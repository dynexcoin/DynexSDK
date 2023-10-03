import os
import numpy as np
import torch
import dimod
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import logging
import dynex

logging.basicConfig(filename="MARBM.log", level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class MARBM(nn.Module):
    """
    Mode-Assisted Restricted Boltzmann Machine (MARBM).
    
    Restricted Boltzmann machines (RBMs) are a class of generative models that
    have historically been challenging to train due to the complex nature of 
    their gradient. The MARBM provides a novel approach to training RBMs by 
    combining the standard gradient updates with an off-gradient direction. 
    This direction is constructed using samples from the RBM ground state, 
    also referred to as 'mode'. By utilizing mode-assisted training, the RBM 
    benefits from faster convergence, improved training stability, and lower 
    converged relative entropy (KL divergence).
    
    
    Parameters:
    - visible_units (int): Number of visible units in the RBM.
    - hidden_units (int): Number of hidden units in the RBM.
    
    Attributes:
    - W (torch.Tensor): Weights connecting the visible and hidden units.
    - h_bias (torch.Tensor): Biases associated with the hidden units.
    - v_bias (torch.Tensor): Biases associated with the visible units.
    - free_energies (list): List to store computed free energies during training.
    
    Methods:
    - forward: Compute the forward pass (probability of hidden given visible).
    - sample_hidden: Sample from the hidden layer given the visible layer.
    - sample_visible: Sample from the visible layer given the hidden layer.
    - contrastive_divergence: Perform a Contrastive Divergence (CD) step.
    - rbm2qubo: Convert RBM parameters to a QUBO matrix.
    - train: Train the RBM using mode-assisted training.
    - reconstruct: Reconstruct input data using the trained RBM.
    - compute_free_energy: Compute the free energy of a given configuration.
    - _mode_train_step: Execute one step of mode-assisted training.
    - _cd_train_step: Execute one step of training using Contrastive Divergence.
    
    """
    
    def __init__(self, visible_units: int, hidden_units: int, sampler: str = 'SA', seed: int = None):
        """
        Initialize the Mode-Assisted Restricted Boltzmann Machine (MARBM).
        
        Parameters
        ----------
        visible_units : int
            Number of visible units, representing the dimensionality of the data input.
        hidden_units : int
            Number of hidden units, representing the latent dimensionality or feature detectors.
        sampler : str, optional
            Sampling method used during training. Options: 'SA' (Simulated Annealing) or 'DYNEX' (Dynex Sampler). 
            Default is 'SA'.
        seed : int, optional
            Random seed for reproducibility. If provided, it sets the seeds for both torch and numpy.
        
        Attributes
        ----------
        visible_units : int
            Dimensionality of data input.
        hidden_units : int
            Latent features or representations.
        W : torch.nn.Parameter
            Weight matrix connecting visible to hidden units.
        h_bias : torch.nn.Parameter
            Bias for hidden units.
        v_bias : torch.nn.Parameter
            Bias for visible units.
        metrics_name : str
            Metric(s) name used during training.
        metrics_values : list
            Metric values during training.
        sigm_values : list
            Sigmoid values during training.
        
        Notes
        -----
        The MARBM class encapsulates a Mode-Assisted Restricted Boltzmann Machine, an unsupervised 
        neural network model tailored for learning representations. It includes features for tracking 
        and visualizing training dynamics, enhancing interpretability.
        """
        
        # Initialize parent class
        super(MARBM, self).__init__()

        # Validate inputs
        assert isinstance(visible_units, int) and visible_units > 0, "visible_units should be a positive integer."
        assert isinstance(hidden_units, int) and hidden_units > 0, "hidden_units should be a positive integer."
        
        # Initialize visible and hidden units
        self.visible_units = visible_units
        self.hidden_units = hidden_units

        # Initialize weights with small random values for symmetry breaking
        self.W = nn.Parameter(torch.randn(hidden_units, visible_units) * 0.01)
        
        # Initialize biases to zero, a common practice in RBMs
        self.h_bias = nn.Parameter(torch.zeros(hidden_units))
        self.v_bias = nn.Parameter(torch.zeros(visible_units))
        
        # Data structures for visualization and analysis
        self.metrics_name = ''
        self.metrics_values  = []
        self.sigm_values = []
        
        # Set the sampler and validate its type
        self.sampler = sampler
        assert self.sampler in ['SA', 'DYNEX'], "sampler should be either 'SA' or 'DYNEX'"
        
        # Sampler parameters
        if self.sampler == 'DYNEX':
            self.num_reads = 5000
            self.annealing_time = 300
        else:
            self.num_reads = 25
            self.annealing_time = 4
        self.debugging = False
        self.mainnet = False
        self.logging = False
            
        # Set seeds for reproducibility if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Log the initialization
        logger.info(f"Initialized MARBM with visible units: {visible_units}, hidden units: {hidden_units}")
    
    def set_sampler_parameters(self, num_reads=None, annealing_time=None, debugging=None, mainnet=None, logging=None):
        """
        Set sampler parameters for the RBM.

        Parameters
        ----------
        num_reads : int, optional
            Number of reads for the sampler. If not provided, the existing value remains unchanged.
        annealing_time : float, optional
            Annealing time for the sampler. If not provided, the existing value remains unchanged.
        debugging : bool, optional
            Debugging mode flag. If not provided, the existing value remains unchanged.
        mainnet : bool, optional
            Flag to determine if the Dynex online computing service should be accessed. If not provided, the existing value remains unchanged.
        logging : bool, optional
            Logging mode flag. If not provided, the existing value remains unchanged.

        Notes
        -----
        This method updates the sampler parameters for the RBM based on the provided values.
        """
        
        if num_reads is not None:
            self.num_reads = num_reads

        if annealing_time is not None:
            self.annealing_time = annealing_time

        if debugging is not None:
            self.debugging = debugging

        if mainnet is not None:
            self.mainnet = mainnet

        if logging is not None:
            self.logging = logging

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """
        Compute the forward pass.
        
        Given the visible units, compute the probability of the hidden units being activated.
        
        Parameters:
            v (torch.Tensor): The visible units.
        
        Returns:
            torch.Tensor: Probability of hidden units being activated.
        """
        h_prob = torch.sigmoid(F.linear(v, self.W, self.h_bias))
        logger.debug("Completed forward pass")
        return h_prob
    
    def lock_weights(self):
        """
        Lock the weights of the RBM to prevent them from being updated during training.
        This is useful when utilizing the RBM in transfer learning scenarios, ensuring the 
        pretrained weights remain unchanged.
        """
        for param in self.parameters():
            param.requires_grad = False
            
    def unlock_weights(self):
        """
        Unlock the weights of the RBM, allowing them to be updated during training.
        """
        for param in self.parameters():
            param.requires_grad = True

    def extract_features(self, v):
        """
        Extract the features from the input using the hidden activations of the RBM.
        
        Parameters:
        - v (torch.Tensor): Input data, corresponding to the visible units.

        Returns:
        - torch.Tensor: The activations of the hidden units which can be used as features for downstream tasks.
        
        Usage:
            rbm = MARBM(visible_units, hidden_units)
            rbm.load_model(path)
            input_data = ...  # Your data for which you wish to extract features
            features = rbm.extract_features(input_data)
            # You can now feed this into the subsequent layer of your model
        """
        
        return self.forward(v)
        
    def sample_hidden(self, v: torch.Tensor) -> torch.Tensor:
        """
        Sample from the hidden layer given the visible layer.

        Given the state of the visible units, this method computes the probability 
        of each hidden unit being activated and then samples from a Bernoulli 
        distribution based on these probabilities.

        Parameters:
        - v (torch.Tensor): A tensor representing the state of the visible units. It should have a shape of (batch_size, visible_units).

        Returns:
        - torch.Tensor: A tensor representing the sampled state of the hidden units. It will have a shape of (batch_size, hidden_units).
        """
        h_prob = self.forward(v)
        h_sample = torch.bernoulli(h_prob)
        logger.debug("Sampled hidden layer with shape: %s", str(h_sample.shape))
        return h_sample

    def sample_visible(self, h: torch.Tensor) -> torch.Tensor:
        """
        Sample from the visible layer given the hidden layer.

        Given the state of the hidden units, this method computes the probability 
        of each visible unit being activated and then samples from a Bernoulli 
        distribution based on these probabilities.

        Parameters:
        - h (torch.Tensor): A tensor representing the state of the hidden units. It should have a shape of (batch_size, hidden_units).

        Returns:
        - torch.Tensor: A tensor representing the sampled state of the visible units. It will have a shape of (batch_size, visible_units).
        """
        v_prob = torch.sigmoid(F.linear(h, self.W.t(), self.v_bias))
        v_sample = torch.bernoulli(v_prob)
        logger.debug("Sampled visible layer with shape: %s", str(v_sample.shape))
        return v_sample

    def contrastive_divergence(self, input_data: torch.Tensor, k: int = 1) -> torch.Tensor:
        """
        Perform one step of Contrastive Divergence (CD) for training the RBM.

        The method approximates the gradient of the log-likelihood of the data 
        by running a Gibbs chain for a specified number of steps, k.
        
        Parameters:
        - input_data (torch.Tensor): The visible layer data, of shape [batch_size, visible_units].
        - k (int, optional): The number of Gibbs sampling steps. Defaults to 1.

        Returns:
        - torch.Tensor: The difference between the outer product of the data and 
                        hidden probabilities at the start and the end of the Gibbs chain, 
                        of shape [visible_units, hidden_units].
        """
        
        v0 = input_data
        vk = v0.clone()  # Use clone to avoid in-place modification issues
        
        for _ in range(k):
            hk = self.sample_hidden(vk)
            vk = self.sample_visible(hk)

        h0_prob = self.forward(v0)
        hk_prob = self.forward(vk)

        positive_phase = torch.mm(v0.t(), h0_prob)
        negative_phase = torch.mm(vk.t(), hk_prob)

        logger.debug("Performed Contrastive Divergence")
        return positive_phase - negative_phase
        
    def rbm2qubo(self) -> np.ndarray:
        """
        Convert RBM parameters to a QUBO (Quadratic Unconstrained Binary Optimization) matrix.
        
        The QUBO matrix is constructed using the weights and biases of the RBM. The diagonal 
        of the QUBO matrix corresponds to biases, and the off-diagonal elements correspond to the weights.

        Returns:
            numpy.ndarray: The QUBO matrix with shape (n_total, n_total), where n_total = n_visible + n_hidden.
        """
        
        # Extract the parameters from the RBM as numpy arrays
        vishid = self.W.detach().numpy()
        hidbiases = self.h_bias.detach().numpy().flatten()
        visbiases = self.v_bias.detach().numpy().flatten()
        
        # Number of visible and hidden nodes
        n_visible, n_hidden = vishid.shape
        n_total = n_visible + n_hidden
        
        # Initialize the QUBO matrix with zeros
        Q = np.zeros((n_total, n_total))
        
        # Populate the diagonal entries with biases
        Q[:n_visible, :n_visible] = np.diag(hidbiases)
        Q[n_visible:, n_visible:] = np.diag(visbiases)
        
        # Populate the off-diagonal entries with weights
        Q[:n_visible, n_visible:] = vishid
        return Q
                
    def extract_data_only_from_batch(self, batch):
        """
        Extracts data from a given batch.

        If the batch is a tuple or a list, it extracts the first element. Otherwise, it returns the batch as it is.

        Parameters:
        - batch (Union[Tuple[torch.Tensor, Any], List[torch.Tensor, Any], torch.Tensor]): Input batch which can be a tuple, list, or a tensor.

        Returns:
        - torch.Tensor: Extracted data from the batch.
        """
        if isinstance(batch, (tuple, list)):
            data = batch[0]
        else:
            data = batch
        return data
    
    def preprocess_to_binary(self, data, threshold=0.5):
        """
        Convert the data into binary format based on a given threshold.
        
        Parameters:
        - data (torch.Tensor): Input data tensor.
        - threshold (float, optional): The threshold for conversion to binary. Default is 0.5.
        
        Returns:
        - binary_data (torch.Tensor): Data converted to binary format.
        """
        binary_data = (data > threshold).float()
        return binary_data  
        
    def train(self, train_loader, val_loader=None, epochs=10, lr=0.01, k=1, sigm_a=20, sigm_b=-6, p_max=0.1, plotper=100, loss_metric='free_energy', save_model_per_epoch=False, save_path='./saved_models'):
        """
        Trains the MARBM model on provided data using the specified parameters.

        Parameters:
        - train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        - val_loader (torch.utils.data.DataLoader, optional): DataLoader for the validation dataset. Default is None.
        - epochs (int, optional): Number of training epochs. Default is 10.
        - lr (float, optional): Learning rate for optimization. Default is 0.01.
        - k (int, optional): Number of Gibbs sampling steps used in contrastive divergence. Default is 1.
        - sigm_a (float, optional): Coefficient for the sigmoidal function determining mode switching. Default is 20.
        - sigm_b (float, optional): Bias for the sigmoidal function determining mode switching. Default is -6.
        - p_max (float, optional): Upper limit for the probability of the sigmoidal switch function. Must be within (0, 1]. Default is 0.1.
        - plotper (int, optional): Frequency for calculating and logging the free energy. Default is 100.
        - loss_metric (str, optional): Metric for loss computation. Accepts 'free_energy', 'kl' or 'mse'. Default is 'free_energy'.
        - save_model_per_epoch (bool, optional): If True, the model will be saved after every epoch. Default is False.
        - save_path (str, optional): Path to the directory where models should be saved. Used only if save_model_per_epoch is True. Default is './saved_models'.
        
        Notes:
        Training alternates between mode-based training and contrastive divergence based on stochastic switching. 
        The probability of selecting mode-based training at each step is given by the sigmoid function 
        'sigm = p_max / (1 + np.exp( -sigm_a * (iter_idx + epoch * steps_per_epoch) / total_steps - sigm_b))'. 
        The sigmoid function ensures that as training progresses, especially in the later epochs, there's 
        an increased likelihood of using mode-based training. This is useful as mode training in the 
        later steps helps the model converge more effectively. 
        Free energy or KL loss is periodically computed and stored based on the `plotper` interval.
        
        """
        
        self.metrics_name = loss_metric
        self.metrics_values  = []
        self.sigm_values = []
        
        lr = float(lr)
        sigm_a = float(sigm_a)
        sigm_b = float(sigm_b)
        p_max = float(p_max)
        
        assert isinstance(train_loader, torch.utils.data.DataLoader), "train_loader should be of type torch.utils.data.DataLoader"
        assert val_loader is None or isinstance(val_loader, torch.utils.data.DataLoader), "val_loader should be either None or of type torch.utils.data.DataLoader"
        assert isinstance(epochs, int) and epochs > 0, "epochs should be a positive integer"
        assert isinstance(lr, float) and lr > 0, "lr should be a positive float"
        assert isinstance(k, int) and k > 0, "k should be a positive integer"
        assert isinstance(sigm_a, float) and sigm_a > 0, "sigm_a should be a positive float"
        assert isinstance(sigm_b, float), "sigm_b should be a float"
        assert isinstance(p_max, float) and 0 <= p_max <= 1, "p_max should be a float in the range (0, 1]"
        assert isinstance(plotper, int) and plotper > 0, "plotper should be a positive integer"

        
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)

        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * epochs
        
        logger.info("Total training steps: %s", total_steps)
        logger.info("Training started for %s epochs", epochs)
        
        for epoch in range(epochs):
            logger.info("Epoch %s started", epoch+1)
            for iter_idx, batch in tqdm(enumerate(train_loader), desc=f"Epoch {epoch+1}/{epochs} Training Progress", ncols=100, total=len(train_loader)):
                
                data = self.extract_data_only_from_batch(batch)
                data = self.preprocess_to_binary(data)
                
                sigm = p_max / (1 + np.exp( -sigm_a * (iter_idx + epoch * steps_per_epoch) / total_steps - sigm_b))
                self._mode_train_step(data, optimizer, lr) if torch.rand(1) <= sigm else self._cd_train_step(data, optimizer, lr, k)
                
                # Calculate metric every 'plotper' steps
                if iter_idx % plotper == 0:
                    self.compute_and_log_metric(val_loader, data, loss_metric, sigm)
                    
            logger.info(f"{self.metrics_name}: {self.metrics_values[-1]}") 
            print(f"{self.metrics_name}: {self.metrics_values[-1]}")
            
            # Saving the model after each epoch if specified
            if save_model_per_epoch:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                model_save_path = os.path.join(save_path, f'marbm_epoch_{epoch+1}.pth')
                self.save_model(model_save_path)
                logger.info(f"Model saved at {model_save_path} after epoch {epoch+1}")
                
        logger.info("Training completed")
        
    def _mode_sampling(self):
        """
        Sample the mode (ground state) of the RBM encoded as a QUBO using simulated annealing.

        Returns
        -------
        mode_v : np.array
            State of the visible units of the sampled mode.
        mode_h : np.array
            State of the hidden units of the sampled mode.
        ground_state_energy : float
            Energy of the sampled ground state.

        Notes
        -----
        This method supports two sampling methods: 'SA' (Simulated Annealing) and 'Dynex'.
        """
        
        # Convert the RBM to QUBO format
        Q = self.rbm2qubo()

        if self.sampler == 'SA':
            # Simulated Annealing parameters
            simulated_annealing_parameters = {
                'beta_range': [0.1, 1.0],
                'num_reads': self.num_reads,
                'num_sweeps': self.annealing_time
            }

            # Initialize and sample using the Simulated Annealing sampler
            sampler = dimod.SimulatedAnnealingSampler()
            response = sampler.sample_qubo(-Q, **simulated_annealing_parameters)

        else:
            # Convert QUBO to BQM format for Dynex sampling
            bqm = dimod.BinaryQuadraticModel.from_qubo(-Q, 0.0)
            model = dynex.BQM(bqm, logging=False)

            # Initialize and sample using the Dynex sampler
            sampler = dynex.DynexSampler(model, mainnet=self.mainnet, logging=self.logging, description='Dynex SDK test')
            response = sampler.sample(num_reads=self.num_reads, annealing_time=self.annealing_time, debugging=self.debugging)
        
        # Extract the ground state and its energy
        ground_state = response.first.sample
        ground_state_energy = response.first.energy

        # Split the state into visible and hidden units
        mode_v = np.array([ground_state[i] for i in range(self.visible_units)]).reshape(1, -1)
        mode_h = np.array([ground_state[i] for i in range(self.visible_units, self.visible_units + self.hidden_units)]).reshape(1, -1)
        
        return mode_v, mode_h, ground_state_energy
 
    def _mode_train_step(self, input_data, optimizer, lr):
        """
        Performs a training step using mode-guided training for the RBM.

        This method applies a quantum-inspired simulated annealing to sample the ground state
        (mode) of the RBM encoded as a QUBO. The mode, once sampled, is used to derive 
        model-driven expectations. These expectations, combined with data-driven expectations,
        guide the optimization of the RBM's parameters.

        Parameters:
        - input_data (torch.Tensor): A batch of training data with shape (batch_size, visible_units).
        - optimizer (torch.optim.Optimizer): The optimizer instance used for parameter updates.
        - lr (float): Base learning rate for parameter updates.
        """
        optimizer.zero_grad()

        # Sampling the mode
        mode_v, mode_h, ground_state_energy = self._mode_sampling()
        
        # Calculate model expectations and averages
        model_expectation = torch.mm(torch.tensor(mode_v.T, dtype=torch.float32), torch.tensor(mode_h, dtype=torch.float32))
        model_vis_avg = torch.tensor(mode_v.T, dtype=torch.float32)
        model_hidden_avg = torch.tensor(mode_h.T, dtype=torch.float32)
        
        # Calculate data-driven expectations and averages (positive phase)
        h0_prob = self.forward(input_data)
        data_expectation = torch.mm(input_data.t(), h0_prob)
        data_vis_avg = torch.mean(input_data, dim=0)
        data_hidden_avg = torch.mean(h0_prob, dim=0)
        
        # Calculate the full weight size and mode push
        fullWsize = np.prod(np.array([self.visible_units, self.hidden_units]) + 1)
        mode_push = (1 / (4 * fullWsize)) * (-ground_state_energy - 0.5 * np.sum(self.h_bias.detach().numpy()) - 0.5 * np.sum(self.v_bias.detach().numpy()) - (1 / 4) * np.sum(self.W.detach().numpy()))
        
        optimizer.param_groups[0]['lr'] = lr * mode_push
        
        # Calculate the gradients based on differences in expectations
        self.W.grad = -(data_expectation - model_expectation).t() / input_data.shape[0]
        self.v_bias.grad = -(data_vis_avg - model_vis_avg).mean(dim=0)
        self.h_bias.grad = -(data_hidden_avg - model_hidden_avg).mean(dim=0)
        optimizer.step()
    
    def _cd_train_step(self, input_data, optimizer, lr, k=1):
        """
        Performs one step of training using Contrastive Divergence (CD).
        
        The function updates the model's parameters using 
        the provided optimizer and learning rate.

        Parameters:
        - input_data (torch.Tensor): The input data tensor of shape [batch_size, visible_units].
        - optimizer (torch.optim.Optimizer): The optimizer used to update the model's parameters.
        - lr (float): The learning rate for the optimizer.
        - k (int, optional): The number of Gibbs sampling steps used in CD. Default is 1.
        """
        optimizer.zero_grad()
        optimizer.param_groups[0]['lr'] = lr
        weight_grad = self.contrastive_divergence(input_data, k=k)
        # Updating weights and biases using gradients from CD
        self.W.grad = -weight_grad.t() / input_data.shape[0]
        self.v_bias.grad = -(input_data - self.sample_visible(self.sample_hidden(input_data))).mean(dim=0)
        self.h_bias.grad = -(self.forward(input_data) - self.forward(self.sample_visible(self.sample_hidden(input_data)))).mean(dim=0)
        optimizer.step()

    def reconstruct(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct the input data by passing it through the RBM's hidden layer and then back to the visible layer.

        Given an input visible layer, this method computes the activation of the hidden layer using the method `sample_hidden`,
        and then reconstructs the visible layer using the method `sample_visible`. This is a common approach in RBMs for data reconstruction.

        Parameters:
        - input_data (torch.Tensor): A tensor representing the visible layer's data to be reconstructed. 
                                    Shape should be (batch_size, visible_units).

        Returns:
        - torch.Tensor: A tensor of the reconstructed visible layer. Shape is (batch_size, visible_units).

        """
        h = self.sample_hidden(input_data)
        v = self.sample_visible(h)
        logger.debug("Reconstructed input data with shape: %s", str(v.shape))
        return v

    def compute_free_energy(self, v: torch.Tensor) -> torch.Tensor:
        """
        Compute the free energy of a given configuration.

        The free energy is calculated using the formula:
        F(v) = - v * v_bias - Σ softplus(W * v + h_bias)

        Parameters:
        v (torch.Tensor): The visible layer configuration, of shape [batch_size, visible_units].

        Returns:
        torch.Tensor: The computed free energy for each configuration in the batch, of shape [batch_size].
        """
        wx_b = F.linear(v, self.W, self.h_bias)
        term_1 = torch.matmul(v, self.v_bias)
        term_2 = torch.sum(F.softplus(wx_b), dim=1)
        logger.debug("Computed free energy for a configuration")
        return -term_1 - term_2
    
    def compute_kl_divergence(self, data: torch.Tensor) -> torch.Tensor:
        """
        Compute the KL divergence between the original data and its reconstruction.

        Parameters:
        ----------
        data (torch.Tensor): Original input data, of shape [batch_size, visible_units].

        Returns:
        -------
        torch.Tensor: KL Divergence between the original data and its reconstruction.
        """
        reconstructed_data_probabilities = self.reconstruct(data)
        
        # Apply softmax to turn data and reconstruction into probability distributions
        data_probabilities = F.softmax(data, dim=-1)
        reconstructed_data_probabilities = F.softmax(reconstructed_data_probabilities, dim=-1)
        
        return self.kl_divergence(data_probabilities, reconstructed_data_probabilities)  

    def compute_mse(self, data: torch.Tensor) -> torch.Tensor:
        """
        Compute the Mean Squared Error (MSE) between the original data and its reconstruction.

        Parameters:
        ----------
        data (torch.Tensor): Original input data, of shape [batch_size, visible_units].

        Returns:
        -------
        torch.Tensor: Mean Squared Error between the original data and its reconstruction.
        """
        reconstructed_data = self.reconstruct(data)
        mse_loss = torch.nn.functional.mse_loss(reconstructed_data, data, reduction='mean')
        return mse_loss
    
    def compute_mae(self, data: torch.Tensor) -> torch.Tensor:
        """
        Compute the Mean Absolute Error (MAE) between the original data and its reconstruction.

        Parameters:
        ----------
        data (torch.Tensor): Original input data, of shape [batch_size, visible_units].

        Returns:
        -------
        torch.Tensor: Mean Absolute Error between the original data and its reconstruction.
        """
        reconstructed_data = self.reconstruct(data)
        mae_loss = torch.nn.functional.l1_loss(reconstructed_data, data, reduction='mean')
        return mae_loss
    
    def compute_ssim(self, data: torch.Tensor, image_shape: tuple) -> torch.Tensor:
        """
        Compute the Structural Similarity Index Measure (SSIM) loss between the original flattened 
        grayscale data and its reconstruction.

        Parameters:
        ----------
        data (torch.Tensor): Original input flattened data, of shape [batch_size, visible_units].
        image_shape (tuple): The shape (height, width) of the original images before flattening.

        Returns:
        -------
        torch.Tensor: SSIM loss between the original data and its reconstruction.
        """
        # Reshape the data
        data_2d = data.view(data.size(0), 1, image_shape[0], image_shape[1])
        reconstructed_data = self.reconstruct(data)
        reconstructed_data_2d = reconstructed_data.view(reconstructed_data.size(0), 1, image_shape[0], image_shape[1])

        ssim_value = self._ssim(data_2d, reconstructed_data_2d)
        return 1 - ssim_value  # SSIM loss
    
    def _ssim(self, x: torch.Tensor, y: torch.Tensor, window_size: int = 11, 
              sigma: float = 1.5, C1: float = 0.01**2, C2: float = 0.03**2) -> torch.Tensor:
        """
        Compute the SSIM between two images.
        
        Parameters:
        ----------
        x, y (torch.Tensor): Images to compare, of shape [batch_size, channels, height, width].
        window_size (int): Size of the Gaussian window used to compute local statistics.
        sigma (float): Standard deviation of the Gaussian window.
        C1, C2 (float): Constants to stabilize division with a weak denominator.
        
        Returns:
        -------
        torch.Tensor: SSIM between x and y.
        """
        # Create a Gaussian window
        window = self._create_gaussian_window(window_size, x.size(1), sigma)
        window = window.to(x.device)
        
        # Compute local means
        mu_x = F.conv2d(x, window, padding=window_size//2, groups=x.size(1))
        mu_y = F.conv2d(y, window, padding=window_size//2, groups=y.size(1))
        
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)
        mu_xy = mu_x * mu_y
        
        # Compute local variances
        sigma_x_sq = F.conv2d(x*x, window, padding=window_size//2, groups=x.size(1)) - mu_x_sq
        sigma_y_sq = F.conv2d(y*y, window, padding=window_size//2, groups=y.size(1)) - mu_y_sq
        sigma_xy = F.conv2d(x*y, window, padding=window_size//2, groups=x.size(1)) - mu_xy
        
        # Compute SSIM
        ssim_map = ((2*mu_xy + C1) * (2*sigma_xy + C2)) / ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
        return ssim_map.mean()
    
    def _create_gaussian_window(self, window_size: int, channels: int, sigma: float) -> torch.Tensor:
        """
        Create a Gaussian window for SSIM computation.
        
        Parameters:
        ----------
        window_size (int): Size of the Gaussian window.
        channels (int): Number of channels of the images.
        sigma (float): Standard deviation of the Gaussian window.
        
        Returns:
        -------
        torch.Tensor: The Gaussian window, of shape [channels, 1, window_size, window_size].
        """
        coords = torch.arange(window_size, dtype=torch.float) - window_size // 2
        g = torch.exp(-(coords**2) / (2*sigma**2))
        g /= g.sum()
        
        # Create a Gaussian window using outer product
        window = g.unsqueeze(1) * g.unsqueeze(0)
        window = window.float().unsqueeze(0).unsqueeze(0)
        return window.repeat(channels, 1, 1, 1)

    def compute_and_log_metric(self, val_loader, data, loss_metric, sigm):
        """
        Computes and logs the specified metric using provided data.
        
        Parameters:
        - val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        - data (torch.Tensor): Input data tensor.
        - loss_metric (str): The loss metric to compute. Can be 'free_energy', 'kl', 'mse', 'mae', or 'ssim'.
        
        Returns:
        - metric_value (float): Computed metric value.
        """
        assert loss_metric in ['free_energy', 'kl', 'mse', 'mae', 'ssim'], \
               "loss_metric should be one of ['free_energy', 'kl', 'mse', 'mae', 'ssim']."
        
        with torch.no_grad():
            if val_loader:
                data_for_metric = next(iter(val_loader))
                data_for_metric = self.extract_data_only_from_batch(data_for_metric)
            else:
                data_for_metric = data

            if loss_metric == 'free_energy':
                metric_value = self.compute_free_energy(data_for_metric).mean().item()
            elif loss_metric == 'kl':
                metric_value = self.compute_kl_divergence(data_for_metric).mean().item()
            elif loss_metric == 'mse':
                metric_value = self.compute_mse(data_for_metric).item()
            elif loss_metric == 'mae':
                metric_value = self.compute_mae(data_for_metric).item()
            elif loss_metric == 'ssim':
                metric_value = self.compute_ssim(data_for_metric, (28, 28)).item()
            else:
                raise ValueError(f"Invalid loss_metric: {loss_metric}. Expected 'free_energy', 'kl', 'mse', 'mae', or 'ssim'.")
            
            self.metrics_values.append(metric_value)
            self.sigm_values.append(sigm)

    def kl_divergence(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        Compute the Kullback-Leibler (KL) Divergence between two probability distributions.

        Parameters:
        ----------
        p (torch.Tensor): True probability distribution, of shape [batch_size, visible_units].
        q (torch.Tensor): Approximated probability distribution, of shape [batch_size, visible_units].

        Returns:
        -------
        torch.Tensor: KL Divergence between p and q.

        Note:
        -----
        Ensure that both p and q are proper probability distributions, i.e., they both sum up to 1
        and do not contain any negative values. If they do not sum up to 1, consider normalizing them.
        """
        # Adding a small value to prevent log(0) and division by zero issues
        epsilon = 1e-10
        p = p + epsilon
        q = q + epsilon

        # KL Divergence computation
        kl = p * (torch.log(p) - torch.log(q))
        return torch.sum(kl, dim=-1)
        
    def save_model(self, path):
        """
        Save the trained weights and biases of the RBM.

        Parameters:
            - path (str): Path to save the model's state.
        """
        torch.save({
            'W': self.W,
            'h_bias': self.h_bias,
            'v_bias': self.v_bias
        }, path)
        
    def load_model(self, path):
        """
        Load the weights and biases of the RBM from a saved state.
        
        Parameters:
            - path (str): Path from where to load the model's state.
        """
        checkpoint = torch.load(path)
        self.W = nn.Parameter(checkpoint['W'])
        self.h_bias = nn.Parameter(checkpoint['h_bias'])
        self.v_bias = nn.Parameter(checkpoint['v_bias'])

    def get_visualization_data(self):
        """
        Retrieve the data used for visualization.

        Returns:
        -------
        tuple:
            - metrics_name (str): Name of the metrics.
            - metrics_values (list): A list of metric values collected during training.
            - sigm_values (list): A list of sigmoid values collected during training.
        """
        return self.metrics_name, self.metrics_values, self.sigm_values