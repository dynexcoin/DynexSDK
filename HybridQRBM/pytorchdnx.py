"""Pytorch Dynex Neuromporhic Layer."""

__copyright__ = "Dynex Developers, 2023"

import dynex
import dimod
import torch
import torch.nn as nn
import numpy as np
import numpy.typing as npt
import logging
from string import ascii_lowercase as letters
from pathlib import Path
from tqdm import tqdm

class dnx(nn.Module):
    def __init__(self, num_hidden: int, steps_per_epoch: int,
                 sampler: "Sampler", 
                 optimizer: "RBMOptimizer",
                 rnd: np.random.RandomState=None, 
                 name: str=None,
                 num_gibbs_updates=1, 
                 mainnet=False, 
                 logging=False, 
                 num_reads=100, 
                 annealing_time = 1000, 
                 debugging=False,
                 minimum_stepsize=0.002):
        super().__init__();
        self.mainnet = mainnet;
        self.logging = logging;
        self.num_reads = num_reads;
        self.annealing_time = annealing_time;
        self.num_gibbs_updates = num_gibbs_updates;
        self.debugging = debugging;
        self.minimum_stepsize = minimum_stepsize;
        self.errors=[1.0];
        self.acc=[0.0];
        
        self.data = []; # batch collection
        self.cnt = 0;
        self.steps_per_epoch = steps_per_epoch;
        
        # default randomizer if not set:
        if rnd is None:
            rnd = np.random.RandomState();
        self.rnd = rnd;
        
        # default name if not set:
        if name is None:
            name = "dnx_layer_" + "".join(self.rnd.choice(list(letters), size=10))
        self.name = name

        self.num_visible = None;
        self.num_hidden = num_hidden;
        self.weights = None
        self.biases_visible = None
        self.biases_hidden = None
        self.ranges = np.array([[],[]])
        self.logger = self._get_logger();

        self.optimizer = optimizer;
        self.optimizer.rbm = self;
        self.sampler = sampler;
        self.sampler.rbm = self;
        
        # PyTorch buffers:
        self.register_buffer('model_nodes', torch.tensor(0))
        self.register_buffer('model_weights', torch.tensor(0))
        self.register_buffer('model_biases_visible', torch.tensor(0))
        self.register_buffer('model_biases_hidden', torch.tensor(0))
        

    def _get_logger(self, level=logging.DEBUG):
        """Returns a logger that logs to log/{self.name}.log."""
        logger = logging.getLogger(self.name)
        logger.setLevel(level)
        formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s")
        if not logger.handlers:
            log_path = Path(f"log/{self.name}.log")
            log_path.parent.mkdir(exist_ok=True)
            handler = logging.FileHandler(log_path)
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def to_qubo_matrix(self, data: np.ndarray = None, clamp_strength: float = 2.0):
        """
        Generates a QUBO matrix from the RBM weights and biases.

        The resulting matrix is of shape (num_visible + num_hidden, num_visible
        + num_hidden) with weights in the upper right (num_visible, num_hidden)
        block and biases on the the diagonal.

        Parameters
        ----------
        data : np.ndarray
            2D binary array of shape (num_samples, num_features). If data is
            passed, the first num_features columns are fixed to the given
            values.
        clamp_strength : float
            If data array is passed, the corresponding visible unit bias values
            are overriden and set to +/- clamp_strength * (maximum weight or
            bias value). If the value is set too low, the returned samples may
            differ from the data. If it is too high, it might not be possible
            to embed to QUBO matrix on the D-Wave annealer without scaling the
            other values too much.
        """
        if data is None:
            linear = np.concatenate((self.biases_visible, self.biases_hidden))
        else:
            max_weight = max(
                self.biases_visible[len(data):].max(),
                self.biases_hidden.max(),
                self.weights.max()
            )
            clamp_strength = clamp_strength * max_weight
            linear = np.concatenate((
                clamp_strength * (2 * data - 1),
                self.biases_visible[len(data):],
                self.biases_hidden
            ))
        blocks = [
            [np.zeros((self.num_visible, self.num_visible)), self.weights],
            [np.zeros((self.num_hidden, linear.shape[0]))]
        ]
        bqm_arr = np.block(blocks) + np.diag(linear)
        # Energy calculation requires that negative weights and biases are used
        # in the QUBO matrix.
        return -bqm_arr

    def _fit_batch(self, data: npt.NDArray) -> float:
        """
        Updates the weights and biases of the RBM for a single batch.

        The hidden layer probabilities in the positive phase can be calculated
        exactly. This is done using the RBM sampler's infer() method, which
        should be the same for all samplers. The methods to sample from an
        approximate model distribution for the negative phase differ from
        sampler to sampler. The corresponding positive and negative samples are
        passed to the RBM's optimizer to update the weights and biases.

        Parameters
        ----------
        data : npt.NDArray
            2D binary or float array, where the features and labels (if any)
            are already combined. If the values are floats, they are
            interpreted as probabilities and randomly binarized.
        callbacks : Callbacks
            Callbacks are passed from the fit() method.
        epoch : int
            The current epoch number. Useful for callbacks / logging.
        batch_num : int
            The current batch number. Useful for callbacks / logging.

        Returns
        -------
        float
            The mean reconstruction error of the batch.
        """
        
        error = 0;
        
        # Initialise weights with QUBO:
        visible_data = (data > self.rnd.random(data.shape)).astype(np.int0)
        hidden, prob_hidden = self.sampler.infer(visible_data)
        visible, prob_visible = self.sampler.generate(hidden)
        error = ((visible_data - prob_visible) ** 2).mean()

        positive_sample = (visible_data, prob_visible, hidden, prob_hidden)
        negative_sample = self.sampler.sample(visible=visible)

        delta = self.optimizer.calculate_update(positive_sample, negative_sample)
        self.weights += delta.weights
        self.biases_visible += delta.biases_visible
        self.biases_hidden += delta.biases_hidden

        # Apply sample on batch:
        print('DynexQRBM PyTorch Layer | applying sampling result...', len(self.data),'x',len(self.data[0]));
        for i in range(0, data.shape[0]):
            visible_data = np.array([data[i]]);
            hidden, prob_hidden = self.sampler.infer(visible_data);
            visible, prob_visible = self.sampler.generate(hidden);
            error += ((visible_data - prob_visible) ** 2).mean();
            positive_sample = (visible_data, prob_visible, hidden, prob_hidden)
            negative_sample = self.sampler.gibbs_updates(visible)
            delta = self.optimizer.calculate_update(positive_sample, negative_sample)
            self.weights += delta.weights
            self.biases_visible += delta.biases_visible
            self.biases_hidden += delta.biases_hidden
        error = error / (data.shape[0] + 1)
        
        return error

    def forward(self, x):
        
        # increase counter:
        self.cnt += 1;
        
        xnp = x.cpu().detach().numpy(); # convert tensor to numpy
        self.num_visible = np.array(xnp[0].flatten().tolist()).shape[-1];
        error = 0;
        
        # initialize weights if not already set:
        if self.weights is None:
            self.weights = self.rnd.normal(scale=0.001, size=(self.num_visible, self.num_hidden));
            self.biases_visible = np.zeros(self.num_visible);
            self.biases_hidden = np.zeros(self.num_hidden);
            
        # combine data from all batches:
        for batch in range(0, len(xnp)):
            # retrieve data from batch:
            self.data.append(xnp[batch].flatten().tolist());
            
        self.logger.info(f"DynexQRBM PyTorch Layer | batch data appended:  {self.cnt}")
        print('DynexQRBM PyTorch Layer | batch data appended:',self.cnt);
            
        # end of batch?
        if self.cnt % self.steps_per_epoch == 0:
            print('DynexQRBM PyTorch Layer | end of batch, sampling...', len(self.data),'x',len(self.data[0]));
            self.data = np.array(self.data);
            error += self._fit_batch(self.data);
            error /= self.steps_per_epoch; 
            self.errors.append(error);
            self.logger.info(f"DynexQRBM PyTorch Layer | SME:  {error}")
            acc = ((1-error)*100);
            self.acc.append(acc);
            self.logger.info(f"DynexQRBM PyTorch Layer | ACCURACY: {acc}%")
            print('DynexQRBM PyTorch Layer | SME:','{:f}'.format(error),'ACCURACY:','{:f}%'.format(acc));
            # Update PyTorch buffers:
            self.model_dmodel_weight = torch.Tensor(self.weights);
            self.model_biases_visible = torch.Tensor(self.biases_visible);
            self.model_biases_hidden = torch.Tensor(self.biases_hidden);
            # reset data container
            self.data = []; 
            
        return torch.Tensor(self.model_biases_hidden);   # hidden nodes are returned for transfer learning

# --------------------------------------------------------------------------------------------------------------------------------------------------------
# QRBM Experimental class:
# --------------------------------------------------------------------------------------------------------------------------------------------------------
class dnx_experimental(nn.Module):
    """ Dynex QRBM Layer """
    def __init__(self, n_visible, n_hidden, batch_size=1, lr=0.1, lr_decay=0.1, mainnet=False, 
                 logging=False, num_reads=4096, annealing_time = 1000):
        super().__init__()
        self.n_visible = n_visible;
        self.n_hidden = n_hidden;
        self.lr = lr; 
        self.lr_decay = lr_decay;
        self.batch_size = batch_size;
        self.epoch = 0;
        self.mainnet = mainnet;
        self.logging = logging;
        self.num_reads = num_reads;
        self.annealing_time = annealing_time;
        self.mse = -1;
        self.v_prim = [];

        # initialize weights and biases
        self.w = (np.random.rand(self.n_visible, self.n_hidden) * 2 - 1) * 1
        self.visible_bias = (np.random.rand(self.n_visible) * 2 - 1) * 1
        self.hidden_bias = (np.random.rand(self.n_hidden) * 2 - 1) * 1

        # initial momentum velocity value
        self.momentum = 0
        self.momentum_w = np.zeros((len(self.visible_bias), len(self.hidden_bias)));
        self.momentum_v = np.zeros(len(self.visible_bias));
        self.momentum_h = np.zeros(len(self.hidden_bias));

    def sample_opposite_layer_pyqubo(self, v, layer, weights, opposite_layer):

        # initialize Hamiltonian
        H = 0
        H_vars = []

        # initialize all variables (one for each node in the opposite layer)
        for j in range(len(opposite_layer)):
            H_vars.append(Binary('DNX'+str(j)))
    
        for i, bias in enumerate(layer):
            # filter only chosen nodes in the first layer
            if not v[i]:
                continue
    
            # add reward to every connection
            for j, opp_bias in enumerate(opposite_layer):
                H += -1 * weights[i][j] * H_vars[j]
    
        for j, opp_bias in enumerate(opposite_layer):
            H += -1 * opp_bias * H_vars[j]
        
        model = H.compile()
        bqm = model.to_bqm()
        
        # use the dynex sampler:
        dnxmodel = dynex.BQM(bqm, logging=False);
        dnxsampler = dynex.DynexSampler(dnxmodel, logging=self.logging, mainnet=self.mainnet, description='PyTorch QRBM');
        sampleset = dnxsampler.sample(num_reads=self.num_reads, annealing_time = self.annealing_time);
        solution1 = dnxsampler.dimod_assignments.first.sample;
        solution1_list = [(k[3:], v) for k, v in solution1.items()]
        solution1_list.sort(key=lambda tup: int(tup[0]))  # sorts in place
        solution1_list_final = [v for (k, v) in solution1_list]
        return solution1_list_final
        
    def forward(self, x):
        v = x.cpu().detach().numpy(); # convert tensor to numpy
        v = v[0].flatten().tolist(); 
        # TODO: take all BATCH_SIZE elements! currently only first element of batch is taken
        print('[DEBUG] DynexQRBM PyTorch Layer - forward invoked...');
        # # 1.2 compute the probabilities of the hidden units
        h = self.sample_opposite_layer_pyqubo(v, self.visible_bias, self.w, self.hidden_bias);
        # 2 Compute the outer product of v and h and call this the positive gradient.
        pos_grad = np.outer(v, h);
        # 3.1 From h, sample a reconstruction v' of the visible units
        v_prim = self.sample_opposite_layer_pyqubo(h, self.hidden_bias, self.w.T, self.visible_bias);
        self.v_prim = v_prim; 
        # 3.2 then resample the hidden activations h' from this. (Gibbs sampling step)
        h_prim = self.sample_opposite_layer_pyqubo(v_prim, self.visible_bias, self.w, self.hidden_bias);
        # 4 Compute the outer product of v' and h' and call this the negative gradient.
        neg_grad = np.outer(v_prim, h_prim);
        # 5 Let the update to the weight matrix W be the positive gradient minus the negative gradient,
        #   times some learning rate
        self.momentum_w = self.momentum * self.momentum_w + self.lr * (pos_grad - neg_grad)
        self.w += self.momentum_w;
        # 6 Update the biases a and b analogously: a=epsilon (v-v'), b=epsilon (h-h')
        #   momentum here
        self.momentum_v = self.momentum * self.momentum_v + self.lr * (np.array(v) - np.array(v_prim))
        self.momentum_h = self.momentum * self.momentum_h + self.lr * (np.array(h) - np.array(h_prim))
        self.visible_bias += self.momentum_v;
        self.hidden_bias  += self.momentum_h;
        print('[DEBUG] DynexQRBM trained, generating output...');
        # generate output:
        sample_v = v
        sample_h = self.sample_opposite_layer_pyqubo(sample_v, self.visible_bias, self.w, self.hidden_bias);
        sample_output = self.sample_opposite_layer_pyqubo(sample_h, self.hidden_bias, self.w.T, self.visible_bias);
        # calculate mse:
        self.mse = np.sum((np.array(v) - np.array(sample_output))**2)/self.n_visible; 
        print('[DEBUG] MSE = ',self.mse);
        # increase internal epoch counter:
        self.epoch += 1;
        # loss rate adjustment:
        if self.epoch % 5 == 0:
                #learning_rate_decay
                self.lr *= (1 - self.lr_decay)
                print("[DEBUG] loss rate set to ", self.lr)
        
        return torch.Tensor(sample_output)