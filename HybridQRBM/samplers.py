"""Samplers for the RBM training."""

__copyright__ = "Dynex Developers, 2023"

import pickle
from pathlib import Path
from hashlib import md5
import numpy as np
import numpy.typing as npt
import dynex
import dimod

class Sampler:
    """Base class for samplers."""

    def __init__(self, num_gibbs_updates: int):
        """
        Initialize the sampler.

        Parameters
        ----------
        num_gibbs_updates : int
            Number of Gibbs updates to perform when sampling from the model
            distribution. For example, for a sampler implementing contrastive
            divergence training, setting this value to 1 will result in a
            standard CD-1 training.
        """
        self.num_gibbs_updates = num_gibbs_updates
        self.rbm = None

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def infer(self, visible: npt.NDArray):
        """
        Infer the hidden layer given the visible layer.

        Parameters
        ----------
        visible : numpy.ndarray
            The visible layer. Shape (num_samples, num_visible).

        Returns
        -------
        hidden : numpy.ndarray
            Binary hidden layer values sampled from their probability
            distribution. Shape (num_samples, num_hidden).
        prob_hidden : numpy.ndarray
            The probability for each unit of the hidden layer to be 1. Shape
            (num_samples, num_hidden).
        """
        prob_hidden = self._sigmoid(visible @ self.rbm.weights + self.rbm.biases_hidden)
        hidden = (prob_hidden > self.rbm.rnd.random(prob_hidden.shape)).astype(np.int0)
        return hidden, prob_hidden

    def generate(self, hidden: npt.NDArray):
        """
        Generate the visible layer given the hidden layer.

        Parameters
        ----------
        hidden : numpy.ndarray
            The hidden layer. Shape (num_samples, num_hidden).

        Returns
        -------
        visible : numpy.ndarray
            Binary visible layer values sampled from their probability
            distribution. Shape (num_samples, num_visible).
        prob_visible : numpy.ndarray
            The probability for each unit of the visible layer to be 1. Shape
            (num_samples, num_visible).
        """
        prob_visible = self._sigmoid(hidden @ self.rbm.weights.T + self.rbm.biases_visible)
        visible = (prob_visible > self.rbm.rnd.random(prob_visible.shape)).astype(np.int0)
        return visible, prob_visible

    def sample(self, visible: npt.NDArray, **kwargs):
        """
        Abstract method for sampling from the model distribution. Possibly with
        some clamped visible layer values.
        """
        raise NotImplementedError

    def gibbs_updates(self, visible: npt.NDArray, num_gibbs_updates=None):
        """
        Perform Gibbs sampling starting from some visible layer values.

        Parameters
        ----------
        visible : numpy.ndarray
            The initial visible layer values. Shape (num_samples, num_visible).
        num_gibbs_updates : int
            The number of full updates to perform. If None, the default for the
            sampler is performed.

        Returns
        -------
        visible: numpy.ndarray
            The visible layer after the Gibbs sampling. Shape (num_samples,
            num_visible).
        prob_visible: numpy.ndarray
            The probability for each unit of the visible layer to be 1. Shape
            (num_samples, num_visible).
        hidden : numpy.ndarray
            Binary hidden layer values sampled from their probability
            distribution. Shape (num_samples, num_hidden).
        prob_hidden : numpy.ndarray
            The probability for each unit of the hidden layer to be 1. Shape
            (num_samples, num_hidden).
        """

        if num_gibbs_updates is None:
            num_gibbs_updates = self.num_gibbs_updates
        for _ in range(num_gibbs_updates):
            hidden, prob_hidden = self.infer(visible)
            visible, prob_visible = self.generate(hidden)
        return visible, prob_visible, hidden, prob_hidden
    
    def predict(self, features: npt.NDArray, num_particles: int = 10,
            num_gibbs_updates: int = None, **kwargs) -> npt.NDArray:
        """
        Predict the features and labels for the given feature values.

        Parameters
        ----------
        features : numpy.ndarray[np.int0]
            The feature values to predict the labels for. Shape (num_samples,
            num_features).
        num_particles : int
            Number of particles to use for the sampling, i.e. how may times to
            run the label sampling process for each sample.
        num_gibbs_updates : int, optional
            Number of Gibbs updates to perform for each particle. If not
            provided, self.num_gibbs_updates will be used.

        Returns
        -------
        labels : numpy.ndarray[np.float]
            The predicted labels. Shape (num_samples, num_labels).
        features : numpy.ndarray[np.float]
            The predicted features. Shape (num_samples, num_features).
        """
        if num_gibbs_updates is None:
            num_gibbs_updates = self.num_gibbs_updates
        num_samples = features.shape[0]
        num_features = features.shape[1]
        num_labels = self.rbm.num_visible - num_features
        label_predictions = np.zeros((num_samples, num_labels))
        features_predictions = np.zeros((num_samples, num_features))
        for _ in range(num_particles):
            output = np.zeros(label_predictions.shape)
            for _ in range(num_gibbs_updates):
                visible = np.hstack((features, output))
                hidden, _ = self.infer(visible)
                visible, visible_prob = self.generate(hidden)
                error = ((visible - visible_prob) ** 2).mean();
                output_prob = visible_prob[:,-num_labels:]
                output = visible[:,-num_labels:]
            if num_labels>0:
                label_predictions += output_prob / num_particles
            features_predictions += visible_prob[:,:num_features] / num_particles
        return label_predictions, features_predictions
    
class ContrastiveDivergenceSampler(Sampler):
    """Implements sampling for contrastive divergence training."""

    def sample(self, visible: npt.NDArray, **kwargs) -> tuple:
        """
        Implements sampling for contrastive divergence training.

        The expected use is to start with feature values for the visible layer
        and then perform a number (defined in the constructor) of Gibbs
        updates. If the number of Gibbs updates is 1, this implements the
        standard CD-1 training.

        Parameters
        ----------
        visible : numpy.ndarray
            The visible layer values. Shape (num_samples, num_visible).

        Returns
        -------
        visible, prob_visible, hidden, prob_hidden : tuple
            See gibbs_updates() method for details.
        """
        return self.gibbs_updates(visible)


class PersistentContrastiveDivergenceSampler(Sampler):
    """Implements sampling for persistent contrastive divergence training."""
    def __init__(self, *args, **kwargs):
        """Persistend contrastive divergence training saves the state of the
        imaginary particles between updates. The batch size is not known at
        sampler construction, so values will be initialized later."""
        super().__init__(*args, **kwargs)
        self.visible = None

    def sample(self, visible, **kwargs) -> tuple:
        """
        Performs a number of Gibbs updates, starting from the state of the
        visible layer after the previous update.

        Parameters
        ----------
        visible : numpy.ndarray
            The visible layer values are always passed from the RBM. For this
            sampler it is only used to determine the shape of the sample.

        Returns
        -------
        visible, prob_visible, hidden, prob_hidden : tuple
            See gibbs_updates() method for details.
        """
        if self.visible is None:
            self.visible = (self.rbm.rnd.random(visible.shape) > 0.5).astype(np.int0)
        visible, prob_visible, hidden, prob_hidden = self.gibbs_updates(self.visible)
        self.visible = visible
        return visible, prob_visible, hidden, prob_hidden


class NaiveSampler(Sampler):
    """Implements sampling for naive training."""

    def sample(self, visible, **kwargs):
        """
        Implements sampling for naive training.

        The particles (Markov chains) for the negative phase are initialized to
        random values at the beginning of each batch update. If the Markov
        chains are properly burnt in, the sampled particles represent the model
        distribution better than those from either contrastive divergence or
        persistent contrastive divergence; howevere the required number of
        Gibbs updates to burn in the Markov chains is unknown and likely high.

        Parameters
        ----------
        visible : numpy.ndarray
            The visible layer values are always passed from the RBM. For this
            sampler it is only used to determine the shape of the sample.

        Returns
        -------
        visible, prob_visible, hidden, prob_hidden : tuple
            See gibbs_updates() method for details.
        """
        visible = (self.rbm.rnd.random(visible.shape) > 0.5).astype(np.int0)
        return self.gibbs_updates(visible)

class DynexSampler(Sampler):

    def __init__(self, num_reads = 100, annealing_time = 300, mainnet=False, minimum_stepsize = 0.00000006, logging=True, debugging=False, num_gibbs_updates=0, **kwargs):
        super().__init__(num_gibbs_updates=num_gibbs_updates, **kwargs)
        self.mainnet = mainnet;
        self.logging = logging;	
        self.num_reads = num_reads;
        self.annealing_time = annealing_time;
        self.debugging = debugging;
        self.minimum_stepsize = minimum_stepsize;

    def _sample_dynex(self, qubo, num_particles):

        # BQM from QUBO:
        bqm = dimod.BinaryQuadraticModel.from_qubo(qubo, 0);
        model = dynex.BQM(bqm);
        dnxsampler = dynex.DynexSampler(model, mainnet=self.mainnet, logging = self.logging, description='PyTorch DNX Layer');
        sampleset = dnxsampler.sample(num_reads=self.num_reads, annealing_time = self.annealing_time, debugging=self.debugging, minimum_stepsize = self.minimum_stepsize);
        if len(sampleset)>0:
            weights = np.array(list(sampleset.first.sample.values()));
            energy = sampleset.first.energy;
            num_units = self.rbm.num_visible + self.rbm.num_hidden
            visible = np.array([weights[:self.rbm.num_visible]]);
            hidden = np.array([weights[self.rbm.num_visible:num_units]]);
            self.rbm.logger.info(f"Dynex Platform: sampled response: energy = {energy}")
        else:
            # an error occured:            
            visible = self.biases_visible;
            hidden  = self.biases_hidden;
            self.rbm.logger.info(f"Dynex Platform: invalid sample returned")
        return visible, hidden

    def sample(self, visible, **kwargs):
        """
        Implements sampling for the Dynex Neuromorphic Platform.

        Parameters
        ----------
        visible : numpy.ndarray
            The visible layer values are always passed from the RBM. When
            sampling from the model distribution, it is only used to determine
            the shape of the sample.

        Returns
        -------
        visible, prob_visible, hidden, prob_hidden : tuple
            visible and hidden are the samples returned by the Dynex sampler.
            prob_visible and prob_hidden are calculated based on the Dynex
            sample values.
        """
        qubo = self.rbm.to_qubo_matrix()
        num_reads = self.num_reads;

        visible, hidden = self._sample_dynex(qubo, num_reads)
        _, prob_hidden = self.infer(visible)
        _, prob_visible = self.generate(hidden)
        if self.num_gibbs_updates > 0:
            visible, prob_visible, hidden, prob_hidden = self.gibbs_updates(visible)
        return visible, prob_visible, hidden, prob_hidden 

class DWaveSampler(Sampler):
    """Implements sampling for the D-Wave machine."""

    def __init__(self, dwave_sampler, num_spin_reversal_transforms, temp=1.0,
            num_gibbs_updates=0, chain_strength=None, dwave_params=None, **kwargs):
        """
        Initialize the sampler.

        Parameters
        ----------
        dwave_sampler : dwave.system.samplers.DWaveSampler
            The D-Wave machine to use. For AWS this would be a
            braket.ocean_plugin.BraketDWaveSampler or, more likely
            dwave.system.FixedEmbeddingComposite. The latter allows to fix the
            embedding, saving a lot of work for each new sample generation,
            especially for larger RBMs.
        num_spin_reversal_transforms : int
            Number of spin-reversal transforms to apply. This mitigates some
            DWave sampling biases.
        temp : float, optional
            Temperature for the sampling. The QUBO matrix is divided by this
            value before passed to the DWave sampler. Higher temperature lead
            to more uniform sampling.
        num_gibbs_updates : int
            Number of Gibbs updates to perform on top of the samples of the
            visual layer from D-Wave.
        chain_strength : float, optional
            The chain strength for D-Wave sampling. If None, the default
            value (method) for D-Wave is used.
        dwave_params : dict, optional
            Additional parameters to pass to the D-Wave sampler.
        """
        super().__init__(num_gibbs_updates=num_gibbs_updates, **kwargs)
        self.dwave_params = {} if dwave_params is None else dwave_params
        self.sampler = dwave_sampler
        self.num_spin_reversal_transforms = num_spin_reversal_transforms
        self.temp = temp
        self.chain_strength = chain_strength

    def _sample_dwave(self, qubo, num_particles, cached_response=None):
        if cached_response:
            with open(cached_response, "rb") as file:
                response = pickle.load(file)
            self.rbm.logger.info(f"Loaded cached response from: {cached_response}")
        else:
            response = self.sampler.sample_qubo(
                qubo,
                num_reads=num_particles,
                auto_scale=False,
                num_spin_reversal_transforms=self.num_spin_reversal_transforms,
                chain_strength=self.chain_strength,
                **self.dwave_params,
            )
            name = response.info["taskMetadata"]["id"].rsplit("/")[-1]
            response_path = Path(f"responses/{name}.response")
            response_path.parent.mkdir(exist_ok=True)
            with open(response_path, "wb") as file:
                pickle.dump(response, file)
            cache_hash = md5(str((qubo.tobytes(), num_particles)).encode("utf-8")).hexdigest()
            with open(f"responses/{cache_hash}", "w") as file:
                file.write(str(response_path))
            self.rbm.logger.info(f"Loaded D-Wave response: {name}")
            self.rbm.logger.info(f"Cached the response: {cache_hash}")
        df = response.to_pandas_dataframe()
        df = df.loc[df.index.repeat(df["num_occurrences"]),:]
        num_units = self.rbm.num_visible + self.rbm.num_hidden
        visible = df.iloc[:,:self.rbm.num_visible].values
        hidden = df.iloc[:,self.rbm.num_visible:num_units].values
        return visible, hidden

    def sample(self, visible, **kwargs):
        """
        Implements sampling for the D-Wave machine.

        Parameters
        ----------
        visible : numpy.ndarray
            The visible layer values are always passed from the RBM. When
            sampling from the model distribution, it is only used to determine
            the shape of the sample.

        Returns
        -------
        visible, prob_visible, hidden, prob_hidden : tuple
            visible and hidden are the samples returned by the DWave sampler.
            prob_visible and prob_hidden are calculated based on the DWave
            sample values.
        """

        qubo = self.rbm.to_qubo_matrix() / self.temp
        num_reads = visible.shape[0]
        visible, hidden = self._sample_dwave(qubo, num_reads)
        _, prob_hidden = self.infer(visible)
        _, prob_visible = self.generate(hidden)
        if self.num_gibbs_updates > 0:
            visible, prob_visible, hidden, prob_hidden = self.gibbs_updates(visible)
        return visible, prob_visible, hidden, prob_hidden


class DWaveInferenceSampler(DWaveSampler):
    """Extends DWaveSampler with inference / prediction."""

    def predict(self, features: npt.NDArray, num_particles: int = 100,
            num_gibbs_updates: int = None, use_cache = False, **kwargs) -> npt.NDArray:
        """
        Predict the labels for the given feature values.

        Parameters
        ----------
        features : numpy.ndarray[np.int0]
            The feature values to predict the labels for. Shape (num_samples,
            num_features).
        num_particles : int
            Number of particles to use for the sampling, i.e. how may times to
            run the label sampling process for each sample.
        num_gibbs_updates : int, optional
            Number of Gibbs updates to perform for each particle. If not
            provided, self.num_gibbs_updates will be used.
        use_cache : bool
            If set to True, and a cached response is available, it will be used.

        Returns
        -------
        labels : numpy.ndarray[np.float]
            The predicted labels. Shape (num_samples, num_label_classes).
        """
        num_samples = features.shape[0]
        num_features = features.shape[1]
        num_labels = self.rbm.num_visible - num_features
        label_predictions = np.zeros((num_samples, num_labels))
        if num_gibbs_updates is None:
            num_gibbs_updates = self.num_gibbs_updates
        for sample in range(num_samples):
            qubo = self.rbm.to_qubo_matrix(features[sample]) / self.temp
            cache_hash = md5(str((qubo.tobytes(), num_particles)).encode("utf-8")).hexdigest()
            cache_file = Path(f"responses/{cache_hash}")
            cached_response = None
            if use_cache and cache_file.exists():
                with open(cache_file, "r") as file:
                    cached_response = file.read()
            visible, _ = self._sample_dwave(qubo, num_particles, cached_response)
            for _ in range(num_gibbs_updates):
                visible[:,:num_features] = features[sample]
                hidden, _ = self.infer(visible)
                visible, visible_prob = self.generate(hidden)
            label_predictions[sample] = visible[:,-num_labels:].mean(axis=0)
        return label_predictions
