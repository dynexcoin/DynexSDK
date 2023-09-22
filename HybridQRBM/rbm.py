""""Restricted Boltzmann Machine"""

__copyright__ = "Dynex Developers, 2023"

import logging
from pathlib import Path
from string import ascii_lowercase as letters
from typing import Optional
from collections import namedtuple

import numpy as np
import numpy.typing as npt

from .callbacks import produce_logger

hooks = (
    "on_fit_start",
    "on_epoch_start",
    "on_batch_start",
    "on_batch_end",
    "on_epoch_end",
    "on_fit_end"
)
Callbacks = namedtuple("Callbacks", hooks, defaults=[[]] * len(hooks))

default_logger = produce_logger(
    "RBM training epoch {epoch:0>3}. "
    "Reconstruction error: {error:.5f}. "
    "Learning rate: {self.optimizer.learning_rate:.5f}"
)
default_callbacks = Callbacks(
    on_fit_end=[default_logger]
)

class RBM:
    """
    Restricted Boltzmann Machine

    To run, the RBM requires two additional components --- a sampler and an
    optimizer.
    """

    def __init__(self, num_hidden: int,sampler: "Sampler", optimizer: "RBMOptimizer",
            name: str=None, rnd: np.random.RandomState=None,  debug=False) -> None:
        """
        RBM constructor.

        Parameters
        ----------
        num_hidden : int
            The number of hidden units. The number of visible units is inferred
            from the data in fit().
        sampler : Sampler
            The package's Sampler instance. The sampler must have a infer(),
            generate(), sample() and predict() methods.
        optimizer : RBMOptimizer
            The package's RBMOptimizer instance. The optimizer must have a
            calculate_update() method.
        name : str
            The name of the RBM.
        rnd : np.random.RandomState
            Pass a random state to ensure reproducibility.
        """
        self.debug = debug;
        self.errors=[];
        if rnd is None:
            rnd = np.random.RandomState()
        self.rnd = rnd
        if name is None:
            name = "rbm_" + "".join(self.rnd.choice(list(letters), size=10))
        self.name = name
        self.num_hidden = num_hidden
        self.num_visible = None
        self.sampler = sampler
        self.optimizer = optimizer
        self.logger = self._get_logger()
        self.weights = None
        self.biases_visible = None
        self.biases_hidden = None
        self.ranges = np.array([[],[]])
        # Links back to the RBM that give direct access to the weights. This
        # could be improved and generalized if necessary.
        self.optimizer.rbm = self
        self.sampler.rbm = self

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

    def fit(
        self,
        features: npt.NDArray,
        labels: Optional[npt.NDArray] = None,
        epochs: int = 20,
        batch_size: int = 100,
        callbacks: Callbacks = default_callbacks
    ) -> float:
        """
        Fit the RBM to the data.

        If labels are given, these are combined with the features to form the
        visible layer.

        Parameters
        ----------
        features : npt.NDArray
            2D binary or float array of shape (num_samples, num_features). If
            the values are floats, they are binarized in each batch update.
        labels : Optional[npt.NDArray]
            Optional 2D binary or float array of shape (num_samples,
            num_labels). Generaly binary values are expected, but if floats are
            passed, they are binarized in each batch update.
        epochs : int
            The number of epochs to train for. Alternatively, the number of
            times each datapoint is used in weight update calculation.
        batch_size : int
            The number of datapoints to use in each weight update calculation.
        callbacks : Callbacks
            Namedtuple of lists of callback functions. The following hooks are
            provided: on_fit_start, on_epoch_start, on_batch_start,
            on_batch_end, on_epoch_end, on_fit_end. The default callback offers
            some basic logging. If you pass any custom callbacks, you also need
            to supply your own callbacks for logging.

        Returns
        -------
        float
            The average reconstruction error of the last epoch. Reconstruction
            error represents the similarity between the input data and visible
            layer after sampling with input data as the initial state. It does
            not fully represent the quality of the model, but is a good proxy
            during training.
        """
        if labels is None:
            self.num_visible = features.shape[-1]
            data = features
        else:
            self.num_visible = features.shape[-1] + labels.shape[-1]
            data = np.hstack((features, labels))

        if self.weights is None:
            self.weights = self.rnd.normal(scale=0.001, size=(self.num_visible, self.num_hidden))
            self.biases_visible = np.zeros(self.num_visible)
            self.biases_hidden = np.zeros(self.num_hidden)

        print('num_visible:',self.num_visible,'num_hidden:',self.num_hidden);

        # Discard some training datapoints so that each batch is the same size.
        num_samples = data.shape[0] - data.shape[0] % batch_size

        print('num_samples:',num_samples);
        
        indices = np.arange(num_samples)
        for callback in callbacks.on_fit_start:
            callback(**locals())
        for epoch in range(epochs):
            if self.debug:
                print('Epoch',epoch,'...');
            error = 0
            self.rnd.shuffle(indices)
            
            batches = np.split(indices, num_samples // batch_size)
            for callback in callbacks.on_epoch_start:
                callback(**locals())

            for batch_num, batch in enumerate(batches):
                print('    batch',batch_num,len(batches));
                print('fit loop: data[batch]:',len(data[batch]));
                error += self._fit_batch(data[batch], callbacks, epoch, batch_num)
            
            error /= len(batches);
            self.errors.append(error);
            print('Epoch',epoch,'Error:',error);
            for callback in callbacks.on_epoch_end:
                callback(**locals())
        for callback in callbacks.on_fit_end:
            callback(**locals())
        return error

    def _fit_batch(self, data: npt.NDArray, callbacks: Callbacks, epoch: int,
            batch_num: int) -> float:
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

        for callback in callbacks.on_batch_start:
            callback(**locals())
        visible_data = (data > self.rnd.random(data.shape)).astype(np.int0)
        print('debug fit_batch visible data=',len(visible_data));
        hidden, prob_hidden = self.sampler.infer(visible_data)
        visible, prob_visible = self.sampler.generate(hidden)
        error = ((visible_data - prob_visible) ** 2).mean()

        positive_sample = (visible_data, prob_visible, hidden, prob_hidden)
        negative_sample = self.sampler.sample(visible=visible)

        delta = self.optimizer.calculate_update(positive_sample, negative_sample)
        self.weights += delta.weights
        self.biases_visible += delta.biases_visible
        self.biases_hidden += delta.biases_hidden
        for callback in callbacks.on_batch_end:
            callback(**locals())
        return error

    def fit_range(self, features: npt.NDArray):
        """
        For some models, the range of the predictions are much more limited
        than the range [0, 1]. This function learns the range of the
        predictions for each class. This information can be used to rescale the
        predictions to the range [0, 1].

        Parameters
        ----------
        features : npt.NDArray
            2D binary or float array of shape (num_samples, num_features).

        Returns
        -------
        np.ndarray
            Return an array of shape (2, num_labels), where the first rows are
            the minima of the predictions, and second row --- the maxima.
        """

        predictions = self.predict(features, rescale=False)
        self.ranges = np.vstack((predictions.min(axis=0), predictions.max(axis=0)))
        return self.ranges

    def energy(
        self,
        visible: npt.NDArray[np.int0],
        hidden: npt.NDArray[np.int0]
    ) -> npt.NDArray[np.float]:
        """
        Calculates the energy of a given state of the RBM.

        Parameters
        ----------
        visible : np.ndarray(dtype=np.int0)
            2D array of shape (num_samples, num_visible).
        hidden : np.ndarray(dtype=np.int0)
            2D array of shape (num_samples, num_hidden).

        Returns
        -------
        np.ndarray(dtype=np.float)
            1D array of shape (num_samples,).
        """
        energy = -visible @ self.biases_visible
        energy -= hidden @ self.biases_hidden
        energy -= (visible @ self.weights * hidden).sum(axis=1)
        return energy

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

    def predict(self, features: npt.NDArray, num_particles: int = 100,
            rescale: bool = False, **kwargs) -> npt.NDArray:
        """
        Predicts the labels of the given features.

        Parameters
        ----------
        features : npt.NDArray
            2D array of shape (num_samples, num_features).
        rescale : bool
            Rescale the predictions to the range [0, 1].
        num_particles  
            If this method is used with DWaveInferenceSampler, an important
            parameter is the num_particles, which determines the number of
            reads to do on the D-Wave annealer. 

        Returns
        -------
        npt.NDArray
            2D array of shape (num_samples, num_labels).
        """

        predictions = self.sampler.predict(features, num_particles, **kwargs)
        if rescale:
            if self.ranges.size == 0:
                raise AttributeError("Range for rescaling is not fitted yet.")
            predictions -= self.ranges[0]
            predictions /= (self.ranges[1] - self.ranges[0])
            predictions[predictions > 1] = 1
            predictions[predictions < 0] = 0
        return predictions
