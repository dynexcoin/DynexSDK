"""Some callbacks to pass to the RBM's fit() method."""

import gc
from collections import namedtuple
from pathlib import Path
from matplotlib import figure
from tensorflow import keras
import numpy as np
import numpy.typing as npt

Sample = namedtuple("Sample", ["visible", "visible_prob", "hidden", "hidden_prob"])

def save_particles(self, negative_sample: npt.NDArray[np.int0],
        epoch: int, batch_num: int, **kwargs) -> None:
    """
    Save a visualization of the particles produced from the model distribution.

    The callback also saves some additional information to inspect how the
    training is going.

    Parameters
    ----------
    self
        The RBM instance that calls the function.
    negative_sample : numpy.ndarray
        The particles produced by the sampler in the negative phase of the
        training.
    epoch : int
    batch_num : int
    **kwargs
        locals() passed from the method that calls the callback.
    """
    fig = figure.Figure(figsize=(16,28), facecolor='white')
    axes = fig.subplots(5,2).flatten()
    weight_plots = [
        "Particles", "Probabilities",
        "Delta weights", "Weights",
    ]
    bias_plots = [
        "Visible biases", "Hidden biases",
        "Delta visible biases", "Delta hidden biases",
        "Energies",
    ]
    plots = weight_plots + bias_plots
    axes = dict(zip(plots, axes))
    for key, ax in axes.items():
        ax.set_title(key)
    neg = Sample(*negative_sample)
    energies = self.energy(neg.visible, neg.hidden)
    ind = np.argsort(energies)
    weight_data = [
        np.hstack((neg.visible[ind], neg.hidden[ind])),
        np.hstack((neg.visible_prob[ind], neg.hidden_prob[ind])),
        self.optimizer.delta_weights,
        self.weights,
    ]
    bias_data = [
        self.biases_visible,
        self.biases_hidden,
        self.optimizer.delta_biases_visible,
        self.optimizer.delta_biases_hidden,
        energies,
    ]
    for plot, data in zip(weight_plots, weight_data):
        color = axes[plot].matshow(data)
        fig.colorbar(color, ax=axes[plot])
    for plot, data in zip(bias_plots, bias_data):
        axes[plot].hist(data)
    particle_path = Path(f"particles/E{epoch:0>2}B{batch_num:0>3}.png")
    particle_path.parent.mkdir(exist_ok=True)
    fig.savefig(particle_path)
    # matplotlib sometimes has a memory leak. This is a workaround.
    gc.collect()

def get_auc_calculator(features: npt.NDArray,
        labels: npt.NDArray[np.int0]) -> callable:
    """
    Return a function that calculates the AUC of a model using the given
    dataset (features) and labels. Using this function ensures that these are
    not overwritten by the locals() passed to the callback. This functionality
    allows to pass AUC calculating functions with different datasets, e.g.
    training and validation.

    Parameters
    ----------
    features : numpy.ndarray
        The features of the dataset.
    labels : numpy.ndarray
        The labels of the dataset.

    Returns
    -------
    callable
        A function (callback) that calculates the AUC of a model.
    """
    frozen = locals()
    def multiclass_aucs(self, **kwargs) -> npt.NDArray[np.float64]:
        # kwargs will include features and labels from the RBM's fit() method.
        # Here we overwrite them with the given datasets. self here refers to
        # the RBM instance that called this function.
        kwargs.update(frozen)
        label_predictions, features_predictions = self.sampler.predict(features)
        auc = keras.metrics.AUC()
        results = []
        for i in range(labels.shape[-1]):
            auc.reset_state()
            auc.update_state(features_predictions[:,i], label_predictions[:,i])
            results.append(auc.result().numpy())
        self.logger.info(np.round(np.array(results), 4))
    return multiclass_aucs

def produce_logger(message: str) -> callable:
    """
    Produce a logger callback. The function will receive the locals of either
    the fit() or the _fit_batch() methods. These can be used in the message.

    Parameters
    ----------
    message : str
        The message to log. For example, "Epoch {epoch}" or "{self.name}". In
        the second case, self refers to the RBM instance.

    Returns
    -------
    log_: callable
        The logger callback.
    """
    def log_(**kwargs):
        kwargs["self"].logger.info(message.format(**kwargs))
    return log_
