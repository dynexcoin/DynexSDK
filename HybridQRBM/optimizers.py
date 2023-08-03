"""Optimizer to update the weights of the RBM."""

__copyright__ = "Dynex Developers, 2023"

from collections import namedtuple
import numpy as np

Parameters = namedtuple("Parameters", ["weights", "biases_visible", "biases_hidden"])

class RBMOptimizer:
    """Optimizer to update the weights of the RBM."""

    def __init__(self, learning_rate: float = 0.05, momentum: float = 0.9,
            decay_factor: float = 1.0005, regularizers: tuple = ()) -> None:
        """Initialize the optimizer.

        Parameters
        ----------
        learning_rate : float, optional
            The initial learning rate of the optimizer. Defaults to 0.05.
        momentum : float, optional
            The momentum of the optimizer. Defaults to 0.9.
        decay_factor : float, optional
            The decay factor of the learning rate. The learning rate is divided
            by the decay factor after each update (batch). Defaults to 1.0005.
        regularizers : list, optional
            The regularizers to apply to the weights. With the current
            implementation only weights and biases are passed to the
            regularizers.
        """
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate
        self.momentum = momentum
        self.decay_factor = decay_factor
        self.regularizers = regularizers
        self.rbm = None
        self.delta_weights = None
        self.delta_biases_visible = None
        self.delta_biases_hidden = None

    def calculate_update(self, positive_sample, negative_sample):
        """
        Calculate the update for the weights and biases.

        Parameters
        ----------
        positive_sample : tuple
            The values and probabilities of the visible and hidden layers
            produced in the positive phase of the training. The tuple should
            contain four elements: (visible_values, visible_probabilities,
            hidden_values, hidden_probabilities).
        negative_sample : tuple
            The values and probabilities of the visible and hidden layers
            produced in the negative phase of the training. The tuple should
            contain four elements: (visible_values, visible_probabilities,
            hidden_values, hidden_probabilities).

        Returns
        -------
        namedtuple
            The delta values for the update of the weights and biases.
        """
        # The shape of the weights is not known at the optimizer's
        # initialization, therefore we set the values now.
        if self.delta_weights is None:
            self.delta_weights = np.zeros(self.rbm.weights.shape)
            self.delta_biases_visible = np.zeros(self.rbm.biases_visible.shape)
            self.delta_biases_hidden = np.zeros(self.rbm.biases_hidden.shape)

        visible, _, _, prob_hidden = positive_sample
        batch_size = visible.shape[0]

        #print('optimizer debug visible=',len(visible),'prob_hidden:',len(prob_hidden),'batch_size:',batch_size);

        pos_weights = visible.T @ prob_hidden / batch_size
        pos_biases_visible = visible.sum(axis=0) / batch_size
        pos_biases_hidden = prob_hidden.sum(axis=0) / batch_size

        visible, _, _, prob_hidden = negative_sample
        batch_size = visible.shape[0]

        neg_weights = visible.T @ prob_hidden / batch_size
        neg_biases_visible = visible.sum(axis=0) / batch_size
        neg_biases_hidden = prob_hidden.sum(axis=0) / batch_size

        batch_delta_weights = pos_weights - neg_weights
        batch_delta_biases_visible = pos_biases_visible - neg_biases_visible
        batch_delta_biases_hidden = pos_biases_hidden - neg_biases_hidden

        self.delta_weights *= self.momentum
        self.delta_biases_visible *= self.momentum
        self.delta_biases_hidden *= self.momentum

        self.delta_weights += self.learning_rate * batch_delta_weights
        self.delta_biases_visible += self.learning_rate * batch_delta_biases_visible
        self.delta_biases_hidden += self.learning_rate * batch_delta_biases_hidden

        for regularizer in self.regularizers:
            self.delta_weights -= self.learning_rate * regularizer(self.rbm.weights)
            self.delta_biases_visible -= self.learning_rate * regularizer(self.rbm.biases_visible)
            self.delta_biases_hidden -= self.learning_rate * regularizer(self.rbm.biases_hidden)

        self.learning_rate /= self.decay_factor
        return Parameters(self.delta_weights, self.delta_biases_visible, self.delta_biases_hidden)
