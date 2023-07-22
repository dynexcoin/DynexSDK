#    Copyright 2023 Dynex Developers
#    Parts of it are Copyright 2018 D-Wave Systems Inc.

#    Licensed under the Apache License, Version 2.0 (the "License")
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http: // www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import dimod
import dynex

from tabulate import tabulate #pip install tabulate


class DecisionStumpClassifier:
    """Decision tree classifier that operates on a single feature with a single splitting rule.

    The index of the feature used in the decision rule is stored
    relative to the original data frame.
    """

    def __init__(self, X, y, feature_index):
        """Initialize and fit the classifier.

        Args:
            X (array): 
                2D array of feature vectors.  Note that the array
                contains all features, while the weak classifier
                itself uses only a single feature.
            y (array):
                1D array of class labels, as ints.  Labels should be
                +/- 1.
            feature_index (int):
                Index for the feature used by the weak classifier,
                relative to the overall data frame.
        """
        self.i = feature_index

        self.clf = DecisionTreeClassifier(max_depth=1)
        self.clf.fit(X[:, [feature_index]], y)

    def predict(self, X):
        """Predict class.

        Args:
            X (array):
                2D array of feature vectors.  Note that the array
                contains all features, while the weak classifier
                itself will make a prediction based only a single
                feature.

        Returns:
            Array of class labels.
        """
        return self.clf.predict(X[:, [self.i]])


def _build_H(classifiers, X, output_scale):
    """Construct matrix of weak classifier predictions on given set of input vectors."""
    H = np.array([clf.predict(X) for clf in classifiers], dtype=float).T

    # Rescale H
    H *= output_scale

    return H


class EnsembleClassifier:
    """Ensemble of weak classifiers."""

    def __init__(self, weak_classifiers, weights, weak_classifier_scaling, offset=1e-9):
        """Initialize ensemble from list of weak classifiers and weights.

        Args:
            weak_classifiers (list):
                List of classifier instances.
            weights (array):
                Weights associated with the weak classifiers.
            weak_classifier_scaling (float):
                Scaling for weak classifier outputs.
            offset (float):
                Offset value for ensemble classifier.  The default
                value is a small positive number used to prevent
                ambiguous 0 predictions when weak classifiers exactly
                balance each other out.
        """
        self.classifiers = weak_classifiers
        self.w = weights
        self.weak_clf_scale = weak_classifier_scaling
        self.offset = offset

    def predict(self, X):
        """Compute ensemble prediction.

        Note that this function returns the numerical value of the
        ensemble predictor, not the class label.  The predicted class
        is sign(predict()).
        """
        H = _build_H(self.classifiers, X, self.weak_clf_scale)

        # If we've already filtered out those with w=0 and we are only
        # using binary weights, this is just a sum
        preds = np.dot(H, self.w)
        return preds - self.offset

    def predict_class(self, X):
        """Compute ensemble prediction of class label."""
        preds = self.predict(X)

        # Add a small perturbation to any predictions that are exactly
        # 0, because these will not count towards either class when
        # passed through the sign function.  Such zero predictions can
        # happen when the weak classifiers exactly balance each other
        # out.
        preds[preds == 0] = 1e-9

        return np.sign(preds)

    def score(self, X, y):
        """Compute accuracy score on given data."""
        if sum(self.w) == 0:
            # Avoid difficulties that occur with handling this below
            return 0.0
        return accuracy_score(y, self.predict_class(X))

    def squared_error(self, X, y):
        """Compute squared error between predicted and true labels.

        Provided for testing purposes.
        """
        p = self.predict(X)
        return sum((p - y)**2)

    def fit_offset(self, X):
        """Fit offset value based on class-balanced feature vectors.

        Currently, this assumes that the feature vectors in X
        correspond to an even split between both classes.
        """
        self.offset = 0.0
        # Todo: review whether it would be appropriate to subtract
        # mean(y) here to account for unbalanced classes.
        self.offset = np.mean(self.predict(X))

    def get_selected_features(self):
        """Return list of features corresponding to the selected weak classifiers."""
        return [clf.i for clf, w in zip(self.classifiers, self.w) if w > 0]


class AllStumpsClassifier(EnsembleClassifier):
    """Ensemble classifier with one decision stump for each feature."""

    def __init__(self, X, y):
        if not all(np.isin(y, [-1, 1])):
            raise ValueError("Class labels should be +/- 1")

        num_featuers = np.size(X, 1)
        classifiers = [DecisionStumpClassifier(
            X, y, i) for i in range(num_featuers)]

        # Note: the weak classifier output scaling is arbitrary in
        # this case and does not affect the predictions.
        super().__init__(classifiers, np.ones(num_featuers), 1/num_featuers)
        self.fit_offset(X)


def _build_bqm(H, y, lam):
    """Build BQM.

    Args:
        H (array):
            2D array of weak classifier predictions.  Each row is a
            sample point, each column is a classifier.
        y (array):
            Outputs
        lam (float):
            Coefficient that controls strength of regularization term
            (larger values encourage decreased model complexity).
    """
    n_samples = np.size(H, 0)
    n_classifiers = np.size(H, 1)

    # samples_factor is a factor that appears in front of the squared
    # loss term in the objective.  In theory, it does not affect the
    # problem solution, but it does affect the relative weighting of
    # the loss and regularization terms, which is otherwise absorbed
    # into the lambda parameter.

    # Using an average seems to be more intuitive, otherwise, lambda
    # is sample-size dependent.
    samples_factor = 1.0 / n_samples

    bqm = dimod.BQM('BINARY')
    bqm.offset = samples_factor * n_samples

    for i in range(n_classifiers):
        # Note: the last term with h_i^2 is part of the first term in
        # Eq. (12) of Neven et al. (2008), where i=j.
        bqm.add_variable(i, lam - 2.0 * samples_factor *
                         np.dot(H[:, i], y) + samples_factor * np.dot(H[:, i], H[:, i]))

    for i in range(n_classifiers):
        for j in range(i+1, n_classifiers):
            # Relative to Eq. (12) from Neven et al. (2008), the
            # factor of 2 appears here because each term appears twice
            # in a sum over all i,j.
            bqm.add_interaction(
                i, j, 2.0 * samples_factor * np.dot(H[:, i], H[:, j]))

    return bqm


def _minimize_squared_loss_binary(H, y, lam, num_reads=10000, annealing_time=300, mainnet=True):
    """Minimize squared loss using binary weight variables."""

    bqm = _build_bqm(H, y, lam)

    sampler = 'DYNEX';
    model = dynex.BQM(bqm);
    dnxsampler = dynex.DynexSampler(model,  mainnet=mainnet); 
    sampleset = dnxsampler.sample(num_reads=num_reads, annealing_time = annealing_time, debugging=False); 
    weights = np.array(list(dnxsampler.dimod_assignments.first.sample.values()));
    energy = dnxsampler.dimod_assignments.first.energy;

    return weights, energy


class QBoostClassifier(EnsembleClassifier):
    """Construct an ensemble classifier using quadratic loss minimization.

    """

    def __init__(self, X, y, lam, weak_clf_scale=None, drop_unused=True, num_reads=10000, annealing_time=300, mainnet=True):
        """Initialize and fit QBoost classifier.

        X should already include all candidate features (e.g., interactions).

        Args:
            X (array):
                2D array of feature vectors.
            y (array):
                1D array of class labels (+/- 1).
            lam (float):
                regularization parameter.
            weak_clf_scale (float or None):
                scale factor to apply to weak classifier outputs.  If
                None, scale by 1/num_classifiers.
            drop_unused (bool):
                if True, only retain the nonzero weighted classifiers.
        """
        if not all(np.isin(y, [-1, 1])):
            raise ValueError("Class labels should be +/- 1")

        num_features = np.size(X, 1)

        if weak_clf_scale is None:
            weak_clf_scale = 1 / num_features

        wclf_candidates = [DecisionStumpClassifier(
            X, y, i) for i in range(num_features)]

        H = _build_H(wclf_candidates, X, weak_clf_scale)

        # For reference, store individual weak classifier scores.
        # Note: we don't check equality h==y here because H might be rescaled.
        self.weak_scores = np.array([np.mean(np.sign(h) * y > 0) for h in H.T])

        weights, self.energy = _minimize_squared_loss_binary(H, y, lam, num_reads, annealing_time, mainnet)

        # Store only the selected classifiers
        if drop_unused:
            weak_classifiers = [wclf for wclf, w in zip(
                wclf_candidates, weights) if w > 0]
            weights = weights[weights > 0]
        else:
            weak_classifiers = wclf_candidates

        super().__init__(weak_classifiers, weights, weak_clf_scale)
        self.fit_offset(X)

        # Save candidates so we can provide a baseline accuracy report.
        self._wclf_candidates = wclf_candidates

    def report_baseline(self, X, y):
        """Report accuracy of weak classifiers.

        This provides context for interpreting the performance of the boosted
        classifier.
        """
        scores = np.array([accuracy_score(y, clf.predict(X))
                           for clf in self._wclf_candidates])
        data = [[len(scores), scores.min(), scores.mean(), scores.max(), scores.std()]]
        headers = ['count', 'min', 'mean', 'max', 'std']

        print('Accuracy of weak classifiers (score on test set):')
        print(tabulate(data, headers=headers, floatfmt='.3f'))


def qboost_lambda_sweep(X, y, lambda_vals, val_fraction=0.4, verbose=False, **kwargs):
    """Run QBoost using a series of lambda values and check accuracy against a validation set.

    Args:
        X (array):
            2D array of feature vectors.
        y (array):
            1D array of class labels (+/- 1).
        lambda_vals (array):
            Array of values for regularization parameter, lambda.
        val_fraction (float):
            Fraction of given data to set aside for validation.
        verbose (bool):
            Print out diagnostic information to screen.
        kwargs:
            Passed to QBoost.__init__.

    Returns:
        QBoostClassifier:
            QBoost instance with best validation score.
        lambda:
            Lambda value corresponding to the best validation score.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_fraction)

    best_score = -1
    best_lambda = -1
    best_clf = None
    best_features = -1;

    for lam in lambda_vals:
        print('[QBOOST] Sweep: testing lam',lam);
        qb = QBoostClassifier(X_train, y_train, lam, **kwargs)
        score = qb.score(X_val, y_val)
                
        if score > best_score:
            best_score = score
            best_clf = qb
            best_lambda = lam
            best_features = len(qb.get_selected_features())
            
        if verbose:
            print('[QBOOST] Sweep:      lamda: {:<7.6f}      n_features: {:<10}      score: {:<6.3f}'.format(
                lam, len(qb.get_selected_features()), score))
            print('[QBOOST] Sweep: best_lamda: {:<7.6f} best_n_features: {:<10} best_score: {:<6.3f}'.format(
                best_lambda, best_features, best_score))
        

    return best_clf, lam
