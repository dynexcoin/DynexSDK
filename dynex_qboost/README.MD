# Dynex QBoost Implementation

The D-Wave quantum computer has been widely studied as a discrete optimization engine that accepts any problem formulated as quadratic unconstrained binary optimization (QUBO). In 2008, Google and D-Wave published a paper, "Training a Binary Classifier with the Quantum Adiabatic Algorithm"[1], which describes how the Qboost ensemble method makes binary classification amenable to quantum computing: the problem is formulated as a thresholded linear superposition of a set of weak classifiers and the D-Wave quantum computer is used to optimize the weights in a learning process that strives to minimize the training error and number of weak classifiers

This code uses the Dynex Neuromorphic Platform with the Qboost algorithm.
