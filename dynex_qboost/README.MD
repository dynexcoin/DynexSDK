# Dynex QBoost Implementation

The D-Wave quantum computer has been widely studied as a discrete optimization engine that accepts any problem formulated as quadratic unconstrained binary optimization (QUBO). In 2008, Google and D-Wave published a paper, [Training a Binary Classifier with the Quantum Adiabatic Algorithm](https://arxiv.org/pdf/0811.0416.pdf), which describes how the Qboost ensemble method makes binary classification amenable to quantum computing: the problem is formulated as a thresholded linear superposition of a set of weak classifiers and the D-Wave quantum computer is used to optimize the weights in a learning process that strives to minimize the training error and number of weak classifiers

This code provides a QBoost algorithm plugin to use the Dynex Neuromorphic Platform.

## References

[1] Neven, H., Denchev, V. S., Rose, G., and Macready, W. G. Training a Binary Classifier with the Quantum Adiabatic Algorithm, 2008, arXiv:0811.0416v1

[2] Neven, H., Denchev, V. S., Rose, G., and Macready, W. G. QBoost: Large Scale Classifier Training with Adiabatic Quantum Optimization, Journal of Machine Learning Research: Workshop and Conference Proceedings, 2012. URL: http://proceedings.mlr.press/v25/neven12/neven12.pdf.

[3] Mott, A., Job, J., Vlimant, J.-R., Lidar, D., and Spiropulu, M. Solving a Higgs optimization problem with quantum annealing for machine learning. Nature, Vol. 550, 2017, doi:10.1038/nature24047.

[4] Boyda, E., Basu, S., Ganguly, S., Michaelis, A., Mukhopadhyay, S., and Nemani, R. R. Deploying a quantum annealing processor to detect tree cover in aerial imagery of California. PLoS One, 2017, doi:10.1371/journal.pone.0172505.

## License

Released under the Apache License 2.0. See LICENSE file.
