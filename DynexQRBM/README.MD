# Dynex QRBM (3-step Qubo)

Implementation of the Quantum-Boltzmann-Machine (QRBM), utilising Dynex' Neuromorphic Computing platform. All training samples are being sampled on the Dynex platform with no Gibbs or CD steps (full QRBM). It is based on https://cmst.eu/articles/applying-a-quantum-annealing-based-restricted-boltzmann-machine-for-mnist-handwritten-digit-classification/

1. sampling probabilities of hidden units on the Dynex platform
2. sampling reconstruction v' from visible units on the Dynex platform
3. resampling hidden activations h' from v' on the Dynex platform

After each training batch (or as configured), a training sample is reconstructed with the Dynex platform and the MSE measured:

reconstructing data to calculate MSE on the Dynex platform
This approach converges extremely fast to a low MSE. The size of the training samples is almost unlimited, but keep in mind that each data sample is being sampled on the Dynex platform.

- [Jupyter Notebook example](https://github.com/dynexcoin/DynexSDK/blob/main/Dynex-Full-QRBM.ipynb)
