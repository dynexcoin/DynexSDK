# Dynex QSVM PyTorch Layer

This class is a custom PyTorch layer which can be seamless used in Machine Learning & Artificial Intelligence models.

## Example

In this example a classical classiﬁcation model, Kernel-Support Vector machine, is implemented as a Quadratic Unconstrained Binary Optimisation problem. Here, data points are classiﬁed by a separating hyperplane while maximizing the function margin. The problem is solved for a public Banknote Authentication dataset. In [1] the authors performed benchmark tests for a Quantum-SVM on multiple D-Wave Quantum machines. This examples significantly outperforms all other tested sytems (D-Wave Quantum Computer, HQPU, QPU), Scikit-Learn, Simulated Annealing) with achieving 100.00% in all tested metrics (f-score, precision, recall, accuracy):

- [Example Usage](https://github.com/dynexcoin/DynexSDK/blob/main/Example_SVM_pytorch.ipynb)
