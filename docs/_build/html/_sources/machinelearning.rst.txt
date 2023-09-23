Neuromorphic Machine Learning
=====================================
Demonstrates various examples of neuromorphic enhanced Machine Learning techniques, for example Quantum-Boltzmann-Machines (QRBMs>`_, Quantum-Support-Vector-Machines (QSVM>`_ or feature optimisation with QBoost.

Quantum computing algorithms for machine learning harness the power of quantum mechanics to enhance various aspects of machine learning tasks. As both, quantum computing and neuromorphic computing are sharing similar features, these algorithms can also be computed efficiently on the Dynex platform – but without the limitations of limited qubits, error correction or availability:

**Quantum Support Vector Machine (QSVM>`_:** QSVM is a quantum-inspired algorithm that aims to classify data using a quantum kernel function. It leverages the concept of quantum superposition and quantum feature mapping to potentially provide computational advantages over classical SVM algorithms in certain scenarios. 

**Quantum Principal Component Analysis (QPCA>`_:** QPCA is a quantum version of the classical Principal Component Analysis (PCA>`_ algorithm. It utilizes quantum linear algebra techniques to extract the principal components from high-dimensional data, potentially enabling more efficient dimensionality reduction in quantum machine learning.
  
**Quantum Neural Networks (QNN>`_:** QNNs are quantum counterparts of classical neural networks. They leverage quantum principles, such as quantum superposition and entanglement, to process and manipulate data. QNNs hold the potential to learn complex patterns and perform tasks like classification and regression, benefiting from quantum parallelism.

**Quantum K-Means Clustering:** Quantum K-means is a quantum-inspired variant of the classical K-means clustering algorithm. It uses quantum algorithms to accelerate the clustering process by exploring multiple solutions simultaneously. Quantum K-means has the potential to speed up clustering tasks for large-scale datasets. 

**Quantum Boltzmann Machines (QBMs>`_:** QBMs are quantum analogues of classical Boltzmann Machines, which are generative models used for unsupervised learning. QBMs employ quantum annealing to sample from a probability distribution and learn patterns and structures in the data.

**Quantum Support Vector Regression (QSVR>`_:** QSVR extends the concept of QSVM to regression tasks. It uses quantum computing techniques to perform regression analysis, potentially offering advantages in terms of efficiency and accuracy over classical regression algorithms.

Here are some example of these algorithms implemented on the Dynex Platform:

- `Example: Quantum-Support-Vector-Machine Implementation on Dynex <https://github.com/dynexcoin/DynexSDK/blob/main/example_support_vector_machine.ipynb>`_
- `Example: Quantum-Boltzmann-Machine (PyTorch) on Dynex <https://github.com/dynexcoin/DynexSDK/blob/main/example_neuromorphic_torch_layers%20(1).ipynb>`_
- `Example: Quantum-Boltzmann-Machine Implementation (3-step QUBO) on Dynex <https://github.com/dynexcoin/DynexSDK/blob/main/Dynex-Full-QRBM.ipynb>`_
- `Example: Quantum-Boltzmann-Machine (Collaborative Filtering) on Dynex <https://github.com/dynexcoin/DynexSDK/blob/main/example_collaborative_filtering_CFQIRBM.ipynb>`_
- `Example: Quantum-Boltzmann-Machine Implementation on Dynex <https://github.com/dynexcoin/DynexSDK/blob/main/example_quantum_boltzmann_machine_QBM.ipynb>`_
- `Example: Feature Selection - Titanic Survivals <https://github.com/dynexcoin/DynexSDK/blob/main/example_feature_selection_titanic_survivals.ipynb>`_
- `Example: Breast Cancer Prediction using the Dynex scikit-learn Plugin <https://github.com/dynexcoin/DynexSDK/blob/main/Dynex%20Scikit-Learn%20Plugin.ipynb>`_