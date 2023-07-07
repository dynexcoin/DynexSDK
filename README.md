# DynexSDK
Dynex is the world’s first neuromorphic supercomputing blockchain based on the DynexSolve chip algorithm, a Proof-of-Useful-Work (PoUW) approach to solving real-world problems. The Dynex SDK is used to interact and compute on the Dynex Platform. All examples require the DynexSDK for Python as well as a valid API key.

## Beginners Guides

- [Example: Computing on the Dynex Platform with Python - BQM](https://github.com/dynexcoin/DynexSDK/blob/main/beginners_guide_example_bqm.ipynb)
- [Example: Computing on the Dynex Platform with Python - BQM K4 Complete Graph](https://github.com/dynexcoin/DynexSDK/blob/main/beginners_guide_example_bqm_k4_complete_graph.ipynb)
- [Example: Computing on the Dynex Platform with Python - Logic Gates](https://github.com/dynexcoin/DynexSDK/blob/main/beginners_guide_example_logic_gates.ipynb)
- [Example: Computing on the Dynex Platform with Python - QUBO](https://github.com/dynexcoin/DynexSDK/blob/main/beginners_guide_example_QUBO.ipynb)
- [Example: Computing on the Dynex Platform with Python - Anti-crossing problem](https://github.com/dynexcoin/DynexSDK/blob/main/beginners_guide_example_anti_crossing_clique.ipynb)
- [Example: Computing on the Dynex Platform with Python - Maximum Independent Set](https://github.com/dynexcoin/DynexSDK/blob/main/beginners_guide_example_MIS.ipynb)
- [Example: Computing on the Dynex Platform with Python - SAT](https://github.com/dynexcoin/DynexSDK/blob/main/beginners_guide_example_SAT.ipynb)
- [Example: Computing on the Dynex Platform with Python - NAE3SAT](https://github.com/dynexcoin/DynexSDK/blob/main/beginners_guide_example_random_nae3sat.ipynb)

## Advanced Examples

- [Example: RNA Folding of the Tobacco Mild Green Mosaic Virus](https://github.com/dynexcoin/DynexSDK/blob/main/example_rna_folding.ipynb)
- [Example: Placement of Charging Stations](https://github.com/dynexcoin/DynexSDK/blob/main/example_placement_of_charging_stations.ipynb)
- [Example: Breast Cancer Prediction using the Dynex scikit-learn Plugin](https://github.com/dynexcoin/DynexSDK/blob/main/Dynex%20Scikit-Learn%20Plugin.ipynb)
- [Example: Quantum Integer Factorization](https://github.com/dynexcoin/DynexSDK/blob/main/example_integer_factorisation.ipynb)

## Machine Learning Examples

Quantum computing algorithms for machine learning harness the power of quantum mechanics to enhance various aspects of machine learning tasks. As both, quantum computing and neuromorphic computing are sharing similar features, these algorithms can also be computed efficiently on the Dynex platform – but without the limitations of limited qubits, error correction or availability:

- Quantum Support Vector Machine (QSVM): QSVM is a quantum-inspired algorithm that aims to classify data using a quantum kernel function. It leverages the concept of quantum superposition and quantum feature mapping to potentially provide computational advantages over classical SVM algorithms in certain scenarios. 

- Quantum Principal Component Analysis (QPCA): QPCA is a quantum version of the classical Principal Component Analysis (PCA) algorithm. It utilizes quantum linear algebra techniques to extract the principal components from high-dimensional data, potentially enabling more efficient dimensionality reduction in quantum machine learning.
  
- Quantum Neural Networks (QNN): QNNs are quantum counterparts of classical neural networks. They leverage quantum principles, such as quantum superposition and entanglement, to process and manipulate data. QNNs hold the potential to learn complex patterns and perform tasks like classification and regression, benefiting from quantum parallelism.

- Quantum K-Means Clustering: Quantum K-means is a quantum-inspired variant of the classical K-means clustering algorithm. It uses quantum algorithms to accelerate the clustering process by exploring multiple solutions simultaneously. Quantum K-means has the potential to speed up clustering tasks for large-scale datasets. 

- Quantum Boltzmann Machines (QBMs): QBMs are quantum analogues of classical Boltzmann Machines, which are generative models used for unsupervised learning. QBMs employ quantum annealing to sample from a probability distribution and learn patterns and structures in the data.

- Quantum Support Vector Regression (QSVR): QSVR extends the concept of QSVM to regression tasks. It uses quantum computing techniques to perform regression analysis, potentially offering advantages in terms of efficiency and accuracy over classical regression algorithms.

- [Example: Quantum-Support-Vector-Machine Implementation on Dynex](https://github.com/dynexcoin/DynexSDK/blob/main/example_support_vector_machine.ipynb)
- [Example: Quantum-Boltzmann-Machine Implementation on Dynex](https://github.com/dynexcoin/DynexSDK/blob/main/example_quantum_boltzmann_machine_QBM.ipynb)
- [Example: Feature Selection - Titanic Survivals](https://github.com/dynexcoin/DynexSDK/blob/main/example_feature_selection_titanic_survivals.ipynb)
- [Example: Breast Cancer Prediction using the Dynex scikit-learn Plugin](https://github.com/dynexcoin/DynexSDK/blob/main/Dynex%20Scikit-Learn%20Plugin.ipynb)

We are working on finalizing the DynexSDK. Come back to check on updates.

## DIMOD: A Shared API for QUBO/ISING Samplers
Dimod is a shared API for samplers. It provides classes for quadratic models—such as the binary quadratic model (BQM) class that contains Ising and QUBO models used by samplers such as the Dynex Neuromorphic Platform or the D-Wave system—and higher-order (non-quadratic) models, reference examples of samplers and composed samplers and abstract base classes for constructing new samplers and composed samplers:

[Dimod documentation](https://docs.ocean.dwavesys.com/en/stable/docs_dimod/)

## PyQUBO: QUBOs or Ising Models from Flexible Mathematical Expressions
PyQUBO allows you to create QUBOs or Ising models from flexible mathematical expressions easily. It is Python based (C++ backend), fully integrated with Ocean SDK, supports automatic validation of constraints and features placeholder for parameter tuning.

[PyQUBO documentation](https://pyqubo.readthedocs.io/)

## Further reading

- [Dynex Website](https://dynexcoin.org/)
- [Dynex for Enterprises](https://dynexcoin.org/learn/dynex-for-enterprises)
- [Dynex SDK](https://dynexcoin.org/learn/dynex-sdk)
- [Dynex SDK Beginner Guides](https://dynexcoin.org/learn/beginner-guides)
- [Dynex SDK Advanced Examples](https://dynexcoin.org/learn/advanced-examples)
