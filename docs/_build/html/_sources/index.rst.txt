.. Dynex SDK documentation master file, created by
   sphinx-quickstart on Sat Sep 23 17:55:25 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Dynex SDK's documentation!
=====================================

.. toctree::
   :maxdepth: 2
   :caption: Dynex SDK Modules:

   Dynex SDK <modules>
   Getting started <welcome>
   Guides <guides>
   Using the Dynex SDK <using>
   Sampler Properties and Parameters <sampler>
   Problem-Solving Handbook <handbook>
   Neuromorphic Machine Learning <machinelearning>
   Dynex PyTorch Library <pytorch>
   Dynex Scikit-Learn Plugin <scikit>
 
Getting Started with the Dynex SDK
=====================================
Introduces the Dynex Neuromorphic Computing Platform and the Dynex SDK, describes the basics of how it works, and explains with simple examples how to use it to solve problems and how to use it for Machine Learning.

You can also join the `Dynex workspace on Slack <https://join.slack.com/t/dynex-workspace/shared_invite/zt-22eb1n4mo-aXS5zsUBoPs613Dofi8Q4A>`_ to learn more and to interact with the developer community.


- `Welcome to the Dynex Platform <https://github.com/dynexcoin/DynexSDK/wiki/Welcome-to-the-Dynex-Platform>`_
- `Workflow: Formulation and Sampling <https://github.com/dynexcoin/DynexSDK/wiki/Workflow:-Formulation-and-Sampling>`_
- `What is Neuromorphic Computing? <https://github.com/dynexcoin/DynexSDK/wiki/What-is-Neuromorphic-Computing%3F>`_
- `Solving Problems With the Dynex SDK <https://github.com/dynexcoin/DynexSDK/wiki/Solving-Problems-with-the-Dynex-SDK>`_
- `Simple Sampling Examples <https://github.com/dynexcoin/DynexSDK/wiki/Simple-Sampling-Examples>`_
- `Appendix: Next Learning Steps <https://github.com/dynexcoin/DynexSDK/wiki/Appendix:-Next-Learning-Steps>`_

Guides
=====================================
Computing on Quantum or neuromorphic systems is fundamentally different than using traditional hardware and is a very active area of research with new algorithms surfacing almost on a weekly basis. Our guides are step-by-step instructions on how to utilise neuromorphic computing with the Dynex SDK. These examples are just some of multiple possibilities to perform machine learning tasks. However, they can be easily adopted to other use cases.

- `Medium: Computing on the Dynex Neuromorphic Platform: Image Classification <https://dynexcoin.medium.com/computing-on-the-dynex-neuromorphic-platform-image-classification-9b880d7ced9c>`_

Using the Dynex SDK 
=====================================
Introduces the Dynex SDK and provides references to usage information. The Dynex Platform is intended to solve arbitrary application problems formulated as quadratic models. Problems submitted directly to the Dynex Platform are in the **binary quadratic model** (BQM>`_ format, unconstrained with binary-valued variables and structured for the topology of the Dynex Chips. They also accept arbitrarily structured quadratic models (QM>`_, constrained or unconstrained, with real, integer, and binary variables. The Dynex Platform, which implement state-of-the-art neuromorphic computations, is designed to accommodate even very large problems.

- `Installing the Dynex SDK <https://github.com/dynexcoin/DynexSDK/wiki/Installing-the-Dynex-SDK>`_
- `License <https://github.com/dynexcoin/DynexSDK/wiki/License>`_
- `Example Usage <https://github.com/dynexcoin/DynexSDK/wiki/Example-Usage>`_

Sampler Properties and Parameters
=====================================
This guide details limitations on problem size, the range of time allowed for solving problems, etc and input arguments that enable you to configure execution.

- `Defining a Model <https://github.com/dynexcoin/DynexSDK/wiki/Defining-a-Model>`_
- `Sampling Models <https://github.com/dynexcoin/DynexSDK/wiki/Sampling-Models>`_
- `Parallel Sampling <https://github.com/dynexcoin/DynexSDK/wiki/Parallel-Sampling>`_

Problem-Solving Handbook
=====================================
Provides advanced guidance on using the Dynex Platform, in particular the Dynex SDK. It lists, explains, and demonstrates techniques of problem formulation and configuring parameters to optimize performance.

- `Advanced Examples <https://github.com/dynexcoin/DynexSDK/wiki/Advanced-Examples>`_
- `112 Optimization Problems and their QUBO formulations <https://github.com/dynexcoin/DynexSDK/wiki/112-Optimization-Problems-and-their-QUBO-formulations>`_

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


Dynex PyTorch Library
=====================================
Details how to work with Dynex Neuromorphic torch layers, which can be used in any NN model, in PyTorch. Examples include hybrid models, neuromorphic-, transfer- and federated-learning.

- `Using a Dynex PyTorch layer <https://github.com/dynexcoin/DynexSDK/blob/main/dynex_pytorch/example_neuromorphic_torch_layers.ipynb>`_

Dynex Scikit-Learn Plugin
=====================================
This package provides a scikit-learn transformer for feature selection using the Dynex Neuromorphic Computing Platform. It is built to integrate seamlessly with scikit-learn, an industry-standard, state-of-the-art ML library for Python. This plug-in makes it easier to use the Dynex platform for the feature selection piece of ML workflows. Feature selection – a key building block of machine learning – is the problem of determining a small set of the most representative characteristics to improve model training and performance in ML. With this new plug-in, ML developers need not be experts in optimization or hybrid solving to get the business and technical benefits of both. Developers creating feature selection applications can build a pipeline with scikit-learn and then embed the Dynex Platform into this workflow more easily and efficiently. ​The package's main class, SelectFromQuadraticModel, can be used in any existing sklearn pipeline.

- `Using the Dynex Scikit-Learn Plugin <https://github.com/dynexcoin/DynexSDK/blob/main/Dynex%20Scikit-Learn%20Plugin.ipynb>`_

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`



