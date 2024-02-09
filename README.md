# DynexSDK
Dynex is the world’s first neuromorphic supercomputing blockchain based on the DynexSolve chip algorithm, a Proof-of-Useful-Work (PoUW) approach to solving real-world problems. The Dynex SDK is used to interact and compute on the Dynex Platform. All examples require the DynexSDK for Python as well as a valid API key. Our repositoriy is continously updated, come back to check on updates.

- [Dynex SDK Repository in Chinese (Community contributed)](https://github.com/DynexCN/DynexCaseCollection)

## Videos

The following videos are available to explain how to use the Dynex SDK:

- [Tutorial: Compute on Dynex: "Hello, world" (using Github CodeSpace)](https://www.youtube.com/watch?v=V46_cOUb9Vo)
- [Tutorial: Compute on Dynex: "Hello, world" (using pip install dynex)](https://www.youtube.com/watch?v=HNUOwEYyTJA)

## Book

**Neuromorphic Computing for Computer Scientists: A complete guide to Neuromorphic Computing on the Dynex Neuromorphic Cloud Computing Platform**, Dynex Developers, 2024, 249 pages, available as eBook, paperback and hardcover

- [Amazon.com](https://www.amazon.com/dp/B0CRQQPBB5)
- [Amazon.co.uk](https://www.amazon.co.uk/dp/B0CRQQPBB5)
- [Amazon.de](https://www.amazon.de/dp/B0CRQQPBB5)

## Getting Started

Download and install the Dynex SDK with the following command:

```
pip install dynex
```

Then follow the steps explained in [Installing the Dynex SDK](https://github.com/dynexcoin/DynexSDK/wiki/Installing-the-Dynex-SDK) to configure the SDK. We suggest to download the [Dynex SDK Hello World Example](https://github.com/dynexcoin/DynexSDK/blob/main/Dynex_hello_world.ipynb) for the first steps of using the Dynex Neuromorphic Platform.

Dynex SDK documentation:
- [Dynex SDK Wiki](https://github.com/dynexcoin/DynexSDK/wiki)
- [Dynex SDK Documentation](https://docs.dynexcoin.org/)

Dynex SDK Professional Community:
- [Dynex Workspace on Slack](https://join.slack.com/t/dynex-workspace/shared_invite/zt-22eb1n4mo-aXS5zsUBoPs613Dofi8Q4A)

Guides:
- [Medium: Computing on the Dynex Neuromorphic Platform: Image Classification](https://dynexcoin.medium.com/computing-on-the-dynex-neuromorphic-platform-image-classification-9b880d7ced9c)
- [Medium: Computing on the Dynex Neuromorphic Platform: IBM Qiskit 4-Qubit Full Adder Circuit](https://medium.com/@dynexcoin/computing-on-the-dynex-neuromorphic-platform-ibm-qiskit-4-qubit-full-adder-circuit-7416084e19dd)
- [Medium: Benchmarking the Dynex Neuromorphic Platform with the Q-Score](https://dynexcoin.medium.com/benchmarking-the-dynex-neuromorphic-platform-with-the-q-score-93402ca19bdd)

Dynex' Scientific Papers:
- [Advancements in Unsupervised Learning: Mode-Assisted Quantum Restricted Boltzmann Machines Leveraging Neuromorphic Computing on the Dynex Platform](https://www.academia.edu/112834961/Advancements_in_Unsupervised_Learning_Mode_Assisted_Quantum_Restricted_Boltzmann_Machines_Leveraging_Neuromorphic_Computing_on_the_Dynex_Platform); Adam Neumann, Dynex Developers; International Journal of Bioinformatics Intelligent Computing. 2024; Volume 3(1):91- 103, ISSN 2816-8089
- [HUBO & QUBO and Prime Factorization](https://www.academia.edu/112871514/HUBO_and_QUBO_and_Prime_Factorization); Samer Rahmeh, Cali Technology Solutions, Dynex Developers; 112871514; International Journal of Bioinformatics Intelligent Computing. 2024; Volume 3(1):91- 103, ISSN 2816-8089
- [Framework for Solving Harrow-Hassidim-Lloyd Problems with Neuromorphic Computing using the Dynex Cloud Computing Platform](https://www.academia.edu/112871175/Framework_for_Solving_Harrow_Hassidim_Lloyd_Problems_with_Neuromorphic_Computing_using_the_Dynex_Cloud_Computing_Platform); Samer Rahmeh, Cali Technology Solutions, Dynex Developers; 112871175; Academia.edu; 2023

## Pricing

Using Dynex technology for computations on the local machine (mainnet=False) is free. It allows sampling of computing problems on the local machine before using the Dynex Neuromorphic Computing cloud and is mainly intended for prototyping and testing of code. Computing on the mainnet is being charged in DNX based on usage. Users can maintain their balances in the [Dynex Market Place](https://live.dynexcoin.org). The cost for compute on Dynex is based on supply & demand, whereas higher paid compute jobs are being prioritized by the workers. The value "CURRENT AVG BLOCK FEE" shows the current average price for compute. It defines the amount to be paid for each block, which is being produced every 2 minutes. Depending on the number of chips (num_reads), duration (annealing_time), size and complexity of your computational problem, only a fraction of the entire network is being used. The price charged for compute is being calculated as a fraction of the base "block fee" and is being displayed during computing in the Python interface as well as in the "Usage" section of the Dynex market place.

## Beginners Guides

To get familiar with the computing possibilities on the Dynex Platform, we have prepared a number of Python Jupyter Notebooks. Here are some of our beginner guides demonstrating the use of the Dynex SDK.

- [Example: Computing on the Dynex Platform with Python - BQM](https://github.com/dynexcoin/DynexSDK/blob/main/beginners_guide_example_bqm.ipynb)
- [Example: Computing on the Dynex Platform with Python - BQM K4 Complete Graph](https://github.com/dynexcoin/DynexSDK/blob/main/beginners_guide_example_bqm_k4_complete_graph.ipynb)
- [Example: Computing on the Dynex Platform with Python - Logic Gates](https://github.com/dynexcoin/DynexSDK/blob/main/beginners_guide_example_logic_gates.ipynb)
- [Example: Computing on the Dynex Platform with Python - QUBO](https://github.com/dynexcoin/DynexSDK/blob/main/beginners_guide_example_QUBO.ipynb)
- [Example: Computing on the Dynex Platform with Python - Anti-crossing problem](https://github.com/dynexcoin/DynexSDK/blob/main/beginners_guide_example_anti_crossing_clique.ipynb)
- [Example: Computing on the Dynex Platform with Python - Maximum Independent Set](https://github.com/dynexcoin/DynexSDK/blob/main/beginners_guide_example_MIS.ipynb)
- [Example: Computing on the Dynex Platform with Python - SAT](https://github.com/dynexcoin/DynexSDK/blob/main/beginners_guide_example_SAT.ipynb)
- [Example: Computing on the Dynex Platform with Python - NAE3SAT](https://github.com/dynexcoin/DynexSDK/blob/main/beginners_guide_example_random_nae3sat.ipynb)

## Advanced Examples

Here are some advanced code examples and notebooks to be used to compute them on the Dynex neuromorphic computing platform:

- [Example: Quantum Computation of Fluid Dynamics (CFD](https://github.com/dynexcoin/QCFD) | Scientific background: An Introduction to Algorithms in Quantum Computation of Fluid Dynamics, Sachin S. Bharadwaj and Katepalli R. Sreenivasan, Department of Mechanical and Aerospace Engineering, STO - Educational Notes Paper, 2022

- [Example: RNA Folding of the Tobacco Mild Green Mosaic Virus](https://github.com/dynexcoin/DynexSDK/blob/main/example_rna_folding.ipynb) | Scientific background: Fox DM, MacDermaid CM, Schreij AMA, Zwierzyna M, Walker RC. RNA folding using quantum computers,. PLoS Comput Biol. 2022 Apr 11;18(4):e1010032. doi: 10.1371/journal.pcbi.1010032. PMID: 35404931; PMCID: PMC9022793

- [Example: Quantum Single Image Super-Resolution](https://github.com/dynexcoin/DynexSDK/tree/main/Quantum-SISR) | Scientific background: Choong HY, Kumar S, Van Gool L. Quantum Annealing for Single Image Super-Resolution. InProceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition 2023 (pp. 1150-1159).

- [Example: Placement of Charging Stations](https://github.com/dynexcoin/DynexSDK/blob/main/example_placement_of_charging_stations.ipynb) | Scientific background: Pagany, Raphaela & Marquardt, Anna & Zink, Roland. (2019). Electric Charging Demand Location Model—A User-and Destination-Based Locating Approach for Electric Vehicle Charging Stations. Sustainability. 11. 2301. 10.3390/su11082301

- [Example: Breast Cancer Prediction using the Dynex scikit-learn Plugin](https://github.com/dynexcoin/DynexSDK/blob/main/Dynex%20Scikit-Learn%20Plugin.ipynb) | Scientific background: Bhatia, H.S., Phillipson, F. (2021). Performance Analysis of Support Vector Machine Implementations on the D-Wave Quantum Annealer. In: Paszynski, M., Kranzlmüller, D., Krzhizhanovskaya, V.V., Dongarra, J.J., Sloot, P.M.A. (eds) Computational Science – ICCS 2021. ICCS 2021. Lecture Notes in Computer Science(), vol 12747. Springer, Cham
  
- [Example: Quantum Integer Factorization](https://github.com/dynexcoin/DynexSDK/blob/main/example_integer_factorisation.ipynb) | Scientific background: Jiang, S., Britt, K.A., McCaskey, A.J. et al. Quantum Annealing for Prime Factorization. Sci Rep 8, 17667 (2018)

- [Example: Enzyme Target Prediction](www.github.com/samgr55/Enzyme-TargetPrediction_QUBO-Ising) | Scientific background: Hoang M Ngo, My T Thai, Tamer Kahveci, QuTIE: Quantum optimization for Target Identification by Enzymes, Bioinformatics Advances, 2023;, vbad112
  
- [Example: Optimal WiFi Hotspot Positioning Prediction](https://github.com/samgr55/OptimalWiFi-HotspotPositioning_QUBO-Ising)

## Machine Learning Examples

Quantum computing algorithms for machine learning harness the power of quantum mechanics to enhance various aspects of machine learning tasks. As both, quantum computing and neuromorphic computing are sharing similar features, these algorithms can also be computed efficiently on the Dynex platform – but without the limitations of limited qubits, error correction or availability:

**Quantum Support Vector Machine (QSVM):** QSVM is a quantum-inspired algorithm that aims to classify data using a quantum kernel function. It leverages the concept of quantum superposition and quantum feature mapping to potentially provide computational advantages over classical SVM algorithms in certain scenarios. 

**Quantum Principal Component Analysis (QPCA):** QPCA is a quantum version of the classical Principal Component Analysis (PCA) algorithm. It utilizes quantum linear algebra techniques to extract the principal components from high-dimensional data, potentially enabling more efficient dimensionality reduction in quantum machine learning.
  
**Quantum Neural Networks (QNN):** QNNs are quantum counterparts of classical neural networks. They leverage quantum principles, such as quantum superposition and entanglement, to process and manipulate data. QNNs hold the potential to learn complex patterns and perform tasks like classification and regression, benefiting from quantum parallelism.

**Quantum K-Means Clustering:** Quantum K-means is a quantum-inspired variant of the classical K-means clustering algorithm. It uses quantum algorithms to accelerate the clustering process by exploring multiple solutions simultaneously. Quantum K-means has the potential to speed up clustering tasks for large-scale datasets. 

**Quantum Boltzmann Machines (QBMs):** QBMs are quantum analogues of classical Boltzmann Machines, which are generative models used for unsupervised learning. QBMs employ quantum annealing to sample from a probability distribution and learn patterns and structures in the data.

**Quantum Support Vector Regression (QSVR):** QSVR extends the concept of QSVM to regression tasks. It uses quantum computing techniques to perform regression analysis, potentially offering advantages in terms of efficiency and accuracy over classical regression algorithms.

Here are some example of these algorithms implemented on the Dynex Platform:

- [Example: Quantum-Support-Vector-Machine Implementation on Dynex](https://github.com/dynexcoin/DynexSDK/blob/main/example_support_vector_machine.ipynb) | Scientific background: Rounds, Max and Phil Goddard. “Optimal feature selection in credit scoring and classification using a quantum annealer.” (2017)

- [Example: Quantum-Support-Vector-Machine (PyTorch) on Dynex](https://github.com/dynexcoin/DynexSDK/blob/main/Example_SVM_pytorch.ipynb) | Scientific background: Rounds, Max and Phil Goddard. “Optimal feature selection in credit scoring and classification using a quantum annealer.” (2017)

- [Example: Quantum-Boltzmann-Machine (PyTorch) on Dynex](https://github.com/dynexcoin/DynexSDK/blob/main/example_neuromorphic_torch_layers%20(1).ipynb) | Scientific background: Dixit V, Selvarajan R, Alam MA, Humble TS and Kais S (2021) Training Restricted Boltzmann Machines With a D-Wave Quantum Annealer. Front. Phys. 9:589626. doi: 10.3389/fphy.2021.589626; Sleeman, Jennifer, John E. Dorband and Milton Halem. “A Hybrid Quantum enabled RBM Advantage: Convolutional Autoencoders For Quantum Image Compression and Generative Learning.” Defense + Commercial Sensing (2020)

- [Example: Quantum-Boltzmann-Machine Implementation (3-step QUBO) on Dynex](https://github.com/dynexcoin/DynexSDK/blob/main/Dynex-Full-QRBM.ipynb) | Scientific background: Dixit V, Selvarajan R, Alam MA, Humble TS and Kais S (2021) Training Restricted Boltzmann Machines With a D-Wave Quantum Annealer. Front. Phys. 9:589626. doi: 10.3389/fphy.2021.589626; Sleeman, Jennifer, John E. Dorband and Milton Halem. “A Hybrid Quantum enabled RBM Advantage: Convolutional Autoencoders For Quantum Image Compression and Generative Learning.” Defense + Commercial Sensing (2020)

- [Example: Quantum-Boltzmann-Machine (Collaborative Filtering) on Dynex](https://github.com/dynexcoin/DynexSDK/blob/main/example_collaborative_filtering_CFQIRBM.ipynb) | Scientific background: Dixit V, Selvarajan R, Alam MA, Humble TS and Kais S (2021) Training Restricted Boltzmann Machines With a D-Wave Quantum Annealer. Front. Phys. 9:589626. doi: 10.3389/fphy.2021.589626; Sleeman, Jennifer, John E. Dorband and Milton Halem. “A Hybrid Quantum enabled RBM Advantage: Convolutional Autoencoders For Quantum Image Compression and Generative Learning.” Defense + Commercial Sensing (2020)

- [Example: Quantum-Boltzmann-Machine Implementation on Dynex](https://github.com/dynexcoin/DynexSDK/blob/main/example_quantum_boltzmann_machine_QBM.ipynb) | Scientific background: Dixit V, Selvarajan R, Alam MA, Humble TS and Kais S (2021) Training Restricted Boltzmann Machines With a D-Wave Quantum Annealer. Front. Phys. 9:589626. doi: 10.3389/fphy.2021.589626; Sleeman, Jennifer, John E. Dorband and Milton Halem. “A Hybrid Quantum enabled RBM Advantage: Convolutional Autoencoders For Quantum Image Compression and Generative Learning.” Defense + Commercial Sensing (2020)

- [Example: Mode-assisted unsupervised learning of restricted Boltzmann machines (MA-QRBM for Pytorch)](https://github.com/dynexcoin/DynexSDK/tree/main/MAQRBM) | Scientific background: Mode-assisted unsupervised learning of restricted Boltzmann machines, Communications Physics volume 3, Article number:105 (2020)

- [Example: Mode-assisted unsupervised learning of restricted Boltzmann machines (MA-QRBM for Tensorflow)](https://github.com/dynexcoin/QRBM_Tensorflow) | Scientific background: Mode-assisted unsupervised learning of restricted Boltzmann machines, Communications Physics volume 3, Article number:105 (2020)

- [Example: Feature Selection - Titanic Survivals](https://github.com/dynexcoin/DynexSDK/blob/main/example_feature_selection_titanic_survivals.ipynb) | Scientific background: Xuan Vinh Nguyen, Jeffrey Chan, Simone Romano, and James Bailey. 2014. Effective global approaches for mutual information based feature selection. In Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining (KDD '14). Association for Computing Machinery, New York, NY, USA, 512–521

- [Example: Breast Cancer Prediction using the Dynex scikit-learn Plugin](https://github.com/dynexcoin/DynexSDK/blob/main/Dynex%20Scikit-Learn%20Plugin.ipynb) | Scientific background: Bhatia, H.S., Phillipson, F. (2021). Performance Analysis of Support Vector Machine Implementations on the D-Wave Quantum Annealer. In: Paszynski, M., Kranzlmüller, D., Krzhizhanovskaya, V.V., Dongarra, J.J., Sloot, P.M.A. (eds) Computational Science – ICCS 2021. ICCS 2021. Lecture Notes in Computer Science(), vol 12747. Springer, Cham

## Dynex Neuromorphic Torch Layers

The Dynex Neuromorphic Torch layer can be used in any NN model. Welcome to hybrid models, neuromorphic-, transfer- and federated-learning with 
[PyTorch](https://pytorch.org/)
 
- [Example: Quantum-Boltzmann-Machine (PyTorch) on Dynex](https://github.com/dynexcoin/DynexSDK/blob/main/example_neuromorphic_torch_layers%20(1).ipynb) | Scientific background: Dixit V, Selvarajan R, Alam MA, Humble TS and Kais S (2021) Training Restricted Boltzmann Machines With a D-Wave Quantum Annealer. Front. Phys. 9:589626. doi: 10.3389/fphy.2021.589626; Sleeman, Jennifer, John E. Dorband and Milton Halem. “A Hybrid Quantum enabled RBM Advantage: Convolutional Autoencoders For Quantum Image Compression and Generative Learning.” Defense + Commercial Sensing (2020)

- [Example: Quantum-Support-Vector-Machine (PyTorch) on Dynex](https://github.com/dynexcoin/DynexSDK/blob/main/Example_SVM_pytorch.ipynb) | Scientific background: Rounds, Max and Phil Goddard. “Optimal feature selection in credit scoring and classification using a quantum annealer.” (2017)

## Dynex Neuromorphic TensorFlow Layers

The Dynex Neuromorphic Torch layer can be used in any NN model. Welcome to hybrid models, neuromorphic-, transfer- and federated-learning with 
[TensorFlow](https://www.tensorflow.org/)

- [Example: Quantum-Support-Vector-Machine (TensorFlow) on Dynex](https://github.com/dynexcoin/QSVM_Tensorflow/blob/main/Example_SVM_Tensorflow.ipynb) | Scientific background: Rounds, Max and Phil Goddard. “Optimal feature selection in credit scoring and classification using a quantum annealer.” (2017)

- [Example: Mode-assisted unsupervised learning of restricted Boltzmann machines (MA-QRBM for Tensorflow)](https://github.com/dynexcoin/QRBM_Tensorflow) | Scientific background: Mode-assisted unsupervised learning of restricted Boltzmann machines, Communications Physics volume 3, Article number:105 (2020)

## Dynex Qiskit Package

Thanks to groundbreaking research from [Richard H. Warren](https://arxiv.org/pdf/1405.2354.pdf), it is possible to directly translate Qiskit quantum circuits into Dynex Neuromorphic chips. The concept behind is a direct translation of Qiskit objects, but instead of running on IBM Q, the circuits are executed on the Dynex Neuromorphic platform. Here is an example of a one-qubit adder circuit using this approach:

```
from dynexsdk.qiskit import QuantumRegister, ClassicalRegister
from dynexsdk.qiskit import QuantumCircuit, execute

# Input Registers: a = qi[0]; b = qi[1]; ci = qi[2]
qi = QuantumRegister(3)
ci = ClassicalRegister(3)

# Output Registers: s = qo[0]; co = qo[1]
qo = QuantumRegister(2)
co = ClassicalRegister(2)
circuit = QuantumCircuit(qi,qo,ci,co)

# Define adder circuit
for idx in range(3):
    circuit.ccx(qi[idx], qi[(idx+1)%3], qo[1])
for idx in range(3):
    circuit.cx(qi[idx], qo[0])

circuit.measure(qo, co)

# Run
execute(circuit)

# Print
print(circuit)
```

[Dynex Qiskit Package (Github)](https://github.com/dynexcoin/Dynex-Qiskit)

## Dynex scikit-learn Plugin

This package provides a scikit-learn transformer for feature selection using the Dynex Neuromorphic Computing Platform. It is built to integrate seamlessly with scikit-learn, an industry-standard, state-of-the-art ML library for Python.

The [Dynex scikit-learn Plugin](https://github.com/dynexcoin/DynexSDK/tree/main/dynex_scikit_plugin) makes it easier to use the Dynex platform for the feature selection piece of ML workflows. Feature selection – a key building block of machine learning – is the problem of determining a small set of the most representative characteristics to improve model training and performance in ML. With this new plug-in, ML developers need not be experts in optimization or hybrid solving to get the business and technical benefits of both. Developers creating feature selection applications can build a pipeline with scikit-learn and then embed the Dynex Platform into this workflow more easily and efficiently. ​


## Dynex QBoost Implementation

The D-Wave quantum computer has been widely studied as a discrete optimization engine that accepts any problem formulated as quadratic unconstrained binary optimization (QUBO). In 2008, Google and D-Wave published a paper, [Training a Binary Classifier with the Quantum Adiabatic Algorithm](https://arxiv.org/pdf/0811.0416.pdf), which describes how the Qboost ensemble method makes binary classification amenable to quantum computing: the problem is formulated as a thresholded linear superposition of a set of weak classifiers and the D-Wave quantum computer is used to optimize the weights in a learning process that strives to minimize the training error and number of weak classifiers

The [Dynex QBoost Implementation](https://github.com/dynexcoin/DynexSDK/tree/main/dynex_qboost) provides a QBoost algorithm plugin to use the Dynex Neuromorphic Platform.

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

## License

LICENSED UNDER GNU GENERAL PUBLIC LICENSE Version 3. SEE LICENSE FILE IN THE DYNEX PACKAGE
