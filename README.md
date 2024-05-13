# What is Neuromorphic Quantum Computing (in simple terms)? 
Neuromorphic quantum computing is a special type of computing that combines ideas from brain-like computing with quantum technology to solve problems. It works differently from regular quantum computing by using a network of connected components that can quickly react to changes, helping the system to swiftly find the best solutions. This setup mimics natural processes like those seen in the human brain and can also simulate aspects of quantum physics like tunneling but uses everyday electrical behavior instead. This means it can be simulated on current computers and built with usual electrical parts, making it potentially more practical for real-world problems. This technology is exciting because it leads to computer systems that are faster and capable of handling complex tasks more efficiently than traditional computers. 

- [Technological background](https://dynexcoin.org/learn/n-quantum-computing)

# DynexSDK
![Dynex SDK](https://github.com/dynexcoin/website/blob/main/dynexsdk.png)
Customers can run computations on the decentralised Dynex n.quantum computing cloud, which is empowered by a growing number of contributing workers. These are miners who are running the proprietary Proof-of-Useful-Work (PoUW) algorithm DynexSolve. Dynex’s proprietary job management and scheduling system Dynex Mallob ensures that computing jobs are being distributed and computed in the fastest way possible. The Dynex SDK is a Python package which is used to compute on the Dynex platform.

- [Dynex Benchmarks](https://dynexcoin.org/learn/dynex-benchmark-(q-score))

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
- [Medium: Real World Use Case: Stock Portfolio Optimisation with Quantum Algorithms on the Dynex Platform](https://dynexcoin.medium.com/real-world-use-case-stock-portfolio-optimisation-with-quantum-algorithms-on-the-dynex-platform-3c6b2a6c559f)
- [Medium: Computing on the Dynex Neuromorphic Platform: Image Classification](https://dynexcoin.medium.com/computing-on-the-dynex-neuromorphic-platform-image-classification-9b880d7ced9c)
- [Medium: Computing on the Dynex Neuromorphic Platform: IBM Qiskit 4-Qubit Full Adder Circuit](https://medium.com/@dynexcoin/computing-on-the-dynex-neuromorphic-platform-ibm-qiskit-4-qubit-full-adder-circuit-7416084e19dd)
- [Medium: Benchmarking the Dynex Neuromorphic Platform with the Q-Score](https://dynexcoin.medium.com/benchmarking-the-dynex-neuromorphic-platform-with-the-q-score-93402ca19bdd)
- [Medium: Enhancing MaxCut Solutions: Dynex’s Benchmark Performance on G70 Using Quantum Computing](https://dynexcoin.medium.com/enhancing-maxcut-solutions-dynexs-benchmark-performance-on-g70-using-quantum-computing-e5340b2197a6)
- [Medium: Dynex Sets New Record for Quantum Computing, Breaking NVIDIA’s Previous Record](https://dynexcoin.medium.com/benchmarking-the-dynex-neuromorphic-platform-with-the-q-score-93402ca19bdd)

Dynex' Scientific Papers:
- [Advancements in Unsupervised Learning: Mode-Assisted Quantum Restricted Boltzmann Machines Leveraging Neuromorphic Computing on the Dynex Platform](https://doi.org/10.61797/ijbic.v3i1.300); Adam Neumann, Dynex Developers; International Journal of Bioinformatics & Intelligent Computing. 2024; Volume 3(1):91- 103, ISSN 2816-8089
- [HUBO & QUBO and Prime Factorization](https://doi.org/10.61797/ijbic.v3i1.301); Samer Rahmeh, Cali Technology Solutions, Dynex Developers; International Journal of Bioinformatics & Intelligent Computing. 2024; Volume 3(1):45-69, ISSN 2816-8089
- [Framework for Solving Harrow-Hassidim-Lloyd Problems with Neuromorphic Computing using the Dynex Cloud Computing Platform](https://www.academia.edu/112871175/Framework_for_Solving_Harrow_Hassidim_Lloyd_Problems_with_Neuromorphic_Computing_using_the_Dynex_Cloud_Computing_Platform); Samer Rahmeh, Cali Technology Solutions, Dynex Developers; 112871175; Academia.edu; 2023
- [Quantum Frontiers on Dynex: Elevating Deep Restricted Boltzmann Machines with Quantum Mode-Assisted Training](https://www.academia.edu/116660843/Quantum_Frontiers_on_Dynex_Elevating_Deep_Restricted_Boltzmann_Machines_with_Quantum_Mode_Assisted_Training); Adam Neumann, Dynex Developers; 116660843, Academia.edu; 2024

## Pricing

Using Dynex technology for computations on the local machine (mainnet=False) is free. It allows sampling of computing problems on the local machine before using the Dynex Neuromorphic Computing cloud and is mainly intended for prototyping and testing of code. Computing on the mainnet is being charged in DNX based on usage. Users can maintain their balances in the [Dynex Market Place](https://live.dynexcoin.org). The cost for compute on Dynex is based on supply & demand, whereas higher paid compute jobs are being prioritized by the workers. The value "CURRENT AVG BLOCK FEE" shows the current average price for compute. It defines the amount to be paid for each block, which is being produced every 2 minutes. Depending on the number of chips (num_reads), duration (annealing_time), size and complexity of your computational problem, only a fraction of the entire network is being used. The price charged for compute is being calculated as a fraction of the base "block fee" and is being displayed during computing in the Python interface as well as in the "Usage" section of the Dynex market place.

The Dynex SDK provides the following method to estimate the actual costs for a computing job before sampling it on the main job:

```
model = dynex.BQM(bqm); 
dynex.estimate_costs(model, num_reads=10000);

[DYNEX] AVERAGE BLOCK FEE: 282.59 DNX
[DYNEX] SUBMITTING COMPUTE FILE FOR COST ESTIMATION...
[DYNEX] COST OF COMPUTE: 0.537993485 DNX PER BLOCK
[DYNEX] COST OF COMPUTE: 0.268996742 DNX PER MINUTE
```

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

## Advanced Examples (Pharmaceutical)

As quantum computing technology matures and becomes more accessible, its integration into the pharmaceutical industry is poised to usher in a new era of medical innovation. The promise of faster, more efficient drug discovery and development, coupled with the potential for personalized treatments, positions quantum computing as a key driver of future advancements in healthcare and medicine, marking a significant leap forward in our ability to combat disease and improve human health.

- [Example: Quantum Protein Folding](https://github.com/dynexcoin/DynexSDK/blob/main/QuantumProteinFolding.ipynb) | Scientific background: Irbäck, Anders & Knuthson, Lucas & Mohanty, Sandipan & Peterson, Carsten. (2022). Folding lattice proteins with quantum annealing.

- [Example: Quantum RNA Folding of the Tobacco Mild Green Mosaic Virus](https://github.com/dynexcoin/DynexSDK/blob/main/example_rna_folding.ipynb) | Scientific background: Fox DM, MacDermaid CM, Schreij AMA, Zwierzyna M, Walker RC. RNA folding using quantum computers,. PLoS Comput Biol. 2022 Apr 11;18(4):e1010032. doi: 10.1371/journal.pcbi.1010032. PMID: 35404931; PMCID: PMC9022793

- [Example: Efficient Exploration of Phenol Derivatives](https://github.com/dynexcoin/DynexSDK/blob/main/molecule_screening.ipynb) | Scientific background: Efficient Exploration of Phenol Derivatives Using QUBO Solvers with Group Contribution-Based Approaches; Chien-Hung Cho, Jheng-Wei Su, Lien-Po Yu, Ching-Ray Chang, Pin-Hong Chen, Tzu-Wei Lin, Shin-Hong Liu, Tsung-Hui Li, and Ying-Yuan Lee; Industrial & Engineering Chemistry Research 2024 63 (10), 4248-4256; DOI: 10.1021/acs.iecr.3c03331

- [Example: Enzyme Target Prediction](www.github.com/samgr55/Enzyme-TargetPrediction_QUBO-Ising) | Scientific background: Hoang M Ngo, My T Thai, Tamer Kahveci, QuTIE: Quantum optimization for Target Identification by Enzymes, Bioinformatics Advances, 2023;, vbad112

- [Example: Breast Cancer Prediction using the Dynex scikit-learn Plugin](https://github.com/dynexcoin/DynexSDK/blob/main/Dynex%20Scikit-Learn%20Plugin.ipynb) | Scientific background: Bhatia, H.S., Phillipson, F. (2021). Performance Analysis of Support Vector Machine Implementations on the D-Wave Quantum Annealer. In: Paszynski, M., Kranzlmüller, D., Krzhizhanovskaya, V.V., Dongarra, J.J., Sloot, P.M.A. (eds) Computational Science – ICCS 2021. ICCS 2021. Lecture Notes in Computer Science(), vol 12747. Springer, Cham

## Advanced Examples (Automotive, Aerospace & Space)

Quantum computing is set to revolutionize the automotive industry by accelerating advancements in design, safety, efficiency, and sustainability. Among its most promising applications is the enhancement of Computational Fluid Dynamics (CFD), a critical tool in vehicle design and optimization. With its unparalleled computational power, quantum computing can significantly speed up CFD simulations, enabling engineers to rapidly analyze and optimize the aerodynamic performance of vehicles.

- [Example: Quantum Computation of Fluid Dynamics (CFD](https://github.com/dynexcoin/QCFD) | Scientific background: An Introduction to Algorithms in Quantum Computation of Fluid Dynamics, Sachin S. Bharadwaj and Katepalli R. Sreenivasan, Department of Mechanical and Aerospace Engineering, STO - Educational Notes Paper, 2022

- [Example: Quantum Satellite Positioning](https://github.com/dynexcoin/DynexSDK/blob/main/QuantumSatellite.ipynb) | Scientific background: G. Bass, C. Tomlin, V. Kumar, P. Rihaczek, J. Dulny III. Heterogeneous Quantum Computing for Satellite Constellation Optimization: Solving the Weighted K-Clique Problem. 2018 Quantum Sci. Technol. 3 024010.
  
- [Example: Aircraft Loading Optimisation](https://github.com/dynexcoin/DynexSDK/blob/main/aircraft-loading-optim.ipynb) | Airbus Quantum Computing Challenge; Problem Statement n°5

- [Example: Placement of Charging Stations](https://github.com/dynexcoin/DynexSDK/blob/main/example_placement_of_charging_stations.ipynb) | Scientific background: Pagany, Raphaela & Marquardt, Anna & Zink, Roland. (2019). Electric Charging Demand Location Model—A User-and Destination-Based Locating Approach for Electric Vehicle Charging Stations. Sustainability. 11. 2301. 10.3390/su11082301

## Advanced Examples (Financial Services)

Quantum computing represents a transformative leap forward for the financial services industry, poised to redefine the landscapes of risk management, fraud detection, portfolio optimization, and beyond with unparalleled computational power. By harnessing the principles of quantum mechanics, financial institutions can unlock new potentials in analyzing vast datasets, optimizing asset allocations, and executing transactions with groundbreaking speed and precision.

- [Example: Quantum Portfolio Optimisation](https://github.com/dynexcoin/DynexSDK/blob/main/Dynex_Portfolio_Optimisation.ipynb) | Scientific Background: Sakuler, Wolfgang & Oberreuter, Johannes & Aiolfi, Riccardo & Asproni, Luca & Roman, Branislav & Schiefer, Jürgen. (2023). A real world test of Portfolio Optimization with Quantum Annealing. 10.21203/rs.3.rs-3959774/v1. 

## Advanced Examples (Telecommunication)

As quantum computing technology continues to mature, its application in the telecommunications sector could usher in a new era of ultra-fast, secure, and efficient communication networks. This evolution will not only enhance the way we connect with each other but also enable the development of future technologies that depend on robust and secure communication infrastructures.

- [Example: Optimal WiFi Hotspot Positioning Prediction](https://github.com/samgr55/OptimalWiFi-HotspotPositioning_QUBO-Ising)

## Advanced Examples (Algorithms)

- [Example: Enhancing MaxCut Solutions: Dynex’s Benchmark Performance on G70 Using Quantum Computing](https://github.com/dynexcoin/DynexSDK/blob/main/maxcut/G70_dynex.ipynb)

- [Example: Quantum Single Image Super-Resolution](https://github.com/dynexcoin/DynexSDK/tree/main/Quantum-SISR) | Scientific background: Choong HY, Kumar S, Van Gool L. Quantum Annealing for Single Image Super-Resolution. InProceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition 2023 (pp. 1150-1159).
  
- [Example: Quantum Integer Factorization](https://github.com/dynexcoin/DynexSDK/blob/main/example_integer_factorisation.ipynb) | Scientific background: Jiang, S., Britt, K.A., McCaskey, A.J. et al. Quantum Annealing for Prime Factorization. Sci Rep 8, 17667 (2018)

- [Example: Quantum Sudoku Algorithm](https://github.com/dynexcoin/DynexSDK/blob/main/QuantumSudoku.ipynb) | Scientific background: Timothy Resnick, Sudoku at the Intersection of Classical and Quantum Computing, University of Auckland, NZ, Centre for Discrete Mathematics and Theoretical Computer Science

## Advanced Examples (Machine Learning)

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

## AutoQUBO: Automated Conversion from Python functions to QUBO
AUTOmated QUBO Generator (by Fujitsu Research) is an automatic tool for converting a high-level description of an optimization problem, written in Python, into an equivalent QUBO representation. It is doing this by using a novel data driven translation method that can completely decouple the input and output representation. The QUBO framework provides a way to model, in principle, any combinatorial optimization problem and enables the use of Ising machines, like available on the Dynex Platform, to solve it. It introduces symbolic sampling, which provides QUBO formulations for entire problem classes.

[AutoQUBO on the Dynex Platform](https://github.com/dynexcoin/autoqubo)

## Qubolite: light-weight toolbox for working with QUBO instances in NumPy
Quantum Computing (QC) has ushered in a new era of computation, promising to solve problems that are practically infeasible for classical computers. One of the most exciting applications of quantum computing is its ability of solving combinatorial optimization problems, such as Quadratic Unconstrained Binary Optimization (QUBO). This problem class has regained significant attention with the advent of Quantum Computing. These hard-to-solve combinatorial problems appear in many different domains, including finance, logistics, Machine Learning and Data Mining. To harness the power of Quantum Computing for QUBO, The Lamarr Institute introduced qubolite, a Python package comprising utilities for creating, analyzing, and solving QUBO instances, which incorporates current research algorithms developed by scientists at the Lamarr Institute. Qubolite is a light-weight toolbox for working with QUBO instances in NumPy. This fork showcases the use of Qubolite to compute on the Dynex Neuromorphic computing platform.

[Qubolite on the Dynex Platform](https://github.com/dynexcoin/qubolite)


## Further reading

- [Dynex Website](https://dynexcoin.org/)
- [Dynex for Enterprises](https://dynexcoin.org/learn/dynex-for-enterprises)
- [Dynex SDK](https://dynexcoin.org/learn/dynex-sdk)
- [Dynex SDK Beginner Guides](https://dynexcoin.org/learn/beginner-guides)
- [Dynex SDK Advanced Examples](https://dynexcoin.org/learn/advanced-examples)

## License

LICENSED UNDER GNU GENERAL PUBLIC LICENSE Version 3. SEE LICENSE FILE IN THE DYNEX PACKAGE
