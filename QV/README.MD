# Quantum Volume Measurement on Dynex
Quantum Volume (QV) is a holistic benchmark that measures the performance of quantum computers, accounting for both gate fidelity and circuit complexity. Achieving a high QV is essential for demonstrating quantum advantage over classical systems. In this paper, we present a detailed account of computing a Quantum Volume of 2^119 using the Dynex neuromorphic quantum computing platform. We describe the methodologies employed, the modifications made to standard QV testing protocols to accommodate the Dynex architecture, and the results obtained. Our work showcases the scalability and computational capabilities of the Dynex platform in handling large-scale quantum computations.

- Paper: [Computing a Quantum Volume of 2^119 Using the Dynex Neuromorphic Quantum Computing Platform](https://www.academia.edu/124276903/Computing_a_Quantum_Volume_of_2_119_Using_the_Dynex_Neuromorphic_Quantum_Computing_Platform); Samer Rahmeh, Head of Quantum Solutions Architecture, Dynex; Adam Neumann, Dynex Developers; 124276903; Academia.edu; 2024
- [Youtube: Real-time screen recording of the measurement](https://youtu.be/eEiu8e8xlMo) *updated

# Quantum Volume Measurement Approach (updated)
#### Samer Rahmeh | Global Head of Quantum Solutions Architecture | Dynex Development Est.
In this modified Quantum Volume (QV) testing, we will attempt to compute QV with *qubits = 119* (i.e., a Quantum Volume of $2^{119}$. The modified changes here are:

- *Use of SU(4) Unitaries on Random Qubit Pairs*:
  - We generate random SU(4) unitaries and apply them to random pairs of qubits.
  - This introduces entanglement and increases circuit complexity, adhering to the proper QV test requirements.

- *Separate Computation of Ideal Outputs*:
  - We compute the ideal output probabilities using the 'probs' method in dynex_circuit.execute.
  - This allows us to obtain the full probability distribution over all $2^{119}$ possible outputs for proper comparison.

- *Correct Calculation of Heavy Outputs*:
  - Heavy outputs are identified based on the combined $2^N$ output probabilities, not on individual qubit probabilities.
  - We calculate the median of the ideal probabilities and define heavy outputs as those exceeding this median.

- *Utilization of Dynex Neuromorphic Quantum Computing Platform*:
  - We use dynex_circuit that is compatible with pennylane and qasm circuits at scale.

- *Handling Large-Scale Quantum Simulations*:
  - The code is adjusted to handle a high number of qubits (n_wires = 119), demonstrating the scalability of the Dynex platform.
  - Parameters like num_reads and integration_steps are set to ensure accurate simulation of the memristor dynamics.

- *Sample Processing and Error Handling*:
  - Implemented error handling and data processing steps to correctly handle outputs from dynex_circuit.execute.
  - This includes handling various data formats and ensuring correct conversion of samples to integer indices.

- *Compliance with Standard QV Testing Methodology*:
  - The code aligns with the standard Quantum Volume testing methodology, incorporating feedback from experts to ensure accuracy.
  - Adjustments were made to address previous shortcomings, such as using proper unitaries and correct heavy output calculations.

- *Number of Shots*:
  - Using a small number of shots (shots = 50) for experimental runs, balancing resource availability with the need for statistical data.
  - The num_reads parameter is set higher (num_reads = 10000) to compensate and gather sufficient statistics.

Reference: 
- https://en.wikipedia.org/wiki/Quantum_volume

## Circuit Depth and Width
Quantum Volume measures the largest random circuit that a quantum computer can implement successfully. For a circuit, the width is the number of qubits 𝑛, and the depth is the number of layers of gates applied, which is also set to 𝑛 to ensure a balanced evaluation of both qubit number and coherence.

A QV circuit has depth 𝑑=𝑛, meaning it applies 𝑛 layers of randomly chosen two-qubit gates between pairs of qubits. The more qubits and gates in the system, the more difficult it becomes to maintain coherence and achieve high accuracy, which is why this balanced approach gives insight into the overall performance of the quantum system.

## Random Unitary Operations
At each depth, random unitary matrices are applied. These unitary matrices are selected from the Haar measure, which guarantees that the operations are uniformly distributed across the possible quantum operations. Each layer of the circuit consists of random unitary gates applied to randomly chosen pairs of qubits.

## Heavy Output Probability
After running the circuit, the system's output probabilities are measured. The next key element is computing the heavy output probability (HOP), which compares the probability distribution from the quantum computer with the ideal distribution (theoretical one derived from the unitary matrices).

The heavy outputs are the results whose probabilities are in the top 50% of all measured outcomes. The heavy output probability 𝑃heavy is defined as the fraction of times that the heavy outputs are observed in the experimental results. Mathematically, if the experimental probability distribution is 𝑝(𝑥) for the output 𝑥, the heavy output set 𝐻 contains the outputs with the largest 𝑝(𝑥) values such that 𝐻 includes at least 50% of the total probability mass.

## Success Criterion
For a quantum computer to be considered successful at implementing the circuit, 𝑃heavy must be greater than 2/3. This threshold indicates that the quantum system is performing significantly better than random guessing.

This metric captures both the size of the quantum system and its ability to maintain coherence and perform useful quantum computations, making it a comprehensive measure of quantum computing performance.
