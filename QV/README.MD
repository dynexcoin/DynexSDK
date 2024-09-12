## Quantum Volume Measurement on Dynex
Quantum Volume (QV) is a metric designed to quantify the performance of a quantum computer. It considers several factors including the number of qubits, gate fidelity, connectivity, measurement errors, and the ability to execute quantum circuits successfully. The QV metric attempts to capture both the quantity (number of qubits) and quality (fidelity and error rates) of a quantum processor.

- [Jupyter Notebook: QV Measurement on Dynex](https://github.com/dynexcoin/DynexSDK/blob/main/QV/QuantumVolume.ipynb)
- [Youtube: Real-time screen recording of the measurement](https://youtu.be/md6xPnQJcF0)

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
