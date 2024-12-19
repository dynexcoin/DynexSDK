# Dynex vs. IBM Benchmark

Dynex has conducted comprehensive head-to-head comparisons by running the same quantum gate circuits on multiple platforms (Dynex, IBM AER and IBM QPU), with increasing circuit complexity. As the complexity of the quantum gate circuits increases, Dynex remains the only platform able to compute these circuits efficiently, while the other tested solutions struggle to handle the same level of complexity. This reinforces Dynex's position as the leading quantum-computing-as-a-service (QaaS) provider for scalable, large-scale quantum circuit computations.

## Benchmark Results

| Complexity    |     Dynex[1]  |    IBM AER[2] |    IBM QPU[3] |      
| ------------- | ------------- | ------------- | ------------- |
| 2^8           | CORRECT       | CORRECT       | FAILED        |
| 2^16          | CORRECT       | CORRECT       | FAILED        |
| 2^32          | CORRECT       | FAILED        | FAILED        |
| 2^64          | CORRECT       | FAILED        | FAILED        |
| 2^72          | CORRECT       | FAILED        | FAILED        |
| 2^80          | CORRECT       | FAILED        | FAILED        |
| 2^88          | CORRECT       | FAILED        | FAILED        |
| 2^96          | CORRECT       | FAILED        | FAILED        |
| 2^104         | CORRECT       | FAILED        | FAILED        |

[1] Performed on Dynex Quantum-as-a-Service (QaaS) on 18-12-2024
[2] Performed on IBM AerSimulator
[3] Performed on IBM Quantum Platform, 127 Qubit, 30K CLOPS, Eagle r3 QPU

In the performed benchmarks, Dynex has successfully computed quantum gate circuits with a complexity of up to 2^104, consistently returning the correct results for the quantum circuit operations. In comparison, IBM Aer can handle computations with a complexity up to 2^16, producing correct results. However, the IBM QPU (127-qubit Eagle r3) fails to compute accurate results beyond 2^8 and is unable to return correct sample results at higher complexities.

## Methodology

A n-bit adder quantum gate circuit has been used to perform the benchmark tests, with an increasing level of complexity:


