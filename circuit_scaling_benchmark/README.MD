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

![](https://github.com/dynexcoin/DynexSDK/blob/main/circuit_scaling_benchmark/circuits.png)

We applied the following n-bit adder circuit definition:

```
params = [58, 142] # two numbers to add

def Nqubits(a, b):
    mxVal = a + b
    return mxVal.bit_length()

wires = Nqubits(*params)

def Kfourier(k, wires):
    for j in range(len(wires)):
        qml.RZ(k * np.pi / (2**j), wires=wires[j])

def FullAdder(params, state=True):
    a, b = params
    wires = Nqubits(a, b)
    qml.BasisEmbedding(a, wires=range(wires))
    qml.QFT(wires=range(wires))
    Kfourier(b, range(wires))
    qml.adjoint(qml.QFT)(wires=range(wires))
    if state:
        return qml.state()
    else:
        return qml.sample()
```

## IBM QPU Results Remarks

The following IBM dashboard screen displays the results of executing the benchmark's 8-bit adder quantum gate circuit on IBM's Eagle r3 QPU. The probabilistically determined best result was "10010011" (147), observed with a frequency of 12, while the expected, correct outcome, "11001000" (58 + 124 = 200), was reported with a frequency of only 5. This discrepancy can be attributed to error correction issues inherent in hardware quantum-mechanical based systems, which is a major disadvantage compared to Dynex. In contrast, Dynex experiences significantly fewer error correction challenges, ensuring more reliable and accurate results.

![](https://github.com/dynexcoin/DynexSDK/blob/main/circuit_scaling_benchmark/ibm_qpu_8bit.png)

