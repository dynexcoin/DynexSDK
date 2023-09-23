## Dynex Qiskit class

Thanks to groundbreaking research from [Richard H. Warren](https://arxiv.org/pdf/1405.2354.pdf), it is possible to directly translate Qiskit quantum circuits into Dynex Neuromorphic chips. The concept behind is a direct translation of Qiskit objects, but instead of running on IBM Q, the circuits are executed on the Dynex Neuromorphic platform. Here is an example of a one-qubit adder circuit using this approach:

```
from dynex.qiskit import QuantumRegister, ClassicalRegister
from dynex.qiskit import QuantumCircuit, execute

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

