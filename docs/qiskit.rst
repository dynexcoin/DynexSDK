Dynex Qiskit class
=====================================
Recent advances in quantum hardware have resulted in the first systems becoming publicly available. On one hand, gate-based quantum computers have been designed, such as the IBM Q, Rigetti’s Aspen, or IonQ’s systems using using superconducting transmons or ion tubes. On the other hand, adibiatic quantum computing and neuromorphic computing has emerged as another possibility to leverage physics inspired computations. It was shown that adiabatic quantum computing can solve the same problems as gate-based (universal) quantum computing given at least two degrees of freedom for 2-local Hamiltonian [3,4,5]. The Dynex Neuromorphic platform supports a 2-local Ising Hamiltonian with a single degree of freedom, which is why it is believed to only solve a subset of the problems that can be expressed by gate-based (universal) quantum machines. In 2014, Warren outlined how a set of universal quantum gates could be realized in adiabatic form using D-Wave’s annealing abstraction [1]. This is demonstrated, among others, for C-NOT, Toffoli (CC-NOT), Swap and C-Swap (Fredkin) gates in a {0, 1} base of qubit states, and for the Hadamard gate in a two-vector 0i,1i base.

- `Dynex Qiskit Package <https://github.com/dynexcoin/Dynex-Qiskit>`_

- `Medium: Computing on the Dynex Neuromorphic Platform: IBM Qiskit 4-Qubit Full Adder Circuit <https://medium.com/@dynexcoin/computing-on-the-dynex-neuromorphic-platform-ibm-qiskit-4-qubit-full-adder-circuit-7416084e19dd>`_ 


Example
=====================================

Thanks to groundbreaking research from Richard H. Warren, it is possible to directly translate Qiskit quantum circuits into Dynex Neuromorphic chips. The concept behind is a direct translation of Qiskit objects, but instead of running on IBM Q, the circuits are executed on the Dynex Neuromorphic platform. Here is an example of a one-qubit adder circuit using this approach:

.. code-block:: Python

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

Technical Background
=====================================
The Dynex Qiskit Package automatically translates quantum programs (Qiskit based) from a subset of common gates to an adibiatic representation, which is used by adibiatic quantum computers and by the Dynex Neuromorphic platform. Adiabatic computing with two degrees of freedom of 2-local Hamiltonians has been theoretically shown to be equivalent to the gate model of universal quantum computing. The Dynex Qiskit Package translates quantum gate circuits comprised of a subset of common gates expressed as an IBM Qiskit program to single-degree 2-local Ising Hamiltonians, which are subsequently embedded on the Dynex Neuromomorphic platform. These gate elements are placed in the chimera graph and augmented by constraints that enforce inter-gate logical relationships, resulting in an annealer embedding that completely characterizes the overall gate circuit.

Given a circuits (in Qiskit format):

1. Construct equivalent QUBO (quadratic unconstrained binary opt.)
    
    1.1 Each gate -> QUBO
    
    1.2 Compose QUBOs into single Hamiltonian

2. Find embedding
    
    2.1 In 16x16 bi-partite graphs (K4,4) sparsely connected as Chimera
    
    2.2 Project per-gate QUBO -> bi-partite unit graph K4,4
    
    2.3 Connect constraints b/w QUBOs of units graphs -> “wires”

3. In Python: Transform IBM Qiskit program
    
    3.1 Replace gates
    
    3.2 Run, but instead of IBM Q exec., sample on the Dynex Neuromorphic sampler

Given 'n' gates and 'q' qubits (IBM), the resulting QUBO/Ising problem translates to ~ 23n qubits (Dynex), but often just to 5-6n due to complex (Toffoli) gates.

An upper bound on the resource requirements on both ends can be given as follows: Given an n-gate quantum circuit (in our case specified as a Qiskit program), a translation to an adiabatic form is provided in no more than 32n adiabatic qubits on its QUBO/Ising representation. The factor is comprised of 8 qubits for the K4,4 representation of a gate, the remaining 24 qubits are used as wiring to the left and below that gate-equivalent K4,4 graph. Notice that an increase by 32X still increases the capabilities by mapping to the Dynex platform dramatically, and problems can often be mapped more efficiently.

Notice that currently we do not provide a translation for the Hadamard gate, H, as it requires a different base of qubit states than the others.

References and Scientific Background
=====================================

[1] Warren, Richard. (2014). Gates for Adiabatic Quantum Computing.

[2] Regan, Malcolm & Eastwood, Brody & Nagabhiru, Mahita & Mueller, Frank. (2019). Automatically Translating Quantum Programs from a Subset of Common Gates to an Adiabatic Representation. 10.1007/978-3-030-21500-2_9.

[3] Aharonov, D., van Dam, W., Kempe, J., Landau, Z., Lloyd, S., Regev, O.: Adiabatic quantum computation is equivalent to standard quantum computation. SIAM J. Comput. 37(1), 166–194 (Apr 2007). https://doi.org/10.1137/S0097539705447323, http://dx.doi.org/10.1137/S0097539705447323

[4] Bacon, D., Flammia, S.T., Crosswhite, G.M.: Adiabatic quantum transistors (2012). https://doi.org/10.1103/PhysRevX.3.021015

[5] Dorit Aharonov, Wim van Dam, J.K.Z.L.S.L.O.R.: Adiabatic quantum computation is equivalent to standard quantum computation. ArXiv e-prints (May 2004), https://arxiv.org/abs/quant-ph/0405098
