"""
Dynex SDK (beta) Neuromorphic Computing Library
Copyright (c) 2021-2024, Dynex Developers

All rights reserved.

1. Redistributions of source code must retain the above copyright notice, this list of
    conditions and the following disclaimer.
 
2. Redistributions in binary form must reproduce the above copyright notice, this list
   of conditions and the following disclaimer in the documentation and/or other
   materials provided with the distribution.
 
3. Neither the name of the copyright holder nor the names of its contributors may be
   used to endorse or promote products derived from this software without specific
   prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import dynex
from pennylane import numpy as np
import pennylane as qml
import secrets
from collections import Counter
import re
import json
import zlib
import base64
import inspect

################################################################################################################################
# Dynex Circuit Functions (private)
################################################################################################################################

################################################################################################################################
# Pennylane to Dynex .qasm file format
################################################################################################################################
def _pennylane_to_file(circuit, params, wires):
    """
    `Internal Function`

    Serialisation of a PennyLane circuit
    """
    
    with qml.tape.QuantumTape() as tape:
        circuit(params)
    ops = tape.operations
    isQPE = any(op.name.startswith('QuantumPhaseEstimation') for op in ops)
    isGrover = any(op.name.startswith('GroverOperator') for op in ops)
    isCQU = any(op.name.startswith('ControlledQubitUnitary') for op in ops)
    def ProcessOps(op):
        opDict = {
            "name": op.name,
            "wires": [int(w) for w in op.wires],  # ensure wires are integers
            "params": [p.tolist() if hasattr(p, 'tolist') else p for p in op.parameters],
            "hyperparams": {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in op.hyperparameters.items() if k != 'wires'}, # For B.E gate
            "adjointD": 0, # supporting nested daggers
            "ctrlD": 0 # supporting nested controlled gates
        }
        name = op.name
        if name.startswith('Snapshot'):
            pass;
        while name.startswith(('Adjoint(', 'C(')):
            if name.startswith('Adjoint('):
                opDict["adjointD"] += 1
                name = name[8:-1]  # remove 'Adjoint(' and ')'
            elif name.startswith('C('):
                opDict["ctrlD"] += 1
                name = name[2:-1]  # remove 'C(' and ')'
        opDict["base_name"] = name
        if opDict["ctrlD"] > 0 or name == 'ControlledQubitUnitary': # handling CQU
            opDict["control_wires"] = [int(w) for w in op.control_wires]
            opDict["target_wires"] = [int(w) for w in op.wires[len(op.control_wires):]]
        if name == 'QuantumPhaseEstimation': # handling QPE
            opDict["estimation_wires"] = [int(w) for w in op.hyperparameters['estimation_wires']]
            U = op.hyperparameters['unitary']
            if isinstance(U, qml.operation.Operation):
                opDict["unitary"] = {
                    "name": U.name,
                    "wires": [int(w) for w in U.wires],
                    "params": [p.tolist() if hasattr(p, 'tolist') else p for p in U.parameters],
                }
            else:
                opDict["unitary"] = U.tolist() if hasattr(U, 'tolist') else U
                opDict["target_wires"] = [int(w) for w in op.target_wires]
        return opDict
    cirINFO = [ProcessOps(op) for op in ops]
    cirI = {
        "operations": cirINFO,
        "nWires": wires,
        "nParams": len(params),
        "params": params
    }
    JSON = json.dumps(cirI, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))
    comp = zlib.compress(JSON.encode('utf-8'))
    dynexCircuit = base64.b85encode(comp).decode('utf-8')
    return dynexCircuit, isQPE, isGrover, isCQU
################################################################################################################################
# save qasm file
################################################################################################################################
def _save_qasm_file(dnx_circuit):
    """
    `Internal Function`

    Saves the circuit as a .qasm file locally in /tmp as defined in dynex.ini 
    """

    filename = dnx_circuit.qasm_filepath+dnx_circuit.qasm_filename;

    with open(filename, 'w', encoding="utf-8") as f:
        f.write(dnx_circuit.circuit_str); 
################################################################################################################################
# Qasm to Pennylane converter
################################################################################################################################
def _qasm_to_circuit(t, params, wires, debugging=False):
    """
    `Internal Function`

    Reads raw qasm text and converts to PennyLane Circuit class object
    """

    # construct circuit:
    _wires = []
    for i in range(0,wires):
        _wires.append(i);
    
    qasm_circuit = qml.from_qasm(t, measurements=[]) # Create from qasm string

    # define bridge circuit:
    def pl_circuit(y):
        # Add qasm circuit 
        qasm_circuit(wires=_wires);
        return qml.state() 

    return pl_circuit;

################################################################################################################################
# Qiskit to Pennylane converter
################################################################################################################################
def _qiskit_to_circuit(qc, params, wires, debugging=False):
    """
    `Internal Function`

    Reads Qiskit circuit and converts to PennyLane Circuit class object
    """
    
    # construct circuit:
    _wires = []
    for i in range(0,wires):
        _wires.append(i);
    
    my_qfunc = qml.from_qiskit(qc)

    # define bridge circuit:
    def pl_circuit(params):
        my_qfunc(wires=_wires)
        return qml.state()

    return pl_circuit;

################################################################################################################################
# Pennylane Circuit inspection (Private) : More efficient approach when pennylane circuit doesn't have QNode or device
################################################################################################################################
def isPennyLaneCirc(circuit):
    """
    `Internal Function`

    Inspects circuit and returns True if it is a PennyLane circuit
    """
    
    if isinstance(circuit, qml.QNode):
        return True
    if hasattr(circuit, 'quantum_instance') and isinstance(circuit.quantum_instance, qml.QNode):
        return True
    if inspect.isfunction(circuit):
        source = inspect.getsource(circuit)
        pOps = [
            'qml.Hadamard', 'qml.CNOT', 'qml.RX', 'qml.RY', 'qml.RZ',
            'qml.BasisEmbedding', 'qml.QFT', 'qml.adjoint', 'qml.state',
            'qml.sample', 'qml.PauliX', 'qml.PauliY', 'qml.PauliZ',
            'qml.S', 'qml.T', 'qml.CZ', 'qml.SWAP', 'qml.CSWAP',
            'qml.Toffoli', 'qml.PhaseShift', 'qml.ControlledPhaseShift',
            'qml.CRX', 'qml.CRY', 'qml.CRZ', 'qml.Rot', 'qml.MultiRZ',
            'qml.QubitUnitary', 'qml.ControlledQubitUnitary', 'qml.IsingXX',
            'qml.IsingYY', 'qml.IsingZZ', 'qml.Identity', 'qml.Kerr',
            'qml.CrossKerr', 'qml.Squeezing', 'qml.DisplacedSqueezed',
            'qml.TwoModeSqueezing', 'qml.ControlledAddition', 'qml.ControlledSubtraction'
        ]
        if 'qml.' in source and any(op in source for op in pOps):
            return True
        if 'wires=' in source:
            return True
    if hasattr(circuit, 'interface') or hasattr(circuit, 'device'):
        return True
    if hasattr(circuit, 'func') and hasattr(circuit, 'device'):
        return True
    return False

################################################################################################################################
# Dynex Circuit Class (private)
################################################################################################################################
class _dnx_circuit:
    """
    `Internal Class` to hold information about the Dynex circuit
    """

    def __init__(self):
        self.qasm_circuit = None;
        self.circuit_str = None;
        self.qasm_filepath = 'tmp/';
        self.qasm_filename = secrets.token_hex(16)+".qasm.dnx";
        self.params = None;
        self.wires = None;
        self.type = 'qasm';
        self.typestr = 'QASM';
        self.bqm = None;
        self.clauses = [];
        self.wcnf_offset = 0;
        self.precision = 1;

################################################################################################################################
# Measurement parsing functions
################################################################################################################################
def getSamples(sampleset, wires, isQPE, isGrover, isCQU):
    samples = []
    for solution, occurrence in zip(sampleset, sampleset.record.num_occurrences):
        sample = SOL2STATE(solution, wires, isQPE, isGrover, isCQU)
        samples.extend([sample] * occurrence)
    return samples

def getProbs(sampleset, wires, isQPE, isGrover, isCQU): 
    state_counts = Counter()
    total_samples = sum(sampleset.record.num_occurrences)
    for solution, occurrence in zip(sampleset, sampleset.record.num_occurrences):
        state = SOL2STATE(solution, wires, isQPE, isGrover, isCQU)
        state_counts[tuple(state)] += occurrence
    qubit_probs = np.zeros(wires)
    for state, count in state_counts.items():
        for i, bit in enumerate(state):
            if bit == 1:
                qubit_probs[i] += count / total_samples
    return qubit_probs[::-1]

def SOL2STATE(sample, wires, isQPE, isGrover, isCQU):
    state = [0] * wires
    for wire in range(wires):
        rKEY = f'q_{wire}_real'
        iKEY = f'q_{wire}_imag'
        qpeKEY = f'q_{wire}_ctrl_qpe_imag'
        if isQPE and qpeKEY in sample:
            state[wire] = 1 if sample[qpeKEY] > sample[rKEY] else 0
        elif rKEY in sample and iKEY in sample:
            if isGrover or isCQU:
                state[wire] = 1 if sample[iKEY] > 0.5 else 0
            else:
                state[wire] = 1 if sample[rKEY] > 0.5 else 0
        else:
            print(f"Warning: No final state found for wire {wire}")
    return state

################################################################################################################################
# Dynex Circuit Functions (public)
################################################################################################################################

def execute(circuit, params, wires, mainnet=False, num_reads=1000, integration_steps=256, description='Dynex SDK Job', 
                    method='measure', logging=True, debugging=False, bnb=True, switchfraction = 0.0,
                    alpha=20, beta=20, gamma=1, delta=1, epsilon=1, zeta=1, minimum_stepsize = 0.00000006, 
                    block_fee=0, is_cluster=False, cluster_type=1, shots=1):
    """
    Function to execute quantum gate based circuits natively on the Dynex Neuromorphic Computing Platform.

    :Parameters:
    - :circuit: A circuit in one of the following formats: [openQASM, PennyLane, Qiskit, Cirq] (circuit class)
    - :params: Parameters for circuit execution (`list`)
    - :wires: number of qubits (`int`)
    - :method: Type of circuit measurement: 
        'measure': samples of a single measurement
        'probs': computational basis state probabilities
        'all': all solutions as arrays
        'sampleset': dimod sampleset
    - :shots: Sets the minimum number of solutions to retrieve from the network. Works both on mainnet=False and mainnet=True (Default: 1). Typically used for situations where not only the best global optimum (sampleset.first) is required, but multiple optima from different workers (`int`).
    - :description: Defines the description for the job, which is shown in Dynex job dashboards as well as in the network explorer (`string`)

    :Sampling Parameters:

    - :num_reads: Defines the number of parallel samples to be performed (`int` value in the range of [32, MAX_NUM_READS] as defined in your license)

    - :annealing_time: Defines the number of integration steps for the sampling. Models are being converted into neuromorphic circuits, which are then simulated with ODE integration by the participating workers (`int` value in the range of [1, MAX_ANNEALING_TIME] as defined in your license)
        
    - :switchfraction: Defines the percentage of variables which are replaced by random values during warm start samplings (`double` in the range of [0.0, 1.0])

    - :alpha: The ODE integration of the QUBU/Ising or SAT model based neuromorphic circuits is using automatic tuning of these parameters for the ODE integration. Setting values defines the upper bound for the automated parameter tuning (`double` value in the range of [0.00000001, 100.0] for alpha and beta, and [0.0 and 1.0] for gamma, delta and epsilon)

    - :beta: The ODE integration of the QUBU/Ising or SAT model based neuromorphic circuits is using automatic tuning of these parameters for the ODE integration. Setting values defines the upper bound for the automated parameter tuning (`double` value in the range of [0.00000001, 100.0] for alpha and beta, and [0.0 and 1.0] for gamma, delta and epsilon)

    - :gamma: The ODE integration of the QUBU/Ising or SAT model based neuromorphic circuits is using automatic tuning of these parameters for the ODE integration. Setting values defines the upper bound for the automated parameter tuning (`double` value in the range of [0.00000001, 100.0] for alpha and beta, and [0.0 and 1.0] for gamma, delta and epsilon)

    - :delta: The ODE integration of the QUBU/Ising or SAT model based neuromorphic circuits is using automatic tuning of these parameters for the ODE integration. Setting values defines the upper bound for the automated parameter tuning (`double` value in the range of [0.00000001, 100.0] for alpha and beta, and [0.0 and 1.0] for gamma, delta and epsilon)

    - :epsilon: The ODE integration of the QUBU/Ising or SAT model based neuromorphic circuits is using automatic tuning of these parameters for the ODE integration. Setting values defines the upper bound for the automated parameter tuning (`double` value in the range of [0.00000001, 100.0] for alpha and beta, and [0.0 and 1.0] for gamma, delta and epsilon)

    - :zeta: The ODE integration of the QUBU/Ising or SAT model based neuromorphic circuits is using automatic tuning of these parameters for the ODE integration. Setting values defines the upper bound for the automated parameter tuning (`double` value in the range of [0.00000001, 100.0] for alpha and beta, and [0.0 and 1.0] for gamma, delta and epsilon)

    - :minimum_stepsize: The ODE integration of the QUBU/Ising or SAT model based neuromorphic circuits is performig adaptive stepsizes for each ODE integration forward Euler step. This value defines the smallest step size for a single adaptive integration step (`double` value in the range of [0.0000000000000001, 1.0])

    - :debugging: Only applicable for test-net sampling. Defines if the sampling process should be quiet with no terminal output (FALSE) or if process updates are to be shown (TRUE) (`bool`)

    - :bnb: Use alternative branch-and-bound sampling when in mainnet=False (`bool`)

    - :block_fee: Computing jobs on the Dynex platform are being prioritised by the block fee which is being offered for computation. If this parameter is not specified, the current average block fee on the platform is being charged. To set an individual block fee for the sampling, specify this parameter, which is the amount of DNX in nanoDNX (1 DNX = 1,000,000,000 nanoDNX) 

    :Returns:

    - Returns the measurement based on the parameter 'measure'
    
    :Example:

    .. code-block:: Python

        import dynex
        import dynex_circuit
        from pennylane import numpy as np
        import pennylane as qml
        
        params = [0.1, 0.2]
        wires = 2
        
        def circuit(params):
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.Hadamard(wires=0)
            return qml.sample()
        
        # Compute circuit on Dynex:
        measure = dynex_circuit.execute(circuit, params, mainnet=True)
        print(measure)

        >>>
        │   DYNEXJOB │   QUBITS │   QUANTUM GATES │   BLOCK FEE │   ELAPSED │   WORKERS READ │   CIRCUITS │   STEPS │   GROUND STATE │
        ├────────────┼──────────┼─────────────────┼─────────────┼───────────┼────────────────┼────────────┼─────────┼────────────────┤
        │      28391 │       21 │              64 │        0.00 │      0.58 │              1 │       1000 │     256 │       38708.00 │
        ╰────────────┴──────────┴─────────────────┴─────────────┴───────────┴────────────────┴────────────┴─────────┴────────────────╯
        ╭────────────┬─────────────────┬────────────┬───────┬──────────┬──────────────┬─────────────────────────────┬───────────┬──────────╮
        │     WORKER │         VERSION │   CIRCUITS │   LOC │   ENERGY │      RUNTIME │                 LAST UPDATE │     STEPS │   STATUS │
        ├────────────┼─────────────────┼────────────┼───────┼──────────┼──────────────┼─────────────────────────────┼───────────┼──────────┤
        │ 1147..9be1 │ 2.3.5.OZM.134.L │       1000 │     0 │     0.00 │ 4.190416548s │ 2024-08-06T19:37:36.148518Z │ 0 (0.00%) │  STOPPED │
        ├────────────┼─────────────────┼────────────┼───────┼──────────┼──────────────┼─────────────────────────────┼───────────┼──────────┤
        │ 6a66..2857 │ 2.3.5.OZM.134.L │       1000 │     0 │     0.00 │ 9.002006172s │  2024-08-06T19:37:31.33693Z │ 0 (0.00%) │  STOPPED │
        ╰────────────┴─────────────────┴────────────┴───────┴──────────┴──────────────┴─────────────────────────────┴───────────┴──────────╯
        [DYNEX] FINISHED READ AFTER 57.94 SECONDS
        [DYNEX] SAMPLESET READY
        [1 0]

    ...
    """

    # enforce param wires to int value:
    if type(wires) == list:
        wires = len(wires);
    
    circuit_str = None;
    isQPE = False
    isGrover = False
    isCQU = False
    
    # pennylane circuit? convert to qasm
    if isPennyLaneCirc(circuit):
        circuit_str, isQPE, isGrover, isCQU  = _pennylane_to_file(circuit, params, wires);
        if logging:
            print('[DYNEX] Executing PennyLane quantum circuit');

    # qasm circuit? convert to pennylane->to_file
    qasmChecker = type(circuit) == str
    if qasmChecker:
        circuit = _qasm_to_circuit(circuit, params, wires, debugging=False);
        circuit_str, isQPE, isGrover, isCQU = _pennylane_to_file(circuit, params, wires);
        if logging:
            print('[DYNEX] Executing OpenQASM quantum circuit');
    
    # cirq circuit? convert to pennylane->to_file
    # TBD
    
    # qiskit circuit? convert to pennylane->to_file
    qiskitChecker = str(type(circuit)).find('qiskit') > 0
    if qiskitChecker:
        circuit = _qiskit_to_circuit(circuit, params, wires, debugging=False);
        circuit_str, isQPE, isGrover, isCQU = _pennylane_to_file(circuit, params, wires);
        if logging:
            print('[DYNEX] Executing Qiskit quantum circuit');
    
    # At this point we can assume its pennylane->to_file format circuit. We generate a dynex circuit model
    circ_model = _dnx_circuit();
    circ_model.circuit_str = circuit_str;
    circ_model.params = params;
    circ_model.wires = wires;

    # write circuit to .qasm file:
    _save_qasm_file(circ_model);

    if debugging:
        print('[DYNEX] ---------------- / Circuit / --------------');
        print(circ_model.circuit_str);
        print('-------------------------------------------');

    # run qasm circuit:
    sampler = dynex._DynexSampler(circ_model, mainnet=mainnet, description=description, bnb=bnb, logging=logging, filename_override=circ_model.qasm_filename);
    
    sampleset = sampler.sample(num_reads=num_reads, annealing_time=integration_steps, switchfraction=switchfraction, alpha=alpha, beta=beta, gamma=gamma,
                               delta=delta, epsilon=epsilon, zeta=zeta, minimum_stepsize=minimum_stepsize, debugging=debugging, 
                               block_fee=block_fee, is_cluster=is_cluster, cluster_type=cluster_type, shots=shots);
   
    # decode solution:
    if method not in ['measure', 'probs','all','sampleset']:
            raise ValueError("Method must be either 'measure', 'probs', 'all' or 'sampleset'");
        
    if debugging:
        print('[DYNEX] -------------- / '+method+' / ------------');

    if method == 'measure':
        samples = getSamples(sampleset, wires, isQPE, isGrover, isCQU)
        if isQPE:
            result = np.array(samples[0])
        else:
            result = np.array(samples[0])[::-1] 
    elif method == 'sampleset':
        result = sampleset;
    elif method == 'all':
        result = [np.array(sample[::-1]) for sample in getSamples(sampleset, wires, isQPE, isGrover, isCQU)]
    else:  # probs
        probs = getProbs(sampleset, wires, isQPE, isGrover, isCQU)
        result = probs
    return result;
        



