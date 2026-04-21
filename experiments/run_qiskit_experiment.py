import random, time
import pandas as pd

from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.transpiler.passes.synthesis.hls_plugins import PauliEvolutionSynthesisRustiq
from qiskit_ibm_runtime.fake_provider import FakeGuadalupeV2, FakeBrisbane
from qiskit.transpiler import generate_preset_pass_manager
from qiskit.synthesis import LieTrotter
from qiskit.circuit import QuantumCircuit

from pytket.circuit import TermSequenceBox
from pytket.transform import PauliSynthStrat
from pytket.architecture import Architecture
from pytket.passes import CXMappingPass, FullPeepholeOptimise
from pytket.placement import GraphPlacement

from utils import create_random_pauli_polynomial, pp_to_list_qiskit, permute_with_mapping
from utils import two_qubit_gates_qiskit, two_qubit_gates_tket, pp_to_pytket_box, two_qubit_gates_pauliopt, topo_from_ibm_backend
from pauliopt.pauli.simplification.simple_simplify import simplify_pauli_polynomial
from pauliopt.pauli.synthesis.pathfinder_in_pauli_grove import pathfinder_in_pauli_grove
from pauliopt.pauli.synthesis.tree_mapping import pauli_tree_mapping
from pauliopt.pauli.synthesis.steiner_gray_synthesis import pauli_polynomial_steiner_gray_clifford


def qiskit_default_test(pp, qiskit_backend):
    pass_manager = generate_preset_pass_manager(
        optimization_level=3,
        backend=qiskit_backend,
        layout_method="sabre",
        routing_method="sabre",
    )
    pauli_op = SparsePauliOp.from_list(pp_to_list_qiskit(pp))
    evolution_gate = PauliEvolutionGate(pauli_op)

    start = time.time()
    circuit = QuantumCircuit(pp.num_qubits)
    circuit.append(evolution_gate, range(pp.num_qubits))
    transpiled_qc = pass_manager.run(circuit)
    elapsed_time = time.time() - start
    results1 = two_qubit_gates_qiskit(circuit)
    results2 = two_qubit_gates_qiskit(transpiled_qc)
    resp = {'method':'Qiskit','synthesis': results1, 'routed': results2, 'time': round(elapsed_time*1000)}
    return resp

def qiskit_rustiq_test(pp, qiskit_backend):
    pass_manager = generate_preset_pass_manager(
        optimization_level=3,
        backend=qiskit_backend,
        layout_method="sabre",
        routing_method="sabre",
    )
    pauli_op = SparsePauliOp.from_list(pp_to_list_qiskit(pp))
    evolution_gate = PauliEvolutionGate(pauli_op)
    start = time.time()
    circuit = PauliEvolutionSynthesisRustiq().run(evolution_gate, preserve_order=False, optimize_count=True, resynth_clifford_method=2)
    transpiled_qc = pass_manager.run(circuit)
    elapsed_time = time.time() - start
    results1 = two_qubit_gates_qiskit(circuit)
    results2 = two_qubit_gates_qiskit(transpiled_qc)
    resp = {'method':'Qiskit-Rustiq','synthesis': results1, 'routed': results2, 'time': round(elapsed_time*1000)}
    return resp

def tket_test(pp, qiskit_backend):
    coupling_map = qiskit_backend.coupling_map.get_edges()
    architecture = Architecture(coupling_map)
    placement = GraphPlacement(architecture)

    start = time.time()
    tsbox = TermSequenceBox(pp_to_pytket_box(pp), synthesis_strategy=PauliSynthStrat.Greedy)
    pauli_circ = tsbox.get_circuit()
    results1 = two_qubit_gates_tket(pauli_circ)
    CXMappingPass(architecture, placement).apply(pauli_circ)
    FullPeepholeOptimise(allow_swaps=False).apply(pauli_circ)
    elapsed_time = time.time() - start
    results2 = two_qubit_gates_tket(pauli_circ)
    resp = {'method':'TKET','synthesis': results1, 'routed': results2, 'time': round(elapsed_time*1000)}
    return resp

def ppg_test(pp, qiskit_backend):
    topo = topo_from_ibm_backend(qiskit_backend)

    start = time.time()
    mapping, tree = pauli_tree_mapping(pp, topo)
    pp_m = permute_with_mapping(mapping, pp, topo.num_qubits)
    circ_out, gadget_perm, perm, benchmarks = pathfinder_in_pauli_grove(pp_m, topo, tree)
    elapsed_time = time.time() - start
    results = two_qubit_gates_pauliopt(circ_out)
    resp = {'method':'PPG', 'routed': results, 'time': round(elapsed_time*1000), 'benchmarks': benchmarks}
    return resp

def sgc_test(pp, qiskit_backend):
    topo = topo_from_ibm_backend(qiskit_backend)

    start = time.time()
    mapping, tree = pauli_tree_mapping(pp, topo)
    pp_m = permute_with_mapping(mapping, pp, topo.num_qubits)
    circ_out, gadget_perm, perm, benchmarks = pauli_polynomial_steiner_gray_clifford(pp_m, topo, tree)
    elapsed_time = time.time() - start
    results = two_qubit_gates_pauliopt(circ_out)
    resp = {'method':'LO', 'routed': results, 'time': round(elapsed_time*1000), 'benchmarks': benchmarks}
    return resp

def experiment(num_qubits, gadgets, qiskit_backend, rounds):
    df = pd.DataFrame(columns=['n_rep','num_qubits','num_gadgets','method','count','depth','time'])
    for num_gadgets in gadgets:
        for i in range(rounds):
            print('num gadgets:', num_gadgets, 'round:', i)
            pp = create_random_pauli_polynomial(num_qubits, num_gadgets)
            pp = simplify_pauli_polynomial(pp, allow_acs=True)
            results = qiskit_default_test(pp, qiskit_backend)
            df.loc[len(df)] = {'n_rep': i, 'num_qubits': num_qubits, 'num_gadgets': num_gadgets, 'method': results['method'],
                            'count': results['routed']['count'], 'depth': results['routed']['depth'], 
                            'time': results['time']}
            results = qiskit_rustiq_test(pp, qiskit_backend)
            df.loc[len(df)] = {'n_rep': i, 'num_qubits': num_qubits, 'num_gadgets': num_gadgets, 'method': results['method'],
                            'count': results['routed']['count'], 'depth': results['routed']['depth'], 
                            'time': results['time']}

            results = tket_test(pp, qiskit_backend)
            df.loc[len(df)] = {'n_rep': i, 'num_qubits': num_qubits, 'num_gadgets': num_gadgets, 'method': results['method'],
                            'count': results['routed']['count'], 'depth': results['routed']['depth'], 
                            'time': results['time']}

            results = ppg_test(pp, qiskit_backend)
            df.loc[len(df)] = {'n_rep': i, 'num_qubits': num_qubits, 'num_gadgets': num_gadgets, 'method': results['method'],
                            'count': results['routed']['count'], 'depth': results['routed']['depth'], 
                            'time': results['time']}

            results = sgc_test(pp, qiskit_backend)
            df.loc[len(df)] = {'n_rep': i, 'num_qubits': num_qubits, 'num_gadgets': num_gadgets, 'method': results['method'],
                            'count': results['routed']['count'], 'depth': results['routed']['depth'], 
                            'time': results['time']}
    df.to_csv('results/qiskit_test_'+str(num_qubits)+'.csv')
    df = df.drop(['n_rep', 'num_qubits'], axis=1).groupby(['method','num_gadgets']).mean().round(1)
    df.to_csv('results/qiskit_test_aggr'+str(num_qubits)+'.csv')
    return df

#qiskit_backend = FakeGuadalupeV2()
qiskit_backend = FakeBrisbane()
seed = 42
# max 80/150, 30/500 
logical_qubits = 16
# gadgets = [10,20,30,40,50,60,70,80,90,100]
gadgets = [10,20]
rounds = 20

print('\n--------------experiment start----------------')
print('Backend:', qiskit_backend.name, 'with', qiskit_backend.configuration().n_qubits, 'qubits')
print('Logical qubits:', logical_qubits)
print('Gadgets:', gadgets)
print('Rounds per setting:', rounds)
random.seed(seed)
results = experiment(logical_qubits, gadgets, qiskit_backend, rounds)
print(results)
print('--------------experiment end----------------\n')