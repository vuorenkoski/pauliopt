import random, time, itertools, sys
import pandas as pd

from pauliopt.pauli.synthesis.steiner_gray_synthesis import pauli_polynomial_steiner_gray_clifford
from pauliopt.pauli.synthesis.dynamic_ordering_synthesis import pauli_polynomial_dynamic_ordering
from pauliopt.pauli.synthesis.pp_mapping import I_index_mapping
from tests.pauli.utils import verify_equality
from pauliopt.pauli.simplification.simple_simplify import simplify_pauli_polynomial

from pauliopt.pauli.pauli_polynomial import PauliPolynomial
from experiments.utils import permute_with_mapping, random_mapping, create_random_pauli_polynomial
from experiments.utils import cnot_count, print_pp, extend_gadgets, aggregate_data, get_topo, aggregate_data_precx

# todo: 1) molecules, 2) sparse gadgets

def check_circuit_equivalence(pp, circ_out, gadget_perm, perm):
    pp2 = PauliPolynomial(pp.num_qubits)
    pp2.pauli_gadgets = [pp[i].copy() for i in gadget_perm]
    circ = circ_out.to_qiskit()
    pp_circ = pp2.to_qiskit()
    return verify_equality(pp_circ,circ)


def random_pauli_experiment(backend, methods, logical_qubits=None, nr_gadgets=100, nr_steps=5, rounds=20, 
                            verify=True, mapping_method=I_index_mapping, steps=None):
    topo = get_topo(backend['name'], backend['qubits'])
    if logical_qubits is None:
        logical_qubits = backend['qubits']
    if logical_qubits > backend['qubits']:
        raise ValueError('Logical qubits cannot be more than physical qubits')
    print('----------------------------------Experiment')
    print('Backend:', backend['name'])
    print('Num physical qubits:',  backend['qubits'])
    print('Num logical qubits:',  logical_qubits)
    print('Random iterations:', rounds)
    print('Mapping method:', mapping_method.__name__)
    print('Synthesis methods:', [m.__name__ for m in methods])
    print('Verifying:', verify)
    df = pd.DataFrame(columns=['n_rep','num_qubits','n_gadgets','method','mapping','cx','time','pre-cx'])
    if not steps:
        steps = list(range(nr_steps, nr_gadgets, nr_steps))
    for num_gadgets in steps:
        for i in range(rounds):
            pp = create_random_pauli_polynomial(logical_qubits, num_gadgets)
#            pp = create_random_pauli_polynomial(logical_qubits, num_gadgets, min_legs=1, max_legs=8)
            pp = simplify_pauli_polynomial(pp, allow_acs=True)
            mapping_time = time.time()
            map = mapping_method(pp, topo)
            mapping_time = int((time.time() - mapping_time) * 1000)
            pp_m = permute_with_mapping(map, pp, topo.num_qubits)
            pp_r = permute_with_mapping(random_mapping(topo), pp, topo.num_qubits)

            for synth_method in methods:
                start = time.time()
                circ_out, gadget_perm, perm, benchmarks = synth_method(pp_m.copy(), topo)
                if verify:
                    correct = check_circuit_equivalence(pp_m.copy(), circ_out, gadget_perm, perm)
                    if not correct:
                        print('Circuit equivalence failed for', synth_method.__name__, 'with algorithm mapping', mapping_method[1])
                        input()
                column = {
                            'n_rep': i,
                            'num_qubits': logical_qubits,
                            'n_gadgets': num_gadgets,
                            'method': synth_method.__name__,
                            'mapping': 'algorithm',
                            'cx': cnot_count(circ_out),
                        } | benchmarks | {'time': mapping_time + int((time.time()-start) * 1000)}
                df.loc[len(df)] = column
                start = time.time()
                circ_out, gadget_perm, perm, benchmarks = synth_method(pp_r.copy(), topo)
                if verify:
                    correct = check_circuit_equivalence(pp_r.copy(), circ_out, gadget_perm, perm)
                    if not correct:
                        print('Circuit equivalence failed for', synth_method.__name__, 'with random mapping', mapping_method[1])
                        input()
                column = {
                            'n_rep': i,
                            'num_qubits': logical_qubits,
                            'n_gadgets': num_gadgets,
                            'method': synth_method.__name__,
                            'mapping': 'random',
                            'cx': cnot_count(circ_out),
                        } | benchmarks | {'time': int((time.time()-start) * 1000)}
                df.loc[len(df)] = column
        df2 = aggregate_data(df, methods[0].__name__, methods[1].__name__)
        print(df2.tail(1).to_string(header=False)) 
    df2 = aggregate_data(df, methods[0].__name__, methods[1].__name__)
    df2.to_csv('aggregated.csv')
    print(df2.to_string())
    df2 = aggregate_data_precx(df, methods[0].__name__, methods[1].__name__)
    df2.to_csv('aggregated_precx.csv')
    print('Pre-cx data')
    print(df2.to_string())

def all_mappings(pp, backend, synth_method=pauli_polynomial_steiner_gray_clifford):
    topo = get_topo(backend['name'], backend['qubits'])
    if pp.num_qubits > backend['qubits']:
        raise ValueError('Logical qubits cannot be more than physical qubits')

    print('---------------------------------- Experiment')
    print("all mappings")
    print('Backend:', backend['name'])
    print('Num physical qubits:', backend['qubits'])
    print('Num logical qubits:', pp.num_qubits)
    print('Number of gadgets:', pp.num_gadgets)
    print('Synthesis method:', synth_method.__name__)
    min_cx = -1
    max_cx = -1
    min_cx_mapping = None
    max_cx_mapping = None
    for m in itertools.permutations(list(range(backend['qubits']))):
        sys.stdout.write(str(m[0]))
        sys.stdout.flush()
        pp_m = permute_with_mapping(m, pp, topo.num_qubits)
        circ_out, gadget_perm, perm, benchmarks = synth_method(pp_m.copy(), topo)

        if min_cx == -1 or benchmarks['pre-cx'] < min_cx:
            min_cx = benchmarks['pre-cx']
            min_cx_mapping = m
        if max_cx == -1 or benchmarks['pre-cx'] > max_cx:
            max_cx = benchmarks['pre-cx']
            max_cx_mapping = m
    print('\nMinimum pre-CX count:', min_cx, 'with mapping', min_cx_mapping) 
    print('Maximum pre-CX count:', max_cx, 'with mapping', max_cx_mapping)


if __name__ == "__main__":
    verify = False
    methods = [pauli_polynomial_dynamic_ordering, pauli_polynomial_steiner_gray_clifford]
    mapping_method = I_index_mapping
    random.seed(42)
    steps = list(range(2, 40, 2)) + list(range(40, 220, 20))
#    steps = list(range(20, 220, 20))
    backend = {'name': 'quito', 'qubits': 5}
    backend = {'name': 'guadalupe', 'qubits': 16}
    backend = {'name': 'grid', 'qubits': 9}
#    backend = {'name': 'line', 'qubits': 6}
    logical_qubits = 9
    random_pauli_experiment(backend, methods, logical_qubits=logical_qubits, nr_gadgets=200, nr_steps=20, rounds=200, verify=verify, mapping_method=mapping_method, steps=steps)
    input()

    random.seed(42)
    pp = create_random_pauli_polynomial(3, 80, seed=42, empty_qubits=0)
    backend = {'name': 'line', 'qubits': 6}
    synth_method=pauli_polynomial_steiner_gray_clifford
#    synth_method=pauli_polynomial_dynamic_ordering
#    all_mappings(pp, backend, synth_method=synth_method)
#    input()

    gadgets = 6
    seed = 43
    logical_qubits = 4
    physical_qubits = 9
    topology = 'line'

    print('\n\nstart-------------------------------------------------')
    random.seed(seed)
    pp = create_random_pauli_polynomial(logical_qubits, gadgets, seed=seed, empty_qubits=0)
    pp = simplify_pauli_polynomial(pp, allow_acs=True)
    topo = get_topo(topology, physical_qubits)
    mapping = I_index_mapping(pp, topo)
    mapping = [0,3,6,8]
    pp = permute_with_mapping(mapping, pp, topo.num_qubits)
    if pp.num_qubits != topo.num_qubits:
        pp = extend_gadgets(pp,topo)
    print('Topology:', topo)
    print('used mapping',mapping)
    print_pp(pp)

    print('\noriginal synthesis with mapping starts')
    circ_out, gadget_perm, perm, benchmarks = pauli_polynomial_steiner_gray_clifford(pp.copy(), topo)
    verify = check_circuit_equivalence(pp.copy(), circ_out, gadget_perm, perm)
    print('CNOT count:',cnot_count(circ_out))
    print('Benchmarks:', benchmarks)
    print('Circuit equivalence:', verify)
    print('Gadget perm:',gadget_perm)
    print_pp(pp, order=gadget_perm)

    print('\nnew synthesis with mapping starts')
    algo_order = [i for i in range(pp.num_gadgets)]
    algo_order = [1, 2, 3, 4, 5, 0]
    circ_out, gadget_perm, perm, benchmarks = pauli_polynomial_dynamic_ordering(pp.copy(), topo, debug=True, random_sel=False, print_order=algo_order)
    print_pp(pp, order=gadget_perm)
    print('CNOT count:',cnot_count(circ_out))
    print('Benchmarks:', benchmarks)
    print('Gadget perm:',gadget_perm)
    verify = check_circuit_equivalence(pp.copy(), circ_out, gadget_perm, perm)
    print('Circuit equivalence:', verify)
