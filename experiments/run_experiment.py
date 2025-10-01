from tabnanny import verbose
import random, time, itertools, sys
import pandas as pd

from pauliopt.pauli.synthesis.steiner_gray_synthesis import pauli_polynomial_steiner_gray_clifford
from pauliopt.pauli.synthesis.dynamic_ordering_synthesis import pauli_polynomial_dynamic_ordering
from pauliopt.pauli.synthesis.pp_mapping import I_index_mapping, pauli_tree_mapping
from tests.pauli.utils import verify_equality
from pauliopt.pauli.simplification.simple_simplify import simplify_pauli_polynomial

from pauliopt.pauli.pauli_polynomial import PauliPolynomial, I, Z, X, Y
from experiments.utils import permute_with_mapping, qubit_correlation_sum, random_mapping, I_index, cnot_depth
from experiments.utils import create_random_pauli_polynomial, steiner_tree_analysis, order_gadgets
from experiments.utils import cnot_count, print_pp, aggregate_data, get_topo, aggregate_data_depth, qubit_graph
from experiments.utils import print_brisbane_mapping, print_brisbane_topo

def check_circuit_equivalence(pp, circ_out, gadget_perm, perm):
    pp2 = PauliPolynomial(pp.num_qubits)
    pp2.pauli_gadgets = [pp[i].copy() for i in gadget_perm]
    circ = circ_out.to_qiskit()
    pp_circ = pp2.to_qiskit()
    return verify_equality(pp_circ,circ)

def trivial_pauli_experiment(trivial_mapping, backend, methods, logical_qubits=None, nr_gadgets=100, nr_steps=5, rounds=20, max_legs=None, 
                            verify=True, mapping_method=I_index_mapping, steps=None):
    topo = get_topo(backend['name'], backend['qubits'])
    if logical_qubits is None:
        logical_qubits = backend['qubits']
    if logical_qubits > backend['qubits']:
        raise ValueError('Logical qubits cannot be more than physical qubits')
    print('----------------------------------Experiment')
    print('mapping versus trivial', trivial_mapping)
    print('Backend:', backend['name'])
    print('Num physical qubits:',  backend['qubits'])
    print('Num logical qubits:',  logical_qubits)
    print('Random iterations:', rounds)
    print('Mapping method:', mapping_method.__name__)
    print('Synthesis methods:', [m.__name__ for m in methods])
    print('Verifying:', verify)
#    print('trivial mapping')
#    print_brisbane_mapping(trivial_mapping)
    df = pd.DataFrame(columns=['n_rep','num_qubits','n_gadgets','method','mapping','cx','cx_depth','time','pre-cx'])
    if not steps:
        steps = list(range(nr_steps, nr_gadgets, nr_steps))
    for num_gadgets in steps:
        for i in range(rounds):
            sys.stdout.write(str(i))
            sys.stdout.flush()
            if max_legs is None:
                pp = create_random_pauli_polynomial(logical_qubits, num_gadgets)
            else:
                pp = create_random_pauli_polynomial(logical_qubits, num_gadgets, min_legs=1, max_legs=max_legs)
            pp = simplify_pauli_polynomial(pp, allow_acs=True)
            mapping_time = time.time()
            map = mapping_method(pp, topo)
#            print_brisbane_mapping(map)
            mapping_time = int((time.time() - mapping_time) * 1000)
#            print(mapping_time)
            topo_t = qubit_graph(trivial_mapping, topo)
            topo_m = qubit_graph(map, topo)
#            pp_m = permute_with_mapping(map, pp, topo.num_qubits)
#            pp_r = permute_with_mapping(trivial_mapping, pp, topo.num_qubits)

            for synth_method in methods:
                start = time.time()
                circ_out, gadget_perm, perm, benchmarks = synth_method(pp.copy(), topo_m)
                if verify:
                    correct = check_circuit_equivalence(pp.copy(), circ_out, gadget_perm, perm)
                    if not correct:
                        print('Circuit equivalence failed for', synth_method.__name__, 'with algorithm mapping', mapping_method.__name__)
                        input()
                column = {
                            'n_rep': i,
                            'num_qubits': logical_qubits,
                            'n_gadgets': num_gadgets,
                            'method': synth_method.__name__,
                            'mapping': 'algorithm',
                            'cx': cnot_count(circ_out),
                            'cx_depth': cnot_depth(circ_out),
                        } | benchmarks | {'time': mapping_time + int((time.time()-start) * 1000)}
                df.loc[len(df)] = column
                start = time.time()
                circ_out, gadget_perm, perm, benchmarks = synth_method(pp.copy(), topo_t)
                if verify:
                    correct = check_circuit_equivalence(pp.copy(), circ_out, gadget_perm, perm)
                    if not correct:
                        print('Circuit equivalence failed for', synth_method.__name__, 'with random mapping', mapping_method.__name__)
                        input()
                column = {
                            'n_rep': i,
                            'num_qubits': logical_qubits,
                            'n_gadgets': num_gadgets,
                            'method': synth_method.__name__,
                            'mapping': 'random',
                            'cx': cnot_count(circ_out),
                            'cx_depth': cnot_depth(circ_out),
                        } | benchmarks | {'time': int((time.time()-start) * 1000)}
                df.loc[len(df)] = column
        print()
        df2 = aggregate_data(df, methods[0].__name__, methods[1].__name__)
        print(df2.tail(1).to_string(header=False)) 
    df2 = aggregate_data(df, methods[0].__name__, methods[1].__name__)
    df2.to_csv('aggregated.csv')
    print(df2.to_string())
    df2 = aggregate_data_depth(df, methods[0].__name__, methods[1].__name__)
    df2.to_csv('aggregated_depth.csv')
    print('Depth data')
    print(df2.to_string())

def random_pauli_experiment(backend, methods, logical_qubits=None, nr_gadgets=100, nr_steps=5, rounds=20, max_legs=None, 
                            verify=True, mapping_method=I_index_mapping, steps=None, allowed_legs=[X,Y,Z]):
    topo = get_topo(backend['name'], backend['qubits'])
    print(topo)
    if logical_qubits is None:
        logical_qubits = backend['qubits']
    if logical_qubits > backend['qubits']:
        raise ValueError('Logical qubits cannot be more than physical qubits')
    print('----------------------------------Experiment')
    print('mapping versus random')
    print('Backend:', backend['name'])
    print('Num physical qubits:',  backend['qubits'])
    print('Num logical qubits:',  logical_qubits)
    print('Random iterations:', rounds)
    print('Mapping method:', mapping_method.__name__)
    print('Synthesis methods:', [m.__name__ for m in methods])
    print('Verifying:', verify)
    df = pd.DataFrame(columns=['n_rep','num_qubits','n_gadgets','method','mapping','cx','cx_depth','time','pre-cx'])
    if not steps:
        steps = list(range(nr_steps, nr_gadgets, nr_steps))
    for num_gadgets in steps:
        for i in range(rounds):
#            print(i)
            if max_legs is None:
                pp = create_random_pauli_polynomial(logical_qubits, num_gadgets, allowed_legs=allowed_legs)
            else:
                pp = create_random_pauli_polynomial(logical_qubits, num_gadgets, min_legs=1, max_legs=max_legs, allowed_legs=allowed_legs)
            pp = simplify_pauli_polynomial(pp, allow_acs=True)
            mapping_time = time.time()
            map = mapping_method(pp, topo)
            mapping_time = int((time.time() - mapping_time) * 1000)
            topo_r = qubit_graph(random_mapping(topo), topo)
            topo_m = qubit_graph(map, topo)
#            print(topo_m)
            pp_m = permute_with_mapping(map, pp, topo.num_qubits)
            pp_r = permute_with_mapping(random_mapping(topo), pp, topo.num_qubits)

            for synth_method in methods:
                start = time.time()
                circ_out, gadget_perm, perm, benchmarks = synth_method(pp.copy(), topo_m)
                if verify:
                    correct = check_circuit_equivalence(pp.copy(), circ_out, gadget_perm, perm)
                    if not correct:
                        print('Circuit equivalence failed for', synth_method.__name__, 'with algorithm mapping', mapping_method.__name__)
                        input()
                column = {
                            'n_rep': i,
                            'num_qubits': logical_qubits,
                            'n_gadgets': num_gadgets,
                            'method': synth_method.__name__,
                            'mapping': 'algorithm',
                            'cx': cnot_count(circ_out),
                            'cx_depth': cnot_depth(circ_out),
                        } | benchmarks | {'time': mapping_time + int((time.time()-start) * 1000)}
                df.loc[len(df)] = column
                start = time.time()
                circ_out, gadget_perm, perm, benchmarks = synth_method(pp.copy(), topo_r)
                if verify:
                    correct = check_circuit_equivalence(pp.copy(), circ_out, gadget_perm, perm)
                    if not correct:
                        print('Circuit equivalence failed for', synth_method.__name__, 'with random mapping', mapping_method.__name__)
                        input()
                column = {
                            'n_rep': i,
                            'num_qubits': logical_qubits,
                            'n_gadgets': num_gadgets,
                            'method': synth_method.__name__,
                            'mapping': 'random',
                            'cx': cnot_count(circ_out),
                            'cx_depth': cnot_depth(circ_out),
                        } | benchmarks | {'time': int((time.time()-start) * 1000)}
                df.loc[len(df)] = column
        df2 = aggregate_data(df, methods[0].__name__, methods[1].__name__)
        print(df2.tail(1).to_string(header=False)) 
    df2 = aggregate_data(df, methods[0].__name__, methods[1].__name__)
    df2.to_csv('aggregated.csv')
    print(df2.to_string())
    df2 = aggregate_data_depth(df, methods[0].__name__, methods[1].__name__)
    df2.to_csv('aggregated_depth.csv')
    print('Depth data')
    print(df2.to_string())


def all_mappings(pp, backend, synth_method=pauli_polynomial_steiner_gray_clifford, verbose=True):
    topo = get_topo(backend['name'], backend['qubits'])
    if pp.num_qubits > backend['qubits']:
        raise ValueError('Logical qubits cannot be more than physical qubits')

    verbose and print('---------------------------------- Experiment')
    verbose and print('all mappings')
    verbose and print('Backend:', backend['name'])
    verbose and print('Num physical qubits:', backend['qubits'])
    verbose and print('Num logical qubits:', pp.num_qubits)
    verbose and print('Number of gadgets:', pp.num_gadgets)
    verbose and print('Synthesis method:', synth_method.__name__)
    min_cx = -1
    max_cx = -1
    cx_sum = 0
    count = 0
    min_cx_mapping = None
    max_cx_mapping = None

    output_csv = f'all_mappings_{backend['name']}_{pp.num_qubits}.csv'
    df = pd.DataFrame(columns=['n_rep','num_qubits','n_gadgets','method','mapping','cx','cx_depth', 'pre-cx','broken_chains','steiner_nodes','I-index','doubles','q-corr'])

    for m in itertools.permutations(list(range(backend['qubits']))):
        verbose and sys.stdout.write(str(m[0]))
        verbose and sys.stdout.flush()
        pp_m = permute_with_mapping(m, pp, topo.num_qubits)
        
        circ_out, gadget_perm, perm, benchmarks = synth_method(pp_m.copy(), topo)
        steiner_nodes, broken_chains, doubles, steiner_nodesx, steiner_nodesz = steiner_tree_analysis(pp_m, topo)
        column = {
                    'num_qubits': topo.num_qubits,
                    'n_gadgets': pp.num_gadgets,
                    'method': synth_method.__name__,
                    'mapping': qubit_order(m),
                    'cx': cnot_count(circ_out),
                    'broken_chains': broken_chains,
                    'steiner_nodes': steiner_nodes,
                    'cx_depth': cnot_depth(circ_out),
                    'I-index': I_index(pp_m, topo),
                    'doubles': doubles,
                    'q-corr': qubit_correlation_sum(pp_m, topo)
                } | benchmarks
        df.loc[len(df)] = column
        cx_sum += benchmarks['pre-cx']
        count += 1
        if min_cx == -1 or benchmarks['pre-cx'] < min_cx:
            min_cx = benchmarks['pre-cx']
            min_cx_mapping = m
        if max_cx == -1 or benchmarks['pre-cx'] > max_cx:
            max_cx = benchmarks['pre-cx']
            max_cx_mapping = m
    verbose and print('\nMinimum pre-CX count:', min_cx, 'with order', qubit_order(min_cx_mapping)) 
    verbose and print('Maximum pre-CX count:', max_cx, 'with order', qubit_order(max_cx_mapping))

    df.to_csv(output_csv)
    return min_cx_mapping, max_cx_mapping, int(cx_sum / count)

def qubit_order(mapping, physical_qubits=None):
    if physical_qubits is None:
        order = [-1 for _ in range(len(mapping))]
    else:
        order = [-1 for _ in range(physical_qubits)]
    for i in range(len(mapping)):
        order[mapping[i]] = i
    return order

def get_mapping_from_order(order):
    mapping = [0 for _ in range(len(order))]
    for i in range(len(order)):
        mapping[order[i]] = i
    return mapping

def test_randomness(pp, backend, mapping, synth_method, rounds=1000, verbose=True):
    topo = get_topo(backend['name'], backend['qubits'])
    pp_m = permute_with_mapping(mapping, pp, topo.num_qubits)
    min_cx = -1
    max_cx = -1
    sum_cx = 0
    count = 0
    verbose and print('Testing randomness of mapping:', qubit_order(mapping),'for', rounds, 'rounds')
    verbose and print('Backend:', backend)
    verbose and print('Synthesis method:', synth_method.__name__)
    for _ in range(rounds):
        circ_out, gadget_perm, perm, benchmarks = synth_method(pp_m.copy(), topo, random_sel=True)
        if min_cx == -1 or benchmarks['pre-cx'] < min_cx:
            min_cx = benchmarks['pre-cx']
        if max_cx == -1 or benchmarks['pre-cx'] > max_cx:
            max_cx = benchmarks['pre-cx']
        sum_cx += benchmarks['pre-cx']
        count += 1
    verbose and print('Minimum pre-CX count:', min_cx)
    verbose and print('Maximum pre-CX count:', max_cx)
    verbose and print('Mean pre-CX count:', int(sum_cx / count))
    return int(sum_cx / count)

def test_randomness_with_several_pps(backend, synth_method, logical_qubits, gadgets):
    for i in range(10):
        pp = create_random_pauli_polynomial(logical_qubits, gadgets, empty_qubits=0)
        pp = simplify_pauli_polynomial(pp, allow_acs=True)
        min_cx_mapping, max_cx_mapping, mean_cx = all_mappings(pp, backend, synth_method=synth_method, verbose=False)
        print('testing min_cx randomness')
        mean_min = test_randomness(pp, backend, min_cx_mapping, synth_method, rounds=1000, verbose=False)
        print('testing max_cx randomness')
        mean_max = test_randomness(pp, backend, max_cx_mapping, synth_method, rounds=1000, verbose=False)
        print('mean min:', mean_min, 'mean max:', mean_max)
    input()

def reorder_gadgets(pp, order):
    ppn = PauliPolynomial(num_qubits=pp.num_qubits)
    orderr = [order[i] for i in range(len(order))]
    for i in range(len(pp.pauli_gadgets)):
        ppn >>= pp.pauli_gadgets[orderr[i]].copy()
    return ppn


def test_random_gadget_ordering(pp, backend, mapping, synth_method, rounds=1000, verbose=True):
    topo = get_topo(backend['name'], backend['qubits'])
    pp_m = permute_with_mapping(mapping, pp, topo.num_qubits)
    min_cx = -1
    max_cx = -1
    sum_cx = 0
    count = 0
    verbose and print('Testing randomness of mapping:', qubit_order(mapping),'for', rounds, 'rounds')
    verbose and print('Backend:', backend)
    verbose and print('Synthesis method:', synth_method.__name__)
    for _ in range(rounds):
        num_gadgets = len(pp_m.pauli_gadgets)
        gadget_order = list(range(num_gadgets))
        random.shuffle(gadget_order)
#        print(gadget_order)
        ppn = reorder_gadgets(pp_m, gadget_order)
#        print(ppn.pauli_gadgets)
#        input()
        circ_out, gadget_perm, perm, benchmarks = synth_method(ppn, topo)
        if min_cx == -1 or benchmarks['pre-cx'] < min_cx:
            min_cx = benchmarks['pre-cx']
            min_gadget_order = gadget_order
        if max_cx == -1 or benchmarks['pre-cx'] > max_cx:
            max_cx = benchmarks['pre-cx']
            max_gadget_order = gadget_order
        sum_cx += benchmarks['pre-cx']
        count += 1
    verbose and print('Minimum pre-CX count:', min_cx)
    verbose and print('with gadget order', min_gadget_order)
    pp = reorder_gadgets(pp_m, min_gadget_order)
    order = order_gadgets(pp, topo)
    verbose and print_pp(pp, order=order)
    verbose and print('Maximum pre-CX count:', max_cx)
    verbose and print('with gadget order', max_gadget_order)
    pp = reorder_gadgets(pp_m, max_gadget_order)
    order = order_gadgets(pp, topo)
    verbose and print_pp(pp, order=order)
    verbose and print('Mean pre-CX count:', int(sum_cx / count))
    return int(sum_cx / count)

if __name__ == "__main__":
    verify = False
    methods = [pauli_polynomial_dynamic_ordering, pauli_polynomial_steiner_gray_clifford]
    mapping_method = I_index_mapping
#    mapping_method = zx_index_mapping
#    mapping_method = mapping_by_balance
#    mapping_method = pauli_forest_mapping
#    mapping_method = I_to_edge
    random.seed(42)
    steps = list(range(2, 40, 2)) + list(range(40, 220, 20))
#    steps = list(range(200, 1000, 50))
#    steps = list(range(20, 220, 20))
#    steps = list(range(20, 220, 20)) + list(range(200, 2000, 100))
#    steps = [320]
#    backend = {'name': 'quito', 'qubits': 5}
 #   backend = {'name': 'guadalupe', 'qubits': 16}
    backend = {'name': 'grid', 'qubits': 9}
#    backend = {'name': 'line', 'qubits': 6}
#    backend = {'name': 'cycle', 'qubits': 10}
#    backend = {'name': 'brisbane', 'qubits': 127}
#    trivial_mapping = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    logical_qubits = 9
    trivial_mapping = list(range(logical_qubits))
    max_legs = None
    
#    print_brisbane_topo()
#    trivial_pauli_experiment(trivial_mapping, backend, methods, logical_qubits=logical_qubits, nr_gadgets=200, nr_steps=20, rounds=200, verify=verify, mapping_method=mapping_method, steps=steps, max_legs=max_legs)
    random_pauli_experiment(backend, methods, logical_qubits=logical_qubits, nr_gadgets=200, nr_steps=20, rounds=200, 
                            allowed_legs=[Z, X, Y], verify=verify, mapping_method=mapping_method, steps=steps, max_legs=max_legs)
    input()

    seed = 42
    gadgets = 80
    backend = {'name': 'line', 'qubits': 6}
    logical_qubits = 6
    random.seed(seed)
    pp = create_random_pauli_polynomial(logical_qubits, gadgets, seed=seed, empty_qubits=0, allowed_legs=[Z,X,Y])
    pp = simplify_pauli_polynomial(pp, allow_acs=True)
    print('native paulis')
    print('gadgets after simplification:', len(pp.pauli_gadgets))
    print_pp(pp)
#    pp = reorder_gadgets(pp, [22, 54, 34, 11, 25, 23, 38, 59, 69, 5, 36, 40, 12, 45, 13, 10, 17, 18, 58, 8, 53, 20, 31, 44, 46, 47, 65, 39, 3, 1, 16, 60, 14, 6, 37, 49, 57, 42, 26, 28, 15, 68, 48, 4, 0, 27, 35, 9, 51, 55, 2, 43, 66, 62, 24, 41, 19, 50, 63, 70, 30, 56, 52, 61, 67, 29, 21, 7, 32, 33, 64])


    topo = get_topo(backend['name'], backend['qubits'])
    synth_method = pauli_polynomial_dynamic_ordering
#    synth_method = pauli_polynomial_steiner_gray_clifford
#    mapping = mapping_method(pp, topo)
    mapping, tree = pauli_tree_mapping(pp, topo)
#    mapping = get_mapping_from_order([5, 2, 1, 4, 0, 3]) # bad, but few steiner nodes
#    mapping = get_mapping_from_order([0, 1, 3, 4, 5, 2]) # good, with same steiner nodes as above
#    mapping = get_mapping_from_order([1, 2, 0, 4, 3, 5]) # best with 80,42
#    mapping = get_mapping_from_order([0, 1, 2, 3, 4, 5]) 
#    mapping = get_mapping_from_order([4, 1, 0, 3, 5, 2]) # worst with 80,42
#    print('mapping algorithm order', qubit_order(mapping))
#    test_randomness(pp, backend, mapping, synth_method, rounds=10000)

#    min_cx_mapping, max_cx_mapping, mean_cx = all_mappings(pp, backend, synth_method=synth_method)
#   print('testing min_cx randomness')
#    mean_min = test_random_gadget_ordering(pp, backend, min_cx_mapping, synth_method, rounds=1000)
#    print('testing max_cx randomness')
#    mean_max = test_random_gadget_ordering(pp, backend, max_cx_mapping, synth_method, rounds=1000)
#    input()
#    test_random_gadget_ordering(pp, backend, mapping, synth_method, rounds=1000)
#    input()

#    for x in [[1, 2, 5, 4, 0, 3], [3, 0, 5, 2, 4, 1],[0, 5, 4, 3, 2, 1],[3, 4, 0, 5, 2, 1],[0, 5, 2, 3, 4, 1],[3, 4, 0, 2, 5, 1],[3, 4, 5, 0, 2, 1],[1, 4, 2, 5, 0, 3],[1, 5, 4, 3, 2, 0],[0, 5, 3, 1, 4, 2],[0, 3, 5, 4, 1, 2],[0, 3, 4, 5, 2, 1],[4, 0, 3, 5, 2, 1],[3, 5, 2, 0, 4, 1],[2, 1, 5, 4, 0, 3],[1, 3, 5, 4, 2, 0],[0, 3, 5, 2, 4, 1],[2, 0, 4, 3, 1, 5],[4, 1, 0, 5, 2, 3],[3, 5, 2, 0, 1, 4],[4, 3, 5, 0, 2, 1],[1, 4, 2, 3, 0, 5],[5, 1, 3, 4, 0, 2],[3, 4, 5, 2, 0, 1],[2, 1, 4, 3, 5, 0]]:
#        test_randomness(pp, backend, get_mapping_from_order(x), synth_method, rounds=1000)
#    input()

#    test_randomness_with_several_pps(backend, synth_method, logical_qubits, 20)
#   input()
    topo_m = qubit_graph(mapping, topo)
    pp_m = permute_with_mapping(mapping, pp, topo.num_qubits)
    order = order_gadgets(pp, topo)
    print('pre defined order with mapping')
    print_pp(pp, order=order)
#    order = [0,1,2,3,4,5,6,7,8,9]
#    order = [3, 24, 26, 27, 49, 63, 64, 50, 20, 4, 55, 32, 17, 8, 45, 47, 57, 51, 11, 2, 37, 61, 30, 38, 43, 34, 66, 46, 14, 65, 36, 29, 70, 19, 52, 0, 60, 31, 39, 1, 5, 12, 9, 10, 54, 16, 69, 33, 18, 44, 25, 58, 68, 35, 15, 7, 67, 48, 28, 21, 42, 40, 56, 6, 62, 23, 22, 41, 53, 59, 13]
#    order =  [3, 24, 26, 27, 49, 63, 64, 65, 47, 57, 38, 14, 34, 5, 8, 66, 20, 61, 29, 32, 55, 33, 68, 40, 10, 31, 54, 39, 17, 16, 59, 67, 4, 51, 44, 58, 25, 50, 9, 23, 2, 69, 0, 21, 45, 7, 48, 12, 6, 28, 11, 37, 56, 13, 60, 62, 1, 30, 43, 41, 46, 35, 36, 18, 19, 22, 42, 70, 53, 15, 52]
#    order = [0, 4, 9, 5, 8, 15, 3, 12, 1, 7, 6, 13, 18, 11, 17, 16, 2, 14, 10] # optimal mapping
#    order = [8, 10, 14, 20, 31, 39, 26, 12, 35, 32, 38, 28, 25, 33, 23, 7, 40, 16, 42, 15, 5, 4, 9, 36, 6, 22, 37, 24, 1, 27, 34, 43, 30, 41, 11, 3, 0, 17, 29, 2, 18, 21, 19, 13]

    print('manual order')
    print_pp(pp, order=order)
    circ_out, gadget_perm, perm, benchmarks = pauli_polynomial_dynamic_ordering(pp.copy(), topo_m, debug=False, print_order=order, random_sel=False)
#    circ_out, gadget_perm, perm, benchmarks = pauli_polynomial_steiner_gray_clifford(pp.copy(),topo, random_sel=False)
    print('CNOT count:',cnot_count(circ_out))
    print('Pre-cx:', benchmarks['pre-cx'])
    print('Density:\n', benchmarks['density'])
    print('seed',seed)
    print('gadget perm', gadget_perm)
    print()
    verify = check_circuit_equivalence(pp.copy(), circ_out, gadget_perm, perm)
    print('Circuit equivalence:', verify)
#    input()

#    test_randomness(pp, backend, mapping, synth_method)
#    input()

