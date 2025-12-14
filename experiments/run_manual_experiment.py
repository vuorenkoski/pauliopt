import random, time, itertools, sys
import pandas as pd

from pauliopt.pauli.synthesis.steiner_gray_synthesis import pauli_polynomial_steiner_gray_clifford
from pauliopt.pauli.synthesis.shortest_path_pauli_forest import shortest_path_pauli_forest
from pauliopt.pauli.synthesis.pp_mapping import I_index_mapping, pauli_tree_mapping, complete_tree
from tests.pauli.utils import verify_equality
from pauliopt.pauli.simplification.simple_simplify import simplify_pauli_polynomial

from pauliopt.pauli.pauli_polynomial import PauliPolynomial, I, Z, X, Y
from experiments.utils import permute_with_mapping, qubit_correlation_sum, random_mapping, I_index, cnot_depth, two_qubit_gates_pauliopt
from experiments.utils import create_random_pauli_polynomial, steiner_tree_analysis, order_gadgets
from experiments.utils import cnot_count, print_pp, aggregate_data, get_topo, aggregate_data_depth, map_topology
from experiments.utils import print_brisbane_mapping, map_tree, create_complete_pauli_polynomial
from experiments.utils import print_grid25_mapping, print_brisbane_topo

from pauliopt.pauli.pauli_gadget import PauliGadget
import pickle
from pauliopt.utils import AngleVar

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
    print('trivial mapping')
    print_brisbane_mapping(trivial_mapping, None)
#    input()
    df = pd.DataFrame(columns=['n_rep','num_qubits','n_gadgets','method','mapping','cx','cx_depth','time','pre-cx'])
    if not steps:
        steps = list(range(nr_steps, nr_gadgets, nr_steps))
    for num_gadgets in steps:
        for i in range(rounds):
#            sys.stdout.write(str(i))
#            sys.stdout.flush()
            if max_legs is None:
                pp = create_random_pauli_polynomial(logical_qubits, num_gadgets)
            else:
                pp = create_random_pauli_polynomial(logical_qubits, num_gadgets, min_legs=1, max_legs=max_legs)
            pp = simplify_pauli_polynomial(pp, allow_acs=True)
            mapping_time = time.time()
            map,tree = mapping_method(pp, topo)
#            print_brisbane_mapping(map, tree)
            mapping_time = int((time.time() - mapping_time) * 1000)
#            print(mapping_time)
            topo_t = map_topology(trivial_mapping, topo)
            topo_m = map_topology(map, topo)
            tree = map_tree(map, tree)
#            pp_m = permute_with_mapping(map, pp, topo.num_qubits)
#            pp_r = permute_with_mapping(trivial_mapping, pp, topo.num_qubits)

            for synth_method in methods:
                start = time.time()
                circ_out, gadget_perm, perm, benchmarks = synth_method(pp.copy(), topo_m, tree)
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
                circ_out, gadget_perm, perm, benchmarks = synth_method(pp.copy(), topo_t, None)
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
            map, tree = mapping_method(pp, topo)
            map_r = random_mapping(topo)
            tree_r = complete_tree(topo)
#            print_grid25_mapping(map, tree)
#            print_grid25_mapping(map_r, tree_r)
            mapping_time = int((time.time() - mapping_time) * 1000)
#            print(topo_m)
            pp_m = permute_with_mapping(map, pp, topo.num_qubits)
            pp_r = permute_with_mapping(map_r, pp, topo.num_qubits)

            for synth_method in methods:
                start = time.time()
                circ_out, gadget_perm, perm, benchmarks = synth_method(pp_m.copy(), topo, tree)
                if verify:
                    correct = check_circuit_equivalence(pp_m.copy(), circ_out, gadget_perm, perm)
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
                circ_out, gadget_perm, perm, benchmarks = synth_method(pp_r.copy(), topo, tree_r)
                if verify:
                    correct = check_circuit_equivalence(pp_r.copy(), circ_out, gadget_perm, perm)
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
        df2.to_csv('aggregated_'+backend['name']+str(backend['qubits'])+'.csv')
        df2 = aggregate_data_depth(df, methods[0].__name__, methods[1].__name__)
        df2.to_csv('aggregated_'+backend['name']+str(backend['qubits'])+'_depth.csv')

    df2 = aggregate_data(df, methods[0].__name__, methods[1].__name__)
    df2.to_csv('aggregated_'+backend['name']+str(backend['qubits'])+'.csv')
    print(df2.to_string())
    df2 = aggregate_data_depth(df, methods[0].__name__, methods[1].__name__)
    df2.to_csv('aggregated_'+backend['name']+str(backend['qubits'])+'_depth.csv')
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
    df = pd.DataFrame(columns=['n_rep','num_qubits','n_gadgets','method','mapping','last_leg','cx','cx_depth', 'pre-cx','broken_chains','steiner_nodes','I-index','doubles','q-corr'])
    mapping, tree = pauli_tree_mapping(pp, topo)

    for m in itertools.permutations(list(range(backend['qubits']))):
        verbose and sys.stdout.write(str(m[0]))
        verbose and sys.stdout.flush()
        pp_m = permute_with_mapping(m, pp, topo.num_qubits)
        
        circ_out, gadget_perm, perm, benchmarks = synth_method(pp_m.copy(), topo, tree)
        steiner_nodes, broken_chains, doubles, steiner_nodesx, steiner_nodesz = steiner_tree_analysis(pp_m, topo)
        column = {
                    'num_qubits': topo.num_qubits,
                    'n_gadgets': pp.num_gadgets,
                    'method': synth_method.__name__,
                    'mapping': qubit_order(m),
                    'last_leg': benchmarks.get('last_leg', None),
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
    verbose and print('Testing randomness of mapping for', rounds, 'rounds')
    print_brisbane_mapping(mapping, tree)

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


def test_random_gadget_ordering(pp, backend, mapping, tree,synth_method, rounds=1000, verbose=True):
    topo = get_topo(backend['name'], backend['qubits'])
#    topo_m = map_topology(mapping, topo)
    pp_m = permute_with_mapping(mapping, pp, topo.num_qubits)
#    tree_m = map_tree(mapping, tree)
    min_cx = -1
    max_cx = -1
    sum_cx = 0
    count = 0
    verbose and print('Testing randomness of mapping: for', rounds, 'rounds')
    verbose and print_brisbane_mapping(mapping, tree)
    verbose and print('Backend:', backend)
    verbose and print('Synthesis method:', synth_method.__name__)
    for _ in range(rounds):
        num_gadgets = len(pp.pauli_gadgets)
        gadget_order = list(range(num_gadgets))
        random.shuffle(gadget_order)
#        print(gadget_order)
        ppn = reorder_gadgets(pp_m, gadget_order)
#        print(ppn.pauli_gadgets)
#        input()
        circ_out, gadget_perm, perm, benchmarks = synth_method(ppn, topo, tree)
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
#    pp = reorder_gadgets(pp_m, min_gadget_order)
#    order = order_gadgets(pp, topo)
#    verbose and print_pp(pp, order=order)
    verbose and print('Maximum pre-CX count:', max_cx)
    verbose and print('with gadget order', max_gadget_order)
#    pp = reorder_gadgets(pp_m, max_gadget_order)
#    order = order_gadgets(pp, topo)
#    verbose and print_pp(pp, order=order)
    verbose and print('Mean pre-CX count:', int(sum_cx / count))
    return int(sum_cx / count)


def molecule_pp(filename):
    with open(filename, "rb") as pickle_in:
        pp = pickle.load(pickle_in)
        gadgets = []
        for gadget in pp.pauli_gadgets:
            if not isinstance(gadget.angle, AngleVar):
                gadget2 = PauliGadget(gadget.angle, gadget.paulis)
            else:
                gadget2 = PauliGadget(AngleVar(gadget.angle._label,gadget.angle._latex_label), gadget.paulis)
            gadgets.append(gadget2)
        pp.pauli_gadgets = gadgets
        pp = simplify_pauli_polynomial(pp, allow_acs=True)
    return pp

def pad_pp_to_larger_backend(pp: PauliPolynomial, n_qubits):
    pp_ = PauliPolynomial(n_qubits)

    pp_qubits = pp.num_qubits

    identity_pad = [I for _ in range(n_qubits - pp_qubits)]

    for gadget in pp:
        assert isinstance(gadget, PauliGadget)
        angle = gadget.angle
        paulis = gadget.paulis + identity_pad

        pp_ >>= PauliGadget(angle, paulis)
    return pp_

if __name__ == "__main__":
    #
    # Testing single pauli polynomial
    #
    seed = 42
    random.seed(seed)
    gadgets = 80
    synth_method = shortest_path_pauli_forest
    synth_method2 = pauli_polynomial_steiner_gray_clifford
#    backend = {'name': 'line', 'qubits': 6}
#    backend = {'name': 'grid', 'qubits': 16}
#    backend = {'name': 'guadalupe', 'qubits': 16}
    backend = {'name': 'brisbane', 'qubits': 127}
    print('Backend:', backend)
    pp = molecule_pp('./pp_molecules/H2O_BK_sto3g.pickle')
    logical_qubits = pp.num_qubits
    print('Number of logical qubits:', logical_qubits)
#    pp = create_random_pauli_polynomial(logical_qubits, gadgets, seed=seed, empty_qubits=0, allowed_legs=[Z,X,Y])
#    pp = create_complete_pauli_polynomial(backend['qubits'], allowed_legs=[Z,X,Y])
    print('simplifying...')
    pp = simplify_pauli_polynomial(pp, allow_acs=True)
#    pp = pad_pp_to_larger_backend(pp, backend['qubits'])
#    print('native paulis')
    print('gadgets after simplification:', len(pp.pauli_gadgets))
#    print_pp(pp)

    topo = get_topo(backend['name'], backend['qubits'])
#    synth_method = pauli_polynomial_steiner_gray_clifford
    print('Mapping...')
    mapping, tree = pauli_tree_mapping(pp, topo) # mapping produces 112 with 80,42
#    print_grid25_mapping(mapping, tree)
#    print('algorithm mapping', qubit_order(mapping))
    print_brisbane_mapping(mapping, tree)
#    print('tree', tree)
#    mapping = get_mapping_from_order([1, 2, 0, 4, 3, 5]) # best with 80,42: pre-cx 100
#    mapping = get_mapping_from_order([1, 2, 4, 0, 5, 3]) # pre-cx 138
#    print('used mapping', qubit_order(mapping))

#    mapping = get_mapping_from_order([4, 3, 2, 0, 1, 5]) # good, with same steiner nodes as above
#    mapping = get_mapping_from_order([1, 2, 0, 4, 3, 5]) # best with 80,42
#    mapping = get_mapping_from_order([0, 1, 2, 3, 4, 5]) 
#    print('mapping algorithm order', qubit_order(mapping))
#    test_randomness(pp, backend, mapping, synth_method, rounds=10000)

#    pp = reorder_gadgets(pp, [53, 60, 21, 57, 33, 49, 41, 23, 51, 22, 43, 13, 28, 15, 5, 40, 59, 54, 61, 31, 7, 69, 46, 36, 47, 10, 9, 1, 12, 52, 48, 38, 26, 70, 20, 44, 68, 66, 18, 11, 42, 67, 56, 8, 37, 24, 63, 29, 14, 16, 25, 65, 19, 58, 27, 55, 0, 64, 4, 39, 62, 3, 2, 30, 35, 17, 50, 32, 6, 45, 34])
#    min_cx_mapping, max_cx_mapping, mean_cx = all_mappings(pp, backend, synth_method=synth_method)
#    print('testing min_cx randomness')
#    mean_min = test_random_gadget_ordering(pp, backend, min_cx_mapping, tree, synth_method, rounds=1000)
#    print('testing max_cx randomness')
#    mean_max = test_random_gadget_ordering(pp, backend, max_cx_mapping, tree, synth_method, rounds=1000)
#    input()
#    mapping = [0,1,2]
#    test_random_gadget_ordering(pp, backend, mapping, tree, synth_method, rounds=10000)
#    input()
    pp_m = permute_with_mapping(mapping, pp, topo.num_qubits)

    # this is ver good ordering for mapping algorithm, produces 95 cnots (compared to 112 in trivial order)
#    pp_m = reorder_gadgets(pp_m, [48, 31, 23, 3, 27, 43, 16, 4, 32, 38, 29, 37, 11, 1, 7, 28, 53, 14, 13, 18, 21, 49, 52, 51, 22, 35, 10, 2, 9, 34, 5, 0, 33, 44, 46, 40, 30, 6, 36, 26, 50, 8, 47, 45, 12, 42, 25, 39, 15, 41, 24, 19, 17, 20])
#    pp_m = reorder_gadgets(pp_m, [12, 33, 46, 31, 35, 14, 49, 25, 41, 21, 2, 51, 6, 39, 53, 13, 17, 19, 44, 48, 1, 10, 15, 38, 7, 3, 34, 43, 45, 36, 16, 42, 52, 20, 24, 29, 23, 0, 9, 5, 32, 37, 11, 8, 30, 18, 27, 50, 40, 26, 28, 22, 47, 4])
#    print_pp(pp_m)
    # this is poor ordering 138
#    pp_m = reorder_gadgets(pp_m, [40, 20, 69, 2, 53, 23, 19, 32, 41, 15, 68, 55, 28, 16, 37, 29, 22, 21, 4, 33, 52, 5, 43, 14, 26, 9, 38, 61, 48, 67, 70, 18, 46, 45, 7, 66, 36, 47, 56, 57, 42, 62, 35, 54, 27, 39, 34, 17, 6, 0, 60, 49, 10, 44, 58, 64, 24, 8, 30, 13, 51, 63, 1, 31, 3, 11, 59, 25, 65, 50, 12])
#    test_randomness_with_several_pps(backend, synth_method, logical_qubits, 20)
#   input()
#    topo_m = map_topology(mapping, topo)
#    tree_m = map_tree(mapping, tree)
    print_order = order_gadgets(pp_m, topo)
#    print('pre defined order with mapping')
#    print_pp(pp_m, order=print_order)
#    print_order = [1, 4, 40, 53, 6, 25, 44, 14, 17, 51, 52, 0, 24, 35, 45, 9, 33, 39, 48, 3, 13, 21, 28, 2, 8, 36, 50, 11, 12, 19, 41, 10, 22, 42, 47, 18, 29, 38, 43, 7, 26, 27, 49, 15, 20, 32, 46, 16, 34, 5, 30, 37, 23, 31]
#    print_order = [0, 4, 30, 47, 2, 31, 33, 3, 46, 1, 40, 43, 44, 6, 8, 23, 9, 28, 38, 50, 5, 10, 12, 32, 7, 17, 53, 11, 13, 20, 35, 21, 26, 22, 48, 14, 18, 16, 24, 36, 37, 15, 25, 41, 39, 42, 27, 34, 45, 52, 49, 51, 19, 29]
#    print('manual order')
#    print_pp(pp_m, order=print_order)
    print('synthesizing...')
    start = time.time()
    circ_out, gadget_perm, perm, benchmarks = synth_method(pp_m.copy(), topo, tree, debug=False, print_order=print_order, random_sel=False)
    print('synthesis time (ms):', int((time.time() - start)*1000))
    active_qubits = two_qubit_gates_pauliopt(circ_out)['active_qubits']
    print_brisbane_topo(active_qubits)
#    circ_out, gadget_perm, perm, benchmarks = pauli_polynomial_steiner_gray_clifford(pp.copy(),topo, random_sel=False)
    print('CNOT count:',cnot_count(circ_out))
    print('CNOT depth:',cnot_depth(circ_out))
    print(benchmarks)
 #   print('gadget perm', gadget_perm)
    print()
    print('steiner-gray synthesis:')
    start = time.time()
    circ_out, gadget_perm, perm, benchmarks = synth_method2(pp, topo, None)
    print('synthesis time (ms):', int((time.time() - start)*1000))
    active_qubits = two_qubit_gates_pauliopt(circ_out)['active_qubits']
    print_brisbane_topo(active_qubits)
#    circ_out, gadget_perm, perm, benchmarks = pauli_polynomial_steiner_gray_clifford(pp.copy(),topo, random_sel=False)
    print('CNOT count:',cnot_count(circ_out))
    print('CNOT depth:',cnot_depth(circ_out))
    print(benchmarks)
#    print('verifying circuit equivalence...')
#    verify = check_circuit_equivalence(pp_m.copy(), circ_out, gadget_perm, perm)
#    print('Circuit equivalence:', verify)
#    input()

#    test_randomness(pp, backend, mapping, synth_method)
#    input()

