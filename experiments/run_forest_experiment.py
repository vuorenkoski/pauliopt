import random, time
import pandas as pd

from pauliopt.pauli.synthesis.steiner_gray_synthesis import pauli_polynomial_steiner_gray_clifford
from pauliopt.pauli.synthesis.shortest_path_pauli_forest import shortest_path_pauli_forest
from pauliopt.pauli.synthesis.pp_mapping import I_index_mapping, pauli_tree_mapping, complete_tree
from tests.pauli.utils import verify_equality
from pauliopt.pauli.simplification.simple_simplify import simplify_pauli_polynomial

from pauliopt.pauli.pauli_polynomial import PauliPolynomial, I, Z, X, Y
from experiments.utils import permute_with_mapping, random_mapping, cnot_depth
from experiments.utils import create_random_pauli_polynomial, order_gadgets
from experiments.utils import cnot_count, print_pp, aggregate_data, get_topo, aggregate_data_depth, map_topology
from experiments.utils import print_brisbane_mapping, map_tree, create_complete_pauli_polynomial
from experiments.utils import print_grid25_mapping

def check_circuit_equivalence(pp, circ_out, gadget_perm, perm):
    pp2 = PauliPolynomial(pp.num_qubits)
    pp2.pauli_gadgets = [pp[i].copy() for i in gadget_perm]
    circ = circ_out.to_qiskit()
    pp_circ = pp2.to_qiskit()
    return verify_equality(pp_circ,circ)


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
    first = True
    for num_gadgets in steps:
        for i in range(rounds):
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
        print(df2.tail(1).to_string(header=first))
        first = False 
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


if __name__ == "__main__":
    methods = [shortest_path_pauli_forest, pauli_polynomial_steiner_gray_clifford]
    verify = False
    mapping_method = pauli_tree_mapping
    random.seed(42)
    steps = list(range(2, 40, 2)) + list(range(40, 220, 20)) + list(range(200, 2000, 100))
#    backend = {'name': 'quito', 'qubits': 5}
#    backend = {'name': 'guadalupe', 'qubits': 16}
#    backend = {'name': 'grid', 'qubits': 25}
    backend = {'name': 'line', 'qubits': 6}
#    backend = {'name': 'cycle', 'qubits': 10}
#    backend = {'name': 'brisbane', 'qubits': 127}
    logical_qubits = 6
    samples = 200
    max_legs = None
    
    random_pauli_experiment(backend, methods, logical_qubits=logical_qubits, nr_gadgets=200, nr_steps=20, rounds=samples, 
                           allowed_legs=[Z, X, Y], verify=verify, mapping_method=mapping_method, steps=steps, max_legs=max_legs)
    input()


