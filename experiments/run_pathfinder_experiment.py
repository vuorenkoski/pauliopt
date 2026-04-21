import os
import random, time
import pandas as pd
import numpy as np

from pauliopt.pauli.synthesis.steiner_gray_synthesis import pauli_polynomial_steiner_gray_clifford
from pauliopt.pauli.synthesis.pathfinder_in_pauli_grove import pathfinder_in_pauli_grove
from pauliopt.pauli.synthesis.tree_mapping import pauli_tree_mapping, complete_tree
from tests.pauli.utils import verify_equality
from pauliopt.pauli.simplification.simple_simplify import simplify_pauli_polynomial

from pauliopt.pauli.pauli_polynomial import PauliPolynomial, I, Z, X, Y
from experiments.utils import permute_with_mapping, random_mapping, cnot_depth, create_random_pauli_polynomial
from experiments.utils import cnot_count, aggregate_data, get_topo, aggregate_data_depth
from pauliopt.pauli.pauli_gadget import PauliGadget
import pickle
from pauliopt.utils import AngleVar

def check_circuit_equivalence(pp, circ_out, gadget_perm, perm):
    pp2 = PauliPolynomial(pp.num_qubits)
    pp2.pauli_gadgets = [pp[i].copy() for i in gadget_perm]
    circ = circ_out.to_qiskit()
    pp_circ = pp2.to_qiskit()
    return verify_equality(pp_circ,circ)

def random_pauli_experiment(backend, methods, logical_qubits=None, nr_gadgets=100, nr_steps=5, rounds=20, max_legs=None, 
                            verify=True, mapping_method=None, steps=None, allowed_legs=[X,Y,Z]):
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
    print('Baseline method (m2):', methods[1].__name__)
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
            pp = pad_pp_to_larger_backend(pp, backend['qubits'])
            mapping_time = time.time()
            map, tree = mapping_method(pp, topo)
            map_r = random_mapping(topo)
            tree_r = complete_tree(topo) # This is needed for forest synth with random mapping
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
                
#                print(column)
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
        df2.to_csv('results/aggregated_'+backend['name']+str(backend['qubits'])+'q.csv')
        df2 = aggregate_data_depth(df, methods[0].__name__, methods[1].__name__)
        df2.to_csv('results/aggregated_'+backend['name']+str(backend['qubits'])+'q_depth.csv')

    df2 = aggregate_data(df, methods[0].__name__, methods[1].__name__)
    df2.to_csv('results/aggregated_'+backend['name']+str(backend['qubits'])+'q.csv')
    print(df2.to_string())
    df2 = aggregate_data_depth(df, methods[0].__name__, methods[1].__name__)
    df2.to_csv('results/aggregated_'+backend['name']+str(backend['qubits'])+'q_depth.csv')
    print('Depth data')
    print(df2.to_string())

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

def analyse_molecules():
    op_directory = "./pp_molecules/"
    files = os.listdir(op_directory)
    for filename in files:
        if filename.endswith(".pickle"):
            print('Analysing molecule:', filename)
            pp = molecule_pp(os.path.join(op_directory, filename))
            analyse_pp(pp)


def analyse_pp(pp: PauliPolynomial):
    print('Number of gadgets:', pp.num_gadgets)
    print('Number of qubits:', pp.num_qubits)
    leg_counts = {}
    z_count = 0
    x_count = 0
    y_count = 0
    i_count = 0
    qubits = np.zeros(pp.num_qubits)
    for gadget in pp.pauli_gadgets:
        n_legs = 0
        for i,p in enumerate(gadget.paulis):
            if p == Z:
                z_count += 1
            elif p == X:
                x_count += 1
            elif p == Y:
                y_count += 1
            elif p == I:
                i_count += 1
            if p != I:
                n_legs += 1
                qubits[i] += 1

        if n_legs not in leg_counts:
            leg_counts[n_legs] = 0
        leg_counts[n_legs] += 1
    for n_legs in sorted(leg_counts.keys()):
        print(f'Gadgets with {n_legs} legs: {leg_counts[n_legs]}')
    print('Total Z legs:', z_count)
    print('Total X legs:', x_count)
    print('Total Y legs:', y_count)
    print('Total I legs:', i_count)
    print('Qubit usage (number of legs on each qubit):', qubits)

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

if __name__ == "__main__":
    methods = [pathfinder_in_pauli_grove, pauli_polynomial_steiner_gray_clifford]
    verify = False
    mapping_method = pauli_tree_mapping
    random.seed(42)
    steps = list(range(2, 40, 2)) + list(range(40, 220, 20)) + list(range(200, 2000, 100))
#    backend = {'name': 'quito', 'qubits': 5}
#    backend = {'name': 'guadalupe', 'qubits': 16}
#    backend = {'name': 'grid', 'qubits': 9}
    backend = {'name': 'line', 'qubits': 16}
#    backend = {'name': 'brisbane', 'qubits': 127}
    logical_qubits = 16
    samples = 200
    max_legs = None
    
    random_pauli_experiment(backend, methods, logical_qubits=logical_qubits, nr_gadgets=200, nr_steps=20, rounds=samples, 
                           allowed_legs=[Z, X, Y], verify=verify, mapping_method=mapping_method, steps=steps, max_legs=max_legs)
    input()


