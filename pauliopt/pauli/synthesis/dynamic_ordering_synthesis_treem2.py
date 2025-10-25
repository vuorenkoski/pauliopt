import networkx as nx
from experiments.utils import cnot_count_density
import numpy as np
import random

from pauliopt.clifford.clifford_tableau import CliffordTableau
from pauliopt.gates import CX, Vdg, Sdg
from pauliopt.circuits import Circuit
from pauliopt.pauli.pauli_polynomial import PauliPolynomial
from pauliopt.pauli_strings import I, X, Y, Z
from pauliopt.topologies import Topology
from tests.pauli.utils import verify_equality

# This idea is that algorithm resolves gadgets in order from smallest to largest, chains strings which does not have I
# in the middle are prioritised. One gadgets is chosen from the top of the order each time. This gadget is resolved from
# the edges of the chain/branches. Edge to be removed is chosen in order so that it best shortens next gadgets and/or
# remove I:s from the middle of chain. Order is updated after each gadget is resolved.

# Algorithm creates steiner tree from each gadget with networkx function in the start of the algorithm

# Architecture agnostic version: https://arxiv.org/abs/2404.03280
# This have two major differences: 1) does not consider topology, 2) does not allow ordering

# This six different single qubit gates are used in the algorithm: I, V, S, VS, SV

# Time complexity O(m^2 n x) where n is number of qubits and m number of gadgets. X depends on topology: maximum degree of
# physical qubit (for example in line it is 2 and in grid 4). max of x is n-1.
# we have to process m gadgets and n qubits, so n*m
# Each opearation takes 9*m*x time

def pauli_polynomial_dynamic_ordering_treem(pp: PauliPolynomial, topo: Topology, tree, print_order=None, debug=False, random_sel=False):
    num_qubits = pp.num_qubits
    num_gadgets = len(pp.pauli_gadgets)
    removed_gadgets_num = 0
    gadget_angles = []
    removed_gadgets = np.zeros((num_gadgets), dtype=np.int8)
    qc_out = Circuit(num_qubits)
    qc_prop = []
    perm_gadgets = []

    # Create datastructures
    last_leg = np.zeros((num_qubits), dtype=int) # Last leg in each qubit
    gadget_data = np.zeros((num_qubits,num_gadgets), dtype=np.int8) # Dynamic matrix representing paulis
    pauligraph_degrees = np.zeros((num_gadgets, num_qubits), dtype=np.int8) # Dynamic matrix represnting degrees of node in steiner trees
    last_edge = np.zeros((num_gadgets, num_qubits), dtype=int) # Dynamic matrix representing last edges in tree branches
    for i,gadget in enumerate(pp.pauli_gadgets):
        for j in range(num_qubits):
            last_edge[i,j] = -1
            if gadget.paulis[j] == I:
                gadget_data[j,i] = 0b00
            elif gadget.paulis[j] == X:
                gadget_data[j,i] = 0b01
            elif gadget.paulis[j] == Y:
                gadget_data[j,i] = 0b11
            elif gadget.paulis[j] == Z:
                gadget_data[j,i] = 0b10
            else:
                raise ValueError(f'Unknown Pauli {gadget_data[j,i]} in gadget {i}')
        gadget_angles.append(gadget.angle)
        tree_graph = mapping_tree(gadget_data, topo, tree, i)
        create_pauligraph_from_tree_graph(tree_graph, pauligraph_degrees, last_edge, i)
    tree_graph = topology_tree(gadget_data, tree)
    gate_combinations = {} # Immutable dictionary of possible gates for each pair of paulis and targets
    for p1 in [0b00, 0b01, 0b10, 0b11]:
        for p2 in [0b00, 0b01, 0b10, 0b11]:
            for target0 in [False, True]: # Is target to have I or not after gates in qubit0
                for target1 in [False, True]: # Is target to have I or not after gates in qubit1
                    options = possible_gates((p1, p2), target0, target1)
                    gate_combinations[((p1, p2), target0, target1)] = options
    general_data = (gadget_data, gadget_angles, removed_gadgets, pauligraph_degrees, last_edge, tree_graph)
    circ_data = (qc_out, qc_prop)

    debug and print('---------------------Initial gadget data')
    debug and print_sorted_gd(gadget_data, order=print_order)

    # Main loop going through gadgets starting from smallest
    removed_gadgets_num += check_for_singles(general_data, circ_data, perm_gadgets, last_leg)
    debug and print_sorted_gd(gadget_data, order=print_order)
    while removed_gadgets_num < num_gadgets:
        # Randomness 1: there are for example many gadgets having same size and no steiner nodes, how to arrange them?
        next_options = next_gadget(general_data)
        debug and print('\n\n-------Next gadget options:', next_options)
        
        # Loop going through qubits in gadget, removing them one by one
        if len(next_options) == 1: # there is only one gadget having min length
            gadget = next_options[0]
            num_legs = 0
            for j in range(num_qubits):
                if gadget_data[j,gadget] != 0b00:
                    num_legs += 1
            if num_legs == 0:
                print('ERROR: qubit map has no nodes')
                print('gadget:', gadget)
                print_sorted_gd(gadget_data)
                print(perm_gadgets)
                input()
            while num_legs > 1:
                edge, gates, score = next_edge_to_remove(gadget, general_data, gate_combinations, removed_gadgets_num, random_sel=random_sel)
                removed_gadgets_num += add_cnot_and_single_qubit_gates(edge, gates, True, general_data, circ_data, perm_gadgets, last_leg)
                if gadget_data[edge[0], gadget] == 0b00:
                    num_legs -= 1
        else:
            possibilties = []
            for gadget in next_options:
                num_legs = 0
                for j in range(num_qubits):
                    if gadget_data[j,gadget] != 0b00:
                        num_legs += 1
                if num_legs == 0:
                    print('ERROR: qubit map has no nodes')
                    print('gadget:', gadget)
                    print_sorted_gd(gadget_data)
                    print(perm_gadgets)
                    input()
                gate_sequence = []
                score_sum = (0,0)
                debug and print('-------Evaluating gadget:', gadget, 'num legs:', num_legs, pauligraph_degrees[gadget,:])
                while num_legs > 1:
                    # randomness 2: if there are two or more edge+gate combinations having similar match, which one to choose?
                    edge, gates, score = next_edge_to_remove(gadget, general_data, gate_combinations, removed_gadgets_num, random_sel=random_sel)
                    debug and print('------- trying edge:', edge, 'gates:', gates, 'gadget', gadget, 'score:', score)
                    add_cnot_and_single_qubit_gates(edge, gates, False, general_data, circ_data, perm_gadgets, last_leg)
                    gate_sequence.append((edge, gates))
                    score_sum = (score_sum[0]+score[0], score_sum[1]+score[1])
                    if gadget_data[edge[0], gadget] == 0b00:
                        num_legs -= 1
                possibilties.append((gadget, gate_sequence, score_sum))
                debug and print('-------Gadget after evaluation:', gadget)
                debug and print_sorted_gd(gadget_data, order=print_order)
                revert_gates(gate_sequence, general_data)
                debug and print('-------Gadget after reversion:', gadget)
                debug and print_sorted_gd(gadget_data, order=print_order)
                debug and input()
            possibilties.sort(key=lambda x: (-x[2][0], -x[2][1]))
            debug and print('-------Chosen gadget:', possibilties[0][0], 'gates:', possibilties[0][1])
            best_gadget = possibilties[0][0]
            gate_sequence = possibilties[0][1]
            debug and print('executing',gate_sequence)
            for edge, gates in gate_sequence:
                removed_gadgets_num += add_cnot_and_single_qubit_gates(edge, gates, True, general_data, circ_data, perm_gadgets, last_leg)

    # do clifford synthesis for the second part
    qc_prop_r = list(reversed(qc_prop))
    ct_prop = CliffordTableau(num_qubits)
    for gate in qc_prop_r:
        ct_prop.append_gate(gate)
    qc_prop_syn, permutation = ct_prop.to_clifford_circuit_perm_row_col(topo, include_swaps=False)

    circ_out = qc_out + qc_prop_syn
    circ_out.final_permutation = qc_prop_syn.final_permutation
    permutation = [circ_out.final_permutation[i] for i in range(pp.num_qubits)]

    pre_cx = 0
    for gate in qc_out.gates:
        if isinstance(gate, CX):
            pre_cx += 1
    density = cnot_count_density(qc_out)
    return circ_out, perm_gadgets, permutation, {'pre-cx': pre_cx, 'density': density, 'last_leg': last_leg}

def revert_gates(gate_sequence, general_data):
    gadget_data, gadget_angles, removed_gadgets, pauligraph_degrees, last_edge, tree_graph = general_data
    num_qubits, num_gadgets = gadget_data.shape
    # Propagate gates through all remaining gadgets
    for gadget_index in range(num_gadgets):
        if removed_gadgets[gadget_index]:
            continue
        for edge, gates in reversed(gate_sequence):
    #        print('--- Reverting edge:', edge, 'gates:', gates)
            qubit1, qubit2 = edge
            gate1, gate2 = gates

            pauli1, pauli2 = gadget_data[qubit1,gadget_index], gadget_data[qubit2,gadget_index]
            if pauli1 == 0b00 and pauli2 == 0b00:
                continue
            phase = 1
            # Apply CNOT gate
            pauli1, pauli2, phase_change = apply_cnot(pauli1, pauli2)
            phase *= phase_change
            # Apply single qubit gates
            pauli1, phase_change = gate1(pauli1, reverse=True)
            phase *= phase_change
            pauli2, phase_change = gate2(pauli2, reverse=True)
            phase *= phase_change

            update_pauligraph(general_data, gadget_index, qubit1, qubit2, pauli1, pauli2)
            gadget_data[qubit1,gadget_index], gadget_data[qubit2,gadget_index] = pauli1, pauli2
            gadget_angles[gadget_index] *= phase


def create_pauligraph_from_tree_graph(tree_graph, pauligraph_degrees, last_edge, gadget_index):
    """Update connection datastructure based on qubit_map provided by networkx steinertree algorithm."""
    num_qubits = pauligraph_degrees.shape[1]
    for j in range(num_qubits):
        if tree_graph.has_node(j):
            pauligraph_degrees[gadget_index,j] = tree_graph.degree[j]
            if tree_graph.degree[j] == 1:
                last_edge[gadget_index,j] = list(tree_graph.edges(j))[0][1] # what is the border of edge node
        else:
            pauligraph_degrees[gadget_index,j] = 0

def check_for_singles(general_data, circ_data, perm_gadgets, last_leg):
    """ check if there are gadgets having only single leg and remove them if so."""
    gadget_data, gadget_angles, removed_gadgets, pauligraph_degrees, last_edge, tree_graph = general_data
    num_qubits, num_gadgets = gadget_data.shape
    removed_gadgets_num = 0

    for i in range(num_gadgets):
        if removed_gadgets[i] == 1:
            continue
        x = 0
        for j in range(num_qubits):
            if gadget_data[j,i] != 0b00:
                x += 1
                qubit = j 
        if x == 1:
            removed_gadgets_num += 1
            remove_single(general_data, circ_data, perm_gadgets, i, qubit)
            last_leg[qubit] += 1
    return removed_gadgets_num


def remove_single(general_data, circ_data, perm_gadgets, gadget_index, qubit):
    """ Remove gadget having single leg indicated by gadget_index."""
    gadget_data, gadget_angles, removed_gadgets, pauligraph_degrees, last_edge, tree_graph = general_data
    qc_out, qc_prop = circ_data
    pauli = gadget_data[qubit,gadget_index]
    removed_gadgets[gadget_index] = 1
    if pauli == 0b01:
        qc_out.h(qubit)
    elif pauli == 0b11:
        qc_out.v(qubit)
    qc_out.rz(gadget_angles[gadget_index], qubit)
    if pauli == 0b01:
        qc_out.h(qubit)
    elif pauli == 0b11:
        qc_out.vdg(qubit)
    perm_gadgets.append(gadget_index)
    gadget_data[qubit,gadget_index] = 0b00


def next_edge_to_remove(gadget_index, general_data, gate_combinations, removed_gadgets_num, random_sel=False):
    """Decide what edge to remove next and what gates to apply"""
    # This version primarily chooses gates which minimizes overall effect to the length of chains. 
    # If there are many options having the same score, it secondarily tries to remove/avoid identity gates from the middle of chains.
    gadget_data, gadget_angles, removed_gadgets, pauligraph_degrees, last_edge, tree_graph = general_data
    num_qubits, num_gadgets = gadget_data.shape

    # find different options to remove qubit (edge)
    edge_options = []
    for q in range(num_qubits):
        if pauligraph_degrees[gadget_index, q] == 1:
            edge_options.append((q,int(last_edge[gadget_index,q])))

    # find possible gates for each edge option
    edge_gates = []
    for qubit0,qubit1 in edge_options:
        if gadget_data[qubit1,gadget_index] == 0b00: #swap needed
            options = gate_combinations[((gadget_data[qubit0,gadget_index],gadget_data[qubit1,gadget_index]),False,False)]
            edge_gates.append(options)
        else:
            options = gate_combinations[((gadget_data[qubit0,gadget_index],gadget_data[qubit1,gadget_index]),True,False)]
            edge_gates.append(options)
    if num_gadgets - removed_gadgets_num == 1: # Only one gadget left, choose any
        return edge_options[0], get_gates(edge_gates[0])[0], (0,0)

    score = np.zeros((len(edge_options),9), dtype=int)
    snode_score = np.zeros((len(edge_options),9), dtype=int)
    for gadget in range(num_gadgets): # For each non-removed gadget, check how different edge+gate combinations would affect it
        if removed_gadgets[gadget]:
            continue

        for e in range(len(edge_options)):
            qubit0,qubit1 = edge_options[e]
            gates0 = edge_gates[e]
            qubit0_degree = pauligraph_degrees[gadget,qubit0]
            qubit1_degree = pauligraph_degrees[gadget,qubit1]
            pauli0 = gadget_data[qubit0,gadget]
            pauli1 = gadget_data[qubit1,gadget]
            legs = (pauli0, pauli1)
            node_gates = 0
            snode_gates = 0
            node_change = 0
            snode_change = 0

            # Pair is in the middle of branch. Avoid I:s
            if qubit0_degree > 1 and qubit1_degree > 1:
                if pauli0 == 0b00 or pauli1 == 0b00:
                    snode_gates = gates0 & gate_combinations[(legs,False,False)]
                    snode_change = 1
                else:
                    snode_gates = gates0 & (gate_combinations[(legs,True,False)] | gate_combinations[(legs,False,True)])
                    snode_change = -1

            # Pair is last pair in a brach in this gadget also. Try turn first qubit to I
            elif qubit0_degree == 1 and qubit1_degree > 1:
                if pauli1 == 0b00:
                    snode_gates = gates0 & gate_combinations[(legs,False,False)]
                    snode_change = +1
                else: #both qubit are non-I
                    node_gates = gates0 & gate_combinations[(legs,True,False)]
                    node_change = 1
                    snode_gates = gates0 & gate_combinations[(legs,False,True)]
                    snode_change = -1

            # Same situation but reversed
            elif qubit0_degree > 1 and qubit1_degree == 1:
                if pauli0 == 0b00:
                    snode_gates = gates0 & gate_combinations[(legs,False,False)]
                    snode_change = +1
                else:
                    snode_gates = gates0 & gate_combinations[(legs,True,False)]
                    snode_change = -1
                    node_gates = gates0 & gate_combinations[(legs,False,True)]
                    node_change = 1

            # Pair touches a branch. Try not to extend branch
            elif qubit1_degree == 0 or qubit0_degree == 0:
                node_gates = gates0 & gate_combinations[(legs,False,False)]
                node_change = -1

            # Pair is last pair in a brach in this gadget. Try turn one qubit to I
            elif qubit0_degree == 1 and qubit1_degree == 1:
                node_gates = gates0 & (gate_combinations[(legs,False,True)] | gate_combinations[(legs,True,False)])
                node_change = 1

            for i in range(9):
                if snode_gates & (1<<i):
                    snode_score[e,i] += snode_change
                if node_gates & (1<<i):
                    score[e,i] += node_change

    option_possibilities = []
    for e in range(len(edge_options)):
        gates = edge_gates[e]
        for i in range(9):
            if (gates & (1<<i)) > 0:
                option_possibilities.append((e, i, int(score[e,i]), int(snode_score[e,i])))
    option_possibilities.sort(key=lambda x: (-x[2], -x[3]))

    if len(option_possibilities) == 0:
        print('ERROR: no edge+gate combinations found')
        print('gadget:', gadget_index)
        print('gadget data:', gadget_data[:,gadget_index])
        print('pauligraph degrees:', pauligraph_degrees[gadget_index,:])
        input()

    edge_gates = option_possibilities[0]
    return edge_options[edge_gates[0]], get_gate(1<<edge_gates[1]), (edge_gates[2],edge_gates[3])


def next_gadget(general_data):
    """ Order non-removed gadgets. Primary sorting is done by number of nodes in steiner tree, secondary sorting by number of steiner nodes."""
    gadget_data, gadget_angles, removed_gadgets, pauligraph_degrees, last_edge, tree_graph = general_data
    num_qubits, num_gadgets = gadget_data.shape
    closest = []
    distance = num_qubits*2
    for i in range(num_gadgets):
        if removed_gadgets[i]:
            continue
        s_nodes, nodes = steiner_nodes(gadget_data, pauligraph_degrees, i)
        dist = nodes + (s_nodes * 2)
        if dist < distance:
            distance = dist
            closest = [i]
        elif dist == distance:
            closest.append(i) 
    return closest

def add_cnot_and_single_qubit_gates(edge, gates, permanent, general_data, circ_data, perm_gadgets, last_leg):
    """apply single qubit gates and CNOT to all non-removed gates."""
    gadget_data, gadget_angles, removed_gadgets, pauligraph_degrees, last_edge, tree_graph = general_data
    num_qubits, num_gadgets = gadget_data.shape

    gate1, gate2 = gates
    qubit1, qubit2 = edge
    if permanent:
        qc_out, qc_prop = circ_data
        for gate, qubit in [(gate1, qubit1), (gate2, qubit2)]:
            if gate == apply_v:
                qc_out.v(qubit)
                qc_prop.append(Vdg(qubit))
            elif gate == apply_s:
                qc_out.s(qubit)
                qc_prop.append(Sdg(qubit))
            elif gate == apply_sv:
                qc_out.s(qubit)
                qc_prop.append(Sdg(qubit))
                qc_out.v(qubit)
                qc_prop.append(Vdg(qubit))
            elif gate == apply_vs:
                qc_out.v(qubit)
                qc_prop.append(Vdg(qubit))
                qc_out.s(qubit)
                qc_prop.append(Sdg(qubit))
            elif gate == apply_svs:
                qc_out.s(qubit)
                qc_prop.append(Sdg(qubit))
                qc_out.v(qubit)
                qc_prop.append(Vdg(qubit))
                qc_out.s(qubit)
                qc_prop.append(Sdg(qubit))
        qc_out.cx(qubit1, qubit2)
        qc_prop.append(CX(qubit1, qubit2))

    # Propagate CNOT gate through all remaining gadgets
    removed_gadgets_num = 0
    for gadget_index in range(num_gadgets):
        if removed_gadgets[gadget_index]:
            continue
        pauli1, pauli2 = gadget_data[qubit1,gadget_index], gadget_data[qubit2,gadget_index]
        if pauli1 == 0b00 and pauli2 == 0b00:
            continue
        phase = 1
        # Apply possible single qubit gates
        pauli1, phase_change = gate1(pauli1)
        phase *= phase_change
        pauli2, phase_change = gate2(pauli2)
        phase *= phase_change
        # Apply CNOT gate
        pauli1, pauli2, phase_change = apply_cnot(pauli1, pauli2)
        phase *= phase_change

        gadget_removed = update_pauligraph(general_data, gadget_index, qubit1, qubit2, pauli1, pauli2)
        gadget_data[qubit1,gadget_index], gadget_data[qubit2,gadget_index] = pauli1, pauli2
        gadget_angles[gadget_index] *= phase

        # Update pauligraph and pauligraph_degrees
        if permanent and gadget_removed:
            if pauli1 == 0b00:
                remove_single(general_data, circ_data, perm_gadgets, gadget_index, qubit2)
                last_leg[qubit2] += 1
            else:
                remove_single(general_data, circ_data, perm_gadgets, gadget_index, qubit1)
                last_leg[qubit1] += 1
            removed_gadgets_num += 1
            gadget_data[qubit1,gadget_index], gadget_data[qubit2,gadget_index] = 0b00, 0b00
#        check_cdconns_integrity(pauligraph, pauligraph_degrees, gadget_data, last_edge)
    return removed_gadgets_num

def update_pauligraph(general_data, gadget_index, qubit1, qubit2, pauli1, pauli2):
    gadget_data, gadget_angles, removed_gadgets, pauligraph_degrees, last_edge, tree_graph = general_data
    num_qubits, num_gadgets = gadget_data.shape

    original_pauli1, original_pauli2 = gadget_data[qubit1,gadget_index], gadget_data[qubit2,gadget_index]
    if pauli1 != 0b00 and original_pauli1 == 0b00: # addition
        if pauligraph_degrees[gadget_index,qubit1] == 0: # outside of chain, negative
            pauligraph_degrees[gadget_index,qubit1] += 1
            pauligraph_degrees[gadget_index,qubit2] += 1
            last_edge[gadget_index,qubit1] = qubit2
        else:                                            # middle of chain, positive
            pass
    if pauli2 != 0b00 and original_pauli2 == 0b00: # addition
        if pauligraph_degrees[gadget_index,qubit2] == 0: # outside of chain, negative
            pauligraph_degrees[gadget_index,qubit1] += 1
            pauligraph_degrees[gadget_index,qubit2] += 1
            last_edge[gadget_index,qubit2] = qubit1
        else:                                            # middle of chain, positive
            pass

    if pauli1 == 0b00 and original_pauli1 != 0b00: # deletion
        if pauligraph_degrees[gadget_index,qubit1] == 1: # end of chain, positive
            pauligraph_degrees[gadget_index,qubit1] -= 1
            pauligraph_degrees[gadget_index,qubit2] -= 1
            if pauligraph_degrees[gadget_index,qubit2] == 1:
                last_edge[gadget_index,qubit2] = find_last_edge(gadget_index, qubit2, pauligraph_degrees, tree_graph)
        else:                                            # middle of chain, negative
            pass
    if pauli2 == 0b00 and original_pauli2 != 0b00: # deletion
        if pauligraph_degrees[gadget_index,qubit2] == 1: # end of chain, positive
            pauligraph_degrees[gadget_index,qubit1] -= 1
            pauligraph_degrees[gadget_index,qubit2] -= 1
            if pauligraph_degrees[gadget_index,qubit1] == 1:
                last_edge[gadget_index,qubit1] = find_last_edge(gadget_index, qubit1, pauligraph_degrees, tree_graph)
        else:                                            # middle of chain, negative
            pass
    return (pauligraph_degrees[gadget_index,qubit1] == 0) and (pauligraph_degrees[gadget_index,qubit2] == 0)


def find_last_edge(gadget_index, qubit, pauligraph_degrees, tree_graph):
    for neighbor in tree_graph.neighbors(qubit):
        if pauligraph_degrees[gadget_index, neighbor] != 0:
            return neighbor
    print('ERROR: could not find last edge')
    print('gadget index:', gadget_index)
    print('qubit:', qubit)
    print('gadget degrees:', pauligraph_degrees[gadget_index])
    input()
    return -1


def topology_tree(gadget_data, tree):
    num_qubits, num_gadgets = gadget_data.shape
    root, tree_children = tree
    edges = []
    for i in range(num_qubits):
        for j in tree_children[i] or []:
            edges.append((i,j))
    G = nx.Graph(edges)
    return G

def mapping_tree(gadget_data, topo, tree, gadget_index):
    root, tree_children = tree
    nodes_removed = []
    edges = []
    remove_nodes_from_leafs(gadget_data, gadget_index, tree_children, root, nodes_removed, edges)
    remove_nodes_from_root(gadget_data, gadget_index, tree_children, root, nodes_removed,edges)
    collect_edges(tree_children, root, nodes_removed, edges)

    G = nx.Graph(edges)

    return G

def collect_edges(tree_children, root, nodes_removed, edges):
    if tree_children[root] is None:
        return
    for node in tree_children[root]:
        if node not in nodes_removed:
            if root not in nodes_removed:
                edges.append((root, node))
        collect_edges(tree_children, node, nodes_removed, edges)

def remove_nodes_from_leafs(gadget_data, gadget_index, tree_children, root, nodes_removed, edges):
    if tree_children[root] is None:
        if gadget_data[root,gadget_index] == 0b00:
            nodes_removed.append(root)
            return True
        else:
            return False
    non_removable_branches = 0
    for node in tree_children[root]:
        removable = remove_nodes_from_leafs(gadget_data, gadget_index, tree_children, node, nodes_removed, edges)
        if not removable:
            non_removable_branches += 1
    if non_removable_branches == 0 and gadget_data[root,gadget_index] == 0b00:
        nodes_removed.append(root)
        return True 
    return False

def remove_nodes_from_root(gadget_data, gadget_index, tree_children, root, nodes_removed, edges):
    if tree_children[root] is None:
        return
    if gadget_data[root,gadget_index] != 0b00:
        return
    branches_left = 0
    branch = -1
    for node in tree_children[root]:
        if node not in nodes_removed:
            branches_left += 1
            branch = node
    if branches_left == 1:
        nodes_removed.append(root)
        remove_nodes_from_root(gadget_data, gadget_index, tree_children, branch, nodes_removed, edges)

def steiner_nodes(gadget_data, pauligraph_degrees, gadget_index):
    """ Defines number of steiner nodes and regular nodes of steiner tree (pauligraph data)"""
    num_qubits = gadget_data.shape[0]
    nodes = 0
    steiner_nodes = 0
    for i in range(num_qubits):
        if gadget_data[i, gadget_index] != 0b00:
            nodes += 1
        elif pauligraph_degrees[gadget_index, i] > 0:
            steiner_nodes += 1
    return steiner_nodes, nodes


def apply_v(p, reverse=False): # z,z xor x
    phase = 1
    if not reverse and p == 0b10:
        phase = -1
    if reverse and p == 0b11:
        phase = -1
    return (0b10 & p) | ((0b01 & p) ^ (p >> 1)), phase

def apply_s(p, reverse=False): # z xor x, x
    phase = 1
    if not reverse and p == 0b11:
        phase = -1
    if reverse and p == 0b01:
        phase = -1
    return 0b01 & p | ((0b10 & p) ^ ((p & 0b01) << 1)), phase

def apply_I(p, reverse=False):
    return p, 1

def apply_sv(p, reverse=False):
    if reverse:
        pauli,phase1 = apply_v(p, reverse=True)
        pauli,phase2 = apply_s(pauli, reverse=True)
        return pauli, phase1*phase2
    pauli,phase1 = apply_s(p)
    pauli,phase2 = apply_v(pauli)
    return pauli, phase1*phase2

def apply_vs(p, reverse=False):
    if reverse:
        pauli,phase1 = apply_s(p, reverse=True)
        pauli,phase2 = apply_v(pauli, reverse=True)
        return pauli, phase1*phase2
    pauli,phase1 = apply_v(p)
    pauli,phase2 = apply_s(pauli)
    return pauli, phase1*phase2

def apply_svs(p, reverse=False):
    pauli,phase1 = apply_s(p, reverse=reverse)
    pauli,phase2 = apply_v(pauli, reverse=reverse)
    pauli,phase3 = apply_s(pauli, reverse=reverse)
    return pauli, phase1*phase2*phase3

def apply_cnot(p1, p2):
    phase = 1
    if (p1 == 0b01 and p2 == 0b10) or (p1 == 0b11 and p2 == 0b11):
        phase = -1
    pauli1 = (p1 & 0b01) | ((p1 ^ p2) & 0b10)
    pauli2 = ((p1 ^ p2) & 0b01) | (p2 & 0b10)
    return pauli1, pauli2, phase

def apply_single_and_cnot(gates, p1, p2):
    phase = 1
    first_gate, second_gate = gates

    pauli1, phase_change = first_gate(p1)
    phase *= phase_change

    pauli2, phase_change = second_gate(p2)
    phase *= phase_change

    pauli1, pauli2, phase_change = apply_cnot(pauli1, pauli2)
    phase *= phase_change
    return pauli1, pauli2, phase

def possible_gates(paulis,target0, target1):
    """Return possible gates for given pair of paulis and targets to have or have not I.
    Possible gates include possible single qubit gate for qubit1, single qubit gate for qubit2 and cnot direction.
    :params paulis: tuple of two chars, e.g. (0b01, 0b11) or (0b00, 0b10)
    :params target0: True if qubit0 should have I, False if it should not have I
    :params target1: True if qubit1 should have I, False if it should not have I
    :returns: 9-bit integer coding possible gate combinations."""
    # CHANGED SO THAT IT RETURNS ONLY 9 POSSIBLE COMBINATIONS, NOT DISTINGUISHING CNOT DIRECTION
    # Possibilities for first: I, V SV
    # Possibilities for second: S, V, VS
    # There is legacy code because of previous 72 possibilities
    gates_none = np.zeros((2,6,6), dtype=object)
    convert_to_X = {0b01: [0,1], 0b11: [2,5], 0b10: [3,4]}
    convert_to_Y = {0b01: [2,4], 0b11: [0,3], 0b10: [1,5]}
    convert_to_Z = {0b01: [3,5], 0b11: [1,4], 0b10: [0,2]}

    if target0 and target1 and paulis[0] == 0b00 and paulis[1] == 0b00:   # current and target is II
        paulis_options = np.ones((2,6,6), dtype=object)                 # all gates
    elif paulis[0] == 0b00 and paulis[1] == 0b00:                         # current is ??, target is II
        paulis_options = gates_none
    elif target0 and target1:  
        paulis_options = gates_none
    elif target0 and not target1 and paulis[0] != 0b00 and paulis[1] == 0b00: # Swap needed, not possible
        paulis_options = gates_none
    elif not target0 and target1 and paulis[0] == 0b00 and paulis[1] != 0b00: # Swap needed, not possible
        paulis_options = gates_none
    elif target0 and not target1:
        paulis_options = gates_none
        if paulis[0] == 0b00:
            for i in range(6):
                paulis_options[np.ix_([0],[i],convert_to_X[paulis[1]])] = 1
                paulis_options[np.ix_([1],[i],convert_to_Z[paulis[1]])] = 1 #1q-h
        else:
            paulis_options[np.ix_([1], convert_to_X[paulis[0]], convert_to_X[paulis[1]])] = 1 # XX
            paulis_options[np.ix_([1], convert_to_X[paulis[0]], convert_to_Y[paulis[1]])] = 1 # XY
            paulis_options[np.ix_([0], convert_to_Z[paulis[0]], convert_to_Z[paulis[1]])] = 1 # ZZ
            paulis_options[np.ix_([0], convert_to_Z[paulis[0]], convert_to_Y[paulis[1]])] = 1 # YZ
    elif not target0 and target1:
        paulis_options = gates_none
        if paulis[1] == 0b00:
            for i in range(6):
                paulis_options[np.ix_([1],convert_to_X[paulis[0]],[i])] = 1
                paulis_options[np.ix_([0],convert_to_Z[paulis[0]],[i])] = 1 #1q-h
        else:
            paulis_options[np.ix_([0], convert_to_X[paulis[0]], convert_to_X[paulis[1]])] = 1 # XX
            paulis_options[np.ix_([0], convert_to_Y[paulis[0]], convert_to_X[paulis[1]])] = 1 # YX
            paulis_options[np.ix_([1], convert_to_Z[paulis[0]], convert_to_Z[paulis[1]])] = 1 # ZZ
            paulis_options[np.ix_([1], convert_to_Y[paulis[0]], convert_to_Z[paulis[1]])] = 1 # YZ

    elif not target0 and not target1:
        paulis_options = gates_none.copy()
        if paulis[0] != 0b00 and paulis[1] != 0b00:   # XY<->YZ, XZ<->YY, ZX<->ZX:   XY,XZ,   ZX,   YZ, YY
            paulis_options[np.ix_([0], convert_to_X[paulis[0]], convert_to_Y[paulis[1]])] = 1
            paulis_options[np.ix_([0], convert_to_X[paulis[0]], convert_to_Z[paulis[1]])] = 1
            paulis_options[np.ix_([0], convert_to_Z[paulis[0]], convert_to_X[paulis[1]])] = 1
            paulis_options[np.ix_([0], convert_to_Y[paulis[0]], convert_to_Z[paulis[1]])] = 1
            paulis_options[np.ix_([0], convert_to_Y[paulis[0]], convert_to_Y[paulis[1]])] = 1

            paulis_options[np.ix_([1], convert_to_Y[paulis[0]], convert_to_X[paulis[1]])] = 1
            paulis_options[np.ix_([1], convert_to_Z[paulis[0]], convert_to_X[paulis[1]])] = 1
            paulis_options[np.ix_([1], convert_to_X[paulis[0]], convert_to_Z[paulis[1]])] = 1
            paulis_options[np.ix_([1], convert_to_Z[paulis[0]], convert_to_Y[paulis[1]])] = 1
            paulis_options[np.ix_([1], convert_to_Y[paulis[0]], convert_to_Y[paulis[1]])] = 1
        if paulis[0] == 0b00 and paulis[1] != 0b00:   # XI, YI, IY, IZ
            for i in range(6):
                paulis_options[np.ix_([0],[i],convert_to_Y[paulis[1]])] = 1
                paulis_options[np.ix_([0],[i],convert_to_Z[paulis[1]])] = 1
                paulis_options[np.ix_([1],[i],convert_to_X[paulis[1]])] = 1
                paulis_options[np.ix_([1],[i],convert_to_Y[paulis[1]])] = 1
        if paulis[0] != 0b00 and paulis[1] == 0b00:
            for i in range(6):
                paulis_options[np.ix_([1],convert_to_Y[paulis[0]],[i])] = 1
                paulis_options[np.ix_([1],convert_to_Z[paulis[0]],[i])] = 1
                paulis_options[np.ix_([0],convert_to_X[paulis[0]],[i])] = 1
                paulis_options[np.ix_([0],convert_to_Y[paulis[0]],[i])] = 1
    else:
        print('XXXX Should not happen')

    options = []
    for j, first_qubit in enumerate([apply_I, apply_v, apply_s, apply_svs, apply_vs, apply_sv]):
        for k, second_qubit in enumerate([apply_I, apply_v, apply_s, apply_svs, apply_vs, apply_sv]):
            if paulis_options[0,j,k] == 1:
                options.append((first_qubit, second_qubit))

    # Adapt the new 9 possibilities coding
    options2 = 0 
    for j, first_qubit in enumerate([apply_I, apply_v, apply_sv]):
        for k, second_qubit in enumerate([apply_v, apply_s, apply_vs]):
            if (first_qubit, second_qubit) in options:
                options2 |= 1 << (j*3 + k)

    return options2

def get_gates(gate_set):
    """Return gate combinations for given gate set indicated by 9-bit integer.
    :params gate_set: 9-bit integer coding possible gate combinations.
    :returns: list of tuples (first_qubit_gate, second_qubit_gate, cnot direction).
    """
    # CHANGED SO THAT IT RETURNS ONLY 9 POSSIBLE COMBINATIONS, NOT DISTINGUISHING CNOT DIRECTION
    gates = []
    for j, first_qubit in enumerate([apply_I, apply_v, apply_sv]):
        for k, second_qubit in enumerate([apply_v, apply_s, apply_vs]):
            gate = 1 << (j*3 + k)
            if gate & gate_set > 0:
                gates.append((first_qubit, second_qubit))
    return gates

def get_gate(gate_set):
    first = [apply_I, apply_v, apply_sv]
    second = [apply_v, apply_s, apply_vs]
    i = -1
    while gate_set > 0:
        gate_set >>= 1
        i +=1
    return first[i//3], second[i%3]

#
#
#
# Functions for testing and debugging
#

def check_circuit_equivalence(pp, circ_out, gadget_perm, perm):
    """Check if the circuit is equivalent to the circuit produced by cbnot-ladders from PauliPolynomial."""
    pp2 = PauliPolynomial(pp.num_qubits)
    pp2.pauli_gadgets = [pp[i].copy() for i in gadget_perm]
    circ = circ_out.to_qiskit()
    pp_circ = pp2.to_qiskit()
    return verify_equality(pp_circ,circ)


def print_sorted_gd(gadget_data, order=None):
    num_qubits, num_gadgets = gadget_data.shape
    if order is None:
        order = [i for i in range(num_gadgets)]
    else:
        order = [order[i] for i in range(len(order))]
    print(' ', end=' ')
    for i in order:
        print(int(i / 10), end=' ')
    print('\n ', end=' ')
    for i in order:
        print(i % 10, end=' ')
    print('')
    for i in range(num_qubits):
        print(i % 10, end=' ')
        for j in range(len(order)):
            if gadget_data[i,order[j]] == 0b01:
                char = 'X'
            elif gadget_data[i,order[j]] == 0b10:
                char = 'Z'
            elif gadget_data[i,order[j]] == 0b11:
                char = 'Y'
            elif gadget_data[i,order[j]] == 0b00:
                char = ' '
            print(char, end=' ')
        print('')
    print('')
