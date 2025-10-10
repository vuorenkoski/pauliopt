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

def pauli_polynomial_dynamic_ordering_tree(pp: PauliPolynomial, topo: Topology, tree, print_order=None, debug=False, random_sel=False):
    num_qubits = pp.num_qubits
    num_gadgets = len(pp.pauli_gadgets)
    removed_gadgets_num = 0
    gadget_angles = []
    removed_gadgets = np.zeros((num_gadgets), dtype=np.int8)
    qc_out = Circuit(num_qubits)
    qc_prop = []
    perm_gadgets = []

    # Create datastructures
    gadget_data = np.zeros((num_qubits,num_gadgets), dtype=np.int8) # Dynamic matrix representing paulis
    pauligraph = np.zeros((num_gadgets,num_qubits,num_qubits), dtype=np.int8) # Dynamic matrix represnting steiner trees
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
        if tree is None:
            tree_graph = steiner_tree(gadget_data, topo, i)
        else:
            tree_graph = mapping_tree(gadget_data, topo, tree, i)
        create_pauligraph_from_tree_graph(tree_graph, pauligraph, pauligraph_degrees, last_edge, i)
    gate_combinations = {} # Immutable dictionary of possible gates for each pair of paulis and targets
    for p1 in [0b00, 0b01, 0b10, 0b11]:
        for p2 in [0b00, 0b01, 0b10, 0b11]:
            for target0 in [False, True]: # Is target to have I or not after gates in qubit0
                for target1 in [False, True]: # Is target to have I or not after gates in qubit1
                    options = possible_gates((p1, p2), target0, target1)
                    gate_combinations[((p1, p2), target0, target1)] = options
    general_data = (gadget_data, gadget_angles, removed_gadgets, pauligraph, pauligraph_degrees, last_edge)
    circ_data = (qc_out, qc_prop)

    if tree is not None:
        node_levels = dict()
        root, tree_children = tree
        node_levels_from_tree(tree_children, root, node_levels)
    else:
        node_levels = None

    debug and print('---------------------Initial gadget data')
    debug and print_sorted_gd(gadget_data, order=print_order)

    # Main loop going through gadgets starting from smallest
    removed_gadgets_num += check_for_singles(general_data, circ_data, perm_gadgets)
    debug and print_sorted_gd(gadget_data, order=print_order)
    while removed_gadgets_num < num_gadgets:
        # Randomness 1: there are for example many gadgets having same size and no steiner nodes, how to arrange them?
#        if tree is None:
        next = next_gadget(general_data, gate_combinations)
#        else:
#            next = next_gadget2(general_data, node_levels)
        debug and print('-------Next gadget:', next)
        num_legs = 0
        for j in range(num_qubits):
            if gadget_data[j,next] != 0b00:
                num_legs += 1
        if num_legs == 0:
            print('ERROR: qubit map has no nodes')
            print('gadget:', next)
            print_sorted_gd(gadget_data)
            print(perm_gadgets)
            input()
        
        # Loop going through qubits in gadget, removing them one by one
        while num_legs > 1:
            # randomness 2: if there are two or more edge+gate combinations having similar match, which one to choose?
            edge, gates, score = next_edge_to_remove(next, general_data, gate_combinations, removed_gadgets_num, random_sel=random_sel)
            debug and print('-------Next edge:', edge, 'gates:', gates, 'gadget', next)
            rgadgets = add_cnot_and_single_qubit_gates(edge, gates, general_data, circ_data, perm_gadgets, topo)
            removed_gadgets_num += rgadgets
            if gadget_data[edge[0], next] == 0b00:
                num_legs -= 1
            debug and print_sorted_gd(gadget_data, order=print_order)
            debug and input()

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
    return circ_out, perm_gadgets, permutation, {'pre-cx': pre_cx, 'density': density}


def create_pauligraph_from_tree_graph(tree_graph, pauligraph, pauligraph_degrees, last_edge, gadget_index):
    """Update connection datastructure based on qubit_map provided by networkx steinertree algorithm."""
    num_qubits = pauligraph.shape[1]
    for j in range(num_qubits):
        if tree_graph.has_node(j):
            pauligraph_degrees[gadget_index,j] = tree_graph.degree[j]
            if tree_graph.degree[j] == 1:
                last_edge[gadget_index,j] = list(tree_graph.edges(j))[0][1] # what is the border of edge node
        else:
            pauligraph_degrees[gadget_index,j] = 0
    for j in range(num_qubits):
        for k in range(num_qubits):
            pauligraph[gadget_index,j,k] = 0
    for edges in tree_graph.edges():
        pauligraph[gadget_index, edges[0], edges[1]] = 1
        pauligraph[gadget_index, edges[1], edges[0]] = 1


def check_for_singles(general_data, circ_data, perm_gadgets):
    """ check if there are gadgets having only single leg and remove them if so."""
    gadget_data, gadget_angles, removed_gadgets, pauligraph, pauligraph_degrees, last_edge = general_data
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
    return removed_gadgets_num


def remove_single(general_data, circ_data, perm_gadgets, gadget_index, qubit):
    """ Remove gadget having single leg indicated by gadget_index."""
    gadget_data, gadget_angles, removed_gadgets, pauligraph, pauligraph_degrees, last_edge = general_data
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
    gadget_data, gadget_angles, removed_gadgets, pauligraph, pauligraph_degrees, last_edge = general_data
    num_qubits, num_gadgets = gadget_data.shape

    # find different option to remove qubit (edge)
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
        return edge_options[0], get_gates(edge_gates[0])[0], 0

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
            legs = (gadget_data[qubit0,gadget], gadget_data[qubit1,gadget])
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
                option_possibilities.append((e, i, score[e,i], snode_score[e,i]))
    option_possibilities.sort(key=lambda x: (-x[2], -x[3]))

    if len(option_possibilities) == 0:
        print('ERROR: no edge+gate combinations found')
        print('gadget:', gadget_index)
        print('gadget data:', gadget_data[:,gadget_index])
        print('pauligraph degrees:', pauligraph_degrees[gadget_index,:])
        print('pauligraphs:', pauligraph[gadget_index,:,:])
        input()

    if random_sel:
        max_score = option_possibilities[0][2]
        max_I_score = option_possibilities[0][3]
        best_options = []
        for x in option_possibilities:
            if x[2] == max_score and x[3] == max_I_score:
                best_options.append((x[0], x[1]))
        edge_gates = random.choice(best_options)
    else:
        edge_gates = option_possibilities[0]
    return edge_options[edge_gates[0]], get_gate(1<<edge_gates[1]), edge_gates[2]


def next_gadget(general_data, gate_combinations):
    """ Order non-removed gadgets. Primary sorting is done by number of nodes in steiner tree, secondary sorting by number of steiner nodes."""
    gadget_data, gadget_angles, removed_gadgets, pauligraph, pauligraph_degrees, last_edge = general_data
    num_qubits, num_gadgets = gadget_data.shape
    closest = -1
    distance = num_qubits*2
    for i in range(num_gadgets):
        if removed_gadgets[i]:
            continue
        s_nodes, nodes = steiner_nodes(gadget_data, pauligraph_degrees, i)
        dist = nodes + (s_nodes * 2)
        if dist < distance:
            distance = dist
            closest = i

    return closest

def next_gadget_efficient_removal(general_data, gate_combinations):
    """ Check all closest gadgets, which one most shortens gadgets. Takes more time, littlebbit more efficient"""
    gadget_data, gadget_angles, removed_gadgets, pauligraph, pauligraph_degrees, last_edge = general_data
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

    max_score = - num_gadgets*2
    max_gadget = -1
    for gadget in closest:
        edge, gates, score = next_edge_to_remove(gadget, general_data, gate_combinations, 0)

        if score > max_score:
            max_score = score
            max_gadget = gadget
    if max_gadget == -1:
        print('ERROR: no gadget found')
        print('closest:', closest)
        print_sorted_gd(gadget_data)
        input()
    return max_gadget   

def next_gadget2(general_data, node_levels):
    """ Order non-removed gadgets. Primary sorting is done by number of nodes in steiner tree, secondary sorting by number of steiner nodes."""
    gadget_data, gadget_angles, removed_gadgets, pauligraph, pauligraph_degrees, last_edge = general_data
    num_qubits, num_gadgets = gadget_data.shape
    closest = -1
    min_level = num_qubits
    distance = -1
    for i in range(num_gadgets):
        if removed_gadgets[i]:
            continue
        s_nodes, nodes = steiner_nodes(gadget_data, pauligraph_degrees, i)
#        level = get_gadget_level(node_levels, pauligraph_degrees, i)
        dist = nodes + (s_nodes * 2)
        if distance == -1 or dist < distance:
            distance = dist
            closest = i
#        elif dist == distance and level < min_level:
#            min_level = level
#            closest = i
    return closest

def get_gadget_level(node_levels, pauligraph_degrees, gadget_index):
    max_level = 0
    for i in range(pauligraph_degrees.shape[1]):
        if pauligraph_degrees[gadget_index,i] > 0:
            max_level = max(max_level, node_levels[i])
    return max_level

def node_levels_from_tree(tree_children, node, node_levels):
    if tree_children[node] is None:
        node_levels[node] = 0
        return 0
    max_level = 0
    for n in tree_children[node]:
        level = node_levels_from_tree(tree_children, n, node_levels)
        if level>max_level:
            max_level = level
    node_levels[node] = max_level + 1
    return max_level + 1


def add_cnot_and_single_qubit_gates(edge, gates, general_data, circ_data, perm_gadgets, topo):
    """apply single qubit gates and CNOT to all non-removed gates."""
    gadget_data, gadget_angles, removed_gadgets, pauligraph, pauligraph_degrees, last_edge = general_data
    qc_out, qc_prop = circ_data
    num_qubits, num_gadgets = gadget_data.shape

    first_qubit, second_qubit, cnot_reversed = gates
    if cnot_reversed:
        qubit2,qubit1 = edge
        gate2 = first_qubit
        gate1 = second_qubit
    else:
        qubit1,qubit2 = edge
        gate1 = first_qubit
        gate2 = second_qubit
    for gate, qubit in [(gate1, qubit1), (gate2, qubit2)]:
        if gate == apply_vdg:
            qc_out.v(qubit)
            qc_prop.append(Vdg(qubit))
        elif gate == apply_sdg:
            qc_out.s(qubit)
            qc_prop.append(Sdg(qubit))
        elif gate == apply_vs:
            qc_out.s(qubit)
            qc_prop.append(Sdg(qubit))
            qc_out.v(qubit)
            qc_prop.append(Vdg(qubit))
        elif gate == apply_sv:
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
    removed_connections = 0
    for gadget_index in range(num_gadgets):
        if removed_gadgets[gadget_index]:
            continue
        pauli1, pauli2 = gadget_data[qubit1,gadget_index], gadget_data[qubit2,gadget_index]
        if pauli1 == 0b00 and pauli2 == 0b00:
            continue
        phase = 1
        # Apply possible single qubit gates
        if gate1 is not None:
            pauli1, phase_change = gate1(pauli1)
            phase *= phase_change
        if gate2 is not None:
            pauli2, phase_change = gate2(pauli2)
            phase *= phase_change
        # Apply CNOT gate
        pauli1, pauli2, phase_change = apply_cnot(pauli1, pauli2)
        phase *= phase_change
        gadget_data[qubit1,gadget_index], gadget_data[qubit2,gadget_index] = pauli1, pauli2
        gadget_angles[gadget_index] *= phase

        
        # Update pauligraph and pauligraph_degrees
        gadget_removed, rconnections = update_pauligraph(general_data, gadget_index, qubit1, qubit2, pauli1, pauli2)
        removed_connections += rconnections
        if gadget_removed:
            if pauli1 == 0b00:
                remove_single(general_data, circ_data, perm_gadgets, gadget_index, qubit2)
            else:
                remove_single(general_data, circ_data, perm_gadgets, gadget_index, qubit1)
            removed_gadgets_num += 1
            gadget_data[qubit1,gadget_index], gadget_data[qubit2,gadget_index] = 0b00, 0b00
#        check_cdconns_integrity(pauligraph, pauligraph_degrees, gadget_data, last_edge)
    return removed_gadgets_num


def update_pauligraph(general_data, gadget_index, qubit1, qubit2, pauli1, pauli2):
    """ Update pauligraph and pauligraph_degrees for gadget. Loop through all edge pairs which 
    are in the edge of branch. Check is there changes for those."""
    gadget_data, gadget_angles, removed_gadgets, pauligraph, pauligraph_degrees, last_edge = general_data
    num_qubits, num_gadgets = gadget_data.shape
    gadget_removed = False
    removed_connections = 0

    # Loop trough all edge pairs of of gadget which are end of branch
    for q1 in range(num_qubits):
        if (q1!=qubit1 and q1!=qubit2) or (pauligraph_degrees[gadget_index,q1] != 1):
            continue
        q2 = last_edge[gadget_index,q1]

        # There are only two neighbouring qubits left which matches for cnot
        if pauligraph_degrees[gadget_index,q2] == 1 and ((q1 == qubit1 and q2 == qubit2) or (q1 == qubit2 and q2 == qubit1)):
            if pauli1 == 0b00 or pauli2 == 0b00:
                remove_connection(general_data, gadget_index, qubit1, qubit2)
                removed_connections += 1
                gadget_removed = True
                break

        # this gadget has last edge matching for cnot
        elif q1 == qubit1 and q2 == qubit2: 
            if pauli1 == 0b00: # Chain shortens
                remove_connection(general_data, gadget_index, qubit1, qubit2)
                removed_connections += 1
                last_edge[gadget_index,qubit1] = -1
                for i in range(num_qubits):
                    if pauligraph[gadget_index,qubit2,i] == 1:
                        last_edge[gadget_index,qubit2] = i
                        break

        # Same but reversed
        elif q1 == qubit2 and q2 == qubit1: 
            if pauli2 == 0b00: # Chain shortens
                remove_connection(general_data, gadget_index, qubit1, qubit2)
                removed_connections += 1
                last_edge[gadget_index,qubit2] = -1
                for i in range(num_qubits):
                    if pauligraph[gadget_index,qubit1,i] == 1:
                        last_edge[gadget_index,qubit1] = i
                        break

        # edge is not and last edge in this gadget, but qubit1 is the last qubit in the branch
        elif q1 == qubit1:
            if pauli1 == 0b00: # chain shortens
                remove_connection(general_data, gadget_index, q1, q2)
                removed_connections += 1
                last_edge[gadget_index,q1] = -1
                gadget_removed = follow_chain_until_not_I(general_data, gadget_index, q2, gadget_data[q2,gadget_index])
            elif pauli2 != 0b00 and pauligraph_degrees[gadget_index,qubit2] == 0: # Branch extends towards qubit2
                add_connection(general_data, gadget_index, q1, qubit2)
                last_edge[gadget_index,qubit2] = q1
                last_edge[gadget_index,q1] = -1

        # Edge is not the last edge in this gadget but qubit2 is the last qubit the in branch
        elif q1 == qubit2:
            if pauli2 == 0b00: # Branch shortens
                remove_connection(general_data, gadget_index, q1, q2)
                removed_connections += 1
                last_edge[gadget_index,q1] = -1
                gadget_removed = follow_chain_until_not_I(general_data, gadget_index, q2, gadget_data[q2,gadget_index])
            elif pauli1 != 0b00 and pauligraph_degrees[gadget_index,qubit1] == 0: # Branch extends
                add_connection(general_data, gadget_index, q1, qubit1)
                last_edge[gadget_index,qubit1] = q1
                last_edge[gadget_index,q1] = -1

    # If one of the qubits is in the midddle of chain
    # and other qubit is not in any branch and turns from I to pauli: new branch
    if pauligraph_degrees[gadget_index,qubit1] == 0 and pauligraph_degrees[gadget_index,qubit2] > 1 and pauli1 != 0b00: # New branch
        add_connection(general_data, gadget_index, qubit1, qubit2)
        last_edge[gadget_index,qubit1] = qubit2
    elif pauligraph_degrees[gadget_index,qubit2] == 0 and pauligraph_degrees[gadget_index,qubit1] > 1 and pauli2 != 0b00: # New branch
        add_connection(general_data, gadget_index, qubit1, qubit2)
        last_edge[gadget_index,qubit2] = qubit1
    return gadget_removed, removed_connections


def follow_chain_until_not_I(general_data, gadget_index, qubit, pauli):
    """Recursive functions following chain of I:s until next qubit is not I. Return True if gadget was removed."""
    gadget_data, gadget_angles, removed_gadgets, pauligraph, pauligraph_degrees, last_edge = general_data
    num_qubits = pauligraph.shape[1]
    if pauligraph_degrees[gadget_index,qubit] > 1:
        return False
    
    qubit2 = None
    j = 0
    for i in range(num_qubits):
        if pauligraph[gadget_index,qubit,i] == 1:
            qubit2 = i
            j += 1
            if j > 1:
                print('ERROR: follow_chain_until_not_I: more than one qubit found for gadget', gadget_index, 'qubit',qubit)
                input()
    if qubit2 is None:
        print('ERROR: follow_chain_until_not_I: no qubit found for gadget', gadget_index, 'qubit',qubit)
        input()

    # chain of I does not continue, this is now the last edge
    if pauli != 0b00:
        last_edge[gadget_index,qubit] = qubit2
        return False

    # pauli is I, we must remove connection
    remove_connection(general_data, gadget_index, qubit, qubit2)

    # If next qubit2 has degree 0 after removing connection, it means that this the only leg left
    if pauligraph_degrees[gadget_index,qubit2] == 0:
        return True
    else: #chain continues beyond qubit2
        return follow_chain_until_not_I(general_data, gadget_index, qubit2, gadget_data[qubit2,gadget_index])


def add_connection(general_data, gadget_index, qubit1, qubit2):
    """Add connection between two qubits in a gadget"""
    gadget_data, gadget_angles, removed_gadgets, pauligraph, pauligraph_degrees, last_edge = general_data
    pauligraph[gadget_index,qubit1,qubit2] = 1
    pauligraph[gadget_index,qubit2,qubit1] = 1
    pauligraph_degrees[gadget_index,qubit1] += 1
    pauligraph_degrees[gadget_index,qubit2] += 1


def remove_connection(general_data, gadget_index, qubit1, qubit2):
    """Add connection between two qubits in a gadget"""
    gadget_data, gadget_angles, removed_gadgets, pauligraph, pauligraph_degrees, last_edge = general_data
    pauligraph[gadget_index,qubit1,qubit2] = 0
    pauligraph[gadget_index,qubit2,qubit1] = 0
    pauligraph_degrees[gadget_index,qubit1] -= 1
    pauligraph_degrees[gadget_index,qubit2] -= 1


def steiner_tree(gadget_data,topo, gadget_index):
    """ Uses NetworkX Steinertree algorithm to make steinertree from gadget."""
    num_qubits, num_gadgets = gadget_data.shape
    nodes = []
    for i in range(num_qubits):
        if gadget_data[i,gadget_index] != 0b00:
            nodes.append(i)
    steiner_stree = nx.algorithms.approximation.steinertree.steiner_tree(topo.to_nx, nodes)
    return nx.Graph(steiner_stree)

def mapping_tree(gadget_data, topo, tree, gadget_index):
    num_qubits, num_gadgets = gadget_data.shape
    root, tree_children = tree
    nodes_removed = []
    edges = []
    remove_nodes_from_leafs(gadget_data, gadget_index, tree_children, root, nodes_removed, edges)
    remove_nodes_from_root(gadget_data, gadget_index, tree_children, root, nodes_removed,edges)
    collect_edges(tree_children, root, nodes_removed, edges)

    G = nx.Graph(edges)
#        print(G.nodes, G.edges, gadget_index)
#        print(gadget_data[:,gadget_index])
#        print('nodes:', G.nodes)
#        print('edges:', G.edges)
#        print('edges:', edges)
#        print('removed:', nodes_removed)

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


def apply_vdg(p):
    phase = 1
    if p == 0b10:
        phase = -1
    return (0b10 & p) | ((0b01 & p) ^ (p >> 1)), phase

def apply_sdg(p):
    phase = 1
    if p == 0b11:
        phase = -1
    return 0b01 & p | ((0b10 & p) ^ ((p & 0b01) << 1)), phase

def apply_I(p):
    return p, 1

def apply_vs(p):
    pauli,phase1 = apply_sdg(p)
    pauli,phase2 = apply_vdg(pauli)
    return pauli, phase1*phase2

def apply_sv(p):
    pauli,phase1 = apply_vdg(p)
    pauli,phase2 = apply_sdg(pauli)
    return pauli, phase1*phase2

def apply_svs(p):
    pauli,phase1 = apply_sdg(p)
    pauli,phase2 = apply_vdg(pauli)
    pauli,phase3 = apply_sdg(pauli)
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
    first_gate, second_gate, cnot_reversed = gates

    pauli1, phase_change = first_gate(p1)
    phase *= phase_change

    pauli2, phase_change = second_gate(p2)
    phase *= phase_change

    if cnot_reversed:
        pauli2, pauli1, phase_change = apply_cnot(pauli2, pauli1)
    else:
        pauli1, pauli2, phase_change = apply_cnot(pauli1, pauli2)
    phase *= phase_change
    return pauli1, pauli2, phase

def possible_gates(paulis,target0, target1):
    """Return possible gates for given pair of paulis and targets to have or have not I.
    Possible gates include possible single qubit gate for qubit1, single qubit gate for qubit2 and cnot direction.
    :params paulis: tuple of two chars, e.g. (0b01, 0b11) or (0b00, 0b10)
    :params target0: True if qubit0 should have I, False if it should not have I
    :params target1: True if qubit1 should have I, False if it should not have I
    :returns: 72-bit integer coding possible gate combinations."""
    # CHANGED SO THAT IT RETURNS ONLY 36 POSSIBLE COMBINATIONS, NOT DISTINGUISHING CNOT DIRECTION
    # coding I, V, S, SVS(H), SV, VS
    # SV X->Y, Y->Z, Z->X: first propagate S, then H
    # VS X->Z, Z->Y, Y->X 
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
#    for i, cnot_reversed in enumerate([False, True]):
    for j, first_qubit in enumerate([apply_I, apply_vdg, apply_sdg, apply_svs, apply_sv, apply_vs]):
        for k, second_qubit in enumerate([apply_I, apply_vdg, apply_sdg, apply_svs, apply_sv, apply_vs]):
            if paulis_options[0,j,k] == 1:
                options.append((first_qubit, second_qubit, False))

    options2 = 0 
    for j, first_qubit in enumerate([apply_I, apply_vdg, apply_vs]):
        for k, second_qubit in enumerate([apply_vdg, apply_sdg, apply_sv]):
            if (first_qubit, second_qubit, False) in options:
                options2 |= 1 << (j*3 + k)

    return options2

def get_gates(gate_set):
    """Return gate combinations for given gate set indicated by 72-bit integer.
    :params gate_set: 72-bit integer coding possible gate combinations.
    :returns: list of tuples (first_qubit_gate, second_qubit_gate, cnot direction).
    """
    # CHANGED SO THAT IT RETURNS ONLY 9 POSSIBLE COMBINATIONS, NOT DISTINGUISHING CNOT DIRECTION
    gates = []
    for j, first_qubit in enumerate([apply_I, apply_vdg, apply_vs]):
        for k, second_qubit in enumerate([apply_vdg, apply_sdg, apply_sv]):
            gate = 1 << (j*3 + k)
            if gate & gate_set > 0:
                gates.append((first_qubit, second_qubit, False))
    return gates

def get_gate(gate_set):
    first = [apply_I, apply_vdg, apply_vs]
    second = [apply_vdg, apply_sdg, apply_sv]
    i = -1
    while gate_set > 0:
        gate_set >>= 1
        i +=1
    return first[i//3], second[i%3], False

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


def test_possible_gates():
    """Testing function for possible_gates"""
    print('Test possible_gates function')
    ok = True
    for p1 in [0b01, 0b11, 0b10, 0b00]:
        for p2 in [0b01, 0b11, 0b10, 0b00]:
            for target0 in [False, True]:
                for target1 in [False, True]:
                    options = possible_gates((p1, p2), target0, target1)
                    options_test = []
                    for cnot_reversed in [False]:
                        for first_gate in [apply_I, apply_vdg, apply_sdg, apply_svs, apply_sv, apply_vs]:
                            for second_gate in [apply_I, apply_vdg, apply_sdg, apply_svs, apply_sv, apply_vs]:
                                if first_gate is not None:
                                    pauli1, phase = first_gate(p1)
                                else:
                                    pauli1 = p1
                                if second_gate is not None:
                                    pauli2, phase = second_gate(p2)
                                else:
                                    pauli2 = p2
                                if cnot_reversed:
                                    pauli2, pauli1, phase = apply_cnot(pauli2, pauli1)
                                else:
                                    pauli1, pauli2, phase = apply_cnot(pauli1, pauli2)
                                if target0 == (pauli1 == 0b00) and target1 == (pauli2 == 0b00):
                                    options_test.append((first_gate, second_gate, cnot_reversed))
    
                    if len(get_gates(options)) != len(options_test):
                        print('ERROR: different number of options', len(get_gates(options)), len(options_test))
                        print('p1:', p1, 'p2:', p2, 'target0:', target0, 'target1:', target1)
                        print('options:', get_gates(options))
                        print('options_test:', options_test)
                        ok = False
                        input()
                    options = get_gates(options)
                    while len(options) > 0:
                        option = options.pop()
                        if option not in options_test:
                            print('ERROR: option not in options_test', option)
                            print('p1:', p1, 'p2:', p2, 'target0:', target0, 'target1:', target1)
                            print('options:', options)
                            print('options_test:', options_test)
                            ok = False
                            input()
    print('Test ok:', ok)

def check_cdconns_integrity(pauligraph, pauligraph_degrees, gadget_data, last_edge):
    """ Check that pauligraph_degrees and pauligraph are consistent, and that last_edge is correct."""
    num_qubits, num_gadgets = gadget_data.shape
    for g in range(num_gadgets):
        for q in range(num_qubits):
            if pauligraph_degrees[g,q] != pauligraph[g,q].sum():
                print('ERROR: pauligraph_degrees does not match pauligraph for gadget', g)
                print('pauligraph_degrees:', pauligraph_degrees[g])
                print('pauligraph:', pauligraph[g])
                print_sorted_gd(gadget_data)
                input()
            if pauligraph_degrees[g,q] < 0:
                print('ERROR: negative degree for gadget', g, 'qubit', q)
                print('pauligraph_degrees:', pauligraph_degrees[g])
                print('pauligraph:', pauligraph[g])
                print_sorted_gd(gadget_data)
                input()
            if pauligraph_degrees[g,q] == 1 and last_edge[g,q] == -1:
                print('ERROR: last_edge is -1 for gadget', g, 'qubit', q)
                print('gadget_data:', gadget_data[:,g])
                print('last edge:', last_edge[g])
                print('pauligraph_degrees:', pauligraph_degrees[g])
                print('pauligraph:', pauligraph[g])
                print_sorted_gd(gadget_data)
                input() 
            if pauligraph_degrees[g,q] == 0 and gadget_data[q,g] != 0b00 and pauligraph_degrees[g].sum() > 0:
                print('ERROR: degree is 0 but gadget_data is not I for gadget', g, 'qubit', q)
                print('pauligraph_degrees:', pauligraph_degrees[g])
                print('gadget_data:', gadget_data[:,g])
                print_sorted_gd(gadget_data)
                input()

def check_equal_gates():
    print('Check equal gates')
    found = False
    count = 0
    for op11 in [apply_I, apply_vdg, apply_vs]:
        for op12 in [apply_vdg, apply_sdg, apply_sv]:
            for cnotr1 in [False]:
                combination = []
                for op21 in [apply_I, apply_vdg, apply_vs]:
                    for op22 in [apply_vdg, apply_sdg, apply_sv]:
                        for cnotr2 in [False]:
                            if op11 == op21 and op12 == op22 and cnotr1 == cnotr2:
                                continue
                            this_is_ok = True
                            for p1 in [0b01,0b11,0b10,0b00]:
                                for p2 in [0b01,0b11,0b10,0b00]:
                                    p11,p12,_ = apply_cnot(op11(p1)[0], op12(p2)[0])
                                    p21,p22,_ = apply_cnot(op21(p1)[0], op22(p2)[0])
                                    if cnotr1:
                                        p12,p11,_ = apply_cnot(op12(p2)[0], op11(p1)[0])
                                    if cnotr2:
                                        p22,p21,_ = apply_cnot(op22(p2)[0], op21(p1)[0])
                                    if ((p11==0b00) != (p21==0b00)) or ((p12==0b00) != (p22==0b00)):
                                        this_is_ok = False
                            if this_is_ok:
                                combination.append((op21.__name__, op22.__name__, cnotr2))
                if len(combination) > 0:
                    count += 1
                    found = True
                    print()
                    print(op11.__name__, op12.__name__, cnotr1, '->', combination)
    print('Found equivalent gates:', found)
    print(count)

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

#test_possible_gates()
#check_equal_gates()