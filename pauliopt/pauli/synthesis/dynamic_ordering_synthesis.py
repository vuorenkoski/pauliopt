import networkx as nx
import numpy as np
import random, time

from pauliopt.clifford.clifford_tableau import CliffordTableau
from pauliopt.gates import CX, H, Vdg, Sdg
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

# This six different single qubit gates are used in the algorithm: I, V, S, VS, SV, SVS(H)

# Time complexity O(m^2 n x) where n is number of qubits and m number of gadgets. X depends on topology: maximum degree of
# physical qubit (for example in line it is 2 and in grid 4). max of x is n-1.
# we have to process m gadgets and n qubits, so n*m
# Each opearation takes 36*m*x time

def pauli_polynomial_dynamic_ordering(pp: PauliPolynomial, topo: Topology, print_order=None, debug=False, random_sel=False):
    num_qubits = pp.num_qubits
    num_gadgets = len(pp.pauli_gadgets)
    removed_gadgets_num = 0
    gadget_angles = []
    removed_gadgets = np.zeros((num_gadgets), dtype=np.int8)
    qc_out = Circuit(num_qubits)
    qc_prop = []
    perm_gadgets = []

    gadget_rand_num = [] # This allows randomness version to order gadgets partly randomly, instead if trivially
    for i in range(num_gadgets):
        gadget_rand_num.append(random.randrange(num_gadgets * 100))

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
        tree_graph = steiner_tree(gadget_data, topo, i)
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

    debug and print_sorted_gd(gadget_data, order=print_order)

    # Main loop going through gadgets starting from smallest
    removed_gadgets_num += check_for_singles(general_data, circ_data, perm_gadgets)
    debug and print_sorted_gd(gadget_data, order=print_order)
    while removed_gadgets_num < num_gadgets:
        # Randomness 1: there are for example many gadgets having same size and no steiner nodes, how to arrange them?
        next = next_gadget(general_data, gadget_rand_num, random_sel=random_sel)
        num_legs = 0
        for j in range(num_qubits):
            if gadget_data[j,next] != 0b00:
                num_legs += 1
        if num_legs == 0:
            print('ERROR: qubit map has no nodes')
            input()
        
        # Loop going through qubits in gadget, removing them one by one
        while num_legs > 1:
            # randomness 2: if there are two or more edge+gate combinations having similar match, which one to choose?
            edge, gates = next_edge_to_remove(next, general_data, gate_combinations, removed_gadgets_num, random_sel=random_sel)
            rgadgets = add_cnot_and_single_qubit_gates(edge, gates, general_data, circ_data, perm_gadgets)
            removed_gadgets_num += rgadgets
            debug and print('-------Next edge to remove:', edge, 'gates:', gates, 'gadget', next)
            debug and print_sorted_gd(gadget_data, order=print_order)
            debug and input()
            if gadget_data[edge[0], next] == 0b00:
                num_legs -= 1

    # do clifford synthesis for the second part
    qc_prop_r = list(reversed(qc_prop))
    ct_prop = CliffordTableau(num_qubits)
#    cnot_table = np.zeros((num_qubits, num_qubits), dtype=np.int8)
    for gate in qc_prop_r:
        ct_prop.append_gate(gate)
#        if isinstance(gate, CX):
#            cnot_table[gate.control, gate.target] += 1
    qc_prop_syn, permutation = ct_prop.to_clifford_circuit_perm_row_col(topo, include_swaps=False)

    circ_out = qc_out + qc_prop_syn
    circ_out.final_permutation = qc_prop_syn.final_permutation
    permutation = [circ_out.final_permutation[i] for i in range(pp.num_qubits)]

    pre_cx = 0
    for gate in qc_out.gates:
        if isinstance(gate, CX):
            pre_cx += 1

    return circ_out, perm_gadgets, permutation, {'pre-cx': pre_cx}

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

def next_edge_to_remove2(gadget_index, general_data, gate_combinations, removed_gadgets_num, random_sel=False):
    """Decide what edge to remove next and what gates to apply"""
    gadget_data, gadget_angles, removed_gadgets, pauligraph, pauligraph_degrees, last_edge = general_data
    num_qubits, num_gadgets = gadget_data.shape

    # find different option to remove qubit (edge)
    edge_options = []
    for q in range(num_qubits):
        if pauligraph_degrees[gadget_index, q] == 1:
            edge_options.append((q,int(last_edge[gadget_index,q])))
    
    edge_gates = []
    for qubit0,qubit1 in edge_options:
        pauli0 = gadget_data[qubit0,gadget_index]
        pauli1 = gadget_data[qubit1,gadget_index]
        gate_options = []
        for i, cnot_reversed in enumerate([False,True]):
            for j, first_qubit in enumerate([apply_I, apply_vdg, apply_sdg, apply_svs, apply_sv, apply_vs]):
                for k, second_qubit in enumerate([apply_I, apply_vdg, apply_sdg, apply_svs, apply_sv, apply_vs]):
                    gate_combination = (first_qubit, second_qubit, cnot_reversed)
                    p0,p1,_ = apply_single_and_cnot(gate_combination, pauli0, pauli1)
                    if pauli1 == 0b00: # swap needed, change to non-I
                        if p1 != 0b00: 
                            gate_options.append(gate_combination)
                    else: # change first one to I
                        if p0 == 0b00: 
                            gate_options.append(gate_combination)
        edge_gates.append(gate_options)
#        print('-------Edge gates:', qubit0, qubit1, len(gate_options))

    if num_gadgets - removed_gadgets_num == 1: # Only one gadget left, choose any
        return edge_options[0], edge_gates[0][0]
    
    options = []
    for i,(qubit0,qubit1) in enumerate(edge_options):
        for gates in edge_gates[i]:
            score_end_both = 0
            score_middle_both = 0
            score_end_one = 0
            score_middle_one = 0

            for gadget in range(num_gadgets):
                if removed_gadgets[gadget]:
                    continue
                pauli0, pauli1 = gadget_data[qubit0,gadget], gadget_data[qubit1,gadget]
                if pauli0==0b00 and pauli1==0b00:
                    continue
#                print(gates)
                p0,p1,_ = apply_single_and_cnot(gates, pauli0, pauli1)
                qubit0_degree = pauligraph_degrees[gadget,qubit0]
                qubit1_degree = pauligraph_degrees[gadget,qubit1]
                pauli0 = pcomponents(pauli0)
                pauli1 = pcomponents(pauli1)
                p0 = pcomponents(p0)
                p1 = pcomponents(p1)
                score_end = [0,0]
                score_middle = [0,0]
                for comp in range(2):
                    if qubit0_degree > 1 and qubit1_degree > 1: # Middle of chain
                        if pauli0[comp] == 0 and p0[comp] == 1:
                            score_middle[comp] += 1
                        if pauli1[comp] == 0 and p1[comp] == 1:
                            score_middle[comp] += 1
                        if pauli0[comp] == 1 and p0[comp] == 0:
                            score_middle[comp] -= 1
                        if pauli1[comp] == 1 and p1[comp] == 0:
                            score_middle[comp] -= 1
                    elif qubit0_degree == 1 and qubit1_degree > 1: # End of chain
                        if p0[comp] == 0:
                            score_end[comp] += 1
                        if pauli1[comp] == 1 and p1[comp] == 0:
                            score_middle[comp] -= 1
                        if pauli1[comp] == 0 and p1[comp] == 0:
                            score_middle[comp] += 1
                    elif qubit0_degree > 1 and qubit1_degree == 1: # End of chain
                        if p1[comp] == 0:
                            score_end[comp] += 1
                        if pauli0[comp] == 1 and p0[comp] == 0:
                            score_middle[comp] -= 1
                        if pauli0[comp] == 0 and p0[comp] == 0:
                            score_middle[comp] += 1
                    elif qubit0_degree == 1 and qubit1_degree == 1: # Last pair
                        if p0[comp] == 0 or p1[comp] == 0:
                            score_end[comp] += 1
                    elif qubit1_degree == 0: # half otuside of chain
                        if p1[comp] == 1:
                            score_end[comp] -=1          
                    elif qubit0_degree == 0: # half otuside of chain
                        if p0[comp] == 1:
                            score_end[comp] -=1          

                if score_end[0] == 1 and score_end[1] == 1:
                    score_end_both += 1
                elif score_end[0] == -1 and score_end[1] == -1:
                    score_end_both -= 1
                else:
                    score_end_one += score_end[0] + score_end[1]

                if score_middle[0] == 1 and score_middle[1] == 1:
                    score_middle_both += 1
                elif score_middle[0] == -1 and score_middle[1] == -1:
                    score_middle_both -= 1
                else:
                    score_middle_one += score_middle[0] + score_middle[1]
            options.append(((qubit0,qubit1), gates, score_end_both, score_middle_both, score_end_one, score_middle_one))
    options.sort(key=lambda x: (-x[2], -x[3], -x[4], -x[5]))
#    options.sort(key=lambda x: (-x[2], -x[3]))
#    print(options[0])
#    print(options[1])
    return options[0][0], options[0][1]

def pcomponents(p):
    return p & 0b10, p & 0b01

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
        return edge_options[0], get_gates(edge_gates[0])[0]

    score = np.zeros((len(edge_options),9), dtype=int)
    score_remove_I = np.zeros((len(edge_options),9), dtype=int)
    for gadget in range(num_gadgets): # For each non-removed gadget, check how different edge+gate combinations would affect it
        if removed_gadgets[gadget]:
            continue

        for e in range(len(edge_options)):
            qubit0,qubit1 = edge_options[e]
            gates0 = edge_gates[e]
            qubit0_degree = pauligraph_degrees[gadget,qubit0]
            qubit1_degree = pauligraph_degrees[gadget,qubit1]
            legs = (gadget_data[qubit0,gadget], gadget_data[qubit1,gadget])
            intersection = 0
            I_intersection = 0
            score_change = 0
            I_score_change = 0

            # Pair is in the middle of branch. Avoid I:s
            if qubit0_degree > 1 and qubit1_degree > 1:
                if gadget_data[qubit0,gadget] == 0b00 or gadget_data[qubit1,gadget] == 0b00:
                    I_intersection = gates0 & gate_combinations[(legs,False,False)]
                    I_score_change = 1
                else:
                    I_intersection = gates0 & (gate_combinations[(legs,True,False)] | gate_combinations[(legs,False,True)])
                    I_score_change = -1

            # Pair is last pair in a brach in this gadget also. Try turn first qubit to I
            elif qubit0_degree == 1 and qubit1_degree > 1:
                intersection = gates0 & gate_combinations[(legs,True,False)]
                if intersection > 0:
                    score_change = 1
                if gadget_data[qubit1,gadget] == 0b00:
                    I_intersection = gates0 & gate_combinations[(legs,False,False)]
                    if I_intersection > 0:
                        I_score_change = +1
                else:
                    I_intersection = gates0 & gate_combinations[(legs,False,True)]
                    if I_intersection > 0:
                        I_score_change = -1

            # Same situation but reversed
            elif qubit0_degree > 1 and qubit1_degree == 1:
                intersection = gates0 & gate_combinations[(legs,False,True)]
                if intersection > 0:
                    score_change = 1
                if gadget_data[qubit0,gadget] == 0b00:
                    I_intersection = gates0 & gate_combinations[(legs,False,False)]
                    if I_intersection > 0:
                        I_score_change = +1
                else:
                    I_intersection = gates0 & gate_combinations[(legs,True,False)]
                    if I_intersection > 0:
                        I_score_change = -1

            # Pair touches a branch. Try not to extend branch
            elif qubit1_degree == 0 or qubit0_degree == 0:
                intersection = gates0 & gate_combinations[(legs,False,False)]
                if intersection > 0:
                    score_change = -1

            # Pair is last pair in a brach in this gadget. Try turn one qubit to I
            elif qubit0_degree == 1 and qubit1_degree == 1:
                intersection = gates0 & (gate_combinations[(legs,False,True)] | gate_combinations[(legs,True,False)])
                if intersection > 0:
                    score_change = 1

            if I_intersection > 0:
                for i in range(9):
                    if I_intersection & (1<<i):
                        score_remove_I[e,i] += I_score_change
            if intersection > 0:
                for i in range(9):
                    if intersection & (1<<i):
                        score[e,i] += score_change

    option_possibilities = []
    for e in range(len(edge_options)):
        gates = edge_gates[e]
        for i in range(9):
            if (gates & (1<<i)) > 0:
                option_possibilities.append((e, i, score[e,i], score_remove_I[e,i]))
    option_possibilities.sort(key=lambda x: (-x[2], -x[3]))

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
    return edge_options[edge_gates[0]], get_gate(1<<edge_gates[1])


def next_gadget(general_data, gadget_rand_num, random_sel=False):
    """ Order non-removed gadgets. Primary sorting is done by number of nodes in steiner tree, secondary sorting by number of steiner nodes."""
    gadget_data, gadget_angles, removed_gadgets, pauligraph, pauligraph_degrees, last_edge = general_data
    num_qubits, num_gadgets = gadget_data.shape
    order = []
    for i in range(num_gadgets):
        if removed_gadgets[i]:
            continue
        s_nodes, nodes = steiner_nodes(gadget_data, pauligraph_degrees, i)
        order.append((i, s_nodes, nodes+s_nodes, gadget_rand_num[i]))
    if random_sel:
        order.sort(key=lambda x: (x[2], x[1], x[3]))
    else:
        order.sort(key=lambda x: (x[2], x[1], x[0]))
    return order[0][0]


def add_cnot_and_single_qubit_gates(edge, gates, general_data, circ_data, perm_gadgets):
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
    """ Update pauligraph and pauligraph_degrees for gadget. Loop through all edge pairs which are in the edge of branch. Check is there changes for those."""
    gadget_data, gadget_angles, removed_gadgets, pauligraph, pauligraph_degrees, last_edge = general_data
    num_qubits, num_gadgets = gadget_data.shape
    gadget_removed = False
    removed_connections = 0
    for q1 in range(num_qubits): # Loop trough all edge pairs of of gadget
        if (q1!=qubit1 and q1!=qubit2) or (pauligraph_degrees[gadget_index,q1] != 1):
            continue
        q2 = last_edge[gadget_index,q1]

        # There are only two neighbouring qubits left which matches for cnot
        if pauligraph_degrees[gadget_index,q2] == 1 and ((q1 == qubit1 and q2 == qubit2) or (q1 == qubit2 and q2 == qubit1)): # This is final pair
            if pauli1 == 0b00 or pauli2 == 0b00:
                remove_connection(general_data, gadget_index, qubit1, qubit2)
                removed_connections += 1
                gadget_removed = True
                break

        # this gadget has exactly same edge as the original one
        elif q1 == qubit1 and q2 == qubit2: 
            if pauli1 == 0b00: # Chain shortens
                remove_connection(general_data, gadget_index, qubit1, qubit2)
                removed_connections += 1
                last_edge[gadget_index,qubit1] = -1
                for i in range(num_qubits):
                    if pauligraph[gadget_index,qubit2,i] == 1:
                        last_edge[gadget_index,qubit2] = i
                        break

        # This gadget has same edge but reversed
        elif q1 == qubit2 and q2 == qubit1: 
            if pauli2 == 0b00: # Chain shortens
                remove_connection(general_data, gadget_index, qubit1, qubit2)
                removed_connections += 1
                last_edge[gadget_index,qubit2] = -1
                for i in range(num_qubits):
                    if pauligraph[gadget_index,qubit1,i] == 1:
                        last_edge[gadget_index,qubit1] = i
                        break

        # edge is not and last edge in this gadget, but q1 is the first qubit of cnot
        elif q1 == qubit1:
            if pauli1 == 0b00 and pauligraph_degrees[gadget_index,qubit1] == 1: # chain shortens
                remove_connection(general_data, gadget_index, qubit1, q2)
                removed_connections += 1
                last_edge[gadget_index,qubit1] = -1
                gadget_removed = follow_chain_until_not_I(general_data, gadget_index, q2, gadget_data[q2,gadget_index])
            elif pauli2 != 0b00 and pauligraph_degrees[gadget_index,qubit2] == 0: # Branch extends towars qubit2
                add_connection(general_data, gadget_index, qubit1, qubit2)
                last_edge[gadget_index,qubit2] = qubit1
                last_edge[gadget_index,qubit1] = -1

        # Edge is not the last edge in this gadget but q1 is the second qubit of cnot
        elif q1 == qubit2:
            if pauli2 == 0b00 and pauligraph_degrees[gadget_index,qubit2] == 1: # Branch shortens
                remove_connection(general_data, gadget_index, qubit2, q2)
                removed_connections += 1
                last_edge[gadget_index,qubit2] = -1
                gadget_removed = follow_chain_until_not_I(general_data, gadget_index, q2, gadget_data[q2,gadget_index])
            elif pauli1 != 0b00 and pauligraph_degrees[gadget_index,qubit1] == 0: # Branch extends
                add_connection(general_data, gadget_index, qubit1, qubit2)
                last_edge[gadget_index,qubit1] = qubit2
                last_edge[gadget_index,qubit2] = -1

        # Edge is not the last edge in this gadget but q2 is the second qubit of cnot, new branch can emerge
        elif q2 == qubit2:
            if pauli1 != 0b00 and pauligraph_degrees[gadget_index,qubit1] == 0: # New branch
                add_connection(general_data, gadget_index, qubit1, qubit2)
                last_edge[gadget_index,qubit1] = qubit2

        # Edge is not the last edge in this gadget but q2 is the first qubit of cnot
        elif q2 == qubit1:
            if pauli2 != 0b00 and pauligraph_degrees[gadget_index,qubit2] == 0: # New branch
                add_connection(general_data, gadget_index, qubit1, qubit2)
                last_edge[gadget_index,qubit2] = qubit1

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
    # CHANGED SO THAT IT RETURNS ONLY 36 POSSIBLE COMBINATIONS, NOT DISTINGUISHING CNOT DIRECTION
    gates = []
#    for i, cnot_reversed in enumerate([False, True]):
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

# Functions for testing and debugging

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
                input()
            if pauligraph_degrees[g,q] < 0:
                print('ERROR: negative degree for gadget', g, 'qubit', q)
                print('pauligraph_degrees:', pauligraph_degrees[g])
                print('pauligraph:', pauligraph[g])
                input()
            if pauligraph_degrees[g,q] == 1 and last_edge[g,q] == -1:
                print('ERROR: last_edge is -1 for gadget', g, 'qubit', q)
                print('gadget_data:', gadget_data[:,g])
                print('last edge:', last_edge[g])
                print('pauligraph_degrees:', pauligraph_degrees[g])
                print('pauligraph:', pauligraph[g])
                input() 
            if pauligraph_degrees[g,q] == 0 and gadget_data[q,g] != 0b00 and pauligraph_degrees[g].sum() > 0:
                print('ERROR: degree is 0 but gadget_data is not I for gadget', g, 'qubit', q)
                print('pauligraph_degrees:', pauligraph_degrees[g])
                print('gadget_data:', gadget_data[:,g])
                input()

def check_equal_gates():
    print('Check equal gates')
    found = False
    count = 0
    for op11 in [apply_I, apply_vdg, apply_vs]:
        for op12 in [apply_vdg, apply_sdg, apply_sv]:
            for cnotr1 in [False]:
#    for op11 in [apply_I, apply_svs, apply_vdg, apply_sdg, apply_sv, apply_vs]:
#        for op12 in [apply_I, apply_svs, apply_vdg, apply_sdg, apply_sv, apply_vs]:
#            for cnotr1 in [False, True]:
                combination = []
                for op21 in [apply_I, apply_vdg, apply_vs]:
                    for op22 in [apply_vdg, apply_sdg, apply_sv]:
                        for cnotr2 in [False]:
#                for op21 in [apply_I, apply_svs, apply_vdg, apply_sdg, apply_sv, apply_vs]:
#                    for op22 in [apply_I, apply_svs, apply_vdg, apply_sdg, apply_sv, apply_vs]:
#                        for cnotr2 in [False, True]:
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