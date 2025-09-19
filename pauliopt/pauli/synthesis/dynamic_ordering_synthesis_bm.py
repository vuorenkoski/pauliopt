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

# This has HS and SH possibilities as single qubit gates. So 6*6*2 = 72 possibilities altogether
# According to above paper we need only 18 different chunks, all of these are in use in this algoritm, but this has more.
# Paper has better datastructure for gadget_data.

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
    gadget_data_bm = np.zeros((2,num_qubits,num_gadgets), dtype=('uint8')) # Dynamic matrix representing paulis
 #   gadget_data = np.zeros((num_qubits,num_gadgets), dtype=('U1')) # Dynamic matrix representing paulis
    gdconns = np.zeros((num_gadgets,num_qubits,num_qubits), dtype=np.int8) # Dynamic matrix represnting steiner trees
    gdconns_degrees = np.zeros((num_gadgets, num_qubits), dtype=np.int8) # Dynamic matrix represnting degrees of node in steiner trees
    last_edge = np.zeros((num_gadgets, num_qubits), dtype=int) # Dynamic matrix representing last edges in tree branches
    for i,gadget in enumerate(pp.pauli_gadgets):
        for j in range(num_qubits):
            last_edge[i,j] = -1
            if gadget.paulis[j] == I:
                gadget_data_bm[0,j,i] = 0
                gadget_data_bm[1,j,i] = 0
            elif gadget.paulis[j] == X:
                gadget_data_bm[0,j,i] = 1
                gadget_data_bm[1,j,i] = 0
            elif gadget.paulis[j] == Y:
                gadget_data_bm[0,j,i] = 1
                gadget_data_bm[1,j,i] = 1
            elif gadget.paulis[j] == Z:
                gadget_data_bm[0,j,i] = 0
                gadget_data_bm[1,j,i] = 1
        gadget_angles.append(gadget.angle)
        qubit_map = map_gadget(gadget_data_bm, topo, i)
        update_gdconns_from_qubit_map(qubit_map, gdconns, gdconns_degrees, last_edge, i)
    gate_combinations = {} # Immutable dictionary of possible gates for each pair of paulis and targets
    for p1 in [(1,0), (1,1), (0,1), (0,0)]:
        for p2 in [(1,0), (1,1), (0,1), (0,0)]:
            for target0 in [False, True]: # Is target to have I or not after gates in qubit0
                for target1 in [False, True]: # Is target to have I or not after gates in qubit1
                    options = possible_gates(p1, p2, target0, target1)
                    gate_combinations[(p1,p2, target0, target1)] = options
    general_data = (gadget_data_bm, gadget_angles, removed_gadgets, gdconns, gdconns_degrees, last_edge)
    circ_data = (qc_out, qc_prop)
    debug and print_sorted_gd(gadget_data_bm, order=print_order)

    # Main loop going through gadgets starting from smallest
    removed_gadgets_num += check_for_singles(general_data, circ_data, perm_gadgets)
    debug and print_sorted_gd(gadget_data_bm, order=print_order)
    while removed_gadgets_num < num_gadgets:
        # Randomness 1: there are for example many gadgets having same size and no steiner nodes, how to arrange them?
        order = order_gadgets(general_data, gadget_rand_num, random_sel=random_sel)
        next = order[0][0]
        qubit_map = make_map_from_gdconns(gdconns, gdconns_degrees, next)
        if qubit_map.number_of_nodes() == 0:
            print('ERROR: qubit map has no nodes')
            input()
        
        # Loop going through qubits in gadget, removing them one by one
        while qubit_map.number_of_nodes() > 1:
            # randomness 2: if there are two or more edge+gate combinations having similar match, which one to choose?
            edge, gates = next_edge_to_remove(qubit_map, next, order, general_data, gate_combinations, random_sel=random_sel)
            rgadgets = add_cnot_and_single_qubit_gates(edge, gates, general_data, circ_data, perm_gadgets)
            removed_gadgets_num += rgadgets
            debug and print('-------Next edge to remove:', edge, 'gates:', gates[0].__name__, gates[1].__name__, gates[2])
            debug and print_sorted_gd(gadget_data_bm, order=print_order)
            debug and input()
            if gadget_data_bm[0,edge[0], next] == 0 and gadget_data_bm[1,edge[0], next] == 0:
                qubit_map.remove_node(edge[0])

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

    return circ_out, perm_gadgets, permutation, {'pre-cx': pre_cx}

def update_gdconns_from_qubit_map(qubit_map, gdconns, gdconns_degrees, last_edge, gadget_index):
    """Update connection datastructure based on qubit_map provided by networkx steinertree algorithm."""
    num_qubits = gdconns.shape[1]
    for j in range(num_qubits):
        if qubit_map.has_node(j):
            gdconns_degrees[gadget_index,j] = qubit_map.degree[j]
            if qubit_map.degree[j] == 1:
                last_edge[gadget_index,j] = list(qubit_map.edges(j))[0][1] # what is the border of edge node
        else:
            gdconns_degrees[gadget_index,j] = 0
    for j in range(num_qubits):
        for k in range(num_qubits):
            gdconns[gadget_index,j,k] = 0
    for edges in qubit_map.edges():
        gdconns[gadget_index, edges[0], edges[1]] = 1
        gdconns[gadget_index, edges[1], edges[0]] = 1


def check_for_singles(general_data, circ_data, perm_gadgets):
    """ check if there are gadgets having only single leg and remove them if so."""
    gadget_data, gadget_angles, removed_gadgets, gdconns, gdconns_degrees, last_edge = general_data
    _, num_qubits, num_gadgets = gadget_data.shape
    removed_gadgets_num = 0

    for i in range(num_gadgets):
        if removed_gadgets[i] == 1:
            continue
        x = 0
        for j in range(num_qubits):
            if gadget_data[0,j,i] or gadget_data[1,j,i]:
                x += 1
                qubit = j 
        if x == 1:
            removed_gadgets_num += 1
            remove_single(general_data, circ_data, perm_gadgets, i, qubit)
    return removed_gadgets_num

def remove_single(general_data, circ_data, perm_gadgets, gadget_index, qubit):
    """ Remove gadget having single leg indicated by gadget_index."""
    gadget_data, gadget_angles, removed_gadgets, gdconns, gdconns_degrees, last_edge = general_data
    qc_out, qc_prop = circ_data
    pauli = (gadget_data[0,qubit,gadget_index], gadget_data[1,qubit,gadget_index])
    removed_gadgets[gadget_index] = 1
    if pauli == (1,0):
        qc_out.h(qubit)
    elif pauli == (1,1):
        qc_out.v(qubit)
    qc_out.rz(gadget_angles[gadget_index], qubit)
    if pauli == (1,0):
        qc_out.h(qubit)
    elif pauli == (1,1):
        qc_out.vdg(qubit)
    perm_gadgets.append(gadget_index)
    gadget_data[0,qubit,gadget_index], gadget_data[1,qubit,gadget_index] = (0,0)

def next_edge_to_remove(qubit_map, gadget_index, order, general_data, gate_combinations, random_sel=False):
    """Decide what edge to remove next and what gates to apply"""
    # This version primarily chooses gates which minimizes overall effect to the length of chains. 
    # If there are many options having the same score, it secondarily tries to remove/avoid identity gates from the middle of chains.
    gadget_data, gadget_angles, removed_gadgets, gdconns, gdconns_degrees, last_edge = general_data
    edge = None
    edge_options = []
    for v in qubit_map.nodes():
        if qubit_map.degree[v] == 1:
            edge_options.append(list(qubit_map.edges(v))[0])
    if len(edge_options) == 0: # current map is cycle (all nodes degrees>1), add edges with degree 2 to edge_options
        for v in qubit_map.nodes():
            if qubit_map.degree[v] == 2:
                edge_options.append(list(qubit_map.edges(v))[0])

    edge_gates = []
    edge_options_left = []
    edge_excellent = []
    good_edges_left = len(edge_options)
    for qubit0,qubit1 in edge_options:
        edge_options_left.append(True)
        edge_excellent.append(0)
        p0 = (gadget_data[0,qubit0,gadget_index], gadget_data[1,qubit0,gadget_index])
        p1 = (gadget_data[0,qubit1,gadget_index], gadget_data[1,qubit1,gadget_index])
        if gadget_data[0,qubit1,gadget_index] == 0 and gadget_data[1,qubit1,gadget_index] == 0: #swap needed
            options = gate_combinations[(p0,p1,False,False)]
            edge_gates.append(options)
        else:
            options = gate_combinations[(p0,p1,True,False)]
            edge_gates.append(options)

    if len(order) == 1: # Only one gadget left, choose any
        return edge_options[0], get_gates(edge_gates[0])[0]
    g=1

    score = np.zeros((len(edge_options),72), dtype=int)
    score_remove_I = np.zeros((len(edge_options),72), dtype=int)
    possibility_to_influence_length = False
    while g < len(order):
        gadget = order[g][0]
        g += 1

        for e in range(len(edge_options)):
            if not edge_options_left[e]:
                continue
            qubit0,qubit1 = edge_options[e]
            gates0 = edge_gates[e]
            qubit0_degree = gdconns_degrees[gadget,qubit0]
            qubit1_degree = gdconns_degrees[gadget,qubit1]
            leg0 = (gadget_data[0,qubit0,gadget], gadget_data[1,qubit0,gadget])
            leg1 = (gadget_data[0,qubit1,gadget], gadget_data[1,qubit1,gadget])
            intersection = 0
            score_change = 0

            if qubit0_degree == 1 or qubit1_degree == 1:
                possibility_to_influence_length = True

            # Pair is in the middle of branch. Avoid I:s
            if qubit0_degree > 1 and qubit1_degree > 1:
                if (gadget_data[0,qubit0,gadget] == 0 and gadget_data[1,qubit0,gadget] == 0) or (gadget_data[0,qubit1,gadget] == 0 and gadget_data[1,qubit1,gadget] == 0):
                    intersection = gates0 & gate_combinations[(leg0,leg1,False,False)]
                    score_change = 1
                else:
                    intersection = gates0 & (gate_combinations[(leg0,leg1,True,False)] | gate_combinations[(leg1,leg0,False,True)])
                    score_change = -1
                if intersection > 0:
                    for i in range(72):
                        if intersection & (1<<i):
                            score_remove_I[e,i] += score_change
                continue

            # Pair is last pair in a brach in this gadget also. Try turn first qubit to I
            elif qubit0_degree == 1 and qubit1_degree > 1:
                intersection = gates0 & gate_combinations[(leg0,leg1,True,False)]
                if intersection > 0:
                    score_change = 1

            # Same situation but reversed
            elif qubit0_degree > 1 and qubit1_degree == 1:
                intersection = gates0 & gate_combinations[(leg0,leg1,False,True)]
                if intersection > 0:
                    score_change = 1

            # Pair is touches last qubit of a branch. Try not to extneded branch
            elif qubit0_degree == 1 and qubit1_degree == 0:
                intersection = gates0 & gate_combinations[(leg0,leg1,False,False)]
                if intersection > 0:
                    score_change = -1

            # Same situation but reversed
            elif qubit0_degree == 0 and qubit1_degree == 1:
                intersection = gates0 & gate_combinations[(leg0,leg1,False, False)]
                if intersection > 0:
                    score_change = -1

            elif qubit0_degree == 1 and qubit1_degree == 1:
                intersection = gates0 & (gate_combinations[(leg0,leg1,False,True)] | gate_combinations[(leg0,leg1,True,False)])
                if intersection > 0:
                    score_change = 1

            if intersection > 0:
                for i in range(72):
                    if intersection & (1<<i):
                        score[e,i] += score_change

    max_score = 0
    max_edge_gates_combination = None
    if possibility_to_influence_length:
        for e in range(len(edge_options)):
            for i in range(72):
                if (edge_gates[e] & (1<<i)) > 0:
                    if max_edge_gates_combination is None or score[e,i] > max_score:
                        max_score = score[e,i]
                        max_edge_gates_combination = [(e, i, score_remove_I[e,i])]
                    elif score[e,i] == max_score:
                        max_edge_gates_combination.append((e, i, score_remove_I[e,i]))
        max_edge_gates_combination.sort(key=lambda x: (-x[2]))
        best = []
        max_remove_I = max_edge_gates_combination[0][2]
        for x in max_edge_gates_combination:
            if x[2] == max_remove_I:
                best.append((x[0], x[1]))
        max_edge_gates_combination = best
    else:
        for e in range(len(edge_options)):
            for i in range(72):
                if (edge_gates[e] & (1<<i)) > 0:
                    if max_edge_gates_combination is None or score_remove_I[e,i] > max_score:
                        max_score = score_remove_I[e,i]
                        max_edge_gates_combination = [(e, i)]
                    elif score[e,i] == max_score:
                        max_edge_gates_combination.append((e, i))
    print(len(max_edge_gates_combination))
    if len(max_edge_gates_combination) % 2 > 0:
        print('Max score:', max_score, 'Possibilities:', max_edge_gates_combination)
        for c in max_edge_gates_combination:
            print(c[0],get_gates(1<<c[1])[0])
    if random_sel:
        edge_gates = random.choice(max_edge_gates_combination)
    else:
        edge_gates = max_edge_gates_combination[0]
    return edge_options[edge_gates[0]], get_gates(1<<edge_gates[1])[0]


def order_gadgets(general_data, gadget_rand_num, random_sel=False):
    """ Order non-removed gadgets. Primary sorting is done by number of nodes in steiner tree, secondary sorting by number of steiner nodes."""
    gadget_data, gadget_angles, removed_gadgets, gdconns, gdconns_degrees, last_edge = general_data
    _, num_qubits, num_gadgets = gadget_data.shape
    order = []
    for i in range(num_gadgets):
        if removed_gadgets[i]:
            continue
        s_nodes, nodes, min_from_edge = steiner_nodes(gadget_data, gdconns_degrees, i)
        order.append((i, s_nodes, nodes+s_nodes, gadget_rand_num[i], min_from_edge))
    if random_sel:
        order.sort(key=lambda x: (x[2], x[1], -x[4], x[3]))
    else:
        order.sort(key=lambda x: (x[2], x[1], -x[4], x[0]))
    return order


def add_cnot_and_single_qubit_gates(edge, gates, general_data, circ_data, perm_gadgets):
    """apply single qubit gates and CNOT to all non-removed gates."""
    gadget_data, gadget_angles, removed_gadgets, gdconns, gdconns_degrees, last_edge = general_data
    qc_out, qc_prop = circ_data
    _, num_qubits, num_gadgets = gadget_data.shape

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
            qc_out.v(qubit)
            qc_prop.append(Vdg(qubit))
            qc_out.s(qubit)
            qc_prop.append(Sdg(qubit))
        elif gate == apply_sv:
            qc_out.s(qubit)
            qc_prop.append(Sdg(qubit))
            qc_out.v(qubit)
            qc_prop.append(Vdg(qubit))
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
        pauli1 = (gadget_data[0,qubit1,gadget_index], gadget_data[1,qubit1,gadget_index])
        pauli2 = (gadget_data[0,qubit2,gadget_index], gadget_data[1,qubit2,gadget_index])
        if pauli1 == (0,0) and pauli2 == (0,0):
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
        gadget_data[0,qubit1,gadget_index], gadget_data[1,qubit1,gadget_index] = pauli1
        gadget_data[0,qubit2,gadget_index], gadget_data[1,qubit2,gadget_index] = pauli2
        gadget_angles[gadget_index] *= phase

        # Update gdconns and gdconns_degrees
        gadget_removed, rconnections = update_gdconns(general_data, gadget_index, qubit1, qubit2, pauli1, pauli2)
        removed_connections += rconnections
        if gadget_removed:
            if pauli1 == (0,0):
                remove_single(general_data, circ_data, perm_gadgets, gadget_index, qubit2)
            else:
                remove_single(general_data, circ_data, perm_gadgets, gadget_index, qubit1)
            removed_gadgets_num += 1
            gadget_data[0,qubit1,gadget_index], gadget_data[1,qubit1,gadget_index] = (0,0)
            gadget_data[0,qubit2,gadget_index], gadget_data[1,qubit2,gadget_index] = (0,0)
#        check_cdconns_integrity(gdconns, gdconns_degrees, gadget_data, last_edge)
    return removed_gadgets_num


def update_gdconns(general_data, gadget_index, qubit1, qubit2, pauli1, pauli2):
    """ Update gdconns and gdconns_degrees for gadget. Loop through all edge pairs which are in the edge of branch. Check is there changes for those."""
    gadget_data, gadget_angles, removed_gadgets, gdconns, gdconns_degrees, last_edge = general_data
    _, num_qubits, num_gadgets = gadget_data.shape
    gadget_removed = False
    removed_connections = 0
    for q1 in range(num_qubits):
        # If qubit is not in the end of branch/chain
        if (q1!=qubit1 or q1!=qubit2) and gdconns_degrees[gadget_index,q1] != 1:
            continue
        q2 = last_edge[gadget_index,q1]

        # There are only two neighbouring qubits left which matches for cnot
        if gdconns_degrees[gadget_index,q2] == 1 and ((q1 == qubit1 and q2 == qubit2) or (q1 == qubit2 and q2 == qubit1)): # This is final pair
            if pauli1 == (0,0) or pauli2 == (0,0):
                remove_connection(general_data, gadget_index, qubit1, qubit2)
                removed_connections += 1
                gadget_removed = True
                break

        # this gadget has exactly same edge as the original one
        elif q1 == qubit1 and q2 == qubit2: 
            if pauli1 == (0,0): # Chain shortens
                remove_connection(general_data, gadget_index, qubit1, qubit2)
                removed_connections += 1
                last_edge[gadget_index,qubit1] = -1
                gadget_removed = follow_chain_until_not_I(general_data, gadget_index, qubit2, pauli2)

        # This gadget has same edge but reversed
        elif q1 == qubit2 and q2 == qubit1: 
            if pauli2 == (0,0): # Chain shortens
                remove_connection(general_data, gadget_index, qubit1, qubit2)
                removed_connections += 1
                last_edge[gadget_index,qubit2] = -1
                gadget_removed = follow_chain_until_not_I(general_data, gadget_index, qubit1, pauli1)

        # edge is not and last edge in this gadget, but q1 is the first qubit of cnot
        elif q1 == qubit1:
            if pauli1 == (0,0) and gdconns_degrees[gadget_index,qubit1] == 1: # chain shortens
                remove_connection(general_data, gadget_index, qubit1, q2)
                removed_connections += 1
                last_edge[gadget_index,qubit1] = -1
                p = (gadget_data[0,q2,gadget_index], gadget_data[1,q2,gadget_index])
                gadget_removed = follow_chain_until_not_I(general_data, gadget_index, q2, p)
            elif pauli2 != (0,0) and gdconns_degrees[gadget_index,qubit2] == 0: # Branch extends towars qubit2
                add_connection(general_data, gadget_index, qubit1, qubit2)
                last_edge[gadget_index,qubit2] = qubit1
                last_edge[gadget_index,qubit1] = -1

        # Edge is not the last edge in this gadget but q1 is the second qubit of cnot
        elif q1 == qubit2:
            if pauli2 == (0,0) and gdconns_degrees[gadget_index,qubit2] == 1: # Branch shortens
                remove_connection(general_data, gadget_index, qubit2, q2)
                removed_connections += 1
                last_edge[gadget_index,qubit2] = -1
                p = (gadget_data[0,q2,gadget_index], gadget_data[1,q2,gadget_index])
                gadget_removed = follow_chain_until_not_I(general_data, gadget_index, q2, p)
            elif pauli1 != (0,0) and gdconns_degrees[gadget_index,qubit1] == 0: # Branch extends
                add_connection(general_data, gadget_index, qubit1, qubit2)
                last_edge[gadget_index,qubit1] = qubit2
                last_edge[gadget_index,qubit2] = -1

        # Edge is not the last edge in this gadget but q2 is the second qubit of cnot, new branch can emerge
        elif q2 == qubit2:
            if pauli1 != (0,0) and gdconns_degrees[gadget_index,qubit1] == 0: # New branch
                add_connection(general_data, gadget_index, qubit1, qubit2)
                last_edge[gadget_index,qubit1] = qubit2

        # Edge is not the last edge in this gadget but q2 is the first qubit of cnot
        elif q2 == qubit1:
            if pauli2 != (0,0) and gdconns_degrees[gadget_index,qubit2] == 0: # New branch
                add_connection(general_data, gadget_index, qubit1, qubit2)
                last_edge[gadget_index,qubit2] = qubit1

    # If one of the qubits of the edge is in the midddle of chain
    # and other qubit is not in any branch and turns from I to pauli: new branch
    if gdconns_degrees[gadget_index,qubit1] == 0 and gdconns_degrees[gadget_index,qubit2] > 1 and pauli1 != (0,0): # New branch
        add_connection(general_data, gadget_index, qubit1, qubit2)
        last_edge[gadget_index,qubit1] = qubit2
    elif gdconns_degrees[gadget_index,qubit2] == 0 and gdconns_degrees[gadget_index,qubit1] > 1 and pauli2 != (0,0): # New branch
        add_connection(general_data, gadget_index, qubit1, qubit2)
        last_edge[gadget_index,qubit2] = qubit1
    return gadget_removed, removed_connections


def follow_chain_until_not_I(general_data, gadget_index, qubit, pauli):
    """Recursive functions following chain of I:s until next qubit is not I. Return True if gadget was removed."""
    gadget_data, gadget_angles, removed_gadgets, gdconns, gdconns_degrees, last_edge = general_data
    num_qubits = gdconns.shape[1]
    if gdconns_degrees[gadget_index,qubit] > 1:
        return False
    
    qubit2 = None
    j = 0
    for i in range(num_qubits):
        if gdconns[gadget_index,qubit,i] == 1:
            qubit2 = i
            j += 1
            if j > 1:
                print('ERROR: follow_chain_until_not_I: more than one qubit found for gadget', gadget_index, 'qubit',qubit)
                input()
    if qubit2 is None:
        print('ERROR: follow_chain_until_not_I: no qubit found for gadget', gadget_index, 'qubit',qubit)
        input()

    # chain of I does not continue, this is now the last edge
    if pauli != (0,0):
        last_edge[gadget_index,qubit] = qubit2
        return False

    # pauli is I, we must remove connection
    remove_connection(general_data, gadget_index, qubit, qubit2)

    # If next qubit2 has degree 0 after removing connection, it means that this the only leg left
    if gdconns_degrees[gadget_index,qubit2] == 0:
        return True
    else: #chain continues beyond qubit2
        p = (gadget_data[0,qubit2,gadget_index], gadget_data[1,qubit2,gadget_index])
        return follow_chain_until_not_I(general_data, gadget_index, qubit2, p)


def add_connection(general_data, gadget_index, qubit1, qubit2):
    """Add connection between two qubits in a gadget"""
    gadget_data, gadget_angles, removed_gadgets, gdconns, gdconns_degrees, last_edge = general_data
    gdconns[gadget_index,qubit1,qubit2] = 1
    gdconns[gadget_index,qubit2,qubit1] = 1
    gdconns_degrees[gadget_index,qubit1] += 1
    gdconns_degrees[gadget_index,qubit2] += 1


def remove_connection(general_data, gadget_index, qubit1, qubit2):
    """Add connection between two qubits in a gadget"""
    gadget_data, gadget_angles, removed_gadgets, gdconns, gdconns_degrees, last_edge = general_data
    gdconns[gadget_index,qubit1,qubit2] = 0
    gdconns[gadget_index,qubit2,qubit1] = 0
    gdconns_degrees[gadget_index,qubit1] -= 1
    gdconns_degrees[gadget_index,qubit2] -= 1


def map_gadget(gadget_data,topo, gadget_index):
    """ Uses NetworkX Steinertree algorithm to make steinertree from gadget."""
    _, num_qubits, num_gadgets = gadget_data.shape
    nodes = []
    for i in range(num_qubits):
        if gadget_data[0,i,gadget_index] or gadget_data[1,i,gadget_index]:
            nodes.append(i)
    steiner_stree = nx.algorithms.approximation.steinertree.steiner_tree(topo.to_nx, nodes)
    return nx.Graph(steiner_stree)


def make_map_from_gdconns(gdconns, gdconns_degrees, gadget_index):
    """ Make steiner tree from gdconns data."""
    num_qubits = gdconns.shape[1]
    qubit_map = nx.Graph()
    for i in range(num_qubits):
        if gdconns_degrees[gadget_index,i] > 0:
            qubit_map.add_node(i)
    for i in range(num_qubits-1):
        for j in range(i+1,num_qubits):
            if gdconns[gadget_index,i,j] == 1:
                qubit_map.add_edge(i,j)
    return qubit_map


def steiner_nodes(gadget_data, gdconns_degrees, gadget_index):
    """ Defines number of steiner nodes and regular nodes of steiner tree (gdconns data)"""
    _, num_qubits, num_gadgets = gadget_data.shape
    nodes = 0
    steiner_nodes = 0
    min_from_edge = num_qubits
    for i in range(num_qubits):
        if gadget_data[0, i, gadget_index] or gadget_data[1, i, gadget_index]:
            nodes += 1
            from_edge = min(i,num_qubits-i-1)
            min_from_edge = min(min_from_edge, from_edge)
        elif gdconns_degrees[gadget_index, i] > 0:
            steiner_nodes += 1
    return steiner_nodes, nodes, min_from_edge

def apply_I(p):
    return p, 1

def apply_vdg(p):
    phase = 1
    if p == (0,1):
        phase = -1
    return (p[0]^p[1],p[1]), phase

def apply_sdg(p):
    phase = 1
    if p == (1,1):
        phase = -1
    return (p[0],p[0] ^ p[1]), phase

def apply_sv(p):
    pauli,phase1 = apply_sdg(p)
    pauli,phase2 = apply_vdg(pauli)
    return pauli, phase1*phase2

def apply_vs(p):
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
    if (p1 == (1,0) and p2 == (0,1)) or (p1 == (1,1) and p2 == (1,1)):
        phase = -1
    return (p1[0],p1[1] ^ p2[1]), (p1[0] ^ p2[0],p2[1]), phase

def possible_gates(p0, p1, target0, target1):
    """Return possible gates for given pair of paulis and targets to have or have not I.
    Possible gates include possible single qubit gate for qubit1, single qubit gate for qubit2 and cnot direction.
    :params paulis: tuple of two chars, e.g. ('X', 'Y') or ('I', 'Z')
    :params target0: True if qubit0 should have I, False if it should not have I
    :params target1: True if qubit1 should have I, False if it should not have I
    :returns: 72-bit integer coding possible gate combinations."""
    # coding I, V, S, H, SH, HS
    # HS X->Y, Y->Z, Z->X: first propagate S, then H
    # SH X->Z, Z->Y, Y->X 
    gates_none = np.zeros((2,6,6), dtype=object) # cnot reversed: (False, True), q0 ( I, V, S, H, HS, SH), q1 ( I, V, S, H, HS, SH)
    convert_to_X = {(1,0): [0,1], (1,1): [2,5], (0,1): [3,4]}
    convert_to_Y = {(1,0): [2,4], (1,1): [0,3], (0,1): [1,5]}
    convert_to_Z = {(1,0): [3,5], (1,1): [1,4], (0,1): [0,2]}

    if target0 and target1 and p0 == (0,0) and p1 == (0,0):       # current and target is II
        paulis_options = np.ones((2,6,6), dtype=object)  # all gates
    elif p0 == (0,0) and p1 == (0,0):                             # current is ??, target is II
        paulis_options = gates_none
    elif target0 and target1:                                     # Trying to remove both paulis, not possible
        paulis_options = gates_none
    elif target0 and not target1 and p0 != (0,0) and p1 == (0,0): # Swap needed, not possible
        paulis_options = gates_none
    elif not target0 and target1 and p0 == (0,0) and p1 != (0,0): # Swap needed, not possible
        paulis_options = gates_none
    elif target0 and not target1:
        paulis_options = gates_none
        if p0 == (0,0):
            for i in range(6):
                paulis_options[np.ix_([0],[i],convert_to_X[p1])] = 1
                paulis_options[np.ix_([1],[i],convert_to_Z[p1])] = 1 #1q-h
        else:
            paulis_options[np.ix_([1], convert_to_X[p0], convert_to_X[p1])] = 1 # XX
            paulis_options[np.ix_([1], convert_to_X[p0], convert_to_Y[p1])] = 1 # XY
            paulis_options[np.ix_([0], convert_to_Z[p0], convert_to_Z[p1])] = 1 # ZZ
            paulis_options[np.ix_([0], convert_to_Z[p0], convert_to_Y[p1])] = 1 # YZ
    elif not target0 and target1:
        paulis_options = gates_none
        if p1 == (0,0):
            for i in range(6):
                paulis_options[np.ix_([1],convert_to_X[p0],[i])] = 1
                paulis_options[np.ix_([0],convert_to_Z[p0],[i])] = 1 #1q-h
        else:
            paulis_options[np.ix_([0], convert_to_X[p0], convert_to_X[p1])] = 1 # XX
            paulis_options[np.ix_([0], convert_to_Y[p0], convert_to_X[p1])] = 1 # YX
            paulis_options[np.ix_([1], convert_to_Z[p0], convert_to_Z[p1])] = 1 # ZZ
            paulis_options[np.ix_([1], convert_to_Y[p0], convert_to_Z[p1])] = 1 # YZ

    elif not target0 and not target1:
        paulis_options = gates_none.copy()
        if p0 != (0,0) and p1 != (0,0):   # XY<->YZ, XZ<->YY, ZX<->ZX:   XY,XZ,   ZX,   YZ, YY
            paulis_options[np.ix_([0], convert_to_X[p0], convert_to_Y[p1])] = 1
            paulis_options[np.ix_([0], convert_to_X[p0], convert_to_Z[p1])] = 1
            paulis_options[np.ix_([0], convert_to_Z[p0], convert_to_X[p1])] = 1
            paulis_options[np.ix_([0], convert_to_Y[p0], convert_to_Z[p1])] = 1
            paulis_options[np.ix_([0], convert_to_Y[p0], convert_to_Y[p1])] = 1

            paulis_options[np.ix_([1], convert_to_Y[p0], convert_to_X[p1])] = 1
            paulis_options[np.ix_([1], convert_to_Z[p0], convert_to_X[p1])] = 1
            paulis_options[np.ix_([1], convert_to_X[p0], convert_to_Z[p1])] = 1
            paulis_options[np.ix_([1], convert_to_Z[p0], convert_to_Y[p1])] = 1
            paulis_options[np.ix_([1], convert_to_Y[p0], convert_to_Y[p1])] = 1
        if p0 == (0,0) and p1 != (0,0):   # XI, YI, IY, IZ
            for i in range(6):
                paulis_options[np.ix_([0],[i],convert_to_Y[p1])] = 1
                paulis_options[np.ix_([0],[i],convert_to_Z[p1])] = 1
                paulis_options[np.ix_([1],[i],convert_to_X[p1])] = 1
                paulis_options[np.ix_([1],[i],convert_to_Y[p1])] = 1
        if p0 != (0,0) and p1 == (0,0):
            for i in range(6):
                paulis_options[np.ix_([1],convert_to_Y[p0],[i])] = 1
                paulis_options[np.ix_([1],convert_to_Z[p0],[i])] = 1
                paulis_options[np.ix_([0],convert_to_X[p0],[i])] = 1
                paulis_options[np.ix_([0],convert_to_Y[p0],[i])] = 1
    else:
        print('XXXX Should not happen')

    options = 0
    for i, cnot_reversed in enumerate([False, True]):
        for j, first_qubit in enumerate([apply_I, apply_vdg, apply_sdg, apply_svs, apply_vs, apply_sv]):
            for k, second_qubit in enumerate([apply_I, apply_vdg, apply_sdg, apply_svs, apply_vs, apply_sv]):
                if paulis_options[i,j,k] == 1:
                    gate = 1 << (i*36 + j*6 + k)
                    options += gate
    return options

def get_gates(gate_set):
    """Return gate combinations for given gate set indicated by 72-bit integer.
    :params gate_set: 72-bit integer coding possible gate combinations.
    :returns: list of tuples (first_qubit_gate, second_qubit_gate, cnot direction).
    """
    gates = []
    for i, cnot_reversed in enumerate([False, True]):
        for j, first_qubit in enumerate([apply_I, apply_vdg, apply_sdg, apply_svs, apply_vs, apply_sv]):
            for k, second_qubit in enumerate([apply_I, apply_vdg, apply_sdg, apply_svs, apply_vs, apply_sv]):
                gate = 1 << (i*36 + j*6 + k)
                if gate & gate_set > 0:
                    gates.append((first_qubit, second_qubit, cnot_reversed))
                gate >>= 1
    return gates

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
    for p0 in [(1,0), (1,1), (0,1), (0,0)]:
        for p1 in [(1,0), (1,1), (0,1), (0,0)]:
            for target0 in [False, True]:
                for target1 in [False, True]:
                    options = possible_gates(p0, p1, target0, target1)
                    options_test = []
                    for cnot_reversed in [True, False]:
                        for first_gate in [apply_I, apply_vdg, apply_sdg, apply_svs, apply_vs, apply_sv]:
                            for second_gate in [apply_I, apply_vdg, apply_sdg, apply_svs, apply_vs, apply_sv]:
                                pauli1, phase = first_gate(p0)
                                pauli2, phase = second_gate(p1)
                                if cnot_reversed:
                                    pauli2, pauli1, phase = apply_cnot(pauli2, pauli1)
                                else:
                                    pauli1, pauli2, phase = apply_cnot(pauli1, pauli2)
                                if target0 == (pauli1 == (0,0)) and target1 == (pauli2 == (0,0)):
                                    options_test.append((first_gate, second_gate, cnot_reversed))
    
                    if len(get_gates(options)) != len(options_test):
                        print('ERROR: different number of options', len(get_gates(options)), len(options_test))
                        print('p1:', p0, 'p2:', p1, 'target0:', target0, 'target1:', target1)
                        print('options:', get_gates(options))
                        print('options_test:', options_test)
                        ok = False
                        input()
                    options = get_gates(options)
                    while len(options) > 0:
                        option = options.pop()
                        if option not in options_test:
                            print('ERROR: option not in options_test', option)
                            print('p1:', p0, 'p2:', p1, 'target0:', target0, 'target1:', target1)
                            print('options:', options)
                            print('options_test:', options_test)
                            ok = False
                            input()
    print('Test ok:', ok)

def check_cdconns_integrity(gdconns, gdconns_degrees, gadget_data, last_edge):
    """ Check that gdconns_degrees and gdconns are consistent, and that last_edge is correct."""
    _, num_qubits, num_gadgets = gadget_data.shape
    for g in range(num_gadgets):
        for q in range(num_qubits):
            if gdconns_degrees[g,q] != gdconns[g,q].sum():
                print('ERROR: gdconns_degrees does not match gdconns for gadget', g)
                print('gdconns_degrees:', gdconns_degrees[g])
                print('gdconns:', gdconns[g])
                input()
            if gdconns_degrees[g,q] < 0:
                print('ERROR: negative degree for gadget', g, 'qubit', q)
                print('gdconns_degrees:', gdconns_degrees[g])
                print('gdconns:', gdconns[g])
                input()
            if gdconns_degrees[g,q] == 1 and last_edge[g,q] == -1:
                print('ERROR: last_edge is -1 for gadget', g, 'qubit', q)
                print('gadget_data:', gadget_data[:,g])
                print('last edge:', last_edge[g])
                print('gdconns_degrees:', gdconns_degrees[g])
                print('gdconns:', gdconns[g])
                input() 
            if gdconns_degrees[g,q] == 0 and gadget_data[q,g] != 'I' and gdconns_degrees[g].sum() > 0:
                print('ERROR: degree is 0 but gadget_data is not I for gadget', g, 'qubit', q)
                print('gdconns_degrees:', gdconns_degrees[g])
                print('gadget_data:', gadget_data[:,g])
                input()

def check_equal_gates():
    print('Check equal gates')
    found = False
    for op11 in [apply_I, apply_svs, apply_vdg, apply_sdg, apply_sv, apply_vs]:
        for op12 in [apply_I, apply_svs, apply_vdg, apply_sdg, apply_sv, apply_vs]:
            for cnotr1 in [True, False]:
                combination = []
                for op21 in [apply_I, apply_svs, apply_vdg, apply_sdg, apply_sv, apply_vs]:
                    for op22 in [apply_I, apply_svs, apply_vdg, apply_sdg, apply_sv, apply_vs]:
                        for cnotr2 in [True, False]:
                            if op11 == op21 and op12 == op22 and cnotr1 == cnotr2:
                                continue
                            this_is_ok = True
                            for p1 in [(1,0),(1,1),(0,1),(0,0)]:
                                for p2 in [(1,0),(1,1),(0,1),(0,0)]:
                                    p11,p12,_ = apply_cnot(op11(p1)[0], op12(p2)[0])
                                    p21,p22,_ = apply_cnot(op21(p1)[0], op22(p2)[0])
                                    if cnotr1:
                                        p12,p11,_ = apply_cnot(op12(p2)[0], op11(p1)[0])
                                    if cnotr2:
                                        p22,p21,_ = apply_cnot(op22(p2)[0], op21(p1)[0])
                                    if p11 != p21 or p12 != p22:
                                        this_is_ok = False
                                    if op11==apply_I and op12==apply_svs and cnotr1==True and op21==apply_sv and op22==apply_vdg and cnotr2==False:
                                        print('Debug:', p1, p2, '->', p11,p12, p21,p22)
                            if this_is_ok:
                                combination.append((op21.__name__, op22.__name__, cnotr2))
                    if len(combination) > 0:
                        found = True
                        print(op11.__name__, op12.__name__, cnotr1, '->', combination)
    print('Found equivalent gates:', found)

def print_sorted_gd(gadget_data, order=None):
    _, num_qubits, num_gadgets = gadget_data.shape
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
            if gadget_data[0,i,order[j]] == 1 and gadget_data[1,i,order[j]] == 0:
                char = 'X'
            elif gadget_data[0,i,order[j]] == 1 and gadget_data[1,i,order[j]] == 1:
                char = 'Y'
            elif gadget_data[0,i,order[j]] == 0 and gadget_data[1,i,order[j]] == 1:
                char = 'Z'
            else:
                char = ' '
            print(char, end=' ')
        print('')
    print('')

#test_possible_gates()
check_equal_gates()
#for i in range(72):
#    print(get_gates(1<<i))