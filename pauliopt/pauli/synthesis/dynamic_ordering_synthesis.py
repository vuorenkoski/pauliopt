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

def pauli_polynomial_dynamic_ordering(pp: PauliPolynomial, topo: Topology, print_order=None, debug=False, random_sel=False):
    num_qubits = pp.num_qubits
    num_gadgets = len(pp.pauli_gadgets)
    removed_gadgets_num = 0
    gadget_angles = []
    removed_gadgets = np.zeros((num_gadgets), dtype=np.int8)
    qc_out = Circuit(num_qubits)
    qc_prop = []
    perm_gadgets = []
    if random_sel:
        random.seed()

    # Create datastructures
    gadget_data = np.zeros((num_qubits,num_gadgets), dtype=('U1')) # Dynamic matrix representing paulis
    gdconns = np.zeros((num_gadgets,num_qubits,num_qubits), dtype=np.int8) # Dynamic matrix represnting steiner trees
    gdconns_degrees = np.zeros((num_gadgets, num_qubits), dtype=np.int8) # Dynamic matrix represnting degrees of node in steiner trees
    last_edge = np.zeros((num_gadgets, num_qubits), dtype=int) # Dynamic matrix representing last edges in tree branches
    for i,gadget in enumerate(pp.pauli_gadgets):
        for j in range(num_qubits):
            last_edge[i,j] = -1
            if gadget.paulis[j] == I:
                gadget_data[j,i] = 'I'
            elif gadget.paulis[j] == X:
                gadget_data[j,i] = 'X'
            elif gadget.paulis[j] == Y:
                gadget_data[j,i] = 'Y'
            elif gadget.paulis[j] == Z:
                gadget_data[j,i] = 'Z'
            else:
                raise ValueError(f'Unknown Pauli {gadget_data[j,i]} in gadget {i}')
        gadget_angles.append(gadget.angle)
        qubit_map = map_gadget(gadget_data, topo, i)
        update_gdconns_from_qubit_map(qubit_map, gdconns, gdconns_degrees, last_edge, i)
    gate_combinations = {} # Immutable dictionary of possible gates for each pair of paulis and targets
    for p1 in ['X', 'Y', 'Z', 'I']:
        for p2 in ['X', 'Y', 'Z', 'I']:
            for target0 in [False, True]: # Is target to have I or not after gates in qubit0
                for target1 in [False, True]: # Is target to have I or not after gates in qubit1
                    options = possible_gates((p1, p2), target0, target1)
                    gate_combinations[(p1 + p2, target0, target1)] = options
    general_data = (gadget_data, gadget_angles, removed_gadgets, gdconns, gdconns_degrees, last_edge)
    circ_data = (qc_out, qc_prop)

    # Main loop going through gadgets starting from smallest
    removed_gadgets_num += check_for_singles(general_data, circ_data, perm_gadgets)
    while removed_gadgets_num < num_gadgets:
        debug and print_sorted_gd(gadget_data, order = print_order)
        # Randomness 1: there are for example many gadgets having same size and no steiner nodes, how to arrange them?
        # !!! However, algorithm benefits if there is similar order from step to step
        order = order_gadgets(general_data, random_sel=False)
        debug and print('\nGadget order:', order)
        next = order[0][0]
        qubit_map = make_map_from_gdconns(gdconns, gdconns_degrees, next)
        if qubit_map.number_of_nodes() == 0:
            print('ERROR: qubit map has no nodes')
            input()
        
        # Loop going through qubits in gadget, removing them one by one
        while qubit_map.number_of_nodes() > 1:
            # If this is uncommented, algorithm improves little bit but time increases a lot
            # order = order_gadgets(general_data, random_sel=random_sel)

            # randomness 2: if there are two or more edges having really good match in terms of next gadgets (removes middle I, shortens gadget etc)
            edge, gates = next_edge_to_remove(qubit_map, next, order, general_data, gate_combinations, random_sel=random_sel)
            debug and print('-------Next edge to remove:', edge, 'gates:', gates)

            removed_gadgets_num += add_cnot_and_single_qubit_gates(edge, gates, general_data, circ_data, perm_gadgets)
            if gadget_data[edge[0], next] == 'I':
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
    num_qubits, num_gadgets = gadget_data.shape
    removed_gadgets_num = 0

    for i in range(num_gadgets):
        if removed_gadgets[i] == 1:
            continue
        x = 0
        for j in range(num_qubits):
            if gadget_data[j,i] != 'I':
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
    pauli = gadget_data[qubit,gadget_index]
    removed_gadgets[gadget_index] = 1
    if pauli == 'X':
        qc_out.h(qubit)
    elif pauli == 'Y':
        qc_out.v(qubit)
    qc_out.rz(gadget_angles[gadget_index], qubit)
    if pauli == 'X':
        qc_out.h(qubit)
    elif pauli == 'Y':
        qc_out.vdg(qubit)
    perm_gadgets.append(gadget_index)
    gadget_data[qubit,gadget_index] = 'I'

def next_edge_to_remove(qubit_map, gadget_index, order, general_data, gate_combinations, random_sel=False):
    """Decide what edge to remove next and what gates to apply"""
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
        if gadget_data[qubit1,gadget_index] == 'I': #swap needed
            options = gate_combinations[(gadget_data[qubit0,gadget_index]+gadget_data[qubit1,gadget_index],False,False)]
            edge_gates.append(options)
        else:
            options = gate_combinations[(gadget_data[qubit0,gadget_index]+gadget_data[qubit1,gadget_index],True,False)]
            edge_gates.append(options)
 #       print('Edge:', (qubit0, qubit1), 'gates:', get_gates(options))

    if len(order) == 1: # Only one gadget left, choose any
        return edge_options[0], get_gates(edge_gates[0])[0]
    g=1

    # Loops through gadgets and possible edges (qubit pairs). Checks which qubit pair gate combination (two possible single qubit gates and CNOT) is
    # best considering next gadgets (trying to shorten chains or removing Is from the middle of chains).
    # For each gadget edge options are evaluated. If part of the edges have suitable gate combinations, they continue to next round, other are dropped.
    # Gate combinations are limited each round.
    while g < len(order):
        gadget = order[g][0]
        g += 1
        edge_good_match = []
        all_non_suitable = True

        for e in range(len(edge_options)):
            if not edge_options_left[e]:
                edge_good_match.append(False)
                continue
            good_match = False
            qubit0,qubit1 = edge_options[e]
            gates0 = edge_gates[e]
            qubit0_degree = gdconns_degrees[gadget,qubit0]
            qubit1_degree = gdconns_degrees[gadget,qubit1]
            legs = gadget_data[qubit0,gadget] + gadget_data[qubit1,gadget]

            # Pair does not touch branch/chain. All options are good
            if qubit0_degree == 0 and qubit1_degree == 0:
                intersection = gates0
                good_match = True

            # Pair is last pair in a brach in this gadget also. Try turn first qubit to I
            if qubit0_degree == 1 and qubit1_degree > 0:
                intersection = gates0 & gate_combinations[(legs,True,False)]
                if intersection > 0:
                    good_match = True
                    edge_excellent[e] += 1

            # Same situation but reversed
            if not good_match and qubit0_degree > 0 and qubit1_degree == 1:
                intersection = gates0 & gate_combinations[(legs,False,True)]
                if intersection > 0:
                    good_match = True
                    edge_excellent[e] += 1

            # Pair is touches last qubit of a branch. Try not to extneded branch
            if not good_match and qubit0_degree == 1 and qubit1_degree == 0:
                intersection = gates0 & gate_combinations[(legs,False,True)]
                good_match = intersection > 0

            # Same situation but reversed
            if not good_match and qubit0_degree == 0 and qubit1_degree == 1:
                intersection = gates0 & gate_combinations[(legs,True, False)]
                good_match = intersection > 0

            # Pair is in the middle of branch. Avoid I:s
            if not good_match and qubit0_degree > 1 and qubit1_degree > 1:
                intersection = gates0 & gate_combinations[(legs,False,False)]
                if intersection > 0:
                    good_match = True
                    if gadget_data[qubit0,gadget] == 'I' or gadget_data[qubit1,gadget] == 'I':
                        edge_excellent[e] += 1

            if good_match:
                edge_gates[e] = intersection
                all_non_suitable = False
            edge_good_match.append(good_match)

        # If all edges are not good matches, continue next gadget. So no edge is above others in terms of priority
        if all_non_suitable:
            continue

        # remove edges which are not good matches from the list
        for e in range(len(edge_options)):
            if edge_options_left[e] and not edge_good_match[e]:
                edge_options_left[e] = False
                good_edges_left -= 1
                good_edge = e
    
        # If there is is only one edge option and only one gate option left, return it
        if good_edges_left == 1:
            gates = get_gates(edge_gates[good_edge])
            if len(gates) == 1:
                return edge_options[good_edge], gates[0]

    # Choose only those edges which have max edge_excellent
    max_excellent = 0
    for e in range(len(edge_options_left)):
        if edge_options_left[e] and edge_excellent[e] > max_excellent:
            max_excellent = edge_excellent[e]
    for e in range(len(edge_options_left)):
        if edge_options_left[e] and edge_excellent[e] < max_excellent:
            edge_options_left[e] = False
            good_edges_left -= 1

    selection = 0
    if random_sel and good_edges_left > 1:
        selection = random.randint(0,good_edges_left-1)
    for e in range(len(edge_options_left)):
        if edge_options_left[e] == 1:
            if selection>0:
                selection -= 1
                continue
            edge = edge_options[e]
            gates = get_gates(edge_gates[e])
            selection = 0
            if random_sel:
                selection = random.randint(0,len(gates)-1)
            return edge, gates[selection]

    print('ERROR: no edge found in next_edge_to_remove') 
    input()

def order_gadgets(general_data, random_sel=False):
    """ Order non-removed gadgets. Primary sorting is done by number of nodes in steiner tree, secondary sorting by number of steiner nodes."""
    gadget_data, gadget_angles, removed_gadgets, gdconns, gdconns_degrees, last_edge = general_data
    num_qubits, num_gadgets = gadget_data.shape
    order = []
    for i in range(num_gadgets):
        if removed_gadgets[i]:
            continue
        s_nodes, nodes = steiner_nodes(gadget_data, gdconns_degrees, i)
        r = random.randrange(num_gadgets * 100)
        order.append((i, s_nodes, nodes+s_nodes, r))
    if random_sel:
        order.sort(key=lambda x: (x[2], x[1], x[3]))
    else:
        order.sort(key=lambda x: (x[2], x[1], x[0]))
    return order

def add_cnot_and_single_qubit_gates(edge, gates, general_data, circ_data, perm_gadgets):
    """apply single qubit gates and CNOT to all non-removed gates."""
    gadget_data, gadget_angles, removed_gadgets, gdconns, gdconns_degrees, last_edge = general_data
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
        if gate == apply_h:
            qc_out.h(qubit)
            qc_prop.append(H(qubit))
        elif gate == apply_vdg:
            qc_out.v(qubit)
            qc_prop.append(Vdg(qubit))
        elif gate == apply_sdg:
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
        if pauli1 == 'I' and pauli2 == 'I':
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

        # Update gdconns and gdconns_degrees
        gadget_removed = update_gdconns(general_data, gadget_index, qubit1, qubit2, pauli1, pauli2)
        if gadget_removed:
            if pauli1 == 'I':
                remove_single(general_data, circ_data, perm_gadgets, gadget_index, qubit2)
            else:
                remove_single(general_data, circ_data, perm_gadgets, gadget_index, qubit1)
            removed_gadgets_num += 1
            gadget_data[qubit1,gadget_index], gadget_data[qubit2,gadget_index] = 'I', 'I'
#        check_cdconns_integrity(gdconns, gdconns_degrees, gadget_data, last_edge)
    return removed_gadgets_num

def update_gdconns(general_data, gadget_index, qubit1, qubit2, pauli1, pauli2):
    """ Update gdconns and gdconns_degrees for gadget. Loop through all edge pairs which are in the edge of branch. Check is there changes for those."""
    gadget_data, gadget_angles, removed_gadgets, gdconns, gdconns_degrees, last_edge = general_data
    num_qubits, num_gadgets = gadget_data.shape
    gadget_removed = False
    for q1 in range(num_qubits):
        # If qubit is not in the end of branch/chain
        if (q1!=qubit1 or q1!=qubit2) and gdconns_degrees[gadget_index,q1] != 1:
            continue
        q2 = last_edge[gadget_index,q1]

        # There are only two neighbouring qubits left which matches for cnot
        if gdconns_degrees[gadget_index,q2] == 1 and ((q1 == qubit1 and q2 == qubit2) or (q1 == qubit2 and q2 == qubit1)): # This is final pair
            if pauli1 == 'I' or pauli2 == 'I':
                remove_connection(general_data, gadget_index, qubit1, qubit2)
                gadget_removed = True
                break

        # this gadget has exactly same edge as the original one
        elif q1 == qubit1 and q2 == qubit2: 
            if pauli1 == 'I': # Chain shortens
                remove_connection(general_data, gadget_index, qubit1, qubit2)
                last_edge[gadget_index,qubit1] = -1
                gadget_removed = follow_chain_until_not_I(general_data, gadget_index, qubit2, pauli2)

        # This gadget has same edge but reversed
        elif q1 == qubit2 and q2 == qubit1: 
            if pauli2 == 'I': # Chain shortens
                remove_connection(general_data, gadget_index, qubit1, qubit2)
                last_edge[gadget_index,qubit2] = -1
                gadget_removed = follow_chain_until_not_I(general_data, gadget_index, qubit1, pauli1)

        # edge is not and last edge in this gadget, but q1 is the first qubit of cnot
        elif q1 == qubit1:
            if pauli1 == 'I' and gdconns_degrees[gadget_index,qubit1] == 1: # chain shortens
                remove_connection(general_data, gadget_index, qubit1, q2)
                last_edge[gadget_index,qubit1] = -1
                gadget_removed = follow_chain_until_not_I(general_data, gadget_index, q2, gadget_data[q2,gadget_index])
            elif pauli2 != 'I' and gdconns_degrees[gadget_index,qubit2] == 0: # Branch extends towars qubit2
                add_connection(general_data, gadget_index, qubit1, qubit2)
                last_edge[gadget_index,qubit2] = qubit1
                last_edge[gadget_index,qubit1] = -1

        # Edge is not the last edge in this gadget but q1 is the second qubit of cnot
        elif q1 == qubit2:
            if pauli2 == 'I' and gdconns_degrees[gadget_index,qubit2] == 1: # Branch shortens
                remove_connection(general_data, gadget_index, qubit2, q2)
                last_edge[gadget_index,qubit2] = -1
                gadget_removed = follow_chain_until_not_I(general_data, gadget_index, q2, gadget_data[q2,gadget_index])
            elif pauli1 != 'I' and gdconns_degrees[gadget_index,qubit1] == 0: # Branch extends
                add_connection(general_data, gadget_index, qubit1, qubit2)
                last_edge[gadget_index,qubit1] = qubit2
                last_edge[gadget_index,qubit2] = -1

        # Edge is not the last edge in this gadget but q2 is the second qubit of cnot, new branch can emerge
        elif q2 == qubit2:
            if pauli1 != 'I' and gdconns_degrees[gadget_index,qubit1] == 0: # New branch
                add_connection(general_data, gadget_index, qubit1, qubit2)
                last_edge[gadget_index,qubit1] = qubit2

        # Edge is not the last edge in this gadget but q2 is the first qubit of cnot
        elif q2 == qubit1:
            if pauli2 != 'I' and gdconns_degrees[gadget_index,qubit2] == 0: # New branch
                add_connection(general_data, gadget_index, qubit1, qubit2)
                last_edge[gadget_index,qubit2] = qubit1

    # If one of the qubits of the edge is in the midddle of chain
    # and other qubit is not in any branch and turns from I to pauli: new branch
    if gdconns_degrees[gadget_index,qubit1] == 0 and gdconns_degrees[gadget_index,qubit2] > 1 and pauli1 != 'I': # New branch
        add_connection(general_data, gadget_index, qubit1, qubit2)
        last_edge[gadget_index,qubit1] = qubit2
    elif gdconns_degrees[gadget_index,qubit2] == 0 and gdconns_degrees[gadget_index,qubit1] > 1 and pauli2 != 'I': # New branch
        add_connection(general_data, gadget_index, qubit1, qubit2)
        last_edge[gadget_index,qubit2] = qubit1
    return gadget_removed

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
    if pauli != 'I':
        last_edge[gadget_index,qubit] = qubit2
        return False

    # pauli is I, we must remove connection
    remove_connection(general_data, gadget_index, qubit, qubit2)

    # If next qubit2 has degree 0 after removing connection, it means that this the only leg left
    if gdconns_degrees[gadget_index,qubit2] == 0:
        return True
    else: #chain continues beyond qubit2
        return follow_chain_until_not_I(general_data, gadget_index, qubit2, gadget_data[qubit2,gadget_index])


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
    num_qubits, num_gadgets = gadget_data.shape
    nodes = []
    for i in range(num_qubits):
        if gadget_data[i,gadget_index] != 'I':
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
    num_qubits = gadget_data.shape[0]
    nodes = 0
    steiner_nodes = 0
    for i in range(num_qubits):
        if gadget_data[i, gadget_index] != 'I':
            nodes += 1
        elif gdconns_degrees[gadget_index, i] > 0:
            steiner_nodes += 1
    return steiner_nodes, nodes


def apply_h(p):
    if p == 'X':
        return 'Z', 1
    elif p == 'Z':
        return 'X', 1
    elif p == 'Y':
        return 'Y', -1
    elif p == 'I':
        return 'I', 1
    else:
        raise ValueError("Invalid Pauli: {}".format(p))

def apply_vdg(p):
    if p == 'X':
        return 'X', 1
    elif p == 'Z':
        return 'Y', -1
    elif p == 'Y':
        return 'Z', 1
    elif p == 'I':
        return 'I', 1
    else:
        raise ValueError("Invalid Pauli: {}".format(p))

def apply_sdg(p):
    if p == 'X':
        return 'Y', 1
    elif p == 'Z':
        return 'Z', 1
    elif p == 'Y':
        return 'X', -1
    elif p == 'I':
        return 'I', 1
    else:
        raise ValueError("Invalid Pauli: {}".format(p))

def apply_cnot(p1, p2):
    # change I:  XX<->XI, YX<->YI, ZY<->IY, ZZ<->IZ
    # others:    XY<->YZ, XZ<->YY
    # no effect: ZX<->ZX, II<->II, ZI<->ZI, IX<->IX

    if p1 == 'X' and p2 == 'X':
        return 'X', 'I', 1
    elif p1 == 'X' and p2 == 'Y':
        return 'Y', 'Z', 1
    elif p1 == 'X' and p2 == 'Z':
        return 'Y', 'Y', -1
    elif p1 == 'X' and p2 == 'I':
        return 'X', 'X', 1

    elif p1 == 'Y' and p2 == 'X':
        return 'Y', 'I', 1
    elif p1 == 'Y' and p2 == 'Y':
        return 'X', 'Z', -1
    elif p1 == 'Y' and p2 == 'Z':
        return 'X', 'Y', 1
    elif p1 == 'Y' and p2 == 'I':
        return 'Y', 'X', 1

    elif p1 == 'Z' and p2 == 'X':
        return 'Z', 'X', 1
    elif p1 == 'Z' and p2 == 'Y':
        return 'I', 'Y', 1
    elif p1 == 'Z' and p2 == 'Z':
        return 'I', 'Z', 1
    elif p1 == 'Z' and p2 == 'I':
        return 'Z', 'I', 1

    elif p1 == 'I' and p2 == 'X':
        return 'I', 'X', 1
    elif p1 == 'I' and p2 == 'Y':
        return 'Z', 'Y', 1
    elif p1 == 'I' and p2 == 'Z':
        return 'Z', 'Z', 1
    elif p1 == 'I' and p2 == 'I':
        return 'I', 'I', 1
    else:
        raise ValueError("Invalid Pauli combination: {}, {}".format(p1, p2))


def possible_gates(paulis,target0, target1):
    """Return possible gates for given pair of paulis and targets to have or have not I.
    Possible gates include possible single qubit gate for qubit1, single qubit gate for qubit2 and cnot direction.
    :params paulis: tuple of two chars, e.g. ('X', 'Y') or ('I', 'Z')
    :params target0: True if qubit0 should have I, False if it should not have I
    :params target1: True if qubit1 should have I, False if it should not have I
    :returns: 32-bit integer coding possible gate combinations."""
    gates_none = np.zeros((2,4,4), dtype=object) # cnot reversed: (False, True), q0 (None, apply_vdg, apply_sdg, apply_h), q1 (None, apply_vdg, apply_sdg, apply_h)
    convert_to_X = {'X': [0,1], 'Y': [2], 'Z': [3]}
    convert_to_Y = {'X': [2], 'Y': [0,3], 'Z': [1]}
    convert_to_Z = {'X': [3], 'Y': [1], 'Z': [0,2]}

    if target0 and target1 and paulis[0] == 'I' and paulis[1] == 'I':   #current and target is II
        paulis_options = np.ones((2,4,4), dtype=object)  # all gates
    elif paulis[0] == 'I' and paulis[1] == 'I':                             #current is ??, target is II
        paulis_options = gates_none
    elif target0 and target1:  
        paulis_options = gates_none
    elif target0 and not target1 and paulis[0] != 'I' and paulis[1] == 'I': # Swap needed, not possible
        paulis_options = gates_none
    elif not target0 and target1 and paulis[0] == 'I' and paulis[1] != 'I': # Swap needed, not possible
        paulis_options = gates_none
    elif target0 and not target1:
        paulis_options = gates_none
        if paulis[0] == 'I':
            for i in range(4):
                paulis_options[np.ix_([0],[i],convert_to_X[paulis[1]])] = 1
                paulis_options[np.ix_([1],[i],convert_to_Z[paulis[1]])] = 1 #1q-h
        else:
            paulis_options[np.ix_([1], convert_to_X[paulis[0]], convert_to_X[paulis[1]])] = 1 # XX
            paulis_options[np.ix_([1], convert_to_X[paulis[0]], convert_to_Y[paulis[1]])] = 1 # XY
            paulis_options[np.ix_([0], convert_to_Z[paulis[0]], convert_to_Z[paulis[1]])] = 1 # ZZ
            paulis_options[np.ix_([0], convert_to_Z[paulis[0]], convert_to_Y[paulis[1]])] = 1 # YZ
    elif not target0 and target1:
        paulis_options = gates_none
        if paulis[1] == 'I':
            for i in range(4):
                paulis_options[np.ix_([1],convert_to_X[paulis[0]],[i])] = 1
                paulis_options[np.ix_([0],convert_to_Z[paulis[0]],[i])] = 1 #1q-h
        else:
            paulis_options[np.ix_([0], convert_to_X[paulis[0]], convert_to_X[paulis[1]])] = 1 # XX
            paulis_options[np.ix_([0], convert_to_Y[paulis[0]], convert_to_X[paulis[1]])] = 1 # YX
            paulis_options[np.ix_([1], convert_to_Z[paulis[0]], convert_to_Z[paulis[1]])] = 1 # ZZ
            paulis_options[np.ix_([1], convert_to_Y[paulis[0]], convert_to_Z[paulis[1]])] = 1 # YZ

    elif not target0 and not target1:
        paulis_options = gates_none.copy()
        if paulis[0] != 'I' and paulis[1] != 'I':   # XY<->YZ, XZ<->YY, ZX<->ZX:   XY,XZ,   ZX,   YZ, YY
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
        if paulis[0] == 'I' and paulis[1] != 'I':   # XI, YI, IY, IZ
            for i in range(4):
                paulis_options[np.ix_([0],[i],convert_to_Y[paulis[1]])] = 1
                paulis_options[np.ix_([0],[i],convert_to_Z[paulis[1]])] = 1
                paulis_options[np.ix_([1],[i],convert_to_X[paulis[1]])] = 1
                paulis_options[np.ix_([1],[i],convert_to_Y[paulis[1]])] = 1
        if paulis[0] != 'I' and paulis[1] == 'I':
            for i in range(4):
                paulis_options[np.ix_([1],convert_to_Y[paulis[0]],[i])] = 1
                paulis_options[np.ix_([1],convert_to_Z[paulis[0]],[i])] = 1
                paulis_options[np.ix_([0],convert_to_X[paulis[0]],[i])] = 1
                paulis_options[np.ix_([0],convert_to_Y[paulis[0]],[i])] = 1
    else:
        print('XXXX Should not happen')

    options = 0
    for i, cnot_reversed in enumerate([False, True]):
        for j, first_qubit in enumerate([None, apply_vdg, apply_sdg, apply_h]):
            for k, second_qubit in enumerate([None, apply_vdg, apply_sdg, apply_h]):
                if paulis_options[i,j,k] == 1:
                    options += 1
                options <<= 1
    return options

def get_gates(gate_set):
    """Return gate combinations for given gate set indicated by 32-bit integer.
    :params gate_set: 32-bit integer coding possible gate combinations.
    :returns: list of tuples (first_qubit_gate, second_qubit_gate, cnot direction).
    """
    gates = []
    gate = 1 << 32
    for i, cnot_reversed in enumerate([False, True]):
        for j, first_qubit in enumerate([None, apply_vdg, apply_sdg, apply_h]):
            for k, second_qubit in enumerate([None, apply_vdg, apply_sdg, apply_h]):
                if gate & gate_set > 1:
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
    for p1 in ['X', 'Y', 'Z', 'I']:
        for p2 in ['X', 'Y', 'Z', 'I']:
            for target0 in [False, True]:
                for target1 in [False, True]:
                    options = possible_gates((p1, p2), target0, target1)
                    options_test = []
                    for cnot_reversed in [True, False]:
                        for first_gate in [None, apply_vdg, apply_sdg, apply_h]:
                            for second_gate in [None, apply_vdg, apply_sdg, apply_h]:
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
                                if target0 == (pauli1 == 'I') and target1 == (pauli2 == 'I'):
                                    options_test.append((first_gate, second_gate, cnot_reversed))
    
                    if len(options) != len(options_test):
                        print('ERROR: different number of options', len(options), len(options_test))
                        print('p1:', p1, 'p2:', p2, 'target0:', target0, 'target1:', target1)
                        print('options:', options)
                        print('options_test:', options_test)
                        input()
                    while len(options) > 0:
                        option = options.pop()
                        if option not in options_test:
                            print('ERROR: option not in options_test', option)
                            print('p1:', p1, 'p2:', p2, 'target0:', target0, 'target1:', target1)
                            print('options:', options)
                            print('options_test:', options_test)
                            input()

def check_cdconns_integrity(gdconns, gdconns_degrees, gadget_data, last_edge):
    """ Check that gdconns_degrees and gdconns are consistent, and that last_edge is correct."""
    num_qubits, num_gadgets = gadget_data.shape
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
            char = ' '
            if gadget_data[i,order[j]] != 'I':
                char = gadget_data[i,order[j]]
            print(char, end=' ')
        print('')
    print('')

