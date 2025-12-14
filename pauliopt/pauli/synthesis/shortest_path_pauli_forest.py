import networkx as nx
import numpy as np

from pauliopt.clifford.clifford_tableau import CliffordTableau
from pauliopt.gates import CX, Vdg, Sdg
from pauliopt.circuits import Circuit
from pauliopt.pauli.pauli_polynomial import PauliPolynomial
from pauliopt.pauli_strings import I, X, Y, Z
from pauliopt.topologies import Topology
from tests.pauli.utils import verify_equality

# The idea is that algorithm resolves gadgets in order by choosing nearest gadget as next to be resolved.
# Distance of gadgets is measured by number of cnots needed to resolve the gadget. Gadget can be resolved
# by many different paths. A path is composed from steps. In each step cnot is applied in addition to
# single qubit gates. Single qubit gate for control is either I, V or S+V and for target V, S or V+S.
# S = sqrt(Z), V = sqrt(X)
# S = Y->X->Y, V = Y->Z->Y, SV = Z->Y->X->Z, VS = X->Y->Z->X

# Path is formed step by step, so that each step removes one leg from gadget, or appends non-I leg  
# where I is in the middle of non-I legs. From multiple step choices, one is selected which brings
# non-resolved gadgets closer (removes legs in endings, or adds non-I leg to the middle)

# Topology of the target device is fitted to a tree, so we usually all connections of device is not used. 
# This tree must be provided for the algorithm. Every gadget is fitted to this tree and legs are removed 
# from edges of that graph

# Time complexity O(num_gadgets * ((num_gadgets * num_qubits) + num_qubits * num_qubits * num_gadgets))
# =O(num_gadgets^2 * num_qubits^2)

def shortest_path_pauli_forest(pp: PauliPolynomial, topo: Topology, tree, print_order=None, debug=False, random_sel=False):
    num_qubits = pp.num_qubits
    num_gadgets = len(pp.pauli_gadgets)
    qc_out = Circuit(num_qubits)
    qc_prop = []
    perm_gadgets = []

    # Create algorithm datastructures
    last_leg = np.zeros((num_qubits), dtype=int) # Last leg of gadget before removal (for analysis)
    gadget_data = np.zeros((num_qubits,num_gadgets), dtype=np.int8) # Matrix representing current status of paulipolynomial 
    gadget_angles = []
    removed_gadgets = np.zeros((num_gadgets), dtype=np.int8) # Array representing removed gadgets
    gadget_graph_degrees = np.zeros((num_gadgets, num_qubits), dtype=np.int8) # Matrix representing degrees of gadget graph
    gadget_graph_last_edge = np.zeros((num_gadgets, num_qubits), dtype=int) # Matrix representing last edges in gadget graph
    for i,gadget in enumerate(pp.pauli_gadgets):
        for j in range(num_qubits):
            gadget_graph_last_edge[i,j] = -1
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
        create_gadget_graph(gadget_data, tree, gadget_graph_degrees, gadget_graph_last_edge, i)
    device_topology = graph_from_tree(gadget_data, tree)
    tree_topo = Topology(topo.num_qubits, list(device_topology.edges))
    general_data = (gadget_data, gadget_angles, removed_gadgets, gadget_graph_degrees, gadget_graph_last_edge)
    circ_data = (qc_out, qc_prop)
    debug and print('---------------------Initial gadget data')
    debug and print_sorted_gd(gadget_data, order=print_order)

    # Remove possible gadgets having initially one leg
    removed_gadgets_num = check_for_singles(general_data, circ_data, perm_gadgets, last_leg)
    debug and print_sorted_gd(gadget_data, order=print_order)

    while removed_gadgets_num < num_gadgets:
        # Select gadget to be processed next
        next = next_gadget(general_data)
        debug and print('-------Next gadget:', next)
        num_legs = 0
        for j in range(num_qubits):
            if gadget_data[j,next] != 0b00:
                num_legs += 1
        if num_legs == 0:
            print('ERROR: qubit map has no nodes, gadget:', next)
            print_sorted_gd(gadget_data)
            input()
        
        # Loop going through legs in gadget, removing them one by one
        while num_legs > 1:
            # Decide next step in path
            edge, gates = next_leg_to_remove(next, general_data)
            debug and print('-------Next edge:', edge, 'gates:', gates, 'gadget', next)
            # Execute that step
            rgadgets = add_cnot_and_single_qubit_gates(edge, gates, general_data, circ_data, perm_gadgets, last_leg, device_topology)
            removed_gadgets_num += rgadgets
            if gadget_data[edge[0], next] == 0b00:
                num_legs -= 1
            debug and print_sorted_gd(gadget_data, order=print_order)
            debug and input()

    # Post processing

    # do clifford synthesis for the second part of the created circuit
    qc_prop_r = list(reversed(qc_prop))
    ct_prop = CliffordTableau(num_qubits)
    for gate in qc_prop_r:
        ct_prop.append_gate(gate)
    qc_prop_syn, permutation = ct_prop.to_clifford_circuit_perm_row_col(tree_topo, include_swaps=True)

    # Combine circuits
    circ_out = qc_out + qc_prop_syn
    circ_out.final_permutation = qc_prop_syn.final_permutation
    permutation = [circ_out.final_permutation[i] for i in range(pp.num_qubits)]

    # Calculate some analysis data
    pre_cx = 0
    for gate in qc_out.gates:
        if isinstance(gate, CX):
            pre_cx += 1
    qc_prop_syn_count = 0
    for gate in qc_prop_syn.gates:
        if isinstance(gate, CX):
            qc_prop_syn_count += 1
    return circ_out, perm_gadgets, permutation, {'pre-cx': pre_cx, 'tableau-cnot':qc_prop_syn_count}

def graph_from_tree(gadget_data, tree):
    """Create device topology from gadget tree structure."""
    num_qubits, num_gadgets = gadget_data.shape
    root, tree_children = tree
    edges = []
    for i in range(num_qubits):
        if i in tree_children:
            for j in tree_children[i] or []:
                edges.append((i,j))
    G = nx.Graph(edges)
    return G

def create_gadget_graph(gadget_data, tree, gadget_graph_degrees, gadget_graph_last_edge, gadget_index):
    """Create gadget graph_data structures from gadget tree graph."""
    num_qubits = gadget_graph_degrees.shape[1]
    tree_graph = map_gadget_as_tree(gadget_data, tree, gadget_index)
    for j in range(num_qubits):
        if tree_graph.has_node(j):
            gadget_graph_degrees[gadget_index,j] = tree_graph.degree[j]
            if tree_graph.degree[j] == 1:
                gadget_graph_last_edge[gadget_index,j] = list(tree_graph.edges(j))[0][1] # what is the border of edge node
        else:
            gadget_graph_degrees[gadget_index,j] = 0

def check_for_singles(general_data, circ_data, perm_gadgets, last_leg):
    """ check if there are gadgets having only single leg and remove them if so."""
    gadget_data, gadget_angles, removed_gadgets, gadget_graph_degrees, gadget_graph_last_edge = general_data
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
    gadget_data, gadget_angles, removed_gadgets, gadget_graph_degrees, gadget_graph_last_edge = general_data
    qc_out, qc_prop = circ_data
    pauli = gadget_data[qubit,gadget_index]
    removed_gadgets[gadget_index] = 1
    if pauli == 0b01:
        qc_out.s(qubit)
        qc_out.v(qubit)
    elif pauli == 0b11:
        qc_out.v(qubit)
    qc_out.rz(gadget_angles[gadget_index], qubit)
    if pauli == 0b01:
        qc_out.vdg(qubit)
        qc_out.sdg(qubit)
    elif pauli == 0b11:
        qc_out.vdg(qubit)
    perm_gadgets.append(gadget_index)
    gadget_data[qubit,gadget_index] = 0b00


def next_leg_to_remove(gadget_index, general_data):
    """ Select next leg (edge+gate combination) to remove from gadget.
    This can be also filling I in the middle of non-I legs.
    time complexity O(num_gadgets * num_qubits)"""
    gadget_data, gadget_angles, removed_gadgets, gadget_graph_degrees, gadget_graph_last_edge = general_data
    num_qubits, num_gadgets = gadget_data.shape

    # find different option to remove leg (edge in gadget graph)
    edge_options = []
    for q in range(num_qubits):
        if gadget_graph_degrees[gadget_index, q] == 1:
            edge_options.append((q,int(gadget_graph_last_edge[gadget_index,q])))
    if len(edge_options) == 0:
        print('ERROR: no edge options found for gadget:', gadget_index)
        print(gadget_graph_degrees[gadget_index])
        print_sorted_gd(gadget_data)
        input()
    # find possible gates for each edge option
    score = dict()
    edge_gates = []
    for qubit0,qubit1 in edge_options:
        if gadget_data[qubit1,gadget_index] == 0b00: # Filling needed
            if gadget_data[qubit0,gadget_index] == 0b11: # We need to change Y to (Y or X): gate I,SV
                gate0 = [apply_I, apply_sv]
            elif gadget_data[qubit0,gadget_index] == 0b01: # We need to change X to (Y or X): gate I, V
                gate0 = [apply_I, apply_v]
            else: # We need to change Z to (Y or X): gate V,SV
                gate0 = [apply_v, apply_sv]
            gate1 = [apply_s, apply_v, apply_vs] # second is I so any gate is good
        else:
            if gadget_data[qubit0,gadget_index] == 0b11: # We need to change Y to Z: gate V
                gate0 = [apply_v]
            elif gadget_data[qubit0,gadget_index] == 0b01: # We need to change X to Z: gate SV
                gate0 = [apply_sv]
            else: # We need to change Z to Z: gate I
                gate0 = [apply_I]
            if gadget_data[qubit1,gadget_index] == 0b11: # We need to change Y to (Y or Z): gate V,VS
                gate1 = [apply_v, apply_vs]
            elif gadget_data[qubit1,gadget_index] == 0b01: # We need to change X to (Y or Z): gate S, VS
                gate1 = [apply_s, apply_vs]
            else: # We need to change Z to (Y or Z): gate S, V
                gate1 = [apply_s, apply_v]
        edge_gates.append((gate0, gate1))

        for g0 in gate0:
            for g1 in gate1:
                score[((qubit0,qubit1),g0,g1)] = (0,0)

    # Score each edge+gate combination in terms of how much it helps to remove legs in other gadgets
    for gadget in range(num_gadgets):
        if removed_gadgets[gadget]:
            continue
        for e in range(len(edge_options)):
            qubit0,qubit1 = edge_options[e]
            pauli0 = int(gadget_data[qubit0,gadget])
            pauli1 = int(gadget_data[qubit1,gadget])

            for gate0 in edge_gates[e][0]:
                for gate1 in edge_gates[e][1]:
                    new_pauli0, phase = gate0(pauli0)
                    new_pauli1, phase = gate1(pauli1)
                    new_pauli0, new_pauli1, phase = apply_cnot(new_pauli0, new_pauli1)
                    leg_change, middle_I_change = get_score(general_data, gadget, qubit0, qubit1, new_pauli0, new_pauli1, pauli0, pauli1)
                    leg, middle_I = score[(edge_options[e],gate0,gate1)]
                    score[(edge_options[e],gate0,gate1)] = (leg + leg_change, middle_I + middle_I_change)

    # Select combination with minimum score (primary target is leg count, secondary middle I count)
    min_score = None
    for e in range(len(edge_options)):
        for gate0 in edge_gates[e][0]:
            for gate1 in edge_gates[e][1]:
                if min_score is None or score[(edge_options[e],gate0,gate1)][0] < min_score[0]:
                    min_score = score[(edge_options[e],gate0,gate1)]
                    edge = edge_options[e]
                    gates = (gate0, gate1)
                elif score[(edge_options[e],gate0,gate1)][0] == min_score[0]:
                    if score[(edge_options[e],gate0,gate1)][1] < min_score[1]:
                        min_score = score[(edge_options[e],gate0,gate1)]
                        edge = edge_options[e]
                        gates = (gate0, gate1)
    return edge, gates

def get_score(general_data, gadget_index, qubit0, qubit1, new_pauli0, new_pauli1, original_pauli0, original_pauli1):
    """ Get score (leg change, middle I change) for given changes in pauli in given gadget and qubits."""
    gadget_data, gadget_angles, removed_gadgets, gadget_graph_degrees, gadget_graph_last_edge = general_data
    leg_change = 0
    middle_I_change = 0
    if new_pauli0 != 0b00 and original_pauli0 == 0b00: # addition
        if gadget_graph_degrees[gadget_index,qubit0] == 0: # outside of chain, negative
            leg_change = 1
        else:                                            # middle of chain, positive
            middle_I_change = -1
    if new_pauli1 != 0b00 and original_pauli1 == 0b00: # addition
        if gadget_graph_degrees[gadget_index,qubit1] == 0: # outside of chain, negative
            leg_change = 1
        else:                                            # middle of chain, positive
            middle_I_change = -1
            pass

    if new_pauli0 == 0b00 and original_pauli0 != 0b00: # deletion
        if gadget_graph_degrees[gadget_index,qubit0] == 1: # end of chain, positive
            leg_change = -1
        else:                                            # middle of chain, negative
            middle_I_change = 1
    if new_pauli1 == 0b00 and original_pauli1 != 0b00: # deletion
        if gadget_graph_degrees[gadget_index,qubit1] == 1: # end of chain, positive
            leg_change = -1
        else:                                            # middle of chain, negative
            middle_I_change = 1
    return leg_change, middle_I_change

def next_gadget(general_data):
    """ Select closest gadget.
        time complexity O(num_gadgets * num_qubits)"""
    gadget_data, gadget_angles, removed_gadgets, gadget_graph_degrees, gadget_graph_last_edge = general_data
    num_qubits, num_gadgets = gadget_data.shape
    closest = -1
    num_closest = 0
    distance = num_qubits*2
    for i in range(num_gadgets):
        if removed_gadgets[i]:
            continue
        I_nodes, nodes = gadget_graph_size(gadget_data, gadget_graph_degrees, i)
        dist = nodes + (I_nodes * 2)
        if dist < distance:
            distance = dist
            closest = i
            num_closest = 1
        elif dist == distance:
            num_closest += 1
    return closest

def add_cnot_and_single_qubit_gates(edge, gates, general_data, circ_data, perm_gadgets, last_leg, device_topology):
    """apply single qubit gates and CNOT to all non-removed gates.
    time complexity O(num_gadgets)"""
    gadget_data, gadget_angles, removed_gadgets, gadget_graph_degrees, gadget_graph_last_edge = general_data
    qc_out, qc_prop = circ_data
    num_qubits, num_gadgets = gadget_data.shape

    gate1,gate2 = gates
    qubit1,qubit2 = edge
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

    # Propagate gates through all remaining gadgets
    removed_gadgets_num = 0
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

        gadget_removed = update_gadget_graph(general_data, gadget_index, qubit1, qubit2, pauli1, pauli2, device_topology)
        gadget_data[qubit1,gadget_index], gadget_data[qubit2,gadget_index] = pauli1, pauli2
        gadget_angles[gadget_index] *= phase

        if gadget_removed:
            if pauli1 == 0b00:
                remove_single(general_data, circ_data, perm_gadgets, gadget_index, qubit2)
                last_leg[qubit2] += 1
            else:
                remove_single(general_data, circ_data, perm_gadgets, gadget_index, qubit1)
                last_leg[qubit1] += 1
            removed_gadgets_num += 1
            gadget_data[qubit1,gadget_index], gadget_data[qubit2,gadget_index] = 0b00, 0b00
    return removed_gadgets_num

def update_gadget_graph(general_data, gadget_index, qubit1, qubit2, pauli1, pauli2, device_topology):
    gadget_data, gadget_angles, removed_gadgets, gadget_graph_degrees, gadget_graph_last_edge = general_data

    original_pauli1, original_pauli2 = gadget_data[qubit1,gadget_index], gadget_data[qubit2,gadget_index]
    if pauli1 != 0b00 and original_pauli1 == 0b00: # addition
        if gadget_graph_degrees[gadget_index,qubit1] == 0: # outside of chain, negative
            gadget_graph_degrees[gadget_index,qubit1] += 1
            gadget_graph_degrees[gadget_index,qubit2] += 1
            gadget_graph_last_edge[gadget_index,qubit1] = qubit2
        else:                                            # middle of chain, positive
            pass
    if pauli2 != 0b00 and original_pauli2 == 0b00: # addition
        if gadget_graph_degrees[gadget_index,qubit2] == 0: # outside of chain, negative
            gadget_graph_degrees[gadget_index,qubit1] += 1
            gadget_graph_degrees[gadget_index,qubit2] += 1
            gadget_graph_last_edge[gadget_index,qubit2] = qubit1
        else:                                            # middle of chain, positive
            pass

    if pauli1 == 0b00 and original_pauli1 != 0b00: # deletion
        if gadget_graph_degrees[gadget_index,qubit1] == 1: # end of chain, positive
            gadget_graph_degrees[gadget_index,qubit1] -= 1
            gadget_graph_degrees[gadget_index,qubit2] -= 1
            if gadget_graph_degrees[gadget_index,qubit2] == 1:
                gadget_graph_last_edge[gadget_index,qubit2] = find_edge(gadget_index, qubit2, gadget_graph_degrees, device_topology)
        else:                                            # middle of chain, negative
            pass
    if pauli2 == 0b00 and original_pauli2 != 0b00: # deletion
        if gadget_graph_degrees[gadget_index,qubit2] == 1: # end of chain, positive
            gadget_graph_degrees[gadget_index,qubit1] -= 1
            gadget_graph_degrees[gadget_index,qubit2] -= 1
            if gadget_graph_degrees[gadget_index,qubit1] == 1:
                gadget_graph_last_edge[gadget_index,qubit1] = find_edge(gadget_index, qubit1, gadget_graph_degrees, device_topology)
        else:                                            # middle of chain, negative
            pass
    return (gadget_graph_degrees[gadget_index,qubit1] == 0) and (gadget_graph_degrees[gadget_index,qubit2] == 0)

def find_edge(gadget_index, qubit, gadget_graph_degrees, device_topology):
    """ Find last edge connected to the qubit in gadget graph."""
    for neighbor in device_topology.neighbors(qubit):
        if gadget_graph_degrees[gadget_index, neighbor] != 0:
            return neighbor
    print('ERROR: could not find last edge')
    print('gadget index:', gadget_index)
    print('qubit:', qubit)
    print('gadget degrees:', gadget_graph_degrees[gadget_index])
    input()
    return -1

def map_gadget_as_tree(gadget_data, tree, gadget_index):
    """ Map gadget as tree graph by removing nodes which are not part of the gadget."""
    """ Time complexity O(n)"""
    root, tree_children = tree
    nodes_to_remove = [] # Nodes which can be removed from tree
    remove_nodes_from_leafs(gadget_data, gadget_index, tree_children, root, nodes_to_remove)
    remove_nodes_from_root(gadget_data, gadget_index, tree_children, root, nodes_to_remove)
    edges = []
    list_edges(tree_children, root, nodes_to_remove, edges)
    return nx.Graph(edges)

def remove_nodes_from_leafs(gadget_data, gadget_index, tree_children, root, nodes_to_remove):
    """ Recursive check from leafs to root which branches can be removed."""
    """ Time complexity O(n)"""
    # If there is no children and node is I, it can be removed
    if tree_children[root] is None:
        if gadget_data[root,gadget_index] == 0b00:
            nodes_to_remove.append(root)
            return True
        else:
            return False
    # Check all children, if all are removable and node is I, it can be removed
    non_removable_branches = 0
    for node in tree_children[root]:
        removable = remove_nodes_from_leafs(gadget_data, gadget_index, tree_children, node, nodes_to_remove)
        if not removable:
            non_removable_branches += 1
    if non_removable_branches == 0 and gadget_data[root,gadget_index] == 0b00:
        nodes_to_remove.append(root)
        return True
    return False

def remove_nodes_from_root(gadget_data, gadget_index, tree_children, root, nodes_to_remove):
    """ Recursively check if root node can be removed due to fact that it has only single branch."""
    """ Time complexity O(n)"""
    # If there is no children and node is I do nothing
    if tree_children[root] is None or gadget_data[root,gadget_index] != 0b00:
        return
    
    # For all children of a node, check if only single branch is left after removals. If so, remove this node too
    branches_left = 0
    branch = -1
    for node in tree_children[root]:
        if node not in nodes_to_remove:
            branches_left += 1
            branch = node
    if branches_left == 1:
        nodes_to_remove.append(root)
        remove_nodes_from_root(gadget_data, gadget_index, tree_children, branch, nodes_to_remove)

def list_edges(tree_children, root, nodes_to_remove, edges):
    """ List edges from tree excluding removed nodes."""
    """ Time complexity O(n)"""
    if tree_children[root] is None:
        return
    for node in tree_children[root]:
        if node not in nodes_to_remove:
            if root not in nodes_to_remove:
                edges.append((root, node))
        list_edges(tree_children, node, nodes_to_remove, edges)

def gadget_graph_size(gadget_data, gadget_graph_degrees, gadget_index):
    """ Defines number of middle I nodes and regular nodes of gadget graph."""
    num_qubits = gadget_data.shape[0]
    nodes = 0
    I_nodes = 0
    for i in range(num_qubits):
        if gadget_data[i, gadget_index] != 0b00:
            nodes += 1
        elif gadget_graph_degrees[gadget_index, i] > 0:
            I_nodes += 1
    return I_nodes, nodes

def apply_v(p): # z,z xor x
    phase = 1
    if p == 0b10:
        phase = -1
    return (0b10 & p) | ((0b01 & p) ^ (p >> 1)), phase

def apply_s(p): # z xor x, x
    phase = 1
    if p == 0b11:
        phase = -1
    return 0b01 & p | ((0b10 & p) ^ ((p & 0b01) << 1)), phase

def apply_I(p):
    return p, 1

def apply_sv(p):
    pauli,phase1 = apply_s(p)
    pauli,phase2 = apply_v(pauli)
    return pauli, phase1*phase2

def apply_vs(p):
    pauli,phase1 = apply_v(p)
    pauli,phase2 = apply_s(pauli)
    return pauli, phase1*phase2

def apply_svs(p):
    pauli,phase1 = apply_s(p)
    pauli,phase2 = apply_v(pauli)
    pauli,phase3 = apply_s(pauli)
    return pauli, phase1*phase2*phase3

def apply_cnot(p1, p2):
    phase = 1
    if (p1 == 0b01 and p2 == 0b10) or (p1 == 0b11 and p2 == 0b11):
        phase = -1
    pauli1 = (p1 & 0b01) | ((p1 ^ p2) & 0b10)
    pauli2 = ((p1 ^ p2) & 0b01) | (p2 & 0b10)
    return pauli1, pauli2, phase

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
