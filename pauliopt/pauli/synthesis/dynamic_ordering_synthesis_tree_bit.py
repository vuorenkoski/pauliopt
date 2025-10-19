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

def pauli_polynomial_dynamic_ordering_tree_bit(pp: PauliPolynomial, topo: Topology, tree, print_order=None, debug=False, random_sel=False):
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
    pauligraph_degrees = np.zeros((num_gadgets, num_qubits), dtype=np.int8) # Dynamic matrix represnting degrees of node in steiner trees
    last_edge = np.zeros((num_gadgets, num_qubits), dtype=int) # Dynamic matrix representing last edges in tree branches
    gmatrix = np.zeros((num_gadgets), dtype=np.int64) # Static matrix representing paulis
    for i,gadget in enumerate(pp.pauli_gadgets):
        value = 0
        for j in range(num_qubits):
            last_edge[i,j] = -1
            if gadget.paulis[j] == I:
                value |= (0b00 << (2*j))
            elif gadget.paulis[j] == X:
                value |= (0b01 << (2*j))
            elif gadget.paulis[j] == Y:
                value |= (0b11 << (2*j))
            elif gadget.paulis[j] == Z:
                value |= (0b10 << (2*j))
            else:
                raise ValueError(f'Unknown Pauli in gadget {i}')
        gmatrix[i] = value
        gadget_angles.append(gadget.angle)
        tree_graph = mapping_tree(gmatrix, topo, tree, i)
        create_pauligraph_from_tree_graph(tree_graph, pauligraph_degrees, last_edge, i)
    tree_graph = topology_tree(num_qubits, tree)
    general_data = (gmatrix, num_qubits, gadget_angles, removed_gadgets, pauligraph_degrees, last_edge)
    circ_data = (qc_out, qc_prop)

    node_levels = dict()
    root, tree_children = tree
    node_levels_from_tree(tree_children, root, node_levels)

    debug and print('---------------------Initial gadget data')
    debug and print_sorted_gd(gmatrix, num_qubits,order=print_order)

    # Main loop going through gadgets starting from smallest
    removed_gadgets_num += check_for_singles(general_data, circ_data, perm_gadgets, last_leg)
    debug and print_sorted_gd(gmatrix, num_qubits, order=print_order)
    while removed_gadgets_num < num_gadgets:
        # Randomness 1: there are for example many gadgets having same size and no steiner nodes, how to arrange them?
        next = next_gadget(general_data)
        debug and print('-------Next gadget:', next)
        num_legs = 0
        for j in range(num_qubits):
            if gadget_datam(gmatrix, j, next) != 0b00:
                num_legs += 1
        if num_legs == 0:
            print('ERROR: qubit map has no nodes')
            print('gadget:', next)
            print_sorted_gd(gmatrix, num_qubits, order=print_order)
            print(perm_gadgets)
            input()
        
        # Loop going through qubits in gadget, removing them one by one
        while num_legs > 1:
            # randomness 2: if there are two or more edge+gate combinations having similar match, which one to choose?
            edge, gates, score = next_edge_to_remove(next, general_data)
            debug and print('-------Next edge:', edge, 'gates:', gates, 'gadget', next)
            rgadgets = add_cnot_and_single_qubit_gates(edge, gates, general_data, circ_data, perm_gadgets, last_leg, tree_graph)
            removed_gadgets_num += rgadgets
            if gadget_datam(gmatrix, edge[0], next) == 0b00:
                num_legs -= 1
            debug and print_sorted_gd(gmatrix, num_qubits, order=print_order)
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
    return circ_out, perm_gadgets, permutation, {'pre-cx': pre_cx, 'density': density, 'last_leg': last_leg}

def topology_tree(num_qubits, tree):
    root, tree_children = tree
    edges = []
    for i in range(num_qubits):
        for j in tree_children[i] or []:
            edges.append((i,j))
    G = nx.Graph(edges)
    return G

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

def gadget_datam(gmatrix, qubit, gadget, value = None):
    if value is not None:
        gmatrix[gadget] &= ~(0b11 << (2*qubit))
        gmatrix[gadget] |= (value & 0b11) << (2*qubit)
        return
    return (gmatrix[gadget] & (0b11 << (2*qubit))) >> (2*qubit)

def check_for_singles(general_data, circ_data, perm_gadgets, last_leg):
    """ check if there are gadgets having only single leg and remove them if so."""
    gmatrix, num_qubits, gadget_angles, removed_gadgets, pauligraph_degrees, last_edge = general_data
    num_gadgets = len(gmatrix)
    removed_gadgets_num = 0

    for i in range(num_gadgets):
        if removed_gadgets[i] == 1:
            continue
        x = 0
        for j in range(num_qubits):
            if gadget_datam(gmatrix, j, i) != 0b00:
                x += 1
                qubit = j 
        if x == 1:
            removed_gadgets_num += 1
            remove_single(general_data, circ_data, perm_gadgets, i, qubit)
            last_leg[qubit] += 1
    return removed_gadgets_num


def remove_single(general_data, circ_data, perm_gadgets, gadget_index, qubit):
    """ Remove gadget having single leg indicated by gadget_index."""
    gmatrix, num_qubits, gadget_angles, removed_gadgets, pauligraph_degrees, last_edge = general_data
    qc_out, qc_prop = circ_data
    pauli = gadget_datam(gmatrix, qubit, gadget_index)
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
    gadget_datam(gmatrix, qubit, gadget_index, value=0b00)


def next_edge_to_remove(gadget_index, general_data):
    """Decide what edge to remove next and what gates to apply"""
    # This version primarily chooses gates which minimizes overall effect to the length of chains. 
    # If there are many options having the same score, it secondarily tries to remove/avoid identity gates from the middle of chains.
    gmatrix, num_qubits, gadget_angles, removed_gadgets, pauligraph_degrees, last_edge = general_data
    num_gadgets = len(gmatrix)

    # find different option to remove qubit (edge)
    edge_options = []
    for q in range(num_qubits):
        if pauligraph_degrees[gadget_index, q] == 1:
            edge_options.append((q,int(last_edge[gadget_index,q])))

    # find possible gates for each edge option
    score = dict()
    edge_gates = []
    for qubit0,qubit1 in edge_options:
        pauli0 = gadget_datam(gmatrix, qubit0, gadget_index)
        pauli1 = gadget_datam(gmatrix, qubit1, gadget_index)
        if pauli1 == 0b00: #swap needed
            if pauli0 == 0b11: # We need to change Y to (Y or X): gate I,SV
                gate0 = [apply_I, apply_sv]
            elif pauli0 == 0b01: # We need to change X to (Y or X): gate I, V
                gate0 = [apply_I, apply_v]
            else: # We need to change Z to (Y or X): gate V,SV
                gate0 = [apply_v, apply_sv]
            gate1 = [apply_s, apply_v, apply_vs] # second is I so any gate is good
        else:
            if pauli0 == 0b11: # We need to change Y to Z: gate V
                gate0 = [apply_v]
            elif pauli0 == 0b01: # We need to change X to Z: gate SV
                gate0 = [apply_sv]
            else: # We need to change Z to Z: gate I
                gate0 = [apply_I]
            if pauli1 == 0b11: # We need to change Y to (Y or Z): gate V,VS
                gate1 = [apply_v, apply_vs]
            elif pauli1 == 0b01: # We need to change X to (Y or Z): gate S, VS
                gate1 = [apply_s, apply_vs]
            else: # We need to change Z to (Y or Z): gate S, V
                gate1 = [apply_s, apply_v]
        edge_gates.append((gate0, gate1))

        for g0 in gate0:
            for g1 in gate1:
                score[((qubit0,qubit1),g0,g1)] = 0

    for gadget in range(num_gadgets):
        if removed_gadgets[gadget]:
            continue
        for e in range(len(edge_options)):
            qubit0,qubit1 = edge_options[e]
            pauli0 = int(gadget_datam(gmatrix, qubit0, gadget))
            pauli1 = int(gadget_datam(gmatrix, qubit1, gadget))

            for gate0 in edge_gates[e][0]:
                for gate1 in edge_gates[e][1]:
                    new_pauli0, phase = gate0(pauli0)
                    new_pauli1, phase = gate1(pauli1)
                    new_pauli0, new_pauli1, phase = apply_cnot(new_pauli0, new_pauli1)
                    score_change = get_score(general_data, gadget, qubit0, qubit1, new_pauli0, new_pauli1, pauli0, pauli1)
                    score[(edge_options[e],gate0,gate1)] += score_change
    min_score = None
    for e in range(len(edge_options)):
        for gate0 in edge_gates[e][0]:
            for gate1 in edge_gates[e][1]:
                if min_score is None or score[(edge_options[e],gate0,gate1)] < min_score:
                    min_score = score[(edge_options[e],gate0,gate1)]
                    edge = edge_options[e]
                    gates = (gate0, gate1)
    return edge, gates, min_score

def get_score(general_data, gadget_index, qubit0, qubit1, new_pauli0, new_pauli1, original_pauli0, original_pauli1):
    # 10x more score if chain is extended/shortened
    gmatrix, num_qubits, gadget_angles, removed_gadgets, pauligraph_degrees, last_edge = general_data
    score = 0
    if new_pauli0 != 0b00 and original_pauli0 == 0b00: # addition
        if pauligraph_degrees[gadget_index,qubit0] == 0: # outside of chain, negative
            score = 10
        else:                                            # middle of chain, positive
            score = -1
    if new_pauli1 != 0b00 and original_pauli1 == 0b00: # addition
        if pauligraph_degrees[gadget_index,qubit1] == 0: # outside of chain, negative
            score = 10
        else:                                            # middle of chain, positive
            score = -1
            pass

    if new_pauli0 == 0b00 and original_pauli0 != 0b00: # deletion
        if pauligraph_degrees[gadget_index,qubit0] == 1: # end of chain, positive
            score = -10
        else:                                            # middle of chain, negative
            score = 1
    if new_pauli1 == 0b00 and original_pauli1 != 0b00: # deletion
        if pauligraph_degrees[gadget_index,qubit1] == 1: # end of chain, positive
            score = -10
        else:                                            # middle of chain, negative
            score = 1
    return score

def next_gadget(general_data):
    """ Order non-removed gadgets. Primary sorting is done by number of nodes in steiner tree, secondary sorting by number of steiner nodes."""
    gmatrix, num_qubits, gadget_angles, removed_gadgets, pauligraph_degrees, last_edge = general_data
    num_gadgets = len(gmatrix)
    closest = -1
    num_closest = 0
    distance = num_qubits*2
    for i in range(num_gadgets):
        if removed_gadgets[i]:
            continue
        s_nodes, nodes = steiner_nodes(gmatrix, pauligraph_degrees, i)
        dist = nodes + (s_nodes * 2)
        if dist < distance:
            distance = dist
            closest = i
            num_closest = 1
        elif dist == distance:
            num_closest += 1
    return closest

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


def add_cnot_and_single_qubit_gates(edge, gates, general_data, circ_data, perm_gadgets, last_leg, tree_graph):
    """apply single qubit gates and CNOT to all non-removed gates."""
    gmatrix, num_qubits, gadget_angles, removed_gadgets, pauligraph_degrees, last_edge = general_data
    qc_out, qc_prop = circ_data
    num_gadgets = len(gmatrix)
    gate1, gate2 = gates
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
        pauli1, pauli2 = gadget_datam(gmatrix, qubit1,gadget_index), gadget_datam(gmatrix, qubit2,gadget_index)
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

        gadget_removed = update_pauligraph(general_data, gadget_index, qubit1, qubit2, pauli1, pauli2, tree_graph)
        gadget_datam(gmatrix, qubit1, gadget_index, value=pauli1)
        gadget_datam(gmatrix, qubit2, gadget_index, value=pauli2)
        gadget_angles[gadget_index] *= phase

        if gadget_removed:
            if pauli1 == 0b00:
                remove_single(general_data, circ_data, perm_gadgets, gadget_index, qubit2)
                last_leg[qubit2] += 1
            else:
                remove_single(general_data, circ_data, perm_gadgets, gadget_index, qubit1)
                last_leg[qubit1] += 1
            removed_gadgets_num += 1
            gadget_datam(gmatrix, qubit1, gadget_index, value=0b00)
            gadget_datam(gmatrix, qubit2, gadget_index, value=0b00)
    return removed_gadgets_num

def update_pauligraph(general_data, gadget_index, qubit1, qubit2, pauli1, pauli2, tree_graph):
    gmatrix, num_qubits, gadget_angles, removed_gadgets, pauligraph_degrees, last_edge = general_data

    original_pauli1, original_pauli2 = gadget_datam(gmatrix, qubit1,gadget_index), gadget_datam(gmatrix, qubit2,gadget_index)
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

def mapping_tree(gmatrix, topo, tree, gadget_index):
    root, tree_children = tree
    nodes_removed = []
    edges = []
    remove_nodes_from_leafs(gmatrix, gadget_index, tree_children, root, nodes_removed, edges)
    remove_nodes_from_root(gmatrix, gadget_index, tree_children, root, nodes_removed,edges)
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

def remove_nodes_from_leafs(gmatrix, gadget_index, tree_children, root, nodes_removed, edges):
    if tree_children[root] is None:
        if gadget_datam(gmatrix, root,gadget_index) == 0b00:
            nodes_removed.append(root)
            return True
        else:
            return False
    non_removable_branches = 0
    for node in tree_children[root]:
        removable = remove_nodes_from_leafs(gmatrix, gadget_index, tree_children, node, nodes_removed, edges)
        if not removable:
            non_removable_branches += 1
    if non_removable_branches == 0 and gadget_datam(gmatrix, root,gadget_index) == 0b00:
        nodes_removed.append(root)
        return True 
    return False

def remove_nodes_from_root(gmatrix, gadget_index, tree_children, root, nodes_removed, edges):
    if tree_children[root] is None:
        return
    if gadget_datam(gmatrix, root,gadget_index) != 0b00:
        return
    branches_left = 0
    branch = -1
    for node in tree_children[root]:
        if node not in nodes_removed:
            branches_left += 1
            branch = node
    if branches_left == 1:
        nodes_removed.append(root)
        remove_nodes_from_root(gmatrix, gadget_index, tree_children, branch, nodes_removed, edges)

def steiner_nodes(gmatrix, pauligraph_degrees, gadget_index):
    """ Defines number of steiner nodes and regular nodes of steiner tree (pauligraph data)"""
    num_qubits = pauligraph_degrees.shape[1]
    nodes = 0
    steiner_nodes = 0
    for i in range(num_qubits):
        if gadget_datam(gmatrix, i, gadget_index) != 0b00:
            nodes += 1
        elif pauligraph_degrees[gadget_index, i] > 0:
            steiner_nodes += 1
    return steiner_nodes, nodes


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


def print_sorted_gd(gmatrix, num_qubits, order=None):
    num_gadgets = gmatrix.shape[0]
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
            if gadget_datam(gmatrix, i, order[j]) == 0b01:
                char = 'X'
            elif gadget_datam(gmatrix, i, order[j]) == 0b10:
                char = 'Z'
            elif gadget_datam(gmatrix, i, order[j]) == 0b11:
                char = 'Y'
            elif gadget_datam(gmatrix, i, order[j]) == 0b00:
                char = ' '
            print(char, end=' ')
        print('')
    print('')
