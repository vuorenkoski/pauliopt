import numpy as np
from pauliopt.pauli.pauli_polynomial import PauliPolynomial, Topology, I

def pauli_tree_mapping(pp: PauliPolynomial, topo: Topology):
    """ Algorithm for initial mapping giving weights to edges between qubits according to the how many paulis have both these legs. 
    :param pp: PauliPolynomial to map
    :param topo: Topology of the physical qubits
    :return: mapping from logical qubits to physical qubits, and tree structure used in the mapping
    """
    weights = edges_by_I_index(pp)
    mapping, tree = map_qubits_as_tree(weights, pp, topo)
    return mapping, tree

def edges_by_I_index(pp: PauliPolynomial):
    """ Gives edges weight by calculating gadgets having both qubits non-I.
    :param pp: PauliPolynomial to map
    :return: Edge weights as a numpy array
    """
    num_qubits = pp.num_qubits
    edges = np.zeros((num_qubits, num_qubits), dtype=int)

    for q1 in range(num_qubits):
        for q2 in range(num_qubits):
            I_index = 0
            for gadget in pp.pauli_gadgets:
                if (gadget.paulis[q1] != I) and (gadget.paulis[q2] != I):
                    I_index += 1
            edges[q1,q2] = I_index
    return edges

def map_qubits_as_tree(correlations, pp, topo):
    """ Mapping starts from center and heaviest logical qubit (Most legs). Maping continues to logical qubit which have most common non-I legs with mapped qubit. Connections are made as a tree structure. 
    :param correlations: Correlation matrix between logical qubits
    :param pp: PauliPolynomial to map
    :param topo: Topology of the physical qubits
    :return: mapping from logical qubits to physical qubits, and tree structure of physical qubits used in the mapping.
    """

    num_logical_qubits = pp.num_qubits
    num_physical_qubits = topo.num_qubits
    mapped_logical_qubits = []
    mapped_physical_qubits = []
    mapped_physical_qubit_depth = {}
    mapping = [-1 for _ in range(num_logical_qubits)]
    mapping_reverse = [-1 for _ in range(num_physical_qubits)]
    edges = []

    # Search most heavily connected qubit (has most non-I legs)
    heaviest_logical_qubit = find_heaviest_logical_qubit(pp)

    # Map this qubit to the center of graph
    center = center_physical_qubit(topo)
    mapped_logical_qubits.append(heaviest_logical_qubit)
    mapped_physical_qubits.append(center)
    mapping[heaviest_logical_qubit] = center
    mapping_reverse[center] = heaviest_logical_qubit
    mapped_physical_qubit_depth[center] = 0

    while len(mapped_logical_qubits) < num_logical_qubits:
        # find possible next physical qubits by these criteria:
        #  - Non-mapped qubit is neighbour of mapped physical qubit
        #  - Of these select ones which has smallest leaf depth (distance to root)
        #  - Of these select those that have max free degree.
        # Select many if there are qual options
        npq = []
        min_leaf_depth = num_physical_qubits
        max_free_degree = -1
        for nmpq in range(num_physical_qubits):

            # Check that qubit is not mapped
            if nmpq in mapped_physical_qubits:
                continue

            # Check that qubit is neighbour of mapped physical qubit
            neighbour = -1
            for mpq in mapped_physical_qubits:
                if topo.dist(mpq, nmpq) == 1:
                    neighbour = mpq
                    break
            if neighbour == -1:
                continue

            # Leaf depth
            leaf_depth = mapped_physical_qubit_depth[neighbour]

            # count connection to non-mapped physical qubits
            free_degree = 0
            for i in range(num_physical_qubits):  
                if topo.dist(nmpq,i) == 1 and i not in mapped_physical_qubits:
                    free_degree += 1

            # compare options
            if leaf_depth < min_leaf_depth:
                min_leaf_depth = leaf_depth
                npq = [nmpq]
                max_free_degree = free_degree
            elif leaf_depth == min_leaf_depth and free_degree > max_free_degree:
                npq = [nmpq]
                min_leaf_depth = leaf_depth
                max_free_degree = free_degree
            elif leaf_depth == min_leaf_depth and free_degree == max_free_degree:
                npq.append(nmpq)

        # select logical qubit having max score with neighbouring mapped logical qubit
        max_correlation = -1
        max_logical_qubit_pair = -1
        max_physical_qubit_pair = -1
        for nmpq in npq:
            for nmlq in range(pp.num_qubits):
                if nmlq in mapped_logical_qubits:
                    continue
                correlation = 0
                for mpq in mapped_physical_qubits:
                    if topo.dist(mpq, nmpq) != 1:
                        continue
                    mlq = mapping_reverse[mpq]
                    correlation = correlations[nmlq, mlq] 
                    if max_correlation == -1 or correlation > max_correlation:
                        max_correlation = correlation
                        max_logical_qubit_pair = (nmlq, mlq)
                        max_physical_qubit_pair = (nmpq, mpq)

        # Make mapping
        mapped_logical_qubits.append(max_logical_qubit_pair[0])
        mapped_physical_qubits.append(max_physical_qubit_pair[0])
        mapping[max_logical_qubit_pair[0]] = max_physical_qubit_pair[0]
        mapping_reverse[max_physical_qubit_pair[0]] = max_logical_qubit_pair[0]
        edges.append(max_physical_qubit_pair)
        mapped_physical_qubit_depth[max_physical_qubit_pair[0]] = 1 + mapped_physical_qubit_depth[max_physical_qubit_pair[1]]
    
    root = center
    tree_children = {}
    for edge in edges:
        if edge[1] in tree_children:
            tree_children[edge[1]].append(edge[0])
        else:
            tree_children[edge[1]] = [edge[0]]

    for pq in mapped_physical_qubits:
        if pq not in tree_children:
            tree_children[pq] = None

    tree = (root, tree_children)
    return mapping, tree

def center_physical_qubit(topo):
    """ Find qubit having max degree and max distance to the closest edge
    :param topo: Topology of the physical qubits
    :return: Index of the cetner physical qubit
    """
    num_physical_qubits = topo.num_qubits

    # Find group of qubits having maximum degree, and group of qubits having degree of 1
    max_degree = 0
    min_degree = -1
    max_degree_qubits = []
    min_degree_qubits = []
    one_degree_qubits = []
    for i in range(num_physical_qubits):
        degree = 0
        for j in range(num_physical_qubits):
            if i != j and topo.dist(i,j) == 1:
                degree += 1
        if degree > max_degree:
            max_degree = degree
            max_degree_qubits = [i]
        elif degree == max_degree:
            max_degree_qubits.append(i)
        if degree == 1:
            one_degree_qubits.append(i)
        if min_degree == -1 or degree < min_degree:
            min_degree = degree
            min_degree_qubits = [i]
        elif degree == min_degree:
            min_degree_qubits.append(i)
    
    # find qubit which distance to closest degree 1 qubit is greatest
    max_distance_to_edge = -1
    max_distance_qubit = -1
    for q1 in max_degree_qubits:
        min_distance_to_edge = -1
        for q2 in min_degree_qubits:
            dist = topo.dist(q1, q2)
            if min_distance_to_edge == -1 or dist < min_distance_to_edge:
                min_distance_to_edge = dist
        if min_distance_to_edge > max_distance_to_edge:
            max_distance_to_edge = min_distance_to_edge
            max_distance_qubit = q1
    return max_distance_qubit

def find_heaviest_logical_qubit(pp):
    """ Find logical qubit having most connections.
    :param pp: PauliPolynomial to map
    :return: Index of heaviest logical qubit.
    """

    max_conn = 0
    heaviest_logical_qubit = -1
    for i in range(pp.num_qubits):
        conn = 0
        for gadget in pp.pauli_gadgets:
            if gadget[i] != I:
                conn += 1
        if conn > max_conn:
            max_conn = conn
            heaviest_logical_qubit = i
    return heaviest_logical_qubit


def complete_tree(topo):
    """ Map all physical qubits to tree strucutre where root is the center physical qubit. 
    :param topo: Topology of the physical qubits
    :return: tree structure of physical qubits containing all physical qubits.
    """

    num_physical_qubits = topo.num_qubits
    mapped_physical_qubits = []
    mapped_physical_qubit_depth = {}
    edges = []

    # Map this qubit to the center of graph
    center = center_physical_qubit(topo)
    mapped_physical_qubits.append(center)
    mapped_physical_qubit_depth[center] = 0

    while len(mapped_physical_qubits) < num_physical_qubits:
        pair = None
        min_leaf_depth = num_physical_qubits
        max_free_degree = -1
        for nmpq in range(num_physical_qubits):

            # Check that qubit is not mapped
            if nmpq in mapped_physical_qubits:
                continue

            # Check that qubit is neighbour of mapped physical qubit
            neighbour = -1
            for mpq in mapped_physical_qubits:
                if topo.dist(mpq, nmpq) == 1:
                    neighbour = mpq
                    break
            if neighbour == -1:
                continue

            # Leaf depth
            leaf_depth = mapped_physical_qubit_depth[neighbour]

            # count connection to non-mapped physical qubits
            free_degree = 0
            for i in range(num_physical_qubits):  
                if topo.dist(nmpq,i) == 1 and i not in mapped_physical_qubits:
                    free_degree += 1

            # compare options
            if leaf_depth < min_leaf_depth:
                min_leaf_depth = leaf_depth
                pair = (neighbour, nmpq)
                max_free_degree = free_degree
            elif leaf_depth == min_leaf_depth and free_degree > max_free_degree:
                pair = (neighbour, nmpq)
                min_leaf_depth = leaf_depth
                max_free_degree = free_degree
    
        # Make mapping
        mapped_physical_qubits.append(pair[1])
        edges.append(pair)
        mapped_physical_qubit_depth[pair[1]] = 1 + mapped_physical_qubit_depth[pair[0]]

    root = center
    tree_children = {}
    for edge in edges:
        if edge[0] in tree_children:
            tree_children[edge[0]].append(edge[1])
        else:
            tree_children[edge[0]] = [edge[1]]

    for pq in mapped_physical_qubits:
        if pq not in tree_children:
            tree_children[pq] = None

    tree = (root, tree_children)
    return tree

def qubit_correlations(pp):
    """ Measure correlations between logical qubits. Correlations represents number of gadgets having both qubits non-I.
    :param pp: PauliPolynomial to map
    :return: Matrix represeting correlations.
    """

    correlations = np.zeros((pp.num_qubits, pp.num_qubits), dtype=int)
    for gadget in pp.pauli_gadgets:
        for i in range(pp.num_qubits):
            for j in range(pp.num_qubits):
                if gadget.paulis[i] != I and gadget.paulis[j] != I:
                    correlations[i,j] += 1
    return correlations