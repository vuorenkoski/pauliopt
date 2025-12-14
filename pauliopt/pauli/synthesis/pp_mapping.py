import random
import numpy as np
from pauliopt.pauli.pauli_polynomial import PauliPolynomial, Topology, I

# Similar to I-index mapping: https://ieeexplore.ieee.org/abstract/document/10771974
# Differences: how mapping is done after initial physical and logical qubit pair. I-index has more straightforward way: what logical qubit woul
# have most common legs with mapped logical qubit when new logical qubit would be placed as a neighbour of mapped logical qubit.

def pauli_tree_mapping(pp: PauliPolynomial, topo: Topology):
    # This mapping gives weights to edges according to the how many paulis have both these legs
    # Mapping starts from center and heaviest logical qubit (Most legs). Maping continues to logical 
    # qubit which have most common non-I legs with mapped qubit. Connections are made as a tree structure.
    weights = edges_by_I_index(pp)
    mapping, tree = map_qubits_as_tree(weights, pp, topo)
    return mapping, tree

def edges_by_I_index(pp: PauliPolynomial):
    """ Gives edges weight by calculating gadgets having both qubits non-I.
    :param pp: PauliPolynomial to map
    :param debug: If True, print debug information
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
    """ Find qubit having max degree and max distance to the closest edge"""
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



# --- Prevous version

def I_index_mapping(pp: PauliPolynomial, topo: Topology):
    # This mapping gives weghts to edges according to the how many paulis have both these legs
    # Mapping starts from center and heaviest logical qubit (Most legs). Maping continues to logical qubit which have most common non-I legs with mapped qubit.
    # Difference to forest: I-index is the same, but finding optimal mappings after initial is little bit different.
    weights = edges_by_I_index(pp)
#    mapping = map_graphs_by_chains(weights, topo)
    mapping = map_qubits_from_center(weights, pp, topo)
    return mapping, None


def map_qubits_from_center(correlations, pp, topo):
    num_logical_qubits = pp.num_qubits
    num_physical_qubits = topo.num_qubits
    mapped_logical_qubits = []
    mapped_physical_qubits = []
    mapping = [-1 for _ in range(num_logical_qubits)]
    mapping_reverse = [-1 for _ in range(num_physical_qubits)]

    # Search most heavily connected qubit (has most non-I legs)
    heaviest_logical_qubit = find_heaviest_logical_qubit(pp)

    # Map this qubit to the center of graph
    center = center_physical_qubit(topo)
    mapped_logical_qubits.append(heaviest_logical_qubit)
    mapped_physical_qubits.append(center)
    mapping[heaviest_logical_qubit] = center
    mapping_reverse[center] = heaviest_logical_qubit

    while len(mapped_logical_qubits) < num_logical_qubits:
        # find possible next physical qubits by these criteria:
        #  - Non-mapped qubit is neighbour of mapped physical qubit
        #  - Of these select one which has shortest distance of second mapped physical qubit (if this is third)
        #  - Of these select those that have max free degree.
        # Select many if there are qual options
        npq = []
        min_2nd_distance = num_physical_qubits
        max_free_degree = -1
        for nmpq in range(num_physical_qubits):

            # Check that qubiut is not mapped
            if nmpq in mapped_physical_qubits:
                continue

            # Check that qubit is neighbour of mapped physical qubit
            neighbour = False
            for mpq in mapped_physical_qubits:
                if topo.dist(mpq, nmpq) == 1:
                    neighbour = True
                    break
            if not neighbour:
                continue

            # Find shortest distance to other mapped physical qubit
            shortest_distance = num_physical_qubits
            for i in mapped_physical_qubits:       
                if i == mpq:
                    continue
                if topo.dist(i,nmpq) < shortest_distance:
                    shortest_distance = topo.dist(i,nmpq)

            # count connection to non-mapped physical qubits
            free_degree = 0
            for i in range(num_physical_qubits):  
                if topo.dist(nmpq,i) == 1 and i not in mapped_physical_qubits:
                    free_degree += 1

            if shortest_distance < min_2nd_distance:
                min_2nd_distance = shortest_distance
                npq = [nmpq]
                max_free_degree = free_degree
            elif shortest_distance == min_2nd_distance and free_degree > max_free_degree:
                min_2nd_distance = shortest_distance
                npq = [nmpq]
                max_free_degree = free_degree
            elif shortest_distance == min_2nd_distance and free_degree == max_free_degree:
                npq.append(nmpq)

        # select logical qubit having max sum score with neighbouring mapped logical qubits
        max_correlation = -1
        max_logical_qubit = -1
        max_physical_qubit = -1
        for nmpq in npq:
            for nmlq in range(pp.num_qubits):
                if nmlq in mapped_logical_qubits:
                    continue
                correlation_sum = 0
                for mpq in mapped_physical_qubits:
                    if topo.dist(mpq, nmpq) != 1:
                        continue
                    mlq = mapping_reverse[mpq]
                    correlation_sum += correlations[nmlq, mlq] 
                if max_correlation == -1 or correlation_sum > max_correlation:
                    max_correlation = correlation_sum
                    max_logical_qubit = nmlq
                    max_physical_qubit = nmpq

        # Make mapping
        mapped_logical_qubits.append(max_logical_qubit)
        mapped_physical_qubits.append(max_physical_qubit)
        mapping[max_logical_qubit] = max_physical_qubit
        mapping_reverse[max_physical_qubit] = max_logical_qubit
    return mapping

def map_graphs_by_chains(edges, topo):
    """Map logical qubits to physical qubits. First create longest chain of physical qubits, then heaviest chain of logical qubits, and map them together.
    After that map rest of the logical qubits to physical qubits from starting haeviest pair which first qubit is mapped and second is not.
    :param edges: Edge weights as a numpy array
    :param tm: Topology as numpy-matrix
    :return: List of logical qubits mapped to physical qubits
    """
    num_physical_qubits = topo.num_qubits
    num_logical_qubits = len(edges)

    # Create longest chain of physical qubits
    physical_chain = best_connected_physical_chain(topo)

    # Create longest chain of logical qubits trying to maximize edge weights, having same length as physical_chain
    logical_chain = heaviest_chain(edges, len(physical_chain))

    # Start mapping
    mapped_physical_qubits = set()
    mapped_logical_qubits = set()
    mapping = [-1 for x in range(num_logical_qubits)]

    # 1) map initial chains together
    for i in range(min(len(logical_chain),len(physical_chain))):
        mapping[logical_chain[i]] = physical_chain[i]
        mapped_physical_qubits.add(physical_chain[i])
        mapped_logical_qubits.add(logical_chain[i])

    # 2) map rest of the qubits
    while len(mapped_logical_qubits)<num_logical_qubits: # Continue until all qubits are mapped
        max_next = search_max_next(edges, mapped_logical_qubits) # Search non mapped logical qubit attached to mapped logical qubit having max weight 
        
        if max_next == None: # no attached qubits found, have to take random logical qubit
            for i in random_sample(0,num_logical_qubits):
                if i not in mapped_logical_qubits:
                    for j in random_sample(0,num_physical_qubits):  # map this qubit to random physical qubit
                        if j not in mapped_physical_qubits:
                            mapped_logical_qubits.add(i)
                            mapped_physical_qubits.add(j)
                            mapping[i] = j
                            break
                    break
        else: # next logical qubit was found
            mapped_logical_qubits.add(max_next[1])
            for i in random_sample(0,num_physical_qubits):   # map logical qubit to physical qubit wich is not mapped and which is connected to mapped other end of edge
                if topo.dist(mapping[max_next[0]],i) == 1 and i not in mapped_physical_qubits:
                    mapping[max_next[1]] = i
                    mapped_physical_qubits.add(i)
                    break
            if mapping[max_next[1]] == -1: # if no connected physical qubit is found, map to nearest physical qubit
                closest_distance = -1
                selected = -1
                for i in random_sample(0,num_physical_qubits):
                    if i not in mapped_physical_qubits:
                        if closest_distance == -1 or topo.dist(mapping[max_next[0]],i)<closest_distance:
                            closest_distance = topo.dist(mapping[max_next[0]],i)
                            selected = i
                mapping[max_next[1]] = selected
                mapped_physical_qubits.add(selected)
    
    return mapping

def heaviest_chain(edges, n): 
    """Create heavy chain having min (n, num_logical_qubits.
    :param edges: Edge weights as a numpy array
    :param n: Length of the chain
    :return: List of logical qubits in chain
    """
    n = min(n, len(edges))  # Limit chain length to number of logical qubits
    # Find heaviest connected pair
    max_value = -1
    visited = np.zeros((n), dtype=int)
    for i in range(n-1):
        for j in range(i+1,n):
            if edges[i,j] > max_value:
                max_value = edges[i,j]
                start = i
                end = j
    visited[start] = 1
    visited[end] = 1
    chain = [start, end]

    # Extend this pair from start or end by choosing heaviest connection, until n is reached
    for i in range(n-2):
        max_value = -1
        next_qubit = -1
        for i in range(n):
            if i!=start and not visited[i] and edges[min(start,i),max(start,i)] > max_value:
                is_start = True
                next_qubit = i
                max_value = edges[min(start,i),max(start,i)]
            if i!=end and not visited[i] and edges[min(i,end),max(i,end)] > max_value:
                is_start = False
                next_qubit = i
                max_value = edges[min(i,end),max(i,end)]
        if next_qubit == -1:
            continue
        if is_start:
            start = next_qubit
            chain.insert(0, start)
            visited[start] = 1
        else:
            end = next_qubit
            chain.append(end)
            visited[end] = 1
    return chain

def best_connected_physical_chain(topo): 
    """Find physical qubit chain so that qubits are well connected
    :param tm: Topology as numpy-matrix
    :return: List of physical qubits in chain
    """
    n = topo.num_qubits

    # Calculate qubit connections
    connections = np.zeros((n), dtype=int)
    for i in range(n):
        for j in range(n):
            if i!=j and topo.dist(i,j) == 1:
                connections[i] += 1

    # Find best connected qubit-pair
    max_value = -1
    for i in range(n-1):
        for j in range(i+1,n):
            if topo.dist(i,j)==1 and connections[i]+connections[j] > max_value:
                max_value = connections[i]+connections[j]
                start = i
                end = j
    connections[start] = -1
    connections[end] = -1
    chain = [start, end]

    # Grow chain by adding best connected qubits either end of chain until no more connections are found
    while True:
        max_value = -1
        next_qubit = -1
        for i in range(n):
            if i!=start and topo.dist(start,i)==1 and connections[i] > max_value:
                is_start = True
                next_qubit = i
                max_value = connections[i]
            if i!=end and topo.dist(i,end)==1 and connections[i] > max_value:
                is_start = False
                next_qubit = i
                max_value = connections[i]
        if next_qubit == -1:
            break
        if is_start:
            start = next_qubit
            chain.insert(0, start)
        else:
            end = next_qubit
            chain.append(end)
        connections[next_qubit] = -1

    return chain

def search_max_next(edges, mapped_qubits): 
    """Search max weight edge where one qubit is mapped and other non-mapped.
    :param edges: Edge weights as a numpy array
    :param mapped_qubits: List of mapped qubits
    :return: (mapped_qubit, non_mapped_qubit) tuple
    """
    n = len(edges)
    max_value = -1
    for i in range(n-1):
        for j in range(i+1,n):
            if edges[i,j] > max_value:
                if i in mapped_qubits and j not in mapped_qubits:
                    max_value = edges[i,j]
                    max_i = i
                    max_j = j
                    flip = False
                if j in mapped_qubits and i not in mapped_qubits:
                    max_value = edges[i,j]
                    max_i = i
                    max_j = j
                    flip = True

    if max_value == -1: 
        return None
    if flip:
        return (max_j, max_i)
    else:
        return (max_i, max_j)

def random_sample(first, last):
    arr = list(range(first, last))
    random.shuffle(arr)
    return arr

def complete_tree(topo):
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
    correlations = np.zeros((pp.num_qubits, pp.num_qubits), dtype=int)
    for gadget in pp.pauli_gadgets:
        for i in range(pp.num_qubits):
            for j in range(pp.num_qubits):
                if gadget.paulis[i] != I and gadget.paulis[j] != I:
                    correlations[i,j] += 1
    return correlations