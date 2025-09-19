from pauliopt.pauli.pauli_polynomial import PauliPolynomial, Topology, I
import numpy as np
import random

def I_index_mapping(pp: PauliPolynomial, topo: Topology):
    """Mapping is based on trying to maximize I and non-I pairs in the PauliPolynomial after mapping.
    :param pp: PauliPolynomial to map
    :param topo: Topology to map to
    :return: Edge weights as a numpy array
    """
    tm = create_topology_matrix(topo)
#    weight = attenuation(pp)
    edges = edges_by_I_index(pp)
#    mapping = map_graphs_by_chains(edges,tm)
    mapping = map_graphs_from_center(edges,tm,pp)
    return mapping

def edges_by_I_index(pp: PauliPolynomial):
    """ Gives edges weight by calculating gadgets having both qubits I or non-I.
    :param pp: PauliPolynomial to map
    :param debug: If True, print debug information
    :return: Edge weights as a numpy array
    """
    num_qubits = pp.num_qubits
    edges = np.zeros((num_qubits, num_qubits), dtype=int)

    for q1 in range(num_qubits-1):
        for q2 in range(q1+1,num_qubits):
            I_index = 0
            for i, gadget in enumerate(pp.pauli_gadgets):
                if (gadget.paulis[q1] != I) and (gadget.paulis[q2] != I):
                    I_index += 1
            edges[q1,q2] = I_index
    return edges

def map_graphs_from_center(edges, tm, pp):
    """Map logical qubits to physical qubits. First create center of physical qubits, then heaviest center of logical qubits, and map them together.
    After that map rest of the logical qubits to physical qubits from starting haeviest pair which first qubit is mapped and second is not.
    :param edges: Edge weights as a numpy array
    :param tm: Topology as numpy-matrix
    :return: List of logical qubits mapped to physical qubits
    """
    num_physical_qubits = len(tm)
    num_logical_qubits = len(edges)

    # calculate shortest path topologymtarix with floyd warshall
    apsp = floydWarshall(tm)

    # find center of physical qubits
    physical_center = find_center_of_topology(tm,apsp)

    # find heaviest logical qubit
    logical_center = heaviest_qubit(pp)
    print('Logical center:', logical_center)    
    input()

    # Start mapping
    mapped_physical_qubits = set()
    mapped_logical_qubits = set()
    mapping = [-1 for x in range(num_logical_qubits)]

    mapping[logical_center] = physical_center
    mapped_physical_qubits.add(physical_center)
    mapped_logical_qubits.add(logical_center)

    while len(mapped_logical_qubits)<num_logical_qubits: 
        max_edge_weight = -1
        selected_lq1 = -1
        selected_pq1 = -1
        for lq2 in mapped_logical_qubits:
            pq2 = mapping[lq2]
            for pq1 in range(num_physical_qubits):
                if tm[pq2,pq1] == 0 or pq1 not in mapped_physical_qubits:
                    continue
                for lq1 in range(num_logical_qubits):
                    if lq1 in mapped_logical_qubits:
                        continue
                    if max_edge_weight == -1 or edges[min(lq1,lq2),max(lq1,lq2)] > max_edge_weight:
                        max_edge_weight = edges[min(lq1,lq2),max(lq1,lq2)]
                        selected_lq1 = lq1
                        selected_pq1 = pq1

        mapping[selected_lq1] = selected_pq1
        mapped_physical_qubits.add(selected_pq1)
        mapped_logical_qubits.add(selected_lq1)
        print(mapping)
    return mapping


def find_center_of_topology(tm, apsp):
    leafs = []
    for i in range (len(tm)):
        if np.sum(tm[i,:]) == 1:
            leafs.append(i)
    max_min_distance = -1
    for i in range(len(tm)):
        min_distance = -1
        for j in leafs:
            if i==j:
                dist = 0
            else:
                dist = apsp[i,j]
            if min_distance == -1 or dist < min_distance:
                min_distance = dist
        if max_min_distance == -1 or min_distance > max_min_distance:
            max_min_distance = min_distance
            center = i
    return center
    
def heaviest_qubit(pp):
    num_qubits = pp.num_qubits

    heaviest = -1
    max_weight = -1
    for q in range(num_qubits):
        weight = 0
        for gadget in pp.pauli_gadgets:
            if gadget.paulis[q] != I:
                weight += 1
        if weight > max_weight:
            max_weight = weight
            heaviest = q
    return heaviest

def attenuation(pp):
    attenuation = []
    for gadget in range(len(pp.pauli_gadgets)):
        weight = 0
        for pauli in pp.pauli_gadgets[gadget].paulis:
            if pauli == I:
                weight += 1
        attenuation.append(weight)
    return attenuation

def map_graphs_by_chains(edges, tm):
    """Map logical qubits to physical qubits. First create longest chain of physical qubits, then heaviest chain of logical qubits, and map them together.
    After that map rest of the logical qubits to physical qubits from starting haeviest pair which first qubit is mapped and second is not.
    :param edges: Edge weights as a numpy array
    :param tm: Topology as numpy-matrix
    :return: List of logical qubits mapped to physical qubits
    """
    num_physical_qubits = len(tm)
    num_logical_qubits = len(edges)

    # calculate shortest path topologymtarix with floyd warshall
    apsp = floydWarshall(tm)

    # Create longest chain of physical qubits
    physical_chain = best_connected_physical_chain(tm)

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
        max_next = search_max_next(edges, mapped_logical_qubits) # Search non mapped logical qubit attached to mapped logical qubits 
        
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
            for i in random_sample(0,num_physical_qubits):   # map logical qubit to random connected physical qubits
                if tm[mapping[max_next[0]],i] == 1 and i not in mapped_physical_qubits:
                    mapping[max_next[1]] = i
                    mapped_physical_qubits.add(i)
                    break
            if mapping[max_next[1]] == -1: # if no connected physical qubit is found, map to nearest physical qubit
                closest_distance = -1
                selected = -1
                for i in random_sample(0,num_physical_qubits):
                    if i not in mapped_physical_qubits:
                        if closest_distance == -1 or apsp[mapping[max_next[0]],i]<closest_distance:
                            closest_distance = apsp[mapping[max_next[0]],i]
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

def best_connected_physical_chain(tm): 
    """Find physical qubit chain so that qubits are well connected
    :param tm: Topology as numpy-matrix
    :return: List of physical qubits in chain
    """
    n = len(tm.data)

    # Calculate qubit connections
    connections = np.zeros((n), dtype=int)
    for i in range(n):
        connections[i] = np.sum(tm[i,:])

    # Find best connected qubit-pair
    max_value = -1
    for i in range(n-1):
        for j in range(i+1,n):
            if tm[i,j]==1 and connections[i]+connections[j] > max_value:
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
            if i!=start and tm[start,i]==1 and connections[i] > max_value:
                is_start = True
                next_qubit = i
                max_value = connections[i]
            if i!=end and tm[i,end]==1 and connections[i] > max_value:
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


def create_topology_matrix(topo):
    tm = np.zeros((topo.num_qubits,topo.num_qubits), dtype=int)
    for i in range(topo.num_qubits):
        for j in range(topo.num_qubits):
            if topo.dist(i,j) == 1:
                tm[i,j] = 1
    return tm


def random_sample(first, last):
    arr = list(range(first, last))
    random.shuffle(arr)
    return arr


def floydWarshall(graph):
    g = graph.copy()
    n = len(g)

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if ((g[i][j] == 0 or g[i][j] > (g[i][k] + g[k][j])) and (g[k][j] != 0 and g[i][k] != 0)):
                    g[i][j] = g[i][k] + g[k][j]
    return g
