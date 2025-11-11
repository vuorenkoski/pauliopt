import random, math, json
import numpy as np
from pauliopt.pauli.pauli_gadget import PauliGadget
from pauliopt.pauli.pauli_polynomial import PauliPolynomial, I, Z, X, Y
from pauliopt.gates import CX
from pauliopt.pauli.pauli_gadget import PPhase
from pauliopt.pauli.synthesis.pp_mapping import qubit_correlations
from pauliopt.utils import pi
from pauliopt.topologies import Topology
import networkx as nx

def permute_with_mapping(mapping, pp, num_physical_qubits):
    """Permute the PauliPolynomial with the mapping"""
    if pp.num_qubits > num_physical_qubits:
        raise ValueError(f"Number of logical qubits {pp.num_qubits} cannot be greater than number of physical qubits {num_physical_qubits}")

    permuted_pp = PauliPolynomial(num_physical_qubits)
    for gadget in pp.pauli_gadgets:
        paulis_p = [I for _ in range(num_physical_qubits)] # If more physical qubits than logical qubits, fill with I
        for i,pauli in enumerate(gadget.paulis):
            paulis_p[mapping[i]] = pauli
        gadget_p = PauliGadget(gadget.angle, paulis_p)
        permuted_pp.pauli_gadgets.append(gadget_p)
    return permuted_pp

def map_topology(mapping, topo):
    """Create a qubit map from the mapping and topology which includes only used qubits"""
    num_logical_qubits = len(mapping)
    edges = []
    for i in range(num_logical_qubits-1):
        for j in range(i,num_logical_qubits):
            if topo.dist(mapping[i], mapping[j]) == 1:
                edges.append((i,j))
    new_topo = Topology(num_logical_qubits, edges)
    return new_topo

def map_tree(mapping, tree):
    if tree is None:
        return None
    root, tree_childrens = tree
    reverse_mapping = {}
    for lq,pq in enumerate(mapping):
        reverse_mapping[pq] = lq
    new_root = reverse_mapping[root]
    new_childrens = {}
    for lq,pq in enumerate(mapping):
        children = tree_childrens[pq]
        if children is None:
            new_childrens[lq] = None
        else:
            new_childrens[lq] = [reverse_mapping[pq] for pq in children]
    return new_root, new_childrens

def extend_gadgets(pp, topo):
    """Extend the PauliPolynomial gadgets to match the topology"""
    if pp.num_qubits == topo.num_qubits:
        return pp
    elif pp.num_qubits > topo.num_qubits:
        raise ValueError(f"Number of logical qubits {pp.num_qubits} cannot be greater than number of physical qubits {topo.num_qubits}")

    extended_pp = PauliPolynomial(topo.num_qubits)
    for gadget in pp.pauli_gadgets:
        paulis_p = [I for _ in range(topo.num_qubits)]
        for i, pauli in enumerate(gadget.paulis):
            paulis_p[i] = pauli
        gadget_p = PauliGadget(gadget.angle, paulis_p)
        extended_pp.pauli_gadgets.append(gadget_p)
    return extended_pp

def random_mapping(topo):
    m = []
    for i in range(topo.num_qubits):
        m.append(i)
    random.shuffle(m)
    return m

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

def create_random_phase_gadget(num_qubits, min_legs, max_legs, allowed_angles, allowed_legs=None, empty_qubits=0):
    if allowed_legs is None:
        allowed_legs = [X, Y, Z]
    angle = random.choice(allowed_angles)
    nr_legs = random.randint(min_legs, max_legs)
    legs = random.choices(
        [i for i in range(num_qubits-empty_qubits)], k=nr_legs)
    phase_gadget = [I for _ in range(num_qubits)]
    for leg in legs:
        phase_gadget[leg] = random.choice(allowed_legs)
    return PPhase(angle) @ phase_gadget


def create_random_pauli_polynomial(num_qubits: int, num_gadgets: int, min_legs=None, max_legs=None, allowed_angles=None, seed=None, empty_qubits=0, allowed_legs=[X, Y, Z]):
    if min_legs is None:
        min_legs = 1
    if max_legs is None:
        max_legs = num_qubits - empty_qubits
    if allowed_angles is None:
        allowed_angles = [pi, pi / 2, pi / 4, pi / 8, pi / 16]

    if seed is not None:
        random.seed(seed)
    pp = PauliPolynomial(num_qubits)
    for _ in range(num_gadgets):
        pp >>= create_random_phase_gadget(num_qubits, min_legs, max_legs, allowed_angles, empty_qubits=empty_qubits, allowed_legs=allowed_legs)

    return pp

def create_complete_pauli_polynomial_r(num_qubits: int, allowed_legs=[X, Y, Z]):
    if num_qubits == 1:
        pstrings = []
        pstrings.append([I])
        for leg in allowed_legs:
            pstrings.append([leg])
        return pstrings
    else:
        pstrings = []
        smaller_pp = create_complete_pauli_polynomial_r(num_qubits - 1, allowed_legs)
        for pstring in smaller_pp:
            pstrings.append(pstring + [I])
            for leg in allowed_legs:
                pstrings.append(pstring + [leg])
        return pstrings

def create_complete_pauli_polynomial(num_qubits: int, allowed_legs=[X, Y, Z]):
    allowed_angles = [pi, pi / 2, pi / 4, pi / 8, pi / 16]
    pps = create_complete_pauli_polynomial_r(num_qubits, allowed_legs)
    pp = PauliPolynomial(num_qubits)
    for pstring in pps:
        if pstring.count(I) < num_qubits-1:
            angle = random.choice(allowed_angles)
            gadget = PauliGadget(angle, pstring)
            pp.pauli_gadgets.append(gadget)
    return pp

def find_square_dimensions(n):
    s = int(math.sqrt(n))
    if s * s == n:
        l = k = s
        return l, k
    lower_n = n - 1
    upper_n = n + 1
    while True:
        s = int(math.sqrt(lower_n))
        if s * s == lower_n:
            l = k = s
            return l, k

        s = int(math.sqrt(upper_n))
        if s * s == upper_n:
            l = k = s
            return l, k
        lower_n -= 1
        upper_n += 1

def cnot_count(circ):
    count = 0
    for gate in circ.gates:
        if isinstance(gate, CX):
            count += 1
    return count

def cnot_depth(circ):
    q_depth = [0 for _ in range(circ.n_qubits)]
    for gate in circ.gates:
        if isinstance(gate, CX):
            max_depth = max(q_depth[gate.control], q_depth[gate.target])
            q_depth[gate.control] = max_depth + 1
            q_depth[gate.target] = max_depth + 1
    depth = 0
    for i in range(circ.n_qubits):
        if q_depth[i] > depth:
            depth = q_depth[i]
    return depth

def cnot_count_density(circ):
    density = np.zeros((circ.n_qubits, circ.n_qubits), dtype=int)
    for gate in circ.gates:
        if isinstance(gate, CX):
            density[gate.control, gate.target] += 1
    return density  

def print_pp(pp, order=None):
    num_qubits = pp.num_qubits
    num_gadgets = len(pp.pauli_gadgets)

    if order is None:
        order = [i for i in range(num_gadgets)]

    print(' ', end=' ')
    for i in order:
        print(int(i / 10), end=' ')
    print('\n ', end=' ')
    for i in order:
        print(i % 10, end=' ')
    print('')
    for i in range(num_qubits):
        print(i % 10, end=' ')
        for j in range(num_gadgets):
            g = order[j]
            if pp.pauli_gadgets[g][i] == X:
                print('X', end=' ')
            elif pp.pauli_gadgets[g][i] == Y:
                print('Y', end=' ')
            elif pp.pauli_gadgets[g][i] == Z:
                print('Z', end=' ')
            else:
                print(' ', end=' ')
        print('')
    print('')


def aggregate_data(df, method1, method2):
    df2 = df.drop(['n_rep','num_qubits','pre-cx','cx_depth'], axis=1)
    df_1 = df2.loc[(df['mapping'] == 'random') & (df['method'] == method1)].drop(['mapping', 'method'], axis=1).reset_index(drop=True).groupby(['n_gadgets']).mean().round(1).reset_index().rename(columns={'cx':'m1','time':'m1 (ms)'})  
    df_1m = df2.loc[(df['mapping'] == 'algorithm') & (df['method'] == method1)].drop(['mapping', 'method'], axis=1).reset_index(drop=True).groupby(['n_gadgets']).mean().round(1).reset_index().rename(columns={'cx':'m1+mapping', 'time':'m1m (ms)'})  
    df_2 = df2.loc[(df['mapping'] == 'random') & (df['method'] == method2)].drop(['mapping', 'method'], axis=1).reset_index(drop=True).groupby(['n_gadgets']).mean().round(1).reset_index().rename(columns={'cx':'m2','time':'m2 (ms)'})  
    df_2m = df2.loc[(df['mapping'] == 'algorithm') & (df['method'] == method2)].drop(['mapping', 'method'], axis=1).reset_index(drop=True).groupby(['n_gadgets']).mean().round(1).reset_index().rename(columns={'cx':'m2+mapping', 'time':'m2m (ms)'})  
    df3 = df_2.merge(df_2m, on=['n_gadgets'], how='left')
    df3 = df3.merge(df_1, on=['n_gadgets'], how='left')
    df3 = df3.merge(df_1m, on=['n_gadgets'], how='left')
    df3['m2m/m2%'] = np.round(((df3['m2+mapping'] / df3['m2']) - 1)*100,1)
    df3['m1m/m1m'] = np.round(((df3['m1+mapping'] / df3['m1']) - 1)*100,1)
    df3['m1/m2%'] = np.round(((df3['m1'] / df3['m2']) - 1)*100,1)
    df3['m1m/m2m%'] = np.round(((df3['m1+mapping'] / df3['m2+mapping']) - 1)*100,1)
    df3['m1m/m2m time%'] = np.round(((df3['m1m (ms)'] / df3['m2m (ms)']) - 1)*100,1)
    df3 = df3.rename(columns={'n_gadgets': 'gadgets'})
    df3 = df3.set_index('gadgets')
    return df3

def aggregate_data_depth(df, method1, method2):
    df2 = df.drop(['n_rep','num_qubits','pre-cx','cx'], axis=1)
    df_1 = df2.loc[(df['mapping'] == 'random') & (df['method'] == method1)].drop(['mapping', 'method'], axis=1).reset_index(drop=True).groupby(['n_gadgets']).mean().round(1).reset_index().rename(columns={'cx_depth':'m1','time':'m1 (ms)'})  
    df_1m = df2.loc[(df['mapping'] == 'algorithm') & (df['method'] == method1)].drop(['mapping', 'method'], axis=1).reset_index(drop=True).groupby(['n_gadgets']).mean().round(1).reset_index().rename(columns={'cx_depth':'m1+mapping', 'time':'m1m (ms)'})  
    df_2 = df2.loc[(df['mapping'] == 'random') & (df['method'] == method2)].drop(['mapping', 'method'], axis=1).reset_index(drop=True).groupby(['n_gadgets']).mean().round(1).reset_index().rename(columns={'cx_depth':'m2','time':'m2 (ms)'})  
    df_2m = df2.loc[(df['mapping'] == 'algorithm') & (df['method'] == method2)].drop(['mapping', 'method'], axis=1).reset_index(drop=True).groupby(['n_gadgets']).mean().round(1).reset_index().rename(columns={'cx_depth':'m2+mapping', 'time':'m2m (ms)'})  
    df3 = df_2.merge(df_2m, on=['n_gadgets'], how='left')
    df3 = df3.merge(df_1, on=['n_gadgets'], how='left')
    df3 = df3.merge(df_1m, on=['n_gadgets'], how='left')
    df3['m2m/m2%'] = np.round(((df3['m2+mapping'] / df3['m2']) - 1)*100,1)
    df3['m1m/m1%'] = np.round(((df3['m1+mapping'] / df3['m1']) - 1)*100,1)
    df3['m1/m2%'] = np.round(((df3['m1'] / df3['m2']) - 1)*100,1)
    df3['m1m/m2m%'] = np.round(((df3['m1+mapping'] / df3['m2+mapping']) - 1)*100,1)
    df3['m1m/m2m time%'] = np.round(((df3['m1m (ms)'] / df3['m2m (ms)']) - 1)*100,1)
    df3 = df3.rename(columns={'n_gadgets': 'gadgets'})
    df3 = df3.set_index('gadgets')
    return df3


def ibm_backend(backend_name):
    if backend_name not in ['kolkata', 'mumbai', 'lima', 'belem', 'quito', 'guadalupe', 'jakarta', 'manila', 'hanoi', 
                            'algiers', 'lagos', 'nairobi', 'cairo', 'auckland', 'perth', 'peekskill', 'ithaca', 'kyiv', 
                            'prague', 'sherbrooke', 'brisbane', 'seattle', 'nazcav', 'cusco']:
        raise ValueError(f"Unknown IBM backend: {backend_name}")
    with open("backends_2023.json", "r") as f:
        backends = json.load(f)
    backend = None
    for b in backends:
        if b['name'] == 'ibmq_'+backend_name or b['name'] == 'ibm_'+backend_name:
            backend = b
    if backend is None:
        raise ValueError(f'Unknown backend: {backend_name}')
    couplings = backend['couplingMap']
    num_qubits = backend['qubits']
    topo = Topology(num_qubits, couplings)
    return topo

def get_topo(topo_name, num_qubits=9):
    if topo_name == 'line':
        return Topology.line(num_qubits)
    elif topo_name == 'complete':
        return Topology.complete(num_qubits)
    elif topo_name == 'cycle':
        return Topology.cycle(num_qubits)
    elif topo_name == 'grid':
        if num_qubits == 6:
            return Topology.grid(2, 3)
        elif num_qubits == 8:
            return Topology.grid(2, 4)
        else:
            n_rows, n_cols = find_square_dimensions(num_qubits)
            return Topology.grid(n_rows, n_cols)
    return ibm_backend(topo_name)

def steiner_tree_analysis(pp, topo):
    steiner_nodes = 0
    steiner_nodesx = 0
    steiner_nodesz = 0
    broken_chains = 0
    doubles = 0
    for gadget in pp.pauli_gadgets:
        nodes = []
        znodes = []
        xnodes = []
        for j in range(len(gadget)):
            if gadget[j] != I:
                nodes.append(j)
            if gadget[j] == X or gadget[j] == Y:
                xnodes.append(j)
            if gadget[j] == Z or gadget[j] == Y:
                znodes.append(j)
        steiner_stree = nx.algorithms.approximation.steinertree.steiner_tree(topo.to_nx, nodes)
        I_nodes = len(steiner_stree.nodes) - len(nodes)
        steiner_nodes += I_nodes
        if len(xnodes)>0:
            steiner_streex = nx.algorithms.approximation.steinertree.steiner_tree(topo.to_nx, xnodes)
            Ix_nodes = len(steiner_streex.nodes) - len(xnodes)
            steiner_nodesx += Ix_nodes
        if len(znodes)>0:
            steiner_streez = nx.algorithms.approximation.steinertree.steiner_tree(topo.to_nx, znodes)
            Iz_nodes = len(steiner_streez.nodes) - len(znodes)
            steiner_nodesz += Iz_nodes
        if I_nodes>0:
            broken_chains += 1
        if I_nodes == 0 and len(nodes) == 2:
            doubles += 1
    return steiner_nodes, broken_chains, doubles, steiner_nodesx, steiner_nodesz

def I_index(pp, topo):
    num_qubits = pp.num_qubits
    ind = 0
    for q1 in range(num_qubits-1):
        for q2 in range(q1+1,num_qubits):
            if topo.dist(q1,q2) == 1:
                for gadget in pp.pauli_gadgets:
                    if (gadget.paulis[q1] == I) != (gadget.paulis[q2] == I):
                        ind += 1
    return ind

def qubit_correlation_sum(pp, topo):
    correlations = qubit_correlations(pp)
    c_sum = 0
    for i in range(pp.num_qubits-1):
        for j in range(i+1, pp.num_qubits):
            if topo.dist(i,j) == 1:
                c_sum += correlations[i,j]
    return c_sum

def create_datastructures(pp, topo):
    num_qubits = pp.num_qubits
    num_gadgets = len(pp.pauli_gadgets)
    gadget_angles = []
    removed_gadgets = np.zeros((num_gadgets), dtype=np.int8)
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
    general_data = (gadget_data, gadget_angles, removed_gadgets, pauligraph, pauligraph_degrees, last_edge)
    return general_data

def steiner_tree(gadget_data,topo, gadget_index):
    """ Uses NetworkX Steinertree algorithm to make steinertree from gadget."""
    num_qubits, num_gadgets = gadget_data.shape
    nodes = []
    for i in range(num_qubits):
        if gadget_data[i,gadget_index] != 0b00:
            nodes.append(i)
    steiner_stree = nx.algorithms.approximation.steinertree.steiner_tree(topo.to_nx, nodes)
    return nx.Graph(steiner_stree)

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

def order_gadgets(pp, topo):
    gadget_data, gadget_angles, removed_gadgets, pauligraph, pauligraph_degrees, last_edge = create_datastructures(pp, topo)
    order = []
    for i in range(pp.num_gadgets):
        s_nodes, nodes = steiner_nodes(gadget_data, pauligraph_degrees, i)
        order.append((i, nodes + (s_nodes*2)))
    order.sort(key=lambda x: (x[1]))
    return [i[0] for i in order]

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


brisbane = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,-1,
14,-1,-1,-1,15,-1,-1,-1,16,-1,-1,-1,17,-1,-1,
18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,
-1,-1,33,-1,-1,-1,34,-1,-1,-1,35,-1,-1,-1,36,
37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,
52,-1,-1,-1,53,-1,-1,-1,54,-1,-1,-1,55,-1,-1,
56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,
-1,-1,71,-1,-1,-1,72,-1,-1,-1,73,-1,-1,-1,74,
75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,
90,-1,-1,-1,91,-1,-1,-1,92,-1,-1,-1,93,-1,-1,
94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,
-1,-1,109,-1,-1,-1,110,-1,-1,-1,111,-1,-1,-1,112,
-1,113,114,115,116,117,118,119,120,121,122,123,124,125,126]

def print_grid16_mapping(mapping, tree):
    root, tree_children = tree
    parent = {root: None}
    for i in tree_children:
        children = tree_children[i]
        if children is not None:
            for c in children:
                parent[c] = i
    for i in range(4):
        for j in range(4):
            if i*4+j == root:
                print('X', end='')
            elif i*4+j in mapping:
                up = down = left = right = False
                neighbours = [parent[i*4+j]]
                if tree_children[i*4+j] is not None:
                    neighbours += tree_children[i*4+j]
                if (i-1)*4+j in neighbours:
                    up = True
                if (i+1)*4+j in neighbours:
                    down = True
                if i*4+j-1 in neighbours:
                    left = True
                if i*4+j+1 in neighbours:
                    right = True
                line_char(up, down, left, right)
            else:
                print('.', end='')
        print()
    print()

def print_grid25_mapping(mapping, tree):
    size = 5
    root, tree_children = tree
    parent = {root: None}
    for i in tree_children:
        children = tree_children[i]
        if children is not None:
            for c in children:
                parent[c] = i
    for i in range(size):
        for j in range(size):
            if i*size+j == root:
                print('X', end='')
            elif i*size+j in mapping:
                up = down = left = right = False
                neighbours = [parent[i*size+j]]
                if tree_children[i*size+j] is not None:
                    neighbours += tree_children[i*size+j]
                if (i-1)*size+j in neighbours:
                    up = True
                if (i+1)*size+j in neighbours:
                    down = True
                if i*size+j-1 in neighbours:
                    left = True
                if i*size+j+1 in neighbours:
                    right = True
                line_char(up, down, left, right)
            else:
                print(chr(183), end='')
        print()
    print()

def line_char(up, down, left, right):
    if (up and down and left and right):
        print('┼', end='')
    elif (up and down and right):
        print('├', end='')
    elif (up and down and left):
        print('┤', end='')
    elif (left and right and up):
        print('┴', end='')
    elif (left and right and down):
        print('┬', end='')
    elif (up and down):
        print('|', end='')
    elif (left and right):
        print('─', end='')
    elif (up and right):
        print('└', end='')
    elif (up and left):
        print('┘', end='')
    elif (down and right):
        print('┌', end='')
    elif (down and left):
        print('┐', end='')
    elif up:
        print('╵', end='')
    elif down:
        print('╷', end='')
    elif left:
        print('╴', end='')
    elif right:
        print('╶', end='')

def print_brisbane_mapping(mapping, tree):
    if tree is not None:
        root, tree_children = tree
        for i in range(13):
            for j in range(15):
                if root is not None and brisbane[i*15+j] == root:
                    print('X', end='')
                elif brisbane[i*15+j] != -1:
                    if brisbane[i*15+j] in mapping:
                        if tree_children[brisbane[i*15+j]] is None:
                            print('o', end='')
                        elif len(tree_children[brisbane[i*15+j]]) > 1:
                            print('+', end='')
                        elif len(tree_children[brisbane[i*15+j]]) == 1:
                            print('*', end='')
                    else:
                        print('.', end='')
                else:
                    print(' ', end='')
            print()
    else:
        for i in range(13):
            for j in range(15):
                if brisbane[i*15+j] != -1:
                    if brisbane[i*15+j] in mapping:
                        print('O', end='')
                    else:
                        print('+', end='')
                else:
                    print(' ', end='')
            print()
    print()

def print_brisbane_topo():
    for i in range(13):
        for j in range(15):
            if brisbane[i*15+j] != -1:
                print('O', end='')
            else:
                print(' ', end='')
        print()