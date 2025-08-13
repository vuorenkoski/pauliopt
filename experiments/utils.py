import random, math, json
import numpy as np
from pauliopt.pauli.pauli_gadget import PauliGadget
from pauliopt.pauli.pauli_polynomial import PauliPolynomial, I, Z, X, Y
from pauliopt.gates import CX
from pauliopt.pauli.pauli_gadget import PPhase
from pauliopt.utils import pi
from pauliopt.topologies import Topology

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

def create_random_phase_gadget(
        num_qubits, min_legs, max_legs, allowed_angels, allowed_legs=None, empty_qubits=0
):
    if allowed_legs is None:
        allowed_legs = [X, Y, Z]
    angle = random.choice(allowed_angels)
    nr_legs = random.randint(min_legs, max_legs)
    legs = random.choices(
        [i for i in range(num_qubits-empty_qubits)], k=nr_legs)
    phase_gadget = [I for _ in range(num_qubits)]
    for leg in legs:
        phase_gadget[leg] = random.choice(allowed_legs)
    return PPhase(angle) @ phase_gadget


def create_random_pauli_polynomial(
        num_qubits: int, num_gadgets: int, min_legs=None, max_legs=None, allowed_angels=None, seed=None, empty_qubits=0
):
    if min_legs is None:
        min_legs = 1
    if max_legs is None:
        max_legs = num_qubits - empty_qubits
    if allowed_angels is None:
        allowed_angels = [pi, pi / 2, pi / 4, pi / 8, pi / 16]

    if seed is not None:
        random.seed(seed)
    pp = PauliPolynomial(num_qubits)
    for _ in range(num_gadgets):
        pp >>= create_random_phase_gadget(
            num_qubits, min_legs, max_legs, allowed_angels, empty_qubits=empty_qubits
        )

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

def print_pp(pp, order=None):
    num_qubits = pp.num_qubits
    num_gadgets = len(pp.pauli_gadgets)

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
    df2 = df.drop(['n_rep','num_qubits','pre-cx'], axis=1)
    df_1 = df2.loc[(df['mapping'] == 'random') & (df['method'] == method1)].drop(['mapping', 'method'], axis=1).reset_index(drop=True).groupby(['n_gadgets']).mean().round(1).reset_index().rename(columns={'cx':'do','time':'do (ms)'})  
    df_1m = df2.loc[(df['mapping'] == 'algorithm') & (df['method'] == method1)].drop(['mapping', 'method'], axis=1).reset_index(drop=True).groupby(['n_gadgets']).mean().round(1).reset_index().rename(columns={'cx':'do+mapping', 'time':'dom (ms)'})  
    df_2 = df2.loc[(df['mapping'] == 'random') & (df['method'] == method2)].drop(['mapping', 'method'], axis=1).reset_index(drop=True).groupby(['n_gadgets']).mean().round(1).reset_index().rename(columns={'cx':'sg','time':'sg (ms)'})  
    df_2m = df2.loc[(df['mapping'] == 'algorithm') & (df['method'] == method2)].drop(['mapping', 'method'], axis=1).reset_index(drop=True).groupby(['n_gadgets']).mean().round(1).reset_index().rename(columns={'cx':'sg+mapping', 'time':'sgm (ms)'})  
    df3 = df_2.merge(df_2m, on=['n_gadgets'], how='left')
    df3 = df3.merge(df_1, on=['n_gadgets'], how='left')
    df3 = df3.merge(df_1m, on=['n_gadgets'], how='left')
    df3['sg-sgm%'] = np.round(((df3['sg+mapping'] / df3['sg']) - 1)*100,1)
    df3['sg-do%'] = np.round(((df3['do'] / df3['sg']) - 1)*100,1)
    df3['do-dom%'] = np.round(((df3['do+mapping'] / df3['do']) - 1)*100,1)
    df3['sg-do time%'] = np.round(((df3['do (ms)'] / df3['sg (ms)']) - 1)*100,1)
    df3 = df3.rename(columns={'n_gadgets': 'gadgets'})
    df3 = df3.set_index('gadgets')
    return df3

def aggregate_data_precx(df, method1, method2):
    df2 = df.drop(['n_rep','num_qubits','cx'], axis=1)
    df_1 = df2.loc[(df['mapping'] == 'random') & (df['method'] == method1)].drop(['mapping', 'method'], axis=1).reset_index(drop=True).groupby(['n_gadgets']).mean().round(1).reset_index().rename(columns={'pre-cx':'do','time':'do (ms)'})  
    df_1m = df2.loc[(df['mapping'] == 'algorithm') & (df['method'] == method1)].drop(['mapping', 'method'], axis=1).reset_index(drop=True).groupby(['n_gadgets']).mean().round(1).reset_index().rename(columns={'pre-cx':'do+mapping', 'time':'dom (ms)'})  
    df_2 = df2.loc[(df['mapping'] == 'random') & (df['method'] == method2)].drop(['mapping', 'method'], axis=1).reset_index(drop=True).groupby(['n_gadgets']).mean().round(1).reset_index().rename(columns={'pre-cx':'sg','time':'sg (ms)'})  
    df_2m = df2.loc[(df['mapping'] == 'algorithm') & (df['method'] == method2)].drop(['mapping', 'method'], axis=1).reset_index(drop=True).groupby(['n_gadgets']).mean().round(1).reset_index().rename(columns={'pre-cx':'sg+mapping', 'time':'sgm (ms)'})  
    df3 = df_2.merge(df_2m, on=['n_gadgets'], how='left')
    df3 = df3.merge(df_1, on=['n_gadgets'], how='left')
    df3 = df3.merge(df_1m, on=['n_gadgets'], how='left')
    df3['sg-sgm%'] = np.round(((df3['sg+mapping'] / df3['sg']) - 1)*100,1)
    df3['sg-do%'] = np.round(((df3['do'] / df3['sg']) - 1)*100,1)
    df3['do-dom%'] = np.round(((df3['do+mapping'] / df3['do']) - 1)*100,1)
    df3['sg-do time%'] = np.round(((df3['do (ms)'] / df3['sg (ms)']) - 1)*100,1)
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