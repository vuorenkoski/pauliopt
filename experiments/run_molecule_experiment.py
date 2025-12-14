from itertools import product, repeat
import os
import pickle
from datetime import datetime
from multiprocess import Lock, Pool
import tqdm
import pandas as pd

from pauliopt.pauli.synthesis.steiner_gray_synthesis import pauli_polynomial_steiner_gray_clifford
from pauliopt.pauli.synthesis.shortest_path_pauli_forest import shortest_path_pauli_forest

from pauliopt.pauli.synthesis.pp_mapping import pauli_tree_mapping, complete_tree
from pauliopt.topologies import Topology
from pauliopt.pauli.simplification.simple_simplify import simplify_pauli_polynomial
from experiments.utils import get_topo, find_square_dimensions, cnot_count, permute_with_mapping, cnot_depth, random_mapping
from pauliopt.pauli.pauli_polynomial import PauliPolynomial, I
from pauliopt.pauli.pauli_gadget import PauliGadget
from pauliopt.utils import AngleVar

import signal


def get_topo_kind(topo_kind, num_qubits):
    if topo_kind == "line":
        return Topology.line(num_qubits)
    elif topo_kind == "complete":
        return Topology.complete(num_qubits)
    elif topo_kind == "cycle":
        return Topology.cycle(num_qubits)
    elif topo_kind == "grid":
        if num_qubits == 6:
            return Topology.grid(2, 3)
        elif num_qubits == 8:
            return Topology.grid(2, 4)
        else:
            n_rows, n_cols = find_square_dimensions(num_qubits)
            return Topology.grid(n_rows, n_cols)
    else:
        raise Exception("Unknown topology kind")

def create_csv_header_real_hw():
    header = [
        "name",
        "backend",
        "num_qubits",
        "n_gadgets",
        "method",
        "cx",
        "cx-depth",
        "time",
    ]
    return header

def get_suitable_ibm_backend(n_qubits):
    available_backends = [
#        ("quito", 5),
#        ("nairobi", 7),
#        ("guadalupe", 15),
#        ("mumbai", 27),
#        ("ithaca", 65),
        ("brisbane", 127),
    ]

    available_backends = list(sorted(available_backends, key=lambda x: x[1]))

    for name, backend_qubits in available_backends:
        if backend_qubits >= n_qubits:
            return name

    raise Exception(
        f"No backend with: {n_qubits} in list: {available_backends}")

def pad_pp_to_ibm_backend(pp: PauliPolynomial, n_qubits):
    pp_ = PauliPolynomial(n_qubits)

    pp_qubits = pp.num_qubits

    identity_pad = [I for _ in range(n_qubits - pp_qubits)]

    for gadget in pp:
        assert isinstance(gadget, PauliGadget)
        angle = gadget.angle
        paulis = gadget.paulis + identity_pad

        pp_ >>= PauliGadget(angle, paulis)
    return pp_

TIMEOUT = 60*60

def signal_handler(signum, frame):
    raise TimeoutError("Timed out!")

def time_out(func):
    def wrapper(data):
        try:
            signal.signal(signal.SIGALRM, signal_handler)
            signal.alarm(TIMEOUT)
            func(data)
        except Exception as e:
            print(data)
            exp_data, _ = data
            synthesis = exp_data[-1]
            synth_name, _ = synthesis

            print(f"{synth_name} failed")
            print(f"Error: {e}")
            return
    return wrapper

def get_lock(new_lock):
    global lock
    lock = new_lock

def synth_pp_pauliopt_steiner_gray(pp: PauliPolynomial, topo: Topology):
    pp = simplify_pauli_polynomial(pp, allow_acs=True)
    circ_out, gadget_perm, perm, benchmarks = pauli_polynomial_steiner_gray_clifford(pp, topo, None)
    return {'cx': cnot_count(circ_out), 'cx-depth': cnot_depth(circ_out)} | benchmarks

def synth_pp_shortest_path_mapping(pp: PauliPolynomial, topo: Topology):
    pp = simplify_pauli_polynomial(pp, allow_acs=True)
    mapping, tree = pauli_tree_mapping(pp, topo)
    pp_m = permute_with_mapping(mapping, pp, topo.num_qubits)
    circ_out, gadget_perm, perm, benchmarks = shortest_path_pauli_forest(pp_m, topo, tree)
    return {'cx': cnot_count(circ_out), 'cx-depth': cnot_depth(circ_out)} | benchmarks 

def synth_pp_shortest_path(pp: PauliPolynomial, topo: Topology):
    pp = simplify_pauli_polynomial(pp, allow_acs=True)
#    map_r = random_mapping(topo)
    tree_c = complete_tree(topo) # This is needed for forest synth with random mapping
#    pp_m = permute_with_mapping(map_r, pp, topo.num_qubits)
    circ_out, gadget_perm, perm, benchmarks = shortest_path_pauli_forest(pp, topo, tree_c)
    return {'cx': cnot_count(circ_out), 'cx-depth': cnot_depth(circ_out)} | benchmarks 

SYNTHESIS_METHODS = {
    "pauliopt_steiner_gray": synth_pp_pauliopt_steiner_gray,
    "shortest_path_synthesis_map": synth_pp_shortest_path_mapping,
    "shortest_path_synthesis": synth_pp_shortest_path,
    }

def threaded_real_hw_ucc_evaluation(max_qubits=30):
    op_directory = "./pp_molecules/"
    files = os.listdir(op_directory)

    results_directory = "./"
    os.makedirs(results_directory, exist_ok=True)

    results_file = os.path.join(results_directory, "results_molecules.csv")
    df = pd.DataFrame({c: [] for c in create_csv_header_real_hw()})
    with open(results_file, "wb") as f:
        df.to_csv(f, header=create_csv_header_real_hw(), index=False)

    total_len = len(files) * len(SYNTHESIS_METHODS.items())
    print(SYNTHESIS_METHODS.items())
    arguments = zip(product(files, SYNTHESIS_METHODS.items()),
                    repeat((op_directory, results_file)))

    lock = Lock()
    n_workers = os.cpu_count() - 1
    with Pool(n_workers, initializer=get_lock, initargs=(lock,)) as p:
        for _ in tqdm.tqdm(p.imap_unordered(threaded_hw_function, arguments, chunksize=1), total=total_len):
            pass
    p.join()


@time_out
def threaded_hw_function(data):
    exp_data, fixed_data = data
    filename, (synth_name, synth_method) = exp_data
    op_directory, results_file = fixed_data
    path = os.path.join(op_directory, filename)
    name = filename.replace(".pickle", "")
    print(path)
    with open(path, "rb") as pickle_in:
        pp = pickle.load(pickle_in)
        gadgets = []
        for gadget in pp.pauli_gadgets:
            if not isinstance(gadget.angle, AngleVar):
                gadget2 = PauliGadget(gadget.angle, gadget.paulis)
            else:
                gadget2 = PauliGadget(AngleVar(gadget.angle._label,gadget.angle._latex_label), gadget.paulis)
            gadgets.append(gadget2)
        pp.pauli_gadgets = gadgets
        pp = simplify_pauli_polynomial(pp, allow_acs=True)

    n_qubits = pp.num_qubits

    backend_name = get_suitable_ibm_backend(n_qubits)
    topo = get_topo(backend_name)
    pp_ = pad_pp_to_ibm_backend(pp, topo.num_qubits)

    start = datetime.now()
    count_dict = synth_method(pp_, topo)
    time_passed = (datetime.now() - start).total_seconds()
    column = {
        "name": name,
        "backend": backend_name,
        "num_qubits": n_qubits,
        "n_gadgets": pp.num_gadgets,
        "method": synth_name,
        "time": time_passed,
    } | count_dict

    df = pd.DataFrame([{c: column[c] for c in create_csv_header_real_hw()}])
    lock.acquire()
    with open(results_file, "ab") as f_ptr:
        df.to_csv(f_ptr, header=False, index=False)
    lock.release()

if __name__ == "__main__":
    threaded_real_hw_ucc_evaluation()