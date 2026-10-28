"""Turn a grammar design plus a load case into surrogate inputs, run the ensemble, and apply the routing rule.
NumPy only: no JAX or PyTorch needed to use a trained model."""
import json
import os
import pathlib

import numpy as np

import surrogate_common as sc
import truss_solver as ts
from gnn_numpy import Ensemble
from train_gnn_config import LAYERS

NMAX = 14
HERE = pathlib.Path(__file__).parent


def load_ensemble(models_dir=None):
    """Loads every models/gnn_s*.npz. TRUSS_MODELS_DIR overrides the folder (used by tests)."""
    folder = models_dir or os.environ.get("TRUSS_MODELS_DIR")
    if folder is None:      # a models folder if there is one, otherwise the app folder itself (flat layout)
        folder = HERE / "models" if list((HERE / "models").glob("gnn_s*.npz")) else HERE
    paths = sorted(pathlib.Path(folder).glob("gnn_s*.npz"))
    return Ensemble([str(p) for p in paths], LAYERS) if paths else None


def data_file(name):
    """data/<name> if it exists, otherwise <name> next to the app (flat layout). None when neither exists."""
    for p in (HERE / "data" / name, HERE / name):
        if p.exists():
            return p
    return None


def load_results():
    p = data_file("gnn_results.json")
    return json.loads(p.read_text()) if p else None


def encode_inputs(state, loads):
    """The five padded input arrays, shape (1, NMAX, ...), in the same layout the dataset uses."""
    if len(state.nodes) > NMAX:
        raise ValueError(f"the surrogate handles at most {NMAX} joints")
    pin, roller = ts.support_nodes(state)
    idx = {n.id: k for k, n in enumerate(state.nodes)}
    pos = np.zeros((NMAX, 2), np.float32)
    mask = np.zeros(NMAX, np.float32)
    role = np.zeros((NMAX, 2), np.float32)
    load = np.zeros((NMAX, 2), np.float32)
    adj = np.zeros((NMAX, NMAX), np.float32)
    for n in state.nodes:
        pos[idx[n.id]], mask[idx[n.id]] = (n.x, n.y), 1
    role[idx[pin], 0] = 1
    role[idx[roller], 1] = 1
    for nid, (fx, fy) in loads.items():
        load[idx[nid]] = (fx, fy)
    for e in state.edges:
        adj[idx[e.a], idx[e.b]] = adj[idx[e.b], idx[e.a]] = 1
    return [a[None] for a in (pos, mask, role, load, adj)]


def predict(ens, state, loads):
    """Surrogate answer for one design. Member forces come back in the order of state.edges."""
    x = encode_inputs(state, loads)
    out = ens.predict(*x)
    idx = {n.id: k for k, n in enumerate(state.nodes)}
    forces = np.array([out["force"][0, idx[e.a], idx[e.b]] for e in state.edges])
    return {"forces": forces, "max_force": float(out["max_force"][0]), "max_disp": float(out["max_disp"][0]),
            "spread_force": float(out["spread_log_force"][0]), "spread_disp": float(out["spread_log_disp"][0])}


def decide(pred, rule):
    """Verdict from the surrogate and whether the routing rule sends the design to the solver instead."""
    ok = bool(sc.feasible(pred["max_force"], pred["max_disp"]))
    if not rule:
        return ok, True
    a, b = rule["margin_a"], rule["margin_b"]
    near_f = abs(np.log10(max(pred["max_force"], 1e-6)) - np.log10(sc.F_LIM)) < a + b * pred["spread_force"]
    near_d = abs(np.log10(max(pred["max_disp"], 1e-6)) - np.log10(sc.D_LIM)) < a + b * pred["spread_disp"]
    return ok, bool(near_f or near_d)
