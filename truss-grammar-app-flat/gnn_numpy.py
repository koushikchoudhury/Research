"""NumPy forward pass of the trained graph network. The Streamlit app uses this, so the app does not need PyTorch.

The architecture is defined once in train_gnn.py (PyTorch); this file mirrors it exactly, and
test_surrogate.py checks that both give the same numbers.

Inputs, for a batch of B padded designs with N joints each:
    pos (B,N,2)  mask (B,N)  role (B,N,2) [is pin, is roller]  load (B,N,2)  adj (B,N,N)
Outputs:
    force  (B,N,N)  predicted signed axial force per member (tension positive), read where adj == 1
    disp   (B,N,2)  predicted joint displacement
"""
import numpy as np


def silu(x):
    return x / (1.0 + np.exp(-np.clip(x, -60, 60)))


def mlp(x, p, name):
    return silu(x @ p[f"{name}.0.weight"].T + p[f"{name}.0.bias"]) @ p[f"{name}.2.weight"].T + p[f"{name}.2.bias"]


def layer_norm(x, w, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def forward(p, pos, mask, role, load, adj, n_layers):
    B, N, _ = pos.shape
    pin = (pos * role[..., 0:1]).sum(1, keepdims=True)
    rel = (pos - pin) * mask[..., None]
    h = mlp(np.concatenate([rel, role, load], -1), p, "enc")
    d = rel[:, None, :, :] - rel[:, :, None, :]                 # d[b, i, j] = pos_j - pos_i
    L = np.sqrt((d ** 2).sum(-1, keepdims=True) + 1e-12)
    e = np.concatenate([d, L], -1)
    A = adj[..., None]
    for k in range(n_layers):
        hi = np.broadcast_to(h[:, :, None, :], (B, N, N, h.shape[-1]))
        hj = np.broadcast_to(h[:, None, :, :], (B, N, N, h.shape[-1]))
        m = mlp(np.concatenate([hi, hj, e], -1), p, f"layers.{k}.msg") * A
        h = layer_norm(h + mlp(np.concatenate([h, m.sum(2)], -1), p, f"layers.{k}.upd"),
                       p[f"layers.{k}.ln.weight"], p[f"layers.{k}.ln.bias"])
    hi = np.broadcast_to(h[:, :, None, :], (B, N, N, h.shape[-1]))
    hj = np.broadcast_to(h[:, None, :, :], (B, N, N, h.shape[-1]))
    dx, dy = d[..., 0:1], d[..., 1:2]
    geo = np.concatenate([L, dx * dx, dy * dy, dx * dy], -1)
    t = mlp(np.concatenate([hi + hj, hi * hj, np.abs(hi - hj), geo], -1), p, "edge_head")[..., 0]
    o = mlp(h, p, "node_head")
    return np.sinh(t) * adj, np.sinh(o) * mask[..., None]      # the network works in asinh space


class Ensemble:
    """One or more trained networks. Their disagreement is the uncertainty used to decide what goes to the solver."""

    def __init__(self, paths, n_layers):
        self.members = [dict(np.load(p)) for p in paths]
        self.n_layers = n_layers

    def predict(self, pos, mask, role, load, adj, batch=256):
        outs_f, outs_d = [], []
        for p in self.members:
            fs, ds = [], []
            for i in range(0, len(pos), batch):
                sl = slice(i, i + batch)
                f, d = forward(p, pos[sl], mask[sl], role[sl], load[sl], adj[sl], self.n_layers)
                fs.append(f)
                ds.append(d)
            outs_f.append(np.concatenate(fs))
            outs_d.append(np.concatenate(ds))
        F, D = np.stack(outs_f), np.stack(outs_d)           # (members, S, ...)
        mf = np.abs(F).reshape(F.shape[0], F.shape[1], -1).max(-1)          # per member: max |force|
        md = np.sqrt((D ** 2).sum(-1)).max(-1)                               # per member: max displacement
        return {"force": F.mean(0), "disp": D.mean(0),
                "max_force": np.median(mf, 0), "max_disp": np.median(md, 0),
                "spread_log_force": np.log10(np.maximum(mf, 1e-6)).std(0),
                "spread_log_disp": np.log10(np.maximum(md, 1e-6)).std(0)}
