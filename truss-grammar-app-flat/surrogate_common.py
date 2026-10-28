"""Shared by the baseline and the graph network: loading, splits, the feasibility rule and the metrics."""
import numpy as np

# Illustrative limits, in units of the applied load (P = 1) and EA = 1. They are placeholders for a real design
# brief: the point is that feasibility is decided by the solver's numbers, and the surrogate has to reproduce it.
F_LIM = 2.0      # largest axial force allowed
D_LIM = 5.0      # largest joint displacement allowed
TRAIN_MAX_JOINTS = 9     # train on designs with up to 9 joints; 10 to 14 joints are held out as the "larger designs" test


def load(path="data/truss_dataset.npz"):
    return dict(np.load(path))


def feasible(max_force, max_disp):
    return (np.asarray(max_force) <= F_LIM) & (np.asarray(max_disp) <= D_LIM)


def make_splits(d, seed=0):
    """Split by design (all load cases and jittered copies of one design stay together), never by sample.
    Returns dict of index arrays: train, val, test (same size range as train), big (more joints than any training design)."""
    rng = np.random.default_rng(seed)
    nj, group = d["nj"], d["group"]
    small = np.flatnonzero(nj <= TRAIN_MAX_JOINTS)
    big = np.flatnonzero(nj > TRAIN_MAX_JOINTS)
    groups = np.unique(group[small])
    rng.shuffle(groups)
    n = len(groups)
    cut1, cut2 = int(0.8 * n), int(0.9 * n)
    which = {g: 0 for g in groups[:cut1]}
    which.update({g: 1 for g in groups[cut1:cut2]})
    which.update({g: 2 for g in groups[cut2:]})
    part = np.array([which[g] for g in group[small]])
    return {"train": small[part == 0], "val": small[part == 1], "test": small[part == 2], "big": big}


def scalar_metrics(true_f, true_d, pred_f, pred_d):
    """Errors on the two headline numbers plus the feasibility decision they drive."""
    out = {}
    for name, t, p in (("max_force", true_f, pred_f), ("max_disp", true_d, pred_d)):
        lt, lp = np.log10(np.maximum(t, 1e-6)), np.log10(np.maximum(p, 1e-6))
        ss = ((lt - lt.mean()) ** 2).sum()
        out[f"{name}_R2_log"] = float(1 - ((lt - lp) ** 2).sum() / ss)
        out[f"{name}_median_rel_err"] = float(np.median(np.abs(p - t) / np.maximum(t, 1e-6)))
    tf, pf = feasible(true_f, true_d), feasible(pred_f, pred_d)
    out["feasible_accuracy"] = float((tf == pf).mean())
    out["false_feasible_rate"] = float((pf & ~tf).sum() / max(1, (~tf).sum()))   # unsafe designs the surrogate would pass
    out["false_reject_rate"] = float((~pf & tf).sum() / max(1, tf.sum()))        # good designs the surrogate would drop
    out["share_feasible"] = float(tf.mean())
    return out
