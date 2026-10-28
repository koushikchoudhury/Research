"""Direct stiffness solver for planar pin-jointed trusses (ground truth for the surrogate).

Modelling choices, stated plainly:
  * Every member has the same axial stiffness EA (default 1). All results are in units of the load.
  * Supports: a pin at the (single) fixed joint, plus a roller that blocks vertical movement at the
    right-most other joint. Two reactions plus one reaction is the 3 reactions behind Maxwell's 2j - 3.
  * Linear, small-displacement analysis. Tension is positive.
  * Only point loads on joints. There is no self-weight, no buckling, no stress limit: those are checks
    you add on top of the forces returned here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from truss_core import State


class SolveError(Exception):
    pass


@dataclass
class Result:
    node_ids: list          # joint ids in the order used by u
    u: np.ndarray           # (j, 2) joint displacements
    forces: np.ndarray      # (b,) axial force per member, in the order of state.edges, tension positive
    reactions: dict         # joint id -> (rx, ry)
    max_force: float        # max |axial force|
    max_disp: float         # max joint displacement magnitude
    residual: float         # max |K u - F| on free degrees of freedom


def support_nodes(state: State):
    """Returns (pin_id, roller_id). Raises SolveError when the supports cannot be set."""
    fixed = [n for n in state.nodes if n.fixed]
    if len(fixed) != 1:
        raise SolveError(f"expected exactly one fixed joint, found {len(fixed)}")
    pin = fixed[0]
    others = [n for n in state.nodes if n.id != pin.id]
    if not others:
        raise SolveError("a single joint cannot be solved")
    roller = sorted(others, key=lambda n: (-n.x, n.y, n.id))[0]
    return pin.id, roller.id


def solve(state: State, loads: dict, EA: float = 1.0) -> Result:
    """loads: {joint id: (fx, fy)}. Returns a Result, or raises SolveError for a singular system."""
    ids = [n.id for n in state.nodes]
    idx = {i: k for k, i in enumerate(ids)}
    xy = np.array([[n.x, n.y] for n in state.nodes], float)
    j = len(ids)
    K = np.zeros((2 * j, 2 * j))
    geo = []
    for e in state.edges:
        a, b = idx[e.a], idx[e.b]
        d = xy[b] - xy[a]
        L = float(np.hypot(*d))
        if L < 1e-9:
            raise SolveError("zero-length member")
        c, s = d / L
        t = np.array([c, s])
        k = EA / L * np.outer(np.r_[-t, t], np.r_[-t, t])
        dofs = [2 * a, 2 * a + 1, 2 * b, 2 * b + 1]
        K[np.ix_(dofs, dofs)] += k
        geo.append((a, b, L, t))
    F = np.zeros(2 * j)
    for nid, (fx, fy) in loads.items():
        F[2 * idx[nid]] += fx
        F[2 * idx[nid] + 1] += fy
    pin, roller = support_nodes(state)
    fixed_dofs = [2 * idx[pin], 2 * idx[pin] + 1, 2 * idx[roller] + 1]
    free = [d for d in range(2 * j) if d not in fixed_dofs]
    Kff = K[np.ix_(free, free)]
    if np.linalg.cond(Kff) > 1e10:
        raise SolveError("stiffness matrix is singular: the truss is a mechanism or badly supported")
    u = np.zeros(2 * j)
    u[free] = np.linalg.solve(Kff, F[free])
    forces = np.array([EA / L * float(t @ (u[2 * b:2 * b + 2] - u[2 * a:2 * a + 2])) for a, b, L, t in geo])
    R = K @ u - F
    reactions = {pin: (float(R[2 * idx[pin]]), float(R[2 * idx[pin] + 1])), roller: (0.0, float(R[2 * idx[roller] + 1]))}
    U = u.reshape(j, 2)
    return Result(ids, U, forces, reactions, float(np.abs(forces).max()) if len(forces) else 0.0,
                  float(np.hypot(U[:, 0], U[:, 1]).max()), float(np.abs((K @ u - F)[free]).max()))


def joint_balance(state: State, res: Result, loads: dict) -> float:
    """Independent equilibrium check built from the member forces alone: the largest |sum of forces| at any joint,
    counting external loads and support reactions."""
    idx = {i: k for k, i in enumerate(res.node_ids)}
    nm = {n.id: n for n in state.nodes}
    tot = np.zeros((len(res.node_ids), 2))
    for e, N in zip(state.edges, res.forces):
        a, b = nm[e.a], nm[e.b]
        d = np.array([b.x - a.x, b.y - a.y])
        t = d / np.hypot(*d)
        tot[idx[e.a]] += N * t      # a tension member pulls joint a toward b
        tot[idx[e.b]] -= N * t
    for nid, (fx, fy) in loads.items():
        tot[idx[nid]] += (fx, fy)
    for nid, (rx, ry) in res.reactions.items():
        tot[idx[nid]] += (rx, ry)    # R = K u - F is the force the support applies to the truss
    return float(np.abs(tot).max())
