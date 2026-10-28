"""Truss grammar core (Python port of core.js). Pure functions, no Streamlit or matplotlib.

State: State(nodes=[Node(id, x, y, fixed)], edges=[Edge(a, b, free)])  with a < b.
Rules are rebuilt from Fig. 4 of Shea, Stankovic, Agrawal, Cagan, McComb,
J. Comput. Inf. Sci. Eng. 25(12), 120802, 2025. They are NOT taken from the paper's
supplementary code; compare the two before presenting this as the paper's grammar.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

TOL = 1e-9


@dataclass
class Node:
    id: int
    x: float
    y: float
    fixed: bool = False


@dataclass
class Edge:
    a: int
    b: int
    free: bool = False


@dataclass
class State:
    nodes: list = field(default_factory=list)
    edges: list = field(default_factory=list)


def clone(s: State) -> State:
    return State([Node(n.id, n.x, n.y, n.fixed) for n in s.nodes],
                 [Edge(e.a, e.b, e.free) for e in s.edges])


def mk_edge(a: int, b: int, free: bool) -> Edge:
    return Edge(a, b, bool(free)) if a < b else Edge(b, a, bool(free))


def ekey(a: int, b: int) -> str:
    return f"{a}-{b}" if a < b else f"{b}-{a}"


def node_map(s: State) -> dict:
    return {n.id: n for n in s.nodes}


def adjacency(s: State) -> dict:
    adj = {n.id: {} for n in s.nodes}
    for e in s.edges:
        adj[e.a][e.b] = e
        adj[e.b][e.a] = e
    return adj


def next_id(s: State) -> int:
    return max((n.id for n in s.nodes), default=-1) + 1


def common(adj: dict, a: int, b: int) -> list:
    return sorted(k for k in adj[a] if k != b and k in adj[b])


def remove_nodes(s: State, ids) -> None:
    ids = set(ids)
    s.nodes = [n for n in s.nodes if n.id not in ids]
    s.edges = [e for e in s.edges if e.a not in ids and e.b not in ids]


def remove_edge(s: State, a: int, b: int) -> None:
    k = ekey(a, b)
    s.edges = [e for e in s.edges if ekey(e.a, e.b) != k]


# ---------- geometry ----------
def orient(p, q, r) -> float:
    return (q.x - p.x) * (r.y - p.y) - (q.y - p.y) * (r.x - p.x)


def sgn(v: float) -> int:
    return 0 if abs(v) < TOL else (1 if v > 0 else -1)


def on_seg(p, q, r) -> bool:
    if abs(orient(p, q, r)) > TOL:
        return False
    return (min(p.x, q.x) - TOL <= r.x <= max(p.x, q.x) + TOL and
            min(p.y, q.y) - TOL <= r.y <= max(p.y, q.y) + TOL)


def segments_meet(p1, p2, p3, p4) -> bool:
    s1, s2 = sgn(orient(p3, p4, p1)), sgn(orient(p3, p4, p2))
    s3, s4 = sgn(orient(p1, p2, p3)), sgn(orient(p1, p2, p4))
    if s1 * s2 < 0 and s3 * s4 < 0:
        return True
    return on_seg(p3, p4, p1) or on_seg(p3, p4, p2) or on_seg(p1, p2, p3) or on_seg(p1, p2, p4)


# ---------- structural checks ----------
def matrix_rank(M: list, tol: float) -> int:
    m = len(M)
    n = len(M[0]) if m else 0
    A = [row[:] for row in M]
    r = 0
    for c in range(n):
        if r >= m:
            break
        p, best = r, abs(A[r][c])
        for i in range(r + 1, m):
            v = abs(A[i][c])
            if v > best:
                best, p = v, i
        if best < tol:
            continue
        A[r], A[p] = A[p], A[r]
        for i in range(r + 1, m):
            f = A[i][c] / A[r][c]
            if f != 0:
                for k in range(c, n):
                    A[i][k] -= f * A[r][k]
        r += 1
    return r


def rigidity_rank(s: State) -> int:
    idx = {n.id: i for i, n in enumerate(s.nodes)}
    nm = node_map(s)
    M = []
    for e in s.edges:
        row = [0.0] * (2 * len(s.nodes))
        p, q = nm[e.a], nm[e.b]
        i, k = idx[e.a], idx[e.b]
        row[2 * i], row[2 * i + 1] = p.x - q.x, p.y - q.y
        row[2 * k], row[2 * k + 1] = q.x - p.x, q.y - p.y
        M.append(row)
    return matrix_rank(M, 1e-7)


def is_connected(s: State) -> bool:
    if not s.nodes:
        return False
    adj = adjacency(s)
    seen, stack = {s.nodes[0].id}, [s.nodes[0].id]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return len(seen) == len(s.nodes)


@dataclass
class Summary:
    j: int
    b: int
    need: int
    rank: int
    dof: int
    connected: bool
    status: str
    free: int
    fixed: int


def summary(s: State) -> Summary:
    j, b = len(s.nodes), len(s.edges)
    need = 2 * j - 3
    rank = rigidity_rank(s)
    connected = is_connected(s)
    status = "redundant"
    if b == need and rank == need:
        status = "isostatic"
    elif rank < need:
        status = "mechanism"
    return Summary(j, b, need, rank, need - rank, connected, status,
                   sum(1 for e in s.edges if e.free), sum(1 for n in s.nodes if n.fixed))


@dataclass
class Validation:
    ok: bool
    sm: Summary
    kind: Optional[str] = None  # 'topology' (a rule bug), 'geometry' (blocked), 'rank' (degenerate)
    reason: Optional[str] = None


def validate(s: State) -> Validation:
    sm = summary(s)
    if not sm.connected:
        return Validation(False, sm, "topology", "a joint is disconnected from the truss")
    if sm.b != sm.need:
        return Validation(False, sm, "topology", f"Maxwell count fails: {sm.b} members, {sm.need} needed")
    nm = node_map(s)
    for i in range(len(s.nodes)):
        for k in range(i + 1, len(s.nodes)):
            p, q = s.nodes[i], s.nodes[k]
            if math.hypot(p.x - q.x, p.y - q.y) < 1e-6:
                return Validation(False, sm, "geometry", f"joints {p.id} and {q.id} coincide")
    for e in s.edges:
        for n in s.nodes:
            if n.id in (e.a, e.b):
                continue
            if on_seg(nm[e.a], nm[e.b], n):
                return Validation(False, sm, "geometry", f"joint {n.id} lies on member {e.a}–{e.b}")
    for i in range(len(s.edges)):
        for k in range(i + 1, len(s.edges)):
            e1, e2 = s.edges[i], s.edges[k]
            shared = [v for v in (e1.a, e1.b) if v in (e2.a, e2.b)]
            if len(shared) == 2:
                return Validation(False, sm, "geometry", f"duplicate member {e1.a}–{e1.b}")
            if len(shared) == 1:
                c = nm[shared[0]]
                p = nm[e1.b if e1.a == c.id else e1.a]
                q = nm[e2.b if e2.a == c.id else e2.a]
                cr = (p.x - c.x) * (q.y - c.y) - (p.y - c.y) * (q.x - c.x)
                dt = (p.x - c.x) * (q.x - c.x) + (p.y - c.y) * (q.y - c.y)
                if abs(cr) < TOL and dt > 0:
                    return Validation(False, sm, "geometry",
                                      f"members {e1.a}–{e1.b} and {e2.a}–{e2.b} overlap")
            elif segments_meet(nm[e1.a], nm[e1.b], nm[e2.a], nm[e2.b]):
                return Validation(False, sm, "geometry",
                                  f"member {e1.a}–{e1.b} crosses member {e2.a}–{e2.b}")
    if sm.status != "isostatic":
        return Validation(False, sm, "rank",
                          f"rigidity rank {sm.rank} of {sm.need}: the truss would be a mechanism")
    return Validation(True, sm)


# ---------- the five rules ----------
@dataclass
class Rule:
    id: str
    name: str
    inverse: str
    text: str
    matches: Callable
    apply: Callable


def _r1_matches(s):
    adj, out = adjacency(s), []
    for e in [e for e in s.edges if e.free]:
        for c in common(adj, e.a, e.b):
            out.append({"a": e.a, "b": e.b, "c": c, "label": f"AB {e.a}–{e.b} · apex {c}"})
    return out


def _r1_apply(s, m):
    t = clone(s)
    nm = node_map(t)
    P = next_id(t)
    A, B = nm[m["a"]], nm[m["b"]]
    remove_edge(t, m["a"], m["b"])
    t.nodes.append(Node(P, (A.x + B.x) / 2, (A.y + B.y) / 2, False))
    t.edges += [mk_edge(m["a"], P, True), mk_edge(P, m["b"], True), mk_edge(m["c"], P, False)]
    return t


def _r2_matches(s):
    adj, out = adjacency(s), []
    for p in s.nodes:
        if p.fixed or len(adj[p.id]) != 3:
            continue
        nb = list(adj[p.id].items())
        solid = [(k, e) for k, e in nb if not e.free]
        fr = [(k, e) for k, e in nb if e.free]
        if len(solid) != 1 or len(fr) != 2:
            continue
        c = solid[0][0]
        a, b = min(fr[0][0], fr[1][0]), max(fr[0][0], fr[1][0])
        if c in adj[a] and c in adj[b] and b not in adj[a]:
            out.append({"p": p.id, "a": a, "b": b, "c": c, "label": f"joint {p.id}"})
    return out


def _r2_apply(s, m):
    t = clone(s)
    remove_nodes(t, [m["p"]])
    t.edges.append(mk_edge(m["a"], m["b"], True))
    return t


def _r3_matches(s):
    adj, out = adjacency(s), []
    for A in [n for n in s.nodes if n.fixed]:
        for c, e in list(adj[A.id].items()):
            if not e.free:
                continue
            for b in common(adj, A.id, c):
                out.append({"a": A.id, "b": b, "c": c,
                            "label": f"support {A.id} · free {A.id}–{c} · B {b}"})
    return out


def _r3_apply(s, m):
    t = clone(s)
    nm = node_map(t)
    D = next_id(t)
    A, B, C = nm[m["a"]], nm[m["b"]], nm[m["c"]]
    L = math.hypot(B.x - A.x, B.y - A.y)
    ux, uy = (B.x - A.x) / L, (B.y - A.y) / L
    nx, ny = -uy, ux
    if nx * (C.x - A.x) + ny * (C.y - A.y) > 0:
        nx, ny = -nx, -ny
    ab_free = next(e for e in t.edges if ekey(e.a, e.b) == ekey(m["a"], m["b"])).free
    remove_edge(t, m["a"], m["b"])
    t.nodes.append(Node(D, A.x + 0.2 * L * (ux + nx), A.y + 0.2 * L * (uy + ny), False))
    t.edges += [mk_edge(m["a"], D, False), mk_edge(D, m["b"], ab_free), mk_edge(D, m["c"], False)]
    return t


def _r4_matches(s):
    adj, out = adjacency(s), []
    for A in [n for n in s.nodes if n.fixed]:
        for d in list(adj[A.id].keys()):
            D = next(n for n in s.nodes if n.id == d)
            if D.fixed or len(adj[d]) != 3:
                continue
            rest = [k for k in adj[d] if k != A.id]
            for x, y in ((rest[0], rest[1]), (rest[1], rest[0])):
                if x in adj[A.id] and y in adj[x] and y not in adj[A.id]:
                    out.append({"a": A.id, "d": d, "x": x, "y": y,
                                "label": f"joint {d} into support {A.id}"})
    return out


def _r4_apply(s, m):
    t = clone(s)
    dy_free = next(e for e in t.edges if ekey(e.a, e.b) == ekey(m["d"], m["y"])).free
    remove_nodes(t, [m["d"]])
    t.edges.append(mk_edge(m["a"], m["y"], dy_free))
    return t


def _r5_matches(s):
    adj, out = adjacency(s), []
    for e in [e for e in s.edges if not e.free]:
        cn = common(adj, e.a, e.b)
        if len(cn) == 2 and cn[1] not in adj[cn[0]]:
            out.append({"a": e.a, "c": e.b, "b": cn[0], "d": cn[1],
                        "label": f"{e.a}–{e.b} → {cn[0]}–{cn[1]}"})
    return out


def _r5_apply(s, m):
    t = clone(s)
    remove_edge(t, m["a"], m["c"])
    t.edges.append(mk_edge(m["b"], m["d"], False))
    return t


RULES = [
    Rule("R1", "Subdivide", "R2",
         "Free member AB of triangle ABC is split at its midpoint P, and a new member joins P to the apex C. "
         "One joint and two members are added.", _r1_matches, _r1_apply),
    Rule("R2", "Merge", "R1",
         "A joint P with three members, two free (PA, PB) and one solid (PC), is removed, and free member AB "
         "is restored. Inverse of R1.", _r2_matches, _r2_apply),
    Rule("R3", "Extend", "R4",
         "At a support A with free member AC, joint D is added beside A. Member AB is replaced by AD, DB and DC, "
         "and DB keeps the status of AB. One joint and two members are added.", _r3_matches, _r3_apply),
    Rule("R4", "Contract", "R3",
         "A three-member joint D next to support A folds into A, and its far neighbour Y is joined to A. "
         "One joint and two members are removed. Inverse of R3.", _r4_matches, _r4_apply),
    Rule("R5", "Flip", "R5",
         "Interior diagonal AC of a quadrilateral ABCD swaps to the other diagonal BD. "
         "Joint and member counts stay the same.", _r5_matches, _r5_apply),
]
RULE_BY_ID = {r.id: r for r in RULES}


@dataclass
class Option:
    rule: str
    match: dict
    label: str
    state: State
    ok: bool
    kind: Optional[str]
    reason: Optional[str]
    sm: Summary


def enumerate_options(s: State) -> list:
    out = []
    for r in RULES:
        for m in r.matches(s):
            result = r.apply(s, m)
            v = validate(result)
            out.append(Option(r.id, m, m["label"], result, v.ok, v.kind, v.reason, v.sm))
    return out


# ---------- presets (the "Before" panels of Fig. 4) ----------
PRESETS = {
    "triangle": ("Triangle (R1, R3)", State(
        [Node(0, 0, 0, True), Node(1, 1, 0), Node(2, 0.5, 1)],
        [mk_edge(0, 1, True), mk_edge(0, 2, False), mk_edge(1, 2, False)])),
    "strip": ("Two triangles (R4)", State(
        [Node(0, 0, 0, True), Node(1, 1, 0), Node(2, 1, 1), Node(3, 2, 1)],
        [mk_edge(0, 1, False), mk_edge(0, 2, False), mk_edge(1, 2, False), mk_edge(1, 3, False), mk_edge(2, 3, False)])),
    "square": ("Square with diagonal (R5)", State(
        [Node(0, 0, 0, True), Node(1, 1, 0), Node(2, 1, 1), Node(3, 0, 1)],
        [mk_edge(0, 1, False), mk_edge(1, 2, False), mk_edge(2, 3, False), mk_edge(3, 0, False), mk_edge(0, 2, False)])),
}


def preset_state(key: str) -> State:
    return clone(PRESETS[key][1])


# ---------- canonical form and seeded random walk ----------
def _r6(v: float) -> str:
    return f"{round(v * 1e6) / 1e6 + 0.0:.6f}"


def nkey(n: Node) -> str:
    return f"{_r6(n.x)},{_r6(n.y)}"


def edge_key_xy(e: Edge, nm: dict) -> str:
    a, b = nkey(nm[e.a]), nkey(nm[e.b])
    return a + "|" + b if a < b else b + "|" + a


def canon(s: State) -> str:
    nm = node_map(s)
    ns = sorted(nkey(n) + ("F" if n.fixed else "") for n in s.nodes)
    es = sorted(edge_key_xy(e, nm) + ("f" if e.free else "s") for e in s.edges)
    return ";".join(ns) + "#" + ";".join(es)


def _imul(a: int, b: int) -> int:
    return (a * b) & 0xFFFFFFFF


def mulberry32(seed: int):
    a = seed & 0xFFFFFFFF

    def rnd() -> float:
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = _imul(a ^ (a >> 15), 1 | a)
        t = ((t + _imul(t ^ (t >> 7), 61 | t)) & 0xFFFFFFFF) ^ t
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0

    return rnd


def random_walk(s: State, steps: int, seed: int) -> list:
    rnd, trail, cur = mulberry32(seed), [], s
    for _ in range(steps):
        options = [o for o in enumerate_options(cur) if o.ok]
        if not options:
            break
        pick = options[int(rnd() * len(options))]
        trail.append({"rule": pick.rule, "label": pick.label, "state": pick.state})
        cur = pick.state
    return trail


# ---------- self-tests ----------
def run_tests() -> list:
    T = []

    def add(name, passed, detail="", info=False):
        T.append({"name": name, "pass": bool(passed), "detail": detail, "info": info})

    def first(s, rule):
        return [o for o in enumerate_options(s) if o.rule == rule]

    tri, strip, sq = (preset_state(k) for k in ("triangle", "strip", "square"))

    o = first(tri, "R1")[0]
    n = next((k for k in o.state.nodes if k.id == 3), None)
    add("R1 splits the free member at its midpoint and adds apex member",
        o.ok and o.sm.j == 4 and o.sm.b == 5 and n and abs(n.x - 0.5) < 1e-12 and abs(n.y) < 1e-12 and
        any(e.a == 2 and e.b == 3 and not e.free for e in o.state.edges),
        "triangle → 4 joints, 5 members, P at (0.5, 0): the right-hand panel of Fig. 4, rule 1")
    back = [x for x in first(o.state, "R2") if x.ok and canon(x.state) == canon(tri)]
    add("R2 undoes R1 exactly", len(back) == 1, "R1 then R2 returns the starting design")

    o = first(tri, "R3")[0]
    add("R3 adds one joint and two members at the support",
        o.ok and o.sm.j == 4 and o.sm.b == 5 and next(k for k in o.state.nodes if k.id == 0).fixed and
        not any(e.a == 0 and e.b == 2 for e in o.state.edges),
        "support 0 stays fixed; member 0–2 is replaced by 0–D, D–2, D–1")
    back = [x for x in first(o.state, "R4") if x.ok and canon(x.state) == canon(tri)]
    add("R4 undoes R3 exactly", len(back) == 1, "R3 then R4 returns the starting design")

    cand = [x for x in first(strip, "R4") if x.ok]
    o = cand[0] if cand else None
    keys = " ".join(sorted(nkey(k) for k in o.state.nodes)) if o else ""
    add("R4 contracts the two-triangle strip to one triangle",
        o and o.sm.j == 3 and o.sm.b == 3 and keys == "0.000000,0.000000 1.000000,1.000000 2.000000,1.000000",
        "matches Fig. 4 rule 4: joint 1 folds into support 0, leaving (0,0), (1,1), (2,1)")

    cand = [x for x in first(sq, "R5") if x.ok]
    o = cand[0] if cand else None
    has = o and any(e.a == 1 and e.b == 3 for e in o.state.edges) and \
        not any(e.a == 0 and e.b == 2 for e in o.state.edges)
    twice = [x for x in first(o.state, "R5") if x.ok and canon(x.state) == canon(sq)] if o else []
    add("R5 flips the diagonal, and flipping twice restores it", has and len(twice) == 1,
        "diagonal 0–2 becomes 1–3 (Fig. 4 rule 5)")

    sq0 = clone(sq)
    sq0.edges = [e for e in sq0.edges if not (e.a == 0 and e.b == 2)]
    sm = summary(sq0)
    add("Rank check detects a mechanism (square without a diagonal)",
        sm.status == "mechanism" and sm.rank == 4 and sm.need == 5, "rank 4 of 5, one degree of freedom")

    bad = clone(sq)
    bad.edges.append(mk_edge(1, 3, False))
    v = validate(bad)
    add("Maxwell count rejects a redundant member (both diagonals of a square)",
        (not v.ok) and v.kind == "topology" and "Maxwell" in v.reason, v.reason)

    bad = clone(sq)
    for n in bad.nodes:
        if n.id == 2:
            n.x, n.y = 0, 1
        if n.id == 3:
            n.x, n.y = 1, 1
    v = validate(bad)
    add("Geometry check rejects crossing members when the count is right",
        (not v.ok) and v.kind == "geometry" and "crosses" in v.reason, v.reason)

    bad = clone(tri)
    bad.nodes.append(Node(3, 2, 2))
    v = validate(bad)
    add("A disconnected joint is rejected", (not v.ok) and v.kind == "topology", v.reason)

    add("R2 does not match when no joint has the R1 pattern", len(RULE_BY_ID["R2"].matches(clone(strip))) == 0,
        "two-triangle strip has no R2 match")

    states = matches = topo_bugs = blocked_geo = blocked_rank = rt_fail = rt_checked = fixed_bad = id_dup = 0
    starts = [tri, strip, sq]
    for seed in range(1, 61):
        cur = starts[seed % 3]
        trail = random_walk(cur, 10, seed)
        seq = [cur] + [t["state"] for t in trail]
        for st in seq:
            states += 1
            if len({n.id for n in st.nodes}) != len(st.nodes):
                id_dup += 1
            for o in enumerate_options(st):
                matches += 1
                topo_bugs += o.kind == "topology"
                blocked_geo += o.kind == "geometry"
                blocked_rank += o.kind == "rank"
                if sum(n.fixed for n in o.state.nodes) != sum(n.fixed for n in st.nodes):
                    fixed_bad += 1
                if o.ok and o.rule in ("R1", "R3", "R5"):
                    rt_checked += 1
                    inv = RULE_BY_ID[o.rule].inverse
                    if not any(x.rule == inv and x.ok and canon(x.state) == canon(st)
                               for x in enumerate_options(o.state)):
                        rt_fail += 1
    add("Every rule application keeps Maxwell count and connectivity", topo_bugs == 0,
        f"{matches} rule applications over {states} designs; {topo_bugs} count or connectivity failures")
    add("No rule changes the number of supports", fixed_bad == 0, f"{fixed_bad} violations")
    add("Joint ids stay unique", id_dup == 0, f"{id_dup} duplicates")
    add("Forward then inverse restores the design (R1→R2, R3→R4, R5→R5)",
        rt_checked > 0 and rt_fail == 0, f"{rt_checked} round trips, {rt_fail} failures")
    add("How often the geometry and rank checks blocked a match", True,
        f"{blocked_geo} blocked for geometry, {blocked_rank} blocked for rank, out of {matches}", info=True)
    return T
