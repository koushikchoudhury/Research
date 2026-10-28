/* ===== Truss grammar core: pure functions, no DOM =====
   State: { nodes: [{id,x,y,fixed}], edges: [{a,b,free}] }  (a < b)
   Rules reconstructed from Fig. 4 of Shea, Stankovic, Agrawal, Cagan, McComb (JCISE 2025). */

const TOL = 1e-9;

function cloneState(s) {
  return { nodes: s.nodes.map(n => ({ ...n })), edges: s.edges.map(e => ({ ...e })) };
}
function mkEdge(a, b, free) { return a < b ? { a, b, free: !!free } : { a: b, b: a, free: !!free }; }
function ekey(a, b) { return a < b ? a + '-' + b : b + '-' + a; }
function nodeMap(s) { const m = new Map(); s.nodes.forEach(n => m.set(n.id, n)); return m; }
function adjacency(s) {
  const adj = new Map();
  s.nodes.forEach(n => adj.set(n.id, new Map()));
  s.edges.forEach(e => { adj.get(e.a).set(e.b, e); adj.get(e.b).set(e.a, e); });
  return adj;
}
function nextId(s) { return s.nodes.reduce((m, n) => Math.max(m, n.id), -1) + 1; }
function common(adj, a, b) {
  const out = [];
  adj.get(a).forEach((_, k) => { if (k !== b && adj.get(b).has(k)) out.push(k); });
  return out.sort((x, y) => x - y);
}
function removeNodes(s, ids) {
  const set = new Set(ids);
  s.nodes = s.nodes.filter(n => !set.has(n.id));
  s.edges = s.edges.filter(e => !set.has(e.a) && !set.has(e.b));
}
function removeEdge(s, a, b) {
  const k = ekey(a, b);
  s.edges = s.edges.filter(e => ekey(e.a, e.b) !== k);
}

/* ---------- geometry ---------- */
function orient(p, q, r) { return (q.x - p.x) * (r.y - p.y) - (q.y - p.y) * (r.x - p.x); }
function sgn(v) { return Math.abs(v) < TOL ? 0 : (v > 0 ? 1 : -1); }
function onSeg(p, q, r) {
  if (Math.abs(orient(p, q, r)) > TOL) return false;
  return r.x >= Math.min(p.x, q.x) - TOL && r.x <= Math.max(p.x, q.x) + TOL &&
         r.y >= Math.min(p.y, q.y) - TOL && r.y <= Math.max(p.y, q.y) + TOL;
}
function segmentsMeet(p1, p2, p3, p4) {
  const s1 = sgn(orient(p3, p4, p1)), s2 = sgn(orient(p3, p4, p2));
  const s3 = sgn(orient(p1, p2, p3)), s4 = sgn(orient(p1, p2, p4));
  if (s1 * s2 < 0 && s3 * s4 < 0) return true;
  return onSeg(p3, p4, p1) || onSeg(p3, p4, p2) || onSeg(p1, p2, p3) || onSeg(p1, p2, p4);
}

/* ---------- structural checks ---------- */
function matrixRank(M, tol) {
  const m = M.length, n = m ? M[0].length : 0;
  const A = M.map(r => r.slice());
  let r = 0;
  for (let c = 0; c < n && r < m; c++) {
    let p = r, best = Math.abs(A[r][c]);
    for (let i = r + 1; i < m; i++) { const v = Math.abs(A[i][c]); if (v > best) { best = v; p = i; } }
    if (best < tol) continue;
    const t = A[r]; A[r] = A[p]; A[p] = t;
    for (let i = r + 1; i < m; i++) {
      const f = A[i][c] / A[r][c];
      if (f !== 0) for (let k = c; k < n; k++) A[i][k] -= f * A[r][k];
    }
    r++;
  }
  return r;
}
function rigidityRank(s) {
  const idx = new Map(); s.nodes.forEach((n, i) => idx.set(n.id, i));
  const nm = nodeMap(s);
  const M = s.edges.map(e => {
    const row = new Array(2 * s.nodes.length).fill(0);
    const p = nm.get(e.a), q = nm.get(e.b);
    const i = idx.get(e.a), k = idx.get(e.b);
    row[2 * i] = p.x - q.x; row[2 * i + 1] = p.y - q.y;
    row[2 * k] = q.x - p.x; row[2 * k + 1] = q.y - p.y;
    return row;
  });
  return matrixRank(M, 1e-7);
}
function isConnected(s) {
  if (!s.nodes.length) return false;
  const adj = adjacency(s), seen = new Set([s.nodes[0].id]), q = [s.nodes[0].id];
  while (q.length) { const u = q.pop(); adj.get(u).forEach((_, v) => { if (!seen.has(v)) { seen.add(v); q.push(v); } }); }
  return seen.size === s.nodes.length;
}
function summary(s) {
  const j = s.nodes.length, b = s.edges.length, need = 2 * j - 3;
  const rank = rigidityRank(s), connected = isConnected(s);
  let status = 'redundant';
  if (b === need && rank === need) status = 'isostatic';
  else if (rank < need) status = 'mechanism';
  return { j, b, need, rank, dof: need - rank, connected, status,
           free: s.edges.filter(e => e.free).length, fixed: s.nodes.filter(n => n.fixed).length };
}
/* kind: 'topology' (a rule bug), 'geometry' (design blocked), 'rank' (degenerate) */
function validate(s) {
  const sm = summary(s);
  if (!sm.connected) return { ok: false, kind: 'topology', reason: 'a joint is disconnected from the truss', sm };
  if (sm.b !== sm.need) return { ok: false, kind: 'topology', reason: 'Maxwell count fails: ' + sm.b + ' members, ' + sm.need + ' needed', sm };
  const nm = nodeMap(s);
  for (let i = 0; i < s.nodes.length; i++) for (let k = i + 1; k < s.nodes.length; k++) {
    const p = s.nodes[i], q = s.nodes[k];
    if (Math.hypot(p.x - q.x, p.y - q.y) < 1e-6) return { ok: false, kind: 'geometry', reason: 'joints ' + p.id + ' and ' + q.id + ' coincide', sm };
  }
  for (const e of s.edges) for (const n of s.nodes) {
    if (n.id === e.a || n.id === e.b) continue;
    if (onSeg(nm.get(e.a), nm.get(e.b), n)) return { ok: false, kind: 'geometry', reason: 'joint ' + n.id + ' lies on member ' + e.a + '–' + e.b, sm };
  }
  for (let i = 0; i < s.edges.length; i++) for (let k = i + 1; k < s.edges.length; k++) {
    const e1 = s.edges[i], e2 = s.edges[k];
    const shared = [e1.a, e1.b].filter(v => v === e2.a || v === e2.b);
    if (shared.length === 2) return { ok: false, kind: 'geometry', reason: 'duplicate member ' + e1.a + '–' + e1.b, sm };
    if (shared.length === 1) {
      const c = nm.get(shared[0]);
      const p = nm.get(e1.a === c.id ? e1.b : e1.a), q = nm.get(e2.a === c.id ? e2.b : e2.a);
      const cr = (p.x - c.x) * (q.y - c.y) - (p.y - c.y) * (q.x - c.x);
      const dt = (p.x - c.x) * (q.x - c.x) + (p.y - c.y) * (q.y - c.y);
      if (Math.abs(cr) < TOL && dt > 0) return { ok: false, kind: 'geometry', reason: 'members ' + e1.a + '–' + e1.b + ' and ' + e2.a + '–' + e2.b + ' overlap', sm };
    } else if (segmentsMeet(nm.get(e1.a), nm.get(e1.b), nm.get(e2.a), nm.get(e2.b))) {
      return { ok: false, kind: 'geometry', reason: 'member ' + e1.a + '–' + e1.b + ' crosses member ' + e2.a + '–' + e2.b, sm };
    }
  }
  if (sm.status !== 'isostatic') return { ok: false, kind: 'rank', reason: 'rigidity rank ' + sm.rank + ' of ' + sm.need + ': the truss would be a mechanism', sm };
  return { ok: true, sm };
}

/* ---------- the five rules (R2 inverts R1, R4 inverts R3, R5 flips back) ---------- */
const RULES = [
  { id: 'R1', name: 'Subdivide', inverse: 'R2',
    text: 'Free member AB of triangle ABC is split at its midpoint P, and a new member joins P to the apex C. One joint and two members are added.',
    matches(s) {
      const adj = adjacency(s), out = [];
      s.edges.filter(e => e.free).forEach(e => common(adj, e.a, e.b).forEach(c =>
        out.push({ a: e.a, b: e.b, c, label: 'AB ' + e.a + '–' + e.b + ' · apex ' + c })));
      return out;
    },
    apply(s, m) {
      const t = cloneState(s), nm = nodeMap(t), P = nextId(t), A = nm.get(m.a), B = nm.get(m.b);
      removeEdge(t, m.a, m.b);
      t.nodes.push({ id: P, x: (A.x + B.x) / 2, y: (A.y + B.y) / 2, fixed: false });
      t.edges.push(mkEdge(m.a, P, true), mkEdge(P, m.b, true), mkEdge(m.c, P, false));
      return t;
    } },
  { id: 'R2', name: 'Merge', inverse: 'R1',
    text: 'A joint P with three members, two free (PA, PB) and one solid (PC), is removed, and free member AB is restored. Inverse of R1.',
    matches(s) {
      const adj = adjacency(s), nm = nodeMap(s), out = [];
      s.nodes.forEach(p => {
        if (p.fixed || adj.get(p.id).size !== 3) return;
        const nb = [...adj.get(p.id).entries()];
        const solid = nb.filter(([, e]) => !e.free), fr = nb.filter(([, e]) => e.free);
        if (solid.length !== 1 || fr.length !== 2) return;
        const c = solid[0][0], a = Math.min(fr[0][0], fr[1][0]), b = Math.max(fr[0][0], fr[1][0]);
        if (adj.get(a).has(c) && adj.get(b).has(c) && !adj.get(a).has(b)) out.push({ p: p.id, a, b, c, label: 'joint ' + p.id });
      });
      return out;
    },
    apply(s, m) {
      const t = cloneState(s);
      removeNodes(t, [m.p]);
      t.edges.push(mkEdge(m.a, m.b, true));
      return t;
    } },
  { id: 'R3', name: 'Extend', inverse: 'R4',
    text: 'At a support A with free member AC, joint D is added beside A. Member AB is replaced by AD, DB and DC, and DB keeps the status of AB. One joint and two members are added.',
    matches(s) {
      const adj = adjacency(s), out = [];
      s.nodes.filter(n => n.fixed).forEach(A => adj.get(A.id).forEach((e, c) => {
        if (!e.free) return;
        common(adj, A.id, c).forEach(b => out.push({ a: A.id, b, c, label: 'support ' + A.id + ' · free ' + A.id + '–' + c + ' · B ' + b }));
      }));
      return out;
    },
    apply(s, m) {
      const t = cloneState(s), nm = nodeMap(t), D = nextId(t);
      const A = nm.get(m.a), B = nm.get(m.b), C = nm.get(m.c);
      const L = Math.hypot(B.x - A.x, B.y - A.y), ux = (B.x - A.x) / L, uy = (B.y - A.y) / L;
      let nx = -uy, ny = ux;
      if (nx * (C.x - A.x) + ny * (C.y - A.y) > 0) { nx = -nx; ny = -ny; }
      const abFree = t.edges.find(e => ekey(e.a, e.b) === ekey(m.a, m.b)).free;
      removeEdge(t, m.a, m.b);
      t.nodes.push({ id: D, x: A.x + 0.2 * L * (ux + nx), y: A.y + 0.2 * L * (uy + ny), fixed: false });
      t.edges.push(mkEdge(m.a, D, false), mkEdge(D, m.b, abFree), mkEdge(D, m.c, false));
      return t;
    } },
  { id: 'R4', name: 'Contract', inverse: 'R3',
    text: 'A three-member joint D next to support A folds into A, and its far neighbour Y is joined to A. One joint and two members are removed. Inverse of R3.',
    matches(s) {
      const adj = adjacency(s), out = [];
      s.nodes.filter(n => n.fixed).forEach(A => adj.get(A.id).forEach((_, d) => {
        const D = s.nodes.find(n => n.id === d);
        if (D.fixed || adj.get(d).size !== 3) return;
        const rest = [...adj.get(d).keys()].filter(k => k !== A.id);
        [[rest[0], rest[1]], [rest[1], rest[0]]].forEach(([x, y]) => {
          if (adj.get(A.id).has(x) && adj.get(x).has(y) && !adj.get(A.id).has(y))
            out.push({ a: A.id, d, x, y, label: 'joint ' + d + ' into support ' + A.id });
        });
      }));
      return out;
    },
    apply(s, m) {
      const t = cloneState(s);
      const dyFree = t.edges.find(e => ekey(e.a, e.b) === ekey(m.d, m.y)).free;
      removeNodes(t, [m.d]);
      t.edges.push(mkEdge(m.a, m.y, dyFree));
      return t;
    } },
  { id: 'R5', name: 'Flip', inverse: 'R5',
    text: 'Interior diagonal AC of a quadrilateral ABCD swaps to the other diagonal BD. Joint and member counts stay the same.',
    matches(s) {
      const adj = adjacency(s), out = [];
      s.edges.filter(e => !e.free).forEach(e => {
        const cn = common(adj, e.a, e.b);
        if (cn.length === 2 && !adj.get(cn[0]).has(cn[1]))
          out.push({ a: e.a, c: e.b, b: cn[0], d: cn[1], label: e.a + '–' + e.b + ' → ' + cn[0] + '–' + cn[1] });
      });
      return out;
    },
    apply(s, m) {
      const t = cloneState(s);
      removeEdge(t, m.a, m.c);
      t.edges.push(mkEdge(m.b, m.d, false));
      return t;
    } }
];
const RULE_BY_ID = Object.fromEntries(RULES.map(r => [r.id, r]));

/* every precondition match with its result and validation */
function enumerate(s) {
  const out = [];
  RULES.forEach(r => r.matches(s).forEach(m => {
    const result = r.apply(s, m), v = validate(result);
    out.push({ rule: r.id, match: m, label: m.label, state: result, ok: v.ok, kind: v.kind, reason: v.reason, sm: v.sm });
  }));
  return out;
}

/* ---------- presets (the "Before" panels of Fig. 4) ---------- */
const PRESETS = {
  triangle: { name: 'Triangle (R1, R3)', state: {
    nodes: [{ id: 0, x: 0, y: 0, fixed: true }, { id: 1, x: 1, y: 0, fixed: false }, { id: 2, x: 0.5, y: 1, fixed: false }],
    edges: [mkEdge(0, 1, true), mkEdge(0, 2, false), mkEdge(1, 2, false)] } },
  strip: { name: 'Two triangles (R4)', state: {
    nodes: [{ id: 0, x: 0, y: 0, fixed: true }, { id: 1, x: 1, y: 0, fixed: false }, { id: 2, x: 1, y: 1, fixed: false }, { id: 3, x: 2, y: 1, fixed: false }],
    edges: [mkEdge(0, 1, false), mkEdge(0, 2, false), mkEdge(1, 2, false), mkEdge(1, 3, false), mkEdge(2, 3, false)] } },
  square: { name: 'Square with diagonal (R5)', state: {
    nodes: [{ id: 0, x: 0, y: 0, fixed: true }, { id: 1, x: 1, y: 0, fixed: false }, { id: 2, x: 1, y: 1, fixed: false }, { id: 3, x: 0, y: 1, fixed: false }],
    edges: [mkEdge(0, 1, false), mkEdge(1, 2, false), mkEdge(2, 3, false), mkEdge(3, 0, false), mkEdge(0, 2, false)] } }
};

/* ---------- canonical form (ids ignored) and seeded random walk ---------- */
function r6(v) { return (Math.round(v * 1e6) / 1e6 + 0).toFixed(6); }
function nkey(n) { return r6(n.x) + ',' + r6(n.y); }
function edgeKeyXY(s, e, nm) {
  const a = nkey(nm.get(e.a)), b = nkey(nm.get(e.b));
  return (a < b ? a + '|' + b : b + '|' + a);
}
function canon(s) {
  const nm = nodeMap(s);
  const ns = s.nodes.map(n => nkey(n) + (n.fixed ? 'F' : '')).sort();
  const es = s.edges.map(e => edgeKeyXY(s, e, nm) + (e.free ? 'f' : 's')).sort();
  return ns.join(';') + '#' + es.join(';');
}
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function randomWalk(s, steps, seed) {
  const rnd = mulberry32(seed), trail = [];
  let cur = s;
  for (let i = 0; i < steps; i++) {
    const options = enumerate(cur).filter(o => o.ok);
    if (!options.length) break;
    const pick = options[Math.floor(rnd() * options.length)];
    trail.push({ rule: pick.rule, label: pick.label, state: pick.state });
    cur = pick.state;
  }
  return trail;
}

/* ---------- self-tests ---------- */
function runTests() {
  const T = [];
  const add = (name, pass, detail, info) => T.push({ name, pass: !!pass, detail: detail || '', info: !!info });
  const first = (s, rule) => enumerate(s).filter(o => o.rule === rule);
  const tri = PRESETS.triangle.state, strip = PRESETS.strip.state, sq = PRESETS.square.state;

  { const o = first(tri, 'R1')[0], n = o && o.state.nodes.find(k => k.id === 3);
    add('R1 splits the free member at its midpoint and adds apex member',
      o && o.ok && o.sm.j === 4 && o.sm.b === 5 && n && Math.abs(n.x - 0.5) < 1e-12 && Math.abs(n.y) < 1e-12 && o.state.edges.some(e => e.a === 2 && e.b === 3 && !e.free),
      'triangle → 4 joints, 5 members, P at (0.5, 0): the right-hand panel of Fig. 4, rule 1'); }
  { const o = first(tri, 'R1')[0], back = first(o.state, 'R2').filter(x => x.ok && canon(x.state) === canon(tri));
    add('R2 undoes R1 exactly', back.length === 1, 'R1 then R2 returns the starting design'); }
  { const o = first(tri, 'R3')[0];
    add('R3 adds one joint and two members at the support',
      o && o.ok && o.sm.j === 4 && o.sm.b === 5 && o.state.nodes.find(n => n.id === 0).fixed && !o.state.edges.some(e => e.a === 0 && e.b === 2),
      'support 0 stays fixed; member 0–2 is replaced by 0–D, D–2, D–1'); }
  { const o = first(tri, 'R3')[0], back = first(o.state, 'R4').filter(x => x.ok && canon(x.state) === canon(tri));
    add('R4 undoes R3 exactly', back.length === 1, 'R3 then R4 returns the starting design'); }
  { const o = first(strip, 'R4').filter(x => x.ok)[0];
    const keys = o && o.state.nodes.map(nkey).sort().join(' ');
    add('R4 contracts the two-triangle strip to one triangle',
      o && o.sm.j === 3 && o.sm.b === 3 && keys === '0.000000,0.000000 1.000000,1.000000 2.000000,1.000000',
      'matches Fig. 4 rule 4: joint 1 folds into support 0, leaving (0,0), (1,1), (2,1)'); }
  { const o = first(sq, 'R5').filter(x => x.ok)[0];
    const has = o && o.state.edges.some(e => e.a === 1 && e.b === 3) && !o.state.edges.some(e => e.a === 0 && e.b === 2);
    const twice = o && first(o.state, 'R5').filter(x => x.ok && canon(x.state) === canon(sq));
    add('R5 flips the diagonal, and flipping twice restores it', has && twice.length === 1, 'diagonal 0–2 becomes 1–3 (Fig. 4 rule 5)'); }
  { const sq0 = cloneState(sq); sq0.edges = sq0.edges.filter(e => !(e.a === 0 && e.b === 2));
    const sm = summary(sq0);
    add('Rank check detects a mechanism (square without a diagonal)', sm.status === 'mechanism' && sm.rank === 4 && sm.need === 5, 'rank 4 of 5, one degree of freedom'); }
  { const bad = cloneState(sq); bad.edges.push(mkEdge(1, 3, false));
    const v = validate(bad);
    add('Maxwell count rejects a redundant member (both diagonals of a square)', !v.ok && v.kind === 'topology' && /Maxwell/.test(v.reason), v.reason); }
  { const bad = cloneState(sq);
    bad.nodes.find(n => n.id === 2).x = 0; bad.nodes.find(n => n.id === 2).y = 1;
    bad.nodes.find(n => n.id === 3).x = 1; bad.nodes.find(n => n.id === 3).y = 1;
    const v = validate(bad);
    add('Geometry check rejects crossing members when the count is right', !v.ok && v.kind === 'geometry' && /crosses/.test(v.reason), v.reason); }
  { const bad = cloneState(tri); bad.nodes.push({ id: 3, x: 2, y: 2, fixed: false });
    const v = validate(bad);
    add('A disconnected joint is rejected', !v.ok && v.kind === 'topology', v.reason); }
  { const r = RULES.find(x => x.id === 'R2'), t = cloneState(strip);
    add('R2 does not match when no joint has the R1 pattern', r.matches(t).length === 0, 'two-triangle strip has no R2 match'); }

  /* property tests over seeded random walks */
  let states = 0, matches = 0, topoBugs = 0, blockedGeo = 0, blockedRank = 0, rtFail = 0, rtChecked = 0, fixedBad = 0, idDup = 0;
  const starts = [tri, strip, sq];
  for (let seed = 1; seed <= 60; seed++) {
    let cur = starts[seed % 3];
    const trail = randomWalk(cur, 10, seed);
    const seq = [cur].concat(trail.map(t => t.state));
    seq.forEach(st => {
      states++;
      if (new Set(st.nodes.map(n => n.id)).size !== st.nodes.length) idDup++;
      const all = enumerate(st);
      all.forEach(o => {
        matches++;
        if (o.kind === 'topology') topoBugs++;
        if (o.kind === 'geometry') blockedGeo++;
        if (o.kind === 'rank') blockedRank++;
        if (o.state.nodes.filter(n => n.fixed).length !== st.nodes.filter(n => n.fixed).length) fixedBad++;
        if (o.ok && (o.rule === 'R1' || o.rule === 'R3' || o.rule === 'R5')) {
          rtChecked++;
          const inv = RULE_BY_ID[RULE_BY_ID[o.rule].inverse];
          const found = enumerate(o.state).some(x => x.rule === inv.id && x.ok && canon(x.state) === canon(st));
          if (!found) rtFail++;
        }
      });
    });
  }
  add('Every rule application keeps Maxwell count and connectivity', topoBugs === 0,
    matches + ' rule applications over ' + states + ' designs; ' + topoBugs + ' count or connectivity failures');
  add('No rule changes the number of supports', fixedBad === 0, fixedBad + ' violations');
  add('Joint ids stay unique', idDup === 0, idDup + ' duplicates');
  add('Forward then inverse restores the design (R1→R2, R3→R4, R5→R5)', rtChecked > 0 && rtFail === 0,
    rtChecked + ' round trips, ' + rtFail + ' failures');
  add('How often the geometry and rank checks blocked a match', true,
    blockedGeo + ' blocked for geometry, ' + blockedRank + ' blocked for rank, out of ' + matches, true);
  return T;
}

if (typeof module !== 'undefined') module.exports = { RULES, RULE_BY_ID, PRESETS, enumerate, validate, summary, canon, randomWalk, runTests, cloneState, mkEdge, mulberry32 };
