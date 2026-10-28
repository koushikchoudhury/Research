# Truss Grammar Workbench (Phase 1)

A planar truss design grammar with five rewrite rules (R1 Subdivide, R2 Merge, R3 Extend,
R4 Contract, R5 Flip), rebuilt from Fig. 4 of Shea, Stankovic, Agrawal, Cagan, McComb,
"Expanding the Generative Power of Large Language Models for Design Through Formal Design
Grammars and Languages", J. Comput. Inf. Sci. Eng. 25(12), 120802, 2025.

NOTE: the rules are reconstructed from the figure and text, not from the paper's supplementary
Python. Compare them with the supplement before presenting this as the paper's grammar.

## Files
- truss-grammar.html  Interactive page. Double-click to open in any browser. Nothing to install.
- core.js             The grammar: rules, checks, random walk, self-tests. No dependencies.
- run_tests.js        Runs the self-tests in Node:  node run_tests.js
- template.html       Page source. build.py inlines core.js into it to make truss-grammar.html.
- build.py            python3 build.py

## Run it
- Browser: open truss-grammar.html (the Google Fonts link is optional; it falls back if offline).
- Tests:   node run_tests.js   (Node 16 or newer)

## Put it online
- GitHub Pages: push truss-grammar.html to a repo, rename it index.html, then
  Settings > Pages > Deploy from branch.
- Or drag the file into netlify.com/drop, or paste it into a CodePen / StackBlitz HTML project.

## Where the rules live
core.js, the RULES array. Each rule has matches(state) (precondition) and apply(state, match).
validate(state) runs the Maxwell count, geometry and rigidity-rank checks.
