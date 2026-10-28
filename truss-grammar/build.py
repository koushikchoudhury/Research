import pathlib
root = pathlib.Path(__file__).parent
tpl = (root / "template.html").read_text()
core = (root / "core.js").read_text()
assert "/*__CORE__*/" in tpl
out = tpl.replace("/*__CORE__*/", core)
(root / "truss-grammar.html").write_text(out)
print("built", len(out), "bytes")
