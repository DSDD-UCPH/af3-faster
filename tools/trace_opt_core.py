"""Build the opt_core subset used by af3-fast: AST walk + optional runtime dump.

Writes core_manifest.txt (paths relative to common/opt_core/opt_core/).
Env:
  AF3_FASTER_TRACE_RUNTIME=1  also import install_fast + lazy Pallas rows (needs jax/GPU).
  AF3_FASTER_TRACE_OUT=path   override output path.
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parent
REPO = PKG_ROOT.parents[1]
CORE = REPO / "common" / "opt_core" / "opt_core"
FAST = REPO / "af3_jax" / "native_fast" / "af3_fast"
OUT = Path(os.environ.get("AF3_FASTER_TRACE_OUT") or (HERE / "core_manifest.txt"))

# Torch / CUDA trees af3-fast never launches. Keep the sibling JAX packages.
SKIP_PREFIXES = (
    "kernels/triattn/",          # 179M torch CUDA triattn
    "kernels/trimul/",           # 61M torch CUDA trimul
    "kernels/triattn_xla/bin/",  # cubins unused on the fast word (fpf_core / rowshared)
    "kernels/triattn_xla/csrc/",
    "kernels/trimul_xla/pkg/",   # cubins; native_xla row is not the fast winner
    "kernels/transition/",
    "kernels/ln/",
    "kernels/apb/",
    "kernels/fpf_trimul",
    "kernels/fpf_triatt",
    "kernels/fpf_glue",
    "kernels/fpf_mkpf",
    "kernels/fpf_transition",
    "kernels/triattn_exact/",
    "capture/",
    "tools/",
    "of3_sampler/",
    "of3_trunk/",
    "ops/",
    "seq/",
    "host/",
    "host_cache/",
    "diffusion_loop/",
    "jax_design/",
    "precision/",
    "attn/",
    "autoload.py",
    "modes.py",
    "cli.py",
    "process.py",
    "instances.py",
    "manifest.py",
    "stock_proof.py",
    "testing.py",
    "compare.py",
    "det.py",
    "extern.py",
    "home.py",
    "jit_cache.py",
    "strategies.py",
    "shape_policy.py",
    "upstream_fix.py",
    "warm.py",
    "arch.py",
    "jax_arch.py",
    "trimul.py",
    "trimul_weights.py",
)

# Whole trees that the fast path (or a named fallback in serve.py) can load.
FORCE_TREES = (
    "kernels/pallas",
    "kernels/fpf_pallas",
    "kernels/pallas_attn",
    "kernels/pallas_triatt",
    "kernels/triattn_xla",
    "kernels/trimul_xla",
    "mem/rowpair_jax",
)

FORCE_FILES = (
    "__init__.py",
    "oom.py",
    "gates.py",
    "cell_census.py",
    "counters.py",
    "report.py",
    "kernels/__init__.py",
    "kernels/fpf_pallas_serve.py",
    "kernels/fpf_pallas_f32.py",
    "kernels/fpf_pallas_bias.py",
    "kernels/pallas_attn_serve.py",
    "kernels/pallas_glut.py",
    "kernels/cell_words.py",
    "kernels/safe_settings.py",
    "kernels/META/fpf_pallas.json",
    "kernels/META/fpf_pallas_f32.json",
    "kernels/META/pallas.json",
    "kernels/META/pallas_attn.json",
    "kernels/META/pallas_glut.json",
    "kernels/META/pallas_triatt.json",
    "kernels/META/triattn_xla.json",
    "kernels/META/trimul_xla.json",
    "mem/__init__.py",
    "mem/primitives.py",
    "mem/ngpu.py",
    "mem/patchset.py",
)

DATA_SUFFIXES = {".json", ".NOTICE", ".md", ".txt", ".LICENSE"}
DATA_NAMES = {"NOTICE", "LICENSE", "VERSION", "SHA256SUMS"}


def skipped(rel: str) -> bool:
    rel = rel.replace("\\", "/")
    if rel.endswith(".pyc") or "/__pycache__/" in rel or rel.endswith(".pyc"):
        return True
    if rel.split("/")[-1] == "__pycache__":
        return True
    for p in SKIP_PREFIXES:
        if rel == p.rstrip("/") or rel.startswith(p):
            return True
    return False


def is_py(path: Path) -> bool:
    return path.suffix == ".py"


def parse_imports(src: str) -> set[str]:
    """Return opt_core module names referenced by import statements or import_module strings."""
    out: set[str] = set()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return out
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "opt_core" or a.name.startswith("opt_core."):
                    out.add(a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module and (node.module == "opt_core" or node.module.startswith("opt_core.")):
                if node.names and node.names[0].name != "*":
                    for a in node.names:
                        out.add(node.module + "." + a.name)
                out.add(node.module)
        elif isinstance(node, ast.Call):
            func = node.func
            name = ""
            if isinstance(func, ast.Attribute) and func.attr == "import_module":
                name = "import_module"
            elif isinstance(func, ast.Name) and func.id in {"__import__", "import_module"}:
                name = func.id
            if name and node.args:
                arg0 = node.args[0]
                if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str) and arg0.value.startswith("opt_core"):
                    out.add(arg0.value)
    return out


def module_to_rel(mod: str) -> list[str]:
    """opt_core.kernels.pallas.serve -> possible file paths under CORE."""
    if not mod.startswith("opt_core"):
        return []
    parts = mod.split(".")[1:]  # drop opt_core
    if not parts:
        return ["__init__.py"]
    base = "/".join(parts)
    cands = [base + ".py", base + "/__init__.py"]
    return cands


def collect_tree(rel_dir: str) -> set[str]:
    root = CORE / rel_dir
    out: set[str] = set()
    if not root.exists():
        return out
    if root.is_file():
        r = str(Path(rel_dir).as_posix())
        if not skipped(r):
            out.add(r)
        return out
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = str(p.relative_to(CORE).as_posix())
        if skipped(rel):
            continue
        if p.suffix == ".py" or p.suffix in DATA_SUFFIXES or p.name in DATA_NAMES:
            out.add(rel)
    return out


def sidecar_data(py_rel: str) -> set[str]:
    p = CORE / py_rel
    out: set[str] = set()
    parent = p.parent if p.suffix == ".py" else p
    if not parent.is_dir():
        return out
    for child in parent.iterdir():
        if not child.is_file():
            continue
        rel = str(child.relative_to(CORE).as_posix())
        if skipped(rel):
            continue
        if child.suffix in DATA_SUFFIXES or child.name in DATA_NAMES:
            out.add(rel)
    return out


def walk_static() -> set[str]:
    files: set[str] = set()
    for rel in FORCE_FILES:
        if (CORE / rel).is_file() and not skipped(rel):
            files.add(rel)
    for tree in FORCE_TREES:
        files |= collect_tree(tree)
    # Seed modules from af3_fast sources.
    queue: list[str] = []
    for src in FAST.rglob("*.py"):
        text = src.read_text(encoding="utf-8", errors="replace")
        for mod in parse_imports(text):
            queue.append(mod)
    seen_mods: set[str] = set()
    while queue:
        mod = queue.pop()
        if mod in seen_mods:
            continue
        seen_mods.add(mod)
        for cand in module_to_rel(mod):
            if skipped(cand):
                continue
            if (CORE / cand).is_file():
                files.add(cand)
                files |= sidecar_data(cand)
                if cand.endswith(".py"):
                    text = (CORE / cand).read_text(encoding="utf-8", errors="replace")
                    for nxt in parse_imports(text):
                        queue.append(nxt)
                    # relative imports: resolve against this file's package
                    try:
                        tree = ast.parse(text)
                    except SyntaxError:
                        continue
                    pkg_parts = cand[:-3].split("/") if not cand.endswith("__init__.py") else cand[: -len("/__init__.py")].split("/")
                    if cand.endswith("__init__.py"):
                        pkg_parts = cand[: -len("/__init__.py")].split("/") if "/" in cand else []
                    else:
                        pkg_parts = cand[:-3].split("/")[:-1]
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ImportFrom) and node.level:
                            base = pkg_parts[: len(pkg_parts) - (node.level - 1)] if node.level else pkg_parts
                            if node.level >= 1:
                                base = pkg_parts[: max(0, len(pkg_parts) - (node.level - (1 if cand.endswith("__init__.py") else 0)))]
                            # Standard: level=1 is current package
                            cur = cand[:-3].split("/") if not cand.endswith("__init__.py") else cand[: -len("__init__.py")].rstrip("/").split("/")
                            if cand.endswith("__init__.py"):
                                cur_pkg = cand[: -len("/__init__.py")].split("/") if cand != "__init__.py" else []
                            else:
                                cur_pkg = cand[:-3].split("/")[:-1]
                            # go up node.level-1 from current package? ImportFrom level: 1 = from ., 2 = from ..
                            up = node.level - 1
                            parent = cur_pkg[: len(cur_pkg) - up] if up else cur_pkg
                            if node.module:
                                parts = parent + node.module.split(".")
                            else:
                                parts = parent
                            nxt_mod = "opt_core." + ".".join(parts) if parts else "opt_core"
                            queue.append(nxt_mod)
                            if node.names:
                                for a in node.names:
                                    if a.name == "*":
                                        continue
                                    queue.append(nxt_mod + "." + a.name)
    # Always keep NOTICE/LICENSE next to included packages.
    extra: set[str] = set()
    for rel in list(files):
        extra |= sidecar_data(rel)
    files |= extra
    return {r for r in files if (CORE / r).is_file() and not skipped(r)}


def walk_runtime(files: set[str]) -> set[str]:
    sys.path.insert(0, str(REPO / "af3_jax" / "native_fast"))
    sys.path.insert(0, str(REPO / "common" / "opt_core"))
    opened: set[str] = set()
    core_s = str(CORE.resolve())

    orig_open = open

    def tracing_open(file, *a, **k):
        try:
            p = Path(file if not hasattr(file, "name") else file).resolve()
            s = str(p)
            if s.startswith(core_s + os.sep) or s == core_s:
                rel = str(p.relative_to(CORE)).replace("\\", "/")
                if not skipped(rel):
                    opened.add(rel)
        except Exception:
            pass
        return orig_open(file, *a, **k)

    import builtins
    builtins.open = tracing_open  # type: ignore[assignment]
    try:
        os.environ.setdefault("AF3_FLASHPAIRFORMER", "both")
        os.environ.setdefault("AF3_DIFFUSION_HOIST", "0")
        import importlib

        seeds = [
            "opt_core",
            "opt_core.oom",
            "opt_core.gates",
            "opt_core.cell_census",
            "opt_core.counters",
            "opt_core.report",
            "opt_core.kernels",
            "opt_core.kernels.pallas",
            "opt_core.kernels.pallas.serve",
            "opt_core.kernels.fpf_pallas_serve",
            "opt_core.kernels.fpf_pallas",
            "opt_core.kernels.fpf_pallas.trimul_pallas",
            "opt_core.kernels.fpf_pallas.triattn_pallas",
            "opt_core.kernels.fpf_pallas.transition_pallas",
            "opt_core.kernels.fpf_pallas_f32",
            "opt_core.kernels.fpf_pallas_bias",
            "opt_core.kernels.pallas_attn_serve",
            "opt_core.kernels.pallas_attn.af2_flash_pallas",
            "opt_core.kernels.pallas_triatt.triatt_attn",
            "opt_core.kernels.pallas.rowshared_flash_pallas",
            "opt_core.kernels.pallas.cd_trimul.trimul_pallas",
            "opt_core.kernels.pallas.cd_layers.layers_ln",
            "opt_core.kernels.pallas.cd_layers.layers_transition",
            "opt_core.kernels.pallas.mlp_transition.mlp_transition_pallas",
            "opt_core.kernels.pallas_glut",
            "opt_core.kernels.triattn_xla",
            "opt_core.kernels.trimul_xla",
            "opt_core.mem",
            "opt_core.mem.rowpair_jax",
            "opt_core.mem.ngpu",
        ]
        for name in seeds:
            try:
                importlib.import_module(name)
            except Exception as e:
                print(f"runtime import skip {name}: {type(e).__name__}: {e}", file=sys.stderr)
        try:
            from af3_fast.install import apply_fast_env, install_fast
            apply_fast_env()
            install_fast()
        except Exception as e:
            print(f"install_fast skip: {type(e).__name__}: {e}", file=sys.stderr)
    finally:
        builtins.open = orig_open  # type: ignore[assignment]

    for name, mod in list(sys.modules.items()):
        fn = getattr(mod, "__file__", None)
        if not fn:
            continue
        try:
            p = Path(fn).resolve()
            if core_s in str(p):
                rel = str(p.relative_to(CORE)).replace("\\", "/")
                if not skipped(rel):
                    files.add(rel)
        except Exception:
            continue
    files |= {r for r in opened if (CORE / r).is_file() and not skipped(r)}
    return files


def write_manifest(files: set[str]) -> None:
    lines = [
        "# opt_core files vendored into af3_faster._core (paths relative to common/opt_core/opt_core/).",
        "# Regenerated by tools/trace_opt_core.py. Do not hand-edit except to force-add a fallback module.",
        "",
    ]
    for rel in sorted(files, key=lambda s: s.lower()):
        lines.append(rel)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    n_py = sum(1 for r in files if r.endswith(".py"))
    print(f"wrote {OUT}  files={len(files)} py={n_py}")


def main() -> int:
    files = walk_static()
    if os.environ.get("AF3_FASTER_TRACE_RUNTIME") == "1":
        files = walk_runtime(files)
    write_manifest(files)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
