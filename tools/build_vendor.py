"""Regenerate af3_faster/ from native_fast + the pruned opt_core manifest.

    python tools/build_vendor.py

Idempotent. Overwrites af3_faster/af3_faster/ (the import package) including _core/.
Does not touch pyproject.toml, tests/, or this tools/ directory.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parent
REPO = PKG_ROOT.parents[1]
FAST_SRC = REPO / "af3_jax" / "native_fast" / "af3_fast"
CORE_SRC = REPO / "common" / "opt_core" / "opt_core"
CORE_PKG = REPO / "common" / "opt_core"
MANIFEST = HERE / "core_manifest.txt"
DST_PKG = PKG_ROOT / "af3_faster"
DST_CORE = DST_PKG / "_core"
NEW_MOD = "af3_faster._core"

OPT_CORE_RE = re.compile(r"(?<![A-Za-z0-9_])opt_core(?![A-Za-z0-9_])")
AF3_FAST_IDENT_RE = re.compile(r"(?<![A-Za-z0-9_])af3_fast(?![A-Za-z0-9_])")

SKIP_NAMES = {"__pycache__", ".pyc"}

EXTRA_LICENSES = (
    (CORE_PKG / "LICENSE", DST_CORE / "LICENSE"),
    (CORE_PKG / "THIRD_PARTY_NOTICES.md", DST_CORE / "THIRD_PARTY_NOTICES.md"),
    (CORE_PKG / "third_party_licenses" / "jax-xla-ffi.Apache-2.0.txt", DST_CORE / "third_party_licenses" / "jax-xla-ffi.Apache-2.0.txt"),
    (CORE_PKG / "third_party_licenses" / "Triton.MIT.txt", DST_CORE / "third_party_licenses" / "Triton.MIT.txt"),
    (CORE_PKG / "LICENSE", PKG_ROOT / "LICENSE"),
)


def die(msg: str) -> None:
    print(f"build_vendor: {msg}", file=sys.stderr)
    raise SystemExit(1)


def load_manifest() -> list[str]:
    if not MANIFEST.is_file():
        die(f"missing {MANIFEST}; run tools/trace_opt_core.py first")
    out = []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


def rewrite_opt_core(text: str) -> str:
    return OPT_CORE_RE.sub(NEW_MOD, text)


def rewrite_wrapper(text: str) -> str:
    text = text.replace("af3-fast", "af3-faster")
    text = AF3_FAST_IDENT_RE.sub("af3_faster", text)
    text = text.replace("AF3_FAST_BOOTED", "AF3_FASTER_BOOTED")
    text = text.replace("AF3_FAST_REPO", "AF3_FASTER_REPO")
    text = rewrite_opt_core(text)
    return text


def copy_file(src: Path, dst: Path, rewriter) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix == ".py":
        text = src.read_text(encoding="utf-8")
        dst.write_text(rewriter(text), encoding="utf-8")
    else:
        shutil.copy2(src, dst)


def copy_wrapper() -> None:
    if not FAST_SRC.is_dir():
        die(f"missing {FAST_SRC}")
    for src in FAST_SRC.rglob("*"):
        if not src.is_file():
            continue
        if "__pycache__" in src.parts or src.suffix == ".pyc":
            continue
        rel = src.relative_to(FAST_SRC)
        copy_file(src, DST_PKG / rel, rewrite_wrapper)


def copy_core(files: list[str]) -> None:
    missing = [r for r in files if not (CORE_SRC / r).is_file()]
    if missing:
        die("manifest paths missing in opt_core:\n  " + "\n  ".join(missing[:20]))
    for rel in files:
        src = CORE_SRC / rel
        copy_file(src, DST_CORE / rel, rewrite_opt_core)
    (DST_CORE / "py.typed").write_text("", encoding="utf-8")
    for src, dst in EXTRA_LICENSES:
        if src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info() -> dict:
    def run(args):
        try:
            return subprocess.check_output(args, cwd=REPO, text=True).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return ""

    sha = run(["git", "rev-parse", "HEAD"])
    dirty = run(["git", "diff", "--name-only", "--", "common/opt_core"])
    ver_src = (CORE_SRC / "__init__.py").read_text(encoding="utf-8")
    m = re.search(r'__version__\s*=\s*"([^"]+)"', ver_src)
    return {
        "git_sha": sha,
        "dirty_opt_core_files": [l for l in dirty.splitlines() if l.strip()],
        "upstream_version": m.group(1) if m else "unknown",
    }


def write_vendored(files: list[str]) -> None:
    info = git_info()
    digest = {rel: sha256_file(DST_CORE / rel) for rel in files if (DST_CORE / rel).is_file()}
    doc = {
        "upstream_package": "opt_core",
        "upstream_version": info["upstream_version"],
        "git_sha": info["git_sha"],
        "folded_uncommitted": [
            "kernels/fpf_pallas_serve.py TILE_TABLES own:8.9 (RTX 4090) and own:12.0 (RTX 5090)",
            "kernels/pallas/serve.py probe: eager_constant_folding(False) instead of ensure_compile_time_eval",
        ],
        "dirty_opt_core_files": info["dirty_opt_core_files"],
        "files": digest,
    }
    (DST_CORE / "VENDORED.json").write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def apply_wrapper_extras() -> None:
    """Native discovery, doctor, and version checks on top of the copied af3-fast sources."""
    compat = DST_PKG / "compat.py"
    text = compat.read_text(encoding="utf-8")
    if "ENV_RUN" not in text:
        text = text.replace(
            'ENV_REPO = "AF3_FASTER_REPO"',
            'ENV_REPO = "AF3_FASTER_REPO"\nENV_RUN = "AF3_RUN_ALPHAFOLD"\nMIN_NATIVE = (3, 1, 7)\n'
            'FLAVOR_COLABFOLD = "colabfold"\nFLAVOR_ORIG = "orig"\nFLAVOR_MISSING = "missing"',
        )
    old_find = '''def find_run_alphafold(environ=os.environ) -> str:
    """Native ``run_alphafold.py``. Editable install: walk up from alphafold3.__file__."""
    for key in (ENV_REPO, "AF3_JAX_REPO"):
        root = (environ.get(key) or "").strip()
        if root:
            p = os.path.join(os.path.abspath(root), "run_alphafold.py")
            if os.path.isfile(p):
                return p
'''
    new_find = '''def find_run_alphafold(environ=os.environ) -> str:
    """Native ``run_alphafold.py``. Editable install: walk up from alphafold3.__file__."""
    explicit = (environ.get(ENV_RUN) or "").strip()
    if explicit:
        p = os.path.abspath(explicit)
        if os.path.isfile(p):
            return p
        raise FileNotFoundError(f"{PREFIX} {ENV_RUN}={explicit} is not a file")
    for key in (ENV_REPO, "AF3_FAST_REPO", "AF3_JAX_REPO"):
        root = (environ.get(key) or "").strip()
        if root:
            p = os.path.join(os.path.abspath(root), "run_alphafold.py")
            if os.path.isfile(p):
                return p
'''
    if old_find not in text:
        die("compat.find_run_alphafold preamble not found for overlay")
    text = text.replace(old_find, new_find)
    text = text.replace(
        'f"{PREFIX} cannot find run_alphafold.py next to alphafold3; set {ENV_REPO} to the native checkout"',
        'f"{PREFIX} cannot find run_alphafold.py next to alphafold3; set {ENV_RUN} to the native script (or {ENV_REPO} to the checkout)"',
    )
    old_ok = '''def native_stack_ok() -> tuple[bool, str]:
    """Whether the importable alphafold3 is the colabfold tree (decoded_ccd + cpp)."""
    try:
        import alphafold3  # noqa: F401
    except Exception as e:
        return False, f"alphafold3_missing:{type(e).__name__}:{e}"
    try:
        from alphafold3.constants import decoded_ccd  # noqa: F401
    except Exception:
        return False, "alphafold3_not_native:missing_decoded_ccd"
    try:
        import alphafold3.cpp  # noqa: F401
    except Exception as e:
        return False, f"alphafold3_cpp_missing:{type(e).__name__}"
    return True, getattr(alphafold3, "__file__", "imported")
'''
    new_ok = '''def _version_tuple(text: str) -> tuple[int, ...]:
    parts = []
    for tok in re.split(r"[^0-9]+", str(text or "").strip()):
        if tok.isdigit():
            parts.append(int(tok))
    return tuple(parts or (0,))


def native_version() -> str:
    """Installed alphafold3 version (colabfold dist, orig dist, or alphafold3.version)."""
    try:
        import importlib.metadata as md
        for name in ("alphafold3-colabfold", "alphafold3"):
            try:
                return md.version(name)
            except Exception:
                continue
    except Exception:
        pass
    try:
        from alphafold3.version import __version__ as ver
        if ver:
            return str(ver)
    except Exception:
        pass
    try:
        import alphafold3
        ver = getattr(alphafold3, "__version__", None)
        if ver:
            return str(ver)
    except Exception:
        pass
    return "0"


def native_has_decoded_ccd() -> bool:
    try:
        from alphafold3.constants import decoded_ccd  # noqa: F401
        return True
    except Exception:
        return False


def native_flavor() -> str:
    """``colabfold`` (sokrypton OF3 fork), ``orig`` (google-deepmind), or ``missing``."""
    try:
        import alphafold3  # noqa: F401
    except Exception:
        return FLAVOR_MISSING
    if native_has_decoded_ccd():
        return FLAVOR_COLABFOLD
    return FLAVOR_ORIG


def of3_supported(flavor: Optional[str] = None) -> bool:
    """OF3 / openbind0 exist only on alphafold3-colabfold (``--model`` + decoded_ccd)."""
    return (flavor or native_flavor()) == FLAVOR_COLABFOLD


def native_stack_ok() -> tuple[bool, str]:
    """Importable alphafold3 with cpp + jax/haiku/tokamax. Colabfold also needs decoded_ccd at >= 3.1.7."""
    import importlib
    try:
        import alphafold3  # noqa: F401
    except Exception as e:
        return False, f"alphafold3_missing:{type(e).__name__}:{e}"
    flavor = native_flavor()
    ver = native_version()
    if flavor == FLAVOR_COLABFOLD and _version_tuple(ver) < MIN_NATIVE:
        return False, f"alphafold3_too_old:{ver}<3.1.7"
    try:
        import alphafold3.cpp  # noqa: F401
    except Exception as e:
        return False, f"alphafold3_cpp_missing:{type(e).__name__}"
    for name in ("jax", "haiku", "tokamax"):
        try:
            importlib.import_module(name)
        except Exception as e:
            return False, f"{name}_missing:{type(e).__name__}:{e}"
    return True, f"{flavor}:{getattr(alphafold3, '__file__', 'imported')}"
'''
    if "import re" not in text.split("def ", 1)[0]:
        text = text.replace("import inspect\nimport os\n", "import inspect\nimport os\nimport re\n")
    if old_ok not in text:
        die("compat.native_stack_ok not found for overlay")
    text = text.replace(old_ok, new_ok)
    old_param = '''def native_has_param(fn, name: str) -> bool:
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def native_has_conditioning_only() -> bool:
'''
    new_param = '''def native_has_param(fn, name: str) -> bool:
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def call_native(fn, *args, **kwargs):
    """Call ``fn``, dropping kwargs the native signature does not accept."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(*args, **kwargs)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return fn(*args, **kwargs)
    accepted = {
        name
        for name, p in sig.parameters.items()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return fn(*args, **{k: v for k, v in kwargs.items() if k in accepted})


def native_has_conditioning_only() -> bool:
'''
    if old_param not in text:
        die("compat.native_has_param not found for overlay")
    text = text.replace(old_param, new_param, 1)
    if "def drop_model_flag(" not in text:
        text = text.rstrip() + '''


def drop_model_flag(argv: list[str]) -> list[str]:
    """Orig has no ``--model``; strip it so absl does not see an unknown flag."""
    out = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok.startswith("--model=") and not tok.startswith("--model_dir"):
            i += 1
            continue
        if tok == "--model" and i + 1 < len(argv) and not argv[i + 1].startswith("-"):
            i += 2
            continue
        out.append(tok)
        i += 1
    return out
'''
        text += "\n"
    compat.write_text(text, encoding="utf-8")

    main_py = DST_PKG / "__main__.py"
    mtext = main_py.read_text(encoding="utf-8")
    needle = '''def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    _reexec_if_needed()
'''
    insert = '''def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "doctor":
        _reexec_if_needed()
        from af3_faster.doctor import main as doctor_main
        doctor_main(argv[1:])
        return
    _reexec_if_needed()
'''
    if needle not in mtext:
        die("__main__.main preamble not found for overlay")
    mtext = mtext.replace(needle, insert, 1)
    old_script = """    script = compat.find_run_alphafold()
    repo = os.path.dirname(script)
"""
    new_script = """    try:
        script = compat.find_run_alphafold()
    except FileNotFoundError as e:
        print(f"{e}; exit {EXIT_NOT_ACTIVE}", flush=True)
        sys.exit(EXIT_NOT_ACTIVE)
    repo = os.path.dirname(script)
"""
    if old_script not in mtext:
        die("__main__ find_run_alphafold call not found for overlay")
    mtext = mtext.replace(old_script, new_script, 1)
    mtext = mtext.replace(
        "from af3_faster import EXIT_NOT_ACTIVE, FAST_MODELS, PREFIX",
        "from af3_faster import EXIT_NOT_ACTIVE, FAST_MODELS, OF3_MODELS, PREFIX",
        1,
    )
    old_gate = '''def _gate_model(argv: list[str]) -> str:
    model = compat.parse_model_flag(argv)
    if not compat.fast_allowed(model):
'''
    new_gate = '''def _gate_model(argv: list[str]) -> str:
    model = compat.parse_model_flag(argv)
    flavor = compat.native_flavor()
    if model in OF3_MODELS and not compat.of3_supported(flavor):
        print(
            f"{PREFIX} NOT ACTIVE: model={model} OF3 weights are not available on google-deepmind alphafold3 "
            f"(use alphafold3-colabfold for --model {'|'.join(OF3_MODELS)}); "
            f"exit {EXIT_NOT_ACTIVE}",
            flush=True,
        )
        sys.exit(EXIT_NOT_ACTIVE)
    if not compat.fast_allowed(model):
'''
    if old_gate not in mtext:
        die("__main__._gate_model not found for overlay")
    mtext = mtext.replace(old_gate, new_gate, 1)
    old_fast = '''    if mode == "fast":
        model = _gate_model(rest)
'''
    new_fast = '''    if mode == "fast" or (
        compat.parse_model_flag(rest) in OF3_MODELS and not compat.of3_supported()
    ):
        model = _gate_model(rest)
'''
    if old_fast not in mtext:
        die("__main__ fast gate call not found for overlay")
    mtext = mtext.replace(old_fast, new_fast, 1)
    old_stack = '''            f"{PREFIX} NOT ACTIVE: {where}; pip install -e the native checkout "
            f"(AF3_FASTER_REPO={repo}) so importable alphafold3 matches {script}; "
            f"exit {EXIT_NOT_ACTIVE}",
'''
    new_stack = '''            f"{PREFIX} NOT ACTIVE: {where}; pip install -e the native checkout "
            f"(google-deepmind/alphafold3 or alphafold3-colabfold; AF3_FASTER_REPO={repo}) "
            f"so importable alphafold3 matches {script}; "
            f"exit {EXIT_NOT_ACTIVE}",
'''
    if old_stack not in mtext:
        die("__main__ native_stack_ok message not found for overlay")
    mtext = mtext.replace(old_stack, new_stack, 1)
    old_off = '''        sys.exit(EXIT_NOT_ACTIVE)
    if mode == "off":
'''
    new_off = '''        sys.exit(EXIT_NOT_ACTIVE)
    if not compat.of3_supported():
        rest = compat.drop_model_flag(rest)
    if mode == "off":
'''
    if old_off not in mtext:
        die("__main__ drop_model_flag insertion point not found for overlay")
    mtext = mtext.replace(old_off, new_off, 1)
    main_py.write_text(mtext, encoding="utf-8")

    init_py = DST_PKG / "__init__.py"
    itext = init_py.read_text(encoding="utf-8")
    itext = itext.replace('__version__ = "0.1.0"', '__version__ = "0.2.0"')
    init_py.write_text(itext, encoding="utf-8")

    sbf = DST_PKG / "levers" / "sampler_bf16.py"
    st = sbf.read_text(encoding="utf-8")
    if "from af3_faster import compat" not in st:
        st = st.replace("from af3_faster import PREFIX\n", "from af3_faster import PREFIX\nfrom af3_faster import compat\n", 1)
    st = st.replace(
        "return stock_aln(x, single_cond, name, global_config=global_config, atom=atom, **kwargs)",
        "return compat.call_native(\n                stock_aln, x, single_cond, name, global_config=global_config, atom=atom, **kwargs\n            )",
        1,
    )
    st = st.replace(
        "return stock_azi(x, num_channels, single_cond, global_config, name, project=project, atom=atom, **kwargs)",
        "return compat.call_native(\n                stock_azi, x, num_channels, single_cond, global_config, name, project=project, atom=atom, **kwargs\n            )",
        1,
    )
    st = st.replace(
        '"compute_dtype": U.compute_dtype,',
        '"compute_dtype": getattr(U, "compute_dtype", None),\n        "utils": U,',
        1,
    )
    st = st.replace(
        "U.compute_dtype = _force_sampler_f32(U.compute_dtype)",
        "if _STATE[\"stock\"][\"compute_dtype\"] is not None:\n        U.compute_dtype = _force_sampler_f32(_STATE[\"stock\"][\"compute_dtype\"])",
        1,
    )
    st = st.replace(
        "U.compute_dtype = st[\"compute_dtype\"]",
        "if st.get(\"compute_dtype\") is not None:\n        st[\"utils\"].compute_dtype = st[\"compute_dtype\"]",
        1,
    )
    sbf.write_text(st, encoding="utf-8")

    dattn = DST_PKG / "levers" / "dattn.py"
    dt = dattn.read_text(encoding="utf-8")
    if "from af3_faster import compat" not in dt:
        dt = dt.replace("import os\nimport re\n", "import os\nimport re\n\nfrom af3_faster import compat\n", 1)
    dt = dt.replace(
        "x = DT.adaptive_layernorm(x, single_cond, name=name, global_config=global_config)",
        "x = compat.call_native(DT.adaptive_layernorm, x, single_cond, name=name, global_config=global_config)",
        1,
    )
    dattn.write_text(dt, encoding="utf-8")

    atom = DST_PKG / "levers" / "atom_attn.py"
    at = atom.read_text(encoding="utf-8")
    if "from af3_faster import compat" not in at:
        at = at.replace(
            "import importlib.util\nimport os\nimport sys\n",
            "import importlib.util\nimport os\nimport sys\n\nfrom af3_faster import compat\n",
            1,
        )
    at = at.replace(
        'x_q = DT.adaptive_layernorm(x_q, single_cond_q, name=f"{name}q", global_config=global_config, atom=True)',
        'x_q = compat.call_native(DT.adaptive_layernorm, x_q, single_cond_q, name=f"{name}q", global_config=global_config, atom=True)',
        1,
    )
    at = at.replace(
        'x_k = DT.adaptive_layernorm(x_k, single_cond_k, name=f"{name}k", global_config=global_config, atom=True)',
        'x_k = compat.call_native(DT.adaptive_layernorm, x_k, single_cond_k, name=f"{name}k", global_config=global_config, atom=True)',
        1,
    )
    at = at.replace(
        "return DT.adaptive_zero_init(weighted_avg, x_q.shape[-1], single_cond_q, global_config, name, atom=True)",
        "return compat.call_native(DT.adaptive_zero_init, weighted_avg, x_q.shape[-1], single_cond_q, global_config, name, atom=True)",
        1,
    )
    atom.write_text(at, encoding="utf-8")

    shutil.copy2(HERE / "overlay" / "doctor.py", DST_PKG / "doctor.py")


def leftover_opt_core(root: Path) -> list[str]:
    hits = []
    for p in root.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        text = p.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), 1):
            if OPT_CORE_RE.search(line):
                hits.append(f"{p.relative_to(root)}:{i}:{line.strip()[:160]}")
    return hits


def unresolved_lazy_imports(root: Path) -> list[str]:
    """import_module('af3_faster._core....') must resolve to a vendored file."""
    bad = []
    pkg = "af3_faster._core"
    for p in root.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except SyntaxError as e:
            bad.append(f"{p}: syntax {e}")
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            is_imp = (isinstance(func, ast.Attribute) and func.attr == "import_module") or (
                isinstance(func, ast.Name) and func.id in {"__import__", "import_module"}
            )
            if not is_imp or not node.args:
                continue
            arg0 = node.args[0]
            if not isinstance(arg0, ast.Constant) or not isinstance(arg0.value, str):
                continue
            name = arg0.value
            if not name.startswith(pkg):
                continue
            rel = name[len(pkg) :].lstrip(".").replace(".", "/")
            if not rel:
                continue
            if not ((root / (rel + ".py")).is_file() or (root / rel / "__init__.py").is_file()):
                bad.append(f"{p.relative_to(DST_PKG)}:{node.lineno} import_module({name!r}) not vendored")
    return bad


def main() -> int:
    files = load_manifest()
    if DST_PKG.exists():
        shutil.rmtree(DST_PKG)
    DST_PKG.mkdir(parents=True)
    copy_wrapper()
    copy_core(files)
    apply_wrapper_extras()
    write_vendored(files)
    hits = leftover_opt_core(DST_PKG)
    if hits:
        die("leftover opt_core references:\n  " + "\n  ".join(hits[:40]))
    lazy = unresolved_lazy_imports(DST_CORE)
    if lazy:
        die("lazy import to unvendored module:\n  " + "\n  ".join(lazy[:40]))
    n_py = sum(1 for p in DST_PKG.rglob("*.py") if "__pycache__" not in p.parts)
    size = sum(p.stat().st_size for p in DST_PKG.rglob("*") if p.is_file())
    print(f"vendored {DST_PKG}  py={n_py} bytes={size} (~{size/1e6:.1f}MB)  core_files={len(files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
