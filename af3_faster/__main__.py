"""af3-faster CLI: native run_alphafold.py with fast levers in-process.

    af3-faster --mode fast --json_path=... --output_dir=... --model_dir=... [--model alphafold3|openfold3]

``--mode off`` runs native unchanged. All other flags pass through to native.
Sampler hybrid bf16 (f32 residual, bf16 GEMMs), pair-logit hoist, and (on
google-deepmind orig) pair/atom conditioning hoist are on in fast. Disable with
``--no-sampler-bf16`` / ``--no-hoist-logits``, or
``AF3_JAX_SAMPLER_BF16=0`` / ``AF3_JAX_HOIST_LOGITS=0`` /
``AF3_DIFFUSION_HOIST=0``, or
``MODEL_OPT_LEVERS_OFF=SAMPLER_BF16,HOIST_LOGITS,DIFFUSION_HOIST``.
``--model openfold3|openbind0`` needs alphafold3-colabfold; google-deepmind orig is AF3 only.
"""
from __future__ import annotations

import os
import sys

from af3_faster import EXIT_NOT_ACTIVE, FAST_MODELS, OF3_MODELS, PREFIX
from af3_faster import compat
from af3_faster import host
from af3_faster import install as inst

XLA_DEFAULT = "--xla_gpu_enable_triton_gemm=false"


def _consume_fast_flags(argv: list[str]) -> list[str]:
    """Strip af3-faster-only flags; set lever env. Native never sees these."""
    out = []
    i = 0
    off = {"0", "off", "false", "no"}

    def _set(env_name: str, value: str) -> None:
        os.environ[env_name] = value

    while i < len(argv):
        tok = argv[i]
        if tok in ("--no-sampler-bf16", "--sampler-bf16=0", "--sampler-bf16=off", "--sampler-bf16=false"):
            _set("AF3_JAX_SAMPLER_BF16", "0")
            i += 1
            continue
        if tok in ("--sampler-bf16=1", "--sampler-bf16=on", "--sampler-bf16=true"):
            _set("AF3_JAX_SAMPLER_BF16", "1")
            i += 1
            continue
        if tok == "--sampler-bf16":
            if i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                _set("AF3_JAX_SAMPLER_BF16", "0" if argv[i + 1].strip().lower() in off else "1")
                i += 2
            else:
                _set("AF3_JAX_SAMPLER_BF16", "1")
                i += 1
            continue
        if tok in ("--no-hoist-logits", "--hoist-logits=0", "--hoist-logits=off"):
            _set("AF3_JAX_HOIST_LOGITS", "0")
            i += 1
            continue
        if tok in ("--hoist-logits", "--hoist-logits=1", "--hoist-logits=on"):
            _set("AF3_JAX_HOIST_LOGITS", "1")
            i += 1
            continue
        out.append(tok)
        i += 1
    return out


def _split_mode(argv: list[str]) -> tuple[str, list[str]]:
    mode = "fast"
    out = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok.startswith("--mode="):
            mode = tok.split("=", 1)[1] or mode
            i += 1
            continue
        if tok == "--mode" and i + 1 < len(argv):
            mode = argv[i + 1]
            i += 2
            continue
        out.append(tok)
        i += 1
    return mode.strip().lower(), out


def _ensure_env() -> None:
    os.environ.setdefault("XLA_FLAGS", XLA_DEFAULT)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
    os.environ.setdefault("XLA_CLIENT_MEM_FRACTION", "0.90")
    os.environ.setdefault("PYTHONHASHSEED", "0")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")


def _reexec_if_needed() -> None:
    """XLA_FLAGS must be set before jax is imported."""
    if os.environ.get("AF3_FASTER_BOOTED") == "1":
        return
    os.environ["AF3_FASTER_BOOTED"] = "1"
    _ensure_env()
    os.execvpe(sys.executable, [sys.executable, "-m", "af3_faster", *sys.argv[1:]], os.environ)


def _gate_model(argv: list[str]) -> str:
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
        print(
            f"{PREFIX} NOT ACTIVE: model={model} (fast levers validated for AF3-architecture trunks only: "
            f"{','.join(FAST_MODELS)}); "
            f"exit {EXIT_NOT_ACTIVE}",
            flush=True,
        )
        sys.exit(EXIT_NOT_ACTIVE)
    return model


def _gate_weights(argv: list[str], model: str) -> None:
    model_dir = compat.parse_model_dir(argv)
    ok, where = compat.of3_weights_present(model, model_dir)
    if not ok:
        print(f"{PREFIX} NOT ACTIVE: weights={where} model={model}; pass --model_dir; exit {EXIT_NOT_ACTIVE}", flush=True)
        sys.exit(EXIT_NOT_ACTIVE)
    print(f"{PREFIX} GATE model={model} weights={where}", flush=True)


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "doctor":
        _reexec_if_needed()
        from af3_faster.doctor import main as doctor_main
        doctor_main(argv[1:])
        return
    _reexec_if_needed()
    mode, rest = _split_mode(argv)
    rest = _consume_fast_flags(rest)
    if mode not in ("off", "fast"):
        sys.exit(f"{PREFIX} --mode must be off|fast, got {mode!r}")
    if mode == "fast" or (
        compat.parse_model_flag(rest) in OF3_MODELS and not compat.of3_supported()
    ):
        model = _gate_model(rest)
    try:
        script = compat.find_run_alphafold()
    except FileNotFoundError as e:
        print(f"{e}; exit {EXIT_NOT_ACTIVE}", flush=True)
        sys.exit(EXIT_NOT_ACTIVE)
    repo = os.path.dirname(script)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    ok, where = compat.native_stack_ok()
    if not ok:
        print(
            f"{PREFIX} NOT ACTIVE: {where}; pip install -e the native checkout "
            f"(google-deepmind/alphafold3 or alphafold3-colabfold; AF3_FASTER_REPO={repo}) "
            f"so importable alphafold3 matches {script}; "
            f"exit {EXIT_NOT_ACTIVE}",
            flush=True,
        )
        sys.exit(EXIT_NOT_ACTIVE)
    if not compat.of3_supported():
        rest = compat.drop_model_flag(rest)
    if mode == "off":
        print(f"{PREFIX} MODE off script={script}", flush=True)
        sys.argv = [script, *rest]
        import runpy
        runpy.run_path(script, run_name="__main__")
        return
    _gate_weights(rest, model)
    rest, _b = host.inject_buckets(rest)
    host.describe_fix1(script)
    host.describe_row_levers()
    inst.apply_fast_env()
    print(f"{PREFIX} MODE fast script={script}", flush=True)
    inst.install_fast()
    sys.argv = [script, *rest]
    import runpy
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
