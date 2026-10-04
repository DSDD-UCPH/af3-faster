"""CPU tests for af3-faster (no GPU)."""
from __future__ import annotations

import os
import re
import types
from pathlib import Path

import pytest

from af3_faster import EXIT_NOT_ACTIVE, FAST_MODELS, PREFIX, __version__, compat, host
from af3_faster.install import FAST_ENV, TREE_LEVERS, apply_fast_env

PKG = Path(__file__).resolve().parents[1] / "af3_faster"


def test_parse_model_flag():
    assert compat.parse_model_flag(["--json_path=x"]) == "alphafold3"
    assert compat.parse_model_flag(["--model=openfold3"]) == "openfold3"
    assert compat.parse_model_flag(["--model", "openbind0", "--out=y"]) == "openbind0"


def test_fast_allowed():
    assert compat.fast_allowed("alphafold3") and compat.fast_allowed("openfold3")
    assert not compat.fast_allowed("protenix") and not compat.fast_allowed("boltz2")


def test_is_of3_from_model_and_flag():
    gc = types.SimpleNamespace(model="openfold3", of3_weights=False)
    assert compat.is_of3(gc) and compat.model_name(gc) == "openfold3"
    gc2 = types.SimpleNamespace(of3_weights=True)
    assert compat.is_of3(gc2)
    gc3 = types.SimpleNamespace(model="alphafold3", of3_weights=False)
    assert not compat.is_of3(gc3)


def test_ending_bias_transposed_falls_back_without_native():
    gc = types.SimpleNamespace(model="openfold3")
    assert compat.ending_bias_transposed(gc, True) is True
    assert compat.ending_bias_transposed(gc, False) is False
    assert compat.ending_bias_transposed(types.SimpleNamespace(model="alphafold3"), True) is False


def test_kernel_tile_buckets():
    b = host.kernel_tile_buckets()
    assert b[0] == 64 and b[-1] == 5120 and 1024 in b and 32 not in b
    argv, src = host.inject_buckets(["--json_path=x.json"])
    assert src == "kernel_tile" and argv[0].startswith("--buckets=64,128")
    argv2, src2 = host.inject_buckets(["--buckets=256,512", "--json_path=x.json"])
    assert src2 == "caller" and argv2[0].startswith("--buckets=256")


def test_apply_fast_env(monkeypatch):
    monkeypatch.delenv("AF3_FLASHPAIRFORMER", raising=False)
    monkeypatch.delenv("AF3_JAX_SAMPLER_BF16", raising=False)
    monkeypatch.delenv("AF3_JAX_HOIST_LOGITS", raising=False)
    monkeypatch.delenv("AF3_DIFFUSION_HOIST", raising=False)
    monkeypatch.delenv("AF3_JAX_COND_SHARE", raising=False)
    monkeypatch.delenv("AF3_JAX_ATOM_COND_HOIST", raising=False)
    monkeypatch.delenv("MODEL_OPT_LEVERS_OFF", raising=False)
    apply_fast_env()
    assert os.environ["AF3_FLASHPAIRFORMER"] == "both"
    assert os.environ["AF3_DIFFUSION_HOIST"] == "1"
    assert os.environ["AF3_JAX_TTR"] == "fast"
    assert os.environ["AF3_JAX_TRIMUL_CD"] == "fast"
    assert os.environ["AF3_JAX_SAMPLER_BF16"] == "1"
    assert os.environ["AF3_JAX_HOIST_LOGITS"] == "1"
    assert os.environ["AF3_JAX_COND_SHARE"] == "1"
    assert os.environ["AF3_JAX_ATOM_COND_HOIST"] == "1"


def test_sampler_bf16_default_on_can_disable(monkeypatch):
    from af3_faster.install import _skip_reason
    from af3_faster.levers import sampler_bf16 as sbf
    monkeypatch.delenv("AF3_JAX_SAMPLER_BF16", raising=False)
    monkeypatch.delenv("MODEL_OPT_LEVERS_OFF", raising=False)
    apply_fast_env()
    assert sbf.wanted() is True
    assert _skip_reason("SAMPLER_BF16") is None
    monkeypatch.setenv("AF3_JAX_SAMPLER_BF16", "0")
    assert sbf.wanted() is False
    monkeypatch.setenv("AF3_JAX_SAMPLER_BF16", "1")
    monkeypatch.setenv("MODEL_OPT_LEVERS_OFF", "SAMPLER_BF16")
    apply_fast_env()
    assert os.environ["AF3_JAX_SAMPLER_BF16"] == "0"
    assert sbf.wanted() is False


def test_consume_fast_flags_cli(monkeypatch):
    monkeypatch.delenv("AF3_JAX_SAMPLER_BF16", raising=False)
    monkeypatch.delenv("AF3_JAX_HOIST_LOGITS", raising=False)
    from af3_faster.__main__ import _consume_fast_flags
    rest = _consume_fast_flags(["--json_path=a.json", "--no-sampler-bf16", "--output_dir=/tmp"])
    assert rest == ["--json_path=a.json", "--output_dir=/tmp"]
    assert os.environ["AF3_JAX_SAMPLER_BF16"] == "0"
    rest = _consume_fast_flags(["--sampler-bf16", "1", "--model=openfold3"])
    assert rest == ["--model=openfold3"]
    assert os.environ["AF3_JAX_SAMPLER_BF16"] == "1"
    rest = _consume_fast_flags(["--no-hoist-logits", "--json_path=x.json"])
    assert rest == ["--json_path=x.json"]
    assert os.environ["AF3_JAX_HOIST_LOGITS"] == "0"


def test_hoist_logits_default_on_can_disable(monkeypatch):
    from af3_faster.install import _skip_reason
    from af3_faster.levers import hoist_logits as hl
    monkeypatch.delenv("AF3_JAX_HOIST_LOGITS", raising=False)
    monkeypatch.delenv("MODEL_OPT_LEVERS_OFF", raising=False)
    apply_fast_env()
    assert os.environ["AF3_JAX_HOIST_LOGITS"] == "1"
    assert hl.wanted() is True
    assert _skip_reason("HOIST_LOGITS") is None
    monkeypatch.setenv("AF3_JAX_HOIST_LOGITS", "0")
    assert hl.wanted() is False
    monkeypatch.setenv("AF3_JAX_HOIST_LOGITS", "1")
    monkeypatch.setenv("MODEL_OPT_LEVERS_OFF", "HOIST_LOGITS")
    apply_fast_env()
    assert os.environ["AF3_JAX_HOIST_LOGITS"] == "0"
    assert hl.wanted() is False


def test_per_block_pair_ln():
    assert compat.per_block_pair_ln(types.SimpleNamespace(model="openfold3")) is True
    assert compat.per_block_pair_ln(types.SimpleNamespace(model="alphafold3")) is False
    assert compat.per_block_pair_ln(types.SimpleNamespace(model="openbind0")) is False


def test_hoist_logits_native_safe_targets():
    from af3_faster.install import NATIVE_SKIP, _skip_reason
    from af3_faster.levers import hoist_logits as hl
    assert "HOIST_LOGITS" not in NATIVE_SKIP
    assert "ATOM_COND_HOIST" in NATIVE_SKIP
    assert _skip_reason("HOIST_LOGITS") is None
    assert any(t.endswith(":DiffusionHead") for t in hl.REBINDS)
    assert any("Model._sample_diffusion" in t for t in hl.REBINDS)
    assert not any("__call__" in t for t in hl.REBINDS)
    assert hl.INSTALL_AFTER_ADDON is True


def test_force_sampler_f32_wrapper():
    try:
        import jax.numpy as jnp
    except Exception:
        return
    from af3_faster.levers.sampler_bf16 import _force_sampler_f32

    def stock(_gc, sampler=False):
        return jnp.bfloat16

    fn = _force_sampler_f32(stock)
    assert fn(None, sampler=True) == jnp.float32
    assert fn(None, sampler=False) == jnp.bfloat16


def test_tree_lever_order():
    names = [n for n, _ in TREE_LEVERS]
    assert names[0] == "cond_share" and names[-1] == "hoist_logits"
    assert names.index("triatt_xla") < names.index("trimul_cd")
    assert "dattn" in names and "ttr" in names and "lnp" in names


def test_split_mode_roundtrip():
    from af3_faster.__main__ import _split_mode
    mode, rest = _split_mode(["--mode", "off", "--json_path=a.json", "--model=openfold3"])
    assert mode == "off" and rest == ["--json_path=a.json", "--model=openfold3"]
    mode, rest = _split_mode(["--mode=fast", "--output_dir=/tmp"])
    assert mode == "fast" and "--mode" not in rest


def test_opt_core_tile_tables():
    from af3_faster._core.kernels import fpf_pallas_serve as S
    assert "8.9" in S.TILE_TABLES and "12.0" in S.TILE_TABLES
    assert S.tiles_label("8.9") == "own:8.9"
    assert S.tiles_label("12.0") == "own:12.0"
    assert S.attn_cfg(1024, "8.9")["bq"] == 64
    assert S.attn_cfg(1024, "12.0")["bq"] == 128


def test_probe_fix_in_serve_source():
    import inspect
    from af3_faster._core.kernels.pallas import serve as PS
    src = inspect.getsource(PS._walk)
    assert "eager_constant_folding(False)" in src
    assert "with jax.ensure_compile_time_eval():" not in src


def test_prefix():
    assert PREFIX == "[af3-faster]"
    assert EXIT_NOT_ACTIVE == 3
    assert "openfold3" in FAST_MODELS
    assert __version__ == "0.2.0"


def test_gate_refuses_protenix():
    from af3_faster.__main__ import _gate_model
    with pytest.raises(SystemExit) as ei:
        _gate_model(["--model=protenix", "--json_path=x.json"])
    assert ei.value.code == EXIT_NOT_ACTIVE
    if compat.of3_supported():
        assert _gate_model(["--model=openfold3"]) == "openfold3"
    else:
        assert _gate_model(["--json_path=x.json"]) == "alphafold3"


def test_of3_weights_from_dir(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    ok, where = compat.of3_weights_present("alphafold3", None)
    assert not ok and where == "no --model_dir"
    ok, where = compat.of3_weights_present("alphafold3", str(empty))
    assert not ok and "no_weights_in" in where
    p = tmp_path / "w"
    p.mkdir()
    (p / "openfold3.bin.zst").write_bytes(b"x")
    (p / "af3.bin.zst").write_bytes(b"y")
    ok, where = compat.of3_weights_present("openfold3", str(p))
    assert ok and where.endswith("openfold3.bin.zst")
    ok, where = compat.of3_weights_present("alphafold3", str(p))
    assert ok and where.endswith("af3.bin.zst")


def test_native_has_fix1_from_script(tmp_path):
    script = tmp_path / "run_alphafold.py"
    script.write_text("class ModelRunner:\n    def __init__(self):\n        self._autotune_attempted = False\n")
    assert compat.native_has_fix1(str(script))
    script.write_text("class ModelRunner:\n    pass\n")
    assert not compat.native_has_fix1(str(script))


def test_main_protenix_exits_before_native(monkeypatch):
    monkeypatch.setenv("AF3_FASTER_BOOTED", "1")
    from af3_faster.__main__ import main
    with pytest.raises(SystemExit) as ei:
        main(["--mode", "fast", "--model", "protenix", "--json_path=x.json"])
    assert ei.value.code == EXIT_NOT_ACTIVE


def test_find_run_alphafold_env(tmp_path, monkeypatch):
    (tmp_path / "run_alphafold.py").write_text("# stub\n")
    monkeypatch.setenv("AF3_FASTER_REPO", str(tmp_path))
    monkeypatch.delenv("AF3_RUN_ALPHAFOLD", raising=False)
    assert compat.find_run_alphafold() == str(tmp_path / "run_alphafold.py")


def test_find_run_alphafold_explicit_script(tmp_path, monkeypatch):
    script = tmp_path / "run_alphafold.py"
    script.write_text("# stub\n")
    monkeypatch.setenv("AF3_RUN_ALPHAFOLD", str(script))
    monkeypatch.delenv("AF3_FASTER_REPO", raising=False)
    assert compat.find_run_alphafold() == str(script)


def test_no_opt_core_module_path_in_sources():
    ident = re.compile(r"(?<![A-Za-z0-9_])opt_core(?![A-Za-z0-9_])")
    hits = []
    for p in PKG.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if ident.search(line):
                hits.append(f"{p.relative_to(PKG)}:{i}")
    assert hits == []


def test_vendored_modules_import():
    import importlib
    mods = [
        "af3_faster._core",
        "af3_faster._core.oom",
        "af3_faster._core.gates",
        "af3_faster._core.cell_census",
        "af3_faster._core.kernels",
        "af3_faster._core.kernels.pallas",
        "af3_faster._core.kernels.pallas.serve",
        "af3_faster._core.kernels.fpf_pallas_serve",
        "af3_faster._core.kernels.fpf_pallas",
        "af3_faster._core.mem",
        "af3_faster.doctor",
        "af3_faster.install",
    ]
    for name in mods:
        importlib.import_module(name)


def test_pallas_cells_via_resources():
    import importlib.resources as ir
    root = ir.files("af3_faster._core.kernels.pallas")
    data = (root / "PALLAS_CELLS.json").read_text(encoding="utf-8")
    assert '"cells"' in data and '"rows"' in data
    vend = ir.files("af3_faster._core").joinpath("VENDORED.json").read_text(encoding="utf-8")
    assert "0.5.229.0" in vend


def test_doctor_module_importable():
    from af3_faster.doctor import main as doctor_main
    assert callable(doctor_main)


def test_version_tuple():
    assert compat._version_tuple("3.1.14") >= (3, 1, 7)
    assert compat._version_tuple("3.1.6") < (3, 1, 7)


def test_native_version_reads_installed_dist():
    ver = compat.native_version()
    assert ver and ver != "0"
    if compat.native_flavor() == compat.FLAVOR_COLABFOLD:
        assert compat._version_tuple(ver) >= (3, 1, 7)
    else:
        assert compat._version_tuple(ver) >= (3, 0, 0)


def test_native_flavor_matches_installed_alphafold3():
    flavor = compat.native_flavor()
    assert flavor in (compat.FLAVOR_COLABFOLD, compat.FLAVOR_ORIG)
    assert compat.of3_supported() is (flavor == compat.FLAVOR_COLABFOLD)
    assert compat.native_has_decoded_ccd() is (flavor == compat.FLAVOR_COLABFOLD)


def test_call_native_drops_unknown_kwargs():
    def orig(x, name):
        return (x, name)

    assert compat.call_native(orig, 1, name="n", global_config="gc", atom=True) == (1, "n")

    def colab(x, name, global_config=None, atom=False):
        return (x, name, global_config, atom)

    assert compat.call_native(colab, 1, name="n", global_config="gc", atom=True) == (1, "n", "gc", True)


def test_drop_model_flag_keeps_model_dir():
    argv = ["--json_path=a.json", "--model", "openfold3", "--model_dir=/w", "--output_dir=/o"]
    assert compat.drop_model_flag(argv) == ["--json_path=a.json", "--model_dir=/w", "--output_dir=/o"]
    assert compat.drop_model_flag(["--model=alphafold3", "--model_dir=/w"]) == ["--model_dir=/w"]


def test_gate_refuses_of3_on_orig(monkeypatch):
    monkeypatch.setattr(compat, "native_flavor", lambda: compat.FLAVOR_ORIG)
    from af3_faster.__main__ import _gate_model
    with pytest.raises(SystemExit) as ei:
        _gate_model(["--model=openfold3", "--json_path=x.json"])
    assert ei.value.code == EXIT_NOT_ACTIVE
    assert _gate_model(["--json_path=x.json"]) == "alphafold3"


def test_main_of3_on_orig_exits_before_native(monkeypatch):
    monkeypatch.setenv("AF3_FASTER_BOOTED", "1")
    monkeypatch.setattr(compat, "native_flavor", lambda: compat.FLAVOR_ORIG)
    monkeypatch.setattr(compat, "of3_supported", lambda flavor=None: False)
    from af3_faster.__main__ import main
    with pytest.raises(SystemExit) as ei:
        main(["--mode", "fast", "--model", "openfold3", "--json_path=x.json"])
    assert ei.value.code == EXIT_NOT_ACTIVE


def test_skip_hoist_logits_on_orig(monkeypatch):
    from af3_faster.install import _skip_reason
    monkeypatch.setattr(compat, "native_has_conditioning_only", lambda: False)
    assert _skip_reason("HOIST_LOGITS") is None
    assert _skip_reason("DIFFUSION_HOIST") is None


def test_skip_diffusion_hoist_on_colabfold(monkeypatch):
    from af3_faster.install import _skip_reason
    monkeypatch.setattr(compat, "native_has_conditioning_only", lambda: True)
    assert _skip_reason("DIFFUSION_HOIST") == "native_pair_atom_cond"


def test_main_missing_script_exits(monkeypatch, tmp_path):
    monkeypatch.setenv("AF3_FASTER_BOOTED", "1")
    monkeypatch.setenv("AF3_RUN_ALPHAFOLD", str(tmp_path / "missing_run_alphafold.py"))
    from af3_faster.__main__ import main
    with pytest.raises(SystemExit) as ei:
        main(["--mode", "off", "--json_path=x.json"])
    assert ei.value.code == EXIT_NOT_ACTIVE
