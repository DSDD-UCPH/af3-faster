"""SERVED / LNP / held-kernel lines (cut down from af3_jax_opt.fpf_launch)."""
from __future__ import annotations

import sys

from af3_faster import EXIT_NOT_ACTIVE, PREFIX

FPF_LEVERS = {"trimul": "FPF_TRIMUL", "triatt": "FPF_TRIATT"}


def _off(d, key="installed"):
    return d.get(key) if d else False


def served_line(tree: dict, hoist: dict, skips: dict) -> str:
    try:
        import af3_flashpairformer as fpf
        sites = fpf.served_report()
    except Exception as e:
        from af3_faster._core.oom import is_oom
        if is_oom(e):
            raise
        return f"{PREFIX} SERVED error={type(e).__name__}: {e}"
    counts = {(k, t): 0 for k in ("trimul", "triatt") for t in ("fused", "fallback")}
    shapes = []
    for (kernel, tag, shape, dtype), n in sorted(sites.items(), key=str):
        counts[(kernel, tag)] = counts.get((kernel, tag), 0) + int(n)
        if tag == "fallback":
            shapes.append(f"{kernel}:{'x'.join(str(d) for d in shape)}:{dtype}")
    if hoist.get("probe_error"):
        hoist_w = f"probe_error:{hoist['probe_error']}"
    elif not hoist.get("installed"):
        hoist_w = "native" if skips else "off"
    elif hoist.get("counted"):
        hoist_w = str(hoist.get("calls") or 0)
    else:
        hoist_w = "uncounted"
    held = {}
    try:
        st = fpf.status()
        tiles = str(st.get("tile_table") or "unknown").replace(" ", "_")
        cc = str(st.get("compute_capability") or "unknown").replace(" ", "")
        held = dict(st.get("held") or {})
    except Exception as e:
        from af3_faster._core.oom import is_oom
        if is_oom(e):
            raise
        tiles, cc = f"error:{type(e).__name__}", "unknown"

    def g(name, default):
        mod = tree.get(name)
        return mod.report() if mod else default

    d = g("dattn", {"installed": False, "traced": 0, "sites": {}})
    t = g("ttr", {"installed": False, "fused": 0, "routed": 0, "fallback": {}})
    b = g("sampler_bf16", {"installed": False, "traced": 0, "sites": {}})
    x = g("triatt_xla", {"installed": False, "traced": 0, "rows": {}, "aside": {}})
    aa = g("atom_attn", {"installed": False, "traced": 0, "sites": {}, "fallback": {}})
    h = g("hoist_logits", {"installed": False, "traced": {"step": 0}, "skipped": "", "dtype": None})
    c = g("cond_share", {"installed": False, "traced": 0})
    a = g("atom_cond_hoist", {"installed": False, "precomputed": 0, "enc": 0, "dec": 0, "passed": 0, "aside": {}})
    cd = g("trimul_cd", {"installed": False, "traced": 0, "rows": {}, "aside": {}, "word": None, "uncovered": {}})

    def csvmap(m):
        return ",".join(f"{k}:{v}" for k, v in sorted((m or {}).items())) or "none"

    dattn = str(d["traced"]) if d["installed"] else "off"
    ttr = str(t["fused"]) if t["installed"] else "off"
    sbf16 = str(b["traced"]) if b["installed"] else ("skipped:" + skips.get("SAMPLER_BF16", "off") if "SAMPLER_BF16" in skips else "off")
    txla = str(x["traced"]) if x["installed"] else "off"
    atomattn = str(aa["traced"]) if aa["installed"] else "off"
    if h["installed"]:
        tr = h["traced"]
        hlog = f"tok:{tr['step']},enc:{tr['atom_enc']},dec:{tr['atom_dec']}"
    else:
        hlog = ("skipped:" + skips.get("HOIST_LOGITS", "off") if "HOIST_LOGITS" in skips else "off")
    cshare = str(c["traced"]) if c["installed"] else ("skipped:" + skips.get("COND_SHARE", "off") if "COND_SHARE" in skips else "off")
    achoist = str(a.get("precomputed") or 0) if a["installed"] else ("skipped:" + skips.get("ATOM_COND_HOIST", "off") if "ATOM_COND_HOIST" in skips else "off")
    tcd = str(cd["traced"]) if cd["installed"] else "off"
    line = (
        f"{PREFIX} SERVED trimul fused={counts[('trimul', 'fused')]} fallback={counts[('trimul', 'fallback')]} "
        f"triatt fused={counts[('triatt', 'fused')]} fallback={counts[('triatt', 'fallback')]} "
        f"fallback_shapes={','.join(shapes) or 'none'} hoist={hoist_w} tiles={tiles} cc={cc} "
        f"dattn={dattn} dattn_sites={csvmap(d.get('sites'))} ttr={ttr} ttr_routed={t.get('routed', 0)} "
        f"ttr_fallback={csvmap(t.get('fallback'))} sbf16={sbf16} sbf16_sites={csvmap(b.get('sites'))} "
        f"txla={txla} txla_rows={csvmap(x.get('rows'))} txla_aside={csvmap(x.get('aside'))} "
        f"atomattn={atomattn} atomattn_sites={csvmap(aa.get('sites'))} atomattn_fallback={csvmap(aa.get('fallback'))} "
        f"hlog={hlog} hlog_dtype={(h.get('dtype') or 'none')} cshare={cshare} achoist={achoist} "
        f"achoist_sites=enc:{a.get('enc', 0)},dec:{a.get('dec', 0)},passed:{a.get('passed', 0)} "
        f"achoist_aside={csvmap(a.get('aside'))} cnoise=off cnoise_rule=none "
        f"tcd={tcd} tcd_rows={csvmap(cd.get('rows'))} tcd_aside={csvmap(cd.get('aside'))} "
        f"tcd_word={cd.get('word') or 'none'} tcd_uncovered={csvmap(cd.get('uncovered'))}"
    )
    if held:
        line += " held=" + ",".join(f"{k}:{w}" for k, w in sorted(held.items()))
    return line


def lnp_line(tree: dict) -> str:
    mod = tree.get("lnp")
    if mod is None:
        return f"{PREFIX} LNP served=off word=none rows=none units=none routed=none aside=none uncovered=none"
    try:
        return mod.line()
    except Exception as e:
        from af3_faster._core.oom import is_oom
        if is_oom(e):
            raise
        return f"{PREFIX} LNP served=error:{type(e).__name__} word=none rows=none units=none routed=none aside=none uncovered=none"


def held_kernels() -> dict:
    try:
        import af3_flashpairformer as fpf
        return dict((fpf.status() or {}).get("held") or {})
    except Exception as e:
        from af3_faster._core.oom import is_oom
        if is_oom(e):
            raise
        return {}


def refuse_held() -> None:
    held = held_kernels()
    if not held:
        return
    names = ",".join(
        f"{FPF_LEVERS.get(k, k)}={w}" for k, w in sorted(held.items(), key=lambda kw: FPF_LEVERS.get(kw[0], kw[0]))
    )
    print(
        f"{PREFIX} NOT ACTIVE: {names}: the lever cannot engage on this GPU "
        f"(af3_faster._core Pallas serve refuses its compute capability); exit {EXIT_NOT_ACTIVE}",
        flush=True,
    )
    sys.exit(EXIT_NOT_ACTIVE)


def print_report(tree: dict, hoist: dict, skips: dict) -> None:
    sys.stdout.flush()
    print(served_line(tree, hoist, skips), flush=True)
    print(lnp_line(tree), flush=True)
    try:
        from af3_faster._core.kernels.pallas import serve as PS
        dattn = tree.get("dattn")
        if dattn and hasattr(dattn, "census_line"):
            print(dattn.census_line(), flush=True)
    except Exception:
        pass
