# af3-faster

One pip package that speeds up an **existing** AlphaFold 3 install. It does not edit native files and does not change the checkpoint. Fast mode is the default.

**Most of the speedups** (FlashPairformer, Pallas pair/sampler kernels, hoists, and the shared runtime) come from Anthropic’s [uplifting-biomolecular-modeling](https://github.com/anthropics/uplifting-biomolecular-modeling) kit (`af3_jax` and `common/opt_core`). This package wraps that work for an already-installed AlphaFold 3 tree. We added compute-capability **8.9** (RTX 4090, RTX 6000 Ada, L40S, L40) and **12.0** (RTX 5090, RTX 5080, RTX PRO 6000) tile rows. Anthropic already shipped **8.0** (A100), **9.0** (H100, H200), **10.0** (B200), and **10.3** (B300). The ColabFold native tree (AF3, OpenFold3, OpenBind0, and the pair/atom/noise hoists already in ≥3.1.7) is [sokrypton/alphafold3](https://github.com/sokrypton/alphafold3). Apache-2.0; Copyright 2026 Anthropic, PBC. See `LICENSE` and `_core/THIRD_PARTY_NOTICES.md`.

Two native trees:

* **alphafold3-colabfold ≥ 3.1.7** (sokrypton / ColabFold): AF3, OpenFold3, and OpenBind0 (`--model alphafold3|openfold3|openbind0`).
* **google-deepmind/alphafold3** (orig): AF3 weights only. `--model openfold3|openbind0` is refused.

Hoists (do the same work once instead of hundreds of times) are **on by default on both trees**. Colabfold 3.1.7+ already does pair/atom conditioning hoist and noise-level sharing in native code, so the kit copies of those are skipped there. Orig gets the kit copies. Pair-logit hoist (`HOIST_LOGITS`) runs on both.

## Install

```bash
# environment already has alphafold3, jax, haiku, tokamax
pip install af3_faster-0.2.0-py3-none-any.whl
```

AlphaFold 3 must be an editable checkout (so `run_alphafold.py` sits above the `alphafold3` package), or set `AF3_RUN_ALPHAFOLD` to that script. A plain wheel does not ship the runner.

Measured cards: RTX 4090 (cc 8.9) and RTX 5090 (cc 12.0). Anthropic’s table already had cc 8.0 (A100), 9.0 (H100, H200), 10.0 (B200), and 10.3 (B300). Precompiled `triattn_xla` cubins are not in this package, so that row steps aside to another attention kernel.

## Run

```bash
af3-faster --json_path=in.json --output_dir=out --model_dir=weights
af3-faster --mode fast --model openfold3 --json_path=in.json --output_dir=out --model_dir=weights   # colabfold only
af3-faster --mode off  ...          # native, no kit levers
af3-faster doctor                   # stack / GPU / tile / lever preview
```

Turn one lever off: `--no-sampler-bf16`, `--no-hoist-logits`, or `MODEL_OPT_LEVERS_OFF=SAMPLER_BF16,HOIST_LOGITS,DIFFUSION_HOIST`.

## Measured per seed

Poly-alanine 1024 tokens with MSA, 4 seeds, 5 diffusion samples. Times are the average of **seed 2 and seed 3** (`Running model inference with seed N took …`). Seed 1 is omitted (compile). On orig fast, seed 2 can still include extra compile for hoist graphs; seed 3 is the first fully steady seed.

| Tree | Weights | GPU | Off | Fast | Speedup |
|------|---------|-----|-----|------|---------|
| colabfold native | AF3 | 4090 | 67.50s | 34.31s | 1.97× |
| colabfold native | AF3 | 5090 | 36.94s | 22.98s | 1.61× |
| colabfold native | OF3 | 4090 | 69.01s | 33.41s | 2.07× |
| colabfold native | OF3 | 5090 | 37.56s | 22.36s | 1.68× |
| orig | AF3 | 4090 | 78.37s | 47.89s | 1.64× |
| orig | AF3 | 5090 | 43.90s | 34.92s | 1.26× |

Orig fast includes pair-logit hoist. Pair-conditioning hoist (`DIFFUSION_HOIST`) is now also on for orig; the orig fast column above was measured before that extra hoist, so orig may improve further.

Most of the gain is the pairformer kernels (triangle multiply + triangle attention) plus sampler attention/GEMMs. Hoists are “don’t redo work” rather than a new algorithm.

## What each optimization does

Two output classes:

* **Same math** — same operations, often in a different order or once instead of many times. Structures should match native within ordinary GPU noise. Not always bitwise (XLA may fuse differently).
* **Tolerance** — same trained weights, different GPU arithmetic (bf16 tensor cores, flash attention, fused kernels). Coordinates can move a little. Not a different model.

Isolated per-lever timings were not run; “how much” below is the role in the 1024-token fold, not a promise of +X seconds on every job.

### 1. FlashPairformer (triangle multiply + triangle attention)

**From.** Anthropic.

**What.** The pairformer is the slow heart of the trunk: for every residue pair it updates a square table of size N×N. Native does that with large matrix multiplies and attention that materialise huge intermediate tensors. FlashPairformer replaces those two blocks with fused GPU kernels that keep the working set in fast memory.

**How much.** Usually the largest slice of the speedup, especially on the 4090. It runs 48 trunk blocks per recycle, plus MSA, templates, and the confidence head.

**Output.** Tolerance. Same weights; bf16 re-association. Not bitwise vs `--mode off`.

Default: `AF3_FLASHPAIRFORMER=both`. Off: `AF3_FLASHPAIRFORMER=off`.

### 2. TRIMUL_CD and TRIATT_XLA

**From.** Anthropic.

**What.** Same two pairformer sites, served through the kit’s kernel table so the fastest available row for this GPU/JAX/size is used (FlashPairformer’s own row, another Pallas row, or a fallback). If a row cannot run, the next one is used by name — the fold does not die.

**How much.** Overlaps FlashPairformer; they are how those kernels are actually dispatched. Small extra when FPF already hit its fused row.

**Output.** Tolerance, same class as FlashPairformer.

### 3. DATTN (diffusion / pairformer single attention)

**From.** Anthropic.

**What.** During structure sampling the model runs a 24-block transformer on tokens, 200 times, with a pair-bias attention. Native builds a full heads×N×N logit tensor. DATTN uses a flash (blockwise) attention kernel instead.

**How much.** Large in the **sampler**, which dominates after the trunk. Also covers pairformer single attention.

**Output.** Tolerance. Flash softmax is not bitwise vs the stock einsum.

### 4. ATOM_ATTN

**From.** Anthropic.

**What.** Atoms attend in small windows (32 queries × 128 keys), 3 encoder + 3 decoder blocks, every denoising step. Native materialises those window logits; this uses a fused windowed kernel.

**How much.** Medium. Always on in the sampler; also the evoformer atom encoder.

**Output.** Tolerance. Padded empty windows write zeros instead of a mean of V (those rows are masked anyway).

### 5. SAMPLER_BF16

**From.** Anthropic.

**What.** Native AF3 keeps the sampler in float32. This keeps the residual stream in float32 (so sums stay stable) but runs the big GEMMs on bf16 tensor cores.

**How much.** Medium in the sampler (24 token blocks + atom stacks × 200 steps × 5 samples).

**Output.** Tolerance. Not native `bfloat16='all'` (that would bf16 the residual too). Off: `--no-sampler-bf16`.

### 6. TTR (fused transitions)

**From.** Anthropic.

**What.** Pairformer “feed-forward” blocks are LayerNorm → SwiGLU → projection. TTR fuses that so the wide intermediate never hits GPU memory.

**How much.** Medium on the pair stack (C=128). Single-channel C=384 stays on tokamax (no fused row).

**Output.** Tolerance.

### 7. LNP (LayerNorm kernels)

**From.** Anthropic.

**What.** Standalone LayerNorms on pair / MSA / template planes go through a row-wise GPU kernel. Adaptive LayerNorms in the diffusion transformer stay native (different formula).

**How much.** Small–medium. Many calls, each cheaper than attention.

**Output.** Tolerance (f32 stats, different summation).

### 8. DIFFUSION_HOIST (pair conditioning, once per sample)

**From.** sokrypton on colabfold ≥3.1.7 (already in native; kit copy skipped). Anthropic kit copy on orig.

**What.** Each denoising step used to rebuild a token-pair conditioning table (LayerNorm, projection, two transitions on N×N×128) that does **not** depend on the noisy coordinates or the noise level. That is the same work ~200 times. Hoist builds it once and reuses it.

**How much.** Medium–large on **orig** (native orig still rebuilt it every step). On colabfold 3.1.7+ native already does this, so the kit copy is skipped (`native_pair_atom_cond`).

**Output.** Same math. Default **on** (`AF3_DIFFUSION_HOIST=1`), including orig.

### 9. ATOM_COND_HOIST (atom pair table, once per sample)

**From.** sokrypton on colabfold ≥3.1.7 (already in native; kit copy skipped). Anthropic kit copy on orig.

**What.** The atom encoder builds a pair table from the reference conformer and trunk embeddings — also independent of the noisy positions. Native orig rebuilds it every step. This computes it once, including the small pair-logit projections.

**How much.** Medium on orig at large atom counts (memory: the stash grows with atoms). Colabfold native already caches this; kit copy skipped (`native_atom_cond`). Needs diffusion hoist’s once-per-sample step on orig.

**Output.** Same math. Default **on**.

### 10. HOIST_LOGITS (pair-logit projections, once per sample)

**From.** Anthropic.

**What.** After the pair tables exist, each transformer still did LayerNorm + Linear to make per-block attention biases, every step. Those projections are loop-invariant. This runs them once and indexes the result.

* AF3: atom encoder/decoder (the token transformer already shares one projection per super-block inside the step).
* OF3: also the 24 token-transformer blocks (per-block pair LayerNorm).

**How much.** Small on AF3 after compile (~0.4 s/seed at 1024 tokens vs fast without this hoist). First seeds can look slower because the extra precompute graph compiles. Larger on OF3 (24 token blocks). **On by default for orig and colabfold.**

**Output.** Same math. Off: `--no-hoist-logits`.

### 11. COND_SHARE (one noise embedding for all samples in a step)

**From.** sokrypton on colabfold ≥3.1.7 (already in native; kit copy skipped). Anthropic kit copy on orig.

**What.** Five samples are drawn per seed. The noise schedule is the same for all five, but orig carried it inside the per-sample loop, so the noise embedding and AdaLN projections were computed five times. This computes them once per step and broadcasts.

**How much.** Medium on orig (5× less of that work). Colabfold native already shares it; skipped (`native_sample_scan_share`).

**Output.** Same math. Default **on**.

## Defaults on orig vs colabfold

| Lever | From | Fast default | Colabfold native | Orig |
|-------|------|--------------|------------------|------|
| FlashPairformer, DATTN, ATOM_ATTN, SAMPLER_BF16, TTR, TRIMUL_CD, TRIATT_XLA, LNP | Anthropic | on | kit | kit |
| HOIST_LOGITS | Anthropic | on | kit | kit |
| DIFFUSION_HOIST | sokrypton (colabfold); Anthropic (orig) | on | skip (native already) | kit |
| ATOM_COND_HOIST | sokrypton (colabfold); Anthropic (orig) | on | skip (native already) | kit |
| COND_SHARE | sokrypton (colabfold); Anthropic (orig) | on | skip (native already) | kit |

## Rebuild the vendor tree (maintainers)

```bash
python tools/trace_opt_core.py
python tools/build_vendor.py
```

## Attribution

Most of this package is the AlphaFold 3 JAX optimization kit from:

https://github.com/anthropics/uplifting-biomolecular-modeling

That repository is the source of the kernels, levers, and `opt_core` runtime that produce most of the measured speedups above. What we added here is the pip wrapper, orig/colabfold install paths, and compute-capability **8.9** (RTX 4090, RTX 6000 Ada, L40S, L40) and **12.0** (RTX 5090, RTX 5080, RTX PRO 6000) tile rows. Anthropic already shipped **8.0** (A100), **9.0** (H100, H200), **10.0** (B200), and **10.3** (B300).

The ColabFold AlphaFold 3 tree this package runs on is [sokrypton/alphafold3](https://github.com/sokrypton/alphafold3) (≥3.1). That fork is the source of OpenFold3 / OpenBind0 support and of the pair-conditioning, atom-conditioning, and noise-share hoists that native already does there (items 8, 9, and 11; the kit copies are skipped on colabfold). Orig remains [google-deepmind/alphafold3](https://github.com/google-deepmind/alphafold3). Third-party kernel notices stay in `_core/THIRD_PARTY_NOTICES.md`.
