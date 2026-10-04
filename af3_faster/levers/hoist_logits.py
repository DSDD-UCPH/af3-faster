"""HOIST_LOGITS — pair-logits hoist (OF3 token transformer + atom encoder/decoder).

Kit FPF_HOIST + HOIST_LOGITS cannot replace native ``DiffusionHead.__call__`` (it would drop
``conditioning_only`` / ``pair_cond`` / ``atom_cond``). Native already builds pair/atom cond
once per ``sample()``; this lever only lifts the pair-logit projections out of the 200-step
scan and indexes them in place.

  * OF3 token transformer (``PER_BLOCK_PAIR_LAYER_NORM``): each of 24 blocks' LN-scale + Linear
    over [N, N, 128] runs once per sample, then ``lax.dynamic_index_in_dim`` feeds the resident
    [24, H, N, N] view. AF3 / openbind keep the shared LN + per-super-block projection in-step.
  * Atom encoder/decoder: native already caches ``pair_act``; the shared ``LayerNorm + Linear
    (num_blocks, heads)`` still ran every step. Those projections run once per sample too.
    Models with ``PER_BLOCK_ATOM_PAIR_LAYER_NORM`` keep the in-block Linear (weights differ).
  * Orig (no ``conditioning_only``): wrap ``Model._sample_diffusion`` to build atom ``pair_act``
    once per sample and stash encoder/decoder pair logits. Token OF3 hoist is N/A.

Params stay under the same Haiku names (layer_stack name forced to
``__layer_stack_no_per_layer`` even with per-layer inputs, matching native). DATTN / TTR /
SAMPLER_BF16 / ATOM_ATTN compose: the step calls module-global ``self_attention`` /
``transition_block`` / ``cross_attention``.

Switch: ``AF3_JAX_HOIST_LOGITS=1`` (fast default). Off: ``=0``, ``--no-hoist-logits``,
``MODEL_OPT_LEVERS_OFF=HOIST_LOGITS``.
"""
from __future__ import annotations

import os

from af3_faster import PREFIX
from af3_faster import compat

ENV_SWITCH = "AF3_JAX_HOIST_LOGITS"
REBINDS = (
    "alphafold3.model.network.diffusion_transformer:Transformer",
    "alphafold3.model.network.diffusion_transformer:CrossAttTransformer",
    "alphafold3.model.network.diffusion_head:DiffusionHead",
    "alphafold3.model.model:Model._sample_diffusion",
)
INSTALL_AFTER_ADDON = True
STORE_DTYPE = "float32"
LS = "__layer_stack_no_per_layer"
OFF_WORDS = {"0", "off", "false", "no"}
_STATE = {
    "installed": False,
    "dtype": None,
    "traced": {"precompute": 0, "step": 0, "atom_pre": 0, "atom_enc": 0, "atom_dec": 0},
    "skipped": "",
    "stock": {},
    "orig": False,
}
_STASH: dict = {}


def wanted(environ=os.environ) -> bool:
    return (environ.get(ENV_SWITCH, "1") or "1").strip().lower() not in OFF_WORDS


def _store_dtype():
    import jax.numpy as jnp
    return {"bfloat16": jnp.bfloat16, "float32": jnp.float32}[STORE_DTYPE]


def _atom_key(name: str) -> str | None:
    n = str(name)
    if n.endswith("atom_transformer_encoder") and "evoformer" not in n:
        return "enc"
    if n.endswith("atom_transformer_decoder"):
        return "dec"
    return None


def _shared_atom_pair_logits(gc) -> bool:
    """AF3/OF3 share one LN+Linear outside the atom stack; OpenDDE/RF3/Protenix do not."""
    try:
        from alphafold3.model import model_config
        names = getattr(model_config, "PER_BLOCK_ATOM_PAIR_LAYER_NORM", ())
        return getattr(gc, "model", None) not in names
    except Exception:
        return True


def _create_offset(gc, name: str):
    try:
        from alphafold3.model import model_config
        fn = getattr(model_config, "affine_norm", None)
        if fn is not None:
            return fn(getattr(gc, "model", None), name)
    except Exception:
        pass
    return False


def _make_transformer(DT):
    import jax
    import jax.numpy as jnp
    from alphafold3.model import model_config

    Stock = DT.Transformer
    dt = _store_dtype()
    _STATE["dtype"] = jnp.dtype(dt).name

    class HoistTransformer(Stock):
        def __call__(self, act, mask, single_cond, pair_cond, extra_pair_bias=None):
            # Precompute must live in ``__call__`` (Haiku does not prefix it) so LN/Linear
            # names match the checkpoint. A named method becomes ``~precompute_pair_logits/``.
            if _STASH.get("pre") and pair_cond is not None and compat.per_block_pair_ln(self.global_config):
                pair_hat = DT._normalized(pair_cond)
                create_offset = _create_offset(self.global_config, "pair_input_layer_norm")
                nsb = self.config.num_blocks // self.config.super_block_size
                sbs = self.config.super_block_size
                nhead = self.config.attention.num_head

                def blk(c, _i):
                    lg = DT._pair_logits_from_normalized(
                        pair_hat, nhead, pair_cond.dtype, create_offset
                    )
                    return c, jnp.transpose(lg, [2, 0, 1]).astype(dt)

                def sb(c, _i):
                    c, lgs = DT._stack(
                        sbs, blk, self.config.block_remat, with_per_layer_inputs=True, name=LS
                    )(c, jnp.arange(sbs, dtype=jnp.int32))
                    return c, lgs

                dummy = jnp.zeros((), jnp.float32)
                _, logits = DT._stack(nsb, sb, False, with_per_layer_inputs=True, name=LS)(
                    dummy, jnp.arange(nsb, dtype=jnp.int32)
                )
                _STATE["traced"]["precompute"] += 1
                return logits

            pre = _STASH.get("token")
            if pre is None or pair_cond is None or not compat.per_block_pair_ln(self.global_config):
                return compat.call_native(
                    super().__call__,
                    act, mask, single_cond, pair_cond,
                    extra_pair_bias=extra_pair_bias,
                )
            nsb = self.config.num_blocks // self.config.super_block_size
            sbs = self.config.super_block_size
            flat = jnp.reshape(pre, (nsb * sbs,) + tuple(pre.shape[2:]))
            chai = self.global_config.model == "chai1"
            kq_norm = self.global_config.model == "rosettafold3"
            parallel = self.global_config.model in ("rosettafold3", "chai1")
            trans_mask = (
                DT._trans_mask(self.global_config, mask) if hasattr(DT, "_trans_mask") else None
            )

            def super_block(act, i):
                def block(act, j):
                    lg = jax.lax.dynamic_index_in_dim(flat, i * sbs + j, keepdims=False)
                    if extra_pair_bias is not None:
                        lg = lg + extra_pair_bias[None].astype(lg.dtype)
                    attn = DT.self_attention(
                        act, mask, lg, self.config.attention, self.global_config, single_cond,
                        name=self.name, kq_norm=kq_norm, use_gating_query=not chai,
                    )
                    if parallel:
                        act = act + attn + compat.call_native(
                            DT.transition_block,
                            act, self.config.num_intermediate_factor, self.global_config, single_cond,
                            name=self.name, mask=trans_mask,
                        )
                    else:
                        act += attn
                        act += compat.call_native(
                            DT.transition_block,
                            act, self.config.num_intermediate_factor, self.global_config, single_cond,
                            name=self.name, mask=trans_mask,
                        )
                    return act, None

                act, _ = DT._stack(
                    sbs, block, self.config.block_remat, with_per_layer_inputs=True, name=LS
                )(act, jnp.arange(sbs, dtype=jnp.int32))
                return act, None

            act, _ = DT._stack(nsb, super_block, False, with_per_layer_inputs=True, name=LS)(
                act, jnp.arange(nsb, dtype=jnp.int32)
            )
            _STATE["traced"]["step"] += 1
            return act

    HoistTransformer.__module__ = Stock.__module__
    return HoistTransformer


def _make_atom_transformer(DT):
    import haiku as hk
    import jax.numpy as jnp
    from alphafold3.model import model_config
    from alphafold3.model.atom_layout import atom_layout

    Stock = DT.CrossAttTransformer
    hm = DT.hm

    class HoistCrossAttTransformer(Stock):
        def __call__(
            self,
            queries_act,
            queries_mask,
            queries_to_keys,
            keys_mask,
            queries_single_cond,
            keys_single_cond,
            pair_cond,
            pair_mask=None,
            rope_q=None,
            rope_k=None,
        ):
            which = _atom_key(self.name)
            if _STASH.get("pre") and which and pair_cond is not None:
                pair_act = hm.LayerNorm(
                    name="pair_input_layer_norm",
                    use_fast_variance=False,
                    create_offset=_create_offset(self.global_config, "pair_input_layer_norm"),
                )(pair_cond)
                pair_logits = hm.Linear(
                    (self.config.num_blocks, self.config.attention.num_head),
                    name="pair_logits_projection",
                )(pair_act)
                _STATE["traced"]["atom_pre"] += 1
                return jnp.transpose(pair_logits, [3, 0, 4, 1, 2])

            pre = (_STASH.get("atom") or {}).get(self.name) if which else None
            if pre is None:
                return compat.call_native(
                    super().__call__,
                    queries_act, queries_mask, queries_to_keys, keys_mask,
                    queries_single_cond, keys_single_cond, pair_cond,
                    pair_mask=pair_mask, rope_q=rope_q, rope_k=rope_k,
                )
            chai = getattr(self.global_config, "model", None) == "chai1"
            mask_per_block = getattr(self.global_config, "model", None) in getattr(
                model_config, "MASK_ATOM_ACT_PER_BLOCK", ()
            )
            trans_mask = (
                DT._trans_mask(self.global_config, queries_mask)
                if hasattr(DT, "_trans_mask") else None
            )

            def block(queries_act, pair_logits):
                if mask_per_block:
                    queries_act = queries_act * queries_mask[..., None].astype(queries_act.dtype)
                block_in = queries_act
                keys_act = atom_layout.convert(queries_to_keys, queries_act, layout_axes=(-3, -2))
                attn = compat.call_native(
                    DT.cross_attention,
                    x_q=queries_act, x_k=keys_act, mask_q=queries_mask, mask_k=keys_mask,
                    config=self.config.attention, global_config=self.global_config,
                    pair_logits=pair_logits, single_cond_q=queries_single_cond,
                    single_cond_k=keys_single_cond, name=self.name,
                    pair_mask=pair_mask, rope_q=rope_q, rope_k=rope_k,
                )
                trans_in = block_in if chai else (queries_act + attn)
                trans = compat.call_native(
                    DT.transition_block,
                    trans_in, self.config.num_intermediate_factor, self.global_config,
                    queries_single_cond, name=self.name, atom=True, mask=trans_mask,
                )
                return block_in + attn + trans, None

            stacked, _ = hk.experimental.layer_stack(
                self.config.num_blocks, with_per_layer_inputs=True
            )(block)(queries_act, pre)
            _STATE["traced"]["atom_enc" if which == "enc" else "atom_dec"] += 1
            return stacked

    HoistCrossAttTransformer.__module__ = Stock.__module__
    return HoistCrossAttTransformer


def _atom_pair_act(head, batch, embeddings, trunk_pair_cond):
    """Orig encoder's step-invariant atom-pair activations (verbatim ops, name='diffusion')."""
    import jax
    import jax.numpy as jnp
    from alphafold3.model.network import atom_cross_attention as ACA

    hm, atom_layout = ACA.hm, ACA.atom_layout
    c = head.config
    global_config = head.global_config
    name = "diffusion"
    token_atoms_single_cond, _ = compat.call_native(
        ACA._per_atom_conditioning, c, batch, name, global_config=global_config
    )
    token_atoms_mask = batch.predicted_structure_info.atom_mask
    queries_single_cond = atom_layout.convert(
        batch.atom_cross_att.token_atoms_to_queries, token_atoms_single_cond, layout_axes=(-3, -2)
    )
    queries_mask = atom_layout.convert(
        batch.atom_cross_att.token_atoms_to_queries, token_atoms_mask, layout_axes=(-2, -1)
    )
    trunk_single_cond = embeddings["single"]
    if trunk_single_cond is not None:
        trunk_single_cond = hm.Linear(
            c.per_atom_channels, precision="highest", initializer=global_config.final_init,
            name=f"{name}_embed_trunk_single_cond",
        )(
            hm.LayerNorm(
                use_fast_variance=False, create_offset=False, name=f"{name}_lnorm_trunk_single_cond"
            )(trunk_single_cond)
        )
        queries_single_cond += atom_layout.convert(
            batch.atom_cross_att.tokens_to_queries, trunk_single_cond, layout_axes=(-2,)
        )
    queries_single_cond = queries_single_cond * queries_mask[..., None]
    row_act = hm.Linear(c.per_atom_pair_channels, name=f"{name}_single_to_pair_cond_row")(
        jax.nn.relu(queries_single_cond)
    )
    pair_cond_keys_input = atom_layout.convert(
        batch.atom_cross_att.queries_to_keys, queries_single_cond, layout_axes=(-3, -2)
    )
    col_act = hm.Linear(c.per_atom_pair_channels, name=f"{name}_single_to_pair_cond_col")(
        jax.nn.relu(pair_cond_keys_input)
    )
    pair_act = row_act[:, :, None, :] + col_act[:, None, :, :]
    if trunk_pair_cond is not None:
        trunk_pair_cond = hm.Linear(
            c.per_atom_pair_channels, precision="highest", initializer=global_config.final_init,
            name=f"{name}_embed_trunk_pair_cond",
        )(
            hm.LayerNorm(
                use_fast_variance=False, create_offset=False, name=f"{name}_lnorm_trunk_pair_cond"
            )(trunk_pair_cond)
        )
        num_tokens = trunk_pair_cond.shape[0]
        tokens_to_queries = batch.atom_cross_att.tokens_to_queries
        tokens_to_keys = batch.atom_cross_att.tokens_to_keys
        trunk_pair_to_atom_pair = atom_layout.GatherInfo(
            gather_idxs=(
                num_tokens * tokens_to_queries.gather_idxs[:, :, None]
                + tokens_to_keys.gather_idxs[:, None, :]
            ),
            gather_mask=(
                tokens_to_queries.gather_mask[:, :, None] & tokens_to_keys.gather_mask[:, None, :]
            ),
            input_shape=jnp.array((num_tokens, num_tokens)),
        )
        pair_act += atom_layout.convert(
            trunk_pair_to_atom_pair, trunk_pair_cond, layout_axes=(-3, -2)
        )
    queries_ref_pos = atom_layout.convert(
        batch.atom_cross_att.token_atoms_to_queries, batch.ref_structure.positions,
        layout_axes=(-3, -2),
    )
    queries_ref_space_uid = atom_layout.convert(
        batch.atom_cross_att.token_atoms_to_queries, batch.ref_structure.ref_space_uid,
        layout_axes=(-2, -1),
    )
    keys_ref_pos = atom_layout.convert(
        batch.atom_cross_att.queries_to_keys, queries_ref_pos, layout_axes=(-3, -2)
    )
    keys_ref_space_uid = atom_layout.convert(
        batch.atom_cross_att.queries_to_keys, batch.ref_structure.ref_space_uid,
        layout_axes=(-2, -1),
    )
    offsets_valid = queries_ref_space_uid[:, :, None] == keys_ref_space_uid[:, None, :]
    offsets = queries_ref_pos[:, :, None, :] - keys_ref_pos[:, None, :, :]
    pair_act += (
        hm.Linear(c.per_atom_pair_channels, precision="highest", name=f"{name}_embed_pair_offsets")(
            offsets
        )
        * offsets_valid[:, :, :, None]
    )
    sq_dists = jnp.sum(jnp.square(offsets), axis=-1)
    pair_act += (
        hm.Linear(c.per_atom_pair_channels, name=f"{name}_embed_pair_distances")(
            1.0 / (1 + sq_dists[:, :, :, None])
        )
        * offsets_valid[:, :, :, None]
    )
    pair_act += hm.Linear(c.per_atom_pair_channels, name=f"{name}_embed_pair_offsets_valid")(
        offsets_valid[:, :, :, None].astype(jnp.float32)
    )
    pair_act2 = hm.Linear(c.per_atom_pair_channels, initializer="relu", name=f"{name}_pair_mlp_1")(
        jax.nn.relu(pair_act)
    )
    pair_act2 = hm.Linear(c.per_atom_pair_channels, initializer="relu", name=f"{name}_pair_mlp_2")(
        jax.nn.relu(pair_act2)
    )
    pair_act += hm.Linear(
        c.per_atom_pair_channels, initializer=global_config.final_init, name=f"{name}_pair_mlp_3"
    )(jax.nn.relu(pair_act2))
    return pair_act


def _make_head(DH):
    import functools
    import inspect

    Stock = DH.DiffusionHead

    class HoistDiffusionHead(Stock):
        """Same native ``__call__`` (conditioning_only / pair_cond / atom_cond); orig uses ``hoist_precompute``."""

        @functools.wraps(Stock.__call__)
        def __call__(self, *args, **kwargs):
            if kwargs.pop("hoist_precompute", False):
                _orig_precompute(self, *args, **kwargs)
                return None
            out = super().__call__(*args, **kwargs)
            flag = kwargs.get("conditioning_only", False)
            if not flag:
                try:
                    ba = inspect.signature(Stock.__call__).bind(self, *args, **kwargs)
                    ba.apply_defaults()
                    flag = bool(ba.arguments.get("conditioning_only"))
                except (TypeError, ValueError):
                    flag = len(args) >= 8 and bool(args[7])
            if flag:
                pair_cond, atom_cond = out
                _precompute(self, pair_cond, atom_cond)
            return out

    HoistDiffusionHead.__module__ = Stock.__module__
    HoistDiffusionHead.__name__ = Stock.__name__
    HoistDiffusionHead.__qualname__ = Stock.__qualname__
    HoistDiffusionHead.__call__.__signature__ = inspect.signature(Stock.__call__)
    return HoistDiffusionHead


def _orig_precompute(head, *args, **kwargs):
    import jax.numpy as jnp
    from alphafold3.model.components import utils as U

    noise_level = args[1] if len(args) > 1 else kwargs.get("noise_level")
    batch = args[2] if len(args) > 2 else kwargs["batch"]
    embeddings = args[3] if len(args) > 3 else kwargs["embeddings"]
    use_conditioning = args[4] if len(args) > 4 else kwargs.get("use_conditioning", True)
    noise = noise_level if noise_level is not None else jnp.zeros((), jnp.float32)
    with U.bfloat16_context():
        _, pair_cond = head._conditioning(
            batch=batch, embeddings=embeddings, noise_level=noise, use_conditioning=use_conditioning
        )
        pair_act = _atom_pair_act(head, batch, embeddings, pair_cond)
    _precompute(head, pair_cond, {"pair_act": pair_act})


def _make_orig_sample(stock_fn):
    import haiku as hk
    import jax.numpy as jnp

    @hk.transparent
    def _sample_diffusion(self, batch, embeddings, *, sample_config):
        self.diffusion_module(
            None, jnp.zeros((), jnp.float32), batch, embeddings, True, hoist_precompute=True
        )
        return stock_fn(self, batch, embeddings, sample_config=sample_config)

    return _sample_diffusion


def _precompute(head, pair_cond, atom_cond) -> None:
    import jax.numpy as jnp
    from alphafold3.model.components import utils as U
    from alphafold3.model.network import diffusion_transformer as DT

    _STASH.pop("token", None)
    _STASH["atom"] = {}
    _STASH["pre"] = True
    try:
        if pair_cond is not None and compat.per_block_pair_ln(head.global_config):
            dtype = (
                U.compute_dtype(head.global_config, sampler=True)
                if hasattr(U, "compute_dtype") else pair_cond.dtype
            )
            tr = DT.Transformer(head.config.transformer, head.global_config)
            _STASH["token"] = tr(None, None, None, jnp.asarray(pair_cond, dtype=dtype))
        pair_act = atom_cond.get("pair_act") if isinstance(atom_cond, dict) else None
        if pair_act is not None and _shared_atom_pair_logits(head.global_config):
            cfg = head.config.atom_transformer
            for suffix in ("encoder", "decoder"):
                name = f"diffusion_atom_transformer_{suffix}"
                cat = DT.CrossAttTransformer(cfg, head.global_config, name=name)
                _STASH["atom"][name] = cat(None, None, None, None, None, None, pair_act)
    finally:
        _STASH.pop("pre", None)


def install() -> bool:
    if _STATE["installed"]:
        return True
    from alphafold3.model import model as af3_model
    from alphafold3.model.network import diffusion_head as DH
    from alphafold3.model.network import diffusion_transformer as DT

    orig = not compat.native_has_conditioning_only()
    _STATE["orig"] = orig
    _STATE["stock"] = {
        "T": DT.Transformer,
        "CAT": DT.CrossAttTransformer,
        "Head": DH.DiffusionHead,
        "sample_diff": af3_model.Model._sample_diffusion,
    }
    DT.Transformer = _make_transformer(DT)
    DT.CrossAttTransformer = _make_atom_transformer(DT)
    DH.DiffusionHead = _make_head(DH)
    if orig:
        af3_model.Model._sample_diffusion = _make_orig_sample(af3_model.Model._sample_diffusion)
    _STATE["installed"] = True
    print(
        f"{PREFIX} HOIST_LOGITS dtype={STORE_DTYPE} "
        f"{'orig_atom_precompute' if orig else 'token=of3_per_block atom=enc+dec'}",
        flush=True,
    )
    return True


def uninstall() -> None:
    if not _STATE["installed"]:
        return
    from alphafold3.model import model as af3_model
    from alphafold3.model.network import diffusion_head as DH
    from alphafold3.model.network import diffusion_transformer as DT

    st = _STATE["stock"]
    DT.Transformer = st["T"]
    DT.CrossAttTransformer = st["CAT"]
    DH.DiffusionHead = st["Head"]
    if st.get("sample_diff") is not None:
        af3_model.Model._sample_diffusion = st["sample_diff"]
    _STATE["installed"] = False
    _STASH.clear()


def report() -> dict:
    return {
        "installed": _STATE["installed"],
        "dtype": _STATE["dtype"],
        "traced": dict(_STATE["traced"]),
        "skipped": _STATE["skipped"],
        "orig": _STATE["orig"],
    }
