"""Run π₀.₅ inference on a single observation and capture attention probs.

Strategy:
- Replicate one forward pass of `Pi0.sample_actions` at small denoise time
  (almost-clean x_t built from the saved policy prediction at this step) and
  call `PaliGemma.llm(..., mutable=['intermediates'])` so the `self.sow(...)`
  in `gemma.Attention` lands in the bridge's `intermediates` attr.
- Decode the prefix layout to know which key indices belong to which camera.
- Aggregate over heads, slice action-query × image-key, return per-camera (T_q,
  N_patches) attention plus token boundaries.

This module exposes:
    AttentionExtractor — load checkpoint once, then call .extract(obs) per frame.

Usage (example):
    extractor = AttentionExtractor()
    out = extractor.extract(image_top, image_left, image_right, state, x_t, time=0.0)
    out["attn_probs"]      # (num_layers, B, num_kv_heads, group, T_q, T_k)
    out["boundaries"]      # {"img_top": (s, e), "img_left": (s, e), ...}
"""

import dataclasses
import pathlib
from typing import Any

import einops
import jax
import jax.numpy as jnp
import numpy as np

from flax.nnx.bridge import variables as _bv
from openpi.models import model as _model
from openpi.models import pi0 as _pi0
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

CONFIG_NAME = "pi05_yam_cup_lora"
CKPT_DIR = pathlib.Path(
    "/home/kewalk/memory_project/openpi/checkpoints/pi05_yam_cup_lora/cup_0429_v1/29999"
)
PROMPT = "pick up the cup with yellow object inside"


@dataclasses.dataclass
class TokenBoundaries:
    """Index ranges within the (prefix||suffix) sequence."""
    img_top: tuple[int, int]
    img_left: tuple[int, int]
    img_right: tuple[int, int]
    prompt: tuple[int, int]
    actions: tuple[int, int]


class AttentionExtractor:
    def __init__(self, ckpt_dir: pathlib.Path = CKPT_DIR, config_name: str = CONFIG_NAME, prompt: str = PROMPT):
        train_config = _config.get_config(config_name)
        # Reuse the canonical loader so transforms + norm stats match training.
        self._policy = _policy_config.create_trained_policy(train_config, ckpt_dir, default_prompt=prompt)
        self._model: _pi0.Pi0 = self._policy._model  # type: ignore[assignment]
        self._input_transform = self._policy._input_transform
        self._prompt = prompt
        self._action_horizon = self._model.action_horizon
        self._action_dim = self._model.action_dim
        # Boundaries depend on (prompt token length, image grid size). Image grid
        # is fixed by the policy, so cache by prompt string.
        self._boundaries_cache: dict[str, TokenBoundaries] = {}
        self._image_keys = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
        self._image_to_camera = {
            "base_0_rgb": "img_top",
            "left_wrist_0_rgb": "img_left",
            "right_wrist_0_rgb": "img_right",
        }
        # Cache the underlying Linen LLM module + its variables so we can bypass
        # the nnx_bridge during inference (the bridge's mutable-merge logic
        # corrupts params on the 2nd call when sown intermediates collide with
        # param attr names). We snapshot before any mutable call has run.
        self._llm_bridge = self._model.PaliGemma.llm
        self._llm_linen_module = self._llm_bridge.module
        self._llm_variables = self._snapshot_variables(self._llm_bridge)
        # action_out_proj operates outside the LLM; keep an nnx-callable handle.
        self._action_out_proj = self._model.action_out_proj

    @staticmethod
    def _snapshot_variables(bridge) -> dict:
        """Reconstruct the Linen `variables` dict from the bridge's nnx attrs."""
        nnx_attrs = {name: getattr(bridge, name) for name in bridge.linen_attributes}
        return _bv.nnx_attrs_to_linen_vars(nnx_attrs)

    def set_prompt(self, prompt: str) -> None:
        """Update the active prompt. Boundaries are recomputed lazily on next extract()."""
        self._prompt = prompt

    def _build_obs(self, top, left, right, state, prompt: str | None = None) -> _model.Observation:
        # Mirror the YamInputs key names used during training.
        raw = {
            "observation/top_image": top,
            "observation/left_image": left,
            "observation/right_image": right,
            "observation/state": np.asarray(state, dtype=np.float32),
            "prompt": prompt if prompt is not None else self._prompt,
        }
        inputs = self._input_transform(jax.tree.map(lambda x: x, raw))
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)
        return _model.Observation.from_dict(inputs)

    def _compute_boundaries(self, obs: _model.Observation, action_horizon: int) -> TokenBoundaries:
        """Run embed_prefix once to discover the per-camera patch counts and prompt length."""
        prefix_tokens, prefix_mask, _ = self._model.embed_prefix(obs)  # (1, S, D)
        # In Pi0.embed_prefix images are appended in dict-iteration order, then prompt last.
        cursor = 0
        ranges: dict[str, tuple[int, int]] = {}
        for name in obs.images:  # same iteration order as embed_prefix
            img_tokens, _ = self._model.PaliGemma.img(obs.images[name], train=False)
            n = int(img_tokens.shape[1])
            cam = self._image_to_camera[name]
            ranges[cam] = (cursor, cursor + n)
            cursor += n
        # Whatever is left of prefix is the prompt.
        prefix_len = int(prefix_tokens.shape[1])
        ranges["prompt"] = (cursor, prefix_len)
        # Suffix is action_horizon tokens after prefix (pi05 has no state token).
        ranges["actions"] = (prefix_len, prefix_len + action_horizon)
        return TokenBoundaries(
            img_top=ranges["img_top"],
            img_left=ranges["img_left"],
            img_right=ranges["img_right"],
            prompt=ranges["prompt"],
            actions=ranges["actions"],
        )

    def extract(
        self,
        top: np.ndarray,
        left: np.ndarray,
        right: np.ndarray,
        state: np.ndarray,
        action_seed: np.ndarray | None = None,
        time: float = 0.0,
        prompt: str | None = None,
        return_visual: bool = False,
    ) -> dict[str, Any]:
        """Run one forward pass and return predicted v_t plus captured attention.

        Args:
          top/left/right: (H, W, 3) uint8 arrays as recorded in the demo.
          state: (7,) float, raw joint positions.
          action_seed: optional (action_horizon, 7); if None we use zeros (pure noise direction
            at time=1 doesn't make sense here — we use the simplest "clean action ≈ 0" prior).
          time: denoise time in [0, 1]. 0 = clean (data), 1 = pure noise. Default 0 captures
            "what does the model attend to right at the data manifold".
          prompt: optional one-shot override; if given, used for this call without changing
            self._prompt. If None, uses the current self._prompt.
          return_visual: if True, also returns "visual_top" — a mean-pooled feature over the
            top camera's patch outputs from prefix_out. Reuses the same forward pass.

        Returns:
          {
            "v_t": (action_horizon, action_dim) np.ndarray,
            "attn_probs": (num_layers, B=1, num_kv_heads, group, T_q, T_k) np.ndarray,
            "boundaries": TokenBoundaries,
            "visual_top": (D,) np.ndarray  # only when return_visual=True
          }
        """
        active_prompt = prompt if prompt is not None else self._prompt
        obs = self._build_obs(top, left, right, state, prompt=active_prompt)
        boundaries = self._boundaries_cache.get(active_prompt)
        if boundaries is None:
            boundaries = self._compute_boundaries(obs, self._action_horizon)
            self._boundaries_cache[active_prompt] = boundaries

        # Build x_t from action_seed at the requested denoise time. action_seed=None => zeros.
        if action_seed is None:
            action_seed = np.zeros((self._action_horizon, self._action_dim), dtype=np.float32)
        action_seed = jnp.asarray(action_seed, dtype=jnp.float32)[None, ...]  # (1, ah, ad)
        # Same convention as Pi0.sample_actions: x_t = t*noise + (1-t)*action_seed.
        # We use noise=0 here so x_t = (1-t) * action_seed.
        x_t = (1.0 - time) * action_seed
        time_arr = jnp.asarray([time], dtype=jnp.float32)

        # Build prefix + suffix
        prefix_tokens, prefix_mask, prefix_ar_mask = self._model.embed_prefix(obs)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self._model.embed_suffix(
            obs, x_t, time_arr
        )
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = _pi0.make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1

        # Forward pass via the underlying Linen module directly, with
        # mutable=['intermediates']. Bypassing the bridge avoids the
        # mutable-merge bug that corrupts params on the 2nd call.
        out, updates = self._llm_linen_module.apply(
            self._llm_variables,
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
            mutable=["intermediates"],
        )
        (prefix_out, suffix_out), _ = out
        attn_probs = self._find_attn_probs(updates)
        if attn_probs is None:
            raise RuntimeError("attn_probs not found in updates tree")

        v_t = self._action_out_proj(suffix_out[:, -self._action_horizon :])
        result = {
            "v_t": np.asarray(v_t[0]),
            "attn_probs": np.asarray(attn_probs),
            "boundaries": boundaries,
            "prefix_len": int(prefix_tokens.shape[1]),
            "suffix_len": int(suffix_tokens.shape[1]),
        }
        if return_visual:
            i0, i1 = boundaries.img_top
            top_patches = np.asarray(prefix_out[0, i0:i1, :])  # (n_patches, D)
            result["visual_top"] = top_patches.mean(axis=0).astype(np.float32)
        return result

    @staticmethod
    def _find_attn_probs(node: Any) -> Any:
        """Walk an arbitrary tree and return the first attn_probs leaf, else None.

        The leaf may be wrapped as an nnx.Variable (has .value), a tuple from sow,
        a jax/np array, or nested dicts.
        """
        def get_val(x):
            return getattr(x, "value", x)

        if node is None:
            return None
        # Direct leaf check.
        if hasattr(node, "value"):
            return AttentionExtractor._find_attn_probs(node.value)
        if isinstance(node, tuple):
            for el in node:
                r = AttentionExtractor._find_attn_probs(el)
                if r is not None:
                    return r
            return None
        if isinstance(node, dict):
            if "attn_probs" in node:
                v = get_val(node["attn_probs"])
                if isinstance(v, tuple):
                    v = v[0] if len(v) > 0 else None
                v = get_val(v) if v is not None else None
                return v
            for v in node.values():
                r = AttentionExtractor._find_attn_probs(v)
                if r is not None:
                    return r
            return None
        # nnx Module instances have iter_children
        if hasattr(node, "iter_children"):
            try:
                for _, child in node.iter_children():
                    r = AttentionExtractor._find_attn_probs(child)
                    if r is not None:
                        return r
            except Exception:
                pass
        return None


def _smoke():
    """Quick sanity check: load checkpoint, run on a synthetic frame, print shapes."""
    extractor = AttentionExtractor()
    H, W = 480, 640
    img = np.zeros((H, W, 3), dtype=np.uint8)
    state = np.zeros(7, dtype=np.float32)
    out = extractor.extract(img, img, img, state, return_visual=True)
    print("v_t shape         :", out["v_t"].shape)
    print("attn_probs shape  :", out["attn_probs"].shape)
    print("visual_top shape  :", out["visual_top"].shape)
    print("prefix_len        :", out["prefix_len"])
    print("suffix_len        :", out["suffix_len"])
    print("boundaries        :", out["boundaries"])
    # Sanity: softmax across keys should sum to 1.
    probs = out["attn_probs"]
    s = np.asarray(probs).sum(axis=-1)
    print(f"key-sum stats      mean={float(s.mean()):.3f} min={float(s.min()):.3f} max={float(s.max()):.3f}")
    # Multi-prompt check: switching prompt should produce a different boundaries.prompt range.
    p1 = out["boundaries"].prompt
    extractor.set_prompt("Observe and remember which object is on each colored plate.")
    out2 = extractor.extract(img, img, img, state)
    p2 = out2["boundaries"].prompt
    print(f"prompt range #1   : {p1}")
    print(f"prompt range #2   : {p2}")
    assert len(extractor._boundaries_cache) == 2, "expected two cached prompts"


if __name__ == "__main__":
    _smoke()
