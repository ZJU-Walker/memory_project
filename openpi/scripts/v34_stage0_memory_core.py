"""v3.4 Stage 0: the synthetic memory-core battery (V34_PLAN_final.md section 9, mandatory
before any VLA run).

Exercises the Titans inner optimizer through the 5.10 key-space API (`write_kv`/`read_key`)
with synthetic unit (K, V) pairs at the production geometry (d_key 512, hiddens 1024^3,
d_value 2048), gates frozen at theta=0.10 / eta=0.90 / alpha=0.01 by default. ``--eta``
selects an explicit core momentum gate for intervention checks:

  1. repeated association -- one random unit pair written 64x: own-key recall must converge
     (late-window error << initial; damped oscillation at eta=0.9 is EXPECTED and accepted;
     strict per-step monotonic decrease is explicitly not required);
  2. near-orthogonal associations -- 8 random pairs written together 32x: own-key recall must
     succeed for each;
  3. rank-1 stress -- 63 writes of near-identical 16-token frames (cos ~0.998, the v3.3
     write-token similarity): bounded fast-weight drift, no exponential growth in gradient
     norm, surprise, or retrieval norm. This is the battery the UN-normalized v3.3 memory
     fails catastrophically (measured 3 -> 500 -> 20M raw gradient norms).

Throughout: raw gradient norm, clip multiplier, fraction of clipped updates, surprise, and
per-layer drift are logged. PASS requires post-transient Pr(||grad|| > 1) < 10% with clip
multipliers near 1 and bounded drift; a short initial clipping transient is acceptable
(zero-output init against unit targets makes early O(1)-2 gradients legitimate).

Run:
    uv run scripts/v34_stage0_memory_core.py            # the v3.4 (mlp_l2norm) core -> exit 0/1
    uv run scripts/v34_stage0_memory_core.py --legacy   # the v3.3 core, expected to FAIL
"""

import argparse
import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import memory as _memory

TRANSIENT_WRITES = 8  # exempt from the steady-state clipping criterion
GATES = (0.10, 0.90, 0.01)  # frozen theta / eta / alpha


@dataclasses.dataclass
class Trace:
    error: list
    grad_norm: list
    clip: list
    surprise: list
    drift: list
    layer_drift: list

    @staticmethod
    def empty() -> "Trace":
        return Trace([], [], [], [], [], [])


def _unit(key, shape):
    x = jax.random.normal(key, shape)
    return x / jnp.linalg.norm(x, axis=-1, keepdims=True)


def _drift(mem, state):
    total = 0.0
    per_layer = {}
    for name, leaf in state.fast_weights.items():
        # Measure from the effective per-episode initializer. With the run4 blank-output
        # invariant, raw checkpoint m0 output leaves can be nonzero but are intentionally
        # dormant and must not contaminate this diagnostic.
        d = float(jnp.sum(jnp.square(leaf[0] - mem._initial_fast_leaf(name))))  # noqa: SLF001
        per_layer[name] = np.sqrt(d)
        total += d
    return float(np.sqrt(total)), per_layer


def _run(mem, k, v, num_writes, *, eta_gate=GATES[1], recall_k=None):
    """Write the same (k, v) frame `num_writes` times; trace the plan-5.7 quantities."""
    recall_k = k if recall_k is None else recall_k
    state = mem.init_state(1)
    theta = jnp.full((1,), GATES[0])
    eta = jnp.full((1,), eta_gate)
    alpha = jnp.full((1,), GATES[2])
    write = jax.jit(lambda s: mem.write_kv(s, k, v, theta, eta, alpha))
    read = jax.jit(lambda s, q: mem.read_key(s, q))
    trace = Trace.empty()
    initial_error = float(jnp.mean(jnp.sum(jnp.square(read(state, k) - v), axis=-1)))
    for _ in range(num_writes):
        state, aux = write(state)
        trace.grad_norm.append(float(aux["grad_norm"][0]))
        trace.clip.append(float(aux["clip_factor"][0]))
        trace.surprise.append(float(aux["surprise"][0]))
        err = float(jnp.mean(jnp.sum(jnp.square(read(state, recall_k) - v), axis=-1)))
        trace.error.append(err)
        drift, per_layer = _drift(mem, state)
        trace.drift.append(drift)
        trace.layer_drift.append(per_layer)
    return state, trace, initial_error


def _steady_state_stats(trace: Trace):
    grads = np.asarray(trace.grad_norm[TRANSIENT_WRITES:])
    clips = np.asarray(trace.clip[TRANSIENT_WRITES:])
    return {
        "p_clipped": float(np.mean(grads > 1.0 + 1e-6)) if len(grads) else 1.0,
        "median_grad": float(np.median(grads)) if len(grads) else np.inf,
        "max_grad": float(np.max(np.asarray(trace.grad_norm))),
        "median_clip": float(np.median(clips)) if len(clips) else 0.0,
        "max_drift": float(np.max(np.asarray(trace.drift))),
    }


def _report(name, stats, checks):
    print(f"\n== {name} ==")
    for key, value in stats.items():
        print(f"  {key:>22}: {value:.4g}")
    ok = True
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        ok &= passed
    return ok


def battery_repeated_association(mem, *, eta_gate=GATES[1]) -> bool:
    k = _unit(jax.random.key(10), (1, 1, mem.config.d_key))
    v = _unit(jax.random.key(11), (1, 1, mem.config.d_value))
    _, trace, initial_error = _run(mem, k, v, 64, eta_gate=eta_gate)
    stats = _steady_state_stats(trace)
    late = float(np.mean(trace.error[-16:]))
    stats.update({"initial_error": initial_error, "late_window_error": late, "final_error": trace.error[-1]})
    print(f"  error curve (every 8): {[f'{e:.3f}' for e in trace.error[::8]]}")
    print(f"  grad curve  (every 8): {[f'{g:.3f}' for g in trace.grad_norm[::8]]}")
    return _report(
        "battery 1: repeated association (64 writes of one unit pair)",
        stats,
        [
            ("late-window error << initial (<0.2x)", late < 0.2 * initial_error),
            ("post-transient Pr(||grad||>1) < 10%", stats["p_clipped"] < 0.10),
            ("median steady-state grad well below the clip", stats["median_grad"] < 1.0),
            ("drift bounded (< 50)", stats["max_drift"] < 50.0),
        ],
    )


def battery_near_orthogonal(mem, *, eta_gate=GATES[1]) -> bool:
    n = 8
    k = _unit(jax.random.key(20), (1, n, mem.config.d_key))
    v = _unit(jax.random.key(21), (1, n, mem.config.d_value))
    state, trace, initial_error = _run(mem, k, v, 32, eta_gate=eta_gate)
    per_key_error = np.asarray(jnp.sum(jnp.square(mem.read_key(state, k) - v), axis=-1))[0]
    stats = _steady_state_stats(trace)
    stats.update({"initial_error": initial_error, "worst_key_error": float(per_key_error.max())})
    print(f"  per-key recall error: {[f'{e:.3f}' for e in per_key_error]}")
    return _report(
        "battery 2: 8 near-orthogonal associations (32 joint writes)",
        stats,
        [
            ("own-key recall succeeds for EVERY key (<0.5)", float(per_key_error.max()) < 0.5),
            ("post-transient Pr(||grad||>1) < 10%", stats["p_clipped"] < 0.10),
            ("drift bounded (< 50)", stats["max_drift"] < 50.0),
        ],
    )


def battery_rank1_stress(mem, *, eta_gate=GATES[1]) -> bool:
    tokens = 16
    base_k = _unit(jax.random.key(30), (1, 1, mem.config.d_key))
    base_v = _unit(jax.random.key(31), (1, 1, mem.config.d_value))
    state = mem.init_state(1)
    theta = jnp.full((1,), GATES[0])
    eta = jnp.full((1,), eta_gate)
    alpha = jnp.full((1,), GATES[2])
    write = jax.jit(lambda s, k, v: mem.write_kv(s, k, v, theta, eta, alpha))
    read = jax.jit(lambda s, q: mem.read_key(s, q))
    trace = Trace.empty()
    cosines = []
    for i in range(63):
        noise_k = _unit(jax.random.fold_in(jax.random.key(32), i), (1, tokens, mem.config.d_key))
        noise_v = _unit(jax.random.fold_in(jax.random.key(33), i), (1, tokens, mem.config.d_value))
        # unit base + 0.06-norm perturbation -> cosine ~0.998, the measured v3.3 write-token
        # self-similarity
        k = base_k + 0.06 * noise_k
        v = base_v + 0.06 * noise_v
        k = k / jnp.linalg.norm(k, axis=-1, keepdims=True)
        v = v / jnp.linalg.norm(v, axis=-1, keepdims=True)
        cosines.append(float(jnp.mean(jnp.sum(k * base_k, axis=-1))))
        state, aux = write(state, k, v)
        trace.grad_norm.append(float(aux["grad_norm"][0]))
        trace.clip.append(float(aux["clip_factor"][0]))
        trace.surprise.append(float(aux["surprise"][0]))
        retrieval_norm = float(jnp.sqrt(jnp.mean(jnp.square(read(state, k)))))
        trace.error.append(retrieval_norm)  # repurposed: retrieval RMS trajectory
        drift, per_layer = _drift(mem, state)
        trace.drift.append(drift)
        trace.layer_drift.append(per_layer)
    stats = _steady_state_stats(trace)
    early = float(np.mean(trace.grad_norm[TRANSIENT_WRITES : TRANSIENT_WRITES + 10]))
    late = float(np.mean(trace.grad_norm[-10:]))
    stats.update(
        {
            "mean_write_cosine": float(np.mean(cosines)),
            "early_grad_mean": early,
            "late_grad_mean": late,
            "max_surprise": float(np.max(trace.surprise)),
            "final_retrieval_rms": trace.error[-1],
        }
    )
    print(f"  grad norms (every 8): {[f'{g:.3f}' for g in trace.grad_norm[::8]]}")
    print(f"  surprise   (every 8): {[f'{s:.3f}' for s in trace.surprise[::8]]}")
    print(f"  drift      (every 8): {[f'{d:.3f}' for d in trace.drift[::8]]}")
    print(f"  per-layer drift @63: { {k: f'{v:.3f}' for k, v in trace.layer_drift[-1].items()} }")
    return _report(
        "battery 3: rank-1 stress (63 writes of ~0.998-cosine 16-token frames)",
        stats,
        [
            ("raw gradients stay O(1) (max < 10)", stats["max_grad"] < 10.0),
            ("no exponential growth (late mean < 3x early mean + 0.5)", late < 3.0 * early + 0.5),
            ("no sustained clip saturation (post-transient Pr < 10%)", stats["p_clipped"] < 0.10),
            ("surprise bounded (< 5)", stats["max_surprise"] < 5.0),
            ("retrieval norm bounded (< 10)", stats["final_retrieval_rms"] < 10.0),
            ("fast-weight drift bounded (< 50)", stats["max_drift"] < 50.0),
        ],
    )


def battery_conflicting_and_degenerate_inputs(mem, *, eta_gate=GATES[1]) -> bool:
    """Adversarial writes: one key with alternating targets, followed by an exact-zero key.

    These associations are intentionally impossible to satisfy simultaneously. The contract is
    bounded, finite dynamics rather than recall convergence; the exact-zero key also exercises
    every epsilon-protected L2-normalization derivative.
    """
    state = mem.init_state(1)
    passive_state = mem.init_state(1)
    theta = jnp.full((1,), GATES[0])
    eta = jnp.full((1,), eta_gate)
    alpha = jnp.full((1,), GATES[2])
    write = jax.jit(lambda s, k, v: mem.write_kv(s, k, v, theta, eta, alpha))
    decay = jax.jit(lambda s, k, v: mem.write_kv(s, k, v, theta, eta, alpha, zero_gradient=True))
    read = jax.jit(lambda s, q: mem.read_key(s, q))
    key = _unit(jax.random.key(40), (1, 16, mem.config.d_key))
    value = _unit(jax.random.key(41), (1, 16, mem.config.d_value))
    zero_key = jnp.zeros_like(key)
    grads, surprises, retrievals, drifts, content_displacements, momenta = [], [], [], [], [], []
    finite = True
    for i in range(128):
        step_key = zero_key if i >= 96 else key
        step_value = value if i % 2 == 0 else -value
        state, aux = write(state, step_key, step_value)
        passive_state, _ = decay(passive_state, step_key, step_value)
        grad = float(aux["grad_norm"][0])
        surprise = float(aux["surprise"][0])
        retrieval = float(jnp.sqrt(jnp.mean(jnp.square(read(state, step_key)))))
        drift, _ = _drift(mem, state)
        content_displacement = float(
            _memory._tree_norm(  # noqa: SLF001
                jax.tree.map(jnp.subtract, state.fast_weights, passive_state.fast_weights)
            )[0]
        )
        momentum = float(_memory._tree_norm(state.momentum)[0])  # noqa: SLF001
        grads.append(grad)
        surprises.append(surprise)
        retrievals.append(retrieval)
        drifts.append(drift)
        content_displacements.append(content_displacement)
        momenta.append(momentum)
        finite &= bool(
            np.isfinite([grad, surprise, retrieval, drift, content_displacement, momentum]).all()
            and all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(state))
        )
    stats = {
        "max_grad": float(np.max(grads)),
        "max_surprise": float(np.max(surprises)),
        "max_retrieval_rms": float(np.max(retrievals)),
        "max_raw_drift": float(np.max(drifts)),
        "max_content_vs_passive": float(np.max(content_displacements)),
        "max_momentum": float(np.max(momenta)),
    }
    return _report(
        "battery 4: conflicting values + exact-zero-key tail (128 writes)",
        stats,
        [
            ("all state and diagnostics remain finite", finite),
            ("raw gradients stay bounded (max < 10)", stats["max_grad"] < 10.0),
            ("surprise stays bounded (< 25)", stats["max_surprise"] < 25.0),
            ("retrieval RMS stays bounded (< 10)", stats["max_retrieval_rms"] < 10.0),
            # Raw drift is reported but not gated: alpha decays the large hidden m0 toward zero,
            # so it grows to ~||m0|| even with no content writes. The paired passive trajectory
            # removes that gauge/forgetting component.
            ("content displacement vs passive stays bounded (< 20)", stats["max_content_vs_passive"] < 20.0),
            ("momentum respects the clipped-input envelope (< 1.1)", stats["max_momentum"] < 1.1),
        ],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy", action="store_true", help="run the v3.3 (un-normalized) core for contrast")
    parser.add_argument("--blank-initial-output", action="store_true", help="exercise the run4 episode initializer")
    parser.add_argument(
        "--eta",
        type=float,
        default=GATES[1],
        help="explicit write_kv momentum gate used by every battery (default: 0.90)",
    )
    parser.add_argument(
        "--adversarial-loaded-output",
        action="store_true",
        help="simulate a legacy checkpoint with large dormant outer output leaves",
    )
    args = parser.parse_args()

    if args.adversarial_loaded_output and not args.blank_initial_output:
        parser.error("--adversarial-loaded-output requires --blank-initial-output")
    if not np.isfinite(args.eta) or not 0.0 <= args.eta <= 1.0:
        parser.error("--eta must be finite and in [0, 1]")
    config = _memory.MemoryConfig(
        mlp_l2norm=not args.legacy,
        blank_initial_output=args.blank_initial_output,
    )
    print(f"memory core: d_key={config.d_key} hiddens={config.hidden_dims} d_value={config.d_value} "
          f"mlp_l2norm={config.mlp_l2norm} blank_initial_output={config.blank_initial_output} "
          f"clip={config.max_grad_norm} gates=(theta={GATES[0]}, eta={args.eta}, alpha={GATES[2]})")
    mem = _memory.TitansMemory(config, rngs=nnx.Rngs(0))
    if args.adversarial_loaded_output:
        output_layer = mem._num_layers - 1  # noqa: SLF001
        mem.m0[f"w{output_layer}"].value = jnp.full_like(mem.m0[f"w{output_layer}"].value, 7.0)
        mem.m0[f"b{output_layer}"].value = jnp.full_like(mem.m0[f"b{output_layer}"].value, -3.0)
        state = mem.init_state(1)
        np.testing.assert_array_equal(np.asarray(state.fast_weights[f"w{output_layer}"]), 0.0)
        np.testing.assert_array_equal(np.asarray(state.fast_weights[f"b{output_layer}"]), 0.0)
        print("adversarial legacy output loaded: effective fresh output remains exact FP32 zero")

    results = [
        battery_repeated_association(mem, eta_gate=args.eta),
        battery_near_orthogonal(mem, eta_gate=args.eta),
        battery_rank1_stress(mem, eta_gate=args.eta),
        battery_conflicting_and_degenerate_inputs(mem, eta_gate=args.eta),
    ]
    print("\n" + ("STAGE 0: ALL BATTERIES PASS" if all(results) else "STAGE 0: FAILURES PRESENT"))
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
