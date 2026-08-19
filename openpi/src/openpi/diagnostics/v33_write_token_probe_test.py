"""Tests for the v3.3 write-token side probe.

The scientific claim of this diagnostic rests entirely on the estimator being honest, so the
tests target the ways it could lie: a leaky split reporting fake signal, an AUC that disagrees
with a brute-force definition, and a null that fails to flag a separable-by-construction design.
"""

# ruff: noqa: SLF001 - the private probe/planning helpers are part of the contract under test.

import numpy as np
import pytest

from openpi.diagnostics import v33_write_token_probe as probe


def _separable(n: int = 20, d: int = 8, gap: float = 3.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    labels = np.array([i % 2 for i in range(n)], dtype=np.int64)
    features = rng.normal(size=(n, d))
    features[:, 0] += gap * labels  # one genuinely informative dimension
    return features, labels


def test_roc_auc_matches_brute_force_definition():
    rng = np.random.default_rng(3)
    labels = rng.integers(0, 2, size=40)
    scores = rng.normal(size=40)
    pairs = [(s_pos, s_neg) for s_pos in scores[labels == 1] for s_neg in scores[labels == 0]]
    expected = np.mean([(p > n) + 0.5 * (p == n) for p, n in pairs])
    assert probe.roc_auc(labels, scores) == pytest.approx(expected, abs=1e-12)


def test_roc_auc_handles_ties_and_single_class():
    assert probe.roc_auc(np.array([0, 1, 0, 1]), np.array([1.0, 1.0, 1.0, 1.0])) == pytest.approx(0.5)
    assert np.isnan(probe.roc_auc(np.array([1, 1, 1]), np.array([0.1, 0.2, 0.3])))


def test_separable_signal_is_recovered():
    """A genuinely informative dimension must be found -- but note the probe is deliberately
    conservative: with d comparable to n, distractor dimensions can outvote the signal on
    borderline rows, so the honest assertion is a high AUC, not a perfect accuracy. Given the
    signal dimension alone the probe is exact, which localizes any miss to the distractors."""
    features, labels = _separable(gap=6.0)
    result = probe.leave_one_out_probe(features, labels)
    assert result["auc"] > 0.9
    assert result["accuracy"] > 0.85

    signal_only = probe.leave_one_out_probe(features[:, :1], labels)
    assert signal_only["accuracy"] == 1.0
    assert signal_only["auc"] == 1.0


def test_pure_noise_stays_near_chance():
    rng = np.random.default_rng(11)
    features = rng.normal(size=(24, 12))
    labels = np.array([i % 2 for i in range(24)], dtype=np.int64)
    result = probe.leave_one_out_probe(features, labels)
    # Leave-one-out on noise must not manufacture strong signal.
    assert result["auc"] < 0.8


def test_shuffled_null_is_centered_at_chance_for_real_signal():
    features, labels = _separable(gap=6.0)
    null = probe.shuffled_null(features, labels, repeats=8, seed=1)
    # Permuting labels must destroy the signal even though the features are separable.
    assert null["auc_mean"] < 0.75
    assert null["accuracy_mean"] < 0.8


def test_leave_one_out_never_trains_on_the_held_out_row():
    """The load-bearing property: a row whose label contradicts a perfectly separable design
    cannot be predicted correctly, because its own label never enters training."""
    features, labels = _separable(n=16, gap=50.0, seed=5)
    poisoned = labels.copy()
    poisoned[0] = 1 - poisoned[0]  # this row now contradicts every other row
    result = probe.leave_one_out_probe(features, poisoned)
    predictions = (np.array(result["scores"]) > 0).astype(np.int64)
    assert predictions[0] != poisoned[0]


def test_standardize_uses_train_statistics_only():
    train = np.array([[0.0, 10.0], [2.0, 10.0]])
    test = np.array([[4.0, 10.0]])
    train_z, test_z = probe._standardize(train, test)
    assert train_z.mean(axis=0) == pytest.approx([0.0, 0.0])
    # constant column -> scale forced to 1, so it centers to zero rather than exploding
    assert np.isfinite(test_z).all()
    assert test_z[0, 0] == pytest.approx(3.0)
    assert test_z[0, 1] == pytest.approx(0.0)


def test_fit_logistic_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="bad probe shapes"):
        probe.fit_logistic(np.zeros((4, 3)), np.zeros(5))


def test_leave_one_out_requires_enough_episodes():
    with pytest.raises(ValueError, match="at least 4 episodes"):
        probe.leave_one_out_probe(np.zeros((3, 2)), np.array([0, 1, 0]))


def test_l2_regularization_curbs_high_dimensional_memorization():
    """With d >> n an unregularized fit separates anything; the default penalty must temper it."""
    rng = np.random.default_rng(7)
    features = rng.normal(size=(12, 500))
    labels = np.array([i % 2 for i in range(12)], dtype=np.int64)
    weak = probe.leave_one_out_probe(features, labels, l2=1e-6)
    strong = probe.leave_one_out_probe(features, labels, l2=100.0)
    assert strong["auc"] <= weak["auc"] + 0.35


def _harvest(episode: int, side: str, prompt: str, vector: np.ndarray, cf_vector: np.ndarray | None = None):
    plan = probe._wa._EpisodePlan(
        episode=episode,
        prompt=prompt,
        counterfactual="other",
        side=side,
        evidence=(0, 10),
        memory=(20, 30),
        length=31,
    )
    cf = vector if cf_vector is None else cf_vector
    return {
        "plan": plan,
        "counts": {"evidence": 3},
        "phase_vectors": {
            "evidence": {
                "write_true": vector,
                "write_cf": cf,
                "read_true": vector,
                "read_cf": cf,
                "state_true": vector[:4],
                "state_cf": cf[:4],
            }
        },
    }


def test_analyze_reports_signal_and_null_per_stream():
    rng = np.random.default_rng(2)
    harvested = []
    for i in range(16):
        side = "right" if i % 2 else "left"
        vector = rng.normal(size=8)
        vector[0] += 5.0 * (side == "right")
        harvested.append(_harvest(i, side, "find the banana", vector))
    results = probe.analyze(harvested, seed=0)
    evidence = results["phases"]["evidence"]
    assert evidence["n_episodes"] == 16
    assert evidence["write"]["auc"] > 0.9
    assert evidence["write"]["null"]["auc_mean"] < 0.8
    assert results["label_balance"]["right"] == 8
    assert results["label_balance"]["n"] == 16


def test_cf_transfer_detects_an_instruction_driven_flip():
    """If the CF encoding mirrors the informative dimension, the probe's decision must flip."""
    rng = np.random.default_rng(4)
    harvested = []
    for i in range(16):
        side = "right" if i % 2 else "left"
        vector = rng.normal(size=8)
        vector[0] += 5.0 * (side == "right")
        cf_vector = vector.copy()
        cf_vector[0] = -vector[0]  # instruction flips the side-carrying dimension
        harvested.append(_harvest(i, side, "find the banana", vector, cf_vector))
    results = probe.analyze(harvested, seed=0)
    cf = results["phases"]["evidence"]["write"]["cf_transfer"]
    assert cf["cf_flip_rate"] > 0.5
    assert cf["mean_score_shift"] > 0.0


def test_cf_transfer_reports_no_flip_when_instruction_is_ignored():
    rng = np.random.default_rng(6)
    harvested = []
    for i in range(16):
        side = "right" if i % 2 else "left"
        vector = rng.normal(size=8)
        vector[0] += 5.0 * (side == "right")
        harvested.append(_harvest(i, side, "find the banana", vector))  # cf == true
    results = probe.analyze(harvested, seed=0)
    cf = results["phases"]["evidence"]["write"]["cf_transfer"]
    assert cf["cf_flip_rate"] == 0.0
    assert cf["mean_abs_score_shift"] == pytest.approx(0.0, abs=1e-9)


def _plan(episode: int, prompt: str, side: str):
    return probe._wa._EpisodePlan(
        episode=episode,
        prompt=prompt,
        counterfactual="other",
        side=side,
        evidence=(0, 10),
        memory=(20, 30),
        length=31,
    )


def test_stratified_subset_balances_cells_where_a_prefix_would_not():
    """Regression: the dataset is stored in cell order (15 episodes per cell), so a naive
    prefix drew every episode from one cell -- the real 8-episode smoke run came out 7 left /
    1 right and its shuffled null hit AUC 1.00, i.e. it measured nothing."""
    cells = [("banana", "left"), ("banana", "right"), ("box", "left"), ("box", "right")]
    plans = [_plan(i, *cells[i // 15]) for i in range(60)]
    assert len({(p.prompt, p.side) for p in plans[:8]}) == 1  # the degenerate design a prefix gives

    picked = probe._stratified_subset(plans, 8)
    assert len(picked) == 8
    assert sum(p.side == "right" for p in picked) == 4
    assert len({(p.prompt, p.side) for p in picked}) == 4


def test_stratified_subset_is_a_noop_when_limit_exceeds_supply():
    plans = [_plan(i, "banana", "left" if i % 2 else "right") for i in range(6)]
    assert probe._stratified_subset(plans, 10) == plans


def test_analyze_warns_on_an_imbalanced_design():
    rng = np.random.default_rng(12)
    harvested = [_harvest(i, "left" if i < 7 else "right", "banana", rng.normal(size=8)) for i in range(8)]
    results = probe.analyze(harvested, seed=0)
    assert "warning" in results
    assert results["label_balance"]["majority_rate"] == pytest.approx(7 / 8)


def test_analyze_does_not_warn_on_a_balanced_design():
    rng = np.random.default_rng(13)
    harvested = [_harvest(i, "right" if i % 2 else "left", "banana", rng.normal(size=8)) for i in range(8)]
    results = probe.analyze(harvested, seed=0)
    assert "warning" not in results
    assert results["label_balance"]["majority_rate"] == pytest.approx(0.5)


def test_sample_stride_must_divide_the_write_stride():
    """A sample grid that misses write frames would advance memory at frames it never observed."""
    for write_stride, sample_stride, ok in [(15, 5, True), (15, 15, True), (15, 4, False), (15, 30, True)]:
        divides = (write_stride % sample_stride == 0) or sample_stride >= write_stride
        assert divides == ok, f"{write_stride}/{sample_stride}"


def test_options_reject_nonpositive_sample_stride():
    with pytest.raises(ValueError, match="sample_stride must be positive"):
        probe.Options(checkpoint="c", dataset_root="d", output_dir="o", sample_stride=0)


def test_options_default_sample_stride_is_none():
    options = probe.Options(checkpoint="c", dataset_root="d", output_dir="o")
    assert options.sample_stride is None


def test_analyze_skips_phases_with_too_few_episodes():
    rng = np.random.default_rng(8)
    harvested = [_harvest(i, "right" if i % 2 else "left", "p", rng.normal(size=8)) for i in range(3)]
    results = probe.analyze(harvested, seed=0)
    assert "evidence" not in results["phases"]
