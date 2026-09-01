"""Launch authorized v3.5 training from a finalized completed-update-0 checkpoint.

The registered config intentionally contains placeholder injection calibration values.  This
wrapper reads the sealed train-54 calibration through the same helper used by the step-0
finalizer, installs the exact frozen rung schedule, and then calls the ordinary fail-closed
training entry point.  It never creates a fresh state: v3.5 optimizer training always resumes
the finalized checkpoint produced by ``v35_step0_bootstrap.py``.

All CLI paths are relative to ``memory_project``.  The official Pi0.5 base remains an online
source URI and may be downloaded independently into each cluster's project-local cache.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import dataclasses
from pathlib import Path
from pathlib import PurePosixPath
import sys

from openpi.shared import project_paths

# Cache/data roots must be installed before importing JAX, LeRobot, or the training module.
project_paths.configure_v35_runtime_environment()
project_paths.validate_executing_openpi_checkout()

_SCRIPTS_DIR = Path(__file__).parent
import openpi.training.v35_authorization as authorization  # noqa: E402

if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
import train as train_script  # noqa: E402
import v35_step0_bootstrap as bootstrap  # noqa: E402

CHECKPOINT_STEPS_BY_TARGET: dict[int, tuple[int, ...]] = {
    1_000: (250, 500, 1_000),
    2_500: (250, 500, 1_000, 2_500),
    10_000: (250, 500, 1_000, 2_500, 5_000, 10_000),
}
DEFAULT_PILOT_AUTHORIZATION = PurePosixPath("v35/diagnostics/authorization/pilot.json")


class V35TrainLaunchError(ValueError):
    """Raised before training when the portable authorized-resume contract is incomplete."""


def _relative_existing(value: str | PurePosixPath, *, name: str) -> PurePosixPath:
    relative = PurePosixPath(value)
    if not relative.parts or relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts):
        raise V35TrainLaunchError(f"{name} must be a normalized memory_project-relative POSIX path")
    path = project_paths.project_path(relative)
    if not path.is_file():
        raise V35TrainLaunchError(f"{name} does not exist: {relative.as_posix()}")
    return relative


def build_config(
    *,
    experiment_name: str,
    calibration: str | PurePosixPath,
    pilot_authorization: str | PurePosixPath = DEFAULT_PILOT_AUTHORIZATION,
    continuation_authorization: str | PurePosixPath | None = None,
    target: int = 1_000,
    fsdp_devices: int | None = None,
):
    """Build the exact calibrated resume config; ``train.main`` still reauthenticates it."""

    try:
        checkpoint_steps = CHECKPOINT_STEPS_BY_TARGET[target]
    except KeyError as exc:
        raise V35TrainLaunchError(f"target must be one of {tuple(CHECKPOINT_STEPS_BY_TARGET)}, got {target}") from exc
    calibration_relative = _relative_existing(calibration, name="calibration artifact")
    pilot_relative = _relative_existing(pilot_authorization, name="pilot authorization")
    if target == 1_000:
        if continuation_authorization is not None:
            raise V35TrainLaunchError("the 1,000-update pilot must not use a continuation authorization")
        continuation_relative = None
    else:
        if continuation_authorization is None:
            raise V35TrainLaunchError(f"target {target} requires a continuation authorization")
        continuation_relative = _relative_existing(
            continuation_authorization,
            name="continuation authorization",
        )

    config = bootstrap._calibrated_config(  # noqa: SLF001
        experiment_name=experiment_name,
        calibration_relative=calibration_relative,
        fsdp_devices=fsdp_devices,
    )
    return dataclasses.replace(
        config,
        resume=True,
        overwrite=False,
        num_train_steps=target,
        checkpoint_steps=checkpoint_steps,
        v35_pilot_authorization_path=pilot_relative.as_posix(),
        v35_continuation_authorization_path=(
            None if continuation_relative is None else continuation_relative.as_posix()
        ),
    )


def semantic_config_sha256(
    *,
    experiment_name: str,
    calibration: str | PurePosixPath,
    fsdp_devices: int | None = None,
) -> str:
    """Return the preauthorization semantic identity for the calibrated run.

    Authorization paths, resume mode, the target, and checkpoint rungs are intentionally
    excluded by the shared identity schema.  Consequently this value is exactly the one that
    finalized checkpoint 0 and every authorized training target must carry, while it can be
    obtained before the pilot authorization exists.
    """

    calibration_relative = _relative_existing(calibration, name="calibration artifact")
    config = bootstrap._calibrated_config(  # noqa: SLF001
        experiment_name=experiment_name,
        calibration_relative=calibration_relative,
        fsdp_devices=fsdp_devices,
    )
    return authorization.semantic_training_config_sha256(config)


def reauthenticate_pilot_evidence(config):
    """Re-run the production Gate-A/B/step-0 loaders against current source bytes."""

    import v35_gate_artifacts as artifacts
    import v35_pilot_gate as pilot
    import v35_training_authorization as reducer

    record = authorization.load_and_validate_pilot_authorization(config)
    manifest_path = Path(config.data.base_config.memory_episode_manifest_path)
    manifest = artifacts.load_frozen_manifest(
        manifest_path,
        expected_sha256=config.data.base_config.memory_episode_manifest_sha256,
    )
    evidence = record.payload["gate_evidence"]

    def evidence_path(name: str) -> Path:
        descriptor = evidence[name]
        relative = descriptor.get("path_relative") if isinstance(descriptor, dict) else None
        if not isinstance(relative, str):
            raise V35TrainLaunchError(f"pilot authorization has no project-relative {name} evidence path")
        return project_paths.project_path(relative)

    gate_a = reducer.load_data_gate_decision(evidence_path("gate_a"), manifest=manifest)
    gate_b = reducer._load_gate_b(evidence_path("gate_b"), manifest=manifest)  # noqa: SLF001
    rung = pilot.load_rung_result(evidence_path("step0"), manifest=manifest)
    loaded_ids = {
        "gate_a": gate_a["envelope"]["artifact_id"],
        "gate_b": gate_b["envelope"]["artifact_id"],
        "step0": rung.artifact_id,
    }
    expected_ids = {name: descriptor["artifact_id"] for name, descriptor in evidence.items()}
    if loaded_ids != expected_ids:
        raise V35TrainLaunchError("pilot evidence artifact identities changed during production revalidation")
    if rung.completed_updates != 0:
        raise V35TrainLaunchError("pilot authorization step0 evidence is not completed-update 0")
    if not pilot.summarize_gate_c(rung)["passes"] or not pilot.summarize_task_health(rung)["passes"]:
        raise V35TrainLaunchError("current production reducer does not pass step-0 Gate C/task health")
    return record


def verify_authorized_checkpoint0(config) -> None:
    """Restore and reauthenticate the complete pilot launch contract without an update."""

    if config.num_train_steps != 1_000 or config.checkpoint_steps != CHECKPOINT_STEPS_BY_TARGET[1_000]:
        raise V35TrainLaunchError("prelaunch verification is defined only for the frozen 1,000-update pilot")
    train_script._validate_v35_training_ready(config)  # noqa: SLF001
    pilot_authorization = reauthenticate_pilot_evidence(config)
    identity_path = Path(config.checkpoint_dir) / "initialization_manifest.json"
    initialization_identity = train_script._validate_v35_root_identity(config, identity_path)  # noqa: SLF001
    checkpoint_manager, resuming = train_script._checkpoints.initialize_checkpoint_dir(  # noqa: SLF001
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=False,
        resume=True,
        allow_step_zero_resume=True,
    )
    if not resuming or checkpoint_manager.latest_step() != 0:
        raise V35TrainLaunchError("verify-only requires the authorization-linked completed-update-0 checkpoint")
    train_script._validate_v35_checkpoint_protocol(config, resuming=True, latest_step=0)  # noqa: SLF001
    train_script._validate_v35_resume_checkpoint_assets(  # noqa: SLF001
        config,
        checkpoint_step=0,
        identity_path=identity_path,
    )
    mesh = train_script.sharding.make_mesh(config.fsdp_devices)
    data_sharding = train_script.jax.sharding.NamedSharding(
        mesh,
        train_script.jax.sharding.PartitionSpec(train_script.sharding.DATA_AXIS)
        if config.gradient_accumulation_steps == 1
        else train_script.jax.sharding.PartitionSpec(None, train_script.sharding.DATA_AXIS),
    )
    loader = train_script._data_loader.create_data_loader(  # noqa: SLF001
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    rng = train_script.jax.random.key(config.seed)
    _, init_rng = train_script.jax.random.split(rng)
    state_shape, _ = train_script.init_train_state(config, init_rng, mesh, resume=True)
    state, _, parameter_tree_sha256 = train_script._restore_and_validate_v35_authorized_source_checkpoint(  # noqa: SLF001
        config,
        checkpoint_manager=checkpoint_manager,
        checkpoint_step=0,
        state_shape=state_shape,
        data_loader=loader,
        source_authorization=pilot_authorization,
    )
    train_script.jax.block_until_ready(state)
    authorization.validate_pilot_run_binding(
        config,
        pilot_authorization,
        initialization_identity=initialization_identity,
        actual_parameter_tree_sha256=parameter_tree_sha256,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--calibration", required=True, help="Project-relative sealed train-54 calibration JSON.")
    parser.add_argument(
        "--pilot-authorization",
        default=DEFAULT_PILOT_AUTHORIZATION.as_posix(),
        help="Project-relative canonical pilot authorization.",
    )
    parser.add_argument(
        "--continuation-authorization",
        help="Required for target 2500 or 10000; forbidden for the 1000-update pilot.",
    )
    parser.add_argument("--target", type=int, choices=tuple(CHECKPOINT_STEPS_BY_TARGET), default=1_000)
    parser.add_argument("--fsdp-devices", type=int)
    parser.add_argument(
        "--print-semantic-config-sha256",
        action="store_true",
        help=(
            "Print the calibrated semantic training-config SHA256 without requiring an "
            "authorization or opening a checkpoint, then exit."
        ),
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Authenticate finalized checkpoint 0 and the pilot authorization, then exit without training.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.print_semantic_config_sha256 and args.verify_only:
        raise V35TrainLaunchError("choose only one of --print-semantic-config-sha256 and --verify-only")
    if args.print_semantic_config_sha256:
        print(
            semantic_config_sha256(
                experiment_name=args.experiment_name,
                calibration=args.calibration,
                fsdp_devices=args.fsdp_devices,
            )
        )
        return 0
    config = build_config(
        experiment_name=args.experiment_name,
        calibration=args.calibration,
        pilot_authorization=args.pilot_authorization,
        continuation_authorization=args.continuation_authorization,
        target=args.target,
        fsdp_devices=args.fsdp_devices,
    )
    if args.verify_only:
        verify_authorized_checkpoint0(config)
        print("v3.5 finalized checkpoint 0 and pilot authorization verified")
        return 0
    reauthenticate_pilot_evidence(config)
    train_script.main(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
