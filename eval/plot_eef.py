"""FK + plots + error metrics for offline eval results.

Loads `eval/results_<demo>.npz` (and optionally `eval/sim_<demo>.npz`),
runs YAM forward kinematics on each joint trajectory, prints quantitative
error metrics, and saves three figures under `eval/figs/`:

    demo<X>_eef.png      EEF x/y/z time-series + 3D trajectory
                         (state | gt_action | pred | sim if available)
    demo<X>_gripper.png  Gripper joint angle (state | gt | pred | sim)
    demo<X>_error.png    Per-joint abs error + EEF error per timestep

Usage:
    cd /home/kewalk/memory_project
    python eval/plot_eef.py --demo demo1
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from i2rt.robots.kinematics import Kinematics
from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml

EVAL_DIR = pathlib.Path("/home/kewalk/memory_project/eval")
FIG_DIR = EVAL_DIR / "figs"
N_ARM = 6  # joints 0..5 are arm; 6 is gripper


def fk_xyz(kin: Kinematics, joints: np.ndarray) -> np.ndarray:
    """(T, 7) joints -> (T, 3) EEF xyz."""
    return np.stack([kin.fk(q[:N_ARM])[:3, 3] for q in joints])


def print_errors(name: str, pred: np.ndarray, gt: np.ndarray) -> None:
    err = pred - gt
    abs_err = np.abs(err)
    print(f"\n[{name} vs gt]")
    print(f"  per-joint MAE  : {np.round(abs_err.mean(axis=0), 4).tolist()}")
    print(f"  per-joint RMSE : {np.round(np.sqrt((err**2).mean(axis=0)), 4).tolist()}")
    print(f"  per-joint max  : {np.round(abs_err.max(axis=0), 4).tolist()}")
    print(f"  overall MAE    : {abs_err.mean():.4f} rad")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", default="demo1")
    args = parser.parse_args()

    res_path = EVAL_DIR / f"results_{args.demo}.npz"
    sim_path = EVAL_DIR / f"sim_{args.demo}.npz"
    print(f"[load] {res_path}")
    res = np.load(res_path)
    state = res["state"]            # (T, 7)
    gt_action = res["gt_action"]    # (T, 7)
    pred = res["pred_first"]        # (T, 7)
    T = len(state)

    sim_pos = None
    if sim_path.exists():
        print(f"[load] {sim_path}")
        sim = np.load(sim_path)
        sim_pos = sim["sim_pos"][:T]   # (T, 7)

    print("[init] YAM kinematics")
    kin = Kinematics(combine_arm_and_gripper_xml(ArmType.YAM, GripperType.NO_GRIPPER), "grasp_site")

    eef_state = fk_xyz(kin, state)
    eef_gt = fk_xyz(kin, gt_action)
    eef_pred = fk_xyz(kin, pred)
    eef_sim = fk_xyz(kin, sim_pos) if sim_pos is not None else None

    # ---- Quantitative errors ------------------------------------------------
    print_errors("pred_action", pred, gt_action)

    eef_err_pred_mm = np.linalg.norm(eef_pred - eef_gt, axis=1) * 1000  # mm
    print(f"\n[EEF pred vs gt]")
    print(f"  mean : {eef_err_pred_mm.mean():.2f} mm")
    print(f"  max  : {eef_err_pred_mm.max():.2f} mm")

    eef_err_sim_mm = None
    if sim_pos is not None:
        eef_err_sim_mm = np.linalg.norm(eef_sim - eef_gt, axis=1) * 1000
        print(f"\n[EEF sim vs gt]")
        print(f"  mean : {eef_err_sim_mm.mean():.2f} mm")
        print(f"  max  : {eef_err_sim_mm.max():.2f} mm")

    # ---- Plot 1: EEF time-series + 3D trajectory ----------------------------
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12, 8))
    for i, lbl in enumerate("xyz"):
        ax = fig.add_subplot(2, 2, i + 1)
        ax.plot(eef_state[:, i], "--", color="gray", alpha=0.6, label="state (current)")
        ax.plot(eef_gt[:, i], color="C0", label="gt action")
        ax.plot(eef_pred[:, i], color="C1", label="pred action")
        if eef_sim is not None:
            ax.plot(eef_sim[:, i], color="C2", label="sim", alpha=0.8)
        ax.set_xlabel("step")
        ax.set_ylabel(f"eef {lbl} [m]")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    ax3d = fig.add_subplot(2, 2, 4, projection="3d")
    ax3d.plot(*eef_gt.T, color="C0", label="gt")
    ax3d.plot(*eef_pred.T, color="C1", label="pred")
    if eef_sim is not None:
        ax3d.plot(*eef_sim.T, color="C2", label="sim", alpha=0.8)
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    ax3d.legend(fontsize=8)
    fig.suptitle(f"{args.demo}: end-effector trajectory")
    fig.tight_layout()
    out = FIG_DIR / f"{args.demo}_eef.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[save] {out}")

    # ---- Plot 2: gripper joint angle ----------------------------------------
    fig2, ax = plt.subplots(figsize=(10, 3))
    ax.plot(state[:, N_ARM], "--", color="gray", alpha=0.6, label="state")
    ax.plot(gt_action[:, N_ARM], color="C0", label="gt action")
    ax.plot(pred[:, N_ARM], color="C1", label="pred action")
    if sim_pos is not None:
        ax.plot(sim_pos[:, N_ARM], color="C2", label="sim", alpha=0.8)
    ax.set_xlabel("step")
    ax.set_ylabel("gripper [0=closed, 1=open]")
    ax.set_title(f"{args.demo}: gripper")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig2.tight_layout()
    out = FIG_DIR / f"{args.demo}_gripper.png"
    fig2.savefig(out, dpi=140)
    plt.close(fig2)
    print(f"[save] {out}")

    # ---- Plot 3: per-step error ---------------------------------------------
    fig3, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    abs_joint_err = np.abs(pred - gt_action)
    for j in range(7):
        axes[0].plot(abs_joint_err[:, j], label=f"j{j}", alpha=0.7)
    axes[0].set_ylabel("|pred - gt| [rad]")
    axes[0].set_title(f"{args.demo}: per-joint absolute error")
    axes[0].legend(ncol=7, fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(eef_err_pred_mm, color="C1", label="pred vs gt")
    if eef_err_sim_mm is not None:
        axes[1].plot(eef_err_sim_mm, color="C2", label="sim vs gt", alpha=0.8)
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("EEF error [mm]")
    axes[1].set_title("EEF position error")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig3.tight_layout()
    out = FIG_DIR / f"{args.demo}_error.png"
    fig3.savefig(out, dpi=140)
    plt.close(fig3)
    print(f"[save] {out}")


if __name__ == "__main__":
    main()
