"""Push policy-predicted joint commands into i2rt's MuJoCo SimRobot.

Loads predicted action chunks from `eval/results_<demo>.npz` and drives the
YAM SimRobot through them at the recorded fps. By default executes each
predicted chunk in full (receding-horizon: infer once per chunk_stride
steps, execute that chunk, re-infer). Set --chunk-stride 1 to fall back to
re-inferring every step (= using `pred_first[t]`, the original behavior,
which is jittery because each inference samples fresh noise).

Optionally with a live MuJoCo viewer. Saves the resulting sim joint
trajectory to `eval/sim_<demo>.npz`.

This is *passive* playback: the policy itself was already run open-loop in
`offline_eval.py`. Here we only visualize what the predicted commands look
like when applied to a kinematic/dynamic YAM model. The cup is *not* in the
sim scene; the arm just moves through empty space.

Usage:
    cd /home/kewalk/memory_project
    python eval/sim_playback.py                       # demo1, viewer ON, chunk_stride=10
    python eval/sim_playback.py --chunk-stride 1      # re-infer every step (original)
    python eval/sim_playback.py --no-viewer           # headless, faster
    python eval/sim_playback.py --speed 2.0           # 2x real time

Outputs:
    eval/sim_<demo>.npz
        sim_pos      (T, 7)       robot.get_joint_pos() after each command
        cmd_pos      (T, 7)       what we commanded (gripper-remapped)
"""

import argparse
import pathlib
import time

import numpy as np

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.utils import ArmType, GripperType

EVAL_DIR = pathlib.Path("/home/kewalk/memory_project/eval")
FPS = 30
N_ARM = 6  # joints 0..5 are arm; joint 6 is gripper


def clip_gripper(cmd: np.ndarray) -> np.ndarray:
    """Clip gripper component to [0, 1] command space (0=closed, 1=open).

    The YAM teleop dataset already records the gripper in [0, 1] command
    space (verified empirically: range ~[0.48, 1.00]), so we only clip and
    do not rescale. SimRobot internally maps [0, 1] -> MuJoCo joint range.
    """
    out = cmd.copy()
    out[:, N_ARM] = np.clip(cmd[:, N_ARM], 0.0, 1.0)
    return out


def build_chunk_schedule(pred_chunks: np.ndarray, stride: int) -> np.ndarray:
    """Stitch saved per-step action chunks into a (T, 7) command schedule.

    With stride=H (=action_horizon) we use pred_chunks[t][0..H-1] for steps
    [t, t+H) and only re-anchor at the next multiple of H. With stride=1 we
    use pred_chunks[t][0] every step (= pred_first, jittery).
    """
    T, H, A = pred_chunks.shape
    stride = max(1, min(stride, H))
    cmd = np.zeros((T, A), dtype=np.float32)
    for t in range(0, T, stride):
        end = min(t + stride, T)
        cmd[t:end] = pred_chunks[t][: end - t]
    return cmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", default="demo1")
    parser.add_argument("--no-viewer", action="store_true", help="Headless run")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier (1.0 = real-time)")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--chunk-stride", type=int, default=10,
                        help="Re-infer every N steps; up to action_horizon (10). "
                             "10 = execute full chunk; 1 = use pred_first every step.")
    args = parser.parse_args()

    in_path = EVAL_DIR / f"results_{args.demo}.npz"
    out_path = EVAL_DIR / f"sim_{args.demo}.npz"
    print(f"[load] {in_path}")
    data = np.load(in_path)
    state = data["state"]            # (T, 7)
    pred_chunks = data["pred_chunks"]  # (T, 10, 7)
    T = len(pred_chunks)
    if args.max_steps is not None:
        T = min(args.max_steps, T)
        state, pred_chunks = state[:T], pred_chunks[:T]
    cmd_raw = build_chunk_schedule(pred_chunks, args.chunk_stride)
    print(f"[load] {T} steps; chunk_stride={args.chunk_stride} "
          f"(re-infer every {args.chunk_stride} steps; "
          f"{int(np.ceil(T / args.chunk_stride))} chunks total)")

    print("[init] SimRobot YAM + LINEAR_4310 gripper")
    robot = get_yam_robot(arm_type=ArmType.YAM, gripper_type=GripperType.LINEAR_4310, sim=True)

    gripper_range = getattr(robot, "_gripper_qpos_range", None)
    print(f"[grip] sim gripper qpos range: {gripper_range}; data already in [0,1] cmd space — clip only")
    cmd = clip_gripper(cmd_raw)
    init_qpos = state[0].copy()
    init_qpos[N_ARM] = float(np.clip(state[0, N_ARM], 0.0, 1.0))

    print(f"[init] commanding initial joint pos {np.round(init_qpos, 3).tolist()}")
    robot.command_joint_pos(init_qpos)
    time.sleep(0.2)

    sim_pos = np.zeros((T, 7), dtype=np.float32)
    dt = (1.0 / FPS) / max(args.speed, 1e-6)

    if args.no_viewer:
        viewer_ctx = None
    else:
        import mujoco.viewer
        # Brighten the scene before launching the viewer so the default dark
        # skybox feels less oppressive. (Render-flag toggles like mjRND_SKYBOX
        # aren't exposed on the passive Handle; press 'R' in the viewer and
        # uncheck Skybox if you want a flat background.)
        robot._model.vis.headlight.ambient[:] = (0.6, 0.6, 0.6)
        robot._model.vis.headlight.diffuse[:] = (0.7, 0.7, 0.7)
        robot._model.vis.headlight.specular[:] = (0.3, 0.3, 0.3)
        robot._model.vis.rgba.haze[:] = (1.0, 1.0, 1.0, 1.0)
        robot._model.vis.rgba.fog[:] = (1.0, 1.0, 1.0, 1.0)
        viewer_ctx = mujoco.viewer.launch_passive(robot._model, robot._data)
        print("[view] passive viewer launched (close window or Ctrl-C to abort)")

    try:
        t0 = time.time()
        for t in range(T):
            robot.command_joint_pos(cmd[t])
            sim_pos[t] = robot.get_joint_pos()
            if viewer_ctx is not None:
                viewer_ctx.sync()
            elapsed = time.time() - t0
            target = (t + 1) * dt
            if elapsed < target:
                time.sleep(target - elapsed)
            if (t + 1) % 50 == 0 or t == T - 1:
                print(f"  step {t+1}/{T}  wall={elapsed:.1f}s")
    finally:
        if viewer_ctx is not None:
            viewer_ctx.close()

    np.savez(out_path, sim_pos=sim_pos, cmd_pos=cmd)
    print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
