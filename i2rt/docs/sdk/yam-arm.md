# YAM Arm API

## Import

```python
from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.utils import ArmType, GripperType
```

## `get_yam_robot()`

Factory function — the recommended way to create a robot instance.

```python
robot = get_yam_robot(
    channel="can0",
    arm_type=ArmType.YAM,
    gripper_type=GripperType.LINEAR_4310,
    zero_gravity_mode=True,
    ee_mass=None,
    ee_inertia=None,
    sim=False,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `channel` | `str` | `"can0"` | CAN interface name (e.g. `can0`, `can_follower_l`). Ignored in sim mode. |
| `arm_type` | `ArmType` | `ArmType.YAM` | Arm variant: `YAM`, `YAM_PRO`, `YAM_ULTRA`, `BIG_YAM` |
| `gripper_type` | `GripperType` | `GripperType.LINEAR_4310` | Gripper type. See [Grippers](/sdk/grippers) |
| `zero_gravity_mode` | `bool` | `True` | Enable gravity compensation on init |
| `ee_mass` | `float \| None` | `None` | End-effector mass override (kg) for MuJoCo inertial |
| `ee_inertia` | `np.ndarray \| None` | `None` | End-effector inertia override — 10-element array `[ipos(3), quat(4), diaginertia(3)]` |
| `sim` | `bool` | `False` | If `True`, return a `SimRobot` (no hardware needed) |

**Returns:** `MotorChainRobot` instance (or `SimRobot` when `sim=True`).

::: tip Arm types
Since v1.0, all arm variants use the same factory function. Pass `arm_type=ArmType.BIG_YAM` instead of the removed `get_big_yam_robot()`.
:::

::: tip Zero-gravity vs PD mode
With `zero_gravity_mode=True` the arm floats — great for teleoperation. With `False`, the arm holds its current joint positions as PD targets. Use `False` when operating without the motor safety timeout.
:::

---

## `MotorChainRobot`

### `num_dofs() → int`

Returns the number of controllable degrees of freedom (arm joints + gripper if present).

```python
n = robot.num_dofs()
# 7 (6 arm joints + 1 gripper), or 6 if no gripper
```

### `get_joint_pos() → np.ndarray`

Returns the current joint positions as a NumPy array in **radians**. Includes all DOFs (arm + gripper).

```python
q = robot.get_joint_pos()
# shape: (7,) for 6-joint arm + gripper
# shape: (6,) for arm-only (no_gripper)
```

### `command_joint_pos(joint_pos: np.ndarray) → None`

Commands all joints to move to `joint_pos` (radians). The controller uses PD tracking. For robots with a gripper, the last element controls the gripper position (normalized 0–1 for linear grippers).

```python
import numpy as np
robot.command_joint_pos(np.zeros(7))  # move arm to home, close gripper
```

::: warning Joint limits
Joint limits are defined in the MuJoCo XML model (e.g. `i2rt/robot_models/arm/yam/yam.xml`). A 0.15 rad safety buffer is applied automatically. Exceeding limits can cause motor errors.
:::

### `command_joint_state(joint_state: dict) → None`

Commands joints with position, velocity, and optional PD gains.

```python
robot.command_joint_state({
    "pos": target_positions,    # np.ndarray
    "vel": target_velocities,   # np.ndarray
    "kp": custom_kp,            # optional, overrides defaults
    "kd": custom_kd,            # optional, overrides defaults
})
```

### `get_observations() → dict`

Returns a dictionary of all robot observations. This is the **primary method** for reading robot state.

```python
obs = robot.get_observations()
# Without gripper:
#   obs["joint_pos"]  — (6,) joint positions
#   obs["joint_vel"]  — (6,) joint velocities
#   obs["joint_eff"]  — (6,) joint efforts (torques)
#
# With gripper:
#   obs["joint_pos"]  — (6,) arm joint positions only
#   obs["joint_vel"]  — (6,) arm joint velocities
#   obs["joint_eff"]  — (6,) arm joint efforts
#   obs["gripper_pos"] — (1,) gripper position
#   obs["gripper_vel"] — (1,) gripper velocity
#   obs["gripper_eff"] — (1,) gripper effort
```

### `move_joints(target, time_interval_s=2.0) → None`

Smoothly interpolates from the current position to `target` over `time_interval_s` seconds (blocking).

```python
robot.move_joints(np.zeros(7), time_interval_s=3.0)
```

### `zero_torque_mode() → None`

Enters zero-torque mode — all PD gains are set to zero, motors produce no active torque.

### `update_kp_kd(kp, kd) → None`

Update the PD gains at runtime.

```python
robot.update_kp_kd(kp=new_kp_array, kd=new_kd_array)
```

### `close() → None`

Safely shuts down the robot: stops the control thread and closes the CAN interface.

```python
robot.close()
```

---

## Command-line: Zero-gravity test

```bash
# Arm enters gravity-compensated floating mode
python i2rt/robots/get_robot.py \
  --channel can0 \
  --gripper linear_4310

# Test with BIG YAM
python i2rt/robots/get_robot.py \
  --arm big_yam \
  --channel can0

# Simulation mode (no hardware needed)
python i2rt/robots/get_robot.py --sim
```

---

## Motor Configuration Utilities

### Set timeout

```bash
# Disable the 400 ms safety timeout
python i2rt/motor_config_tool/set_timeout.py --channel can0
python i2rt/motor_config_tool/set_timeout.py --channel can0  # run twice

# Re-enable
python i2rt/motor_config_tool/set_timeout.py --channel can0 --timeout
```

### Zero motor offset

```bash
# Zero motor ID 1 (run for each motor 1–6 as needed)
python i2rt/motor_config_tool/set_zero.py --channel can0 --motor_id 1
```

---

## MuJoCo Models

The SDK uses MuJoCo for gravity computation, simulation, and visualization. Arm and gripper models are combined at runtime via `combine_arm_and_gripper_xml()`.

```
i2rt/robot_models/
├── arm/
│   ├── yam/yam.xml
│   ├── yam_pro/yam_pro.xml
│   ├── yam_ultra/yam_ultra.xml
│   └── big_yam/big_yam.xml
└── gripper/
    ├── crank_4310/crank_4310.xml
    ├── linear_3507/linear_3507.xml
    ├── linear_4310/linear_4310.xml
    ├── yam_teaching_handle/yam_teaching_handle.xml
    └── no_gripper/no_gripper.xml
```

Launch the visualizer without hardware:

```bash
python examples/minimum_gello/minimum_gello.py --mode visualizer_local
```

---

## Leader–Follower Script

`examples/minimum_gello/minimum_gello.py` is the primary teleoperation entry point:

```
usage: minimum_gello.py [options]

  --gripper TYPE        Gripper type (default: linear_4310)
  --mode MODE           follower | leader | visualizer_local
  --can-channel CHAN    CAN interface (default: can0)
  --bilateral_kp FLOAT  Bilateral stiffness 0.1–0.3 (default: 0.2)
```

Higher `bilateral_kp` = leader arm feels heavier (more force feedback from follower load).

---

## Arm Hardware Variants

| Arm Type | Shoulder Motors | Elbow/Wrist Motors | Gravity Factor | Notes |
|----------|----------------|-------------------|----------------|-------|
| `YAM` | 3× DM4340 | 3× DM4310 | 1.3 | Standard arm |
| `YAM_PRO` | 3× DM4340 | 3× DM4310 | 1.3 | Same as YAM |
| `YAM_ULTRA` | 3× DM4340 | 3× DM4310 | 1.3 | Different joint 3 limit |
| `BIG_YAM` | 2× DM6248 | 2× DM4340 + 2× DM4310 | 1.0 | Heavier, reversed joint 2 |

---

## See Also

- [Quick Start](/getting-started/quick-start)
- [Grippers](/sdk/grippers)
- [Bimanual Teleoperation Example](/examples/bimanual-teleoperation)
