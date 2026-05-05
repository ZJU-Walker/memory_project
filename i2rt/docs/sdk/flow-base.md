# Flow Base API

The Flow Base SDK has two layers:

| Class | Location | Use |
|-------|----------|-----|
| `Vehicle` | `flow_base_controller.py` | Runs **on-board** the Pi — joystick demo |
| `FlowBaseClient` | `flow_base_client.py` | Runs **remotely** — network Python API |

## `FlowBaseClient`

For remote control from your development machine.

```python
from i2rt.flow_base.flow_base_client import FlowBaseClient

client = FlowBaseClient(
    host="172.6.2.20",
    with_linear_rail=False,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `host` | `str` | `"localhost"` | IP address of the Flow Base Pi |
| `with_linear_rail` | `bool` | `False` | Set `True` if the linear rail module is installed |

::: tip No port parameter
The port is hardcoded internally using `BASE_DEFAULT_PORT`. You only need to specify the host IP.
:::

---

## Movement Commands

### `set_target_velocity(velocity, frame)`

```python
import numpy as np

# 3D (base only)
client.set_target_velocity(np.array([vx, vy, omega]), frame="local")

# 4D (base + linear rail)
client.set_target_velocity(np.array([vx, vy, omega, rail_vel]), frame="local")
```

| Parameter | Unit | Description |
|-----------|------|-------------|
| `vx` | m/s | Forward/backward |
| `vy` | m/s | Left/right (strafe) |
| `omega` | rad/s | Rotation (yaw rate) |
| `rail_vel` | rad/s | Linear rail speed (positive = up) |
| `frame` | — | `"local"` (relative to base) or `"global"` (world frame) |

::: warning Velocity must be a NumPy array
`set_target_velocity()` expects a `np.ndarray` with shape `(3,)` or `(4,)`. Pass `np.array([...])`, not a plain Python list.
:::

::: warning Velocity timeout
The base stops automatically if no command arrives within **0.25 seconds**. `FlowBaseClient` maintains a heartbeat automatically while connected via a background thread (20 ms interval).
:::

---

## Odometry

### `get_odometry() → dict`

```python
odom = client.get_odometry()
# {'translation': array([x, y]), 'rotation': array(theta)}
```

Wheel odometry only. Errors accumulate over time — integrate visual odometry (RealSense T265, ZED) for precise localization.

### `reset_odometry() → None`

Resets position and heading to zero.

---

## Linear Rail API

Only available when `with_linear_rail=True`.

### `get_linear_rail_state() → dict`

```python
state = client.get_linear_rail_state()
# {
#   'position': float,
#   'velocity': float,
#   'upper_limit_triggered': bool,
#   'lower_limit_triggered': bool,
# }
```

### `set_linear_rail_velocity(velocity: float) → None`

```python
client.set_linear_rail_velocity(0.5)    # raise
client.set_linear_rail_velocity(0.0)    # stop
client.set_linear_rail_velocity(-0.5)   # lower
```

::: tip Auto-homing
The rail homes to the **lower limit switch** on init. Ensure clearance below before powering on.
:::

---

## Cleanup

Always close the client when done to stop the background heartbeat thread:

```python
client.close()
```

---

## Command-line Client

Quick functional tests without writing Python:

```bash
# Read odometry
python i2rt/flow_base/flow_base_client.py --command get_odometry --host 172.6.2.20

# Reset odometry
python i2rt/flow_base/flow_base_client.py --command reset_odometry --host 172.6.2.20

# Run a short movement test (base will move)
python i2rt/flow_base/flow_base_client.py --command test_command --host 172.6.2.20

# Test linear rail (rail will move)
python i2rt/flow_base/flow_base_client.py --command test_linear_rail --host 172.6.2.20

# Monitor linear rail state
python i2rt/flow_base/flow_base_client.py --command get_linear_rail_state --host 172.6.2.20
```

---

## `Vehicle` (On-board Controller)

Runs directly on the Pi. Used for the joystick demo.

```python
from i2rt.flow_base.flow_base_controller import Vehicle
import time

v = Vehicle()
v.start_control()

start = time.time()
while time.time() - start < 2.0:
    v.set_target_velocity((0.15, 0.0, 0.0), frame="local")
```

---

## Coordinate Frames

| Frame | Description |
|-------|-------------|
| `local` | Relative to the current base orientation. Joystick forward = robot forward regardless of heading. |
| `global` | World frame from odometry zero. Similar to drone headless mode. Accumulates error. |

Switch frames at runtime via the remote **Mode** button, or programmatically by passing `frame=` to `set_target_velocity`.
