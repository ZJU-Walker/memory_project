<script setup>
import { withBase } from 'vitepress'
</script>

# YAM Cell

<div class="product-badges">
  <span class="product-badge available">✓ Python SDK</span>
  <span class="product-badge available">✓ Bimanual</span>
  <span class="product-badge available">✓ Data Generation</span>
</div>

**YAM Cell** is a complete bimanual teleoperation workstation built around two YAM leader arms and two YAM follower arms. It is designed for collecting high-quality manipulation demonstrations for training embodied AI models.

<div class="product-gallery">
  <figure>
    <img :src="withBase('/images/yam-station/DS-ST-1.webp')"  alt="YAM Cell full workstation" />
    
  </figure>
  <figure>
    <img :src="withBase('/images/yam-station/DS-ST-2.webp')"  alt="YAM Cell side view" />
    
  </figure>
  <figure>
    <img :src="withBase('/images/yam-station/DS-ST-3.webp')"  alt="YAM Cell detail" />
    
  </figure>
</div>


#### Mobile Variant

<div class="product-gallery">
  <figure>
    <img :src="withBase('/images/yam-mobile/YAM-Mobile-1.webp')"  alt="YAM Cell Mobile system" />
    
  </figure>
  <figure>
    <img :src="withBase('/images/yam-mobile/YAM-Mobile-2.webp')"  alt="YAM Cell Mobile 3/4 view" />
    
  </figure>
</div>

## System Overview

The YAM Cell pairs **leader arms** (with teaching handles) and **follower arms** in a bilateral teleoperation loop. An operator moves the leader arms naturally; the follower arms mirror the motion in real time with configurable force feedback.

```
Operator ──► Leader (YAM + teaching handle)
               │  bilateral link (CAN bus)
             Follower (YAM + task gripper)  ──► Task workspace
```

## Hardware Requirements

| Component | Qty | Notes |
|-----------|-----|-------|
| YAM Follower arms | 2 | Any YAM tier |
| YAM Leader arms | 2 | Must use `yam_teaching_handle` gripper |
| CANable USB-CAN adapters | 4 | One per arm |
| Workstation PC | 1 | Ubuntu 22.04 recommended |

## CAN Bus Layout

Each arm requires a dedicated CAN channel. Assign persistent names using udev rules (see [Hardware Setup](/getting-started/hardware-setup)):

| Arm | CAN name |
|-----|----------|
| Left follower | `can_follower_l` |
| Right follower | `can_follower_r` |
| Left leader | `can_leader_l` |
| Right leader | `can_leader_r` |

## Videos

<MediaPlaceholder
  type="video"
  description="Operator sitting in front of two YAM leader arms, controlling two follower arms picking objects. Side-by-side view of leader and follower workspace. 1–2 min demo."
/>

<video controls style="width:100%;border-radius:8px;margin:16px 0 8px">
  <source :src="withBase('/images/yam-station/DS-ST.mp4')" type="video/mp4" />
</video>


## Quick Start

### 1. Configure CAN IDs

Plug one CANable device at a time and assign each arm a persistent name:

```bash
# Follow the instructions in:
doc/set_persist_id_socket_can.md
```

### 2. Verify connectivity

```bash
ip a | grep can
# Expected:
# can_follower_r  UP
# can_follower_l  UP
# can_leader_r    UP
# can_leader_l    UP
```

### 3. Launch bimanual teleoperation

```bash
cd /path/to/i2rt
source .venv/bin/activate
python examples/bimanual_lead_follower/bimanual_lead_follower.py
```

### 4. Enable synchronization

Press the **top button** on either teaching handle once to begin tracking. Press again to pause.

## Control Reference

| Action | Result |
|--------|--------|
| Move leader arm | Follower mirrors motion |
| Squeeze trigger | Follower gripper closes |
| Top button (1×) | Enable sync |
| Top button (2×) | Pause sync |
| `--bilateral_kp` flag | Increase for more force feedback (default 0.2) |

::: tip Bilateral stiffness
Start with `--bilateral_kp 0.1` and increase gradually. Values above `0.3` make the leader arm feel noticeably heavy.
:::

## Data Logging

Trajectory recording is built into the example:

```python
from i2rt.robots.motor_chain_robot import get_yam_robot
# See examples/record_replay_trajectory/ for full dataset collection pipeline
```

## See Also

- [Bimanual Teleoperation Example](/examples/bimanual-teleoperation)
- [Record & Replay](/examples/record-replay)
- [YAM Arm API](/sdk/yam-arm)

<style scoped>
.product-badges { display: flex; flex-wrap: wrap; gap: 8px; margin: 16px 0 24px; }
.product-badge { display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; border: 1px solid; }
.product-badge.available { color: #4C6762; border-color: rgba(76,103,98,0.4); background: rgba(76,103,98,0.08); }
.product-gallery { display: flex; flex-wrap: wrap; gap: 16px; margin: 16px 0 8px; }
.product-gallery figure { flex: 1 1 220px; margin: 0; }
.product-gallery img { width: 100%; border-radius: 8px; }
.product-gallery figcaption { font-size: 0.8rem; color: var(--vp-c-text-2); text-align: center; margin-top: 6px; }
</style>
