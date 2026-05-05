# Motors

<div class="product-badges">
  <span class="product-badge available">✓ CAN Bus</span>
  <span class="product-badge available">✓ Dual Encoder</span>
  <span class="product-badge available">✓ Integrated Driver</span>
  <span class="product-badge available">✓ Planetary Reduction</span>
</div>

I2RT's **GF series** are all-in-one joint modules — planetary gearbox, brushless DC motor, and driver integrated into a single compact housing. Gears are vacuum-nitrided for high strength and service life. The concentrated-winding motor design delivers high power density with optimized pole-arc ratio for low cogging torque and smooth motion.

## Model Comparison

| Model | Reduction Ratio | Rated Torque | Peak Torque | Rated Speed | Weight | Size (⌀ × L) |
|-------|:-:|:-:|:-:|:-:|:-:|:-:|
| [GF43X40-16](/products/motor-gf43x40-16) | 40 | 14 N·m | 23.5 N·m | 44 RPM | 464.5 g | 57 × 62.5 mm |
| [GF43X40-10](/products/motor-gf43x40-10) | 40 | 8.9 N·m | 28 N·m | 43 RPM | 362 g | 57 × 56.5 mm |
| [GF43X10-10](/products/motor-gf43x10-10) | 10 | 2.2 N·m | 7 N·m | 172 RPM | 300 g | 57 × 45.9 mm |

All models share the same 24 V supply and CAN bus interface.

## Key Features

- **Dual encoder** — single-turn absolute output-shaft position; survives power-off without losing zero reference
- **Integrated driver** — compact single-unit assembly with no external controller required
- **CAN bus feedback** — real-time speed, position, torque, and motor temperature
- **Dual temperature protection** — hardware and software over-temperature shutdown
- **Trapezoidal motion profile** — configurable acceleration/deceleration in position mode
- **Visual debugging** — host-PC GUI for tuning and firmware upgrade

## Communication

All GF series motors communicate over **CAN bus (1 Mbit/s)**, the same bus used by YAM arm actuators. Multiple motors can share a single CAN channel using standard node addressing.

```python
# Motor control follows the same CAN pattern as YAM joints
# See the YAM Arm SDK for protocol details
```

## Naming Convention

```
GF  43  X  40  -  16
│   │      │      └─ Motor variant (current/winding spec)
│   │      └─────── Reduction ratio
│   └────────────── Outer diameter class (43 mm motor stator)
└────────────────── GF joint module series
```

<style scoped>
.product-badges { display: flex; flex-wrap: wrap; gap: 8px; margin: 16px 0 24px; }
.product-badge { display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; border: 1px solid; }
.product-badge.available { color: #4C6762; border-color: rgba(76,103,98,0.4); background: rgba(76,103,98,0.08); }
</style>
