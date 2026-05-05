<script setup>
import { withBase } from 'vitepress'
</script>

# GF43X40-16

<div class="product-badges">
  <span class="product-badge">Reduction 40×</span>
  <span class="product-badge">14 N·m Rated</span>
  <span class="product-badge">23.5 N·m Peak</span>
</div>

The **GF43X40-16** is the highest-power variant of the GF43 planetary joint module. It combines a 40× planetary gearbox with a 24 V brushless DC motor and integrated CAN driver. Gears are vacuum-nitrided for durability; the concentrated-winding motor is optimized for low cogging torque and smooth motion.

## Dimensions

<div class="dim-gallery">
  <figure>
    <img :src="withBase('/images/motors/image2.webp')" alt="GF43X40-16 overall dimensions" />
    <figcaption>Overall dimensions — three-view</figcaption>
  </figure>
  <figure>
    <img :src="withBase('/images/motors/image3.webp')" alt="GF43X40-16 detailed drawing" />
    <figcaption>Detailed drawing with hole pattern and tolerances</figcaption>
  </figure>
</div>

## Specifications

| Parameter | Value |
|-----------|-------|
| Reduction Ratio | 40 |
| Rated Voltage | 24 V |
| Rated Bus Current | 4 A |
| Rated Phase Current | 3.5 A |
| Rated Power | 65 W |
| Rated Speed (output) | 44 RPM |
| Rated Torque | 14 N·m |
| Peak Power | 90 W |
| Peak Torque | 23.5 N·m |
| Torque Constant | 0.096 N·m/A |
| Weight | 464.5 g |
| Gearbox Backlash | 15 arcmin |
| Motor Size (⌀ × L, excl. pins) | 57 × 62.5 mm |
| Communication | CAN bus |

> All speeds refer to the **output shaft** unless otherwise noted.

## Features

- Dual encoder with single-turn absolute output-shaft position — survives power-off
- Motor + driver in one housing; no external controller required
- CAN bus feedback: speed, position, torque, motor temperature
- Dual temperature protection (hardware + software)
- Trapezoidal acceleration/deceleration in position mode
- Visual debugging via host PC; supports firmware upgrade

## Where to Buy

Visit [i2rt.com](https://i2rt.com) or contact [sales@i2rt.com](mailto:sales@i2rt.com).

## See Also

- [GF43X40-10](/products/motor-gf43x40-10) — lighter variant, same reduction, lower continuous torque
- [GF43X10-10](/products/motor-gf43x10-10) — 10× reduction for higher-speed applications
- [Motors Overview](/products/motors)

<style scoped>
.product-badges { display: flex; flex-wrap: wrap; gap: 8px; margin: 16px 0 24px; }
.product-badge { display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; border: 1px solid; color: #4C6762; border-color: rgba(76,103,98,0.4); background: rgba(76,103,98,0.08); }
.dim-gallery { display: flex; flex-direction: column; gap: 16px; margin: 16px 0 24px; }
.dim-gallery figure { margin: 0; }
.dim-gallery img { width: 100%; border-radius: 6px; background: #fff; padding: 8px; box-sizing: border-box; }
.dim-gallery figcaption { font-size: 0.8rem; color: var(--vp-c-text-2); text-align: center; margin-top: 6px; }
</style>
