<script setup>
import { withBase } from 'vitepress'
</script>

# GF43X40-10

<div class="product-badges">
  <span class="product-badge">Reduction 40×</span>
  <span class="product-badge">8.9 N·m Rated</span>
  <span class="product-badge">28 N·m Peak</span>
</div>

The **GF43X40-10** is a compact mid-power variant of the GF43 joint module with a 40× planetary gearbox. Lighter and shorter than the GF43X40-16, it offers the same reduction ratio with a lower continuous torque rating but a notably higher peak torque ceiling — suitable for applications that require strong intermittent bursts.

## Dimensions

<div class="dim-gallery">
  <figure>
    <img :src="withBase('/images/motors/image5.webp')" alt="GF43X40-10 dimensions" />
    <figcaption>Detailed drawing with hole pattern and tolerances</figcaption>
  </figure>
</div>

## Specifications

| Parameter | Value |
|-----------|-------|
| Reduction Ratio | 40 |
| Rated Voltage | 24 V |
| Rated Bus Current | 2.3 A |
| Rated Phase Current | 2 A |
| Rated Power | 40 W |
| Rated Speed (output) | 43 RPM |
| Rated Torque | 8.9 N·m |
| Peak Power | 70 W |
| Peak Torque | 28 N·m |
| Torque Constant | 0.095 N·m/A |
| Weight | 362 g |
| Gearbox Backlash | 15 arcmin |
| Motor Size (⌀ × L, excl. pins) | 57 × 56.5 mm |
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

- [GF43X40-16](/products/motor-gf43x40-16) — higher continuous torque variant
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
