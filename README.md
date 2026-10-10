# Real-Time Neural Denoiser for Path-Traced Frames

A U-Net denoiser for 1-spp path-traced frames, conditioned on G-buffers and a
temporally reprojected history buffer. Trained on a custom Unreal Engine 5
corpus. The model is **not yet real-time**. This will be addressed in the next iteration.


---

## Spatial Build 

Full-frame PSNR at 960×536, computed on remodulated, Reinhard-tonemapped output,
reduced per image before the log. The noisy input's own PSNR is reported
alongside every number

### Held-out scene (`San Miguel`)

| Model | PSNR | Gain over input |
|---|---|---|
| 1-spp input | ~10 dB | — |
| Spatial only | 28.29 dB | ~+18 dB |



## Spatial Build Video Demo 
[![noisy / denoised / reference Demo](assets/screenshot.png)](https://youtu.be/3pdgesbkxd0)

- Trained on the BMFR Dataset

## Setup

- **Dataset:** BMFR (Koskela et al. 2019), 7 scenes × 60 frames, 720×1280
  - Buffers: 1-spp color, albedo, shading normal, world position, reference
  - [Dataset link](https://etsin.fairdata.fi/dataset/0ab24b68-4658-4259-9f1d-3150be898c63)
- **Split:** held out `san-miguel`
- **Hardware:**  (ROCm 10 / Radeon RX 7900 XTX, Ubuntu)
- **Training:**  epochs= 25, Adam lr=1e-4, batch 32, 256×256 random crops, 16 patches/frame
---


## Temporal Build 

Temporal reprojection resulted in **33.71 dB** compared to the spatial baseline.

### Temporal stability

**Dataset** — 13 scenes rendered in Unreal Engine 5.8, 4 clips × 20 frames each,
960×540, discontinuous clips for variation per frame rendered.

| Pass | Contents |
|---|---|
| `A_1spp` | 1 spp path-traced color (noisy input) |
| `B_2048spp` | 2048 spp path-traced color (reference) |
| `C_gbuffer` | Albedo, Normal, WorldPosition, Depth, MotionVectors |

6 bounces, path-tracer denoiser disabled, no anti-aliasing
(`r.PathTracing.FilterWidth 0.01`), fixed manual exposure.

### Held-out scene (`Sponza`)

**Input representation** — 16 channels:

| Channels | Buffer |
|---|---|
| 3 | demodulated log-space color |
| 3 | shading normal, decoded from `[0,1]` to `[-1,1]` |
| 3 | albedo |
| 3 | world position, per-scene normalised, background masked |
| 3 | warped previous-frame output |
| 1 | reprojection validity mask |

- **Albedo demodulation** — `color / (albedo + 1e-2)` on both input and target.
  Strips texture so the network only reconstructs lighting.
- **`log1p`** on color and reference. L1 in log space approximates relative error
  in linear space, so fireflies don't dominate the gradient.
- **Loss** — L1 + 0.5 × finite-difference gradient loss (the edge term counters
  the blurring L1 alone produces).

**Training** — Adam, lr 1e-4, batch 32, 256×256 random crops, 16 patches/frame,
20 epochs, bf16 autocast. 


| Model | PSNR | Gain over input |
|---|---|---|
| 1-spp input | ~10 dB | — |
| **+ temporal reprojection** | **33.71 dB** | **~+24 dB** |


## Temporal Build Video Demo 
[![noisy / denoised / reference Demo](assets/Screenshot_01.png)](https://youtu.be/awtGA9TlpNk)

- Trained on a custom dataset captured from Unreal Engine
- Side by side comparison: https://youtu.be/prNKZ6O2gko

---

## Setup

## Findings

### 1. Scene diversity

BMFR ships 7 scenes, but 4 are Sponza variants — effectively 3 distinct
environments. 

Replacing it with 13 distinct Unreal scenes raised held-out performance, 
despite a comparable total frame count. Diversity was the binding constraint throughout.


### 2. Motion vectors: engine-provided, convention validated empirically

The capture manifest documents the encoding:

```
dx_px = (R - 0.5) * 2 * W,  dy_px = -(G - 0.5) * 2 * H
pixel (x, y) at frame t came from (x - dx_px, y - dy_px) at frame t-1
```

Validated by warping frame n−1's reference into frame n and measuring error
against the real frame n:

| | Validity | Mean err | Median err |
|---|---|---|---|
| Unreal engine vectors | 96.1% | 0.0070 | 0.0021 |

On BMFR, which ships no motion-vector buffers, the vectors were instead derived
by projecting world position through the previous frame's view-projection matrix
from `camera_matrices.h`. Matrix convention was determined from the data rather
than assumed: column 3's upper-3 norm is exactly `1.000000` (pure rotation row →
row-vector convention), and the column-1/column-0 ratio is `1.7777780` against
16/9 = `1.7777778`, confirming the projection is baked in at 1280×720.

The Y-axis direction was settled by measurement, not reasoning:

| | Validity | Median err |
|---|---|---|
| `flip_y=True` | 98.9% | 0.0873 |
| `flip_y=False` | 98.9% | **0.0018** |

48× separation. Note validity was *identical* for both — only the error metric
could distinguish them, because the camera pans horizontally.

### 3. Disocclusion rejection

Bounds checking alone only catches pixels that left the frame. It misses pixels
whose history landed on a *different surface* — the case that causes ghosting.
Reprojecting world position and shading normal and rejecting on squared
difference (BMFR's shipped thresholds: `position_limit_squared = 0.01`,
`normal_limit_squared = 1.0`):

| | Validity | Mean err | Median err |
|---|---|---|---|
| Bounds only | 98.9% | 0.0108 | 0.0018 |
| + surface validation | 95.6% | 0.0095 | 0.0016 |

The 3.3% rejected carried disproportionate error while the median barely moved —
the test fires on genuine disocclusions rather than over-rejecting.

Validation buffers must be sampled with `INTER_NEAREST`; bilinear across a depth
discontinuity blends two surfaces into a value belonging to neither.

## Deployment (in progress)

- [ ] Reduce architecture to a real-time budget 
- [ ] ONNX export
- [ ] TensorRT fp16 engine
- [ ] Parity check: PyTorch fp32 vs TRT fp16 on identical input
- [ ] Falcor integration and frame-time measurement

---

## References

- Koskela et al., *Blockwise Multi-Order Feature Regression for Real-Time Path
  Tracing Reconstruction* (BMFR), 2019
- Chaitanya et al., *Interactive Reconstruction of Monte Carlo Image Sequences
  using a Recurrent Denoising Autoencoder*, 2017
