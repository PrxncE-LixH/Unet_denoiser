# Real-Time Monte Carlo Denoiser

A U-Net denoiser for 1-spp path-traced frames, trained on the BMFR dataset with
albedo demodulation and log-space loss. Currently spatial-only, temporal
reprojection is validated and pending integration.

![noisy / denoised / reference](docs/comparison.gif)

---

## Results

All PSNR computed on remodulated, Reinhard-tonemapped output at full 720×1280
resolution, reduced per image before the log.

**The input floor is reported alongside every number.** 

| Scene | Split | Input | Output | Gain |
|---|---|---|---|---|
| classroom | train | 9.89 dB | 33.25 dB | +23.4 dB |
| living-room | train | 10.92 dB | 34.73 dB | +23.8 dB |
| san-miguel | **held out** | 10.18 dB | 19.39 dB | +9.21 dB |

---

## Setup

<!-- TODO: fill in. Keep it short — someone should be able to reproduce in 5 minutes. -->

- **Dataset:** BMFR (Koskela et al. 2019), 7 scenes × 60 frames, 720×1280
  - Buffers: 1-spp color, albedo, shading normal, world position, reference
  - [Dataset link](https://etsin.fairdata.fi/dataset/0ab24b68-4658-4259-9f1d-3150be898c63)
- **Split:** held out `san-miguel` - grouped by base scene so Sponza variants stay together
- **Hardware:**  (ROCm 10 / Radeon RX 7900 XTX, Ubuntu)
- **Training:**  epochs= 25, Adam lr=1e-4, batch 32, 256×256 random crops, 16 patches/frame

### Input representation

12 channels: demodulated log-space color (3), shading normal (3), albedo (3),
world position (3).

- **Albedo demodulation** — `color / (albedo + 1e-2)`, applied to both input and
  target. Strips texture detail so the network only reconstructs lighting.
- **`log1p` transform** on color and reference only. L1 in log space approximates
  relative error in linear space, so fireflies don't dominate the gradient.
- **Loss:** L1 + 0.5 × finite-difference gradient loss (the edge term counters
  the blurring L1 alone produces).

---

## Findings

### 1. Generalization gap traced to dataset composition

14 dB between training scenes (~34 dB) and held-out (~19 dB).

Root cause: 4 of BMFR's 7 scenes are Sponza variants (base, glossy, moving-light,
static-camera), so effective scene diversity is 3 distinct environments, not 7.
The model learns environments rather than learning to denoise.

_Remedy in progress:_ rendering additional scenes in Unreal Engine.

---

## Temporal extension (in progress)

Reprojection is validated end-to-end (findings 3 and 4). Remaining work:

- [ ] Dataset returns consecutive frame pairs with a shared crop origin
- [ ] Cache reprojected history per frame pair
- [ ] Input channels 12 - 16 (warped history 3 + validity mask 1)
- [ ] Baseline vs temporal A/B on identical split and seed
- [ ] Frame-delta stability metric (PSNR cannot measure temporal flicker)

_Known limitation:_ training on ground-truth history while deploying on
self-generated history is exposure bias. Mitigation is fine-tuning with unrolled
self-generated history, not attempted yet.

---

## Deployment (in progress)

- [ ] ONNX export
- [ ] TensorRT fp16 engine
- [ ] Parity check: PyTorch fp32 vs TRT fp16 on identical input
- [ ] Falcor integration
- [ ] Frame-time measurement at 1080p

`expm1` and albedo remodulation stay outside the network — quantizing an
exponential amplifies error exponentially.

---

## References

- Koskela et al., *Blockwise Multi-Order Feature Regression for Real-Time
  Path Tracing Reconstruction* (BMFR), 2019
- Chaitanya et al., *Interactive Reconstruction of Monte Carlo Image Sequences
  using a Recurrent Denoising Autoencoder*, 2017

