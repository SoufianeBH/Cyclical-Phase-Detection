# CyclePhase: Robust Phase Detection in Cardiovascular Imaging through Cyclic Motion Estimation

This repository contains the implementation for the MIDL 2026 submission:

**"CyclePhase: Robust phase detection in cardiovascular imaging through cyclic motion estimation"**
*Soufiane Ben Haddou, Rudolf L. M. van Herten, Connie R. Bezzina, R. Nils Planken, Joost Daemen, Jolanda J. Wentzel, José P. Henriques, Ivana Išgum*

## Overview

This repository contains the implementation of our method for accurate cardiac phase detection across cardiovascular imaging modalities. Unlike existing approaches that treat phase detection as discrete frame classification, we propose a fundamentally different approach that models cardiac phase as a **continuous cyclic variable on the unit circle**.

### Key Features

- **Continuous phase representation**: Treats cardiac phase as a cyclic variable on S¹, naturally encoding periodicity
- **Multi-modality support**: Unified framework for both IVUS and cardiac MRI without modality-specific preprocessing
- **Robust to artifacts**: Motion-focused gradient transforms isolate motion from appearance, handling calcifications and other artifacts
- **End-to-end learning**: No dependency on segmentation masks or ECG gating
- **Multi-objective optimization**: Combines balanced BCE, temporal Earth mover's distance (EMD), and circular phase regression

![Phase Detection Comparison](assests/phase_off_and_on.jpg)
*Qualitative comparison between our model without phase supervision (top) and with phase supervision (bottom) on challenging IVUS subsequences with severe imaging artifacts. The phase-supervised model produces cleaner, more temporally coherent predictions with well-localized ED frame peaks.*

## Method

### Architecture

Our framework consists of:

1. **Framewise Encoder**: ResNet-based 2D convolutional encoder processes each frame independently
2. **Temporal Decoder**: Recurrent neural network (GRU/LSTM or BiRNN) models temporal dependencies
3. **Dual Output Heads**:
   - Binary end-diastolic (ED) frame detection (logits ℓₜ)
   - Continuous phase vectors on unit circle (φ̂ₜ ∈ S¹)

### Input Transformations

**IVUS images**: Cartesian to polar coordinate transformation converts radial vessel motion into vertical displacement patterns easily captured by CNNs.

**CMR images**: Processed in native Cartesian coordinates to maintain end-to-end learning without segmentation dependencies.

**Gradient transforms**: Sobel filters in temporal direction create motion-sensitive representations emphasizing cardiac dynamics.

### Loss Function

Our multi-objective optimization combines:

```
L_total = λ_bce * L_BCE + λ_emd * L_EMD + λ_phase * L_phase
```

- **Balanced BCE**: Class-balanced binary cross-entropy for ED frame detection
- **Temporal EMD**: Earth mover's distance on cumulative distributions for temporal coherence
- **Phase regression**: Cosine similarity loss on S¹ for continuous phase alignment

### Continuous Phase Representation

Phase values φₜ ∈ [0, 2π) are embedded on the unit circle:

```
ϕₜ = (cos φₜ, sin φₜ), ‖ϕₜ‖₂ = 1
```

This avoids wrap-around discontinuities and ensures smooth transitions across cardiac cycle boundaries.

## Results

### Quantitative Performance

**IVUS Test Set:**

| Method | AUROC ↑ | AD [s] ↓ | F1 ↑ | THM ↑ |
|--------|---------|----------|------|-------|
| Bajaj et al. (2021) | 0.84 | 0.17 | 0.86 | 0.73 |
| Ours (BCE only) | 0.89 | 0.07 | 0.97 | 0.91 |
| **Ours (full)** | **0.90** | **0.06** | **0.99** | **0.93** |

**CMR Test Set:**

| Method | AUROC ↑ | AD [s] ↓ | F1 ↑ | THM ↑ |
|--------|---------|----------|------|-------|
| Bajaj et al. (2021) | 0.92 | 0.09 | 0.43 | 0.40 |
| Ours (BCE only) | 0.95 | 0.03 | 0.78 | 0.76 |
| **Ours (full)** | **0.97** | **0.02** | **0.78** | **0.76** |

### Robustness to Artifacts

On challenging IVUS subsequences with severe calcifications and acoustic shadowing:

- **IMPACT007-RCA**: AUROC improved from 0.672 → 0.907, AD reduced from 0.196s → 0.073s
- **IMPACT013-RCA**: AUROC improved from 0.642 → 0.785, AD reduced from 0.136s → 0.075s
- Consistently lower prediction entropy indicating more confident predictions

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{benhaddou2026robust,
  title={CyclePhase: Robust phase detection in cardiovascular imaging through cyclic motion estimation},
  author={Ben Haddou, Soufiane and van Herten, Rudolf L. M. and Bezzina, Connie R. and
          Planken, R. Nils and Daemen, Joost and Wentzel, Jolanda J. and
          Henriques, Jos{\'e} P. and I{\v{s}}gum, Ivana},
  note={Submitted to Medical Imaging with Deep Learning (MIDL) 2026},
  year={2025}
}
```

## License

© 2025 CC-BY 4.0
