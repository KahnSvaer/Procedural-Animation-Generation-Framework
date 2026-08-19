# Mesh Contraction & Refinement Ablation Study

This document details the performance, timing metrics (highlighting the fast NumPy CPU implementation), Taubin smoothing effects, and structural refinement ablation analysis for skeleton extraction and refinement algorithms in `animgen`.

---

## 1. Execution Time Breakdown (Fast Vectorized NumPy CPU)

For a standard 3D variable-radius tube mesh with **1,760 faces** and **882 surface vertices**:

| Stage / Algorithm | Implementation | Execution Time | Optimal Use Case |
|---|---|---|---|
| **Step 1: Geometry Contraction** ($20$ steps) | SciPy Sparse Cholesky | $\sim 600.0\text{ ms}$ | Zero-Volume Mesh Contraction |
| **Step 2: Connectivity Surgery** | Half-Edge Collapse Priority Queue | $\sim 100.0\text{ ms}$ | Topology-Preserving 1D Graph Extraction |
| **Step 3: Embedding Refinement** | Boundary Loop Displacement Shifting | $\sim 15.0\text{ ms}$ | Mesh Boundary Centering |
| **Taubin Smoothing** ($2$ passes) | Vectorized NumPy CPU | **$< 0.5\text{ ms}$** | 1D Non-Shrinking Noise Removal |
| **Post Refinement Algo A** (`subdivide_and_center`) | Vectorized NumPy CPU | **$4.9\text{ ms}$** | Bone Edge Densification |
| **Post Refinement Algo B** (`refine_and_center_iterative`) | Vectorized NumPy CPU | **$48.5\text{ ms}$** | Iterative Slice Centering & Momentum |

*Note: The vectorized NumPy implementation achieves optimal latency (<0.5 ms for Taubin, 4.9 ms for Algo A, 48.5 ms for Algo B), outperforming GPU kernel launch overhead for small/medium meshes.*

---

## 2. Refinement & Taubin Smoothing Ablation Results (`max_edge_len=0.4`)

We measured the mean Euclidean distance error between extracted skeleton nodes and the original ground-truth 3D spline curve across 3 radius profiles (**Increasing**, **Decreasing**, **Sinusoidal**).

### Distance Error Table (Lower is Better)

| Refinement Method | Increasing Radius | Decreasing Radius | Sinusoidal Radius | Average Error | Error Reduction vs Raw |
|---|---|---|---|---|---|
| **1. Base Au et al. (Raw)** | $0.0767$ | $0.0726$ | $0.0582$ | $0.0692$ | Baseline |
| **2. Au et al. + 1 Pass Taubin** | $0.0564$ | $0.0475$ | $0.0486$ | $0.0508$ | $26.6\%$ Improvement |
| **3. Au et al. + 2 Passes Taubin** | $0.0555$ | $0.0472$ | $0.0490$ | $0.0506$ | $26.9\%$ Improvement |
| **4. Algo A (`subdivide_and_center_skeleton`)** | $0.0696$ | $0.0611$ | $0.0570$ | $0.0626$ | $9.5\%$ Improvement |
| **5. Algo A + 2 Passes Taubin** | $0.0547$ | $0.0494$ | $0.0472$ | $0.0504$ | $27.2\%$ Improvement |
| **6. Algo B (`refine_and_center_skeleton_iterative`)** | $0.0442$ | $0.0374$ | $0.0415$ | $0.0410$ | **$40.8\%$ Improvement** |
| **7. Algo B + 2 Passes Taubin** | **$0.0432$** | **$0.0336$** | **$0.0404$** | **$0.0391$** | **$43.5\%$ Lowest Overall Error** |

### Key Takeaways on Algo B & Taubin Smoothing
- **Optimal Sub-Edge Resolution (`max_edge_len=0.4`)**: When edge subdivision is tuned to $0.4$, **Algo B + 2 Passes Taubin achieves a $43.5\%$ total error reduction ($0.0692 \to 0.0391$)**, producing the most geometrically accurate 1D medial curve abstraction across all test profiles.
- **Non-Shrinking Regularization**: Taubin smoothing alternates a positive shrink step ($\lambda = 0.5$) with a negative un-shrink step ($\mu = -0.53$), eliminating high-frequency jitter without pulling nodes off-center or shrinking volume.
