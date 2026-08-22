# Procedural Animation Generation Framework (`animgen`)
## Comprehensive Technical System Report

**Author / Maintainer:** Shivansh Pachnanda
**Date:** August 2026

---

## 1. Project Overview & Vision

The **Procedural Animation Generation Framework (`animgen`)** is an open-source end-to-end Python library and pipeline built to bridge the gap between static 3D aquatic mesh models, produced from Large Image to 3D Foundational models, with realistic procedural 3D motion synthesis.

Generating realistic organic motion (such as fish, marine mammals, or cephalopods swimming) requires manual rigging (constructing bone hierarchies, weight painting) and painstaking keyframe animation. While modern deep-learning-based motion generation tools exist, they often suffer from high compute latencies, lack of physical interpretability, mesh artifacts/distortions, and difficulty in fine-tuning kinematic parameters.

`animgen` solves this by combining **classical geometric algorithms** (Cotangent Laplace-Beltrami mesh contraction, medial graph extraction) with **biomechanical wave dynamics equations** and **zero-shot vision models** (SAM3) for morphological understanding of the input mesh.

### Key Architectural Pillars
1. **Zero-Shot Automatic Rigging:** Automatic transformation of raw, un-rigged 3D surface meshes into refined 1D skeletal armatures without manual landmark annotation.
2. **Procedural Kinematic Synthesis:** Mathematical formulation of aquatic locomotion (travelling and standing wave motion) applied dynamically across skeletal chains using Forward Kinematics (FK).
3. **Locomotion-Based Taxonomy:** Modular classification grouping species by locomotion mechanics (carangiform, anguilliform, rajiform) rather than strictly biological taxonomies.
4. **High-Performance CPU/GPU Processing Pipeline:** Memory-efficient, fast vectorized operations built on NumPy, SciPy Sparse solvers, Trimesh, and PyRender. 

---

## 2. Current Implementation Architecture (`animgen` Engine)

The framework is currently structured into five core sub-packages under the `animgen/` module, supported by exploratory experiments and automated test suites.

```
animgen/
├── core/         # Core data structures (Armature, Bone, Spline, Types)
├── rigging/      # Geometric mesh contraction, skeleton extraction, SDF, refinement
├── animation/    # Wave mechanics generators, FK propagation, spine straightening
├── renderer/     # OpenGL off-screen renderer, shader programs, PyRender wrappers
└── utils/        # Vector/matrix operations, math helpers, camera utilities
```

### 2.1 Core Data Structures (`animgen.core`)
* **[Armature & Bone](file:///media/kahnsvaer/Datasets/PsnlProjects/AnimationGenerationGSoC/animgen/core/armature.py):** Implements a flexible graph/tree structure for skeletal hierarchies. Supports connected bones (where child head directly meets parent tail) and disconnected offset bones, DFS graph validation, and serialization.
* **[Centripetal Catmull-Rom Spline](file:///media/kahnsvaer/Datasets/PsnlProjects/AnimationGenerationGSoC/animgen/core/spline.py):** Implements parameter-free spline interpolation avoiding self-intersections and cusp formation ($\alpha=0.5$). Automatically converts discrete control points into continuous 3D skeletal chains.

### 2.2 Skeleton Extraction & Rigging (`animgen.rigging`)
* **Mesh Contraction Algorithm (`mesh_contraction.py`):**
  Adapted from the seminal work *Au et al. (2008) "Skeleton Extraction by Mesh Contraction"*.
  - **Cotangent Laplace-Beltrami Operator:** Solves the sparse linear system $(W_L L + W_H) V^{t+1} = W_H V^t$ using SciPy's sparse Cholesky solver (`spsolve`).
  - **Topology-Preserving Graph Collapse:** Simplifies zero-volume contracted geometry into a single 1D skeletal graph using a half-edge priority queue.
* **Skeleton Refinement Algorithms (`refine_skelaton.py`):**
  - **Iterative Slice Centering (Algo B):** Computes cross-sectional boundary loops of original mesh surfaces and iteratively shifts skeletal nodes toward boundary centroids, while subdeviding extra nodes to increase node density.

### 2.3 Wave Motion Synthesis (`animgen.animation`)
* **Travelling Wave Model (`wave.py`):**
  Formulates continuous spatial-temporal undulation along the creature's spine:
  $$u(s, t) = A \cdot e^{g \cdot s} \cdot \sin\left(\frac{2\pi N}{L} s - \frac{2\pi}{T} t + \phi_s + \phi_t\right)$$
  - Parameters control spatial wavenumber ($N$), temporal period ($T$), base amplitude ($A$), and exponential growth rate ($g$) towards the tail.
* **Forward Kinematics & Spine Straightening (`straight.py`):**
  Algorithms to detect arbitrary curved initial mesh poses, compute spine curvature vectors, and unroll/straighten meshes into canonical rest configurations before wave propagation.

### 2.4 Rendering & Visualization (`animgen.renderer`)
* **PyRender Integration (`renderer.py`):** Off-screen headless OpenGL rendering supporting multi-view projection, surface normals, depth maps, face color mapping, and skeletal overlay rendering (`visualizations.py`).

---

## 3. Empirical Performance & Ablation Results

Based on our benchmarks documented in `docs/mesh_contraction_refinement_ablation.md`:

### 3.1 Timing Metrics (Tested on 1,760 Face / 882 Vertex Tube Mesh)
| Processing Stage | Sub-System / Method | Latency | Optimization Target |
|---|---|---|---|
| Geometry Contraction | SciPy Sparse Cholesky (20 Iterations) | $\sim 600.0\text{ ms}$ | Surface Contraction |
| Connectivity Collapse | Half-Edge Priority Queue | $\sim 100.0\text{ ms}$ | 1D Graph Extraction |
| Embedding Refinement | Boundary Displacement Shifting | $\sim 15.0\text{ ms}$ | Boundary Alignment |
| **Taubin Smoothing (2 Passes)** | **Vectorized CPU NumPy** | **$< 0.5\text{ ms}$** | **Jitter Removal** |
| **Refinement Algo A** | `subdivide_and_center` | **$4.9\text{ ms}$** | Edge Densification |
| **Refinement Algo B** | `refine_and_center_iterative` | **$48.5\text{ ms}$** | Centroid Centering |

*Key Insight:* The vectorized NumPy CPU implementation keeps post-contraction skeleton refinement under **$50\text{ ms}$**, eliminating GPU kernel dispatch overhead for typical real-time animation pipelines.

### 3.2 Geometrical Accuracy (Distance Error vs Ground Truth Spline)
| Refinement Pipeline Strategy | Average Distance Error | % Improvement vs Baseline |
|---|---|---|
| **1. Base Au et al. (Raw)** | $0.0692$ | Baseline |
| **2. Au et al. + 2 Passes Taubin** | $0.0506$ | $26.9\%$ |
| **3. Algo A + 2 Passes Taubin** | $0.0504$ | $27.2\%$ |
| **4. Algo B (`iterative_slice_centering`)** | $0.0410$ | $40.8\%$ |
| **5. Algo B + 2 Passes Taubin** | **$0.0391$** | **$43.5\%$ (Lowest Error)** |

---

## 4. Technical Stack Summary

* **Language:** Python 3.11+
* **Core Libraries:** `numpy`, `scipy`, `trimesh`, `open3d`, `pymeshlab`, `igraph`
* **Rendering & I/O:** `pyrender`, `bpy` (Blender), `pygltflib`, `pillow`, `opencv-python`
* **Quality Assurance:** `pytest`, `pytest-cov`, `pyright`, `ruff`
