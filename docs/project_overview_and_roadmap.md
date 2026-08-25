# Procedural Animation Generation Framework (`animgen`)
## Comprehensive Technical System Report

**Author / Maintainer:** Shivansh Pachnanda  
**Date:** August 2026  
**Repository:** `KahnSvaer/Procedural-Animation-Generation-Framework`  

---

## 1. Project Overview & System Architecture

The **Procedural Animation Generation Framework (`animgen`)** is an open-source Python library and pipeline designed to transform static 3D aquatic mesh models (generated from Large Image-to-3D Foundational models) into fully rigged, articulated, and procedurally animated assets with biomechanically grounded swimming motions.

### Core Architectural Pillars
1. **Zero-Shot Automatic Rigging:** Automatically extracts 1D skeletal curves and graph hierarchies from arbitrary watertight or non-manifold meshes.
2. **Robust Skin Weight Diffusion:** Computes geometry-aware per-vertex bone influences via discrete Laplace-Beltrami heat diffusion with automated UV seam welding.
3. **Volume-Preserving Straightening & Deformation:** Unrolls curved rest poses via parallel-transport Bishop frames and deforms surfaces using Dual Quaternion Skinning (DQS / QBS) to eliminate joint collapse.
4. **Biomechanical Locomotion Synthesis:** Generates continuous travelling, standing, and pulse wave motions across skeletal chains using Forward Kinematics (FK) and SLERP keyframing.
5. **Standardized Interoperability:** Exports self-contained glTF 2.0 binary (`.glb`) packages with bone transforms, inverse bind matrices, skinning weights, and animation samplers.

---

## 2. Standardized Subsystem Specifications

### 2.1 Core Data Structures & Models (`animgen.core`)

* **Role & Overview:** Provides foundational skeletal graph hierarchies, curve parameterizations, and unified object-oriented model management.
* **Mathematical & Algorithmic Formulation:**
  - **Centripetal Catmull-Rom Spline (`spline.py`):** Parameterized knot intervals $t_{i+1} = t_i + \|P_{i+1} - P_i\|^\alpha$ with $\alpha = 0.5$ (centripetal), mathematically preventing self-intersections and cusp formation.
  - **Phantom Boundary Extrapolation:** Employs boundary velocity $v_0 = P_1 - P_0$ and acceleration $a_0 = P_2 - 2P_1 + P_0$ to construct phantom control points $P_{-1} = P_0 - v_0 + \frac{1}{2}w_{\text{acc}}a_0$, ensuring $C^1$ boundary continuity.
* **Key Implementation Highlights:**
  - **`Armature` & `Bone` (`armature.py`):** Directed acyclic graph hierarchy supporting connected and offset bones, cycle validation, and serialization.
  - **`BaseModelClass` (`model.py`):** High-level pipeline unifying geometry pre-processing, multi-view rendering, rigging, skinning, baking, and GLB export.

---

### 2.2 Morphological Segmentation (`animgen.rigging`)

* **Role & Overview:** Isolates anatomical appendages (dorsal fins, caudal fins, pectoral flippers) from the main torso to inform skeletal branching and bone influence masking.
* **Mathematical & Algorithmic Formulation:**
  - **Shape Diameter Function (`shape_diameter_function.py`):** Casts inward ray cones ($N=64, \theta=120^\circ$) from face centroids to measure local diameter $\text{SDF}(f)$, normalized via logarithmic scaling:
    $$\text{SDF}_{\text{norm}}(f) = \frac{\ln\left(\frac{\text{SDF}(f) - \text{SDF}_{\min}}{\text{SDF}_{\max} - \text{SDF}_{\min}} \cdot \alpha + 1\right)}{\ln(\alpha + 1)}$$
  - **Alpha-Expansion Graph Cuts:** Fits a 1D Gaussian Mixture Model (GMM) to SDF values and solves the Markov Random Field energy:
    $$E(\mathcal{P}) = \sum_{f \in F} -\ln P(\text{SDF}(f) \mid \mathcal{P}_f) + \lambda \sum_{(f_1, f_2) \in \mathcal{E}_{\text{adj}}} -\ln\left(\frac{\theta_{f_1, f_2}}{\pi} + \epsilon\right) \cdot \mathbb{I}(\mathcal{P}_{f_1} \neq \mathcal{P}_{f_2})$$
* **Key Implementation Highlights:**
  - **Zero-Shot Vision Foundation Model (`SAM3.py`):** Evaluates multi-view 2D open-vocabulary text prompts (`"fin"`, `"tail"`) with Meta SAM3 and backprojects masks onto 3D mesh triangles.

---

### 2.3 Skeleton Extraction & Refinement (`animgen.rigging`)

* **Role & Overview:** Contracts 3D surface meshes into degenerate zero-volume medial curves and extracts smooth, centered 1D skeletal graphs.
* **Mathematical & Algorithmic Formulation:**
  - **Cotangent Laplace-Beltrami Contraction (`mesh_contraction.py`):** Solves the linear system:
    $$\begin{bmatrix} W_L L \\ W_H \end{bmatrix} V^{t+1} = \begin{bmatrix} 0 \\ W_H V^t \end{bmatrix}$$
    where $L_{ij} = \frac{1}{2}(\cot \alpha_{ij} + \cot \beta_{ij})$ and diagonal weights $W_L, W_H$ iteratively contract geometry while preserving topological shape.
  - **Boundary Slice Centering (`refine_skelaton.py`):** Intersects orthogonal cutting planes $\Pi = (\mathbf{p}, \mathbf{t})$ with uncontracted mesh surfaces and shifts skeletal nodes to the cross-sectional 2D polygon centroids.
* **Key Implementation Highlights:**
  - **Topology-Preserving Collapse:** Simplifies contracted meshes into 1D line graphs using a half-edge priority queue.
  - **Taubin Low-Pass Filter:** Vectorized 2-pass smoothing ($\lambda=0.5, \mu=-0.53$) eliminates high-frequency jitter without skeletal shrinkage.

---

### 2.4 Automatic Bone Heat Skinning Weight Engine (`animgen.rigging.skinning`)

* **Role & Overview:** Computes smooth, geometry-aware per-vertex bone influence matrices $W \in \mathbb{R}^{N \times K}$ based on the discrete heat diffusion formulation (Baran & Popović 2007, Blender `meshlaplacian.cc`).
* **Mathematical & Algorithmic Formulation:**
  - **Heat Diffusion Linear System:** Solves the steady-state boundary conduction system:
    $$(L + M H) W = M H P$$
    where $L$ is the discrete cotangent Laplacian, $M_{ii} = \frac{1}{3}\sum_{f \in \mathcal{F}(i)} \text{Area}(f)$ is the lumped Voronoi mass matrix, and $P_{ik} = \mathbb{I}(k = \arg\min_j d(v_i, \text{bone}_j))$ is the nearest-bone Dirichlet prior.
  - **Conduction Operator:** $H_{ii} = \frac{c}{\max(d_{\min}(v_i)^2, \epsilon)}$ scales heat transfer based on shortest point-to-segment Euclidean distance $d_{\min}(v_i)$.
* **Key Implementation Highlights:**
  - **Sparse SPD Cholesky Solve:** Factorizes $(L + MH)$ once via `scipy.sparse.linalg.factorized` for simultaneous multi-bone computation in $< 20\text{ ms}$.
  - **Topological Seam Welding:** Automatically detects coincident vertices along UV seams/split normals ($\delta=10^{-5}$), solves diffusion on the welded manifold, and broadcasts weights back to prevent visual seam tearing.
  - **Normalization & Fallback:** Enforces $\sum_k W_{ik} = 1.0$ and non-negativity $W_{ik} \ge 0$, with automatic inverse-distance fallback for degenerate non-manifold geometry.

---

### 2.5 Canonical Mesh Straightening & Kinematics (`animgen.animation`)

* **Role & Overview:** Canonicalizes curved rest poses from image-to-3D generators into straight configurations and propagates skeletal rotations down the hierarchy.
* **Mathematical & Algorithmic Formulation:**
  - **Bishop Parallel Transport Frame (`straight.py`):** Constructs continuous singularity-free orthonormal moving frames $(T, N, B)$ along spine paths:
    $$T_i = \frac{P_{i+1} - P_i}{\|P_{i+1} - P_i\|}, \quad N_i = \frac{N_{i-1} - \langle N_{i-1}, T_i \rangle T_i}{\|N_{i-1} - \langle N_{i-1}, T_i \rangle T_i\|}, \quad B_i = T_i \times N_i$$
  - **Cylindrical Coordinate Unrolling:** Maps mesh vertices to $(s, r, \theta)$ spine coordinates and evaluates them onto a straight target spine, achieving **exact volume conservation**.
  - **Forward Kinematics (`kinematics.py`):** Propagates global orientations and positions:
    $$R_{\text{global}, b} = R_{\text{global}, \text{parent}(b)} \cdot R_{\text{local}, b}, \quad h_{\text{pose}, b} = h_{\text{pose}, \text{parent}(b)} + R_{\text{global}, \text{parent}(b)} (h_{\text{rest}, b} - h_{\text{rest}, \text{parent}(b)})$$
* **Key Implementation Highlights:**
  - **GPU Acceleration:** Provides both vectorized NumPy and CUDA PyTorch tensor implementations (`deform_mesh_to_spine_torch`).
  - **Singularity Free:** Parallel transport avoids artificial twisting at spine inflection points where Frenet-Serret frames fail.

---

### 2.6 Procedural Wave Locomotion Dynamics (`animgen.animation.wave`)

* **Role & Overview:** Synthesizes spatial-temporal harmonic undulations for axial swimming, oscillatory fin flapping, and localized pulse swimming.
* **Mathematical & Algorithmic Formulation:**
  - **Travelling Wave Generator:** Models carangiform and anguilliform swimming along cumulative spine distance $s$:
    $$u(s, t) = A \cdot e^{g \cdot s} \cdot \sin\left(\frac{2\pi N}{L} s - \frac{2\pi}{T} t + \phi_s + \phi_t\right)$$
  - **Standing Wave Generator:** Models oscillatory fin flapping (pectoral and dorsal fins):
    $$u(s, t) = A \cdot e^{g \cdot s} \cdot \sin\left(\frac{2\pi N}{L} s + \phi_s\right) \cos\left(\frac{2\pi}{T} t + \phi_t\right)$$
  - **Pulse Wave Generator:** Models transient escape responses ("C-start") and jellyfish contractions:
    $$u(s, t) = A \cdot \exp\left(-\frac{(s - v t - s_0)^2}{2\sigma^2}\right) \cdot \sin\left(k(s - v t) + \phi\right)$$
* **Key Implementation Highlights:**
  - **Chain Wave Dispatcher (`chain_wave_generator`):** Maps spatial wave offsets into parent-relative rotation matrices with customizable 3D steering rotations (`steer_rotation`) in $SO(3)$.
  - **`AnimationClip` & `Animator` (`clip.py`, `animator.py`):** Multi-track timeline aggregation, SLERP keyframe interpolation, and offline mesh baking (`animator.bake()`).

---

### 2.7 Mesh Skinning Deformation Engines (`animgen.animation.deformation`)

* **Role & Overview:** Deforms 3D surface meshes from skeletal pose transformations using Linear Blend Skinning (LBS) or Dual Quaternion Skinning (DQS / QBS).
* **Mathematical & Algorithmic Formulation:**
  - **Linear Blend Skinning (LBS):** Convex combination of bone affine transformations:
    $$v' = \sum_{b=1}^{K} w_{v, b} \left( h_{\text{pose}, b} + R_{\text{global}, b} (v_{\text{rest}} - h_{\text{rest}, b}) \right)$$
  - **Dual Quaternion Skinning (DQS / QBS):** Unit dual quaternion $\hat{q}_b = q_{0, b} + \epsilon q_{d, b}$ with rotation $q_{0, b} = \text{quat}(R_{\text{global}, b})$ and translation $q_{d, b} = \frac{1}{2}(0, t_b) \otimes q_{0, b}$ where $t_b = h_{\text{pose}, b} - R_{\text{global}, b} h_{\text{rest}, b}$.
  - **Dual Quaternion Linear Blending (DLB):** Blends antipodally aligned dual quaternions:
    $$\hat{q}_{\text{norm}} = \frac{\sum_{b=1}^K w_{v, b} \hat{q}_b}{\|\sum_{b=1}^K w_{v, b} q_{0, b}\|} = [w_0, v_0] + \epsilon [w_d, v_d]$$
    $$v' = v_{\text{rest}} + 2 w_0 (v_0 \times v_{\text{rest}}) + 2 (v_0 \times (v_0 \times v_{\text{rest}})) + 2(w_0 v_d - w_d v_0 + v_0 \times v_d)$$
* **Key Implementation Highlights:**
  - **Volume Preservation:** DQS strictly preserves cross-sectional muscle volume and eliminates "candy-wrapper" joint collapsing during spine twisting.
  - **High Throughput:** Vectorized DLB processes 470+ FPS on CPU, while LBS delivers 800+ FPS.

---

### 2.8 Headless Rendering & Camera Systems (`animgen.renderer`, `animgen.utils`)

* **Role & Overview:** Produces off-screen multi-channel visual buffers (RGB, surface normals, depth maps, face IDs) and spatial visualizations.
* **Mathematical & Algorithmic Formulation:**
  - **Spherical Camera Distributions (`camera_position.py`):** Generates deterministic viewpoints using Dodecahedron (12 views), Icosahedron (20 views), or Fibonacci golden spiral distributions on $S^2$.
* **Key Implementation Highlights:**
  - **PyRender OpenGL Engine (`renderer.py`):** Headless rendering supporting EGL and OSMesa backends with custom Phong/Gouraud shaders.
  - **Overlays (`visualizations.py`):** Visualizes 3D skeletal bone graphs with colored joint spheres and surface heatmaps for skinning weights and SDF values.

---

### 2.9 glTF 2.0 / GLB Binary Output Pipeline (`animgen.io.glb_output`)

* **Role & Overview:** Packages rigged meshes, skeletal hierarchies, skinning weights, and animation clips into standard `.glb` binary files.
* **Mathematical & Algorithmic Formulation:**
  - **Inverse Bind Matrices:** Computes aligned $4 \times 4$ model-space inverse bind matrices $(\text{BindMatrix}_b)^{-1}$ packed into `MAT4` binary buffer views.
  - **Attribute Quantization:** Encodes per-vertex bone indices and normalized weights into standard `JOINTS_0` (`VEC4` uint16) and `WEIGHTS_0` (`VEC4` float32) accessors.
* **Key Implementation Highlights:**
  - **Direct Engine Compatibility:** Verified across Blender, Three.js, Unity, Unreal Engine, and VS Code 3D Mesh Viewer.
  - **Animation Samplers:** Encodes procedural wave tracks into glTF `translation` and `rotation` channels with `LINEAR` interpolation samplers.

---

### 2.10 Biomechanical Locomotion Taxonomy (`docs/species-classification.md`)

Organizes aquatic motion generation into 7 distinct biomechanical swimming paradigms:

| Category | Representative Species | Propulsion Mechanism | Procedural Wave Configuration |
|---|---|---|---|
| **1. Carangiform / Fish-Like** | Fish, Sharks, Tuna, Dolphins | Posterior caudal fin oscillation | Travelling wave with $g > 0$, concentrated on posterior $50\%$ of spine |
| **2. Anguilliform / Serpentine** | Eels, Sea Snakes, Lampreys | Continuous full-body undulation | Full-spine travelling wave ($100\%$ length, $N \ge 2.0$) |
| **3. Rajiform / Flat Fin** | Manta Rays, Stingrays, Skates | Pectoral fin-wave undulation | Symmetrical dual standing/travelling waves along lateral fin bones |
| **4. Labriform / Flipper** | Sea Turtles, Penguins, Sea Lions | Pectoral stroke cycles | Forward kinematic cyclic power and recovery stroke state machine |
| **5. Cephalopods** | Octopuses, Squids, Cuttlefish | Multi-tentacle reach & pulse jetting | Multi-chain tentacle wave coordination + pulse wave mantle contraction |
| **6. Gelatinous Organisms** | Jellyfish, Comb Jellies | Rhythmic bell contraction | Radial pulse wave displacement with passive drag relaxation |
| **7. Arthropods** | Crabs, Lobsters, Shrimp | Multi-limb segmented gait | Multi-leg synchronized gait kinematic state machine |

---

### 2.11 Interactive Demonstrators & Tooling (`demo/`)

* **Unity Real-Time Visualizer (`demo/static_visualizer/UnityStaticVisualizer`):** Unity C# project featuring real-time GPU skinning (`gpuSkinning: 1`), camera orbit controls, and live skeletal animation playback for exported `.glb` models.
* **Kaggle Server Compute Backend (`demo/static_visualizer/KaggleServer.ipynb`):** Remote GPU compute pipeline executing heavy foundation models (SAM3, mesh contraction) with ngrok streaming.
* **Web Animation Editor (`demo/web_animation_generation/animation-editor`):** Interactive web UI for adjusting wave amplitude, period, growth factor, and previewing DQS/LBS mesh deformations.

---

## 3. Empirical Performance, Accuracy & Ablation Benchmarks

### 3.1 End-to-End Latency Benchmark
*(Tested on 1,760 Face / 882 Vertex Mesh, 3-Bone Armature, CPU Single-Threaded)*

| Processing Stage | Sub-System / Method | Latency | Target Function |
|---|---|---|---|
| Geometry Contraction | SciPy Sparse Cholesky ($W_L L + W_H$) (20 Iters) | $\sim 600.0\text{ ms}$ | Medial Surface Contraction |
| Connectivity Collapse | Half-Edge Priority Queue | $\sim 100.0\text{ ms}$ | 1D Graph Extraction |
| Embedding Refinement | Boundary Displacement Shifting | $\sim 15.0\text{ ms}$ | Boundary Alignment |
| **Taubin Smoothing (2 Passes)** | **Vectorized CPU NumPy** | **$< 0.5\text{ ms}$** | **High-Frequency Jitter Removal** |
| **Refinement Algo A** | `subdivide_and_center` | **$4.9\text{ ms}$** | Edge Densification |
| **Refinement Algo B** | `refine_and_center_iterative` | **$48.5\text{ ms}$** | Centroid Centering |
| **Shape Diameter Function (SDF)** | Ray-Cone Sampling + GMM Partition | **$\sim 320.0\text{ ms}$** | Anatomical Part Segmentation |
| **Bishop Frame Straightening (NumPy)** | Parallel Transport Coordinate Unrolling | **$\sim 12.4\text{ ms}$** | Canonical Rest-Pose Alignment |
| **Bishop Frame Straightening (CUDA)** | Batched PyTorch GPU Tensors | **$< 1.8\text{ ms}$** | Real-Time GPU Unrolling |
| **Bone Heat Weight Solver** | **Sparse Cotangent Laplacian ($L + MH$) SPD** | **$\sim 18.2\text{ ms}$** | **Skinning Weight Field** |
| **LBS Mesh Deformation** | Vectorized CPU Matrix Formulation | **$< 1.2\text{ ms}$ / frame** | Linear Surface Deformation |
| **DQS / QBS Mesh Deformation** | Vectorized Dual Quaternion DLB | **$< 2.1\text{ ms}$ / frame** | Volume-Preserving Deformation |

---

### 3.2 Skeletal Extraction Accuracy (Distance Error vs Ground Truth Spline)
| Refinement Strategy | Average Distance Error | % Improvement vs Baseline |
|---|---|---|
| **1. Base Au et al. (Raw)** | $0.0692$ | Baseline |
| **2. Au et al. + 2 Passes Taubin** | $0.0506$ | $26.9\%$ |
| **3. Algo A + 2 Passes Taubin** | $0.0504$ | $27.2\%$ |
| **4. Algo B (`iterative_slice_centering`)** | $0.0410$ | $40.8\%$ |
| **5. Algo B + 2 Passes Taubin** | **$0.0391$** | **$43.5\%$ (Lowest Error)** |

---

### 3.3 Bone Heat Skinning Parity vs. Blender Ground Truth (`meshlaplacian.cc`)

| Metric | Measured Value | Acceptance Threshold | Evaluation |
|---|---|---|---|
| **Mean Absolute Weight Difference ($\Delta W_{\text{mean}}$)** | **$0.024$** | $< 0.05$ | **Passed (High Parity)** |
| **Maximum Weight Difference ($\Delta W_{\text{max}}$)** | **$0.071$** | $< 0.10$ | **Passed (High Parity)** |
| **Partition of Unity ($\sum_k W_{ik}$)** | **$1.0 \pm 10^{-6}$** | $1.0 \pm 10^{-5}$ | **Exact** |
| **Monotonic Influence Gradient** | **$100\%$** | $100\%$ | **Consistent Bone Gradient** |

---

### 3.4 Deformation Engine Comparison: LBS vs. DQS (QBS)

| Feature / Metric | Linear Blend Skinning (LBS) | Dual Quaternion Skinning (DQS / QBS) |
|---|---|---|
| **Mathematical Basis** | Linear convex combination of affine matrices | Dual Quaternion Linear Blending (DLB) on $SE(3)$ |
| **Volume Preservation** | Poor (collapses joint cross-section at acute bends) | **Strictly Preserved (Zero Volume Loss)** |
| **Twisting Artifact Resistance** | Severe "Candy-wrapper" pinching | **Completely Eliminated** |
| **CPU Deformation Throughput** | $\sim 800\text{ FPS}$ | $\sim 470\text{ FPS}$ |
| **glTF 2.0 Standard Direct Export** | Native (`JOINTS_0`, `WEIGHTS_0`) | Exported via LBS format or baked vertex morphs |
| **Biomechanical Locomotion Suitability** | Acceptable for rigid fins | **Optimal for flexible undulating spines** |

---

## 4. Technical Stack & Quality Assurance

* **Language & Core Framework:** Python 3.11+
* **Numerical & Differential Geometry:** `numpy`, `scipy` (Sparse Cholesky factorizations), `torch` (PyTorch CUDA tensors), `trimesh`, `open3d`, `pymeshlab`, `igraph`, `networkx`, `scikit-learn`
* **Vision Foundation Models:** `transformers` (Meta SAM3), Hugging Face Hub
* **Rendering & Export:** `pyrender` (EGL/OSMesa OpenGL), `pygltflib` (glTF 2.0 / GLB), `bpy` (Blender verification oracle), `pillow`, `opencv-python`, `matplotlib`
* **Quality Assurance:** `pytest`, `pytest-cov`, `pyright`, `ruff` (Full test suite: 72 passed tests across 12 modules)
