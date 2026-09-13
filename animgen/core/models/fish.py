from pathlib import Path
from typing import Any, Optional
from collections import defaultdict
import copy
import networkx as nx
import numpy as np
import torch
import trimesh

from animgen.core.models.pipeline import Pipeline
from animgen.core.models.model import BaseModelClass
from animgen.core.spline import Spline
from animgen.core.armature import Armature, Bone
from animgen.animation.animator import Animator, AnimationClip
from animgen.animation.kinematics import successive_rotations
from animgen.animation.straight import straighten_lateral
from animgen.rigging.mesh_contraction import extract_skeleton
from animgen.rigging.shape_diameter_function import shape_diameter_function
from animgen.utils.mesh import triangle_areas


def _fill_face_gaps(
    mesh: trimesh.Trimesh,
    face_labels: np.ndarray,
    adj_dict: dict[int, list[int]],
    max_iters: int = 2,
) -> np.ndarray:
    """
    Fills isolated face gaps and smooths part boundaries using majority voting across neighbors.
    """
    labels = face_labels.copy()
    for _ in range(max_iters):
        changed = 0
        for f in range(len(mesh.faces)):
            curr_lbl = labels[f]
            neighbors = adj_dict.get(f, [])
            if not neighbors:
                continue
            nb_labels = [labels[nb] for nb in neighbors]
            majority_lbl = max(set(nb_labels), key=nb_labels.count)
            if (
                nb_labels.count(majority_lbl) >= len(neighbors) * 0.7
                and majority_lbl != curr_lbl
            ):
                labels[f] = majority_lbl
                changed += 1
        if changed == 0:
            break
    return labels


def _expand_fin_boundaries(
    mesh: trimesh.Trimesh,
    face_labels: np.ndarray,
    adj_dict: dict[int, list[int]],
    rounds: int = 1,
) -> np.ndarray:
    """
    Expands non-body appendage labels (fins) outward across face adjacency.
    Body faces bordering a fin are absorbed into that fin if >= half of their neighbors share that fin label.
    """
    labels = face_labels.copy()
    for _ in range(rounds):
        new_labels = labels.copy()
        body_faces = np.where(labels == 0)[0]
        for f in body_faces:
            fin_neighbors = [
                labels[nb] for nb in adj_dict.get(f, []) if labels[nb] != 0
            ]
            if not fin_neighbors:
                continue
            candidate_label = max(set(fin_neighbors), key=fin_neighbors.count)
            if (
                fin_neighbors.count(candidate_label) >= 2
                or fin_neighbors.count(candidate_label)
                >= len(adj_dict.get(f, [])) * 0.5
            ):
                new_labels[f] = candidate_label
        labels = new_labels
    return labels


def _remove_orphan_islands(
    mesh: trimesh.Trimesh,
    face_labels: np.ndarray,
    adj_dict: dict[int, list[int]],
    min_area_ratio: float = 0.08,
) -> np.ndarray:
    """
    For every fin appendage (label != 0), finds all connected components on the face adjacency graph.
    Retains ONLY the largest/dominant connected component per fin.
    All smaller disconnected orphan islands are merged directly back into Main Body (label = 0).
    """
    cleaned_labels = face_labels.copy()
    face_areas = triangle_areas(mesh.vertices, mesh.faces)

    unique_labels = np.unique(face_labels)
    for lbl in unique_labels:
        if lbl == 0:
            continue

        lbl_faces = np.where(cleaned_labels == lbl)[0]
        if len(lbl_faces) == 0:
            continue

        visited = set()
        components = []
        lbl_set = set(lbl_faces)

        for f in lbl_faces:
            if f in visited:
                continue
            comp = []
            queue = [f]
            visited.add(f)
            while queue:
                curr = queue.pop()
                comp.append(curr)
                for nb in adj_dict.get(curr, []):
                    if nb in lbl_set and nb not in visited:
                        visited.add(nb)
                        queue.append(nb)
            components.append(np.array(comp, dtype=np.int32))

        components.sort(key=lambda c: float(np.sum(face_areas[c])), reverse=True)
        primary_area = float(np.sum(face_areas[components[0]]))

        for comp in components[1:]:
            comp_area = float(np.sum(face_areas[comp]))
            if comp_area < primary_area * min_area_ratio:
                cleaned_labels[comp] = 0

    return cleaned_labels


DEFAULT_FISH_PROMPTS: list[str] = [
    "tail",
    "top fin",
    "side fin",
]

DEFAULT_FISH_PARAMS: dict[str, Any] = {
    "num_bones": 12,
    "num_tail_bones": 2,
    "rig_tail": True,
    "rig_pectoral_fins": True,
    "rig_dorsal_fin": False,
    "frame_rate": 30.0,
    "sdf_threshold": 0.30,
    "use_sam": True,
    "animations": {
        "swim": {
            "wave_amplitude": 0.22,
            "wave_duration": 1.4,
            "num_waves": 0.85,
            "head_amplitude_ratio": 0.08,
            "growth_factor": 0.18,
            "pectoral_mode": "active",
            "pectoral_flap_deg": 14.0,
            "pectoral_pitch_deg": 6.0,
            "dorsal_flex_deg": 4.0,
            "is_loopable": True,
        },
        "idle": {
            "wave_amplitude": 0.06,
            "wave_duration": 3.5,
            "num_waves": 1.0,
            "head_amplitude_ratio": 0.04,
            "growth_factor": 0.12,
            "pectoral_mode": "active",
            "pectoral_flap_deg": 4.0,
            "pectoral_pitch_deg": 2.0,
            "dorsal_flex_deg": 1.5,
            "is_loopable": True,
        },
        "sprint": {
            "wave_amplitude": 0.30,
            "wave_duration": 0.85,
            "num_waves": 0.85,
            "head_amplitude_ratio": 0.14,
            "growth_factor": 0.22,
            "pectoral_mode": "closed",
            "pectoral_close_deg": 35.0,
            "dorsal_flex_deg": 5.0,
            "is_loopable": True,
        },
    },
}

PART_COLOR_PALETTE: dict[str, list[int]] = {
    "body": [75, 85, 95, 255],
    "tail": [255, 60, 60, 255],
    "dorsal_fin": [0, 200, 255, 255],
    "left_pectoral_fin": [0, 230, 118, 255],
    "right_pectoral_fin": [255, 180, 0, 255],
}


class FishModels(Pipeline):
    """
    End-to-end procedural animation pipeline for fish and aquatic organisms.

    Performs:
    1. 3D Shape Diameter Function (SDF) and mesh graph anatomical fin and body segmentation.
    2. Automatic tail orientation detection (vertical for fish/sharks vs. horizontal for cetaceans).
    3. Lateral Bishop-frame straightening: Flattens curvature in the lateral swimming plane
       (Z-axis for vertical-tail fish, Y-axis for horizontal-tail cetaceans), preserving the
       natural tail blade profile and longitudinal length.
    4. Hierarchical armature construction with dedicated tail bones, side pectoral fin bones,
       and optional dorsal fin bones.
    5. Direction-aligned procedural wave animations steered into the detected swimming plane.
    6. Comprehensive intermediate artifact export for diagnostics and inspection.
    """

    def __init__(
        self,
        model: BaseModelClass,
        prompts: list[str] | None = None,
        prompts_embedding_path: Path | None = None,
        num_bones: int = DEFAULT_FISH_PARAMS["num_bones"],
        num_tail_bones: int = DEFAULT_FISH_PARAMS["num_tail_bones"],
        rig_tail: bool = DEFAULT_FISH_PARAMS["rig_tail"],
        rig_pectoral_fins: bool = DEFAULT_FISH_PARAMS["rig_pectoral_fins"],
        rig_dorsal_fin: bool = DEFAULT_FISH_PARAMS["rig_dorsal_fin"],
        frame_rate: float = DEFAULT_FISH_PARAMS["frame_rate"],
        sdf_threshold: float = DEFAULT_FISH_PARAMS["sdf_threshold"],
        straighten_mesh: bool = True,
        use_sam: bool = DEFAULT_FISH_PARAMS["use_sam"],
        animations: dict[str, dict[str, Any]] | None = None,
        **kwargs: Any,
    ):
        """
        Initializes the FishModels pipeline.

        Parameters
        ----------
        model : BaseModelClass
            The input 3D model container.
        prompts : list[str] | None, optional
            List of prompt strings. Defaults to DEFAULT_FISH_PROMPTS if None.
        prompts_embedding_path : Path | None, optional
            Path to precomputed text embeddings (.pt).
        num_bones : int, default=12
            Number of longitudinal spine/body bones.
        num_tail_bones : int, default=2
            Number of dedicated tail bones along the caudal fin.
        rig_tail : bool, default=True
            Whether to rig the tail/caudal fin.
        rig_pectoral_fins : bool, default=True
            Whether to rig the left and right pectoral side fins.
        rig_dorsal_fin : bool, default=False
            Whether to rig the top dorsal fin (excluded by default).
        frame_rate : float, default=30.0
            Frame rate (FPS) for keyframe animation sampling.
        sdf_threshold : float, default=0.30
            Normalized Shape Diameter Function threshold for thin appendage extraction.
        straighten_mesh : bool, default=True
            Whether to apply lateral Bishop-frame straightening to the mesh.
        use_sam : bool, default=True
            Whether to run SAM3 vision model multi-view zero-shot segmentation.
        animations : dict[str, dict[str, Any]] | None, optional
            Dictionary mapping animation clip names to wave parameters.
        """
        if prompts is None and prompts_embedding_path is None:
            prompts = list(DEFAULT_FISH_PROMPTS)

        super().__init__(
            model,
            prompts=prompts,
            prompts_embedding_path=prompts_embedding_path,
        )

        self.num_bones: int = num_bones
        self.num_tail_bones: int = num_tail_bones
        self.rig_tail: bool = rig_tail
        self.rig_pectoral_fins: bool = rig_pectoral_fins
        self.rig_dorsal_fin: bool = rig_dorsal_fin
        self.frame_rate: float = frame_rate
        self.sdf_threshold: float = sdf_threshold
        self.straighten_mesh: bool = straighten_mesh
        self.use_sam: bool = use_sam

        self.animations: dict[str, dict[str, Any]] = copy.deepcopy(
            DEFAULT_FISH_PARAMS["animations"]
        )
        if animations is not None:
            for clip_name, clip_cfg in animations.items():
                if clip_name in self.animations:
                    self.animations[clip_name].update(clip_cfg)
                else:
                    self.animations[clip_name] = clip_cfg

        self.initial_mesh: trimesh.Trimesh = self.model.mesh.copy()

        self.segments: dict[str, list[int]] = {}
        self.face_prompt_detected: Optional[dict[str, np.ndarray]] = None
        self.raw_spine_points: Optional[np.ndarray] = None
        self.source_spine: Optional[np.ndarray] = None
        self.target_spine: Optional[np.ndarray] = None
        self.spline: Optional[Spline] = None
        self.armature: Optional[Armature] = None
        self.tail_orientation: Optional[str] = None

    def segment(self) -> dict[str, list[int]]:
        """
        Runs the anatomical appendage segmentation pipeline on the fish model.
        Uses the robust hybrid approach (3D Shape Diameter Function + SAM3 multi-view vision)
        combined with standardized morphological classification and bilateral pairing.

        Returns
        -------
        dict[str, list[int]]
            Dictionary mapping anatomical part names to lists of face indices:
            - 'body'
            - 'tail'
            - 'dorsal_fin'
            - 'left_pectoral_fin'
            - 'right_pectoral_fin'
        """
        mesh = self.model.mesh
        num_faces = len(mesh.faces)
        face_areas = triangle_areas(mesh.vertices, mesh.faces)
        total_mesh_area = float(np.sum(face_areas))

        # Multi-view SAM3 Vision Inference on high-res / subdivided geometry
        if self.use_sam and self.face_prompt_detected is None:
            try:
                from animgen.rigging.SAM3 import SAM3Segmentation
                from animgen.rigging.backproject import backproject_masks_to_faces
                from scipy.spatial import cKDTree

                sam_prompts = (
                    self.prompts if self.prompts else list(DEFAULT_FISH_PROMPTS)
                )

                # Create an 8x-16x subdivided proxy mesh for ultra-fine SAM rasterization
                # so SAM camera views rasterize micro-triangles at fin attachment creases
                # rather than coarse polygons spanning into the torso.
                sub_mesh = mesh.copy()
                try:
                    if num_faces < 15000:
                        sub_mesh = sub_mesh.subdivide().subdivide()
                    elif num_faces < 50000:
                        sub_mesh = sub_mesh.subdivide()
                except Exception:
                    sub_mesh = mesh.copy()

                sub_model = self.model if sub_mesh is mesh else BaseModelClass(sub_mesh)
                with SAM3Segmentation(prompts=sam_prompts) as sam3:
                    masks_dict = sam3(sub_model, threshold=0.5, mask_threshold=0.5)
                    sub_face_prompts = backproject_masks_to_faces(
                        masks_dict,
                        sub_model.views_output["faces"],
                        len(sub_mesh.faces),
                    )

                if sub_mesh is not mesh:
                    # Transfer fine sub-mesh face votes back to original mesh faces via KD-Tree
                    sub_centroids = sub_mesh.triangles.mean(axis=1)
                    tree = cKDTree(sub_centroids)
                    orig_centroids = mesh.triangles.mean(axis=1)
                    _, nearest_idxs = tree.query(orig_centroids, k=1)
                    self.face_prompt_detected = {
                        p: sub_face_prompts[p][nearest_idxs] for p in sub_face_prompts
                    }
                else:
                    self.face_prompt_detected = sub_face_prompts

            except Exception as e:
                print(
                    f"[FishModels.segment] SAM3 inference unavailable or failed ({e}), falling back to pure 3D SDF."
                )
                self.face_prompt_detected = None
        elif not self.use_sam and self.face_prompt_detected is None:
            self.face_prompt_detected = None

        # 3D Shape Diameter Function (SDF)
        try:
            sdf = shape_diameter_function(mesh, norm=True)
        except Exception:
            sdf = np.ones(num_faces, dtype=np.float32)

        is_thin = sdf < self.sdf_threshold

        # SAM3 multi-view candidate filtering
        if self.face_prompt_detected is not None:
            total_sam_votes = np.sum(list(self.face_prompt_detected.values()), axis=0)
            is_sam_candidate = total_sam_votes >= 2
        else:
            is_sam_candidate = np.zeros(num_faces, dtype=bool)

        # Geometric Snout Exclusion
        x_min, x_max = float(mesh.bounds[0, 0]), float(mesh.bounds[1, 0])
        y_min, y_max = float(mesh.bounds[0, 1]), float(mesh.bounds[1, 1])
        z_min, z_max = float(mesh.bounds[0, 2]), float(mesh.bounds[1, 2])
        y_mid = 0.5 * (y_min + y_max)
        z_mid = 0.5 * (z_min + z_max)
        x_span = max(x_max - x_min, 1e-6)
        y_span = max(y_max - y_min, 1e-6)
        z_span = max(z_max - z_min, 1e-6)

        face_centroids = mesh.triangles.mean(axis=1)
        is_snout = (
            (face_centroids[:, 0] < x_min + 0.12 * x_span)
            & (np.abs(face_centroids[:, 2] - z_mid) < 0.12 * z_span)
            & (np.abs(face_centroids[:, 1] - y_mid) < 0.12 * y_span)
        )

        hybrid_candidate = (is_thin | is_sam_candidate) & (~is_snout)

        # Adjacency Graph Setup (welded copy to bridge UV seams)
        welded_mesh = mesh.copy()
        try:
            welded_mesh.merge_vertices(merge_tex=True, merge_norm=True)
            adj_source = welded_mesh.face_adjacency
        except Exception:
            adj_source = mesh.face_adjacency

        adj_dict: dict[int, list[int]] = defaultdict(list)
        for f1, f2 in adj_source:
            adj_dict[f1].append(f2)
            adj_dict[f2].append(f1)

        visited = np.zeros(num_faces, dtype=bool)
        raw_clusters: list[np.ndarray] = []
        for f in np.where(hybrid_candidate)[0]:
            if visited[f]:
                continue
            comp = []
            queue = [f]
            visited[f] = True
            while queue:
                curr = queue.pop()
                comp.append(curr)
                for nb in adj_dict.get(curr, []):
                    if hybrid_candidate[nb] and not visited[nb]:
                        visited[nb] = True
                        queue.append(nb)
            if len(comp) > 0:
                raw_clusters.append(np.array(comp, dtype=np.int32))

        raw_clusters.sort(key=lambda c: float(np.sum(face_areas[c])), reverse=True)

        # Standardized Morphological Classification & Bilateral Pairing
        classified_appendages = self._classify_appendages(
            mesh=mesh,
            raw_clusters=raw_clusters,
            total_mesh_area=total_mesh_area,
        )

        # Part Assembly & Morphological Refinement
        face_label_array = np.zeros(num_faces, dtype=np.int32)
        label_to_id: dict[str, int] = {"body": 0}
        id_to_label: dict[int, str] = {0: "body"}

        part_id = 1
        for label, comp_list in classified_appendages.items():
            label_to_id[label] = part_id
            id_to_label[part_id] = label
            for comp in comp_list:
                face_label_array[comp] = part_id
            part_id += 1

        # 1. Fill isolated face gaps with majority voting
        refined_face_labels = _fill_face_gaps(
            mesh, face_label_array, adj_dict, max_iters=2
        )
        # 2. Island removal FIRST (kills any detached false-positive noise before dilation)
        cleaned_face_labels = _remove_orphan_islands(
            mesh, refined_face_labels, adj_dict, min_area_ratio=0.08
        )
        # 3. Fin boundary expansion LAST (cleanly seals the root attachment crease)
        final_face_labels = _expand_fin_boundaries(
            mesh, cleaned_face_labels, adj_dict, rounds=1
        )

        segments: dict[str, list[int]] = {}
        for pid, label in id_to_label.items():
            if pid == 0:
                continue
            segments[label] = np.where(final_face_labels == pid)[0].tolist()

        all_fin_faces = set()
        for k, v in segments.items():
            all_fin_faces.update(v)
        segments["body"] = [f for f in range(num_faces) if f not in all_fin_faces]

        self.segments = segments
        self.tail_orientation = self._detect_tail_orientation()

        return segments

    def _classify_appendages(
        self,
        mesh: trimesh.Trimesh,
        raw_clusters: list[np.ndarray],
        total_mesh_area: float,
    ) -> dict[str, list[np.ndarray]]:
        """
        Classifies geometric candidate clusters into tail, dorsal_fin, and left_pectoral_fin / right_pectoral_fin.
        """
        face_areas = triangle_areas(mesh.vertices, mesh.faces)
        major_area_threshold = 0.0020 * total_mesh_area

        cluster_meta = []
        for cluster in raw_clusters:
            area = float(np.sum(face_areas[cluster]))
            if area < major_area_threshold:
                continue

            c_verts = mesh.vertices[np.unique(mesh.faces[cluster])]
            centroid = c_verts.mean(axis=0)
            max_abs_z = float(np.max(np.abs(c_verts[:, 2])))

            cluster_meta.append(
                {
                    "faces": cluster,
                    "area": area,
                    "centroid": centroid,
                    "max_abs_z": max_abs_z,
                    "label": None,
                }
            )

        # 1. Pass 1: Median & Terminal Fins (Tail & Dorsal) + Keel Anomaly Filter
        for c in cluster_meta:
            cx, cy, cz = c["centroid"]

            # Keel / Peduncle ridge anomaly filter -> retain in body
            if 0.45 <= cx < 0.70 and abs(cz) < 0.06 and cy < -0.12:
                c["label"] = "body"
                continue

            # Tail Fin: extreme posterior clusters
            if cx >= 0.65 or (cx > 0.58 and abs(cz) < 0.08 and abs(cy) < 0.20):
                c["label"] = "tail"
            # Dorsal Fin: top midline ridge
            elif cy > 0.08 and abs(cz) < 0.15:
                c["label"] = "dorsal_fin"

        # 2. Pass 2: Bilateral Paired Pectoral Fins (Side Fins)
        unlabeled = [c for c in cluster_meta if c["label"] is None]

        left_candidates = [c for c in unlabeled if c["centroid"][2] > 0.02]
        right_candidates = [c for c in unlabeled if c["centroid"][2] < -0.02]

        left_candidates.sort(
            key=lambda c: (c["max_abs_z"], c["centroid"][1]), reverse=True
        )
        right_candidates.sort(
            key=lambda c: (c["max_abs_z"], c["centroid"][1]), reverse=True
        )

        for i, c in enumerate(left_candidates):
            if i == 0 and (c["max_abs_z"] > 0.12 or c["centroid"][1] >= -0.15):
                c["label"] = "left_pectoral_fin"
            else:
                c["label"] = "body"

        for i, c in enumerate(right_candidates):
            if i == 0 and (c["max_abs_z"] > 0.12 or c["centroid"][1] >= -0.15):
                c["label"] = "right_pectoral_fin"
            else:
                c["label"] = "body"

        classified: dict[str, list[np.ndarray]] = defaultdict(list)
        for c in cluster_meta:
            if c["label"] and c["label"] != "body":
                classified[c["label"]].append(c["faces"])

        return classified

    def _get_fin_verts(self, fin: str | list[int] | np.ndarray) -> np.ndarray:
        """Helper to extract 3D vertex coordinates from a fin name, face indices, or vertex array."""
        mesh = self.model.mesh
        if mesh is None:
            raise ValueError("Model mesh is not loaded.")

        if isinstance(fin, str):
            faces = self.segments.get(fin, [])
            if not faces:
                raise ValueError(f"Fin segment '{fin}' not found in model segments.")
            return mesh.vertices[np.unique(mesh.faces[faces])]
        elif isinstance(fin, np.ndarray) and fin.ndim == 2 and fin.shape[1] == 3:
            return fin
        else:
            arr = np.asarray(fin, dtype=int)
            if len(arr) == 0:
                raise ValueError("Empty fin indices provided.")
            if np.max(arr) < len(mesh.faces):
                return mesh.vertices[np.unique(mesh.faces[arr])]
            else:
                return mesh.vertices[arr]

    def _get_body_center(self) -> np.ndarray:
        """Helper to compute the 3D centroid of the fish trunk."""
        mesh = self.model.mesh
        if mesh is None:
            return np.zeros(3)
        body_faces = self.segments.get("body", [])
        if len(body_faces) > 0:
            return mesh.vertices[np.unique(mesh.faces[body_faces])].mean(axis=0)
        return mesh.vertices.mean(axis=0)

    def _get_fin_endpoints(
        self, fin: str | list[int] | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes (root_point, tip_point) 3D coordinates for a fin.
        Root is closest to the trunk/centerline; Tip is furthest from the root.
        """
        fin_verts = self._get_fin_verts(fin)
        base_idx = int(np.argmin(np.abs(fin_verts[:, 2])))
        base_pt = fin_verts[base_idx]

        dists_from_base = np.linalg.norm(fin_verts - base_pt, axis=1)
        k_sample = max(1, len(fin_verts) // 10)
        root_order = np.argsort(dists_from_base)
        root_pt = fin_verts[root_order[:k_sample]].mean(axis=0)

        dists_from_root = np.linalg.norm(fin_verts - root_pt, axis=1)
        tip_order = np.argsort(dists_from_root)
        tip_pt = fin_verts[tip_order[-k_sample:]].mean(axis=0)

        return root_pt, tip_pt

    def _classify_side_fin_type(
        self,
        fin: str | list[int] | np.ndarray,
    ) -> str:
        """
        Determines whether a pectoral/side fin is:
        - 'loosy': flexible, lies along the body flank extending longitudinally (+X).
        - 'straight': rigid/hydrofoil, extends outward away from the body (+/-Z).

        Parameters
        ----------
        fin : str | list[int] | np.ndarray
            Segment key ('left_pectoral_fin', 'right_pectoral_fin'), face indices, or vertex coordinates.

        Returns
        -------
        str
            'loosy' or 'straight'
        """
        fin_verts = self._get_fin_verts(fin)
        span_x = float(fin_verts[:, 0].max() - fin_verts[:, 0].min())
        span_z = float(fin_verts[:, 2].max() - fin_verts[:, 2].min())
        return "loosy" if (span_x / max(span_z, 1e-6)) >= 1.75 else "straight"

    def _classify_fin_type(
        self,
        fin: str | list[int] | np.ndarray,
    ) -> str:
        """
        Classifies a fin by its primary extension direction away from the main body:
        - 'caudal': posterior extension along +X (tail fin)
        - 'dorsal': vertical upward extension along +Y (top fin)
        - 'ventral': vertical downward extension along -Y (anal fin)
        - 'pectoral': lateral extension along +/-Z (side fins)

        Parameters
        ----------
        fin : str | list[int] | np.ndarray
            Segment key, face indices, or vertex array.

        Returns
        -------
        str
            'caudal', 'dorsal', 'ventral', 'pectoral', or 'other'
        """
        mesh = self.model.mesh
        fin_verts = self._get_fin_verts(fin)
        center = fin_verts.mean(axis=0)
        body_center = self._get_body_center()
        ext_vec = center - body_center

        # Lateral displacement away from midline -> pectoral
        if abs(ext_vec[2]) > 0.05 or abs(center[2]) > 0.05:
            return "pectoral"

        # Check if rearmost posterior component -> caudal
        if mesh is not None:
            x_max_mesh = mesh.vertices[:, 0].max()
            if center[0] > (body_center[0] + (x_max_mesh - body_center[0]) * 0.5):
                return "caudal"

        if ext_vec[1] > 0:
            return "dorsal"
        elif ext_vec[1] < 0:
            return "ventral"

        return "other"

    def _detect_tail_orientation(self) -> str:
        """
        Analyzes caudal/tail fin geometry to detect swimming plane:
        - 'vertical' (Fish, Sharks, Tuna, Mackerel): Fin spans along dorsoventral Y-axis.
        - 'horizontal' (Dolphins, Whales, Porpoises): Fluke spans laterally along mediolateral Z-axis.

        Returns
        -------
        str
            'vertical' or 'horizontal'
        """
        tail_faces = self.segments.get("tail", [])
        if not tail_faces:
            return "vertical"

        t_verts = self.model.mesh.vertices[np.unique(self.model.mesh.faces[tail_faces])]
        span_y = float(t_verts[:, 1].max() - t_verts[:, 1].min())
        span_z = float(t_verts[:, 2].max() - t_verts[:, 2].min())
        return "vertical" if (span_y / max(span_z, 1e-6)) >= 1.0 else "horizontal"

    def align(self) -> None:
        """
        Determines head vs tail orientation using segmented Tail Fin and Top Fin relative positions,
        and aligns the mesh so that the snout/head is at negative X (-X) and the caudal tail is at positive X (+X).
        """
        mesh = self.model.mesh
        if mesh is None:
            return

        verts = mesh.vertices
        x_min, x_max = float(mesh.bounds[0, 0]), float(mesh.bounds[1, 0])
        x_mid = 0.5 * (x_min + x_max)

        # Use segmentation results: compare Tail Fin and Top Fin (Dorsal Fin) positions
        tail_faces = self.segments.get("tail", [])
        top_faces = self.segments.get("dorsal_fin", [])

        is_backwards = False
        if tail_faces:
            t_verts = mesh.vertices[np.unique(mesh.faces[tail_faces])]
            tail_x = float(t_verts[:, 0].mean())

            if top_faces:
                top_verts = mesh.vertices[np.unique(mesh.faces[top_faces])]
                top_x = float(top_verts[:, 0].mean())
                # Tail must be posterior to Top Fin (tail_x > top_x in canonical orientation)
                is_backwards = (tail_x < top_x) or (tail_x < x_mid)
            else:
                # If top fin is not present, tail must be at positive end relative to mid-body
                is_backwards = tail_x < x_mid
        else:
            # Fallback to geometric cross-sectional anisotropy / thickness metric
            x_span = x_max - x_min
            mask_neg = verts[:, 0] <= x_min + 0.15 * x_span
            mask_pos = verts[:, 0] >= x_max - 0.15 * x_span
            v_neg = verts[mask_neg]
            v_pos = verts[mask_pos]

            z_neg = (
                float(v_neg[:, 2].max() - v_neg[:, 2].min()) if len(v_neg) > 0 else 1e-4
            )
            z_pos = (
                float(v_pos[:, 2].max() - v_pos[:, 2].min()) if len(v_pos) > 0 else 1e-4
            )
            y_neg = (
                float(v_neg[:, 1].max() - v_neg[:, 1].min()) if len(v_neg) > 0 else 1e-4
            )
            y_pos = (
                float(v_pos[:, 1].max() - v_pos[:, 1].min()) if len(v_pos) > 0 else 1e-4
            )

            aniso_neg = max(y_neg / max(z_neg, 1e-4), z_neg / max(y_neg, 1e-4))
            aniso_pos = max(y_pos / max(z_pos, 1e-4), z_pos / max(y_pos, 1e-4))
            min_dim_neg = min(y_neg, z_neg)
            min_dim_pos = min(y_pos, z_pos)

            is_backwards = (aniso_neg > 2.0 and aniso_neg > aniso_pos) or (
                min_dim_pos > 1.5 * min_dim_neg
            )

        if is_backwards:
            rot_180 = trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0])
            mesh.apply_transform(rot_180)

            # Swap left and right pectoral fins if they were already segmented
            left_f = self.segments.get("left_pectoral_fin", [])
            right_f = self.segments.get("right_pectoral_fin", [])
            if left_f or right_f:
                self.segments["left_pectoral_fin"] = right_f
                self.segments["right_pectoral_fin"] = left_f

            # Re-detect tail orientation in aligned coordinate space
            self.tail_orientation = self._detect_tail_orientation()

    def canonicalize(self, segments: dict[str, list[int]]) -> trimesh.Trimesh:
        """
        Performs Bishop-frame lateral straightening along the axis orthogonal to the tail:
        - If tail is vertical (Fish/Sharks): Straightens lateral Z curvature while leaving
          the natural vertical Y tail profile and X longitudinal length completely intact.
        - If tail is horizontal (Dolphins/Whales): Straightens dorsoventral Y curvature while
          leaving the horizontal Z tail fluke and X longitudinal length intact.

        Parameters
        ----------
        segments : dict[str, list[int]]
            Part face segmentation dictionary.

        Returns
        -------
        trimesh.Trimesh
            The laterally straightened canonical mesh.
        """
        mesh = self.model.mesh

        if self.tail_orientation is None:
            self.tail_orientation = self._detect_tail_orientation()

        straighten_axis = "z" if self.tail_orientation == "vertical" else "y"

        # Extract 1D spine from body (excluding tail fin lobes to prevent distortion)
        body_faces = segments.get("body", [])
        tail_faces = segments.get("tail", [])

        if body_faces and len(body_faces) > 50:
            body_verts = mesh.vertices[np.unique(mesh.faces[body_faces])]
            x_min = float(mesh.bounds[0, 0])
            x_max = float(mesh.bounds[1, 0])

            if tail_faces:
                tail_verts = mesh.vertices[np.unique(mesh.faces[tail_faces])]
                tail_min_x = float(tail_verts[:, 0].min())
                x_tail_start = max(tail_min_x, x_min + 0.5 * (x_max - x_min))
            else:
                x_tail_start = x_min + 0.8 * (x_max - x_min)

            num_slices = 15
            x_slices = np.linspace(x_min, x_tail_start, num_slices)
            slice_centers: list[np.ndarray] = []

            for i in range(len(x_slices) - 1):
                x_low, x_high = x_slices[i], x_slices[i + 1]
                mask = (body_verts[:, 0] >= x_low) & (body_verts[:, 0] <= x_high)
                if np.sum(mask) > 0:
                    center = body_verts[mask].mean(axis=0)
                    center[0] = 0.5 * (x_low + x_high)
                    slice_centers.append(center)

            if len(slice_centers) >= 2:
                snout_pt = np.array(
                    [x_min, slice_centers[0][1], slice_centers[0][2]],
                    dtype=np.float32,
                )
                peduncle_pt = np.array(
                    [x_tail_start, slice_centers[-1][1], slice_centers[-1][2]],
                    dtype=np.float32,
                )
                spine_pts = np.vstack([[snout_pt], slice_centers, [peduncle_pt]])
            else:
                spine_pts = np.array(
                    [[x_min, 0.0, 0.0], [x_tail_start, 0.0, 0.0]], dtype=np.float32
                )
        else:
            # Fallback: Extract skeleton from mesh
            skel_v, skel_e = extract_skeleton(
                mesh,
                max_iters=20,
                threshold=0.5,
                no_1d_collapses=True,
                return_tuple=True,
            )

            G = nx.Graph()
            for u, v in skel_e:
                d = float(np.linalg.norm(skel_v[u] - skel_v[v]))
                G.add_edge(u, v, weight=d)

            min_x_node = int(np.argmin(skel_v[:, 0]))
            max_x_node = int(np.argmax(skel_v[:, 0]))

            if nx.has_path(G, min_x_node, max_x_node):
                spine_path = nx.shortest_path(
                    G, source=min_x_node, target=max_x_node, weight="weight"
                )
            else:
                comp_nodes = list(nx.node_connected_component(G, min_x_node))
                sub_g = G.subgraph(comp_nodes)
                sub_max_node = max(comp_nodes, key=lambda n: skel_v[n, 0])
                spine_path = nx.shortest_path(
                    sub_g, source=min_x_node, target=sub_max_node, weight="weight"
                )

            spine_pts = skel_v[spine_path]
            x_indices = np.argsort(spine_pts[:, 0])
            spine_pts = spine_pts[x_indices]

        dists = np.linalg.norm(np.diff(spine_pts, axis=0), axis=1)
        valid_mask = np.concatenate(([True], dists > 1e-4))
        spine_pts = spine_pts[valid_mask]

        if len(spine_pts) < 4:
            if len(spine_pts) >= 2:
                t_interp = np.linspace(0.0, 1.0, 5)
                spine_pts = (1.0 - t_interp[:, None]) * spine_pts[0] + t_interp[
                    :, None
                ] * spine_pts[-1]
            else:
                x_min, x_max = float(mesh.bounds[0, 0]), float(mesh.bounds[1, 0])
                spine_pts = np.linspace([x_min, 0.0, 0.0], [x_max * 0.8, 0.0, 0.0], 5)

        self.raw_spine_points = spine_pts
        pts_t = [torch.tensor(v, dtype=torch.float32) for v in spine_pts]

        # Build Catmull-Rom Spline strictly covering the body
        self.spline = Spline(pts_t, alpha=0.5, phantom_num_points=1)

        # Apply Bishop-Frame Lateral Straightening along orthogonal axis
        if self.straighten_mesh:
            straight_mesh = straighten_lateral(
                mesh,
                spine_points=self.spline,
                straighten_axis=straighten_axis,
            )
        else:
            straight_mesh = mesh.copy()

        # Construct Target Straightened Spine along the canonical mesh coordinates
        eval_pts = self.spline.evaluate_curve(num_points_per_segment=10)
        self.source_spine = np.array([pt.detach().cpu().numpy() for pt in eval_pts])

        target_spine = self.source_spine.copy()
        if self.straighten_mesh:
            if straighten_axis == "z":
                # Flatten Z to centerline
                target_spine[:, 2] = float(np.mean(target_spine[:, 2]))
            elif straighten_axis == "y":
                # Flatten Y to centerline
                target_spine[:, 1] = float(np.mean(target_spine[:, 1]))
            elif straighten_axis == "x":
                target_spine[:, 0] = float(np.mean(target_spine[:, 0]))

        # Straight line from dorsal fin level to caudal peduncle along target spine
        dorsal_faces = segments.get("dorsal_fin", [])
        tail_faces = segments.get("tail", [])
        if dorsal_faces:
            d_verts = straight_mesh.vertices[
                np.unique(straight_mesh.faces[dorsal_faces])
            ]
            x_dorsal = float(d_verts[:, 0].mean())
        else:
            x_dorsal = float(
                target_spine[0, 0] + 0.4 * (target_spine[-1, 0] - target_spine[0, 0])
            )

        if tail_faces:
            t_verts = straight_mesh.vertices[np.unique(straight_mesh.faces[tail_faces])]
            tail_mid_y = float(0.5 * (t_verts[:, 1].min() + t_verts[:, 1].max()))
            tail_mid_z = float(0.5 * (t_verts[:, 2].min() + t_verts[:, 2].max()))
        else:
            tail_mid_y = float(target_spine[-1, 1])
            tail_mid_z = float(target_spine[-1, 2])

        # Straight line at tail level (tail_mid_y, tail_mid_z) from the tail forward to dorsal fin x-level
        for i in range(len(target_spine)):
            if target_spine[i, 0] >= x_dorsal:
                target_spine[i, 1] = tail_mid_y
                target_spine[i, 2] = tail_mid_z
            else:
                t_blend = (target_spine[i, 0] - target_spine[0, 0]) / max(
                    x_dorsal - target_spine[0, 0], 1e-6
                )
                t_blend = float(np.clip(t_blend, 0.0, 1.0))
                target_spine[i, 1] = (1.0 - t_blend) * target_spine[
                    0, 1
                ] + t_blend * tail_mid_y
                target_spine[i, 2] = (1.0 - t_blend) * target_spine[
                    0, 2
                ] + t_blend * tail_mid_z

        self.target_spine = target_spine
        return straight_mesh

    def rig(self, segments: dict[str, list[int]]) -> Armature:
        """
        Constructs a hierarchical connected bone Armature along the centerline spine
        from anterior snout to caudal tail, including dedicated tail fin bones and
        side pectoral fin bones.
        """
        if self.target_spine is None:
            raise ValueError(
                "Target spine has not been computed. Run canonicalize() first."
            )

        mesh = self.model.mesh

        total_longitudinal_bones = self.num_bones
        num_body_bones = (
            max(1, total_longitudinal_bones - self.num_tail_bones)
            if self.rig_tail
            else total_longitudinal_bones
        )

        # Main body spine bone chain (from snout to peduncle)
        body_armature_indices = np.linspace(
            0, len(self.target_spine) - 1, num_body_bones + 1, dtype=int
        )
        body_armature_verts = self.target_spine[body_armature_indices].copy()

        # Tail bone chain continuing in a straight line at tail level
        tail_faces = segments.get("tail", [])
        if tail_faces:
            tail_v = mesh.vertices[np.unique(mesh.faces[tail_faces])]
            x_tail_end = float(tail_v[:, 0].max())
            y_tail_mid = float(0.5 * (tail_v[:, 1].min() + tail_v[:, 1].max()))
            z_tail_mid = float(0.5 * (tail_v[:, 2].min() + tail_v[:, 2].max()))
        else:
            x_tail_end = float(mesh.bounds[1, 0])
            y_tail_mid = float(body_armature_verts[-1, 1])
            z_tail_mid = float(body_armature_verts[-1, 2])

        if self.rig_tail and self.num_tail_bones > 0:
            tail_x_pts = np.linspace(
                body_armature_verts[-1, 0], x_tail_end, self.num_tail_bones + 1
            )
            tail_armature_verts = np.zeros((self.num_tail_bones + 1, 3))
            tail_armature_verts[:, 0] = tail_x_pts
            tail_armature_verts[:, 1] = y_tail_mid
            tail_armature_verts[:, 2] = z_tail_mid

            straight_armature_verts = np.vstack(
                [body_armature_verts, tail_armature_verts[1:]]
            )
        else:
            straight_armature_verts = body_armature_verts

        root_bone = Bone(
            id="spine_0",
            head=tuple(straight_armature_verts[0]),
            tail=tuple(straight_armature_verts[1]),
        )
        armature = Armature(root_bone)
        curr_bone = root_bone
        spine_bones: list[Bone] = [root_bone]

        for i in range(2, len(straight_armature_verts)):
            curr_bone = armature.add_connected_bone(
                curr_bone, tail=tuple(straight_armature_verts[i])
            )
            bone_idx = i - 1
            if self.rig_tail and bone_idx >= num_body_bones:
                tail_idx = bone_idx - num_body_bones
                curr_bone.id = f"tail_{tail_idx}"
            else:
                curr_bone.id = f"spine_{bone_idx}"
            spine_bones.append(curr_bone)

        # Rig Side Pectoral Fins (Left & Right)
        if self.rig_pectoral_fins:
            l_faces = segments.get("left_pectoral_fin", [])
            r_faces = segments.get("right_pectoral_fin", [])

            root_l, tip_l = None, None
            if l_faces:
                root_l, tip_l = self._get_fin_endpoints(l_faces)

            root_r, tip_r = None, None
            if r_faces:
                root_r, tip_r = self._get_fin_endpoints(r_faces)

            # Enforce strict bilateral left-right symmetry across sagittal plane (Z=0)
            if root_l is not None and root_r is not None:
                sym_root_x = float(0.5 * (root_l[0] + root_r[0]))
                sym_root_y = float(0.5 * (root_l[1] + root_r[1]))
                sym_root_z = float(0.5 * (abs(root_l[2]) + abs(root_r[2])))

                sym_tip_x = float(0.5 * (tip_l[0] + tip_r[0]))
                sym_tip_y = float(0.5 * (tip_l[1] + tip_r[1]))
                sym_tip_z = float(0.5 * (abs(tip_l[2]) + abs(tip_r[2])))

                root_l = np.array([sym_root_x, sym_root_y, sym_root_z])
                tip_l = np.array([sym_tip_x, sym_tip_y, sym_tip_z])
                root_r = np.array([sym_root_x, sym_root_y, -sym_root_z])
                tip_r = np.array([sym_tip_x, sym_tip_y, -sym_tip_z])
            elif root_l is not None:
                root_r = np.array([root_l[0], root_l[1], -abs(root_l[2])])
                tip_r = np.array([tip_l[0], tip_l[1], -abs(tip_l[2])])
            elif root_r is not None:
                root_l = np.array([root_r[0], root_r[1], abs(root_r[2])])
                tip_l = np.array([tip_r[0], tip_r[1], abs(tip_r[2])])

            # Determine the common parent bone along the spine for both side fins
            fin_x = None
            if root_l is not None:
                fin_x = root_l[0]

            if fin_x is not None:
                common_parent_bone = min(
                    spine_bones[:num_body_bones],
                    key=lambda b: abs(b.head[0] - fin_x),
                )

                # Left Pectoral Fin (+Z flank)
                if root_l is not None:
                    fin_l = armature.add_unconnected_bone(
                        parent=common_parent_bone,
                        head=tuple(root_l),
                        tail=tuple(tip_l),
                    )
                    fin_l.id = "left_pectoral_fin_0"

                # Right Pectoral Fin (-Z flank)
                if root_r is not None:
                    fin_r = armature.add_unconnected_bone(
                        parent=common_parent_bone,
                        head=tuple(root_r),
                        tail=tuple(tip_r),
                    )
                    fin_r.id = "right_pectoral_fin_0"

        # Rig Dorsal Fin (Top Fin) if enabled
        if self.rig_dorsal_fin:
            d_faces = segments.get("dorsal_fin", [])
            if d_faces:
                d_verts = mesh.vertices[np.unique(mesh.faces[d_faces])]
                if len(d_verts) > 0:
                    y_order = np.argsort(d_verts[:, 1])
                    k_sample = max(1, len(d_verts) // 10)
                    root_d = d_verts[y_order[:k_sample]].mean(axis=0)
                    tip_d = d_verts[y_order[-k_sample:]].mean(axis=0)

                    parent_bone = min(
                        spine_bones[:num_body_bones],
                        key=lambda b: abs(b.head[0] - root_d[0]),
                    )
                    fin_d = armature.add_unconnected_bone(
                        parent=parent_bone, head=tuple(root_d), tail=tuple(tip_d)
                    )
                    fin_d.id = "dorsal_fin_0"

        self.armature = armature
        return armature

    def animate(self) -> Animator:
        """
        Dynamically creates and registers procedural swimming, idle, and sprint
        animation clips steered into the plane of the detected tail orientation
        with anchored head stabilization and synchronized fin kinematics.

        Returns
        -------
        Animator
            The Animator configured with steered AnimationClip objects.
        """
        armature = self.model.armature or self.armature
        if armature is None:
            raise ValueError(
                "Armature is not set on model. Run rig() before animate()."
            )

        animator = Animator(armature=armature)

        # Primary longitudinal chain (spine + tail bones)
        spine_indices = [
            i
            for i, b in enumerate(armature.bones_list)
            if "pectoral" not in b.id and "dorsal" not in b.id
        ]
        bones = [armature.bones_list[i] for i in spine_indices]

        # Extract cumulative arc-length distances along the spine
        positions = [np.array(bones[0].head, dtype=np.float64)]
        for b in bones:
            positions.append(np.array(b.tail, dtype=np.float64))
        positions = np.array(positions)

        seg_vectors = np.diff(positions, axis=0)
        seg_lengths = np.linalg.norm(seg_vectors, axis=1)
        cum_dist = np.concatenate(([0.0], np.cumsum(seg_lengths)))
        total_len = cum_dist[-1]
        norm_s = cum_dist / max(total_len, 1e-6)

        bind_positions = np.stack(
            [cum_dist, np.zeros_like(cum_dist), np.zeros_like(cum_dist)], axis=-1
        )

        if self.tail_orientation is None:
            self.tail_orientation = self._detect_tail_orientation()

        # Steer rotation: Fish undulate across Z (transverse/yaw), Cetaceans across Y (sagittal/pitch)
        if self.tail_orientation == "vertical":
            steer_rot = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])[
                :3, :3
            ]
        else:
            steer_rot = np.eye(3)

        l_pec_idx = next(
            (i for i, b in enumerate(armature.bones_list) if "left_pectoral" in b.id),
            None,
        )
        r_pec_idx = next(
            (i for i, b in enumerate(armature.bones_list) if "right_pectoral" in b.id),
            None,
        )
        dorsal_idx = next(
            (i for i, b in enumerate(armature.bones_list) if "dorsal" in b.id),
            None,
        )

        num_total_bones = len(armature.bones_list)

        for clip_name, clip_cfg in self.animations.items():
            wave_amplitude = clip_cfg.get("wave_amplitude", 0.22)
            wave_duration = clip_cfg.get("wave_duration", 1.4)
            num_waves = clip_cfg.get("num_waves", 0.85)
            head_amplitude_ratio = clip_cfg.get("head_amplitude_ratio", 0.08)
            pectoral_mode = clip_cfg.get("pectoral_mode", "active")
            pectoral_flap_deg = clip_cfg.get("pectoral_flap_deg", 14.0)
            pectoral_pitch_deg = clip_cfg.get("pectoral_pitch_deg", 6.0)
            pectoral_close_deg = clip_cfg.get("pectoral_close_deg", 35.0)
            dorsal_flex_deg = clip_cfg.get("dorsal_flex_deg", 4.0)
            is_loopable = clip_cfg.get("is_loopable", True)

            # Smooth Quadratic Envelope: Anchors snout and smoothly expands toward caudal fin
            envelope = head_amplitude_ratio + (1.0 - head_amplitude_ratio) * (norm_s**2)

            num_frames = int(round(wave_duration * self.frame_rate))
            times = np.linspace(0.0, wave_duration, num_frames, endpoint=False)
            omega = 2.0 * np.pi / wave_duration
            k = 2.0 * np.pi * num_waves / total_len

            clip_positions: dict[float, list[np.ndarray]] = {}

            for t in times:
                phase = k * cum_dist - omega * t
                amp = wave_amplitude * envelope
                d_env_ds = np.gradient(amp, cum_dist)
                dy_dx = d_env_ds * np.sin(phase) + amp * k * np.cos(phase)

                tangents = np.stack(
                    [np.ones_like(dy_dx), dy_dx, np.zeros_like(dy_dx)], axis=-1
                )
                tangents /= np.linalg.norm(tangents, axis=-1, keepdims=True)

                frame_pts = [np.array([0.0, 0.0, 0.0])]
                for i in range(len(seg_lengths)):
                    frame_pts.append(frame_pts[-1] + seg_lengths[i] * tangents[i])
                frame_pts = np.array(frame_pts)
                frame_pts[:, 1] -= np.mean(frame_pts[:, 1])

                raw_rots = successive_rotations(
                    bind_positions, frame_pts, is_positions=True
                )
                rot_matrices = [
                    r.detach().cpu().numpy()
                    if isinstance(r, torch.Tensor)
                    else np.asarray(r)
                    for r in raw_rots
                ]
                if steer_rot is not None:
                    rot_matrices = [steer_rot @ R @ steer_rot.T for R in rot_matrices]

                full_frame = [
                    np.eye(3, dtype=np.float32) for _ in range(num_total_bones)
                ]
                for idx_in_spine, bone_idx in enumerate(spine_indices):
                    full_frame[bone_idx] = rot_matrices[idx_in_spine].astype(np.float32)

                # Pectoral Fin Kinematics
                if pectoral_mode == "closed":
                    tuck_angle = np.radians(pectoral_close_deg)
                    flutter = np.radians(2.0) * np.sin(omega * t)
                    if l_pec_idx is not None:
                        R_fold_l = trimesh.transformations.rotation_matrix(
                            -tuck_angle, [0, 1, 0]
                        )[:3, :3]
                        R_roll_l = trimesh.transformations.rotation_matrix(
                            flutter, [1, 0, 0]
                        )[:3, :3]
                        full_frame[l_pec_idx] = (R_fold_l @ R_roll_l).astype(np.float32)
                    if r_pec_idx is not None:
                        R_fold_r = trimesh.transformations.rotation_matrix(
                            tuck_angle, [0, 1, 0]
                        )[:3, :3]
                        R_roll_r = trimesh.transformations.rotation_matrix(
                            -flutter, [1, 0, 0]
                        )[:3, :3]
                        full_frame[r_pec_idx] = (R_fold_r @ R_roll_r).astype(np.float32)
                else:
                    flap_angle = np.radians(pectoral_flap_deg) * np.sin(
                        omega * t + np.pi / 4
                    )
                    pitch_angle = np.radians(pectoral_pitch_deg) * np.cos(
                        omega * t + np.pi / 4
                    )
                    if l_pec_idx is not None:
                        R_roll_l = trimesh.transformations.rotation_matrix(
                            flap_angle, [1, 0, 0]
                        )[:3, :3]
                        R_pitch_l = trimesh.transformations.rotation_matrix(
                            pitch_angle, [0, 1, 0]
                        )[:3, :3]
                        full_frame[l_pec_idx] = (R_roll_l @ R_pitch_l).astype(
                            np.float32
                        )
                    if r_pec_idx is not None:
                        R_roll_r = trimesh.transformations.rotation_matrix(
                            -flap_angle, [1, 0, 0]
                        )[:3, :3]
                        R_pitch_r = trimesh.transformations.rotation_matrix(
                            pitch_angle, [0, 1, 0]
                        )[:3, :3]
                        full_frame[r_pec_idx] = (R_roll_r @ R_pitch_r).astype(
                            np.float32
                        )

                # Dorsal Fin Stabilization Flexing
                if dorsal_idx is not None:
                    d_angle = np.radians(dorsal_flex_deg) * np.sin(
                        omega * t - np.pi / 3
                    )
                    R_dorsal = trimesh.transformations.rotation_matrix(
                        d_angle, [0, 0, 1]
                    )[:3, :3]
                    full_frame[dorsal_idx] = R_dorsal.astype(np.float32)

                clip_positions[float(t)] = full_frame

            clip = AnimationClip(
                name=clip_name,
                duration=wave_duration,
                armature=armature,
                is_loopable=is_loopable,
            )
            clip.positions = clip_positions
            animator.add_animation_clip(clip)

        return animator

    def save_intermediate_artifacts(self, output_dir: str | Path) -> dict[str, Path]:
        """
        Exports all intermediate stages to a dedicated folder for diagnostics and visualization:
        1. 01_segmented_mesh.glb: Color-coded 3D mesh highlighting anatomical parts on initial mesh.
        2. 02_extracted_1d_spine.glb: Extracted 1D centerline spine control points.
        3. 03_evaluated_spline.glb: Dense Catmull-Rom spline curve.
        4. 04_canonical_mesh.glb: Canonical rest-pose 3D mesh.
        5. 05_rigged_armature.glb: Rest-pose mesh with rigged bones and skin weights.
        6. 06_animated_fish.glb: Full skeletal animated GLB.

        Parameters
        ----------
        output_dir : str | Path
            Directory where intermediate artifacts should be saved.

        Returns
        -------
        dict[str, Path]
            Mapping from step name to output file path.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        saved_files: dict[str, Path] = {}

        # Segmented Mesh with Color Map (textures stripped, pure ColorVisuals)
        if self.segments:
            seg_mesh = self.initial_mesh.copy()
            seg_mesh.visual = trimesh.visual.ColorVisuals(mesh=seg_mesh)
            face_colors = np.tile(
                np.array([120, 120, 120, 255], dtype=np.uint8),
                (len(seg_mesh.faces), 1),
            )
            for part_name, faces in self.segments.items():
                if part_name in PART_COLOR_PALETTE and len(faces) > 0:
                    face_colors[faces] = PART_COLOR_PALETTE[part_name]
            seg_mesh.visual.face_colors = face_colors
            p1 = out_path / "01_segmented_mesh.glb"
            seg_mesh.export(p1)
            saved_files["01_segmented_mesh"] = p1

        # Extracted 1D Spine Nodes
        if self.raw_spine_points is not None:
            skel_spheres = []
            for pt in self.raw_spine_points:
                s = trimesh.creation.icosphere(subdivisions=1, radius=0.02)
                s.apply_translation(pt)
                skel_spheres.append(s)
            skel_mesh = trimesh.util.concatenate(skel_spheres)
            p2 = out_path / "02_extracted_1d_spine.glb"
            skel_mesh.export(p2)
            saved_files["02_extracted_1d_spine"] = p2

        # Dense Spline Curve
        if self.source_spine is not None:
            curve_spheres = []
            for pt in self.source_spine[::3]:
                s = trimesh.creation.icosphere(subdivisions=1, radius=0.012)
                s.apply_translation(pt)
                curve_spheres.append(s)
            curve_mesh = trimesh.util.concatenate(curve_spheres)
            p3 = out_path / "03_evaluated_spline.glb"
            curve_mesh.export(p3)
            saved_files["03_evaluated_spline"] = p3

        # Canonical Rest-Pose Mesh
        if self.model.mesh is not None:
            p4 = out_path / "04_canonical_mesh.glb"
            self.model.mesh.export(p4)
            saved_files["04_canonical_mesh"] = p4

        # Rigged Armature GLB
        if self.model.armature is not None:
            p5 = out_path / "05_rigged_armature.glb"
            self.model.export(p5)
            saved_files["05_rigged_armature"] = p5

        # Animated Fish GLB
        if self.model.animator is not None:
            p6 = out_path / "06_animated_fish.glb"
            self.model.export(p6, animation=self.model.animator)
            saved_files["06_animated_fish"] = p6

        return saved_files
