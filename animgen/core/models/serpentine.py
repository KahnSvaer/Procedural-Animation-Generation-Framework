from pathlib import Path
from typing import Any, Optional
import copy
import numpy as np
import torch
import trimesh

from animgen.core.models.pipeline import Pipeline
from animgen.core.models.model import BaseModelClass
from animgen.core.spline import Spline
from animgen.core.armature import Armature, Bone
from animgen.animation.animator import Animator, AnimationClip
from animgen.animation.wave import chain_wave_generator
from animgen.animation.straight import straighten
from animgen.rigging.mesh_contraction import extract_skeleton
from animgen.rigging.refine_skelaton import refine_and_center_skeleton_iterative


DEFAULT_SERPENTINE_PARAMS: dict[str, Any] = {
    "num_bones": 20,
    "frame_rate": 30.0,
    "animations": {
        "slow": {
            "wave_amplitude": 0.15,
            "wave_duration": 3.0,
            "growth_factor": 0.0,
            "num_waves": 2.0,
            "wave_type": "travelling",
        },
        "fast": {
            "wave_amplitude": 0.25,
            "wave_duration": 1.2,
            "growth_factor": 0.05,
            "num_waves": 2.2,
            "wave_type": "travelling",
        },
    },
}


class SerpentineModels(Pipeline):
    """
    End-to-end procedural animation pipeline for serpentine creatures (snakes, eels, worms).

    Performs:
    1. Single continuous body segmentation.
    2. Mesh contraction, Algo B centerline refinement, 0.5-alpha Catmull-Rom spline construction,
       and parallel-transport mesh straightening.
    3. Hierarchical armature construction along the straightened spine.
    4. Procedural wave animations steered into the lateral Z-axis plane.
    """

    def __init__(
        self,
        model: BaseModelClass,
        prompts: list[str] | None = None,
        prompts_embedding_path: Path | None = None,
        num_bones: int = DEFAULT_SERPENTINE_PARAMS["num_bones"],
        frame_rate: float = DEFAULT_SERPENTINE_PARAMS["frame_rate"],
        animations: dict[str, dict[str, Any]] | None = None,
    ):
        """
        Initializes the SerpentineModels pipeline.

        Parameters
        ----------
        model : BaseModelClass
            The input 3D model container.
        prompts : list[str] | None, optional
            List of prompt strings. Defaults to ["body"] if None.
        prompts_embedding_path : Path | None, optional
            Path to precomputed text embeddings.
        num_bones : int, default=20
            Number of bones to construct along the straightened spine.
        frame_rate : float, default=30.0
            Frame rate (FPS) for keyframe animation sampling.
        animations : dict[str, dict[str, Any]] | None, optional
            Dictionary mapping animation clip names to wave parameters.
        """
        if prompts is None and prompts_embedding_path is None:
            prompts = ["body"]

        super().__init__(
            model,
            prompts=prompts,
            prompts_embedding_path=prompts_embedding_path,
        )

        self.num_bones: int = num_bones
        self.frame_rate: float = frame_rate

        self.animations: dict[str, dict[str, Any]] = copy.deepcopy(
            DEFAULT_SERPENTINE_PARAMS["animations"]
        )
        if animations is not None:
            for clip_name, clip_cfg in animations.items():
                if clip_name in self.animations:
                    self.animations[clip_name].update(clip_cfg)
                else:
                    self.animations[clip_name] = clip_cfg

        self.source_spine: Optional[np.ndarray] = None
        self.target_spine: Optional[np.ndarray] = None
        self.spline: Optional[Spline] = None
        self.armature: Optional[Armature] = None

    def segment(self) -> dict[str, list[int]]:
        """
        Segments the serpentine creature. For serpentine bodies, the entire
        mesh is treated as a continuous body chain.

        Returns
        -------
        dict[str, list[int]]
            Mapping from part name ("body") to list of mesh face indices.
        """
        return {"body": list(range(len(self.model.mesh.faces)))}

    def canonicalize(self, segments: dict[str, list[int]]) -> trimesh.Trimesh:
        """
        Extracts the 1D centerline skeleton, constructs a 0.5-alpha Catmull-Rom spline,
        and straightens the mesh along the X-axis into canonical rest pose.

        Parameters
        ----------
        segments : dict[str, list[int]]
            Part face segmentation dictionary.

        Returns
        -------
        trimesh.Trimesh
            The straightened canonical mesh.
        """
        # 1. Mesh Contraction (Auto-welds manifold geometry internally)
        skel_v, skel_e = extract_skeleton(
            self.model.mesh,
            max_iters=20,
            threshold=0.5,
            no_1d_collapses=True,
            return_tuple=True,
        )

        # 2. Algo B Iterative Slice Centering
        skel_v_final, skel_e_ref = refine_and_center_skeleton_iterative(
            self.model.mesh.vertices,
            skel_v,
            skel_e,
            max_edge_len=0.1,
            num_iters=10,
        )

        # 3. Trace 1D continuous node chain from endpoint to endpoint
        adj: dict[int, list[int]] = {i: [] for i in range(len(skel_v_final))}
        for u, v in skel_e_ref:
            adj[u].append(v)
            adj[v].append(u)

        endpoints = [i for i, nbs in adj.items() if len(nbs) == 1]
        start_node = endpoints[0] if len(endpoints) > 0 else 0

        chain = [start_node]
        visited = {start_node}
        curr = start_node
        while True:
            next_nodes = [nb for nb in adj[curr] if nb not in visited]
            if not next_nodes:
                break
            next_node = next_nodes[0]
            visited.add(next_node)
            chain.append(next_node)
            curr = next_node

        ordered_verts = skel_v_final[chain]
        pts_t = [torch.tensor(v, dtype=torch.float32) for v in ordered_verts]

        # 4. Catmull-Rom Spline (alpha=0.5 centripetal, phantom_num_points=1)
        self.spline = Spline(pts_t, alpha=0.5, phantom_num_points=1)
        eval_pts = self.spline.evaluate_curve(num_points_per_segment=5)
        self.source_spine = np.array([pt.detach().cpu().numpy() for pt in eval_pts])

        # 5. Straighten mesh along spline
        straight_mesh = straighten(self.model.mesh, spine_points=self.spline, axis="x")

        # 6. Compute target straight spine matching cumulative arc length
        seg_lens = np.linalg.norm(np.diff(self.source_spine, axis=0), axis=1)
        s = np.concatenate(([0.0], np.cumsum(seg_lens)))
        self.target_spine = np.zeros_like(self.source_spine)
        self.target_spine[:, 0] = s

        return straight_mesh

    def rig(self, segments: dict[str, list[int]]) -> Armature:
        """
        Constructs a hierarchical connected bone Armature along the straightened spine.

        Parameters
        ----------
        segments : dict[str, list[int]]
            Part face segmentation dictionary.

        Returns
        -------
        Armature
            The constructed hierarchical Armature.
        """
        if self.target_spine is None:
            raise ValueError(
                "Target spine has not been computed. Run canonicalize() first."
            )

        armature_indices = np.linspace(
            0, len(self.target_spine) - 1, self.num_bones + 1, dtype=int
        )
        straight_armature_verts = self.target_spine[armature_indices]

        root_bone = Bone(
            head=tuple(straight_armature_verts[0]),
            tail=tuple(straight_armature_verts[1]),
        )
        armature = Armature(root_bone)
        curr_bone = root_bone
        for i in range(2, len(straight_armature_verts)):
            curr_bone = armature.add_connected_bone(
                curr_bone, tail=tuple(straight_armature_verts[i])
            )

        self.armature = armature
        return armature

    def animate(self) -> Animator:
        """
        Dynamically creates and registers procedural wave animation clips
        steered into the lateral Z-axis plane using steer_rotation.

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
        bone_indices = list(range(len(armature.bones_list)))

        # 90-degree SO(3) rotation around X-axis to steer wave peaks into the lateral Z-axis
        steer_z = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])[:3, :3]

        for clip_name, clip_cfg in self.animations.items():
            clip_duration = clip_cfg.get("wave_duration", 2.0)

            clip = AnimationClip(
                name=clip_name,
                duration=clip_duration,
                armature=armature,
                is_loopable=clip_cfg.get("is_loopable", True),
            )
            clip.add_animation_movements(
                chain_wave_generator,
                list_bones=bone_indices,
                wave_amplitude=clip_cfg.get("wave_amplitude", 0.2),
                wave_duration=clip_duration,
                frame_rate=self.frame_rate,
                growth_factor=clip_cfg.get("growth_factor", 0.0),
                num_waves=clip_cfg.get("num_waves", 2.0),
                steer_rotation=steer_z,
                wave=clip_cfg.get("wave_type", "travelling"),
            )
            animator.add_animation_clip(clip)

        return animator
