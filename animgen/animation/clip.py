"""
Module for managing procedural animation clips on armatures.

Provides AnimationClip for combining multi-track bone animations, keyframe interpolation,
Forward Kinematics (FK), Dual Quaternion Skinning (DQS), Linear Blend Skinning (LBS),
and mesh baking for vertex animation/morphing.
"""

from typing import (
    Any,
    Callable,
    Literal,
    Optional,
    TypedDict,
    Union,
)
import inspect
import numpy as np
import torch
import trimesh

from animgen.core.armature import Armature
from animgen.core.types import AnimationFrame, TimeSeconds, Animation
from animgen.utils.math import slerp_rotation_matrix
from animgen.animation.kinematics import compute_forward_kinematics
from animgen.animation.deformation import apply_mesh_deformation


AnimationCallback = Callable[..., Union[Animation, tuple[list[int], Animation]]]


class AnimationCallbackData(TypedDict, total=False):
    callback: AnimationCallback
    list_bones: list[int]
    offset: TimeSeconds
    timeline_offset: TimeSeconds
    steer_rotation: Optional[Union[torch.Tensor, np.ndarray]]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


class AnimationClip:
    """
    Responsible for aggregating multiple motion clips into a single animation.
    Supports multi-bone track combination, keyframe evaluation, SLERP interpolation,
    seamless loopability verification, and Dual Quaternion / Linear Blend Skinning mesh baking.
    """

    def __init__(
        self,
        name: str,
        duration: float,
        armature: Armature,
        is_loopable: bool = False,
        skin_weights: Optional[dict[str, np.ndarray]] = None,
    ):
        self.name: str = name
        self.duration: TimeSeconds = duration
        self.armature: Armature = armature
        self.is_loopable: bool = is_loopable
        self.skin_weights: Optional[dict[str, np.ndarray]] = skin_weights

        self.positions: dict[TimeSeconds, AnimationFrame] = {}
        self.animation_callbacks: list[AnimationCallbackData] = []
        self.baked_meshes: dict[TimeSeconds, trimesh.Trimesh] = {}

    @property
    def loopable(self) -> bool:
        """Alias for self.is_loopable."""
        return self.is_loopable

    def add_animation_movements(
        self,
        callback: AnimationCallback,
        list_bones: list[int],
        start_offset: TimeSeconds = 0.0,
        timeline_offset: Optional[TimeSeconds] = None,
        steer_rotation: Optional[Union[torch.Tensor, np.ndarray]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Adds an animation movement callback for a specified subset of bones.

        Parameters
        ----------
        callback : AnimationCallback
            The animation callback to use (e.g. chain_wave_generator).
        list_bones : list[int]
            The list of bone indices to which this movement applies.
        offset : TimeSeconds, default=0.0
            Timeline offset in seconds (delays/shifts when the movement begins on the clip timeline).
        timeline_offset : Optional[TimeSeconds], default=None
            Explicit timeline offset in seconds. If provided, overrides offset.
        steer_rotation : Optional[torch.Tensor | np.ndarray], default=None
            3x3 rotation matrix in SO(3) to steer / rotate the coordinate frame of the movement.
        *args : Any
            Additional positional arguments to pass to the animation callback.
        **kwargs : Any
            Additional keyword arguments to pass to the animation callback (e.g. time_offset, wave_amplitude, etc.).
        """
        actual_timeline_offset = (
            timeline_offset if timeline_offset is not None else start_offset
        )

        self.animation_callbacks.append(
            AnimationCallbackData(
                callback=callback,
                list_bones=list_bones,
                offset=actual_timeline_offset,
                timeline_offset=actual_timeline_offset,
                steer_rotation=steer_rotation,
                args=args,
                kwargs=kwargs,
            )
        )

    def _invoke_callback(
        self,
        cb_data: AnimationCallbackData,
        override_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[list[int], Animation]:
        """Helper to invoke an animation callback with automatically bound arguments."""
        callback = cb_data["callback"]
        list_bones = cb_data["list_bones"]
        args = cb_data.get("args", ())
        kwargs = cb_data.get("kwargs", {}).copy()
        if override_kwargs:
            kwargs.update(override_kwargs)

        sig = inspect.signature(callback)
        if "armature" in sig.parameters and "armature" not in kwargs:
            kwargs["armature"] = self.armature
        if "list_bones" in sig.parameters and "list_bones" not in kwargs:
            kwargs["list_bones"] = list_bones
        if "index_bones" in sig.parameters and "index_bones" not in kwargs:
            kwargs["index_bones"] = list_bones
        if "bones" in sig.parameters and "bones" not in kwargs:
            kwargs["bones"] = list_bones

        raw_res = callback(*args, **kwargs)
        if isinstance(raw_res, tuple) and len(raw_res) == 2:
            return raw_res[0], raw_res[1]
        return list_bones, raw_res

    def check_loopable(
        self,
        atol: float = 1e-4,
    ) -> bool:
        """
        Verifies if the animation is periodic and smoothly loops at its endpoints.
        Updates self.is_loopable.

        Parameters
        ----------
        atol : float, default=1e-4
            Absolute tolerance for comparing boundary frame rotations.

        Returns
        -------
        bool
            True if the animation loops seamlessly.
        """
        if not self.positions:
            self.generate_animation()

        if len(self.positions) < 2:
            self.is_loopable = True
            return True

        sorted_times = sorted(self.positions.keys())
        first_frame = self.positions[sorted_times[0]]
        last_frame = self.positions[sorted_times[-1]]

        if np.isclose(sorted_times[-1], self.duration, atol=1e-4):
            is_loop = True
            for r0, r1 in zip(first_frame, last_frame):
                r0_np = (
                    r0.detach().cpu().numpy()
                    if hasattr(r0, "detach")
                    else np.asarray(r0, dtype=np.float64)
                )
                r1_np = (
                    r1.detach().cpu().numpy()
                    if hasattr(r1, "detach")
                    else np.asarray(r1, dtype=np.float64)
                )
                if not np.allclose(r0_np, r1_np, atol=atol):
                    is_loop = False
                    break
            self.is_loopable = is_loop
            return self.is_loopable

        is_loop = True
        for cb_data in self.animation_callbacks:
            kwargs = cb_data.get("kwargs", {})
            wave_type = kwargs.get("wave", "travelling")
            if wave_type == "pulse":
                is_loop = False
                break

        self.is_loopable = is_loop
        return self.is_loopable

    def generate_animation(self) -> Animation:
        """
        Generates and aggregates all registered procedural animation tracks into self.positions.

        Returns
        -------
        Animation
            Mapping from timestamp (float) to AnimationFrame (list of 3x3 rotation matrices).
        """
        num_bones = len(self.armature.bones_list)
        all_timestamps: set[TimeSeconds] = set()
        track_results: list[tuple[list[int], Animation]] = []

        for cb_data in self.animation_callbacks:
            returned_bones, anim_dict = self._invoke_callback(cb_data)
            tl_offset = cb_data.get("timeline_offset", cb_data.get("offset", 0.0))
            steer = cb_data.get("steer_rotation", None)
            if steer is not None:
                if isinstance(steer, np.ndarray):
                    steer = torch.tensor(steer, dtype=torch.float32)
                else:
                    steer = steer.to(dtype=torch.float32)

            if not anim_dict:
                continue

            # Apply steering rotation and timeline offset
            shifted_anim: Animation = {}
            for t_raw, frame in anim_dict.items():
                t_shifted = round(float(t_raw + tl_offset), 6)
                if steer is not None and len(frame) > 0:
                    steered_frame = []
                    for r in frame:
                        if isinstance(r, np.ndarray):
                            r = torch.tensor(r, dtype=torch.float32)
                        steer_matched = steer.to(dtype=r.dtype, device=r.device)
                        steered_frame.append(steer_matched @ r @ steer_matched.T)
                    shifted_anim[t_shifted] = steered_frame
                else:
                    shifted_anim[t_shifted] = frame
                all_timestamps.add(t_shifted)

            track_results.append((returned_bones, shifted_anim))

        if not all_timestamps:
            return self.positions

        sorted_times = sorted(all_timestamps)
        self.positions.clear()

        for t in sorted_times:
            frame_matrices = [
                torch.eye(3, dtype=torch.float32) for _ in range(num_bones)
            ]

            for bones_subset, anim_dict in track_results:
                if t in anim_dict:
                    sub_frame = anim_dict[t]
                    for sub_idx, bone_idx in enumerate(bones_subset):
                        if sub_idx < len(sub_frame) and bone_idx < num_bones:
                            mat = sub_frame[sub_idx]
                            if isinstance(mat, np.ndarray):
                                mat = torch.tensor(mat, dtype=torch.float32)
                            else:
                                mat = mat.to(dtype=torch.float32)
                            frame_matrices[bone_idx] = mat @ frame_matrices[bone_idx]
                else:
                    sub_times = sorted(anim_dict.keys())
                    if t < sub_times[0]:
                        eval_t = sub_times[0]
                    elif t > sub_times[-1]:
                        eval_t = sub_times[-1]
                    else:
                        eval_t = None
                        for i in range(len(sub_times) - 1):
                            t0, t1 = sub_times[i], sub_times[i + 1]
                            if t0 <= t <= t1:
                                alpha = (t - t0) / (t1 - t0)
                                f0 = anim_dict[t0]
                                f1 = anim_dict[t1]
                                for sub_idx, bone_idx in enumerate(bones_subset):
                                    if (
                                        sub_idx < len(f0)
                                        and sub_idx < len(f1)
                                        and bone_idx < num_bones
                                    ):
                                        interp_mat = slerp_rotation_matrix(
                                            f0[sub_idx], f1[sub_idx], alpha
                                        )
                                        if isinstance(interp_mat, np.ndarray):
                                            interp_mat = torch.tensor(
                                                interp_mat, dtype=torch.float32
                                            )
                                        else:
                                            interp_mat = interp_mat.to(
                                                dtype=torch.float32
                                            )
                                        frame_matrices[bone_idx] = (
                                            interp_mat @ frame_matrices[bone_idx]
                                        )
                                break
                    if eval_t is not None:
                        sub_frame = anim_dict[eval_t]
                        for sub_idx, bone_idx in enumerate(bones_subset):
                            if sub_idx < len(sub_frame) and bone_idx < num_bones:
                                mat = sub_frame[sub_idx]
                                if isinstance(mat, np.ndarray):
                                    mat = torch.tensor(mat, dtype=torch.float32)
                                else:
                                    mat = mat.to(dtype=torch.float32)
                                frame_matrices[bone_idx] = (
                                    mat @ frame_matrices[bone_idx]
                                )

            self.positions[t] = frame_matrices

        self.check_loopable()
        return self.positions

    def evaluate(self, time: TimeSeconds) -> AnimationFrame:
        """
        Evaluates the animation at an arbitrary continuous timestamp using SLERP interpolation.

        Parameters
        ----------
        time : TimeSeconds
            The timestamp in seconds to evaluate.

        Returns
        -------
        AnimationFrame
            List of 3x3 rotation matrices for each bone at the requested time.
        """
        if not self.positions:
            self.generate_animation()

        sorted_times = sorted(self.positions.keys())
        if time <= sorted_times[0]:
            return self.positions[sorted_times[0]]
        if time >= sorted_times[-1]:
            return self.positions[sorted_times[-1]]

        for i in range(len(sorted_times) - 1):
            t0, t1 = sorted_times[i], sorted_times[i + 1]
            if t0 <= time <= t1:
                alpha = float((time - t0) / (t1 - t0))
                f0 = self.positions[t0]
                f1 = self.positions[t1]
                interpolated: AnimationFrame = []
                for m0, m1 in zip(f0, f1):
                    r_interp = slerp_rotation_matrix(m0, m1, alpha)
                    interpolated.append(r_interp)
                return interpolated

        return self.positions[sorted_times[-1]]

    def bake(
        self,
        mesh: Optional[trimesh.Trimesh] = None,
        skin_weights: Optional[dict[str, np.ndarray]] = None,
        method: Literal["dqs", "lbs"] = "dqs",
    ) -> dict[TimeSeconds, trimesh.Trimesh]:
        """
        Bakes the animation clip into per-frame deformed meshes for vertex animation/morphing.

        Parameters
        ----------
        mesh : Optional[trimesh.Trimesh], default=None
            The reference 3D mesh geometry to deform.
        skin_weights : Optional[dict[str, np.ndarray]], default=None
            Mapping from bone ID to per-vertex skin weights.
            If None, self.skin_weights is used.
        method : {"dqs", "lbs"}, default="dqs"
            Deformation algorithm (Dual Quaternion Skinning or Linear Blend Skinning).

        Returns
        -------
        dict[TimeSeconds, trimesh.Trimesh]
            Mapping from timestamp to deformed trimesh.Trimesh.
        """
        if not self.positions:
            self.generate_animation()

        if mesh is None:
            if hasattr(self.armature, "mesh") and isinstance(
                self.armature.mesh, trimesh.Trimesh
            ):
                mesh = self.armature.mesh
            else:
                raise ValueError(
                    "A trimesh.Trimesh must be provided to bake the animation."
                )

        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.to_mesh()

        if skin_weights is None:
            skin_weights = self.skin_weights

        if skin_weights is None:
            raise ValueError(
                "skin_weights must be provided to bake the animation clip. "
                "Please precompute skin weights (e.g., via animgen.rigging.skinning.compute_auto_skin_weights) "
                "and pass them to bake() or assign to clip.skin_weights."
            )

        self.baked_meshes.clear()
        for t, frame in self.positions.items():
            rotations, positions = compute_forward_kinematics(self.armature, frame)
            heads = {b_id: pos[0] for b_id, pos in positions.items()}
            deformed = apply_mesh_deformation(
                mesh=mesh,
                armature=self.armature,
                global_bone_rotations=rotations,
                global_bone_heads=heads,
                skin_weights=skin_weights,
                method=method,
            )
            self.baked_meshes[t] = deformed

        return self.baked_meshes
