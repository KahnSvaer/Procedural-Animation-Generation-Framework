"""
Module for managing and aggregating procedural animations on armatures.

Provides Animator for managing multiple animation clips, evaluating them,
managing skin weights, and baking deformed meshes.
"""

from typing import (
    Any,
    ItemsView,
    Iterator,
    KeysView,
    Literal,
    Optional,
    ValuesView,
)
import numpy as np
import trimesh

from animgen.core.armature import Armature
from animgen.core.types import AnimationFrame, TimeSeconds, Animation
from animgen.animation.clip import (
    AnimationClip,
    AnimationCallback,
    AnimationCallbackData,
)


class Animator:
    """
    Manages procedural animations on armatures.
    Supports registering animations, tracking them, evaluating them, managing skin weights, and baking.

    Can be treated directly as a dict of AnimationClips.
    """

    def __init__(
        self,
        armature: Optional[Armature] = None,
        skin_weights: Optional[dict[str, np.ndarray]] = None,
    ) -> None:
        self.armature: Optional[Armature] = armature
        self._skin_weights: Optional[dict[str, np.ndarray]] = skin_weights
        self._animations: dict[str, AnimationClip] = {}

    @property
    def animations(self) -> dict[str, AnimationClip]:
        """Dictionary of registered AnimationClip objects."""
        return self._animations

    @property
    def skin_weights(self) -> Optional[dict[str, np.ndarray]]:
        """Cached skin weights mapping bone ID -> per-vertex skin weights."""
        return self._skin_weights

    @skin_weights.setter
    def skin_weights(self, weights: Optional[dict[str, np.ndarray]]) -> None:
        self._skin_weights = weights

    def add_animation_clip(self, animation_clip: AnimationClip) -> None:
        """
        Adds an animation clip to the animator.
        """
        if self.armature is None and animation_clip.armature is not None:
            self.armature = animation_clip.armature

        if animation_clip.skin_weights is None and self._skin_weights is not None:
            animation_clip.skin_weights = self._skin_weights

        self._animations[animation_clip.name] = animation_clip

    def get_animation_clip(self, name: str) -> Optional[AnimationClip]:
        """
        Retrieves an animation clip by name.
        """
        return self._animations.get(name)

    def generate_all_animations(self) -> dict[str, Animation]:
        """
        Generates animation data for all registered animation clips.
        """
        results: dict[str, Animation] = {}
        for name, clip in self._animations.items():
            results[name] = clip.generate_animation()
        return results

    def evaluate(self, clip_name: str, time: TimeSeconds) -> AnimationFrame:
        """
        Evaluates a specific animation clip at a given timestamp.
        """
        clip = self.get_animation_clip(clip_name)
        if clip is None:
            raise KeyError(f"Animation clip '{clip_name}' not found in Animator.")
        return clip.evaluate(time)

    def bake(
        self,
        mesh: Optional[trimesh.Trimesh] = None,
        skin_weights: Optional[dict[str, np.ndarray]] = None,
        method: Literal["dqs", "lbs"] = "dqs",
    ) -> dict[str, dict[TimeSeconds, trimesh.Trimesh]]:
        """
        Bakes all registered animation clips into per-frame deformed meshes for mesh morphing.

        Parameters
        ----------
        mesh : Optional[trimesh.Trimesh]
            The reference 3D mesh geometry.
        skin_weights : Optional[dict[str, np.ndarray]]
            Precomputed skin weights. If None, self.skin_weights is used.
        method : {"dqs", "lbs"}, default="dqs"
            Deformation algorithm (Dual Quaternion Skinning or Linear Blend Skinning).

        Returns
        -------
        dict[str, dict[TimeSeconds, trimesh.Trimesh]]
            Mapping from clip name to dict of timestamp -> deformed trimesh.Trimesh.
        """
        if skin_weights is None:
            skin_weights = self._skin_weights

        if skin_weights is None:
            raise ValueError(
                "skin_weights must be provided to bake animations. "
                "Please precompute skin weights (e.g., via animgen.rigging.skinning.compute_auto_skin_weights) "
                "and pass them to bake() or assign to animator.skin_weights."
            )

        baked_dict: dict[str, dict[TimeSeconds, trimesh.Trimesh]] = {}
        for name, clip in self._animations.items():
            baked_dict[name] = clip.bake(
                mesh=mesh, skin_weights=skin_weights, method=method
            )
        return baked_dict

    def keys(self) -> KeysView[str]:
        return self._animations.keys()

    def values(self) -> ValuesView[AnimationClip]:
        return self._animations.values()

    def items(self) -> ItemsView[str, AnimationClip]:
        return self._animations.items()

    def get(self, key: str, default: Any = None) -> Optional[AnimationClip]:
        return self._animations.get(key, default)

    def __len__(self) -> int:
        return len(self._animations)

    def __getitem__(self, key: str) -> AnimationClip:
        return self._animations[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._animations)

    def __contains__(self, item: str) -> bool:
        return item in self._animations


__all__ = [
    "Animator",
    "AnimationClip",
    "AnimationCallback",
    "AnimationCallbackData",
]
