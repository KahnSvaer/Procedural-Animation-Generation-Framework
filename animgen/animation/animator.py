from typing import Any, Callable

from animgen.core.armature import Armature
from animgen.core.types import AnimationFrame, TimeSeconds, Animation


AnimationCallback = Callable[..., tuple[list[int], Animation]]


class AnimationClip:
    """
    Would be responsible for aggregating multiple motion clips into a single animation
    """

    def __init__(
        self,
        name: str,
        duration: float,
        armature: Armature,
        is_loopable: bool = False,
    ):
        self.name: str = name
        self.duration: TimeSeconds = duration
        self.armature: Armature = armature
        self.is_loopable: bool = is_loopable

        self.positions: dict[TimeSeconds, AnimationFrame] = {}

    def add_animation_movements(
        self,
        callback: AnimationCallback,
        list_bones: list[int],
        *args: Any,
    ) -> None:
        """
        Adds animation movement
        """
        pass

    def check_loopable(
        self,
    ) -> None:
        # Essentially check if the last posvector is equal to starting pose vector
        # WOuld change the value of self.loopable in place
        pass


class Animator:
    """
    Manages procedural animations on armatures.
    Supports registering animations, tracking them, and evaluating them.
    """

    def __init__(self) -> None:
        self.animations: dict[str, AnimationClip] = {}
        self.animation_lists: dict[str, AnimationClip] = {}

    def add_animation_clip(self, animation_clip: AnimationClip) -> None:
        """
        Adds an animation clip to the animator.
        """
        self.animations[animation_clip.name] = animation_clip
