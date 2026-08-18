"""
Module for standing wave generation along an armature.

Reference:
- https://en.wikipedia.org/wiki/Standing_wave


"""

import numpy as np

from animgen.core.armature import Armature

from numpy.typing import NDArray
from typing import Literal
from animgen.core.types import Animation
from animgen.utils.math import successive_rotations


def _check_armature_chain(armature: Armature, index_bones: list[int]) -> bool:
    """
    Confirms if the index_bones corresponding to armature.bones_list make a continuous chain
    """
    if len(index_bones) < 2:
        return True  # A single bone is trivially a chain

    bones_list = armature.bones_list
    bones_chain_list = [bones_list[i] for i in index_bones]

    for i in range(len(bones_chain_list) - 1):
        parent_bone = bones_chain_list[i]
        child_bone = bones_chain_list[i + 1]
        if not child_bone.parent == parent_bone:
            return False
        if child_bone.head != parent_bone.tail:
            return False

    return True


def _travelling_wave_generator(
    distances: list[float] | NDArray[np.float64],
    wave_amplitude: float,
    wave_duration: float,
    time_stamps: float | list[float] | NDArray[np.float64],
    growth_factor: float = 0,
    num_waves: float = 2.6,
    phi_s: float = 0.0,
    phi_t: float = 0.0,
) -> Animation:
    """
    Generate a travelling-wave animation from spatial distances and timestamps.

    The travelling wave is modeled as:

        u(s, t) = A exp(g s)
                sin(k s - omega t + phi_s + phi_t)

    where the spatial wave number ``k`` and temporal angular frequency
    ``omega`` are defined as:

        k = 2 pi N / L

        omega = 2 pi / T

    Here, ``N`` is the number of spatial waves, ``L`` is the total
    spatial length, and ``T`` is the temporal period of the wave.

    The resulting displacement is applied along the y-axis to a chain whose
    rest configuration lies along the x-axis. Consequently, each generated
    frame contains the 3D positions of the chain at a given timestamp.

    Parameters
    ----------
    distances : list[float] | NDArray[np.float64]
        Cumulative spatial distances ``s`` at which to evaluate the wave.
        The distances are measured from the root of the armature and define
        the x-coordinate of each point in the generated chain.

    wave_amplitude : float
        Base amplitude ``A`` of the wave at ``s = 0``.

    wave_duration : float
        Temporal period ``T`` of the wave in seconds. The wave completes
        one full temporal oscillation every ``wave_duration`` seconds.

    time_stamps : float | list[float] | NDArray[np.float64]
        Timestamp or timestamps ``t`` at which to evaluate the wave,
        in seconds.

    growth_factor : float, default=0.0
        Exponential spatial growth rate ``g`` of the wave amplitude.

        The amplitude at distance ``s`` is:

            A(s) = A * exp(g * s)

        Positive values increase the amplitude with distance, zero
        produces a constant amplitude, and negative values produce
        exponential damping.

    num_waves : float, default=2.6
        Number of complete spatial wavelengths across the total spatial
        length ``L``.

    phi_s : float, default=0.0
        Spatial phase offset in radians.

    phi_t : float, default=0.0
        Temporal phase offset in radians.

    Returns
    -------
    Animation
        Mapping from timestamps to animation frames containing local bone rotations.

        Each timestamp maps to a list of rotation matrices (shape ``(num_bones, 3, 3)``)
        representing local bone rotations that deform the armature chain from its bind pose
        to the wave shape.
    """

    distances = np.asarray(distances, dtype=np.float64)
    time_stamps = np.asarray(time_stamps, dtype=np.float64)

    total_length = distances[-1]
    num_bones = len(distances) - 1

    bone_length = total_length / num_bones

    if num_waves <= 0.0:
        ceil_waves = 1.0
        ceil_length_target = total_length
        ceil_num_bones = num_bones
    else:
        ceil_waves = float(np.ceil(num_waves))
        ceil_length_target = total_length * (ceil_waves / num_waves)
        ceil_num_bones = int(np.ceil(ceil_length_target / bone_length))

    # Extend distances by appending segments of the same bone_length
    ceil_distances = np.arange(ceil_num_bones + 1) * bone_length

    wave_number = 2 * np.pi * num_waves / total_length
    angular_frequency = 2 * np.pi / wave_duration

    spatial_amplitude = wave_amplitude * np.exp(growth_factor * ceil_distances)

    phase = (
        wave_number * ceil_distances
        - angular_frequency * time_stamps[..., None]
        + phi_s
        + phi_t
    )

    # Spatial derivative of the wave.
    spatial_derivative = spatial_amplitude * (
        growth_factor * np.sin(phase) + wave_number * np.cos(phase)
    )

    # Tangent of the desired curve.
    tangent = np.stack(
        (
            np.ones_like(spatial_derivative),
            spatial_derivative,
            np.zeros_like(spatial_derivative),
        ),
        axis=-1,
    )

    tangent /= np.linalg.norm(
        tangent,
        axis=-1,
        keepdims=True,
    )

    if time_stamps.ndim == 0:
        tangent = tangent[None, ...]

    animation: Animation = {}

    bind_positions = np.stack(
        (
            distances,
            np.zeros_like(distances),
            np.zeros_like(distances),
        ),
        axis=-1,
    )

    for time, frame_tangent in zip(
        np.atleast_1d(time_stamps),
        tangent,
    ):
        frame = [
            (0.0, 0.0, 0.0),
        ]

        for index in range(1, len(ceil_distances)):
            seg_length = ceil_distances[index] - ceil_distances[index - 1]

            previous_position = np.asarray(
                frame[-1],
                dtype=np.float64,
            )

            direction = frame_tangent[index - 1]

            position = previous_position + seg_length * direction

            frame.append(tuple(position.tolist()))

        y_coords = np.array([p[1] for p in frame])
        mean_y_ceil = np.mean(y_coords)
        shifted_frame = [(p[0], p[1] - mean_y_ceil, p[2]) for p in frame]
        truncated_frame = shifted_frame[: len(distances)]

        animation[float(time)] = successive_rotations(
            bind_positions,
            np.array(truncated_frame),
            is_positions=True,
        )

    return animation


def _standing_wave_generator(
    distances: list[float] | NDArray[np.float64],
    wave_amplitude: float,
    wave_duration: float,
    time_stamps: float | list[float] | NDArray[np.float64],
    growth_factor: float = 0,
    num_waves: float = 2.6,
    phi_s: float = 0.0,
    phi_t: float = 0.0,
) -> Animation:
    """
    Generate a standing-wave animation from spatial distances and timestamps.

    The standing wave is modeled as:

        u(s, t) = 2 A exp(g s)
                cos(k s + phi_s) sin(omega t + phi_t)

    where the spatial wave number ``k`` and temporal angular frequency
    ``omega`` are defined as:

        k = 2 pi N / L

        omega = 2 pi / T

    Here, ``N`` is the number of spatial waves, ``L`` is the total
    spatial length, and ``T`` is the temporal period of the wave.

    The resulting displacement is applied along the y-axis to a chain whose
    rest configuration lies along the x-axis. Consequently, each generated
    frame contains the 3D positions of the chain at a given timestamp.

    Parameters
    ----------
    distances : list[float] | NDArray[np.float64]
        Cumulative spatial distances ``s`` at which to evaluate the wave.
        The distances are measured from the root of the armature and define
        the x-coordinate of each point in the generated chain.

    wave_amplitude : float
        Base amplitude ``A`` of the wave at ``s = 0``.

    wave_duration : float
        Temporal period ``T`` of the wave in seconds. The wave completes
        one full temporal oscillation every ``wave_duration`` seconds.

    time_stamps : float | list[float] | NDArray[np.float64]
        Timestamp or timestamps ``t`` at which to evaluate the wave,
        in seconds.

    growth_factor : float, default=0.0
        Exponential spatial growth rate ``g`` of the wave amplitude.

        The amplitude at distance ``s`` is:

            A(s) = 2 * A * exp(g * s)

        Positive values increase the amplitude with distance, zero
        produces a constant amplitude, and negative values produce
        exponential damping.

    num_waves : float, default=2.6
        Number of complete spatial wavelengths across the total spatial
        length ``L``.

    phi_s : float, default=0.0
        Spatial phase offset in radians.

    phi_t : float, default=0.0
        Temporal phase offset in radians.

    Returns
    -------
    Animation
        Mapping from timestamps to animation frames containing local bone rotations.

        Each timestamp maps to a list of rotation matrices (shape ``(num_bones, 3, 3)``)
        representing local bone rotations that deform the armature chain from its bind pose
        to the wave shape.
    """
    distances = np.asarray(distances, dtype=np.float64)
    time_stamps = np.asarray(time_stamps, dtype=np.float64)

    total_length = distances[-1]
    num_bones = len(distances) - 1

    bone_length = total_length / num_bones

    if num_waves <= 0.0:
        ceil_waves = 1.0
        ceil_length_target = total_length
        ceil_num_bones = num_bones
    else:
        ceil_waves = float(np.ceil(num_waves))
        ceil_length_target = total_length * (ceil_waves / num_waves)
        ceil_num_bones = int(np.ceil(ceil_length_target / bone_length))

    # Extend distances by appending segments of the same bone_length
    ceil_distances = np.arange(ceil_num_bones + 1) * bone_length

    wave_number = 2 * np.pi * num_waves / total_length
    angular_frequency = 2 * np.pi / wave_duration

    spatial_amplitude = 2 * wave_amplitude * np.exp(growth_factor * ceil_distances)

    cos_part = np.cos(wave_number * ceil_distances + phi_s)
    sin_part = np.sin(wave_number * ceil_distances + phi_s)

    spatial_deriv = growth_factor * cos_part - wave_number * sin_part

    t_arr = np.atleast_1d(time_stamps)
    temporal_part = np.sin(angular_frequency * t_arr[..., None] + phi_t)

    spatial_combined = spatial_amplitude * spatial_deriv
    derivative = temporal_part * spatial_combined[None, :]

    tangent = np.stack(
        (
            np.ones_like(derivative),
            derivative,
            np.zeros_like(derivative),
        ),
        axis=-1,
    )

    tangent /= np.linalg.norm(
        tangent,
        axis=-1,
        keepdims=True,
    )

    animation: Animation = {}

    bind_positions = np.stack(
        (
            distances,
            np.zeros_like(distances),
            np.zeros_like(distances),
        ),
        axis=-1,
    )

    for time, frame_tangent in zip(
        np.atleast_1d(time_stamps),
        tangent,
    ):
        frame = [
            (0.0, 0.0, 0.0),
        ]

        for index in range(1, len(ceil_distances)):
            seg_length = ceil_distances[index] - ceil_distances[index - 1]

            previous_position = np.asarray(
                frame[-1],
                dtype=np.float64,
            )

            direction = frame_tangent[index - 1]

            position = previous_position + seg_length * direction

            frame.append(tuple(position.tolist()))

        y_coords = np.array([p[1] for p in frame])
        mean_y_ceil = np.mean(y_coords)
        shifted_frame = [(p[0], p[1] - mean_y_ceil, p[2]) for p in frame]
        truncated_frame = shifted_frame[: len(distances)]

        animation[float(time)] = successive_rotations(
            bind_positions,
            np.array(truncated_frame),
            is_positions=True,
        )

    return animation


def chain_wave_generator(
    armature: Armature,
    index_bones: list[int],
    wave_amplitude: float,
    wave_duration: float,
    frame_rate: float,
    growth_factor: float = 0,
    num_waves: float = 2.6,
    phi_s: float = 0.0,
    phi_t: float = 0.0,
    axis=2,
    wave: Literal["standing", "travelling"] = "travelling",
) -> Animation:
    """
    Generate a sequence of armature positions representing a wave animation (standing or travelling).

    The wave is evaluated at discrete time steps determined by
    ``frame_rate`` and ``wave_duration``. The resulting sequence can be
    used to create a looping animation of the armature.

    Parameters
    ----------
    armature : Armature
        Armature whose bones are used to generate the wave.

    index_bones : list[int]
        Indices of the bones in ``armature`` to which the wave is applied.
        The bones must form a continuous chain.

    wave_amplitude : float
        Base amplitude ``A`` of the wave at the root of the armature.
        The maximum displacement at the root is ``2 * wave_amplitude`` for standing waves,
        and ``wave_amplitude`` for travelling waves.

    wave_duration : float
        Period ``T`` of the wave in seconds. After one wave duration,
        the temporal component completes one full oscillation, allowing
        the generated animation to loop seamlessly.

    frame_rate : float
        Number of animation frames generated per second.

    growth_factor : float, default=0
        Exponential spatial growth rate ``g`` of the wave amplitude.

        The amplitude at arc length ``s`` is given by::

            A(s) = A * exp(g * s)

        Positive values increase the amplitude toward the end of the
        armature, zero produces a constant amplitude, and negative
        values produce exponential damping.

    num_waves : float, default=2.6
        Number of complete spatial wavelengths across the total armature
        length ``L``. The corresponding wave number is::

            k = 2 * pi * num_waves / L

    phi_s : float, default=0.0
        Spatial phase offset in radians. This shifts the wave along the
        armature without changing its wavelength.

    phi_t : float, default=0.0
        Temporal phase offset in radians. This shifts the starting point
        of the oscillation in time.

    axis : int, default=2
        Spatial axis along which the displacement is applied.
        ``0`` corresponds to X, ``1`` corresponds to Y, and ``2``
        corresponds to Z.

    wave : str, literal, default='travelling'
        Type of wave to generate.
        ``standing`` corresponds to a standing wave.
        ``travelling`` corresponds to a travelling wave.

    Returns
    -------
    Animation
        Mapping from timestamps to animation frames containing local bone rotations.

    Raises
    ------
    ValueError
        If the bones specified by ``index_bones`` do not form a
        continuous chain.

    Notes
    -----
    For travelling waves, the displacement for each bone is calculated using::

        u(s, t) = A exp(g s) sin(k s - omega t + phi_s + phi_t)

    For standing waves, the displacement is calculated using::

        u(s, t) = 2 A exp(g s) cos(k s + phi_s) sin(omega t + phi_t)

    where the spatial wave number ``k`` and temporal angular frequency
    ``omega`` are defined as::

        k = 2 * pi * N / L

        omega = 2 * pi / T

    ``N`` is the number of spatial waves, ``L`` is the total armature
    length, and ``T`` is ``wave_duration``.

    The temporal component is periodic with period ``T``, so evaluating
    the function over integer multiples of ``wave_duration`` produces
    the same wave state and allows the animation to loop seamlessly.
    """

    if not _check_armature_chain(armature, index_bones):
        raise ValueError("Bones in index_bones do not form a continuous chain.")

    distances = [0.0]
    for bone_idx in index_bones:
        bone = armature.bones_list[bone_idx]
        bone_len = np.linalg.norm(np.array(bone.tail) - np.array(bone.head))
        distances.append(distances[-1] + bone_len)

    total_timestamps = int(frame_rate * wave_duration)
    time_stamps = np.linspace(0, wave_duration, num=total_timestamps, endpoint=False)

    if wave == "standing":
        return _standing_wave_generator(
            distances=distances,
            wave_amplitude=wave_amplitude,
            wave_duration=wave_duration,
            time_stamps=time_stamps,
            growth_factor=growth_factor,
            num_waves=num_waves,
            phi_s=phi_s,
            phi_t=phi_t,
        )
    elif wave == "travelling":
        return _travelling_wave_generator(
            distances=distances,
            wave_amplitude=wave_amplitude,
            wave_duration=wave_duration,
            time_stamps=time_stamps,
            growth_factor=growth_factor,
            num_waves=num_waves,
            phi_s=phi_s,
            phi_t=phi_t,
        )
    else:
        raise ValueError(
            f"Invalid wave type: {wave}, Permitted types: {'standing', 'travelling'}"
        )
