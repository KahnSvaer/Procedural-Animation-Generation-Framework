"""
Module for standing wave generation along an armature.

Reference:
- https://en.wikipedia.org/wiki/Standing_wave


"""

import numpy as np

from animgen.core.armature import Armature

from numpy.typing import NDArray


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


def _standing_wave_calculator(
    distances: list[float] | NDArray[np.float64],
    wave_amplitude: float,
    wave_duration: float,
    time_stamps: float | list[float] | NDArray[np.float64],
    growth_factor: float = 0.0,
    num_waves: float = 2.6,
    phi_s: float = 0.0,
    phi_t: float = 0.0,
) -> NDArray[np.float64]:
    """
    Evaluate a standing wave at specified spatial distances and timestamps.

    The displacement is modeled as:

        u(s, t) = 2 A exp(g s)
                  cos(k s + phi_s)
                  sin(omega t + phi_t)

    where the spatial wave number ``k`` and temporal angular frequency
    ``omega`` are defined as:

        k = 2 pi N / L

        omega = 2 pi / T

    Here, ``N`` is the number of spatial waves, ``L`` is the total
    spatial length, and ``T`` is the temporal period of the wave.

    Parameters
    ----------
    distances : list[float] | NDArray[np.float64]
        Spatial distances ``s`` at which to evaluate the wave. The
        distances are measured from the root of the armature.

    wave_amplitude : float
        Base amplitude ``A`` of the wave at ``s = 0``. The maximum
        displacement is ``2 * wave_amplitude`` before applying the
        spatial growth factor.

    wave_duration : float
        Temporal period ``T`` of the wave in seconds. The wave completes
        one full oscillation every ``wave_duration`` seconds.

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
    NDArray[np.float64]
        Wave displacement evaluated at every combination of timestamp
        and spatial distance.

        For multiple timestamps, the returned array has shape
        ``(n_timestamps, n_distances)``.

        For a single timestamp, the returned array has shape
        ``(1, n_distances)``.
    """

    distances = np.asarray(distances, dtype=np.float64)
    time_stamps = np.asarray(time_stamps, dtype=np.float64)

    total_length = distances[-1]

    wave_number = 2 * np.pi * num_waves / total_length
    angular_frequency = 2 * np.pi / wave_duration

    spatial_part = (
        2
        * wave_amplitude
        * np.exp(growth_factor * distances)
        * np.cos(wave_number * distances + phi_s)
    )

    temporal_part = np.sin(angular_frequency * time_stamps + phi_t)

    if time_stamps.ndim == 0:
        return spatial_part[None, :] * temporal_part

    return temporal_part[:, None] * spatial_part[None, :]


def standing_wave_generator(
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
):
    """
    Generate a sequence of armature positions representing a standing wave.

    The standing wave is evaluated at discrete time steps determined by
    ``frame_rate`` and ``wave_duration``. The resulting sequence can be
    used to create a looping animation of the armature.

    Parameters
    ----------
    armature : Armature
        Armature whose bones are used to generate the standing wave.

    index_bones : list[int]
        Indices of the bones in ``armature`` to which the wave is applied.
        The bones must form a continuous chain.

    wave_amplitude : float
        Base amplitude ``A`` of the wave at the root of the armature.
        The maximum displacement at the root is ``2 * wave_amplitude``.

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

    Returns
    -------
    ...
        A sequence containing the generated armature positions for each
        animation frame.

    Raises
    ------
    ValueError
        If the bones specified by ``index_bones`` do not form a
        continuous chain.

    Notes
    -----
    The displacement for each bone is calculated using the standing-wave
    equation::

        u(s, t) = 2 A exp(g s)
                  cos(k s + phi_s)
                  sin(omega t + phi_t)

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

    pass

    # if not _check_armature_chain(armature, index_bones):
    #     raise ValueError("Bones in index_bones do not form a continuous chain.")

    # total_timestamps = int(frame_rate * wave_duration)
    # frame_timestamps = np.linspace(
    #     0, wave_duration, num=total_timestamps, endpoint=False
    # )

    # TODO: complete here
