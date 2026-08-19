import numpy as np
import pytest
from animgen.animation.wave import (
    _travelling_wave_generator,
    _standing_wave_generator,
    _pulse_wave_generator,
    chain_wave_generator,
)
from animgen.core.armature import Armature, Bone


def reconstruct_positions(distances, rotations):
    bind_positions = np.stack(
        [
            distances,
            np.zeros_like(distances),
            np.zeros_like(distances),
        ],
        axis=-1,
    )
    bind_vectors = np.diff(bind_positions, axis=0)

    positions = [np.array([0.0, 0.0, 0.0])]
    R_accum = np.eye(3)
    for i, R_local in enumerate(rotations):
        if hasattr(R_local, "detach"):
            R_local = R_local.detach().cpu().numpy()
        R_accum = R_local @ R_accum
        positions.append(positions[-1] + R_accum @ bind_vectors[i])

    positions = np.array(positions)
    positions[:, 1] = positions[:, 1] - np.mean(positions[:, 1])
    return positions


@pytest.mark.parametrize(
    "wave_generator",
    [_travelling_wave_generator, _standing_wave_generator, _pulse_wave_generator],
)
def test_bone_length_conservation(wave_generator):
    """Verify that bone lengths are perfectly conserved (rigid bones) across all frames."""
    distances = np.linspace(0.0, 5.0, 11)  # 10 segments of length 0.5 each
    wave_amplitude = 0.4
    wave_duration = 2.0
    time_stamps = [0.0, 0.5, 1.0, 1.5, 2.0]

    animation = wave_generator(
        distances=distances,
        wave_amplitude=wave_amplitude,
        wave_duration=wave_duration,
        time_stamps=time_stamps,
        growth_factor=0.1,
        num_waves=1.5,
        phi_s=0.2,
        phi_t=-0.4,
    )

    expected_lengths = np.diff(distances)

    for t, frame in animation.items():
        actual_lengths = []
        pts = reconstruct_positions(distances, frame)
        for i in range(1, len(pts)):
            p1 = pts[i - 1]
            p2 = pts[i]
            actual_lengths.append(np.linalg.norm(p2 - p1))

        np.testing.assert_allclose(actual_lengths, expected_lengths, atol=1e-12)


@pytest.mark.parametrize(
    "wave_generator",
    [_travelling_wave_generator, _standing_wave_generator],
)
def test_wave_periodicity(wave_generator):
    """Verify that the wave is periodic over one wave duration (loopability)."""
    distances = np.linspace(0.0, 5.0, 20)
    wave_amplitude = 0.4
    wave_duration = 2.0
    t1 = 0.3
    t2 = t1 + wave_duration

    animation = wave_generator(
        distances=distances,
        wave_amplitude=wave_amplitude,
        wave_duration=wave_duration,
        time_stamps=[t1, t2],
        growth_factor=0.1,
        num_waves=1.5,
        phi_s=0.0,
        phi_t=0.0,
    )

    frame1 = reconstruct_positions(distances, animation[t1])
    frame2 = reconstruct_positions(distances, animation[t2])

    np.testing.assert_allclose(frame1, frame2, atol=1e-12)


@pytest.mark.parametrize(
    "wave_generator",
    [_travelling_wave_generator, _standing_wave_generator, _pulse_wave_generator],
)
def test_wave_spatial_growth(wave_generator):
    """Verify that spatial growth factor properly scales the wave amplitude along the armature."""
    distances = np.linspace(0.0, 5.0, 20)
    wave_amplitude = 0.4
    wave_duration = 2.0
    time_stamps = np.linspace(0.0, 2.0, 20)

    # 1. No growth (growth_factor = 0.0)
    anim_no_growth = wave_generator(
        distances=distances,
        wave_amplitude=wave_amplitude,
        wave_duration=wave_duration,
        time_stamps=time_stamps,
        growth_factor=0.0,
        num_waves=1.5,
    )

    # 2. Positive growth (growth_factor = 0.2)
    anim_growth = wave_generator(
        distances=distances,
        wave_amplitude=wave_amplitude,
        wave_duration=wave_duration,
        time_stamps=time_stamps,
        growth_factor=0.2,
        num_waves=1.5,
    )

    max_y_no_growth = np.max(
        [
            np.max(np.abs(reconstruct_positions(distances, f)[:, 1]))
            for f in anim_no_growth.values()
        ]
    )
    max_y_growth = np.max(
        [
            np.max(np.abs(reconstruct_positions(distances, f)[:, 1]))
            for f in anim_growth.values()
        ]
    )

    # Amplitude with growth factor > 0 should be larger due to exponential scaling
    assert max_y_growth > max_y_no_growth


@pytest.mark.parametrize(
    "wave_generator",
    [_travelling_wave_generator, _standing_wave_generator, _pulse_wave_generator],
)
def test_wave_single_timestamp(wave_generator):
    """Verify that passing a single timestamp works correctly and returns one frame."""
    distances = np.linspace(0.0, 5.0, 11)
    wave_amplitude = 0.4
    wave_duration = 2.0
    time_stamp = 0.5

    animation = wave_generator(
        distances=distances,
        wave_amplitude=wave_amplitude,
        wave_duration=wave_duration,
        time_stamps=time_stamp,
        growth_factor=0.1,
        num_waves=1.5,
    )

    assert len(animation) == 1
    assert time_stamp in animation
    assert len(animation[float(time_stamp)]) == len(distances) - 1


@pytest.mark.parametrize("wave_type", ["standing", "travelling", "pulse"])
def test_chain_wave_generator(wave_type):
    """Verify that chain_wave_generator builds the animation with the correct shape and keys."""
    # Build a simple mock armature with a single line of bones
    # 3 bones along the x-axis: [0, 0, 0] -> [1, 0, 0] -> [2, 0, 0] -> [3, 0, 0]
    b1 = Bone(head=(0.0, 0.0, 0.0), tail=(1.0, 0.0, 0.0))
    armature = Armature(b1)
    b2 = armature.add_connected_bone(b1, tail=(2.0, 0.0, 0.0))
    _ = armature.add_connected_bone(b2, tail=(3.0, 0.0, 0.0))

    index_bones = [0, 1, 2]

    wave_duration = 2.0
    frame_rate = 10.0  # 10 fps -> 20 frames total

    animation = chain_wave_generator(
        armature=armature,
        index_bones=index_bones,
        wave_amplitude=0.2,
        wave_duration=wave_duration,
        frame_rate=frame_rate,
        wave=wave_type,
    )

    assert len(animation) == int(frame_rate * wave_duration)
    for t, frame in animation.items():
        # There are 3 bones, so we expect 3 rotation matrices in each frame
        assert len(frame) == 3
        for R in frame:
            assert R.shape == (3, 3)
            # Rotation matrices should be orthogonal
            np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)


def test_pulse_wave_propagation():
    """Verify that pulse envelope propagates in the positive x direction over time."""
    distances = np.linspace(0.0, 10.0, 101)
    wave_amplitude = 0.5
    wave_duration = 4.0
    num_waves = 1.0

    t1 = 0.5
    t2 = 1.5

    animation = _pulse_wave_generator(
        distances=distances,
        wave_amplitude=wave_amplitude,
        wave_duration=wave_duration,
        time_stamps=[t1, t2],
        num_waves=num_waves,
        pulse_width=1.5,
    )

    pts_t1 = reconstruct_positions(distances, animation[t1])
    pts_t2 = reconstruct_positions(distances, animation[t2])

    # Find position of max absolute y-displacement (envelope peak location)
    peak_x_t1 = pts_t1[np.argmax(np.abs(pts_t1[:, 1])), 0]
    peak_x_t2 = pts_t2[np.argmax(np.abs(pts_t2[:, 1])), 0]

    # As time progresses from t1 to t2, peak should move in positive x direction
    assert peak_x_t2 > peak_x_t1
