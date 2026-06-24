"""
Camera Position Generators to experiment with SAM mesh segmentation later to choose which works better overall

Reference: https://github.com/gtangg12/samesh/blob/main/src/samesh/utils/polyhedra.py
"""

import numpy as np


def _build_position_registry():
    return {
        name.removeprefix("pos_"): obj
        for name, obj in globals().items()
        if callable(obj) and name.startswith("pos_")
    }
POSITION_GENERATORS = _build_position_registry()


def _golden_ratio():
    return (1 + np.sqrt(5)) / 2


def pos_tetrahedron():
    return np.array([
        [ 1,  1,  1],
        [-1, -1,  1],
        [-1,  1, -1],
        [ 1, -1, -1],
    ])


def pos_octahedron():
    return np.array([
        [ 1,  0,  0],
        [ 0,  0,  1],
        [-1,  0,  0],
        [ 0,  0, -1],
        [ 0,  1,  0],
        [ 0, -1,  0],
    ])


def pos_cube():
    return np.array([
        [ 1,  1,  1],
        [-1,  1,  1],
        [-1, -1,  1],
        [ 1, -1,  1],
        [ 1,  1, -1],
        [-1,  1, -1],
        [-1, -1, -1],
        [ 1, -1, -1],
    ])


def pos_icosahedron():
    phi = _golden_ratio()
    return np.array([
        [-1,  phi,  0],
        [-1, -phi,  0],
        [ 1,  phi,  0],
        [ 1, -phi,  0],
        [ 0, -1,  phi],
        [ 0,  1,  phi],
        [ 0, -1, -phi],
        [ 0,  1, -phi],
        [ phi,  0, -1],
        [ phi,  0,  1],
        [-phi,  0, -1],
        [-phi,  0,  1],
    ]) / np.sqrt(1 + phi ** 2)


def pos_dodecahedron():
    phi = _golden_ratio()
    a, b = 1 / phi, 1 / (phi * phi)
    return np.array([
        [-a, -a,  b], [ a, -a,  b], [ a,  a,  b], [-a,  a,  b],
        [-a, -a, -b], [ a, -a, -b], [ a,  a, -b], [-a,  a, -b],
        [ b, -a, -a], [ b,  a, -a], [ b,  a,  a], [ b, -a,  a],
        [-b, -a, -a], [-b,  a, -a], [-b,  a,  a], [-b, -a,  a],
        [-a,  b, -a], [ a,  b, -a], [ a,  b,  a], [-a,  b,  a],
    ]) / np.sqrt(a ** 2 + b ** 2)


def pos_ring(n=8, elevation=15):
    pphi =  elevation * np.pi / 180
    nphi = -elevation * np.pi / 180
    coords = []
    for phi in [pphi, nphi]:
        for theta in np.linspace(0, 2 * np.pi, n, endpoint=False):
            coords.append([
                np.cos(theta) * np.cos(phi),
                np.sin(phi),
                np.sin(theta) * np.cos(phi),
            ])
    return np.array(coords)


def pos_sphere(
    n=8,
    elevation_list=(-30, 0, 30)
):
    coords = []
    for elevation in elevation_list:
        coords.append(pos_ring(n, elevation))
    return np.concatenate(coords, axis=0)


def pos_swirl(
        n=120, 
        cycles=1, 
        elevation_range=(-45, 60)
):
    pphi = elevation_range[0] * np.pi / 180
    nphi = elevation_range[1] * np.pi / 180
    thetas = np.linspace(0, 2 * np.pi, n, endpoint=False)
    coords = []
    for i, phi in enumerate(np.linspace(pphi, nphi, n)):
        coords.append([
            np.cos(cycles * thetas[i]) * np.cos(phi),
            np.sin(phi),
            np.sin(cycles * thetas[i]) * np.cos(phi),
        ])
    return np.array(coords)
