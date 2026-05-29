from pathlib import Path
import trimesh
import numpy as np
from PIL import Image

from animgen.render_views import render_views


TEST_MODEL_PATH = "assets/models/goldfish_untex.glb"
TEST_DIR_PATH = "tests/artifacts/"

def test_model_exists():
    assert Path(TEST_MODEL_PATH).exists(), f"Test model not found at {TEST_MODEL_PATH}"

def test_render_views_returns_expected_views():
    mesh = trimesh.load_mesh(TEST_MODEL_PATH)

    test_vals = [
        {"hori": 8, "vert": [-30, 0, 30]},
        {"hori": 4, "vert": [0]},
        {"hori": 16, "vert": [-45, 0]}
    ]
    for vals in test_vals:
        num_views = vals["hori"]
        vertical_angles = vals["vert"]
        views, _ = render_views(mesh, num_views=num_views, vertical_angles=vertical_angles)
        assert len(views) == num_views * len(vertical_angles), f"Expected {num_views * len(vertical_angles)} views, got {len(views)}"
        for view in views:
            assert isinstance(view, np.ndarray)

def test_render_views_not_blank():
    mesh = trimesh.load_mesh(TEST_MODEL_PATH)
    
    views, _ = render_views(
        mesh,
        num_views=4,
        vertical_angles=[0]
    )

    for view in views:
        assert view.mean() < 253 # ie not blank white
        assert view.mean() > 5 # ie not blank black
        assert np.std(view) > 0

def test_render_views():
    mesh = trimesh.load_mesh(TEST_MODEL_PATH)

    views, _ = render_views(
        mesh,
        num_views=4,
        vertical_angles=[0]
    )

    output_dir = Path(TEST_DIR_PATH) / "render_views_test"
    output_dir.mkdir(exist_ok=True, parents=True)

    for i, view in enumerate(views):
        image = Image.fromarray(view)
        image.save(output_dir / f"view_{i}.png")
        