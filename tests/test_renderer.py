import pytest
import trimesh
import numpy as np
from pathlib import Path
from PIL import Image

from animgen.renderer.renderer import Renderer, render_multiview


@pytest.fixture(scope="module")
def render_output():
    """Fixture providing rendered multiview outputs for test functions."""
    mesh = trimesh.creation.cylinder(radius=1.0, height=4.0, sections=16)

    renderer = Renderer(viewport_width=256, viewport_height=256)
    renderer.set_object(mesh)
    renderer.set_camera()

    expected_num_views = 20
    results = render_multiview(
        renderer,
        camera_generation_method="random_sphere",
        renderer_args={},
        sampling_args={"n": expected_num_views, "radius": 6.0},
        verbose=False,
    )
    return results, expected_num_views


def test_renderer_multiview_output_count(render_output):
    """Test that render_multiview generates the expected number of view images."""
    results, expected_num_views = render_output
    mattes = results.get("matte", [])
    assert len(mattes) == expected_num_views, (
        f"Expected {expected_num_views} images, got {len(mattes)}"
    )


def test_renderer_multiview_image_content(render_output):
    """Test that rendered view images are not completely black or white and have valid contrast."""
    results, _ = render_output
    mattes = results.get("matte", [])

    for idx, img_data in enumerate(mattes):
        img_arr = (
            np.array(img_data)
            if isinstance(img_data, Image.Image)
            else np.asarray(img_data)
        )
        mean_val = float(np.mean(img_arr))
        std_val = float(np.std(img_arr))

        assert mean_val > 1.0, f"View {idx} is completely black (mean={mean_val:.2f})"
        assert mean_val < 254.0, f"View {idx} is completely white (mean={mean_val:.2f})"
        assert std_val > 1.0, f"View {idx} lacks contrast (std={std_val:.2f})"


def test_renderer_save_view_artifacts(render_output):
    """Test saving rendered preview views inside tests/artifacts/render_views/."""
    results, _ = render_output
    mattes = results.get("matte", [])

    artifact_dir = Path("tests/artifacts/render_views")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    for idx, img_data in enumerate(mattes):
        img_path = artifact_dir / f"rendered_view_{idx:02d}.png"
        if isinstance(img_data, Image.Image):
            img_data.save(str(img_path))
        else:
            Image.fromarray(np.asarray(img_data).astype(np.uint8)).save(str(img_path))

        assert img_path.exists(), f"Failed to save artifact {img_path}"
        assert img_path.stat().st_size > 0, f"Artifact {img_path} is empty"
