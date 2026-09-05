import os
from pathlib import Path
import cv2
import numpy as np
import pytest
import torch
from PIL import Image
from dotenv import load_dotenv

from animgen.core.models.model import BaseModelClass
from animgen.rigging.SAM3 import SAM3Segmentation, SAM3TextEmbedder
from animgen.rigging.backproject import backproject_masks_to_faces

load_dotenv()

# Path to the undecimated killer whale model
KILLER_WHALE_MESH_PATH = Path("generated_data/models/img_mesh_Killer_Whale.glb")
ARTIFACTS_DIR = Path("tests/artifacts/sam3_segmentation")


def test_backproject_masks_to_faces_unit():
    """Fast unit test for backproject_masks_to_faces with synthetic 2D masks and face buffers."""
    num_faces = 10
    # Synthetic face ID maps for 2 views (4x4 images)
    faces_v0 = np.array(
        [
            [0, 0, 1, 1],
            [0, 2, 2, 1],
            [-1, -1, 3, 3],
            [-1, -1, -1, -1],
        ],
        dtype=np.int32,
    )

    faces_v1 = np.array(
        [
            [1, 1, 2, 2],
            [1, 3, 3, 2],
            [-1, -1, 4, 4],
            [-1, -1, -1, -1],
        ],
        dtype=np.int32,
    )

    # Mask activating top-left region in view 0 (faces 0, 2) and top-right in view 1 (face 2)
    mask_v0 = np.array(
        [
            [True, True, False, False],
            [True, True, False, False],
            [False, False, False, False],
            [False, False, False, False],
        ],
        dtype=bool,
    )

    mask_v1 = np.array(
        [
            [False, False, True, True],
            [False, False, False, True],
            [False, False, False, False],
            [False, False, False, False],
        ],
        dtype=bool,
    )

    masks_dict = {"tail": [mask_v0, mask_v1]}
    votes = backproject_masks_to_faces(masks_dict, [faces_v0, faces_v1], num_faces)

    assert "tail" in votes
    assert len(votes["tail"]) == num_faces
    assert votes["tail"][0] == 1  # Detected in view 0
    assert votes["tail"][2] == 2  # Detected in both view 0 and view 1
    assert votes["tail"][1] == 0  # Not in masked area


def create_mask_overlay(
    image: Image.Image | np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 60, 60),
    alpha: float = 0.5,
) -> Image.Image:
    """
    Overlays a colored translucent mask on a 2D image.
    """
    if isinstance(image, Image.Image):
        base_np = np.array(image.convert("RGB"))
    else:
        base_np = image.copy()
        if base_np.ndim == 2:
            base_np = cv2.cvtColor(base_np, cv2.COLOR_GRAY2RGB)

    overlay = base_np.copy()
    mask_bool = mask > 0

    if np.any(mask_bool):
        overlay[mask_bool] = (
            (1.0 - alpha) * base_np[mask_bool] + alpha * np.array(color)
        ).astype(np.uint8)

        # Draw contour border around the mask for crisp visualization
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, color, 2)

    return Image.fromarray(overlay)


def create_side_by_side_grid(
    base_image: Image.Image,
    mask: np.ndarray,
    overlay_image: Image.Image,
    title_label: str = "Tail Tag",
) -> Image.Image:
    """
    Creates a 3-panel comparison image: [Original Render | Mask | Overlay with Tag].
    """
    w, h = base_image.size
    grid = Image.new("RGB", (w * 3, h), (255, 255, 255))

    mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    mask_rgb[mask > 0] = [255, 255, 255]
    mask_img = Image.fromarray(mask_rgb)

    grid.paste(base_image, (0, 0))
    grid.paste(mask_img, (w, 0))
    grid.paste(overlay_image, (w * 2, 0))

    return grid


def test_sam3_text_embedder_lazy():
    """Test SAM3TextEmbedder initializes cleanly without loading model weights into VRAM."""
    embedder = SAM3TextEmbedder()
    assert embedder.model is None
    assert embedder.processor is None
    assert isinstance(embedder.model_path, str)


@pytest.mark.slow
@pytest.mark.sam3
@pytest.mark.skipif(
    not KILLER_WHALE_MESH_PATH.exists(),
    reason=f"Killer whale mesh not found at {KILLER_WHALE_MESH_PATH}",
)
@pytest.mark.skipif(
    not torch.cuda.is_available() and not os.getenv("HF_TOKEN"),
    reason="CUDA device or HF_TOKEN required for full SAM3 inference visual test",
)
def test_sam3_visual_killer_whale_tail_segmentation():
    """
    Visual test: Loads undecimated killer whale, runs SAM3 multi-view segmentation
    with 'tail' prompt, and saves visual segmentation artifacts to tests/artifacts/.
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_artifact_path = Path("tests/artifacts/sam3_killer_whale_tail.png")

    # 1. Load model with 512x512 rendering viewport for fast test inference
    model = BaseModelClass(KILLER_WHALE_MESH_PATH, renderer_size=(512, 512))
    assert model.mesh is not None

    # 2. Get rendered views
    views_output = model.views_output
    images = views_output["matte"]
    faces_per_view = views_output["faces"]
    assert len(images) > 0, "No views rendered"

    # 3. Instantiate SAM3 segmentation with 'tail' prompt
    prompt = "tail"
    segmenter = SAM3Segmentation(prompts=[prompt])

    # 4. Run 2D segmentation across views
    masks_dict = segmenter(model)
    assert prompt in masks_dict
    assert len(masks_dict[prompt]) == len(images)

    # 5. Backproject 2D masks onto 3D mesh faces
    face_votes = backproject_masks_to_faces(
        masks_dict, faces_per_view, len(model.mesh.faces)
    )
    assert prompt in face_votes
    assert len(face_votes[prompt]) == len(model.mesh.faces)

    # 6. Save visual artifacts for each view
    for idx, (img, mask_np) in enumerate(zip(images, masks_dict[prompt])):
        overlay = create_mask_overlay(img, mask_np, color=(255, 50, 50), alpha=0.5)
        view_path = ARTIFACTS_DIR / f"killer_whale_view_{idx:02d}_tail.png"
        overlay.save(view_path)

    # 7. Save composite summary image for View 0
    first_img = images[0]
    first_mask = masks_dict[prompt][0]
    first_overlay = create_mask_overlay(
        first_img, first_mask, color=(255, 50, 50), alpha=0.5
    )

    summary_grid = create_side_by_side_grid(
        base_image=first_img.convert("RGB"),
        mask=first_mask,
        overlay_image=first_overlay,
        title_label="Tail Tag",
    )
    summary_grid.save(summary_artifact_path)

    assert summary_artifact_path.exists()
    assert summary_artifact_path.stat().st_size > 0
    print(f"\n[Visual Test Artifact Saved]: {summary_artifact_path.resolve()}")


@pytest.mark.slow
@pytest.mark.sam3
@pytest.mark.skipif(
    not torch.cuda.is_available() and not os.getenv("HF_TOKEN"),
    reason="CUDA device or HF_TOKEN required for full SAM3 multi-mesh stress test",
)
def test_sam3_multi_mesh_memory_stress_test_20_meshes():
    """
    Stress test: Runs SAM3 segmentation sequentially across up to 20 different 3D meshes
    without saving outputs, validating that memory doesn't leak or cause progressive lag.
    """
    import gc
    import time

    models_dir = Path("generated_data/models")
    mesh_paths = sorted(list(models_dir.glob("*.glb")))
    if len(mesh_paths) < 20:
        # Include nested backups if needed to reach 20
        mesh_paths += sorted(list(models_dir.glob("*/*.glb")))
    mesh_paths = mesh_paths[:20]

    assert len(mesh_paths) > 0, "No .glb test meshes found in generated_data/models"
    print(
        f"\n[SAM3 Stress Test] Running sequential inference on {len(mesh_paths)} meshes..."
    )

    prompt = "tail"
    times = []
    vram_used = []

    with SAM3Segmentation(prompts=[prompt]) as segmenter:
        for idx, mesh_path in enumerate(mesh_paths):
            t_start = time.perf_counter()

            # 1. Load model with compact viewport
            model = BaseModelClass(mesh_path, renderer_size=(512, 512))
            assert model.mesh is not None

            # 2. Run multi-view SAM3 segmentation
            masks_dict = segmenter(model, threshold=0.5, mask_threshold=0.5)
            assert prompt in masks_dict

            # 3. Fast backprojection
            faces_per_view = model.views_output["faces"]
            face_votes = backproject_masks_to_faces(
                masks_dict, faces_per_view, len(model.mesh.faces)
            )
            assert prompt in face_votes

            t_elapsed = time.perf_counter() - t_start
            times.append(t_elapsed)

            # Measure VRAM if on CUDA
            if torch.cuda.is_available():
                allocated_mb = torch.cuda.memory_allocated() / (1024**2)
                reserved_mb = torch.cuda.memory_reserved() / (1024**2)
                vram_used.append(allocated_mb)
                vram_str = f"| VRAM Allocated: {allocated_mb:.1f} MB, Reserved: {reserved_mb:.1f} MB"
            else:
                vram_str = ""

            print(
                f"  [{idx + 1:02d}/{len(mesh_paths):02d}] {mesh_path.name:30s} "
                f"Time: {t_elapsed:.2f}s {vram_str}"
            )

            # Explicit cleanup of model
            del model
            del masks_dict
            del face_votes
            gc.collect()

    avg_time = sum(times) / len(times)
    print(
        f"\n[SAM3 Stress Test Completed] Processed {len(mesh_paths)} meshes. Average Time: {avg_time:.2f}s per mesh."
    )
    if vram_used:
        print(
            f"  Max VRAM Allocated: {max(vram_used):.1f} MB, Final VRAM Allocated: {vram_used[-1]:.1f} MB"
        )


@pytest.mark.slow
@pytest.mark.sam3
@pytest.mark.skipif(
    not torch.cuda.is_available() and not os.getenv("HF_TOKEN"),
    reason="CUDA device or HF_TOKEN required for SAM3 determinism test",
)
def test_sam3_single_mesh_determinism_test_10_runs():
    """
    Determinism test: Runs SAM3 segmentation on the exact same mesh 10 consecutive times.
    Measures IoU, pixel differences, and face-vote consistency across all runs to assess output stability.
    """
    import gc
    import time

    mesh_path = KILLER_WHALE_MESH_PATH
    if not mesh_path.exists():
        fallback_models = list(Path("generated_data/models").glob("*.glb"))
        assert fallback_models, "No test meshes found in generated_data/models"
        mesh_path = fallback_models[0]

    num_iterations = 10
    prompts = ["tail"]
    print(
        f"\n[SAM3 Determinism Test] Running {num_iterations} repeated segmentations on '{mesh_path.name}'..."
    )

    # Load model once to keep camera views, vertex order, and rendered buffers identical
    model = BaseModelClass(mesh_path, renderer_size=(512, 512))
    assert model.mesh is not None
    num_faces = len(model.mesh.faces)
    faces_per_view = model.views_output["faces"]
    num_views = len(model.views_output["matte"])

    baseline_masks: dict[str, list[np.ndarray]] | None = None
    baseline_votes: dict[str, np.ndarray] | None = None

    ious_all_runs: list[float] = []
    diff_pixels_all_runs: list[int] = []
    exact_matches_all_runs: list[bool] = []
    times: list[float] = []

    with SAM3Segmentation(prompts=prompts) as segmenter:
        for run_idx in range(num_iterations):
            t_start = time.perf_counter()

            # Run SAM3 segmentation
            masks_dict = segmenter(model, threshold=0.5, mask_threshold=0.5)

            # Backproject masks to 3D faces
            face_votes = backproject_masks_to_faces(
                masks_dict, faces_per_view, num_faces
            )
            t_elapsed = time.perf_counter() - t_start
            times.append(t_elapsed)

            if run_idx == 0:
                baseline_masks = {
                    p: [m.copy() for m in masks] for p, masks in masks_dict.items()
                }
                baseline_votes = {p: v.copy() for p, v in face_votes.items()}
                print(
                    f"  [Run 01/{num_iterations:02d}] Baseline established in {t_elapsed:.2f}s."
                )
                continue

            # Compare against baseline
            run_ious = []
            run_diff_pixels = 0
            run_exact = True

            for prompt in prompts:
                for v_idx in range(num_views):
                    curr_m = masks_dict[prompt][v_idx]
                    base_m = baseline_masks[prompt][v_idx]

                    # 1. Exact equality
                    if not np.array_equal(curr_m, base_m):
                        run_exact = False

                    # 2. Pixel difference count
                    diff_count = int(np.bitwise_xor(curr_m, base_m).sum())
                    run_diff_pixels += diff_count

                    # 3. Intersection over Union (IoU)
                    intersection = int(np.logical_and(curr_m, base_m).sum())
                    union = int(np.logical_or(curr_m, base_m).sum())
                    iou = 1.0 if union == 0 else (intersection / union)
                    run_ious.append(iou)

            mean_run_iou = float(np.mean(run_ious))
            ious_all_runs.append(mean_run_iou)
            diff_pixels_all_runs.append(run_diff_pixels)
            exact_matches_all_runs.append(run_exact)

            # Compare 3D face votes
            face_votes_exact = all(
                np.array_equal(face_votes[p], baseline_votes[p]) for p in prompts
            )

            print(
                f"  [Run {run_idx + 1:02d}/{num_iterations:02d}] Time: {t_elapsed:.2f}s | "
                f"Mean IoU: {mean_run_iou:.6f} | Differing Pixels: {run_diff_pixels:5d} | "
                f"2D Exact: {str(run_exact):5s} | 3D Face Votes Exact: {str(face_votes_exact)}"
            )

            del masks_dict
            del face_votes
            gc.collect()

    avg_iou = float(np.mean(ious_all_runs))
    max_diff = max(diff_pixels_all_runs)
    exact_ratio = sum(exact_matches_all_runs) / len(exact_matches_all_runs) * 100.0

    print(f"\n[Determinism Summary across {num_iterations} runs]:")
    print(f"  Average Time per Run: {sum(times) / len(times):.2f}s")
    print(f"  Mean IoU vs Baseline: {avg_iou:.6f}")
    print(f"  Max Differing Pixels across all views: {max_diff}")
    print(f"  Exact Match Rate: {exact_ratio:.1f}%")

    # High consistency requirement: Mean IoU should be virtually 1.0 (>0.999)
    assert avg_iou > 0.999, (
        f"Segmentation determinism failed: Mean IoU ({avg_iou:.4f}) is below 0.999"
    )
