import torch
import numpy as np

from animgen.core.armature import Armature
from animgen.core.spline import Spline


def test_spline_generation_and_evaluation():
    """Test Spline parameterization, evaluation, and duplicate checks."""
    pts = [
        torch.tensor([0.0, 0.0, 0.0]),
        torch.tensor([1.0, 0.0, 0.0]),
        torch.tensor([2.0, 1.0, 0.0]),
        torch.tensor([3.0, 1.0, 1.0]),
    ]

    spline = Spline(pts, alpha=0.5)
    assert len(spline.points) == 4

    # Check evaluation
    eval_pts = spline.evaluate_curve(num_points_per_segment=10)
    assert len(eval_pts) > 4

    # Check duplicate filtering
    pts_with_dups = [
        torch.tensor([0.0, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 0.0]),  # Duplicate
        torch.tensor([1.0, 0.0, 0.0]),
        torch.tensor([2.0, 1.0, 0.0]),
        torch.tensor([3.0, 1.0, 1.0]),
    ]
    spline_filtered = Spline(pts_with_dups)
    assert len(spline_filtered.points) == 4


def test_spline_armature_generation():
    """Test generating armature hierarchy along spline paths."""
    pts = [
        torch.tensor([0.0, 0.0, 0.0]),
        torch.tensor([1.0, 0.0, 0.0]),
        torch.tensor([2.0, 0.0, 0.0]),
        torch.tensor([3.0, 0.0, 0.0]),
    ]
    spline = Spline(pts, alpha=0.5)

    num_bones = 3
    armature = spline.get_armature(num_bone=num_bones)
    assert isinstance(armature, Armature)
    assert len(armature.bones_list) == num_bones

    # Validate tail coordinate matching the curve end point
    assert np.allclose(armature.bones_list[-1].tail, (3.0, 0.0, 0.0))
