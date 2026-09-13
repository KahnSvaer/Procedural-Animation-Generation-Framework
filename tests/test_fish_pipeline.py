from pathlib import Path
import numpy as np
import pytest
import trimesh

from animgen.core.models.model import BaseModelClass
from animgen.core.models.fish import FishModels
from animgen.core.armature import Armature
from animgen.animation.animator import Animator


def create_synthetic_fish_mesh() -> trimesh.Trimesh:
    """
    Creates a synthetic tapered fish-shaped mesh aligned along the X-axis:
    - Snout at negative X
    - Caudal tail at positive X
    - Tapered ends with a wider central torso
    """
    cylinder = trimesh.creation.cylinder(radius=0.3, height=3.0, sections=24)
    rot = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
    cylinder.apply_transform(rot)

    verts = cylinder.vertices.copy()
    x = verts[:, 0]
    profile = np.maximum(0.2, 1.0 - 0.7 * (x / 1.5) ** 2)
    verts[:, 1] *= profile
    verts[:, 2] *= profile
    cylinder.vertices = verts
    return cylinder


def test_fish_pipeline_on_synthetic_mesh():
    """
    Test FishModels end-to-end on a synthetic fish mesh.
    """
    mesh = create_synthetic_fish_mesh()
    model = BaseModelClass(mesh)
    assert model.mesh is not None

    pipeline = FishModels(
        model=model,
        num_bones=10,
        num_tail_bones=2,
        rig_tail=True,
        straighten_mesh=False,
        frame_rate=10.0,
    )

    # Execute end-to-end pipeline
    processed_model = pipeline.process()

    assert processed_model.mesh is not None
    assert isinstance(processed_model.armature, Armature)
    assert len(processed_model.armature.bones_list) == 10

    # Verify bone chain connectivity
    armature = processed_model.armature
    spine_tail_bones = [b for b in armature.bones_list if "pectoral" not in b.id]
    for i in range(len(spine_tail_bones) - 1):
        parent_bone = spine_tail_bones[i]
        child_bone = spine_tail_bones[i + 1]
        assert child_bone.parent == parent_bone
        np.testing.assert_allclose(child_bone.head, parent_bone.tail, atol=1e-5)

    assert any("spine_" in b.id for b in armature.bones_list)
    assert any("tail_" in b.id for b in armature.bones_list)

    assert isinstance(processed_model.animator, Animator)
    assert "swim" in processed_model.animator.animations
    assert "idle" in processed_model.animator.animations
    assert "sprint" in processed_model.animator.animations

    assert processed_model.skin_weights is not None
    assert len(processed_model.skin_weights) == 10

    # Test saving intermediate artifacts
    out_dir = Path("tests/artifacts/synthetic_fish_pipeline")
    saved = pipeline.save_intermediate_artifacts(out_dir)
    assert "04_canonical_mesh" in saved
    assert "05_rigged_armature" in saved
    assert "06_animated_fish" in saved
    assert saved["06_animated_fish"].exists()


def test_tail_orientation_detection():
    """
    Test automatic caudal/tail fin orientation detection on Tuna (vertical) vs Dolphin (horizontal).
    """
    tuna_path = Path("generated_data/models/models_backup_3/dec_mesh_Tuna.glb")
    dolphin_path = Path("generated_data/models/models_backup_3/dec_mesh_Dolphin.glb")

    if tuna_path.exists():
        tuna_model = BaseModelClass(tuna_path)
        tuna_pipe = FishModels(tuna_model)
        tuna_pipe.segment()
        orient_tuna = tuna_pipe._detect_tail_orientation()
        assert orient_tuna == "vertical"

    if dolphin_path.exists():
        dolphin_model = BaseModelClass(dolphin_path)
        dolphin_pipe = FishModels(dolphin_model)
        dolphin_pipe.segment()
        orient_dolphin = dolphin_pipe._detect_tail_orientation()
        assert orient_dolphin == "horizontal"


@pytest.mark.slow
def test_fish_process_and_intermediate_artifacts_on_tuna():
    """
    Test FishModels process() with lateral Bishop straightening and save_intermediate_artifacts on Tuna.
    Saves all diagnostic steps into tests/artifacts/fish_pipeline_tuna/.
    """
    mesh_path = Path("generated_data/models/models_backup_3/paint_mesh_Tuna.glb")
    if not mesh_path.exists():
        mesh_path = Path("generated_data/models/paint_mesh_Tuna.glb")
    if not mesh_path.exists():
        mesh_path = Path("generated_data/models/models_backup_3/dec_mesh_Tuna.glb")
    if not mesh_path.exists():
        pytest.skip(f"Tuna mesh not found at {mesh_path}")

    model = BaseModelClass(mesh_path)
    pipeline = FishModels(
        model=model,
        num_bones=12,
        num_tail_bones=2,
        rig_tail=True,
        rig_pectoral_fins=True,
        straighten_mesh=True,
        frame_rate=30.0,
        use_sam=True,
    )

    processed_model = pipeline.process()

    assert processed_model.mesh is not None
    assert isinstance(processed_model.armature, Armature)
    assert len(processed_model.armature.bones_list) == 14

    bone_ids = [b.id for b in processed_model.armature.bones_list]
    assert "left_pectoral_fin_0" in bone_ids
    assert "right_pectoral_fin_0" in bone_ids
    assert "tail_0" in bone_ids

    # Verify tail orientation detection
    assert pipeline.tail_orientation == "vertical"

    # Export all intermediate step artifacts
    out_dir = Path("tests/artifacts/fish_pipeline_tuna")
    saved = pipeline.save_intermediate_artifacts(out_dir)

    assert "01_segmented_mesh" in saved
    assert "02_extracted_1d_spine" in saved
    assert "03_evaluated_spline" in saved
    assert "04_canonical_mesh" in saved
    assert "05_rigged_armature" in saved
    assert "06_animated_fish" in saved

    for name, path in saved.items():
        assert path.exists()
        assert path.stat().st_size > 0
        print(
            f"  [Artifact Saved] {name:25s} -> {path.resolve()} ({path.stat().st_size} bytes)"
        )


@pytest.mark.slow
def test_fish_process_on_shark():
    """
    Test FishModels on Shark (dec_mesh_Shark.glb):
    - Vertical caudal tail
    - Lateral Bishop straightening along Z-axis
    - Lateral XZ travelling wave motion
    """
    mesh_path = Path("generated_data/models/models_backup_3/dec_mesh_Shark.glb")
    if not mesh_path.exists():
        pytest.skip(f"Shark mesh not found at {mesh_path}")

    model = BaseModelClass(mesh_path)
    pipeline = FishModels(
        model=model,
        num_bones=12,
        num_tail_bones=2,
        rig_tail=True,
        rig_pectoral_fins=True,
        straighten_mesh=True,
        frame_rate=30.0,
        use_sam=True,
    )

    processed_model = pipeline.process()

    assert processed_model.mesh is not None
    assert isinstance(processed_model.armature, Armature)
    assert len(processed_model.armature.bones_list) == 14

    assert pipeline.tail_orientation == "vertical"

    # Export intermediate diagnostic artifacts
    out_dir = Path("tests/artifacts/fish_pipeline_shark")
    saved = pipeline.save_intermediate_artifacts(out_dir)
    assert saved["06_animated_fish"].exists()


@pytest.mark.slow
def test_fish_process_on_dolphin():
    """
    Test FishModels on Dolphin (dec_mesh_Dolphin.glb):
    - Horizontal caudal flukes
    - Dorsoventral Bishop straightening along Y-axis
    - Dorsoventral XY travelling wave motion
    """
    mesh_path = Path("generated_data/models/models_backup_3/dec_mesh_Dolphin.glb")
    if not mesh_path.exists():
        pytest.skip(f"Dolphin mesh not found at {mesh_path}")

    model = BaseModelClass(mesh_path)
    pipeline = FishModels(
        model=model,
        num_bones=12,
        num_tail_bones=2,
        rig_tail=True,
        rig_pectoral_fins=True,
        straighten_mesh=True,
        frame_rate=30.0,
        use_sam=True,
    )

    processed_model = pipeline.process()

    assert processed_model.mesh is not None
    assert isinstance(processed_model.armature, Armature)
    assert len(processed_model.armature.bones_list) == 14

    assert pipeline.tail_orientation == "horizontal"

    # Export intermediate diagnostic artifacts
    out_dir = Path("tests/artifacts/fish_pipeline_dolphin")
    saved = pipeline.save_intermediate_artifacts(out_dir)
    assert saved["06_animated_fish"].exists()


def test_side_fin_classification_loosy_vs_straight():
    """
    Test pectoral/side fin classification into 'loosy' (flexible, lies along flank in +X)
    vs 'straight' (rigid/hydrofoil, extends outward in +/-Z away from body).
    - Tuna: loosy (aspect ratio dx/dz >= 2.0, extends primarily along body)
    - Shark: straight (aspect ratio dx/dz < 2.0, extends primarily away from body)
    - Dolphin: straight (aspect ratio dx/dz < 2.0, extends primarily away from body)
    """
    # 1. Tuna (Loosy side fins)
    tuna_path = Path("generated_data/models/models_backup_3/dec_mesh_Tuna.glb")
    if tuna_path.exists():
        tuna_model = BaseModelClass(tuna_path)
        tuna_pipe = FishModels(tuna_model)
        tuna_pipe.segment()
        assert tuna_pipe._classify_side_fin_type("left_pectoral_fin") == "loosy"
        assert tuna_pipe._classify_side_fin_type("right_pectoral_fin") == "loosy"

    # 2. Shark (Straight / rigid side fins)
    shark_path = Path("generated_data/models/models_backup_3/dec_mesh_Shark.glb")
    if shark_path.exists():
        shark_model = BaseModelClass(shark_path)
        shark_pipe = FishModels(shark_model)
        shark_pipe.segment()
        assert shark_pipe._classify_side_fin_type("left_pectoral_fin") == "straight"
        assert shark_pipe._classify_side_fin_type("right_pectoral_fin") == "straight"

    # 3. Dolphin (Straight / flipper side fins)
    dolphin_path = Path("generated_data/models/models_backup_3/dec_mesh_Dolphin.glb")
    if dolphin_path.exists():
        dolphin_model = BaseModelClass(dolphin_path)
        dolphin_pipe = FishModels(dolphin_model)
        dolphin_pipe.segment()
        assert dolphin_pipe._classify_side_fin_type("left_pectoral_fin") == "straight"
        assert dolphin_pipe._classify_side_fin_type("right_pectoral_fin") == "straight"


def test_fin_type_classification():
    """
    Test anatomical fin type classification based on primary extension direction away from the body:
    - Caudal / Tail fin: Extends along +X away from body -> 'caudal'
    - Dorsal / Top fin: Extends along +Y -> 'dorsal'
    - Pectoral / Side fins: Extend along +/-Z -> 'pectoral'
    """
    tuna_path = Path("generated_data/models/models_backup_3/dec_mesh_Tuna.glb")
    if not tuna_path.exists():
        tuna_path = Path("generated_data/models/dec_mesh_Tuna.glb")
    if not tuna_path.exists():
        pytest.skip(f"Tuna mesh not found at {tuna_path}")

    model = BaseModelClass(tuna_path)
    pipe = FishModels(model)
    segments = pipe.segment()

    # 1. Test Tail / Caudal fin
    if "tail" in segments and len(segments["tail"]) > 0:
        assert pipe._classify_fin_type("tail") == "caudal"

    # 2. Test Dorsal / Top fin
    if "dorsal_fin" in segments and len(segments["dorsal_fin"]) > 0:
        assert pipe._classify_fin_type("dorsal_fin") == "dorsal"

    # 3. Test Left Pectoral fin
    if "left_pectoral_fin" in segments and len(segments["left_pectoral_fin"]) > 0:
        assert pipe._classify_fin_type("left_pectoral_fin") == "pectoral"

    # 4. Test Right Pectoral fin
    if "right_pectoral_fin" in segments and len(segments["right_pectoral_fin"]) > 0:
        assert pipe._classify_fin_type("right_pectoral_fin") == "pectoral"

    # 5. Test passing raw face indices and vertex arrays
    if "tail" in segments:
        tail_faces = segments["tail"]
        assert pipe._classify_fin_type(tail_faces) == "caudal"

        tail_verts = pipe.model.mesh.vertices[
            np.unique(pipe.model.mesh.faces[tail_faces])
        ]
        assert pipe._classify_fin_type(tail_verts) == "caudal"


def test_fish_rig_dorsal_fin_enabled():
    """
    Test that setting rig_dorsal_fin=True builds a dorsal_fin_0 bone attached to the spine.
    """
    tuna_path = Path("generated_data/models/models_backup_3/dec_mesh_Tuna.glb")
    if not tuna_path.exists():
        tuna_path = Path("generated_data/models/dec_mesh_Tuna.glb")
    if not tuna_path.exists():
        pytest.skip("Tuna mesh not found")

    model = BaseModelClass(tuna_path)
    pipe = FishModels(model, rig_dorsal_fin=True)
    processed = pipe.process()

    bone_ids = [b.id for b in processed.armature.bones_list]
    assert "dorsal_fin_0" in bone_ids
    dorsal_bone = next(
        b for b in processed.armature.bones_list if b.id == "dorsal_fin_0"
    )
    assert dorsal_bone.parent is not None
    assert "spine_" in dorsal_bone.parent.id


def test_side_fins_parented_to_same_bone():
    """
    Test that left and right pectoral side fins are parented to the exact same spine bone.
    """
    tuna_path = Path("generated_data/models/models_backup_3/dec_mesh_Tuna.glb")
    if not tuna_path.exists():
        tuna_path = Path("generated_data/models/dec_mesh_Tuna.glb")
    if not tuna_path.exists():
        pytest.skip("Tuna mesh not found")

    model = BaseModelClass(tuna_path)
    pipe = FishModels(model, rig_pectoral_fins=True)
    processed = pipe.process()

    bone_ids = [b.id for b in processed.armature.bones_list]
    assert "left_pectoral_fin_0" in bone_ids
    assert "right_pectoral_fin_0" in bone_ids

    left_fin = next(
        b for b in processed.armature.bones_list if b.id == "left_pectoral_fin_0"
    )
    right_fin = next(
        b for b in processed.armature.bones_list if b.id == "right_pectoral_fin_0"
    )

    assert left_fin.parent is not None
    assert right_fin.parent is not None
    assert left_fin.parent == right_fin.parent
    assert left_fin.parent.id == right_fin.parent.id


def test_tail_body_bones_perfect_join_and_growth_wave():
    """
    Verifies that:
    1. Tail bones perfectly join with body bones (head of first tail bone == tail of last body bone, with matching Y and Z).
    2. Animations configured along the body feature exponential growth-based wave motion (growth_factor > 0).
    """
    mesh = create_synthetic_fish_mesh()
    model = BaseModelClass(mesh)
    pipeline = FishModels(
        model=model,
        num_bones=10,
        num_tail_bones=3,
        rig_tail=True,
    )
    processed = pipeline.process()
    armature = processed.armature

    # Find the last spine bone and the first tail bone
    body_bones = [b for b in armature.bones_list if "spine_" in b.id]
    tail_bones = [b for b in armature.bones_list if "tail_" in b.id]

    assert len(body_bones) == 7
    assert len(tail_bones) == 3

    last_body_bone = body_bones[-1]
    first_tail_bone = tail_bones[0]

    # Perfect join checks
    assert first_tail_bone.parent == last_body_bone
    np.testing.assert_allclose(first_tail_bone.head, last_body_bone.tail, atol=1e-6)
    # Check that tail continues in straight line from last body bone tail in Y and Z
    for t_bone in tail_bones:
        np.testing.assert_allclose(t_bone.head[1], last_body_bone.tail[1], atol=1e-5)
        np.testing.assert_allclose(t_bone.head[2], last_body_bone.tail[2], atol=1e-5)
        np.testing.assert_allclose(t_bone.tail[1], last_body_bone.tail[1], atol=1e-5)
        np.testing.assert_allclose(t_bone.tail[2], last_body_bone.tail[2], atol=1e-5)

    # Growth-based wave checks
    for clip_name in ["swim", "idle", "sprint"]:
        clip_cfg = pipeline.animations[clip_name]
        assert "growth_factor" in clip_cfg
        assert clip_cfg["growth_factor"] > 0.0


def test_dorsal_to_tail_spine_straightness():
    """
    Verifies that from the dorsal fin X level forward to the tail end, the spine/armature
    remains in a straight line at the caudal tail level (constant Y and Z).
    """
    for model_path in [
        Path("generated_data/models/models_backup_3/dec_mesh_Tuna.glb"),
        Path("generated_data/models/models_backup_3/dec_mesh_Shark.glb"),
        Path("generated_data/models/models_backup_3/dec_mesh_Dolphin.glb"),
    ]:
        if not model_path.exists():
            continue

        model = BaseModelClass(model_path)
        pipe = FishModels(model)
        processed = pipe.process()

        tail_faces = pipe.segments.get("tail", [])
        dorsal_faces = pipe.segments.get("dorsal_fin", [])

        if tail_faces and dorsal_faces:
            t_v = processed.mesh.vertices[np.unique(processed.mesh.faces[tail_faces])]
            tail_mid_y = float(0.5 * (t_v[:, 1].min() + t_v[:, 1].max()))
            tail_mid_z = float(0.5 * (t_v[:, 2].min() + t_v[:, 2].max()))

            d_v = processed.mesh.vertices[np.unique(processed.mesh.faces[dorsal_faces])]
            x_dorsal = float(d_v[:, 0].mean())

            # Check spine points where x >= x_dorsal
            dorsal_tail_spine = [pt for pt in pipe.target_spine if pt[0] >= x_dorsal]
            assert len(dorsal_tail_spine) >= 2

            for pt in dorsal_tail_spine:
                np.testing.assert_allclose(pt[1], tail_mid_y, atol=1e-4)
                np.testing.assert_allclose(pt[2], tail_mid_z, atol=1e-4)


def test_fish_align_head_tail_detection():
    """
    Tests that FishModels.align() automatically detects when a model is oriented backwards
    (e.g. tail at -X and snout at +X) and rotates it so snout is at -X and tail at +X.
    """
    tuna_path = Path("generated_data/models/models_backup_3/dec_mesh_Tuna.glb")
    if not tuna_path.exists():
        pytest.skip(f"Tuna mesh not found at {tuna_path}")

    # Load Tuna and deliberately rotate 180 degrees around Y (putting snout at +X and tail at -X)
    mesh = trimesh.load(tuna_path, force="mesh")
    rot_180 = trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0])
    mesh.apply_transform(rot_180)

    model = BaseModelClass(mesh)
    pipe = FishModels(model)

    # 1. Segment first on rotated mesh (detects tail and top fin)
    pipe.segment()

    # 2. Align using segmentation results
    pipe.align()

    # Verify tail faces are now at positive X
    tail_faces = pipe.segments.get("tail", [])
    assert len(tail_faces) > 0
    t_verts = pipe.model.mesh.vertices[np.unique(pipe.model.mesh.faces[tail_faces])]
    assert t_verts[:, 0].mean() > 0.0, "Tail was not aligned to positive X"

    # Verify top fin is positioned before the tail fin
    top_faces = pipe.segments.get("dorsal_fin", [])
    if top_faces:
        top_verts = pipe.model.mesh.vertices[
            np.unique(pipe.model.mesh.faces[top_faces])
        ]
        assert t_verts[:, 0].mean() > top_verts[:, 0].mean(), (
            "Tail is not posterior to top fin"
        )


def test_fish_sam_configuration_and_prompts():
    """
    Tests that FishModels initializes with SAM prompts and toggles use_sam cleanly.
    """
    mesh = create_synthetic_fish_mesh()
    model = BaseModelClass(mesh)

    # 1. Test default initialization (use_sam=True)
    pipe_default = FishModels(model)
    assert pipe_default.use_sam is True
    assert pipe_default.prompts == ["tail", "top fin", "side fin"]
    assert pipe_default.face_prompt_detected is None

    # 2. Test toggling use_sam=False
    pipe = FishModels(model, use_sam=False)
    assert pipe.use_sam is False

    # 2. Test custom prompts and use_sam=True flag initialization
    custom_prompts = ["body", "caudal tail", "dorsal fin", "pectoral fins"]
    pipe_sam = FishModels(model, prompts=custom_prompts, use_sam=True)
    assert pipe_sam.use_sam is True
    assert pipe_sam.prompts == custom_prompts
