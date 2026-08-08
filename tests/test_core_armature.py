import pytest

from animgen.core.armature import Bone, Armature


def test_bone_and_armature_structure():
    """Test basic bone creation and armature hierarchy building."""
    # 1. Root bone
    root = Bone(head=(0.0, 0.0, 0.0), tail=(0.0, 0.0, 1.0))
    assert root.parent is None
    assert root.head == (0.0, 0.0, 0.0)
    assert root.tail == (0.0, 0.0, 1.0)

    armature = Armature(root)
    assert armature.root_bone == root
    assert len(armature.bones_list) == 1

    # 2. Add connected bone
    bone_1 = armature.add_connected_bone(root, tail=(0.0, 1.0, 1.0))
    assert bone_1.parent == root
    assert bone_1.head == root.tail
    assert bone_1.is_connected_to_parent is True
    assert len(armature.bones_list) == 2
    assert bone_1 in root.child

    # 3. Add unconnected bone
    bone_2 = armature.add_unconnected_bone(
        bone_1, head=(1.0, 1.0, 1.0), tail=(2.0, 1.0, 1.0)
    )
    assert bone_2.parent == bone_1
    assert bone_2.head == (1.0, 1.0, 1.0)
    assert bone_2.is_connected_to_parent is False
    assert len(armature.bones_list) == 3

    # Validate structure
    armature.validify_tree()


def test_armature_cycle_and_invalid_relationships():
    """Test validation errors for invalid armature relationships."""
    root = Bone(head=(0.0, 0.0, 0.0), tail=(0.0, 0.0, 1.0))
    armature = Armature(root)
    bone_1 = armature.add_connected_bone(root, tail=(0.0, 1.0, 1.0))

    # Introduce a cyclic link
    bone_1.child.append(root)

    with pytest.raises(ValueError, match="Cycle detected"):
        armature.validify_tree()

    # Clean up cycle
    bone_1.child.remove(root)

    # Make parent-child relationship inconsistent
    root.child.remove(bone_1)
    with pytest.raises(ValueError, match="is not listed as a child of its parent"):
        armature.validify_tree()
