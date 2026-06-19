import sys
sys.path.append(".")

from animgen.core.armature import Armature, Bones

import bpy

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Remove orphaned datablocks
    for collection in (
        bpy.data.meshes,
        bpy.data.armatures,
        bpy.data.materials,
        bpy.data.cameras,
        bpy.data.lights,
        bpy.data.images,
    ):
        for block in collection:
            if block.users == 0:
                collection.remove(block)
                

def create_blender_armature(armature_data):
    # Create armature datablock
    arm_data = bpy.data.armatures.new("TestArmature")
    arm_obj = bpy.data.objects.new("TestArmature", arm_data)

    bpy.context.collection.objects.link(arm_obj)

    # Enter edit mode
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='EDIT')

    edit_bones = {}

    def create_bone_recursive(custom_bone):
        eb = arm_data.edit_bones.new(custom_bone.id)

        eb.head = custom_bone.head
        eb.tail = custom_bone.tail

        edit_bones[custom_bone.id] = eb

        if custom_bone.parent is not None:
            eb.parent = edit_bones[custom_bone.parent.id]
            eb.use_connect = custom_bone.is_connected_to_parent

        for child in custom_bone.child:
            create_bone_recursive(child)

    create_bone_recursive(armature_data.root_bone)

    bpy.ops.object.mode_set(mode='OBJECT')

    return arm_obj


if __name__ == "__main__":

    clear_scene()

    root = Bones(
        id="root",
        head=(0, 0, 0),
        tail=(0, 0, 1)
    )

    armature = Armature(root)

    spine = armature.add_connected_bone(
        parent=root,
        tail=(0, 0, 2)
    )

    neck = armature.add_connected_bone(
        parent=spine,
        tail=(0, 0, 3)
    )

    arm = armature.add_unconnected_bone(
        parent=root,
        head=(0.5, 0, 1),
        tail=(1.5, 0, 1)
    )

    create_blender_armature(armature)

    bpy.ops.wm.save_as_mainfile(
        filepath="./generated_data/rigged/test_armature.blend"
    )

