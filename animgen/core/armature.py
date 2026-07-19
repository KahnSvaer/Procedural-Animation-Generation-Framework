import uuid
from .types import Vec3


def generate_unique_id() -> str:
    return str(uuid.uuid4())


class Bone:
    """
    Represents a single bone in an armature.
    Each bone has a head and tail position in 3D space, a parent bone (if any), and a list of child bones.
    """

    def __init__(
        self,
        id: str = "",
        parent: "Bone | None" = None,
        head: Vec3 | None = None,
        tail: Vec3 = (0, 0, 0),
        is_connected_to_parent: bool = False,
    ):
        self.id = id
        if self.id == "":
            self.id = generate_unique_id()

        self.tail: Vec3 = tail
        self.parent: Bone | None = parent
        self.child: list[Bone] = []
        self.is_connected_to_parent: bool = is_connected_to_parent

        if head is not None:
            self.head = head
        elif parent is not None and self.is_connected_to_parent:
            self.head = parent.tail
        else:
            raise ValueError(
                "Head must be provided if parent is None or if parent is not connected."
            )


class Armature:
    """
    Represents a hierarchical structure of bones (an armature).
    """

    def __init__(self, root_bone: Bone):
        self.root_bone: Bone = root_bone
        self.bones_list: list[Bone] = [root_bone]
        self.disconnected_chain_roots: list[Bone] = [
            self.root_bone
        ]  # List stores unconnected bones for traversal purposes

    def add_connected_bone(self, parent: Bone, tail: Vec3) -> Bone:
        """
        Adds a connected bone to the armature.
        """
        new_bone = Bone(
            parent=parent, head=parent.tail, tail=tail, is_connected_to_parent=True
        )
        parent.child.append(new_bone)
        self.bones_list.append(new_bone)
        return new_bone

    def add_unconnected_bone(self, parent: Bone, head: Vec3, tail: Vec3) -> Bone:
        """
        Adds an unconnected bone to the armature.
        """
        new_bone = Bone(
            parent=parent, head=head, tail=tail, is_connected_to_parent=False
        )
        parent.child.append(new_bone)
        self.bones_list.append(new_bone)
        self.disconnected_chain_roots.append(new_bone)
        return new_bone

    def add_chain(self, armature: "Armature", parent_bone: Bone):
        """
        Adds a disconnected chain of bones (another armature) to the current armature under the specified parent bone.
        """
        parent_bone.child.append(armature.root_bone)
        armature.root_bone.parent = parent_bone
        self.bones_list.extend(armature.bones_list)
        self.disconnected_chain_roots.extend(armature.disconnected_chain_roots)

    def add_root_bone(self, head: Vec3):
        """
        Adds a new root bone to the armature. Added to add negative root bones to the armature for better manipulation.
        """
        old_root = self.root_bone
        new_root = Bone(head=head, tail=old_root.head, is_connected_to_parent=False)
        new_root.child.append(old_root)
        old_root.parent = new_root
        old_root.is_connected_to_parent = True
        self.root_bone = new_root
        self.bones_list.insert(0, new_root)

        self.disconnected_chain_roots.remove(old_root)
        self.disconnected_chain_roots.append(new_root)

    def validify_tree(self):
        """
        Validates the tree structure of the armature.
        Ensures that there are no cycles and that each bone's parent-child relationships are consistent.
        """
        # Parent child relationship validation
        for bone in self.bones_list:
            if bone.parent is not None and bone not in bone.parent.child:
                raise ValueError(
                    f"Bone {bone.id} is not listed as a child of its parent {bone.parent.id}."
                )
            if bone.parent is None and bone != self.root_bone:
                raise ValueError(
                    f"Bone {bone.id} has no parent but is not the root bone."
                )

        # Cycle detection using DFS
        visited = set()

        def dfs(bone: Bone):
            if bone in visited:
                raise ValueError(f"Cycle detected at bone {bone.id}.")
            visited.add(bone)
            for child in bone.child:
                dfs(child)
            visited.remove(bone)

        dfs(self.root_bone)
