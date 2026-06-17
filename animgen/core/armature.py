from typing import List, Union, Tuple, Optional   
import uuid

Vec3 = tuple[float, float, float]

def generate_unique_id() -> str:
    return str(uuid.uuid4())

class Bones:
    def __init__(
            self, id: str = "",  
            parent: Union['Bones', None] = None, 
            head: Optional[Vec3] = None, 
            tail: Vec3 = (0, 0, 0),
            is_connected_to_parent: bool = False
        ):
        
        self.id = id
        if self.id == "":
            self.id = generate_unique_id()

        self.tail: Vec3 = tail
        self.parent: Optional[Bones] = parent
        self.child: List[Bones] = []
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
    
    def __init__(self, root_bone: Bones):
        self.root_bone: Bones = root_bone
        self.bones_list: List[Bones] = [root_bone]
        self.disconnected_chain_roots: List[Bones] = [root_bone]  # List stores unconnected bones

    def add_connected_bone(self, parent: Bones, tail: Vec3) -> Bones:
        new_bone = Bones(parent=parent, head=parent.tail, tail=tail, is_connected_to_parent=True)
        parent.child.append(new_bone)
        self.bones_list.append(new_bone)
        return new_bone
    
    def add_unconnected_bone(self,  parent: Bones, head: Vec3, tail: Vec3) -> Bones:
        new_bone = Bones(parent=parent, head=head, tail=tail, is_connected_to_parent=False)
        parent.child.append(new_bone)
        self.bones_list.append(new_bone)
        self.disconnected_chain_roots.append(new_bone)
        return new_bone
    
    # DFS traversal function for linked list like armature structure
    def traverse_bones(self, bone: Bones, depth=0):
        bones_list: List[Bones] = []
        for child in bone.child:
            bones_list.append(child)
            self.traverse_bones(child, depth + 1)