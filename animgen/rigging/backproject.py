import numpy as np


def backproject_masks_to_faces(
    masks_dict: dict[str, list[np.ndarray]],
    faces_per_view: list[np.ndarray] | np.ndarray,
    num_faces: int,
) -> dict[str, np.ndarray]:
    """
    Backprojects 2D binary masks across multiple camera views onto 3D mesh faces.

    Parameters
    ----------
    masks_dict : dict[str, list[np.ndarray]]
        Mapping from prompt name to a list of 2D boolean/uint8 masks (one per view).
    faces_per_view : list[np.ndarray] | np.ndarray
        Array or list of 2D face ID maps (H, W) for each view rendered by the camera system.
        Pixels with value < 0 indicate background.
    num_faces : int
        Total number of faces on the 3D mesh.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from each prompt name to a 1D int32 array of shape (num_faces,)
        containing the total vote count per face.
    """
    face_prompt_detected: dict[str, np.ndarray] = {
        prompt: np.zeros(num_faces, dtype=np.int32) for prompt in masks_dict.keys()
    }

    for view_idx, v_faces in enumerate(faces_per_view):
        for prompt, view_masks in masks_dict.items():
            if view_idx < len(view_masks):
                mask = view_masks[view_idx]
                if mask is not None and np.any(mask):
                    m_faces = v_faces[mask > 0]
                    m_faces = m_faces[m_faces >= 0]
                    if len(m_faces) > 0:
                        detected_faces = np.unique(m_faces)
                        face_prompt_detected[prompt][detected_faces] += 1

    return face_prompt_detected
