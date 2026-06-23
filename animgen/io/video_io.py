import cv2
import tempfile
import numpy as np
from pathlib import Path


def frames_to_video(images, fps=30, codec="mp4v"):
    """
    Returns video bytes in memory.
    """

    if not images:
        raise ValueError("Image list is empty")

    h, w = images[0].shape[:2]

    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        writer = cv2.VideoWriter(
            tmp.name,
            cv2.VideoWriter_fourcc(*codec),  # type: ignore
            fps,
            (w, h),
        )

        for img in images:
            frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            writer.write(frame)

        writer.release()

        tmp.seek(0)
        return tmp.read()


def save_video(video_bytes, output_path):
    """
    Saves video bytes to disk.
    """

    Path(output_path).write_bytes(video_bytes)