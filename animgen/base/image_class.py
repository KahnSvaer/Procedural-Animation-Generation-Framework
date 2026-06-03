from PIL import Image
from pathlib import Path
from typing import Union, cast

import torch # Allows rembg to use cuda libraries added with torch+cu124
import rembg

class ImageClass:
    def __init__(self, path: Union[str, Path]):
        self.image: Image.Image = Image.open(path)
        self.name = Path(path).stem.replace("_", " ").title()
    
    def preprocess(self, resize: tuple = (256, 256)):
        # Resize the image while maintaining aspect ratio, then crop to the desired size
        self.image.thumbnail(resize, Image.Resampling.LANCZOS)
        w, h = self.image.size
        left = (w - resize[0]) // 2
        top = (h - resize[1]) // 2

        self.image = self.image.crop(
            (
                left,
                top,
                left + resize[0],
                top + resize[1]
            )
        )

        self.image = self.image.convert("RGBA")
        self.image = cast(Image.Image, rembg.remove(self.image))
    
    def show(self):
        self.image.show()
    
    def save(self, path: Union[str, Path]):
        self.image.save(path)

if __name__ == "__main__":
    # Preprocessing all images in the raw folder and saving them to the processed folder
    image_extensions = {".png", ".jpg", ".jpeg", ".webp"}
    for path in Path("./assets/images/raw").iterdir():
        if path.is_file() and path.suffix.lower() in image_extensions:
            img = ImageClass(path)
            img.preprocess()
            img.save(Path("./assets/images/processed") / f"{path.stem}.png")

