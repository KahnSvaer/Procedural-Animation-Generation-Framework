from transformers import Sam3Processor, Sam3Model
from pathlib import Path

from animgen.core.generated_asset_class import GeneratedAssetClass
import trimesh

from dotenv import load_dotenv
import os

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
os.environ["HF_HOME"] = "./models_cache/SAM_original/"

MODEL_PATH = "facebook/sam3"


class SAM3_apendage_finding:
    def __init__(self, device):
        self.device = device
        self.model, self.processor = self.load_model()
        self.text_embeddings = None

    def load_model(self):
        model = Sam3Model.from_pretrained(MODEL_PATH).to(self.device)  # type: ignore
        processor = Sam3Processor.from_pretrained(MODEL_PATH)
        return model, processor

    def get_text_embeddings(self, PROMPT: list[str]):
        pass

    def __call__(
        self,
        mesh: GeneratedAssetClass | str | Path | trimesh.Trimesh,
        text_inputs: list[str] | None = None,
    ):

        if not isinstance(mesh, GeneratedAssetClass):
            mesh = GeneratedAssetClass(mesh)
        if text_inputs is None and self.text_embeddings is None:
            raise ValueError(
                "Add text prompts either to the class or inside the __call__ function"
            )
