import gc
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
from dotenv import load_dotenv
import numpy as np
import torch
from transformers import Sam3Model, Sam3Processor
import trimesh

from animgen.core.models.model import BaseModelClass

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
os.environ["HF_HOME"] = os.getenv("HF_HOME", "./models_cache/SAM_original/")

MODEL_PATH = "facebook/sam3"
DEFAULT_FIXTURE_FOLDER = Path("./models_cache/fixtures")
DEFAULT_FIXTURE_PATH = DEFAULT_FIXTURE_FOLDER / "sam3_text_embeddings.pt"


def _resolve_torch_dtype(
    device: str | torch.device,
    dtype: torch.dtype | str | None = None,
) -> torch.dtype:
    """
    Resolves the appropriate torch.dtype for SAM3 execution.

    By default on CUDA, uses bfloat16 if supported (Ampere+), otherwise float16.
    On CPU, defaults to float32.
    """
    if dtype is not None:
        if isinstance(dtype, str):
            dtype_map = {
                "float16": torch.float16,
                "fp16": torch.float16,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }
            return dtype_map.get(dtype.lower(), torch.float32)
        return dtype

    device_str = str(device)
    if "cuda" in device_str and torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


class SAM3TextEmbedder:
    """
    Backend service/model to convert text prompts into SAM3 text embeddings
    and save/load them to/from known fixture files.
    """

    def __init__(
        self,
        device: str | torch.device | None = None,
        model_path: str = MODEL_PATH,
        dtype: torch.dtype | str | None = None,
        token: str | None = None,
        model: Sam3Model | None = None,
        processor: Sam3Processor | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = _resolve_torch_dtype(self.device, dtype)
        self.model_path = model_path
        self.token = token or hf_token
        self.model = model
        self.processor = processor

    def load_model(self) -> tuple[Sam3Model, Sam3Processor]:
        """
        Loads the SAM3 model and processor to the configured device.
        """
        model = Sam3Model.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            token=self.token,
        ).to(self.device)  # type: ignore
        model.eval()

        processor = Sam3Processor.from_pretrained(
            self.model_path,
            token=self.token,
        )
        return model, processor

    def unload(self) -> None:
        """
        Unloads the SAM3 model from memory and clears GPU VRAM cache.
        """
        self.model = None
        self.processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self) -> "SAM3TextEmbedder":
        if self.model is None or self.processor is None:
            self.model, self.processor = self.load_model()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.unload()

    def encode(self, prompts: list[str] | str) -> dict[str, dict[str, torch.Tensor]]:
        """
        Encodes a single prompt or list of prompts into SAM3 text embeddings.

        Parameters
        ----------
        prompts : list[str] | str
            The prompt string(s) to encode.

        Returns
        -------
        dict[str, dict[str, torch.Tensor]]
            Dictionary mapping each prompt to its CPU-detached 'text_embeds',
            'input_ids', and 'attention_mask' tensors.
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        embeddings: dict[str, dict[str, torch.Tensor]] = {}

        if self.model is None or self.processor is None:
            self.model, self.processor = self.load_model()

        with torch.no_grad():
            for prompt in prompts:
                text_input = self.processor(
                    text=prompt,
                    return_tensors="pt",
                ).to(self.device)

                text_embed = self.model.get_text_features(
                    input_ids=text_input.input_ids,
                    attention_mask=text_input.attention_mask,
                )

                if not isinstance(text_embed, torch.Tensor):
                    text_embed = text_embed[0]

                embeddings[prompt] = {
                    "text_embeds": text_embed.detach().cpu(),
                    "input_ids": text_input.input_ids.detach().cpu(),
                    "attention_mask": text_input.attention_mask.detach().cpu(),
                }
                del text_embed
                del text_input

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return embeddings

    def save(
        self,
        embeddings: dict[str, dict[str, torch.Tensor]],
        output_path: str | Path = DEFAULT_FIXTURE_PATH,
    ) -> Path:
        """
        Saves the text embeddings dictionary to a fixture file (.pt).

        Parameters
        ----------
        embeddings : dict[str, dict[str, torch.Tensor]]
            The embeddings dictionary returned by encode().
        output_path : str | Path, default=DEFAULT_FIXTURE_PATH
            Path where the serialized fixture file should be stored.

        Returns
        -------
        Path
            The absolute or relative path to the saved file.
        """
        save_path = Path(output_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(embeddings, save_path)
        return save_path

    @staticmethod
    def load(
        fixture_path: str | Path = DEFAULT_FIXTURE_PATH,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """
        Loads precomputed text embeddings from a fixture file.

        Parameters
        ----------
        fixture_path : str | Path, default=DEFAULT_FIXTURE_PATH
            Path to the .pt fixture file.

        Returns
        -------
        dict[str, dict[str, torch.Tensor]]
            Dictionary mapping prompt strings to their embedding dictionaries.
        """
        load_path = Path(fixture_path)
        if not load_path.exists():
            raise FileNotFoundError(f"Fixture file not found: {load_path}")
        return torch.load(load_path, weights_only=True)

    def generate_and_save(
        self,
        prompts: list[str],
        output_path: str | Path = DEFAULT_FIXTURE_PATH,
    ) -> Path:
        """
        Convenience method to encode prompts and immediately serialize them to disk.
        """
        embeddings = self.encode(prompts)
        return self.save(embeddings, output_path=output_path)


class SAM3Segmentation:
    """
    SAM3-based multi-view appendage finder and 3D face prompt detector.

    Renders multi-view projections of a 3D mesh, performs zero-shot text-prompted
    instance segmentation using SAM3, and back-projects 2D masks onto 3D mesh faces.
    """

    def __init__(
        self,
        device: str | torch.device | None = None,
        model_path: str = MODEL_PATH,
        dtype: torch.dtype | str | None = None,
        prompts: list[str] | None = None,
        token: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = _resolve_torch_dtype(self.device, dtype)
        self.model_path = model_path
        self.token = token or hf_token

        self.model, self.processor = self.load_model()
        self.prompts: list[str] | None = None
        self.text_embeddings: list[Any] | None = None
        self.text_inputs: list[Any] | None = None

        if prompts is not None:
            self.get_text_embeddings(prompts)

    def load_model(self) -> tuple[Sam3Model, Sam3Processor]:
        """
        Loads the SAM3 model and processor to the configured device.
        """
        model = Sam3Model.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            token=self.token,
        ).to(self.device)  # type: ignore
        model.eval()

        processor = Sam3Processor.from_pretrained(
            self.model_path,
            token=self.token,
        )
        return model, processor

    def unload(self) -> None:
        """
        Unloads the SAM3 model from VRAM, cleans up cached tensors, and releases GPU memory.
        """
        self.model = None
        self.processor = None
        self.text_embeddings = None
        self.text_inputs = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self) -> "SAM3Segmentation":
        if self.model is None or self.processor is None:
            self.model, self.processor = self.load_model()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.unload()

    def get_text_embeddings(self, PROMPT: list[str]) -> list[Any]:
        """
        Extracts and caches text feature embeddings for the given list of prompts.
        """
        if self.model is None or self.processor is None:
            self.model, self.processor = self.load_model()

        self.prompts = list(PROMPT)
        self.text_embeddings = []
        self.text_inputs = []

        with torch.no_grad():
            for prompt in self.prompts:
                text_input = self.processor(
                    text=prompt,
                    return_tensors="pt",
                ).to(self.device)
                text_embed = self.model.get_text_features(
                    input_ids=text_input.input_ids,
                    attention_mask=text_input.attention_mask,
                )
                self.text_inputs.append(text_input)
                self.text_embeddings.append(text_embed)

        return self.text_embeddings

    def __call__(
        self,
        mesh: BaseModelClass | str | Path | trimesh.Trimesh,
        text_inputs: list[str] | None = None,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        max_masks_per_view: int = 2,
        return_instances: bool = False,
    ) -> dict[str, list[Any]]:
        """
        Runs multi-view segmentation on the 3D mesh views using SAM3.

        Parameters
        ----------
        mesh : BaseModelClass | str | Path | trimesh.Trimesh
            The input 3D mesh or BaseModelClass instance.
        text_inputs : list[str] | None, optional
            List of prompt strings. If None, uses pre-cached text embeddings.
        threshold : float, default=0.5
            Confidence threshold for instance segmentation predictions.
        mask_threshold : float, default=0.5
            Binarization threshold for output masks.
        max_masks_per_view : int, default=2
            Maximum number of top instance masks to combine per view.
        return_instances : bool, default=False
            If True, returns a list of individual instance masks per view (list[list[np.ndarray]]).
            If False, combines instance masks per view with logical OR (list[np.ndarray]).

        Returns
        -------
        dict[str, list[np.ndarray]] | dict[str, list[list[np.ndarray]]]
            Mapping from each prompt string to either a list of 2D boolean masks or a list
            of per-instance mask lists corresponding to each rendered view.
        """
        if self.model is None or self.processor is None:
            self.model, self.processor = self.load_model()

        if not isinstance(mesh, BaseModelClass):
            mesh = BaseModelClass(mesh)

        if text_inputs is not None:
            self.get_text_embeddings(text_inputs)

        if (
            self.text_embeddings is None
            or self.text_inputs is None
            or self.prompts is None
        ):
            raise ValueError(
                "Add text prompts either to the class or inside the __call__ function"
            )

        view_outputs = mesh.views_output
        images = view_outputs["matte"]

        results_masks: dict[str, list[Any]] = defaultdict(list)

        with torch.no_grad():
            for image_idx, image in enumerate(images):
                image_inputs = self.processor(
                    images=image,
                    return_tensors="pt",
                ).to(self.device)

                vision_embeds = self.model.get_vision_features(
                    pixel_values=image_inputs.pixel_values.to(dtype=self.dtype),
                )
                target_sizes = [[image.height, image.width]]

                for text_embeds, text_in, prompt in zip(
                    self.text_embeddings, self.text_inputs, self.prompts
                ):
                    outputs = self.model(
                        vision_embeds=vision_embeds,
                        text_embeds=text_embeds,
                        attention_mask=text_in.attention_mask,
                    )
                    results = self.processor.post_process_instance_segmentation(
                        outputs,
                        threshold=threshold,
                        mask_threshold=mask_threshold,
                        target_sizes=target_sizes,
                    )[0]

                    masks = results.get("masks", [])
                    if return_instances:
                        view_instance_masks = []
                        if len(masks) > 0:
                            for m in masks[:max_masks_per_view]:
                                m_np = m.cpu().numpy()
                                if m_np.shape != (image.height, image.width):
                                    m_np = cv2.resize(
                                        m_np.astype(np.uint8),
                                        (image.width, image.height),
                                        interpolation=cv2.INTER_NEAREST,
                                    ).astype(bool)
                                view_instance_masks.append(m_np)
                        results_masks[prompt].append(view_instance_masks)
                    else:
                        if len(masks) > 0:
                            combined_mask = (
                                masks[:max_masks_per_view].any(dim=0).cpu().numpy()
                            )
                            if combined_mask.shape != (image.height, image.width):
                                combined_mask = cv2.resize(
                                    combined_mask.astype(np.uint8),
                                    (image.width, image.height),
                                    interpolation=cv2.INTER_NEAREST,
                                ).astype(bool)
                        else:
                            combined_mask = np.zeros(
                                (image.height, image.width), dtype=bool
                            )

                        results_masks[prompt].append(combined_mask)

                    del outputs
                    del results
                    del masks

                del vision_embeds
                del image_inputs

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return dict(results_masks)
