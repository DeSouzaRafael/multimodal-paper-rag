from __future__ import annotations

from pathlib import Path

import open_clip
import torch
from PIL import Image


class ImageEmbedder:
    def __init__(self, model_name: str = "ViT-L-14", pretrained: str = "openai"):
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self._tokenizer = open_clip.get_tokenizer(model_name)
        self._model.train(False)  # set inference mode
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)

    def embed_image(self, image_path: str | Path) -> list[float]:
        img = Image.open(image_path).convert("RGB")
        tensor = self._preprocess(img).unsqueeze(0).to(self._device)
        with torch.no_grad():
            vec = self._model.encode_image(tensor)
            vec = vec / vec.norm(dim=-1, keepdim=True)
        return vec.squeeze().cpu().tolist()

    def embed_text(self, text: str) -> list[float]:
        tokens = self._tokenizer([text]).to(self._device)
        with torch.no_grad():
            vec = self._model.encode_text(tokens)
            vec = vec / vec.norm(dim=-1, keepdim=True)
        return vec.squeeze().cpu().tolist()
