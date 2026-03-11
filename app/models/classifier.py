import json
import pickle

import torch
import torch.nn as nn

from app.config import LABELS_PATH, MODEL_PATH, SCALER_PATH


class _MLP(nn.Module):
    """Minimal MLP definition matching the saved weights."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BirdClassifier:
    def __init__(self) -> None:
        # Load label mapping
        with open(LABELS_PATH, "r") as f:
            self.label_mapping: dict[str, str] = json.load(f)

        # Load StandardScaler
        with open(SCALER_PATH, "rb") as f:
            self.scaler = pickle.load(f)

        # Load MLP weights
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
        num_classes = len(self.label_mapping)
        # TODO: BirdNET embedding extraction will be added separately
        # Input/hidden dims are placeholders until BirdNET integration
        self.model = _MLP(input_dim=6522, hidden_dim=512, output_dim=num_classes)
        self.model.load_state_dict(checkpoint)
        self.model.eval()

    def predict(self, audio_path: str) -> dict:
        # TODO: implement full inference pipeline once BirdNET embedding
        #       extraction is available. For now return a stub response.
        return {
            "species": "Collared Kingfisher",
            "confidence": 0.87,
        }
