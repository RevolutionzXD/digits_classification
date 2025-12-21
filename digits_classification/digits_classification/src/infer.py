import os
import torch

from src.models.model import SimpleMLP

try:
    from src.models.model import CNN
    TargetCNNClass = CNN
except Exception:
    TargetCNNClass = None


def loadModel(device, model_type: str):
    model_type = model_type.lower()

    if model_type == "mlp":
        model = SimpleMLP(input_dim=784, hidden_dim=128, output_dim=10).to(device)
        path = "assets/model_final.pth"

    elif model_type == "cnn":

        model = TargetCNNClass().to(device)
        path = "assets/model_cnn_final.pth"

    else:
        raise ValueError(f"model_type không hợp lệ: {model_type}")

    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model
