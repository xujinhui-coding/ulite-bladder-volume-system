import os

from backend.config import settings


MODEL_REGISTRY = {
    "u-lite": {
        "model_path": os.path.join(settings.project_root, "runs", "dataset1_scratch", "dataset1_scratch_sim.onnx"),
        "display_name": "U-Lite",
        "input_size": (256, 256),
    },
}


def get_model_config(model_name: str | None = None) -> dict:
    key = settings.default_model
    if model_name and model_name.strip().lower() != settings.default_model:
        raise ValueError(f"当前仅支持模型: {settings.default_model}")
    config = MODEL_REGISTRY[key]
    if not os.path.exists(config["model_path"]):
        raise FileNotFoundError(f"模型文件不存在: {config['model_path']}")
    return {"key": key, **config}
