import cv2
import numpy as np
from PIL import Image


def load_image(image_path: str) -> np.ndarray:
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)


def preprocess_image(image_bgr: np.ndarray, input_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    resized = cv2.resize(image_bgr, input_size)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=0)
    return tensor, resized
