import cv2
import numpy as np

_THRESHOLD = 0.5


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def build_mask(output: np.ndarray, orig_shape: tuple[int, int]) -> np.ndarray:
    heatmap = output[0, 0]
    if heatmap.min() < 0.0 or heatmap.max() > 1.0:
        heatmap = sigmoid(heatmap)
    mask = (heatmap > _THRESHOLD).astype(np.uint8) * 255
    height, width = orig_shape
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)


def build_visualization(orig_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = np.zeros_like(orig_bgr)
    overlay[:, :, 1] = mask
    result = cv2.addWeighted(orig_bgr, 0.6, overlay, 0.4, 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
    return result


def calc_metrics(mask: np.ndarray) -> tuple[int, float]:
    lesion_area = int(np.count_nonzero(mask))
    total_pixels = int(mask.shape[0] * mask.shape[1])
    lesion_ratio = float(lesion_area / total_pixels) if total_pixels else 0.0
    return lesion_area, lesion_ratio
