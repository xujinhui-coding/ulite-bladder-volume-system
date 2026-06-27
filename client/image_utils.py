from __future__ import annotations

import base64
from io import BytesIO
from urllib import request

import cv2
import numpy as np
from PIL import Image
from PySide6.QtCore import QBuffer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel


def set_label_pixmap(label: QLabel, pixmap: QPixmap) -> None:
    if pixmap.isNull():
        label.clear()
        return
    available_width = max(label.width() - 24, 1)
    available_height = max(label.height() - 24, 1)
    if available_width <= 1 or available_height <= 1:
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignCenter)
        return
    scaled = pixmap.scaled(available_width, available_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    label.setPixmap(scaled)
    label.setAlignment(Qt.AlignCenter)


def _pil_to_pixmap(image: Image.Image) -> QPixmap:
    if image.mode == "RGBA":
        fmt = QImage.Format_RGBA8888
        channels = 4
    elif image.mode == "RGB":
        fmt = QImage.Format_RGB888
        channels = 3
    else:
        image = image.convert("RGBA")
        fmt = QImage.Format_RGBA8888
        channels = 4
    data = image.tobytes("raw", image.mode)
    qimage = QImage(data, image.width, image.height, image.width * channels, fmt)
    return QPixmap.fromImage(qimage)


def load_pixmap(path: str) -> QPixmap:
    with Image.open(path) as image:
        return _pil_to_pixmap(image)


def pixmap_to_pil(pixmap: QPixmap) -> Image.Image:
    """将 QPixmap 安全转换为 PIL Image，避免 PySide6 memoryview 问题"""
    buffer = QBuffer()
    buffer.open(QBuffer.ReadWrite)
    pixmap.toImage().save(buffer, "PNG")
    return Image.open(BytesIO(buffer.data().data()))


def decode_mask(mask_base64: str) -> np.ndarray:
    """解码 base64 PNG mask 为灰度 numpy 数组"""
    mask_bytes = base64.b64decode(mask_base64)
    mask_array = cv2.imdecode(
        np.frombuffer(mask_bytes, dtype=np.uint8),
        cv2.IMREAD_GRAYSCALE,
    )
    return mask_array


def build_binary_mask_pixmap(mask_base64: str, target_width: int, target_height: int) -> QPixmap:
    """生成二值化黑白 mask 图像（白底黑掩膜 -> 黑底白病灶更常用，这里用黑底白病灶）"""
    mask = decode_mask(mask_base64)
    if mask.shape[1] != target_width or mask.shape[0] != target_height:
        mask = cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    # 黑底白病灶
    rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    image = Image.fromarray(rgb)
    return _pil_to_pixmap(image)


def build_overlay_pixmap(
    orig_pixmap: QPixmap,
    mask_base64: str,
    target_width: int,
    target_height: int,
) -> QPixmap:
    """将 mask 叠加到原图上：绿色半透明填充 + 绿色轮廓"""
    mask = decode_mask(mask_base64)
    if mask.shape[1] != target_width or mask.shape[0] != target_height:
        mask = cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

    # QPixmap -> PIL -> numpy
    orig_pil = pixmap_to_pil(orig_pixmap).convert("RGB")
    orig_arr = np.array(orig_pil)
    orig_bgr = cv2.cvtColor(orig_arr, cv2.COLOR_RGB2BGR)

    overlay = np.zeros_like(orig_bgr)
    overlay[:, :, 1] = mask  # 绿色通道
    result = cv2.addWeighted(orig_bgr, 0.6, overlay, 0.4, 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(result_rgb)
    return _pil_to_pixmap(image)


def overlay_mask(
    orig_pixmap: QPixmap,
    mask_base64: str,
    orig_width: int,
    orig_height: int,
) -> QPixmap:
    """保持原接口兼容：返回叠加后的结果图"""
    return build_overlay_pixmap(orig_pixmap, mask_base64, orig_width, orig_height)


def overlay_mask_from_path(
    image_path: str,
    mask_base64: str,
    orig_width: int,
    orig_height: int,
) -> QPixmap:
    """从本地文件加载原图，再叠加 mask"""
    orig_pix = load_pixmap(image_path)
    if orig_pix.isNull():
        return QPixmap()
    return build_overlay_pixmap(orig_pix, mask_base64, orig_width, orig_height)
