from __future__ import annotations

from urllib import request

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QLabel


def set_label_pixmap(label: QLabel, pixmap: QPixmap) -> None:
    if pixmap.isNull():
        label.clear()
        return
    scaled = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
    label.setPixmap(scaled)
    label.setAlignment(Qt.AlignCenter)


def load_remote_pixmap(url: str) -> QPixmap:
    with request.urlopen(url, timeout=30) as resp:
        data = resp.read()
    pixmap = QPixmap()
    pixmap.loadFromData(data)
    return pixmap
