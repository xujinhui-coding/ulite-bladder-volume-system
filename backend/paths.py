import os
import uuid
from datetime import datetime

from backend.config import settings


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def ensure_runtime_dirs() -> None:
    os.makedirs(settings.temp_upload_dir, exist_ok=True)


def is_allowed_image(filename: str) -> bool:
    _, ext = os.path.splitext(filename.lower())
    return ext in IMAGE_EXTENSIONS


def build_runtime_name(filename: str) -> str:
    stem, ext = os.path.splitext(os.path.basename(filename))
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stem}_{now}_{uuid.uuid4().hex[:8]}{ext.lower()}"


def to_url_path(prefix: str, filename: str) -> str:
    return f"{prefix}/{filename}"
