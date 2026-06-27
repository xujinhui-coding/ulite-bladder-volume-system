from __future__ import annotations

import json
import time
from typing import Any
from urllib import request


class ApiError(Exception):
    pass


class SegmentationApiClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def health(self) -> bool:
        try:
            with request.urlopen(f"{self.base_url}/", timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False

    def submit_task(self, image_path: str, threshold: float = 0.5, model_name: str = "u-lite") -> dict[str, Any]:
        """提交推理任务，返回 task_id + original_image_url"""
        try:
            boundary = "----TraeFormBoundary7MA4YWxkTrZu0gW"
            body = self._build_multipart_body(image_path, threshold, model_name, boundary)
            req = request.Request(
                url=f"{self.base_url}/predict",
                data=body,
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                method="POST",
            )
            with request.urlopen(req, timeout=30) as resp:
                payload = resp.read().decode("utf-8")
                return json.loads(payload)
        except Exception as exc:
            raise ApiError(f"提交任务失败: {exc}") from exc

    def poll_result(self, task_id: str, poll_interval: float = 1.0, timeout: float = 120.0) -> dict[str, Any]:
        """轮询 GET /predict/{task_id} 直到 completed 或 failed"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                req = request.Request(url=f"{self.base_url}/predict/{task_id}", method="GET")
                with request.urlopen(req, timeout=10) as resp:
                    payload = resp.read().decode("utf-8")
                    data = json.loads(payload)
                    status = data.get("status", "")
                    if status == "completed":
                        return data
                    if status == "failed":
                        raise ApiError(f"推理失败: {data.get('error', '未知错误')}")
                    # pending / processing → 继续轮询
            except ApiError:
                raise
            except Exception as exc:
                raise ApiError(f"轮询失败: {exc}") from exc
            time.sleep(poll_interval)
        raise ApiError(f"轮询超时 ({timeout}s)")

    def build_url(self, relative_url: str) -> str:
        return f"{self.base_url}{relative_url}"

    def _build_multipart_body(self, image_path: str, threshold: float, model_name: str, boundary: str) -> bytes:
        with open(image_path, "rb") as file_obj:
            image_bytes = file_obj.read()

        filename = image_path.replace("\\", "/").split("/")[-1]
        lines: list[bytes] = []
        fields = {
            "model_name": model_name,
            "threshold": str(threshold),
        }
        for key, value in fields.items():
            lines.append(f"--{boundary}".encode())
            lines.append(f'Content-Disposition: form-data; name="{key}"'.encode())
            lines.append(b"")
            lines.append(str(value).encode())

        lines.append(f"--{boundary}".encode())
        lines.append(f'Content-Disposition: form-data; name="file"; filename="{filename}"'.encode())
        lines.append(b"Content-Type: image/jpeg")
        lines.append(b"")
        lines.append(image_bytes)
        lines.append(f"--{boundary}--".encode())
        lines.append(b"")
        return b"\r\n".join(lines)
