from __future__ import annotations

import json
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

    def submit_predict(self, image_path: str, model_name: str = "u-lite") -> dict[str, Any]:
        boundary = "----TraeFormBoundary7MA4YWxkTrZu0gW"
        body = self._build_multipart_body(image_path, model_name, boundary)
        req = request.Request(
            url=f"{self.base_url}/predict",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        return self._request_json(req, timeout=60)

    def get_predict_result(self, task_id: str) -> dict[str, Any]:
        req = request.Request(
            url=f"{self.base_url}/predict/{task_id}",
            method="GET",
        )
        return self._request_json(req, timeout=30)

    def predict(self, image_path: str, model_name: str = "u-lite") -> dict[str, Any]:
        return self.submit_predict(image_path, model_name)

    def build_url(self, relative_url: str) -> str:
        return f"{self.base_url}{relative_url}"

    def _request_json(self, req: request.Request, timeout: int) -> dict[str, Any]:
        try:
            with request.urlopen(req, timeout=timeout) as resp:
                payload = resp.read().decode("utf-8")
                return json.loads(payload)
        except Exception as exc:
            raise ApiError(f"请求失败: {exc}") from exc

    def _build_multipart_body(self, image_path: str, model_name: str, boundary: str) -> bytes:
        with open(image_path, "rb") as file_obj:
            image_bytes = file_obj.read()

        filename = image_path.replace("\\", "/").split("/")[-1]
        lines: list[bytes] = []
        fields = {
            "model_name": model_name,
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
