import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    project_root: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    temp_upload_name: str = "temp_upload"
    upload_url_prefix: str = "/temp_upload"
    default_model: str = "u-lite"
    default_input_size: tuple[int, int] = (256, 256)
    uvicorn_host: str = os.getenv("UVICORN_HOST", "0.0.0.0")
    uvicorn_port: int = int(os.getenv("UVICORN_PORT", "8000"))

    @property
    def temp_upload_dir(self) -> str:
        return os.path.join(self.project_root, self.temp_upload_name)


settings = Settings()
