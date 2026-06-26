from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.api import router
from backend.config import settings
from backend.paths import ensure_runtime_dirs
from backend.worker import start_worker


ensure_runtime_dirs()

app = FastAPI(title="U-Lite Medical Segmentation Backend", version="1.0.0")
app.include_router(router)
app.mount(settings.upload_url_prefix, StaticFiles(directory=settings.temp_upload_dir), name="temp_upload")
app.mount(settings.result_url_prefix, StaticFiles(directory=settings.result_img_dir), name="result_img")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
async def startup_event() -> None:
    start_worker()


@app.get("/")
def root() -> FileResponse:
    return FileResponse("static/index.html")


def run() -> None:
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.uvicorn_host,
        port=settings.uvicorn_port,
        workers=1,
    )


if __name__ == "__main__":
    run()
