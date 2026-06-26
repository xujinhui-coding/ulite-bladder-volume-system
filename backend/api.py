import logging
import os

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.config import settings
from backend.paths import build_runtime_name, is_allowed_image, to_url_path
from backend.schemas import PredictResultResponse, PredictSubmitResponse
from backend.task_queue import get_task_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/predict", response_model=PredictSubmitResponse)
async def predict(
    file: UploadFile = File(...),
) -> PredictSubmitResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="未接收到上传文件")
    if not is_allowed_image(file.filename):
        raise HTTPException(status_code=400, detail="仅支持常见图片格式上传")

    upload_name = build_runtime_name(file.filename)
    upload_path = os.path.join(settings.temp_upload_dir, upload_name)
    original_image_url = to_url_path(settings.upload_url_prefix, upload_name)

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="上传文件为空")
        with open(upload_path, "wb") as out_file:
            out_file.write(content)

        task = get_task_manager().submit_task(upload_path, settings.default_model, original_image_url)
        logger.info(
            "收到推理任务 task_id=%s file=%s model=%s",
            task.task_id,
            file.filename,
            settings.default_model,
        )
        return PredictSubmitResponse(
            code=0,
            message="U-Lite inference task submitted",
            task_id=task.task_id,
            status=task.status,
            original_image_url=task.original_image_url,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("提交推理任务失败 file=%s error=%s", file.filename, exc)
        raise HTTPException(status_code=500, detail=f"提交推理任务失败: {exc}") from exc


@router.get("/predict/{task_id}", response_model=PredictResultResponse)
def get_predict_result(task_id: str) -> PredictResultResponse:
    task = get_task_manager().get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    result = task.result or {}
    message = "任务处理中"
    if task.status == "completed":
        message = "U-Lite inference success"
    elif task.status == "failed":
        message = "推理失败"

    return PredictResultResponse(
        code=0,
        message=message,
        task_id=task.task_id,
        status=task.status,
        original_image_url=task.original_image_url,
        result_image_url=result.get("result_image_url"),
        inference_time_ms=result.get("inference_time_ms"),
        lesion_area=result.get("lesion_area"),
        lesion_ratio=result.get("lesion_ratio"),
        model_name=result.get("model_name"),
        error=task.error,
    )
