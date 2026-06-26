from pydantic import BaseModel


class PredictSubmitResponse(BaseModel):
    code: int
    message: str
    task_id: str
    status: str
    original_image_url: str


class PredictResultResponse(BaseModel):
    code: int
    message: str
    task_id: str
    status: str
    original_image_url: str
    result_image_url: str | None = None
    inference_time_ms: float | None = None
    lesion_area: int | None = None
    lesion_ratio: float | None = None
    model_name: str | None = None
    error: str | None = None


class PredictResponse(PredictResultResponse):
    pass
