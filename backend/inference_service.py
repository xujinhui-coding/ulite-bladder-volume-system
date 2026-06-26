import gc
import logging
import os
import time
from functools import lru_cache

import cv2
import onnxruntime as ort

from backend.model_registry import get_model_config
from backend.paths import build_result_name, to_url_path
from backend.postprocess import build_mask, build_visualization, calc_metrics
from backend.preprocess import load_image, preprocess_image
from backend.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def load_session(model_path: str) -> ort.InferenceSession:
    logger.info("加载ONNX模型: %s", model_path)
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session_options.enable_cpu_mem_arena = False
    session_options.enable_mem_pattern = False
    return ort.InferenceSession(
        model_path,
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )


def run_inference(image_path: str, model_name: str) -> dict:
    image_bgr = None
    tensor = None
    outputs = None
    mask = None
    result_bgr = None
    try:
        config = get_model_config(model_name)
        image_bgr = load_image(image_path)
        orig_h, orig_w = image_bgr.shape[:2]
        tensor, _ = preprocess_image(image_bgr, config["input_size"])

        session = load_session(config["model_path"])
        input_name = session.get_inputs()[0].name

        start_time = time.perf_counter()
        outputs = session.run(None, {input_name: tensor})
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 3)

        mask = build_mask(outputs[0], (orig_h, orig_w))
        result_bgr = build_visualization(image_bgr, mask)
        lesion_area, lesion_ratio = calc_metrics(mask)

        result_name = build_result_name(image_path)
        result_path = os.path.join(settings.result_img_dir, result_name)
        cv2.imwrite(result_path, result_bgr)

        logger.info(
            "推理完成 file=%s model=%s time_ms=%s lesion_area=%s lesion_ratio=%.6f",
            os.path.basename(image_path),
            config["display_name"],
            elapsed_ms,
            lesion_area,
            lesion_ratio,
        )

        return {
            "inference_time_ms": elapsed_ms,
            "lesion_area": lesion_area,
            "lesion_ratio": round(lesion_ratio, 6),
            "model_name": config["display_name"],
            "result_image_url": to_url_path(settings.result_url_prefix, result_name),
        }
    finally:
        _cleanup(outputs, tensor, mask, result_bgr, image_bgr)


def _cleanup(*objs: object) -> None:
    for obj in objs:
        del obj
    gc.collect()
