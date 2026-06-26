"""后台工作者 — 从内存队列中取出任务并执行推理"""
from __future__ import annotations

import logging
import threading
import time

from backend.inference_service import run_inference
from backend.task_queue import TaskStatus, task_queue

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 0.5


def _process_task(task_id: str) -> None:
    task = task_queue.get_task(task_id)
    if task is None:
        return

    logger.info("开始处理任务 task_id=%s", task_id)
    task_queue.update(task_id, TaskStatus.PROCESSING)

    try:
        result = run_inference(task.file_path, task.model_name)
        task_queue.update(task_id, TaskStatus.COMPLETED, result=result)
        logger.info("任务完成 task_id=%s", task_id)
    except Exception as exc:
        logger.exception("任务失败 task_id=%s", task_id)
        task_queue.update(task_id, TaskStatus.FAILED, error=str(exc))


def _poll_loop() -> None:
    """轮询内存队列，逐个处理 PENDING 任务"""
    while True:
        task = None
        for t in list(task_queue._tasks.values()):
            if t.status == TaskStatus.PENDING:
                task = t
                break
        if task:
            _process_task(task.task_id)
        time.sleep(_POLL_INTERVAL)


def start_worker() -> threading.Thread:
    """以守护线程启动后台工作者"""
    thread = threading.Thread(target=_poll_loop, daemon=True, name="task-worker")
    thread.start()
    logger.info("后台工作者已启动")
    return thread
