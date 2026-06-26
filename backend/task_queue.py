"""P0 内存任务队列 — 异步解耦推理请求与执行"""
from __future__ import annotations

import logging
import threading
import uuid
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Task:
    __slots__ = ("task_id", "file_path", "model_name",
                 "original_image_url", "status", "result", "error")

    def __init__(
        self,
        task_id: str,
        file_path: str,
        model_name: str,
        original_image_url: str = "",
    ) -> None:
        self.task_id = task_id
        self.file_path = file_path
        self.model_name = model_name
        self.original_image_url = original_image_url
        self.status = TaskStatus.PENDING
        self.result: dict[str, Any] | None = None
        self.error: str | None = None


class MemoryTaskQueue:
    """线程安全的进程内任务队列"""

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}
        self._lock = threading.Lock()

    def submit_task(
        self,
        file_path: str,
        model_name: str,
        original_image_url: str = "",
    ) -> Task:
        task_id = uuid.uuid4().hex[:12]
        with self._lock:
            task = Task(task_id, file_path, model_name, original_image_url)
            self._tasks[task_id] = task
        logger.info("任务已提交 task_id=%s file=%s", task_id, file_path)
        return task

    def get_task(self, task_id: str) -> Task | None:
        with self._lock:
            return self._tasks.get(task_id)

    def update(
        self,
        task_id: str,
        status: TaskStatus,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.status = status
                task.result = result
                task.error = error


# 全局单例
task_queue = MemoryTaskQueue()


def get_task_manager() -> MemoryTaskQueue:
    """供 api.py 获取任务管理器实例"""
    return task_queue
