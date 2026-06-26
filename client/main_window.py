from __future__ import annotations

import base64
import json
import os
from datetime import datetime
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import QSettings, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QFont, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from client.api_client import ApiError, SegmentationApiClient
from client.config import CLIENT_RESULT_DIR, DEFAULT_API_BASE, WINDOW_TITLE
from client.image_utils import load_remote_pixmap, set_label_pixmap

_HISTORY_FILE = os.path.join(CLIENT_RESULT_DIR, "inference_history.json")
_POLL_INTERVAL_MS = 1000
_MAX_POLL_COUNT = 120

LIGHT_QSS = """
QMainWindow {
    background-color: #F5F7FA;
}
QWidget {
    font-family: "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", "WenQuanYi Micro Hei", sans-serif;
    font-size: 13px;
    color: #2C3E50;
}
QGroupBox {
    border: 1px solid #DDE4ED;
    border-radius: 8px;
    margin-top: 14px;
    padding-top: 16px;
    background-color: #FFFFFF;
    font-weight: bold;
    font-size: 14px;
    color: #34495E;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
}
QPushButton {
    border: 1px solid #B0C4DE;
    border-radius: 6px;
    padding: 7px 18px;
    background-color: #E8F0FE;
    color: #2C3E50;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #D0E3FC;
    border-color: #7BA7D0;
}
QPushButton:pressed {
    background-color: #B8D4F0;
}
QPushButton#connectBtn {
    background-color: #6C9BD2;
    color: #FFFFFF;
    border-color: #5A8AC0;
}
QPushButton#connectBtn:hover {
    background-color: #5A8AC0;
}
QPushButton#inferBtn {
    background-color: #5CB8A5;
    color: #FFFFFF;
    border-color: #4AA38F;
    font-size: 15px;
    padding: 10px 32px;
}
QPushButton#inferBtn:hover {
    background-color: #4AA38F;
}
QPushButton#inferBtn:disabled {
    background-color: #C0C0C0;
    border-color: #A0A0A0;
}
QLineEdit, QComboBox, QDoubleSpinBox {
    border: 1px solid #CCD5E0;
    border-radius: 5px;
    padding: 5px 8px;
    background-color: #FFFFFF;
}
QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus {
    border-color: #7BA7D0;
}
QTabWidget::pane {
    border: 1px solid #DDE4ED;
    border-radius: 6px;
    background-color: #FFFFFF;
}
QTabBar::tab {
    background-color: #ECF0F5;
    border: 1px solid #DDE4ED;
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    padding: 8px 18px;
    margin-right: 2px;
    color: #5A6A7E;
}
QTabBar::tab:selected {
    background-color: #FFFFFF;
    color: #2C3E50;
    font-weight: bold;
}
QTableWidget {
    background-color: #FFFFFF;
    alternate-background-color: #F7F9FC;
    border: 1px solid #E0E7EF;
    border-radius: 6px;
    gridline-color: #E8EEF4;
}
QTableWidget::item {
    padding: 4px 8px;
}
QHeaderView::section {
    background-color: #E8F0FE;
    border: none;
    border-bottom: 1px solid #D0DCE8;
    padding: 6px 8px;
    font-weight: bold;
}
QListWidget {
    background-color: #FFFFFF;
    border: 1px solid #DDE4ED;
    border-radius: 6px;
}
QListWidget::item {
    padding: 4px 8px;
}
QListWidget::item:selected {
    background-color: #D0E3FC;
}
QStatusBar {
    background-color: #E8F0FE;
    color: #5A6A7E;
    border-top: 1px solid #D0DCE8;
}
QSplitter::handle {
    background-color: #DDE4ED;
    width: 2px;
}
QLabel#previewLabel {
    border: 1px dashed #B0C4DE;
    border-radius: 8px;
    background-color: #FAFBFC;
}
"""


def _ensure_client_dir() -> None:
    os.makedirs(CLIENT_RESULT_DIR, exist_ok=True)


def _load_history() -> list[dict[str, Any]]:
    _ensure_client_dir()
    if not os.path.exists(_HISTORY_FILE):
        return []
    with open(_HISTORY_FILE, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _save_history(records: list[dict[str, Any]]) -> None:
    _ensure_client_dir()
    with open(_HISTORY_FILE, "w", encoding="utf-8") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=2)


class _SubmitInferThread(QThread):
    finished_signal = Signal(dict)
    error_signal = Signal(str)

    def __init__(self, client: SegmentationApiClient, image_path: str,
                 model_name: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._client = client
        self._image_path = image_path
        self._model_name = model_name

    def run(self) -> None:
        try:
            result = self._client.submit_predict(self._image_path, self._model_name)
            self.finished_signal.emit(result)
        except ApiError as exc:
            self.error_signal.emit(str(exc))


class _PollResultThread(QThread):
    finished_signal = Signal(dict)
    error_signal = Signal(str)

    def __init__(self, client: SegmentationApiClient, task_id: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._client = client
        self._task_id = task_id

    def run(self) -> None:
        try:
            result = self._client.get_predict_result(self._task_id)
            self.finished_signal.emit(result)
        except ApiError as exc:
            self.error_signal.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(1200, 800)
        self.setMinimumSize(960, 640)

        _ensure_client_dir()
        self._settings = QSettings("ULiteMedSeg", "Client")
        self._client = SegmentationApiClient(
            self._settings.value("api_base_url", DEFAULT_API_BASE)
        )
        self._current_image: str = ""
        self._history: list[dict[str, Any]] = _load_history()
        self._submit_thread: _SubmitInferThread | None = None
        self._poll_thread: _PollResultThread | None = None
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll_task_result)
        self._pending_task_id: str | None = None
        self._poll_count = 0

        self._build_ui()
        self._apply_qss()
        self._refresh_history_table()

    def _apply_qss(self) -> None:
        self.setStyleSheet(LIGHT_QSS)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)
        root_layout.addWidget(splitter)

        sidebar = self._build_sidebar()
        splitter.addWidget(sidebar)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(12, 12, 12, 12)
        right_layout.setSpacing(10)

        top_row = QHBoxLayout()
        top_row.addWidget(self._build_image_panel())
        top_row.addWidget(self._build_control_panel())
        right_layout.addLayout(top_row, 3)

        results_area = QTabWidget()
        results_area.addTab(self._build_result_panel(), "推理结果")
        results_area.addTab(self._build_history_panel(), "历史记录")
        right_layout.addWidget(results_area, 2)

        splitter.addWidget(right_panel)
        splitter.setSizes([240, 960])

        self.status_bar = QStatusBar()
        self.status_bar.showMessage("就绪 - 请配置服务器地址后连接")
        self.setStatusBar(self.status_bar)

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setFixedWidth(230)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        title = QLabel("导航")
        title_font = QFont()
        title_font.setPointSize(15)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        conn_group = QGroupBox("服务器连接")
        conn_layout = QVBoxLayout(conn_group)

        conn_layout.addWidget(QLabel("服务器地址"))
        self.api_url_edit = QLineEdit(self._settings.value("api_base_url", DEFAULT_API_BASE))
        self.api_url_edit.setPlaceholderText("http://192.168.x.x:8000")
        conn_layout.addWidget(self.api_url_edit)

        self.connect_btn = QPushButton("连接服务器")
        self.connect_btn.setObjectName("connectBtn")
        self.connect_btn.clicked.connect(self._on_connect)
        conn_layout.addWidget(self.connect_btn)

        self.conn_status = QLabel("● 未连接")
        self.conn_status.setStyleSheet("color: #999;")
        conn_layout.addWidget(self.conn_status)

        layout.addWidget(conn_group)

        model_group = QGroupBox("推理配置")
        model_layout = QVBoxLayout(model_group)

        model_layout.addWidget(QLabel("模型选择"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["U-Lite"])
        model_layout.addWidget(self.model_combo)

        model_layout.addWidget(QLabel("暂无"))
        self.dummy_label = QLabel("默认参数")
        model_layout.addWidget(self.dummy_label)

        layout.addWidget(model_group)

        img_group = QGroupBox("已导入图像")
        img_layout = QVBoxLayout(img_group)
        self.image_list = QListWidget()
        self.image_list.itemClicked.connect(self._on_image_selected)
        img_layout.addWidget(self.image_list)

        import_btn = QPushButton("导入图像")
        import_btn.clicked.connect(self._on_import_images)
        img_layout.addWidget(import_btn)

        layout.addWidget(img_group)

        layout.addStretch()
        return sidebar

    def _build_image_panel(self) -> QWidget:
        panel = QGroupBox("图像预览")
        layout = QVBoxLayout(panel)
        self.preview_label = QLabel("请导入医学图像")
        self.preview_label.setObjectName("previewLabel")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(300, 300)
        self.preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.preview_label)
        return panel

    def _build_control_panel(self) -> QWidget:
        panel = QGroupBox("操作")
        layout = QVBoxLayout(panel)
        layout.setAlignment(Qt.AlignCenter)

        self.infer_btn = QPushButton("开始推理")
        self.infer_btn.setObjectName("inferBtn")
        self.infer_btn.setMinimumHeight(48)
        self.infer_btn.clicked.connect(self._on_infer)
        self.infer_btn.setEnabled(False)
        layout.addWidget(self.infer_btn)

        info_label = QLabel("选择图像后点击推理\n结果将自动保存至历史记录")
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setStyleSheet("color: #888; font-size: 12px;")
        layout.addWidget(info_label)
        return panel

    def _build_result_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        img_row = QHBoxLayout()
        left = QGroupBox("原图")
        left_layout = QVBoxLayout(left)
        self.orig_label = QLabel("等待推理")
        self.orig_label.setObjectName("previewLabel")
        self.orig_label.setAlignment(Qt.AlignCenter)
        self.orig_label.setMinimumSize(240, 200)
        left_layout.addWidget(self.orig_label)
        img_row.addWidget(left)

        right = QGroupBox("分割结果")
        right_layout = QVBoxLayout(right)
        self.result_label = QLabel("等待推理")
        self.result_label.setObjectName("previewLabel")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setMinimumSize(240, 200)
        right_layout.addWidget(self.result_label)
        img_row.addWidget(right)

        layout.addLayout(img_row, 2)

        metrics_group = QGroupBox("量化指标")
        metrics_layout = QVBoxLayout(metrics_group)
        self.metrics_table = QTableWidget(4, 2)
        self.metrics_table.setHorizontalHeaderLabels(["指标", "数值"])
        self.metrics_table.horizontalHeader().setStretchLastSection(True)
        self.metrics_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.metrics_table.verticalHeader().setVisible(False)
        self.metrics_table.setEditTriggers(QTableWidget.NoEditTriggers)
        metrics_layout.addWidget(self.metrics_table)
        layout.addWidget(metrics_group, 1)

        return panel

    def _build_history_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        bar = QHBoxLayout()
        clear_btn = QPushButton("清空历史")
        clear_btn.clicked.connect(self._on_clear_history)
        bar.addWidget(clear_btn)
        bar.addStretch()
        export_btn = QPushButton("导出记录")
        export_btn.clicked.connect(self._on_export_history)
        bar.addWidget(export_btn)
        layout.addLayout(bar)

        self.history_table = QTableWidget(0, 6)
        self.history_table.setHorizontalHeaderLabels([
            "时间", "文件名", "模型", "病灶面积 (px)", "病灶占比", "推理耗时 (ms)"
        ])
        self.history_table.horizontalHeader().setStretchLastSection(True)
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.history_table.setAlternatingRowColors(True)
        self.history_table.setSelectionBehavior(QTableWidget.SelectRows)
        layout.addWidget(self.history_table)
        return panel

    def _on_connect(self) -> None:
        url = self.api_url_edit.text().strip()
        if not url:
            QMessageBox.warning(self, "提示", "请输入服务器地址")
            return
        self._client = SegmentationApiClient(url)
        self.conn_status.setText("⏳ 正在检测...")
        self.conn_status.setStyleSheet("color: #E67E22;")
        QApplication.processEvents()

        if self._client.health():
            self._settings.setValue("api_base_url", url)
            self.conn_status.setText("● 已连接")
            self.conn_status.setStyleSheet("color: #27AE60; font-weight: bold;")
            self.status_bar.showMessage(f"已连接至 {url}")
        else:
            self.conn_status.setText("● 连接失败")
            self.conn_status.setStyleSheet("color: #E74C3C;")
            self.status_bar.showMessage("服务器连接失败，请检查地址及网络")

    def _on_import_images(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择医学图像", "",
            "图像文件 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp);;所有文件 (*)"
        )
        for f in files:
            for i in range(self.image_list.count()):
                if self.image_list.item(i).data(Qt.UserRole) == f:
                    break
            else:
                item = QListWidgetItem(os.path.basename(f))
                item.setData(Qt.UserRole, f)
                self.image_list.addItem(item)

        if files and not self._current_image:
            self.image_list.setCurrentRow(0)
            self._on_image_selected(self.image_list.item(0))

    def _on_image_selected(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.UserRole)
        self._current_image = path
        pix = QPixmap(path)
        if pix.isNull():
            QMessageBox.warning(self, "警告", f"无法加载图像: {path}")
            return
        set_label_pixmap(self.preview_label, pix)
        self.infer_btn.setEnabled(True)
        self.status_bar.showMessage(f"已选择: {os.path.basename(path)}")

    def _on_infer(self) -> None:
        if not self._current_image:
            return
        self._pending_task_id = None
        self._poll_count = 0
        self._poll_timer.stop()
        self.infer_btn.setEnabled(False)
        self.infer_btn.setText("提交中...")
        self.status_bar.showMessage("正在提交推理任务，请稍候...")
        QApplication.processEvents()

        self._submit_thread = _SubmitInferThread(
            self._client, self._current_image,
            "u-lite", self
        )
        self._submit_thread.finished_signal.connect(self._on_submit_done)
        self._submit_thread.error_signal.connect(self._on_infer_error)
        self._submit_thread.start()

    def _on_submit_done(self, result: dict[str, Any]) -> None:
        task_id = result.get("task_id")
        if not task_id:
            self._on_infer_error("未收到任务ID")
            return
        self._pending_task_id = task_id
        self.infer_btn.setText("轮询中...")
        self.status_bar.showMessage(f"任务已提交，正在等待结果: {task_id}")
        self._poll_timer.start(_POLL_INTERVAL_MS)
        self._poll_task_result()

    def _poll_task_result(self) -> None:
        if not self._pending_task_id:
            self._poll_timer.stop()
            return
        if self._poll_thread and self._poll_thread.isRunning():
            return
        if self._poll_count >= _MAX_POLL_COUNT:
            self._on_infer_error("轮询结果超时")
            return
        self._poll_count += 1
        self._poll_thread = _PollResultThread(self._client, self._pending_task_id, self)
        self._poll_thread.finished_signal.connect(self._on_poll_result)
        self._poll_thread.error_signal.connect(self._on_infer_error)
        self._poll_thread.start()

    def _on_poll_result(self, result: dict[str, Any]) -> None:
        status = result.get("status")
        if status in {"pending", "processing"}:
            self.status_bar.showMessage(f"任务处理中: {self._pending_task_id}")
            return
        self._poll_timer.stop()
        self._pending_task_id = None
        if status == "failed":
            self._on_infer_error(result.get("error") or "推理失败")
            return
        self._on_infer_done(result)

    def _on_infer_done(self, result: dict[str, Any]) -> None:
        self.infer_btn.setEnabled(True)
        self.infer_btn.setText("开始推理")

        orig_url = self._client.build_url(result.get("original_image_url", ""))
        self.orig_label.clear()
        self.result_label.clear()

        def _load_results() -> None:
            try:
                orig_pix = load_remote_pixmap(orig_url) if orig_url else QPixmap()
                set_label_pixmap(self.orig_label, orig_pix)

                # 从 mask_base64 生成分割叠加图
                mask_b64 = result.get("mask_base64")
                if mask_b64 and not orig_pix.isNull():
                    mask_bytes = base64.b64decode(mask_b64)
                    mask_img = QImage.fromData(mask_bytes, "PNG")
                    if mask_img.isNull():
                        self.result_label.clear()
                    else:
                        # 将 mask 转为 numpy（二值）
                        mask_arr = np.frombuffer(mask_img.bits().tobytes(), dtype=np.uint8).reshape(
                            mask_img.height(), mask_img.width(), 4
                        )
                        mask_gray = (mask_arr[:, :, 0] > 128).astype(np.uint8) * 255

                        # 原图 numpy
                        orig_img = orig_pix.toImage()
                        orig_bits = orig_img.bits().tobytes()
                        orig_np = np.frombuffer(orig_bits, dtype=np.uint8).reshape(
                            orig_img.height(), orig_img.width(), 4
                        )
                        orig_bgr = cv2.cvtColor(orig_np[:, :, :3], cv2.COLOR_RGB2BGR)

                        # 缩小 mask 到原图尺寸
                        if mask_gray.shape[:2] != orig_bgr.shape[:2]:
                            mask_gray = cv2.resize(
                                mask_gray, (orig_bgr.shape[1], orig_bgr.shape[0]),
                                interpolation=cv2.INTER_NEAREST
                            )

                        # 生成叠加图
                        overlay = np.zeros_like(orig_bgr, dtype=np.uint8)
                        overlay[:, :, 1] = mask_gray
                        result_bgr = cv2.addWeighted(orig_bgr, 0.6, overlay, 0.4, 0)

                        # 绘制轮廓
                        contours, _ = cv2.findContours(
                            mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        cv2.drawContours(result_bgr, contours, -1, (0, 255, 0), 2)

                        # 转回 QPixmap
                        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
                        h, w, ch = result_rgb.shape
                        result_qimg = QImage(result_rgb.data, w, h, w * ch, QImage.Format_RGB888)
                        set_label_pixmap(self.result_label, QPixmap.fromImage(result_qimg))
                else:
                    self.result_label.clear()
            except Exception:
                self.status_bar.showMessage("结果图片加载失败")

        QTimer.singleShot(100, _load_results)

        metrics = [
            ("推理耗时", f"{result.get('inference_time_ms', 0)} ms"),
            ("病灶像素面积", str(result.get("lesion_area", 0))),
            ("病灶占比", f"{result.get('lesion_ratio', 0):.6f}"),
            ("使用模型", result.get("model_name", "")),
        ]
        self.metrics_table.setRowCount(len(metrics))
        for idx, (label, value) in enumerate(metrics):
            self.metrics_table.setItem(idx, 0, QTableWidgetItem(label))
            self.metrics_table.setItem(idx, 1, QTableWidgetItem(value))

        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filename": os.path.basename(self._current_image),
            "model": result.get("model_name", ""),
            "lesion_area": result.get("lesion_area", 0),
            "lesion_ratio": result.get("lesion_ratio", 0),
            "inference_time_ms": result.get("inference_time_ms", 0),
        }
        self._history.insert(0, record)
        _save_history(self._history)
        self._refresh_history_table()
        self.status_bar.showMessage(
            f"推理完成 - 耗时 {result.get('inference_time_ms', 0)} ms"
        )

    def _on_infer_error(self, msg: str) -> None:
        self._poll_timer.stop()
        self._pending_task_id = None
        self.infer_btn.setEnabled(True)
        self.infer_btn.setText("开始推理")
        QMessageBox.critical(self, "推理失败", msg)
        self.status_bar.showMessage(f"推理失败: {msg}")

    def _refresh_history_table(self) -> None:
        self.history_table.setRowCount(len(self._history))
        for idx, rec in enumerate(self._history):
            self.history_table.setItem(idx, 0, QTableWidgetItem(rec.get("timestamp", "")))
            self.history_table.setItem(idx, 1, QTableWidgetItem(rec.get("filename", "")))
            self.history_table.setItem(idx, 2, QTableWidgetItem(rec.get("model", "")))
            self.history_table.setItem(idx, 3, QTableWidgetItem(str(rec.get("lesion_area", ""))))
            self.history_table.setItem(idx, 4, QTableWidgetItem(str(rec.get("lesion_ratio", ""))))
            self.history_table.setItem(idx, 5, QTableWidgetItem(str(rec.get("inference_time_ms", ""))))

    def _on_clear_history(self) -> None:
        reply = QMessageBox.question(self, "确认", "确定要清空所有历史记录吗？")
        if reply == QMessageBox.Yes:
            self._history.clear()
            _save_history(self._history)
            self._refresh_history_table()

    def _on_export_history(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "导出历史记录", "inference_history.csv", "CSV (*.csv)"
        )
        if not path:
            return
        import csv
        with open(path, "w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.writer(fh)
            writer.writerow(["时间", "文件名", "模型", "病灶面积(px)", "病灶占比", "推理耗时(ms)"])
            for rec in self._history:
                writer.writerow([
                    rec.get("timestamp", ""),
                    rec.get("filename", ""),
                    rec.get("model", ""),
                    rec.get("lesion_area", ""),
                    rec.get("lesion_ratio", ""),
                    rec.get("inference_time_ms", ""),
                ])
        self.status_bar.showMessage(f"历史记录已导出至 {path}")
