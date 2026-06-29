from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

from PySide6.QtCore import QDateTime, QPointF, QSettings, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PySide6.QtCharts import QChart, QChartView, QDateTimeAxis, QLineSeries, QScatterSeries, QValueAxis
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
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
    QScrollArea,
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
from client.image_utils import (
    build_binary_mask_pixmap,
    build_overlay_pixmap,
    load_pixmap,
    overlay_mask_from_path,
    set_label_pixmap,
)

_HISTORY_FILE = os.path.join(CLIENT_RESULT_DIR, "inference_history.json")

LIGHT_QSS = """
QMainWindow {
    background-color: #F8FAFC;
}
QWidget {
    font-family: "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", "WenQuanYi Micro Hei", sans-serif;
    font-size: 13px;
    color: #334155;
}
QGroupBox {
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    margin-top: 14px;
    padding-top: 16px;
    background-color: #FFFFFF;
    font-weight: 600;
    font-size: 14px;
    color: #1E293B;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 16px;
    padding: 0 8px;
}
QPushButton {
    border: none;
    border-radius: 8px;
    padding: 8px 20px;
    background-color: #F1F5F9;
    color: #475569;
    font-weight: 500;
    font-size: 13px;
}
QPushButton:hover {
    background-color: #E2E8F0;
}
QPushButton:pressed {
    background-color: #CBD5E1;
}
QPushButton#connectBtn {
    background-color: #3B82F6;
    color: #FFFFFF;
}
QPushButton#connectBtn:hover {
    background-color: #2563EB;
}
QPushButton#connectBtn:pressed {
    background-color: #1D4ED8;
}
QPushButton#inferBtn {
    background-color: #10B981;
    color: #FFFFFF;
    font-size: 15px;
    font-weight: 600;
    padding: 12px 36px;
    border-radius: 10px;
}
QPushButton#inferBtn:hover {
    background-color: #059669;
}
QPushButton#inferBtn:pressed {
    background-color: #047857;
}
QPushButton#inferBtn:disabled {
    background-color: #94A3B8;
}
QLineEdit, QComboBox, QDoubleSpinBox {
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 7px 12px;
    background-color: #FFFFFF;
    font-size: 13px;
}
QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus {
    border-color: #3B82F6;
    outline: none;
}
QTabWidget::pane {
    border: none;
    background-color: transparent;
}
QTabBar::tab {
    background-color: transparent;
    border: none;
    padding: 10px 24px;
    margin-right: 4px;
    color: #94A3B8;
    font-weight: 500;
    font-size: 14px;
    border-radius: 8px;
}
QTabBar::tab:selected {
    background-color: #FFFFFF;
    color: #1E293B;
    font-weight: 600;
}
QTableWidget {
    background-color: #FFFFFF;
    alternate-background-color: #FAFBFC;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    gridline-color: #E2E8F0;
}
QTableWidget::item {
    padding: 6px 12px;
}
QHeaderView::section {
    background-color: #F8FAFC;
    border: none;
    border-bottom: 2px solid #E2E8F0;
    padding: 8px 12px;
    font-weight: 600;
    color: #64748B;
}
QListWidget {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    outline: none;
}
QListWidget::item {
    padding: 10px 14px;
    border-radius: 6px;
    margin: 2px 4px;
    color: #475569;
}
QListWidget::item:hover {
    background-color: #F1F5F9;
}
QListWidget::item:selected {
    background-color: #DBEAFE;
    color: #1D4ED8;
    font-weight: 500;
}
QStatusBar {
    background-color: #FFFFFF;
    color: #64748B;
    border-top: 1px solid #E2E8F0;
}
QSplitter::handle {
    background-color: #E2E8F0;
    width: 3px;
}
QSplitter::handle:hover {
    background-color: #CBD5E1;
}
QFrame#resultCard {
    background-color: #FFFFFF;
    border: 1px solid #DCE6F2;
    border-radius: 18px;
}
QLabel#panelTitle {
    color: #0F172A;
    font-size: 16px;
    font-weight: 700;
    padding: 2px 4px 8px 4px;
}
QLabel#previewLabel {
    border: 1px solid #D8E1F0;
    border-radius: 16px;
    background-color: #F8FAFC;
    padding: 12px;
    color: #64748B;
}
QComboBox::drop-down {
    border: none;
    border-radius: 0 8px 8px 0;
}
QComboBox::down-arrow {
    image: none;
}
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    border: none;
    border-radius: 0;
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


class _InferThread(QThread):
    finished_signal = Signal(dict)
    error_signal = Signal(str)

    def __init__(
        self,
        client: SegmentationApiClient,
        image_path: str,
        threshold: float,
        model_name: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._client = client
        self._image_path = image_path
        self._threshold = threshold
        self._model_name = model_name

    def run(self) -> None:
        try:
            # Step 1: 提交任务 → 获取 task_id
            submit_result = self._client.submit_task(self._image_path, self._threshold, self._model_name)
            task_id = submit_result.get("task_id", "")
            original_image_url = submit_result.get("original_image_url", "")
            if not task_id:
                raise ApiError("后端未返回 task_id")

            # Step 2: 轮询结果
            poll_result = self._client.poll_result(task_id)
            poll_result["original_image_url"] = original_image_url
            self.finished_signal.emit(poll_result)
        except ApiError as exc:
            self.error_signal.emit(str(exc))
        except Exception as exc:
            self.error_signal.emit(f"未知错误: {exc}")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(1200, 800)
        self.setMinimumSize(960, 640)

        _ensure_client_dir()
        self._settings = QSettings("ULiteMedSeg", "Client")
        self._client = SegmentationApiClient(self._settings.value("api_base_url", DEFAULT_API_BASE))
        self._current_image: str = ""
        self._history: list[dict[str, Any]] = _load_history()
        self._infer_thread: _InferThread | None = None

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
        splitter.addWidget(self._build_sidebar())

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(12, 12, 12, 12)
        right_layout.setSpacing(10)

        top_row = QHBoxLayout()
        top_row.setSpacing(12)
        top_row.addWidget(self._build_image_panel(), 4)
        top_row.addWidget(self._build_control_panel(), 1)
        right_layout.addLayout(top_row, 1)

        tabs = QTabWidget()
        tabs.addTab(self._build_result_panel(), "推理结果")
        tabs.addTab(self._build_history_panel(), "历史记录")
        tabs.addTab(self._build_trend_panel(), "趋势分析")
        right_layout.addWidget(tabs, 4)

        splitter.addWidget(right_panel)
        splitter.setSizes([250, 950])

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setFixedWidth(240)
        sidebar.setStyleSheet("background-color: #1E293B;")
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(16, 20, 16, 20)
        layout.setSpacing(16)

        logo_layout = QHBoxLayout()
        logo_label = QLabel("🩺")
        logo_font = QFont()
        logo_font.setPointSize(24)
        logo_label.setFont(logo_font)
        logo_layout.addWidget(logo_label)

        title_layout = QVBoxLayout()
        app_title = QLabel("膀胱超声")
        app_title.setStyleSheet("color: #FFFFFF; font-size: 16px; font-weight: 600;")
        app_subtitle = QLabel("图像处理系统")
        app_subtitle.setStyleSheet("color: #94A3B8; font-size: 11px;")
        title_layout.addWidget(app_title)
        title_layout.addWidget(app_subtitle)
        logo_layout.addLayout(title_layout)
        logo_layout.addStretch()
        layout.addLayout(logo_layout)

        divider = QWidget()
        divider.setFixedHeight(1)
        divider.setStyleSheet("background-color: #334155;")
        layout.addWidget(divider)

        conn_group = QGroupBox("服务器连接")
        conn_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #334155;
                border-radius: 10px;
                background-color: #0F172A;
                color: #E2E8F0;
                font-size: 12px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: #94A3B8;
            }
        """)
        conn_layout = QVBoxLayout(conn_group)
        conn_layout.setContentsMargins(12, 14, 12, 14)
        conn_layout.setSpacing(10)

        url_label = QLabel("服务器地址")
        url_label.setStyleSheet("color: #94A3B8; font-size: 12px;")
        conn_layout.addWidget(url_label)

        self.api_url_edit = QLineEdit(self._settings.value("api_base_url", DEFAULT_API_BASE))
        self.api_url_edit.setPlaceholderText("http://192.168.x.x:8000")
        self.api_url_edit.setStyleSheet("""
            QLineEdit {
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 8px 10px;
                background-color: #1E293B;
                color: #E2E8F0;
                font-size: 12px;
            }
            QLineEdit:focus {
                border-color: #3B82F6;
            }
            QLineEdit::placeholder {
                color: #475569;
            }
        """)
        conn_layout.addWidget(self.api_url_edit)

        self.connect_btn = QPushButton("连接服务器")
        self.connect_btn.setObjectName("connectBtn")
        self.connect_btn.clicked.connect(self._on_connect)
        self.connect_btn.setStyleSheet("""
            QPushButton {
                background-color: #3B82F6;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 0;
                font-size: 13px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #2563EB;
            }
            QPushButton:pressed {
                background-color: #1D4ED8;
            }
        """)
        conn_layout.addWidget(self.connect_btn)

        self.conn_status = QLabel("● 未连接")
        self.conn_status.setStyleSheet("color: #64748B; font-size: 12px;")
        self.conn_status.setAlignment(Qt.AlignCenter)
        conn_layout.addWidget(self.conn_status)
        layout.addWidget(conn_group)

        model_group = QGroupBox("推理配置")
        model_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #334155;
                border-radius: 10px;
                background-color: #0F172A;
                color: #E2E8F0;
                font-size: 12px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: #94A3B8;
            }
        """)
        model_layout = QVBoxLayout(model_group)
        model_layout.setContentsMargins(12, 14, 12, 14)
        model_layout.setSpacing(10)

        model_label = QLabel("模型选择")
        model_label.setStyleSheet("color: #94A3B8; font-size: 12px;")
        model_layout.addWidget(model_label)

        self.model_combo = QComboBox()
        self.model_combo.addItems(["U-Lite"])
        self.model_combo.setStyleSheet("""
            QComboBox {
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 8px 10px;
                background-color: #1E293B;
                color: #E2E8F0;
                font-size: 12px;
            }
            QComboBox:focus {
                border-color: #3B82F6;
            }
        """)
        model_layout.addWidget(self.model_combo)

        threshold_label = QLabel("分割阈值")
        threshold_label.setStyleSheet("color: #94A3B8; font-size: 12px;")
        model_layout.addWidget(threshold_label)

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.01, 0.99)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setValue(0.5)
        self.threshold_spin.setStyleSheet("""
            QDoubleSpinBox {
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 8px 10px;
                background-color: #1E293B;
                color: #E2E8F0;
                font-size: 12px;
            }
            QDoubleSpinBox:focus {
                border-color: #3B82F6;
            }
        """)
        model_layout.addWidget(self.threshold_spin)
        layout.addWidget(model_group)

        img_group = QGroupBox("已导入图像")
        img_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #334155;
                border-radius: 10px;
                background-color: #0F172A;
                color: #E2E8F0;
                font-size: 12px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: #94A3B8;
            }
        """)
        img_layout = QVBoxLayout(img_group)
        img_layout.setContentsMargins(8, 12, 8, 12)
        img_layout.setSpacing(8)

        self.image_list = QListWidget()
        self.image_list.itemClicked.connect(self._on_image_selected)
        self.image_list.setStyleSheet("""
            QListWidget {
                background-color: #1E293B;
                border: 1px solid #334155;
                border-radius: 6px;
                color: #E2E8F0;
                font-size: 12px;
            }
            QListWidget::item {
                padding: 8px 12px;
                border-radius: 4px;
                margin: 2px 4px;
            }
            QListWidget::item:hover {
                background-color: #334155;
            }
            QListWidget::item:selected {
                background-color: #3B82F6;
                color: white;
            }
        """)
        img_layout.addWidget(self.image_list)

        import_btn = QPushButton("导入图像")
        import_btn.clicked.connect(self._on_import_images)
        import_btn.setStyleSheet("""
            QPushButton {
                background-color: #334155;
                color: #E2E8F0;
                border: none;
                border-radius: 6px;
                padding: 8px 0;
                font-size: 12px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #475569;
            }
        """)
        img_layout.addWidget(import_btn)
        layout.addWidget(img_group)

        layout.addStretch()
        return sidebar

    def _build_image_panel(self) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet("""
            QWidget {
                background-color: #FFFFFF;
                border-radius: 12px;
                border: 1px solid #E2E8F0;
            }
        """)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title_label = QLabel("图像预览")
        title_label.setStyleSheet("color: #1E293B; font-size: 14px; font-weight: 600;")
        layout.addWidget(title_label)

        self.preview_label = QLabel("请导入医学图像")
        self.preview_label.setObjectName("previewLabel")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumHeight(140)
        self.preview_label.setMaximumHeight(180)
        self.preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(self.preview_label)
        return panel

    def _build_control_panel(self) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet("""
            QWidget {
                background-color: #FFFFFF;
                border-radius: 12px;
                border: 1px solid #E2E8F0;
            }
        """)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)
        layout.setAlignment(Qt.AlignCenter)

        title_label = QLabel("操作")
        title_label.setStyleSheet("color: #1E293B; font-size: 14px; font-weight: 600;")
        layout.addWidget(title_label)

        self.infer_btn = QPushButton("开始推理")
        self.infer_btn.setObjectName("inferBtn")
        self.infer_btn.setMinimumHeight(48)
        self.infer_btn.setStyleSheet("""
            QPushButton {
                background-color: #10B981;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 12px 36px;
                font-size: 15px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #059669;
            }
            QPushButton:pressed {
                background-color: #047857;
            }
            QPushButton:disabled {
                background-color: #D1D5DB;
            }
        """)
        self.infer_btn.clicked.connect(self._on_infer)
        self.infer_btn.setEnabled(False)
        layout.addWidget(self.infer_btn)

        info_label = QLabel("选择图像后点击推理\n结果将自动保存至历史记录")
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setStyleSheet("color: #94A3B8; font-size: 12px; line-height: 1.6;")
        layout.addWidget(info_label)

        layout.addStretch()
        return panel

    def _build_result_panel(self) -> QWidget:
        panel = QWidget()
        root_layout = QVBoxLayout(panel)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(12)

        images_container = QWidget()
        images_layout = QHBoxLayout(images_container)
        images_layout.setContentsMargins(0, 0, 0, 0)
        images_layout.setSpacing(16)

        left = QFrame()
        left.setObjectName("resultCard")
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(18, 18, 18, 18)
        left_layout.setSpacing(12)

        left_title = QLabel("分割区域")
        left_title.setObjectName("panelTitle")
        left_layout.addWidget(left_title)

        self.orig_label = QLabel("等待推理")
        self.orig_label.setObjectName("previewLabel")
        self.orig_label.setAlignment(Qt.AlignCenter)
        self.orig_label.setMinimumHeight(220)
        self.orig_label.setMaximumHeight(220)
        self.orig_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        left_layout.addWidget(self.orig_label)
        images_layout.addWidget(left, 1)

        right = QFrame()
        right.setObjectName("resultCard")
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(18, 18, 18, 18)
        right_layout.setSpacing(12)

        right_title = QLabel("分割结果")
        right_title.setObjectName("panelTitle")
        right_layout.addWidget(right_title)

        self.result_label = QLabel("等待推理")
        self.result_label.setObjectName("previewLabel")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setMinimumHeight(220)
        self.result_label.setMaximumHeight(220)
        self.result_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        right_layout.addWidget(self.result_label)
        images_layout.addWidget(right, 1)

        metrics_group = QFrame()
        metrics_group.setObjectName("resultCard")
        metrics_layout = QVBoxLayout(metrics_group)
        metrics_layout.setContentsMargins(18, 18, 18, 18)
        metrics_layout.setSpacing(12)

        metrics_title = QLabel("量化指标")
        metrics_title.setObjectName("panelTitle")
        metrics_layout.addWidget(metrics_title)

        self.metrics_table = QTableWidget(4, 2)
        self.metrics_table.setHorizontalHeaderLabels(["指标", "数值"])
        self.metrics_table.horizontalHeader().setStretchLastSection(True)
        self.metrics_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.metrics_table.verticalHeader().setVisible(False)
        self.metrics_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.metrics_table.setAlternatingRowColors(True)
        self.metrics_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.metrics_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.metrics_table.setMinimumHeight(220)
        self.metrics_table.setMaximumHeight(220)
        self.metrics_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.metrics_table.setStyleSheet("""
            QTableWidget {
                background-color: #F8FAFC;
                border: 1px solid #E2E8F0;
                border-radius: 12px;
                gridline-color: #E2E8F0;
                selection-background-color: #DBEAFE;
                alternate-background-color: #F1F5F9;
            }
            QTableWidget::item {
                padding: 10px 14px;
                color: #334155;
            }
            QHeaderView::section {
                background-color: #EEF2FF;
                border: none;
                border-bottom: 1px solid #D8E1F0;
                padding: 10px 14px;
                font-weight: 600;
                color: #475569;
            }
        """)
        metrics_layout.addWidget(self.metrics_table)

        root_layout.addWidget(images_container)
        root_layout.addWidget(metrics_group)
        root_layout.setStretch(0, 2)
        root_layout.setStretch(1, 1)
        return panel

    def _build_trend_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setStyleSheet("QScrollArea { background: #EEF4F8; border: none; } QScrollBar:vertical { background: #E4ECF2; width: 10px; margin: 0; border-radius: 5px; } QScrollBar::handle:vertical { background: #B8C8D4; min-height: 40px; border-radius: 5px; } QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; background: transparent; }")

        content = QWidget()
        content.setStyleSheet("background: #EEF4F8;")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(16, 16, 16, 16)
        content_layout.setSpacing(16)

        trend_group = QFrame()
        trend_group.setObjectName("trendMainCard")
        trend_group.setStyleSheet("QFrame#trendMainCard { background-color: #FFFFFF; border: 1px solid #D7E3EC; border-radius: 24px; }")
        trend_layout = QVBoxLayout(trend_group)
        trend_layout.setContentsMargins(22, 20, 22, 22)
        trend_layout.setSpacing(16)

        trend_title = QLabel("区域分割趋势监控")
        trend_title.setAlignment(Qt.AlignCenter)
        trend_title.setStyleSheet("color: #0F172A; font-size: 20px; font-weight: 800; padding: 2px 0 6px 0;")
        trend_layout.addWidget(trend_title)

        charts_top_row = QHBoxLayout()
        charts_top_row.setSpacing(16)

        chart_card_style = "background: #FFFFFF; border: 1px solid #DDE7EF; border-radius: 18px;"

        self.area_chart_view = QChartView()
        self.area_chart_view.setMinimumHeight(310)
        self.area_chart_view.setRenderHint(QPainter.Antialiasing)
        self.area_chart_view.setStyleSheet(chart_card_style)
        charts_top_row.addWidget(self.area_chart_view, 1)

        self.ratio_chart_view = QChartView()
        self.ratio_chart_view.setMinimumHeight(310)
        self.ratio_chart_view.setRenderHint(QPainter.Antialiasing)
        self.ratio_chart_view.setStyleSheet(chart_card_style)
        charts_top_row.addWidget(self.ratio_chart_view, 1)

        self.time_chart_view = QChartView()
        self.time_chart_view.setMinimumHeight(360)
        self.time_chart_view.setRenderHint(QPainter.Antialiasing)
        self.time_chart_view.setStyleSheet(chart_card_style)

        trend_layout.addLayout(charts_top_row)
        trend_layout.addWidget(self.time_chart_view)
        content_layout.addWidget(trend_group)
        content_layout.addStretch()
        scroll_area.setWidget(content)
        layout.addWidget(scroll_area)
        return panel

    def _build_history_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        toolbar = QWidget()
        toolbar.setStyleSheet("""
            QWidget {
                background-color: #FFFFFF;
                border-radius: 12px;
                border: 1px solid #E2E8F0;
            }
        """)
        bar = QHBoxLayout(toolbar)
        bar.setContentsMargins(16, 12, 16, 12)
        bar.setSpacing(12)

        title_label = QLabel("推理历史")
        title_label.setStyleSheet("color: #1E293B; font-size: 14px; font-weight: 600;")
        bar.addWidget(title_label)

        bar.addStretch()

        export_btn = QPushButton("导出记录")
        export_btn.setStyleSheet("""
            QPushButton {
                background-color: #F1F5F9;
                color: #475569;
                border: none;
                border-radius: 6px;
                padding: 6px 16px;
                font-size: 12px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #E2E8F0;
            }
        """)
        export_btn.clicked.connect(self._on_export_history)
        bar.addWidget(export_btn)

        clear_btn = QPushButton("清空历史")
        clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #FEF2F2;
                color: #EF4444;
                border: none;
                border-radius: 6px;
                padding: 6px 16px;
                font-size: 12px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #FEE2E2;
            }
        """)
        clear_btn.clicked.connect(self._on_clear_history)
        bar.addWidget(clear_btn)

        layout.addWidget(toolbar)

        table_container = QWidget()
        table_container.setStyleSheet("""
            QWidget {
                background-color: #FFFFFF;
                border-radius: 12px;
                border: 1px solid #E2E8F0;
            }
        """)
        table_layout = QVBoxLayout(table_container)
        table_layout.setContentsMargins(16, 16, 16, 16)

        self.history_table = QTableWidget(0, 6)
        self.history_table.setHorizontalHeaderLabels([
            "时间", "文件名", "模型", "区域面积(px)", "区域占比", "推理耗时(ms)"
        ])
        self.history_table.horizontalHeader().setStretchLastSection(True)
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.history_table.setAlternatingRowColors(True)
        self.history_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.history_table.cellClicked.connect(self._on_history_clicked)
        self.history_table.setStyleSheet("""
            QTableWidget {
                background-color: #F8FAFC;
                border: none;
                border-radius: 8px;
                gridline-color: #E2E8F0;
            }
            QTableWidget::item {
                padding: 8px 12px;
                color: #475569;
            }
            QTableWidget::item:selected {
                background-color: #DBEAFE;
                color: #1D4ED8;
            }
            QHeaderView::section {
                background-color: #F1F5F9;
                border: none;
                border-bottom: 2px solid #E2E8F0;
                padding: 8px 12px;
                font-weight: 600;
                color: #64748B;
            }
        """)
        table_layout.addWidget(self.history_table)
        layout.addWidget(table_container)

        return panel

    def _on_connect(self) -> None:
        url = self.api_url_edit.text().strip()
        if not url:
            QMessageBox.warning(self, "提示", "请输入服务器地址")
            return

        self._client = SegmentationApiClient(url)
        self.conn_status.setText("⏳ 正在检测...")
        self.conn_status.setStyleSheet("color: #F59E0B; font-size: 12px;")
        QApplication.processEvents()

        if self._client.health():
            self._settings.setValue("api_base_url", url)
            self.conn_status.setText("● 已连接")
            self.conn_status.setStyleSheet("color: #10B981; font-weight: bold; font-size: 12px;")
            self.status_bar.showMessage(f"已连接至 {url}")
        else:
            self.conn_status.setText("● 连接失败")
            self.conn_status.setStyleSheet("color: #EF4444; font-size: 12px;")
            self.status_bar.showMessage("服务器连接失败，请检查地址及网络")

    def _on_import_images(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "选择医学图像",
            "",
            "图像文件 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp);;所有文件 (*)",
        )
        if not files:
            return

        invalid_rows = []
        for i in range(self.image_list.count()):
            existing_path = self.image_list.item(i).data(Qt.UserRole)
            if existing_path and not os.path.exists(existing_path):
                invalid_rows.append(i)
        for i in reversed(invalid_rows):
            self.image_list.takeItem(i)

        for file_path in files:
            for i in range(self.image_list.count()):
                if self.image_list.item(i).data(Qt.UserRole) == file_path:
                    break
            else:
                item = QListWidgetItem(os.path.basename(file_path))
                item.setData(Qt.UserRole, file_path)
                self.image_list.addItem(item)

        last_path = files[-1]
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            if item.data(Qt.UserRole) == last_path:
                self.image_list.setCurrentRow(i)
                self._on_image_selected(item)
                break

    def _on_image_selected(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.UserRole)
        pix = load_pixmap(path)
        if pix.isNull():
            self._current_image = ""
            self.infer_btn.setEnabled(False)
            self.preview_label.clear()
            self.preview_label.setText("请导入医学图像")
            QMessageBox.warning(self, "警告", f"无法加载图像: {path}")
            return

        self._current_image = path
        set_label_pixmap(self.preview_label, pix)
        self.infer_btn.setEnabled(True)
        self.status_bar.showMessage(f"已选择: {os.path.basename(path)}")

    def _on_infer(self) -> None:
        if not self._current_image:
            return

        self.infer_btn.setEnabled(False)
        self.infer_btn.setText("推理中...")
        self.status_bar.showMessage("正在推理，请稍候...")
        QApplication.processEvents()

        self._infer_thread = _InferThread(
            self._client,
            self._current_image,
            self.threshold_spin.value(),
            "u-lite",
            self,
        )
        self._infer_thread.finished_signal.connect(self._on_infer_done)
        self._infer_thread.error_signal.connect(self._on_infer_error)
        self._infer_thread.start()

    def _on_infer_done(self, result: dict[str, Any]) -> None:
        self.infer_btn.setEnabled(True)
        self.infer_btn.setText("开始推理")

        mask_b64 = result.get("mask_base64", "")
        img_w = result.get("image_width", 0)
        img_h = result.get("image_height", 0)
        original_image_url = result.get("original_image_url", "")
        self.orig_label.clear()
        self.result_label.clear()

        def _render_results() -> None:
            errors: list[str] = []

            if mask_b64 and img_w and img_h:
                try:
                    binary_pix = build_binary_mask_pixmap(mask_b64, img_w, img_h)
                    set_label_pixmap(self.orig_label, binary_pix)
                except Exception as exc:
                    errors.append(f"二值化 mask 渲染失败: {exc}")

                if self._current_image and os.path.exists(self._current_image):
                    try:
                        orig_pix = load_pixmap(self._current_image)
                        if not orig_pix.isNull():
                            result_pix = build_overlay_pixmap(orig_pix, mask_b64, img_w, img_h)
                            set_label_pixmap(self.result_label, result_pix)
                    except Exception as exc:
                        errors.append(f"叠加渲染失败: {exc}")
            else:
                errors.append("后端未返回有效的 mask_base64")

            if errors:
                self.status_bar.showMessage("；".join(errors))

        QTimer.singleShot(100, _render_results)

        metrics = [
            ("推理耗时", f"{result.get('inference_time_ms', 0)} ms"),
            ("区域像素面积", str(result.get("lesion_area", 0))),
            ("区域占比", f"{result.get('lesion_ratio', 0):.6f}"),
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
            "original_image_path": self._current_image,
            "original_image_url": original_image_url,
            "mask_base64": mask_b64,
            "image_width": img_w,
            "image_height": img_h,
        }
        self._history.insert(0, record)
        _save_history(self._history)
        self._refresh_history_table()
        self._refresh_trend_charts()
        self.status_bar.showMessage(f"推理完成 - 耗时 {result.get('inference_time_ms', 0)} ms")

    def _on_infer_error(self, msg: str) -> None:
        self.infer_btn.setEnabled(True)
        self.infer_btn.setText("开始推理")
        QMessageBox.critical(self, "推理失败", msg)
        self.status_bar.showMessage(f"推理失败: {msg}")

    def _on_history_clicked(self, row: int, _column: int) -> None:
        if row < 0 or row >= len(self._history):
            return

        record = self._history[row]
        original_path = record.get("original_image_path", "")
        if original_path and os.path.exists(original_path):
            pix = load_pixmap(original_path)
            if not pix.isNull():
                self._current_image = original_path
                set_label_pixmap(self.preview_label, pix)
                self.infer_btn.setEnabled(True)

        self.orig_label.clear()
        self.result_label.clear()

        mask_b64 = record.get("mask_base64", "")
        img_w = record.get("image_width", 0)
        img_h = record.get("image_height", 0)

        if mask_b64 and img_w and img_h:
            try:
                binary_pix = build_binary_mask_pixmap(mask_b64, img_w, img_h)
                set_label_pixmap(self.orig_label, binary_pix)
            except Exception as exc:
                self.status_bar.showMessage(f"历史二值化 mask 渲染失败: {exc}")

            if original_path and os.path.exists(original_path):
                try:
                    orig_pix = load_pixmap(original_path)
                    if not orig_pix.isNull():
                        result_pix = build_overlay_pixmap(orig_pix, mask_b64, img_w, img_h)
                        set_label_pixmap(self.result_label, result_pix)
                except Exception as exc:
                    self.status_bar.showMessage(f"历史叠加渲染失败: {exc}")

        metrics = [
            ("推理耗时", f"{record.get('inference_time_ms', 0)} ms"),
            ("区域像素面积", str(record.get("lesion_area", 0))),
            ("区域占比", f"{record.get('lesion_ratio', 0):.6f}"),
            ("使用模型", record.get("model", "")),
        ]
        self.metrics_table.setRowCount(len(metrics))
        for idx, (label, value) in enumerate(metrics):
            self.metrics_table.setItem(idx, 0, QTableWidgetItem(label))
            self.metrics_table.setItem(idx, 1, QTableWidgetItem(value))

        self._refresh_trend_charts()
        self.status_bar.showMessage(f"已加载历史记录: {record.get('filename', '')}")

    def _refresh_history_table(self) -> None:
        self.history_table.setRowCount(len(self._history))
        for idx, rec in enumerate(self._history):
            self.history_table.setItem(idx, 0, QTableWidgetItem(rec.get("timestamp", "")))
            self.history_table.setItem(idx, 1, QTableWidgetItem(rec.get("filename", "")))
            self.history_table.setItem(idx, 2, QTableWidgetItem(rec.get("model", "")))
            self.history_table.setItem(idx, 3, QTableWidgetItem(str(rec.get("lesion_area", ""))))
            self.history_table.setItem(idx, 4, QTableWidgetItem(str(rec.get("lesion_ratio", ""))))
            self.history_table.setItem(idx, 5, QTableWidgetItem(str(rec.get("inference_time_ms", ""))))

    def _build_trend_chart(self, title: str, records: list[dict[str, Any]], value_key: str, color: QColor, y_title: str, is_percent: bool = False) -> QChart:
        chart = QChart()
        chart.setTitle(title)
        chart.legend().hide()
        chart.setBackgroundVisible(False)
        chart.setPlotAreaBackgroundVisible(True)
        chart.setPlotAreaBackgroundBrush(QColor("#FBFDFE"))
        chart.setAnimationOptions(QChart.SeriesAnimations)
        chart.setMargins(type(chart.margins())(28, 22, 28, 48))
        chart.setTitleBrush(QColor("#0F172A"))
        chart.setTitleFont(QFont("Microsoft YaHei", 12, QFont.Bold))

        if not records:
            chart.setTitle(f"{title}（暂无历史数据）")
            return chart

        line_series = QLineSeries()
        line_pen = QPen(color)
        line_pen.setWidthF(3.0)
        line_pen.setCapStyle(Qt.RoundCap)
        line_pen.setJoinStyle(Qt.RoundJoin)
        line_series.setPen(line_pen)

        marker_series = QScatterSeries()
        marker_series.setMarkerShape(QScatterSeries.MarkerShapeCircle)
        marker_series.setMarkerSize(10.5)
        marker_series.setColor(QColor("#FFFFFF"))
        marker_series.setBorderColor(QColor(color))

        values: list[float] = []
        timestamps: list[int] = []
        point_infos: dict[tuple[int, float], str] = {}

        for record in records:
            ts = record.get("timestamp", "")
            value = float(record.get(value_key, 0) or 0)
            if is_percent:
                value *= 100.0
            dt = QDateTime.fromString(ts, "yyyy-MM-dd HH:mm:ss")
            if not dt.isValid():
                continue
            ms = dt.toMSecsSinceEpoch()
            timestamps.append(ms)
            values.append(value)
            line_series.append(ms, value)
            marker_series.append(ms, value)
            value_text = f"{value:.2f}%" if is_percent else f"{value:.2f}"
            point_infos[(ms, value)] = f"{title}｜推理时间: {ts}｜{y_title}: {value_text}"

        if not values or not timestamps:
            chart.setTitle(f"{title}（暂无历史数据）")
            return chart

        chart.addSeries(line_series)
        chart.addSeries(marker_series)

        axis_x = QDateTimeAxis()
        axis_x.setTitleText("推理时间")
        axis_x.setFormat("MM-dd\nHH:mm")
        axis_x.setRange(QDateTime.fromMSecsSinceEpoch(min(timestamps)), QDateTime.fromMSecsSinceEpoch(max(timestamps)))
        axis_x.setTickCount(max(2, min(len(timestamps), 4)))
        axis_x.setLabelsColor(QColor("#1F2937"))
        axis_x.setGridLineVisible(True)
        axis_x.setGridLineColor(QColor("#D9E2EA"))
        axis_x.setLineVisible(True)
        axis_x.setLinePen(QPen(QColor("#94A3B8"), 1))
        axis_x.setTitleBrush(QColor("#0F172A"))
        axis_x.setTitleFont(QFont("Microsoft YaHei", 9, QFont.Bold))
        axis_x.setLabelsFont(QFont("Microsoft YaHei", 8))

        min_value = min(values)
        max_value = max(values)
        if min_value == max_value:
            padding = max(1.0, abs(min_value) * 0.1 or 1.0)
            min_value -= padding
            max_value += padding
        else:
            padding = (max_value - min_value) * 0.1
            min_value -= padding
            max_value += padding

        if is_percent:
            min_value = max(0.0, min_value)

        axis_y = QValueAxis()
        axis_y.setTitleText(y_title)
        axis_y.setRange(min_value, max_value)
        axis_y.setTickCount(5)
        axis_y.setLabelsColor(QColor("#1F2937"))
        axis_y.setGridLineVisible(True)
        axis_y.setGridLineColor(QColor("#D9E2EA"))
        axis_y.setLineVisible(True)
        axis_y.setLinePen(QPen(QColor("#94A3B8"), 1))
        axis_y.setTitleBrush(QColor("#0F172A"))
        axis_y.setTitleFont(QFont("Microsoft YaHei", 9, QFont.Bold))
        axis_y.setLabelsFont(QFont("Microsoft YaHei", 8))

        chart.addAxis(axis_x, Qt.AlignBottom)
        chart.addAxis(axis_y, Qt.AlignLeft)
        line_series.attachAxis(axis_x)
        line_series.attachAxis(axis_y)
        marker_series.attachAxis(axis_x)
        marker_series.attachAxis(axis_y)

        def _show_point_info(point: QPointF) -> None:
            key = (int(point.x()), float(point.y()))
            self.status_bar.showMessage(point_infos.get(key, f"{title}｜数据点: ({point.x():.0f}, {point.y():.2f})"))

        marker_series.clicked.connect(_show_point_info)
        return chart

    def _refresh_trend_charts(self) -> None:
        recent = list(reversed(self._history[-12:]))
        self.area_chart_view.setChart(self._build_trend_chart("区域面积趋势分析", recent, "lesion_area", QColor("#111A5E"), "区域面积（px）"))
        self.ratio_chart_view.setChart(self._build_trend_chart("区域占比趋势分析", recent, "lesion_ratio", QColor("#3E8E88"), "区域占比（%）", is_percent=True))
        self.time_chart_view.setChart(self._build_trend_chart("推理耗时趋势分析", recent, "inference_time_ms", QColor("#D8732A"), "推理耗时（ms）"))

    def _on_clear_history(self) -> None:
        reply = QMessageBox.question(self, "确认", "确定要清空所有历史记录吗？")
        if reply == QMessageBox.Yes:
            self._history.clear()
            _save_history(self._history)
            self._refresh_history_table()
            self._refresh_trend_charts()

    def _on_export_history(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "导出历史记录",
            "inference_history.csv",
            "CSV (*.csv)",
        )
        if not path:
            return

        import csv

        with open(path, "w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.writer(fh)
            writer.writerow(["时间", "文件名", "模型", "区域面积(px)", "区域占比", "推理耗时(ms)"])
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
