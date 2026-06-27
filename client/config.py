import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_API_BASE = "http://127.0.0.1:8000"
WINDOW_TITLE = "膀胱超声图像处理"
CLIENT_RESULT_DIR = os.path.join(PROJECT_ROOT, "client_result")
