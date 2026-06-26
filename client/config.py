import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_API_BASE = "http://127.0.0.1:8000"
WINDOW_TITLE = "医学影像病灶分割分析系统"
CLIENT_RESULT_DIR = os.path.join(PROJECT_ROOT, "client_result")
