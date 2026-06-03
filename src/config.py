"""集中管理路徑、模型名稱與環境變數。"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# 載入專案根目錄的 .env（若存在）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# ---- 資料夾 ----
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma"
DB_PATH = DATA_DIR / "app.db"

for _d in (DATA_DIR, UPLOAD_DIR, CHROMA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---- 模型設定 ----
# 教材為繁體中文，使用多語 embedding 模型
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# UI 下拉可選的常見模型（第一個為預設）
GEMINI_MODEL_CHOICES = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
    "gemini-flash-latest",
]

# ChromaDB collection 名稱（單一 collection 支援多文件檢索）
COLLECTION_NAME = "course_materials"

# ---- 檢索 / 切割參數 ----
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
RETRIEVE_K = 4


def google_api_key() -> str | None:
    """回傳 Gemini 用的 API key（支援兩種常見變數名）。"""
    return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
