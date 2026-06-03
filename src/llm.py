"""初始化 Gemini LLM。"""
from __future__ import annotations

from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI

from . import config


class MissingAPIKeyError(RuntimeError):
    """缺少 Gemini API key 時拋出。"""


@lru_cache(maxsize=16)
def _build(api_key: str, model: str) -> ChatGoogleGenerativeAI:
    """依 (key, model) 建立並快取 LLM。"""
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=0.3,
    )


def get_llm(api_key: str | None = None, model: str | None = None) -> ChatGoogleGenerativeAI:
    """取得 Gemini LLM。

    優先使用呼叫端傳入的 api_key／model（例如使用者在 UI 選的）；
    未提供則退回 .env / 預設值。
    """
    key = (api_key or "").strip() or config.google_api_key()
    if not key:
        raise MissingAPIKeyError(
            "尚未提供 Gemini API key，請在左側欄位輸入後再使用問答/摘要/出題功能。"
        )
    return _build(key, (model or "").strip() or config.GEMINI_MODEL)


def list_models(api_key: str) -> list[str]:
    """查詢這把 key 可用、且支援 generateContent 的模型名稱清單。"""
    from google import genai  # google-genai SDK

    client = genai.Client(api_key=api_key.strip())
    names: list[str] = []
    for m in client.models.list():
        actions = getattr(m, "supported_actions", None) or []
        if not actions or "generateContent" in actions:
            name = (m.name or "").removeprefix("models/")
            if name:
                names.append(name)
    return sorted(set(names))
