"""文件匯入：載入 PDF → 切割(Chunking) → 寫入向量庫(Embedding)。"""
from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config
from .vectorstore import get_vectorstore


def _splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "．", " ", ""],
    )


def ingest_pdf(path: str | Path) -> int:
    """匯入單一 PDF，回傳新增的 chunk 數量。

    metadata 會保留來源檔名與頁碼，供問答時顯示「引用來源」。
    """
    path = Path(path)
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"目前 MVP 僅支援 PDF，收到：{path.suffix}")

    pages = PyPDFLoader(str(path)).load()  # 每頁一個 Document，含 page metadata
    chunks = _splitter().split_documents(pages)

    for c in chunks:
        c.metadata["source"] = path.name
        # PyPDF 的 page 由 0 起算，轉成人類習慣的頁碼
        if "page" in c.metadata:
            c.metadata["page"] = int(c.metadata["page"]) + 1

    if chunks:
        get_vectorstore().add_documents(chunks)
    return len(chunks)
