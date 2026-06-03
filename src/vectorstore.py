"""ChromaDB 向量庫與 HuggingFace embedding 的封裝。

知識庫『依 session 分離』：每位使用者(session_id)有自己的 Chroma collection，
彼此的教材互不可見、互不影響。embedding 模型則全域共用（無狀態）。
"""
from __future__ import annotations

from functools import lru_cache

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from . import config


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """載入 embedding 模型（首次呼叫會下載權重，之後快取於記憶體）。"""
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def _collection_name(session_id: str) -> str:
    """把 session_id 轉成合法的 Chroma collection 名稱（3-63 字、英數/底線）。"""
    sid = "".join(c for c in (session_id or "default") if c.isalnum()) or "default"
    return f"{config.COLLECTION_NAME}_{sid}"[:63]


@lru_cache(maxsize=128)
def get_vectorstore(session_id: str) -> Chroma:
    """取得（或建立）該 session 專屬的持久化 Chroma 向量庫。"""
    return Chroma(
        collection_name=_collection_name(session_id),
        embedding_function=get_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
    )


def get_retriever(session_id: str, k: int = config.RETRIEVE_K):
    """回傳該 session 的相似度檢索器（跨該使用者所有已上傳文件）。"""
    return get_vectorstore(session_id).as_retriever(search_kwargs={"k": k})


def document_count(session_id: str) -> int:
    """該 session 向量庫中的 chunk 數量。"""
    try:
        return get_vectorstore(session_id)._collection.count()
    except Exception:
        return 0


def list_sources(session_id: str) -> dict[str, int]:
    """回傳該 session 的 {來源檔名: 片段數}，依檔名排序。"""
    try:
        data = get_vectorstore(session_id)._collection.get(include=["metadatas"])
    except Exception:
        return {}
    counts: dict[str, int] = {}
    for m in data.get("metadatas") or []:
        src = (m or {}).get("source", "未知")
        counts[src] = counts.get(src, 0) + 1
    return dict(sorted(counts.items()))


def delete_source(session_id: str, name: str) -> int:
    """刪除該 session 某來源檔的所有片段，回傳刪除前該檔的片段數。"""
    vs = get_vectorstore(session_id)
    before = len(vs._collection.get(where={"source": name}).get("ids") or [])
    if before:
        vs._collection.delete(where={"source": name})
    return before
