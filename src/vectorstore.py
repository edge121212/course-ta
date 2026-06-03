"""ChromaDB 向量庫與 HuggingFace embedding 的封裝。"""
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


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    """取得（或建立）持久化的 Chroma 向量庫。"""
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
    )


def get_retriever(k: int = config.RETRIEVE_K):
    """回傳相似度檢索器；跨所有已上傳文件檢索（多文件檢索）。"""
    return get_vectorstore().as_retriever(search_kwargs={"k": k})


def document_count() -> int:
    """目前向量庫中的 chunk 數量。"""
    try:
        return get_vectorstore()._collection.count()
    except Exception:
        return 0


def list_sources() -> dict[str, int]:
    """回傳 {來源檔名: 片段數}，依檔名排序。"""
    try:
        data = get_vectorstore()._collection.get(include=["metadatas"])
    except Exception:
        return {}
    counts: dict[str, int] = {}
    for m in data.get("metadatas") or []:
        src = (m or {}).get("source", "未知")
        counts[src] = counts.get(src, 0) + 1
    return dict(sorted(counts.items()))


def delete_source(name: str) -> int:
    """刪除某來源檔的所有片段，回傳刪除前該檔的片段數。"""
    vs = get_vectorstore()
    before = len(vs._collection.get(where={"source": name}).get("ids") or [])
    if before:
        vs._collection.delete(where={"source": name})
    return before
