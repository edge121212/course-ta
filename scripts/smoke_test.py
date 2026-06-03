"""煙霧測試：驗證 ingest → 向量檢索 能正常運作（不需要 LLM key）。

用法： python scripts/smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.graph import classify          # noqa: E402
from src.ingest import ingest_pdf       # noqa: E402
from src.vectorstore import document_count, get_retriever  # noqa: E402

PDF = ROOT / "期中報告_111704028.pdf"


def main() -> int:
    print(f"[1] 匯入 PDF：{PDF.name}")
    n = ingest_pdf(PDF)
    print(f"    新增 {n} 個片段，知識庫共 {document_count()} 片段")
    assert n > 0, "切割後沒有任何片段"

    print("[2] 檢索測試：'什麼是 RAG'")
    docs = get_retriever().invoke("什麼是 RAG？")
    assert docs, "檢索沒有回傳任何結果"
    top = docs[0]
    print(f"    Top 命中：{top.metadata.get('source')} 第{top.metadata.get('page')}頁")
    print(f"    內容預覽：{top.page_content[:60].strip()}…")
    assert "RAG" in " ".join(d.page_content for d in docs), "檢索結果未包含 RAG 相關內容"

    print("[3] 規則式任務分流測試")
    cases = {
        "什麼是 RAG？": "qa",
        "幫我整理第 3 章重點": "summary",
        "幫我出 5 題選擇題": "quiz",
    }
    for q, expected in cases.items():
        got = classify(q)
        print(f"    {q!r} -> {got} (預期 {expected})")
        assert got == expected, f"分流錯誤：{q}"

    print("\n✅ 煙霧測試通過：RAG 檢索與任務分流皆正常。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
