---
title: 智慧課程助教系統
emoji: 🎓
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
short_description: RAG 課程助教 — 問答/摘要/適性測驗與學習成效分析
---

# 智慧課程助教系統（Intelligent Course TA System）

依《生成式 AI 系統設計與實務 期中報告》(111704028 夏銜呈) 規格實作的核心 MVP。
讓學生以自然語言查詢課程教材，並提供 **課程問答（含引用來源）**、**重點摘要**、**練習題生成**。

## 架構

| 層 | 技術 |
|---|---|
| UI | Streamlit |
| Orchestrator（任務分流） | LangGraph（規則式 router → 問答／摘要／出題節點） |
| LLM | Google Gemini（`langchain-google-genai`） |
| RAG Pipeline | LangChain + HuggingFace Embedding + ChromaDB |
| 對話紀錄 | SQLite |

流程：上傳 PDF → 切割(Chunking) → 嵌入(Embedding) → ChromaDB → 檢索 → 依任務分流 → Gemini 生成。

## 安裝

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> 首次執行會自動下載 HuggingFace embedding 模型權重（約 400MB，需網路）。

### Gemini API Key
key 在**網頁左側欄位輸入**即可（只存在該瀏覽器分頁，關閉或重新整理後需重新輸入，不會寫入磁碟）。
取得：https://aistudio.google.com/app/apikey

> 進階：若想設成共用、免每次輸入，可 `cp .env.example .env` 並填入 `GOOGLE_API_KEY`；
> UI 有輸入時以 UI 為優先，否則退回 `.env`。

## 執行

```bash
streamlit run app.py
```

在左側上傳課程 PDF 並按「匯入教材」，即可在主畫面對話。範例輸入：

- `什麼是 RAG？` → 課程問答，附引用來源
- `幫我整理第 3 章重點` → 重點摘要
- `幫我出 5 題選擇題` → 練習題（含參考答案）

## 煙霧測試（不需 API key）

驗證 RAG 檢索與任務分流：

```bash
python scripts/smoke_test.py
```

## 專案結構

```
app.py              Streamlit 介面
src/
  config.py         路徑／模型／env 設定
  ingest.py         PDF 載入 → 切割 → 寫入向量庫
  vectorstore.py    ChromaDB + HuggingFace embedding
  llm.py            Gemini 初始化
  workflows.py      問答／摘要／出題 prompt chain
  graph.py          LangGraph orchestrator（規則式分流）
  db.py             SQLite 對話紀錄
scripts/smoke_test.py
data/               執行時產生（uploads / chroma / app.db）
```

## MVP 範圍與後續

已實作：PDF 上傳、自動切割／嵌入、多文件檢索、問答+引用、摘要、出題、LangGraph 分流、對話紀錄保存。
後續可擴充：PPTX 支援、角色／權限管理、回應串流、UI 精修。
