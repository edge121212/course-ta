"""課程助教 Agent：規劃 → 分步執行 → 整合（功能 1）。

與一般「單次分流」不同，這裡讓 LLM 先把使用者需求拆解成步驟：
- 單一簡單需求 → 一個步驟（行為等同原本的問答/摘要/出題）。
- 複合需求（比較多章、又問又出題、跨章節整理）→ 多步驟，逐步執行後再整合成一份回答。

問答步驟會套用 workflows.verify 做答案自我查核（功能 2）。
"""
from __future__ import annotations

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from . import workflows
from .llm import get_llm
from .vectorstore import get_retriever


class Step(BaseModel):
    action: Literal["qa", "summary", "quiz"] = Field(description="這一步的任務類型")
    query: str = Field(description="這一步要檢索與處理的子問題或主題")
    note: str = Field(description="這一步在做什麼，用一句話描述")


class Plan(BaseModel):
    steps: list[Step]


_PLAN_SYS = (
    "你是課程助教的任務規劃器。把使用者需求拆成『最少的必要步驟』。"
    "若是單一簡單需求（只問一個問題、只要一份摘要、或只要出一次題）→ 只回傳一個步驟。"
    "若包含多個子任務（例如：比較不同章節、同時又問又出題、跨章節整理重點）→ 才拆成多步。"
    "每步指定 action：qa(問答) / summary(摘要) / quiz(出題)；query(該步要檢索的主題或子問題)；"
    "note(這步在做什麼)。步驟最多 4 步。請用繁體中文。"
)


def make_plan(question: str, api_key: str | None = None,
              model: str | None = None) -> list[dict]:
    llm = get_llm(api_key, model).with_structured_output(Plan)
    plan: Plan = llm.invoke([("system", _PLAN_SYS), ("human", f"使用者需求：{question}")])
    steps = [s.model_dump() for s in plan.steps]
    return steps or [{"action": "qa", "query": question, "note": "回答問題"}]


def _retrieve(query: str):
    docs = get_retriever().invoke(query or "課程重點")
    sources = [
        {"source": d.metadata.get("source", "未知"), "page": d.metadata.get("page", "?")}
        for d in docs
    ]
    return docs, sources


def _exec_step(step: dict, api_key, model, verify_qa: bool) -> dict:
    docs, sources = _retrieve(step["query"])
    action = step["action"]
    grounded = None
    if action == "summary":
        result = workflows.summarize(step["query"], docs, api_key, model)
    elif action == "quiz":
        result = workflows.generate_quiz(step["query"], docs, api_key=api_key, model=model)
    else:  # qa
        result = workflows.qa(step["query"], docs, api_key, model)
        if verify_qa:
            v = workflows.verify(step["query"], result, docs, api_key, model)
            result, grounded = v["answer"], v["grounded"]
    return {"action": action, "note": step["note"], "result": result,
            "sources": sources, "grounded": grounded}


_SYNTH_SYS = (
    "把以下各步驟的結果，整合成一份條理清晰、直接回應使用者『原始問題』的完整回答。"
    "用小標題分段、保持精簡、不要重複內容。請用繁體中文。"
)


def _synthesize(question: str, results: list[dict], api_key, model) -> str:
    blocks = "\n\n".join(f"【{r['note']}】\n{r['result']}" for r in results)
    msgs = ChatPromptTemplate.from_messages([
        ("system", _SYNTH_SYS),
        ("human", "原始問題：{q}\n\n各步驟結果：\n{b}"),
    ]).format_messages(q=question, b=blocks)
    return get_llm(api_key, model).invoke(msgs).content


def run_agent(question: str, api_key: str | None = None, model: str | None = None,
              verify_qa: bool = True) -> dict:
    """規劃並執行，回傳 {answer, steps, sources, grounded, multi}。"""
    steps = make_plan(question, api_key, model)
    results = [_exec_step(s, api_key, model, verify_qa) for s in steps]
    multi = len(results) > 1

    answer = _synthesize(question, results, api_key, model) if multi else results[0]["result"]

    # 合併來源（去重）
    sources, seen = [], set()
    for r in results:
        for s in r["sources"]:
            key = (s["source"], s["page"])
            if key not in seen:
                seen.add(key)
                sources.append(s)

    flags = [r["grounded"] for r in results if r["grounded"] is not None]
    grounded = all(flags) if flags else None

    return {
        "answer": answer,
        "steps": [{"action": r["action"], "note": r["note"]} for r in results],
        "sources": sources,
        "grounded": grounded,
        "multi": multi,
    }
