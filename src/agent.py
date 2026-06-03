"""課程助教 Agent。

能力：
- 規劃 → 分步執行 → 整合（複合需求）
- A. 反問澄清：需求太模糊時先反問，不硬答
- B. 自我修正檢索：問答依據不足時自動換關鍵字再查一次
- C. 個人化考前衝刺：讀取學生過去測驗的弱點，排重點＋針對弱點出題
- D. 進度回報：透過 progress callback 即時回報每一步在做什麼

問答步驟會套用 workflows.verify 做答案自我查核。
"""
from __future__ import annotations

from typing import Callable, Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from . import db, workflows
from .llm import get_llm
from .vectorstore import get_retriever

Progress = Callable[[str], None]


def _noop(_msg: str) -> None:
    pass


# ---------------------------------------------------------------------------
# 規劃（含反問澄清）
# ---------------------------------------------------------------------------
class Step(BaseModel):
    action: Literal["qa", "summary", "quiz", "exam_prep"] = Field(description="任務類型")
    query: str = Field(description="這一步要檢索與處理的子問題或主題")
    note: str = Field(description="這一步在做什麼，用一句話描述")


class Plan(BaseModel):
    clarify: str = Field(
        default="",
        description="若需求太模糊、有多種明顯不同的解讀，這裡填一句『反問使用者』的澄清問題；否則務必留空字串",
    )
    steps: list[Step] = Field(default_factory=list)


_PLAN_SYS = (
    "你是課程助教的任務規劃器，要把使用者需求拆成『最少的必要步驟』。規則：\n"
    "1. 若需求明顯模糊、有多種完全不同的解讀（例如同一詞有多種意思、範圍不清）→ 只填 clarify "
    "（一句反問），steps 留空。不要為了反問而反問，能合理推斷就直接做。\n"
    "2. 若使用者想『準備考試 / 複習 / 考前衝刺 / 幫我準備』→ 用一個 exam_prep 步驟，query 放範圍。\n"
    "3. 單一簡單需求 → 一個步驟（qa 問答 / summary 摘要 / quiz 出題）。\n"
    "4. 複合需求（比較不同章節、又問又出題、跨章節整理）→ 拆成多步，最多 4 步。\n"
    "請用繁體中文。"
)


def make_plan(question: str, api_key=None, model=None) -> dict:
    llm = get_llm(api_key, model).with_structured_output(Plan)
    plan: Plan = llm.invoke([("system", _PLAN_SYS), ("human", f"使用者需求：{question}")])
    steps = [s.model_dump() for s in plan.steps]
    clarify = (plan.clarify or "").strip()
    if not clarify and not steps:
        steps = [{"action": "qa", "query": question, "note": "回答問題"}]
    return {"clarify": clarify, "steps": steps}


# ---------------------------------------------------------------------------
# 檢索 + 自我修正（B）
# ---------------------------------------------------------------------------
def _retrieve(query: str):
    docs = get_retriever().invoke(query or "課程重點")
    sources = [
        {"source": d.metadata.get("source", "未知"), "page": d.metadata.get("page", "?")}
        for d in docs
    ]
    return docs, sources


def _rewrite_query(question: str, api_key, model) -> str:
    """產生替代搜尋關鍵字，用於自我修正檢索。"""
    msgs = [
        ("system", "你是檢索關鍵字優化器。使用者問題用原本關鍵字檢索教材效果不佳，"
                   "請改寫成更適合向量檢索的關鍵詞（同義詞、更具體的學術用語），只回傳關鍵字，不要解釋。"),
        ("human", question),
    ]
    return get_llm(api_key, model).invoke(msgs).content.strip()


def _qa_step(query: str, api_key, model, verify_qa: bool, prog: Progress):
    """問答 + 自我查核 + 自我修正檢索（依據不足時換關鍵字再查一次）。"""
    docs, sources = _retrieve(query)
    answer = workflows.qa(query, docs, api_key, model)
    grounded = None
    if verify_qa:
        v = workflows.verify(query, answer, docs, api_key, model)
        answer, grounded = v["answer"], v["grounded"]
        if grounded is False:
            prog("教材依據不足，換個關鍵字再查一次…")
            nq = _rewrite_query(query, api_key, model)
            docs2, sources2 = _retrieve(nq)
            ans2 = workflows.qa(query, docs2, api_key, model)
            v2 = workflows.verify(query, ans2, docs2, api_key, model)
            if v2["grounded"] or len(docs2):
                answer, grounded, sources = v2["answer"], v2["grounded"], sources2
    return answer, sources, grounded


# ---------------------------------------------------------------------------
# 個人化考前衝刺（C）
# ---------------------------------------------------------------------------
_EXAM_SYS = (
    "你是個人化考前助教。根據『教材內容』與『學生的弱點觀念』，產出一份考前複習指引："
    "①先針對學生的弱點觀念做重點講解與提醒；②列出該範圍的必考重點；③給出具體複習建議。"
    "若沒有弱點資料，就做一般重點整理並提醒可先做測驗以找出弱點。請用繁體中文、條列清晰。"
)


def _exam_prep_step(query: str, session_id: str, api_key, model, prog: Progress):
    weak = db.weak_concepts(session_id, limit=6)
    prog("讀取你的測驗弱點…" + ("（" + "、".join(weak) + "）" if weak else "（尚無紀錄）"))
    topics = "、".join(weak) if weak else query
    docs, sources = _retrieve(f"{query} {topics}")
    msgs = ChatPromptTemplate.from_messages([
        ("system", _EXAM_SYS),
        ("human", "教材內容：\n{context}\n\n複習範圍：{scope}\n\n學生的弱點觀念：{weak}"),
    ]).format_messages(
        context=workflows.format_context(docs),
        scope=query or "全部範圍",
        weak="、".join(weak) if weak else "（尚無測驗紀錄）",
    )
    review = get_llm(api_key, model).invoke(msgs).content

    prog("針對弱點出練習題…")
    quiz_text = workflows.generate_quiz(
        topics, docs, api_key=api_key, model=model, num=5, qtype="選擇題")

    body = f"### 考前複習指引\n{review}\n\n### 針對弱點的練習題\n{quiz_text}"
    if weak:
        body = f"_本指引已針對你較弱的觀念：{('、'.join(weak))}_\n\n" + body
    return body, sources


# ---------------------------------------------------------------------------
# 執行單一步驟
# ---------------------------------------------------------------------------
def _exec_step(step: dict, session_id: str, api_key, model,
               verify_qa: bool, prog: Progress) -> dict:
    action = step["action"]
    grounded = None
    if action == "summary":
        docs, sources = _retrieve(step["query"])
        result = workflows.summarize(step["query"], docs, api_key, model)
    elif action == "quiz":
        docs, sources = _retrieve(step["query"])
        result = workflows.generate_quiz(step["query"], docs, api_key=api_key, model=model)
    elif action == "exam_prep":
        result, sources = _exam_prep_step(step["query"], session_id, api_key, model, prog)
    else:  # qa
        result, sources, grounded = _qa_step(step["query"], api_key, model, verify_qa, prog)
    return {"action": action, "note": step["note"], "result": result,
            "sources": sources, "grounded": grounded}


# ---------------------------------------------------------------------------
# 整合
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def run_agent(question: str, session_id: str, api_key: str | None = None,
              model: str | None = None, verify_qa: bool = True,
              progress: Progress | None = None) -> dict:
    prog = progress or _noop

    prog("規劃步驟…")
    plan = make_plan(question, api_key, model)

    # A. 反問澄清
    if plan["clarify"]:
        return {"answer": plan["clarify"], "steps": None, "sources": [],
                "grounded": None, "multi": False, "is_clarify": True}

    steps = plan["steps"]
    results = []
    for i, s in enumerate(steps):
        prog(f"步驟 {i+1}/{len(steps)}：{s['note']}")
        results.append(_exec_step(s, session_id, api_key, model, verify_qa, prog))

    multi = len(results) > 1
    if multi:
        prog("整合結果…")
        answer = _synthesize(question, results, api_key, model)
    else:
        answer = results[0]["result"]

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
        "is_clarify": False,
    }
