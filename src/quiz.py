"""適性化測驗：結構化出題、計分、依弱點觀念出下一份測驗。

系統目標：「讓使用者變強」——記錄學生答錯的觀念，下一份測驗針對弱點加強。
出題使用 Gemini 結構化輸出，每題帶 `concept`（觀念標籤），以利後續成效分析。
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from . import db
from .llm import get_llm
from .vectorstore import get_retriever, get_vectorstore
from .workflows import format_context


class QuizQuestion(BaseModel):
    concept: str = Field(description="此題對應的核心觀念標籤，例如『記憶體階層』")
    question: str = Field(description="題目敘述")
    options: dict[str, str] = Field(
        description="選項，鍵為 A/B/C/D，值為選項文字；至少四個"
    )
    answer: str = Field(description="正確選項的字母，例如 'A'")
    explanation: str = Field(description="簡短解析，說明為什麼是正解")


class Quiz(BaseModel):
    questions: list[QuizQuestion]


_SYSTEM = (
    "你是課程出題助教。請『僅根據提供的教材內容』出 {num} 題單選選擇題（每題四個選項 A-D，"
    "只有一個正解）。每題都要標註對應的核心觀念(concept)。題目須源自教材，勿超出教材範圍。"
    "{focus}請用繁體中文。"
)


def _retrieve_for(query: str, k: int = 6):
    return get_retriever(k=k).invoke(query or "課程重點")


# ---------------------------------------------------------------------------
# 讓助教讀教材、列出可測主題（給 UI 字卡選擇用）
# ---------------------------------------------------------------------------
class _Topics(BaseModel):
    topics: list[str] = Field(description="適合出測驗的核心主題清單，每個簡短")


def suggest_topics(api_key: str | None = None, model: str | None = None,
                   n: int = 8) -> list[str]:
    """讀取已上傳教材，回傳 n 個適合出測驗的核心主題（給字卡選擇）。"""
    data = get_vectorstore()._collection.get(include=["documents"])
    docs = data.get("documents") or []
    if not docs:
        return []
    # 在整份教材均勻取樣，避免只看到開頭
    stride = max(1, len(docs) // 40)
    sample = docs[::stride][:40]
    corpus = "\n---\n".join((d or "")[:300] for d in sample)

    llm = get_llm(api_key, model).with_structured_output(_Topics)
    system = (
        f"你是課程助教。根據以下教材片段，歸納出 {n} 個最適合用來出測驗的『核心主題／觀念』。"
        "每個主題簡短（不超過 12 字）、彼此不重複、由重要到次要。請用繁體中文。"
    )
    result: _Topics = llm.invoke([("system", system), ("human", f"教材片段：\n{corpus}")])
    # 去重並截斷
    seen, out = set(), []
    for t in result.topics:
        t = (t or "").strip()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out[:n]


def generate(topic: str, num: int = 5, api_key: str | None = None,
             model: str | None = None,
             focus_concepts: list[str] | None = None) -> dict:
    """產生一份結構化測驗。

    回傳 {questions: [..], sources: [..], focus: [..]}。
    focus_concepts 有值時，會偏向針對這些弱點觀念出題（適性模式）。
    """
    # 檢索查詢：適性模式用弱點觀念串起來，否則用主題
    query = "、".join(focus_concepts) if focus_concepts else topic
    docs = _retrieve_for(query)
    sources = [
        {"source": d.metadata.get("source", "未知"), "page": d.metadata.get("page", "?")}
        for d in docs
    ]

    if focus_concepts:
        focus = (f"請『優先針對這些學生較弱的觀念』加強出題：{('、'.join(focus_concepts))}；"
                 "可少量加入其他重點觀念複習。")
    else:
        focus = ""

    system = _SYSTEM.format(num=num, focus=focus)
    llm = get_llm(api_key, model).with_structured_output(Quiz)
    human = (f"教材內容：\n{format_context(docs)}\n\n"
             f"出題主題：{topic or '課程重點'}")
    result: Quiz = llm.invoke([("system", system), ("human", human)])

    questions = [q.model_dump() for q in result.questions]
    return {"questions": questions, "sources": sources, "focus": focus_concepts or []}


def generate_adaptive(session_id: str, topic: str = "", num: int = 5,
                      api_key: str | None = None, model: str | None = None) -> dict:
    """依該 session 的弱點觀念出下一份測驗。"""
    weak = db.weak_concepts(session_id, limit=5)
    return generate(topic or "課程重點", num=num, api_key=api_key, model=model,
                    focus_concepts=weak or None)


def grade(questions: list[dict], user_answers: list[str]) -> list[dict]:
    """比對作答與正解，回傳 responses：[{q_index, user_answer, is_correct}]。"""
    responses = []
    for i, q in enumerate(questions):
        ua = (user_answers[i] or "").strip().upper()[:1] if i < len(user_answers) else ""
        correct = ua == (q.get("answer", "") or "").strip().upper()[:1]
        responses.append({"q_index": i, "user_answer": ua, "is_correct": correct})
    return responses
