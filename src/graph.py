"""LangGraph Orchestrator：依任務分流到 問答 / 摘要 / 出題 節點。

依報告 Part 2 的設計，任務分流(Prompt Routing) 採用『規則式』判斷而非交由 LLM，
較為穩定可靠；再透過 LangGraph 的 conditional edge 導向對應節點。
"""
from __future__ import annotations

from typing import Literal, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from . import workflows
from .vectorstore import get_retriever

Task = Literal["qa", "summary", "quiz"]

# 關鍵字規則：命中即分流到對應任務
_SUMMARY_KW = ("摘要", "整理", "重點", "總結", "歸納", "summary")
_QUIZ_KW = ("出題", "題目", "考題", "練習題", "選擇題", "測驗", "quiz", "出.*題")


class AssistantState(TypedDict, total=False):
    question: str
    api_key: str
    model: str
    task: Task
    docs: list[Document]
    answer: str
    sources: list[dict]


def classify(question: str) -> Task:
    """規則式任務分類。"""
    q = question.lower()
    if any(k in question or k in q for k in _QUIZ_KW):
        return "quiz"
    if any(k in question or k in q for k in _SUMMARY_KW):
        return "summary"
    return "qa"


# ---- 節點 ----
def route_node(state: AssistantState) -> AssistantState:
    return {"task": classify(state["question"])}


def retrieve_node(state: AssistantState) -> AssistantState:
    docs = get_retriever().invoke(state["question"])
    sources = [
        {"source": d.metadata.get("source", "未知"), "page": d.metadata.get("page", "?")}
        for d in docs
    ]
    return {"docs": docs, "sources": sources}


def qa_node(state: AssistantState) -> AssistantState:
    return {"answer": workflows.qa(
        state["question"], state["docs"], state.get("api_key"), state.get("model"))}


def summary_node(state: AssistantState) -> AssistantState:
    return {"answer": workflows.summarize(
        state["question"], state["docs"], state.get("api_key"), state.get("model"))}


def quiz_node(state: AssistantState) -> AssistantState:
    return {"answer": workflows.generate_quiz(
        state["question"], state["docs"], state.get("api_key"), state.get("model"))}


def _select(state: AssistantState) -> Task:
    return state["task"]


def build_graph():
    g = StateGraph(AssistantState)
    g.add_node("route", route_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("qa", qa_node)
    g.add_node("summary", summary_node)
    g.add_node("quiz", quiz_node)

    g.add_edge(START, "route")
    g.add_edge("route", "retrieve")
    # 檢索後依任務分流
    g.add_conditional_edges(
        "retrieve", _select,
        {"qa": "qa", "summary": "summary", "quiz": "quiz"},
    )
    for node in ("qa", "summary", "quiz"):
        g.add_edge(node, END)
    return g.compile()


# 編譯一次重複使用
_APP = None


def make_quiz(topic: str, num: int = 5, qtype: str = "選擇題",
              api_key: str | None = None, model: str | None = None) -> dict:
    """專用出題：依主題檢索教材 → 生成測驗。供 UI 的「生成小測驗」按鈕使用。

    回傳 {quiz, sources}。topic 直接拿來做向量檢索，故主題越明確、題目越貼近教材。
    """
    docs = get_retriever().invoke(topic or "課程重點")
    sources = [
        {"source": d.metadata.get("source", "未知"), "page": d.metadata.get("page", "?")}
        for d in docs
    ]
    quiz = workflows.generate_quiz(
        topic, docs, api_key=api_key, model=model, num=num, qtype=qtype
    )
    return {"quiz": quiz, "sources": sources}


def run(question: str, api_key: str | None = None,
        model: str | None = None) -> AssistantState:
    """執行完整流程，回傳最終 state（含 answer 與 sources）。

    api_key／model：使用者在 UI 輸入/選的；未提供則退回 .env / 預設。
    """
    global _APP
    if _APP is None:
        _APP = build_graph()
    return _APP.invoke({"question": question, "api_key": api_key, "model": model})
