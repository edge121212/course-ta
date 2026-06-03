"""三個任務 workflow：課程問答、摘要生成、練習題生成。

每個函式接收使用者問題與檢索到的文件，回傳 LLM 生成的文字。
Prompt Engineering 強調「僅依教材內容作答、標註來源、避免幻覺」。
"""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from .llm import get_llm


def format_context(docs: list[Document]) -> str:
    """把檢索結果組成帶來源標記的 context。"""
    blocks = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "未知")
        page = d.metadata.get("page", "?")
        blocks.append(f"[來源{i}] {src} 第{page}頁\n{d.page_content.strip()}")
    return "\n\n".join(blocks) if blocks else "（無相關教材）"


_QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是課程智慧助教。請『僅根據以下教材內容』回答學生問題，並在答案中標註引用的來源編號。"
     "若教材內容不足以回答，請誠實說明『教材中查無相關內容』，不要自行編造。請用繁體中文回答。"),
    ("human", "教材內容：\n{context}\n\n學生問題：{question}"),
])

_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是課程智慧助教。請根據以下教材內容，整理出條理清晰的重點摘要（使用條列式），"
     "聚焦於學生指定的範圍。僅依教材內容摘要，勿加入教材以外的資訊。請用繁體中文。"),
    ("human", "教材內容：\n{context}\n\n摘要需求：{question}"),
])

# 各題型的出題格式說明
_QUIZ_FORMATS = {
    "選擇題": "每題包含題目與四個選項(A-D)，僅一個正確答案。",
    "是非題": "每題為一個敘述句，學生判斷正確(○)或錯誤(✗)。",
    "簡答題": "每題為一個開放式問題，需簡短文字作答。",
}


def _quiz_prompt(num: int, qtype: str) -> ChatPromptTemplate:
    fmt = _QUIZ_FORMATS.get(qtype, _QUIZ_FORMATS["選擇題"])
    system = (
        f"你是課程出題助教。請『僅根據以下教材內容』出 {num} 題{qtype}。{fmt} "
        "題目須源自教材，勿超出教材範圍。請先列出所有題目，"
        "最後再附上『參考答案』與每題的簡短解析。請用繁體中文。"
    )
    return ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "教材內容：\n{context}\n\n出題主題／需求：{question}"),
    ])


def _run(prompt: ChatPromptTemplate, question: str, docs: list[Document],
         api_key: str | None = None, model: str | None = None) -> str:
    chain = prompt | get_llm(api_key, model)
    resp = chain.invoke({"context": format_context(docs), "question": question})
    return resp.content


def qa(question: str, docs: list[Document],
       api_key: str | None = None, model: str | None = None) -> str:
    return _run(_QA_PROMPT, question, docs, api_key, model)


def summarize(question: str, docs: list[Document],
              api_key: str | None = None, model: str | None = None) -> str:
    return _run(_SUMMARY_PROMPT, question, docs, api_key, model)


def generate_quiz(question: str, docs: list[Document],
                  api_key: str | None = None, model: str | None = None,
                  num: int = 5, qtype: str = "選擇題") -> str:
    return _run(_quiz_prompt(num, qtype), question, docs, api_key, model)


# ---------------------------------------------------------------------------
# 答案自我查核（功能 2）：生成後回教材逐句核對，刪除/修正無依據內容
# ---------------------------------------------------------------------------
class _Verdict(BaseModel):
    grounded: bool = Field(description="修正後答案是否每句都有教材依據")
    revised_answer: str = Field(description="刪除/修正無依據內容後的答案")


_VERIFY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是嚴格的查核員。檢查『待查核答案』是否每一句都能在『教材內容』中找到依據。"
     "刪除或修正教材裡找不到依據、或與教材矛盾的句子；保留有依據的部分。"
     "若修正後仍有實質內容且全部有依據，grounded 設為 true；"
     "若答案大多無依據，保留有依據的部分並在結尾註明『教材未涵蓋其餘部分』，grounded 設為 false。"
     "請用繁體中文，保持精簡。"),
    ("human", "教材內容：\n{context}\n\n待查核答案：\n{answer}\n\n原問題：{question}"),
])


def verify(question: str, answer: str, docs: list[Document],
           api_key: str | None = None, model: str | None = None) -> dict:
    """回教材核對答案，回傳 {grounded: bool, answer: 修正後答案}。"""
    llm = get_llm(api_key, model).with_structured_output(_Verdict)
    msgs = _VERIFY_PROMPT.format_messages(
        context=format_context(docs), answer=answer, question=question)
    v: _Verdict = llm.invoke(msgs)
    return {"grounded": v.grounded, "answer": v.revised_answer}
