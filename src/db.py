"""SQLite 對話紀錄保存。"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from . import config


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT    NOT NULL,
                role       TEXT    NOT NULL,
                content    TEXT    NOT NULL,
                sources    TEXT,
                created_at TEXT    NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id)"
        )
        # ---- 適性化測驗：作答紀錄與成效分析 ----
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS quiz_attempt (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT    NOT NULL,
                topic      TEXT,
                qtype      TEXT,
                mode       TEXT,                 -- normal / adaptive
                total      INTEGER NOT NULL,
                score      INTEGER NOT NULL,
                created_at TEXT    NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS quiz_question (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                attempt_id   INTEGER NOT NULL,
                concept      TEXT,
                question     TEXT    NOT NULL,
                options_json TEXT,
                answer       TEXT,
                explanation  TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS quiz_response (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                attempt_id  INTEGER NOT NULL,
                question_id INTEGER NOT NULL,
                concept     TEXT,
                user_answer TEXT,
                is_correct  INTEGER NOT NULL,    -- 0 / 1
                created_at  TEXT    NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_attempt_session ON quiz_attempt(session_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_response_attempt ON quiz_response(attempt_id)"
        )


def save_message(session_id: str, role: str, content: str,
                 sources: list[dict] | None = None) -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, sources, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                session_id,
                role,
                content,
                json.dumps(sources, ensure_ascii=False) if sources else None,
                datetime.now(timezone.utc).isoformat(),
            ),
        )


def load_history(session_id: str) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT role, content, sources FROM messages "
            "WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
    history = []
    for r in rows:
        history.append({
            "role": r["role"],
            "content": r["content"],
            "sources": json.loads(r["sources"]) if r["sources"] else None,
        })
    return history


def clear_messages(session_id: str) -> None:
    """清空某 session 的對話訊息（保留教材與測驗紀錄）。"""
    with _conn() as conn:
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))


def list_sessions() -> list[str]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT session_id, MAX(created_at) AS last "
            "FROM messages GROUP BY session_id ORDER BY last DESC"
        ).fetchall()
    return [r["session_id"] for r in rows]


# ---------------------------------------------------------------------------
# 適性化測驗：寫入作答結果
# ---------------------------------------------------------------------------
def save_quiz_result(session_id: str, topic: str, qtype: str, mode: str,
                     questions: list[dict], responses: list[dict]) -> int:
    """寫入一次完整作答（attempt + 題目 + 作答），回傳 attempt_id。

    questions: [{concept, question, options(dict), answer, explanation}]
    responses: [{q_index, user_answer, is_correct}]  q_index 對應 questions 順序
    """
    ts = datetime.now(timezone.utc).isoformat()
    total = len(questions)
    score = sum(1 for r in responses if r.get("is_correct"))
    with _conn() as conn:
        cur = conn.execute(
            "INSERT INTO quiz_attempt (session_id, topic, qtype, mode, total, score, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (session_id, topic, qtype, mode, total, score, ts),
        )
        attempt_id = cur.lastrowid
        q_ids = []
        for q in questions:
            c = conn.execute(
                "INSERT INTO quiz_question "
                "(attempt_id, concept, question, options_json, answer, explanation) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    attempt_id, q.get("concept"), q.get("question"),
                    json.dumps(q.get("options"), ensure_ascii=False) if q.get("options") else None,
                    q.get("answer"), q.get("explanation"),
                ),
            )
            q_ids.append(c.lastrowid)
        for r in responses:
            qi = r["q_index"]
            conn.execute(
                "INSERT INTO quiz_response "
                "(attempt_id, question_id, concept, user_answer, is_correct, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    attempt_id, q_ids[qi], questions[qi].get("concept"),
                    r.get("user_answer"), 1 if r.get("is_correct") else 0, ts,
                ),
            )
    return attempt_id


# ---------------------------------------------------------------------------
# 適性化測驗：分析查詢
# ---------------------------------------------------------------------------
def attempts_summary(session_id: str) -> list[dict]:
    """每次 attempt 的 score/total，依時間排序（給趨勢圖）。"""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT id, topic, qtype, mode, total, score, created_at "
            "FROM quiz_attempt WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def concept_stats(session_id: str) -> list[dict]:
    """各觀念的累積作答數、答對數、正確率。"""
    with _conn() as conn:
        rows = conn.execute(
            """
            SELECT r.concept AS concept,
                   COUNT(*)            AS attempts,
                   SUM(r.is_correct)   AS correct
            FROM quiz_response r
            JOIN quiz_attempt a ON a.id = r.attempt_id
            WHERE a.session_id = ?
            GROUP BY r.concept
            ORDER BY (1.0 * SUM(r.is_correct) / COUNT(*)) ASC
            """,
            (session_id,),
        ).fetchall()
    out = []
    for r in rows:
        attempts = r["attempts"] or 0
        correct = r["correct"] or 0
        out.append({
            "concept": r["concept"] or "未標註",
            "attempts": attempts,
            "correct": correct,
            "accuracy": round(correct / attempts, 3) if attempts else 0.0,
        })
    return out


def weak_concepts(session_id: str, limit: int = 5) -> list[str]:
    """回傳最該加強的觀念：曾答錯、且正確率最低者優先。"""
    stats = concept_stats(session_id)
    weak = [s for s in stats if s["correct"] < s["attempts"]]  # 至少錯過一次
    weak.sort(key=lambda s: (s["accuracy"], -s["attempts"]))
    return [s["concept"] for s in weak[:limit]]


def concept_response_log(session_id: str) -> list[dict]:
    """逐筆作答紀錄（給 CSV 匯出）。"""
    with _conn() as conn:
        rows = conn.execute(
            """
            SELECT a.id AS attempt_id, a.created_at, a.mode, r.concept,
                   r.user_answer, r.is_correct
            FROM quiz_response r
            JOIN quiz_attempt a ON a.id = r.attempt_id
            WHERE a.session_id = ?
            ORDER BY a.id ASC, r.id ASC
            """,
            (session_id,),
        ).fetchall()
    return [dict(r) for r in rows]
