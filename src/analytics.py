"""學習成效量化：把作答紀錄轉成指標與可匯出報表。

核心指標（回答「系統是否讓使用者變強」）：
- 正確率趨勢：每次測驗 score/total。
- 概念掌握度：各觀念累積正確率。
- 進步幅度：最新測驗正確率 − 首次測驗正確率。
- 弱點克服率：首次答錯的觀念中、後續答對的比例。
"""
from __future__ import annotations

import csv
import io

from . import db


def overview(session_id: str) -> dict:
    """彙整關鍵指標。"""
    attempts = db.attempts_summary(session_id)
    metrics = {
        "num_attempts": len(attempts),
        "first_accuracy": None,
        "latest_accuracy": None,
        "improvement": None,
        "weakness_overcome_rate": None,
        "avg_accuracy": None,
    }
    if not attempts:
        return {"attempts": attempts, "metrics": metrics, "concepts": []}

    def acc(a):
        return a["score"] / a["total"] if a["total"] else 0.0

    accs = [acc(a) for a in attempts]
    metrics["first_accuracy"] = round(accs[0], 3)
    metrics["latest_accuracy"] = round(accs[-1], 3)
    metrics["improvement"] = round(accs[-1] - accs[0], 3)
    metrics["avg_accuracy"] = round(sum(accs) / len(accs), 3)
    metrics["weakness_overcome_rate"] = _weakness_overcome_rate(session_id)

    return {
        "attempts": attempts,
        "metrics": metrics,
        "concepts": db.concept_stats(session_id),
    }


def _weakness_overcome_rate(session_id: str) -> float | None:
    """首次測驗答錯的觀念中，之後（後續任何一次）有答對的比例。"""
    log = db.concept_response_log(session_id)
    if not log:
        return None
    # 第一個 attempt_id
    first_id = log[0]["attempt_id"]
    first_wrong = {r["concept"] for r in log
                   if r["attempt_id"] == first_id and not r["is_correct"]}
    if not first_wrong:
        return None
    later_correct = {r["concept"] for r in log
                     if r["attempt_id"] != first_id and r["is_correct"]}
    overcome = first_wrong & later_correct
    return round(len(overcome) / len(first_wrong), 3)


def trend_series(session_id: str) -> list[dict]:
    """給折線圖：[{第幾次, 正確率(%)}]。"""
    attempts = db.attempts_summary(session_id)
    return [
        {"測驗次序": i + 1,
         "正確率(%)": round(100 * a["score"] / a["total"], 1) if a["total"] else 0.0}
        for i, a in enumerate(attempts)
    ]


def responses_csv(session_id: str) -> str:
    """逐筆作答紀錄輸出成 CSV 字串。"""
    rows = db.concept_response_log(session_id)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["attempt_id", "時間", "模式", "觀念", "作答", "是否答對"])
    for r in rows:
        w.writerow([
            r["attempt_id"], r["created_at"], r["mode"], r["concept"],
            r["user_answer"], "是" if r["is_correct"] else "否",
        ])
    return buf.getvalue()


def build_report(session_id: str) -> str:
    """產生 Markdown 成效報告。"""
    data = overview(session_id)
    m = data["metrics"]
    if not m["num_attempts"]:
        return "# 學習成效報告\n\n尚無作答紀錄。"

    def pct(x):
        return f"{round(100 * x, 1)}%" if x is not None else "—"

    lines = [
        "# 學習成效報告",
        f"- session：`{session_id}`",
        f"- 測驗次數：{m['num_attempts']}",
        f"- 首次正確率：{pct(m['first_accuracy'])}",
        f"- 最新正確率：{pct(m['latest_accuracy'])}",
        f"- **進步幅度：{pct(m['improvement'])}**",
        f"- 平均正確率：{pct(m['avg_accuracy'])}",
        f"- **弱點克服率：{pct(m['weakness_overcome_rate'])}**",
        "",
        "## 各次測驗正確率",
        "| 次序 | 模式 | 主題 | 分數 | 正確率 |",
        "|---|---|---|---|---|",
    ]
    for i, a in enumerate(data["attempts"], 1):
        rate = f"{round(100 * a['score'] / a['total'], 1)}%" if a["total"] else "—"
        lines.append(f"| {i} | {a['mode']} | {a['topic'] or '—'} | "
                     f"{a['score']}/{a['total']} | {rate} |")

    lines += ["", "## 各觀念掌握度", "| 觀念 | 作答數 | 答對 | 正確率 |", "|---|---|---|---|"]
    for c in data["concepts"]:
        lines.append(f"| {c['concept']} | {c['attempts']} | {c['correct']} | "
                     f"{round(100 * c['accuracy'], 1)}% |")

    return "\n".join(lines)
