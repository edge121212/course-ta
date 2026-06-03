"""智慧課程助教系統 — Streamlit 介面。

功能：上傳課程 PDF → 自動切割/嵌入 → 自然語言問答(含引用來源) / 摘要 / 出題。
對話紀錄保存於 SQLite。介面樣式集中於 src/ui.py（編輯學院風）。
"""
from __future__ import annotations

import logging
import uuid

import pandas as pd
import streamlit as st

from src import agent, analytics, config, db, quiz, ui
from src.graph import classify, run
from src.ingest import ingest_pdf
from src.llm import MissingAPIKeyError, list_models
from src.vectorstore import delete_source, document_count, list_sources

st.set_page_config(page_title="智慧課程助教", page_icon="📖", layout="wide")
ui.inject()

# 把使用者在網頁遇到的錯誤完整寫進 data/app.log，方便事後診斷
logging.basicConfig(
    filename=str(config.DATA_DIR / "app.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True,
)
log = logging.getLogger("course_ta")

db.init_db()

TASK_LABEL = {"qa": "課程問答", "summary": "摘要生成", "quiz": "練習題生成"}

# ---- session 狀態 ----
# 每位訪客自動分配獨立 session id，存在網址 ?sid= 以便重整後仍保留，
# 不同訪客資料互相隔離（多人測試成效數據不會混在一起）。
if "session_id" not in st.session_state:
    sid = st.query_params.get("sid")
    if not sid:
        sid = uuid.uuid4().hex[:12]
        st.query_params["sid"] = sid
    st.session_state.session_id = sid
if "messages" not in st.session_state:
    st.session_state.messages = db.load_history(st.session_state.session_id)
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

# ---- 側邊欄：金鑰、模型、教材、對話 ----
with st.sidebar:
    ui.section("存取金鑰")
    st.session_state.api_key = st.text_input(
        "Gemini API Key",
        value=st.session_state.api_key,
        type="password",
        help="僅保存在此瀏覽器分頁，關閉或重新整理後需重新輸入。取得：https://aistudio.google.com/app/apikey",
        placeholder="AIza…",
    )
    if st.session_state.api_key.strip():
        st.success("金鑰已就緒")
    else:
        st.info("尚未設定金鑰：可上傳與檢索，問答與測驗需要金鑰")

    # 模型選擇：預設清單 + 可用 key 查詢實際可用模型
    if "model_options" not in st.session_state:
        st.session_state.model_options = list(config.GEMINI_MODEL_CHOICES)
    if st.button("查詢可用模型") and st.session_state.api_key.strip():
        try:
            models = list_models(st.session_state.api_key)
            if models:
                st.session_state.model_options = models
                st.success(f"找到 {len(models)} 個可用模型")
            else:
                st.warning("查無支援 generateContent 的模型")
        except Exception as e:  # noqa: BLE001
            st.error(f"查詢失敗：{e}")
    st.session_state.model = st.selectbox(
        "模型", st.session_state.model_options, index=0,
        help="若預設模型回報 404，按上方查詢實際可用的模型再選。",
    )

    st.markdown("")
    ui.section("助教模式")
    st.session_state.agent_mode = st.toggle(
        "Agent 模式", value=st.session_state.get("agent_mode", True),
        help="開啟後，助教會先『規劃步驟』再執行，能一次處理比較／又問又出題等複合需求。",
    )
    st.session_state.verify_on = st.toggle(
        "答案自我查核", value=st.session_state.get("verify_on", True),
        help="問答生成後，回教材逐句核對，刪除無依據內容並標示已核對。",
    )

    st.markdown("")
    ui.section("課程教材")

    # ---- 新增文件 ----
    uploaded = st.file_uploader("上傳課程 PDF", type=["pdf"], accept_multiple_files=True)
    if uploaded and st.button("匯入教材", type="primary"):
        for f in uploaded:
            dest = config.UPLOAD_DIR / f.name
            dest.write_bytes(f.getbuffer())
            try:
                with st.spinner(f"切割並嵌入 {f.name} …"):
                    n = ingest_pdf(dest)
                st.success(f"{f.name}：新增 {n} 個片段")
                log.info("匯入成功 %s：%d 片段", f.name, n)
            except Exception as e:  # noqa: BLE001
                st.error(f"{f.name} 匯入失敗：{e}")
                log.exception("匯入失敗 %s", f.name)
        st.rerun()

    # ---- 已匯入文件清單（可刪除）----
    sources = list_sources()
    if not sources:
        st.caption("尚無文件，請於上方上傳並匯入。")
    else:
        for name, cnt in sources.items():
            col_a, col_b = st.columns([0.74, 0.26])
            col_a.markdown(
                f'<div class="doc-row">{name}<br><small>{cnt} 個片段</small></div>',
                unsafe_allow_html=True,
            )
            if col_b.button("移除", key=f"del_{name}", help=f"從知識庫移除 {name}"):
                removed = delete_source(name)
                fp = config.UPLOAD_DIR / name
                if fp.exists():
                    fp.unlink()
                st.success(f"已移除 {name}（{removed} 片段）")
                log.info("刪除文件 %s：%d 片段", name, removed)
                st.rerun()
        st.caption(f"知識庫共 {document_count()} 個片段")

    st.markdown("")
    ui.section("對話")
    if st.button("開新對話"):
        sid = uuid.uuid4().hex[:12]
        st.session_state.session_id = sid
        st.query_params["sid"] = sid
        st.session_state.messages = []
        st.session_state.iquiz = st.session_state.iquiz_result = None
        st.session_state.iquiz_submitted = False
        st.rerun()
    st.caption(f"專屬 session：{st.session_state.session_id}　·　資料與他人隔離")

# ---- 主畫面 ----
ui.hero(
    "智慧課程助教",
    "上傳課程教材，即可進行有憑有據的課程問答、重點摘要，"
    "以及依個人弱點自動調整的學習測驗與成效分析。",
    kicker="COURSE ASSISTANT · RAG",
)

# ---- 互動測驗（計分・依弱點適性出題）----
ss = st.session_state
ss.setdefault("iquiz", None)             # {questions, sources, meta}
ss.setdefault("iquiz_submitted", False)
ss.setdefault("iquiz_result", None)      # {responses, attempt_id, score, total}


def _generate_quiz(topic: str, num: int, adaptive: bool):
    """產生測驗並寫入 session_state；回傳 True 表示成功。"""
    try:
        if adaptive:
            res = quiz.generate_adaptive(ss.session_id, topic, num=num,
                                         api_key=ss.api_key, model=ss.get("model"))
        else:
            res = quiz.generate(topic or "課程重點", num=num,
                                api_key=ss.api_key, model=ss.get("model"))
        ss.iquiz = {"questions": res["questions"], "sources": res["sources"],
                    "meta": {"topic": topic, "mode": "adaptive" if adaptive else "normal",
                             "focus": res.get("focus", [])}}
        ss.iquiz_submitted = False
        ss.iquiz_result = None
        log.info("互動出題 mode=%s topic=%r n=%s focus=%s",
                 ss.iquiz["meta"]["mode"], topic, num, res.get("focus"))
        return True
    except MissingAPIKeyError as e:
        st.warning(str(e))
    except Exception as e:  # noqa: BLE001
        st.error(f"出題失敗：{e}")
        log.exception("互動出題失敗 topic=%r", topic)
    return False


with st.expander("互動測驗　·　計分與依弱點適性出題", expanded=True):
    c1, c2, c3 = st.columns([0.5, 0.25, 0.25])
    it_topic = c1.text_input("主題", placeholder="例如：記憶體階層、CPU 結構", key="iquiz_topic")
    it_num = c2.selectbox("題數", [3, 5, 10], index=1, key="iquiz_num")
    it_mode = c3.selectbox("出題模式", ["一般（依主題）", "適性（依我的弱點）"], key="iquiz_mode")

    if st.button("出題", type="primary", key="iquiz_gen"):
        if document_count() == 0:
            st.warning("知識庫是空的，請先在左側上傳並匯入教材。")
        elif not ss.api_key.strip():
            st.warning("請先在左側輸入 Gemini API Key。")
        else:
            with st.spinner("出題中…"):
                if _generate_quiz(it_topic, it_num, it_mode.startswith("適性")):
                    st.rerun()

    iq = ss.iquiz
    # 作答中
    if iq and not ss.iquiz_submitted:
        if iq["meta"].get("focus"):
            st.info("本次聚焦弱點觀念：" + "、".join(iq["meta"]["focus"]))
        with st.form("iquiz_form"):
            answers = []
            for i, q in enumerate(iq["questions"]):
                opts = q.get("options") or {}
                letters = sorted(opts.keys())
                st.markdown(f"**第 {i+1} 題**　{q.get('question', '')}"
                            f"　<small style='color:#6F6B61'>{q.get('concept', '')}</small>",
                            unsafe_allow_html=True)
                ans = st.radio("選擇答案", letters, index=None, key=f"iq_{i}",
                               format_func=lambda L, o=opts: f"{L}.　{o.get(L, '')}",
                               label_visibility="collapsed")
                answers.append(ans)
            if st.form_submit_button("送出作答", type="primary"):
                responses = quiz.grade(iq["questions"], answers)
                attempt_id = db.save_quiz_result(
                    ss.session_id, iq["meta"]["topic"], "選擇題",
                    iq["meta"]["mode"], iq["questions"], responses)
                score = sum(1 for r in responses if r["is_correct"])
                ss.iquiz_result = {"responses": responses, "attempt_id": attempt_id,
                                   "score": score, "total": len(iq["questions"])}
                ss.iquiz_submitted = True
                log.info("作答完成 attempt=%s score=%s/%s", attempt_id, score, len(iq["questions"]))
                st.rerun()

    # 作答結果
    if iq and ss.iquiz_submitted and ss.iquiz_result:
        r = ss.iquiz_result
        pct = round(100 * r["score"] / r["total"]) if r["total"] else 0
        st.success(f"得分　{r['score']} / {r['total']}　（{pct}%）")
        for i, q in enumerate(iq["questions"]):
            resp = r["responses"][i]
            if resp["is_correct"]:
                mark = '<span class="mk mk-ok">答對</span>'
            else:
                mark = '<span class="mk mk-no">答錯</span>'
            st.markdown(f'{mark}　**第 {i+1} 題**　{q.get("question", "")}', unsafe_allow_html=True)
            st.caption(f"你的答案：{resp['user_answer'] or '未作答'}　·　正解：{q.get('answer')}"
                       f"　·　{q.get('explanation', '')}")
        cc1, cc2 = st.columns(2)
        if cc1.button("依弱點再練一輪", type="primary", key="iquiz_again"):
            with st.spinner("依弱點出題中…"):
                if _generate_quiz("", r["total"], adaptive=True):
                    st.rerun()
        if cc2.button("重新出題", key="iquiz_reset"):
            ss.iquiz = ss.iquiz_result = None
            ss.iquiz_submitted = False
            st.rerun()

# ---- 學習成效分析儀表板 ----
with st.expander("學習成效分析　·　量化學習進步", expanded=False):
    ov = analytics.overview(ss.session_id)
    m = ov["metrics"]
    if not m["num_attempts"]:
        st.info("尚無作答紀錄，完成上方一次互動測驗後即可看到成效數據。")
    else:
        def _p(x):
            return f"{round(100 * x, 1)}%" if x is not None else "—"

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("測驗次數", m["num_attempts"])
        k2.metric("最新正確率", _p(m["latest_accuracy"]),
                  delta=_p(m["improvement"]) if m["improvement"] is not None else None)
        k3.metric("進步幅度", _p(m["improvement"]))
        k4.metric("弱點克服率", _p(m["weakness_overcome_rate"]),
                  help="首次答錯的觀念中，之後答對的比例")

        trend = analytics.trend_series(ss.session_id)
        if len(trend) >= 2:
            st.line_chart(pd.DataFrame(trend).set_index("測驗次序"))

        if ov["concepts"]:
            st.markdown("**各觀念掌握度**")
            cdf = pd.DataFrame([
                {"觀念": c["concept"], "作答數": c["attempts"], "答對": c["correct"],
                 "正確率(%)": round(100 * c["accuracy"], 1)}
                for c in ov["concepts"]
            ])
            st.dataframe(cdf, hide_index=True, use_container_width=True)

        weak = db.weak_concepts(ss.session_id)
        if weak:
            st.caption("待加強觀念：" + "、".join(weak))

        e1, e2 = st.columns(2)
        e1.download_button("下載作答紀錄（CSV）", analytics.responses_csv(ss.session_id),
                           file_name="quiz_responses.csv", mime="text/csv")
        e2.download_button("下載成效報告（Markdown）", analytics.build_report(ss.session_id),
                           file_name="learning_report.md", mime="text/markdown")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            srcs = "、".join(f"{s['source']} 第{s['page']}頁" for s in m["sources"])
            st.caption(f"引用來源：{srcs}")

prompt = st.chat_input("輸入問題，例如：什麼是 RAG？／幫我整理重點／出 5 題測驗")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": None})
    db.save_message(st.session_state.session_id, "user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    if document_count() == 0:
        warn = "知識庫目前是空的，請先在左側上傳並匯入課程 PDF。"
        with st.chat_message("assistant"):
            st.warning(warn)
        st.session_state.messages.append({"role": "assistant", "content": warn, "sources": None})
        db.save_message(st.session_state.session_id, "assistant", warn)
    else:
        with st.chat_message("assistant"):
            use_agent = st.session_state.get("agent_mode", True)
            verify_on = st.session_state.get("verify_on", True)
            steps = grounded = None
            is_clarify = False
            log.info("提問 agent=%s verify=%s model=%s q=%r",
                     use_agent, verify_on, st.session_state.get("model"), prompt)
            try:
                if use_agent:
                    history = st.session_state.messages[:-1]
                    with st.status("助教思考中…", expanded=True) as status:
                        res = agent.run_agent(
                            prompt, st.session_state.session_id, history=history,
                            api_key=st.session_state.api_key,
                            model=st.session_state.get("model"), verify_qa=verify_on,
                            progress=lambda msg: status.write("· " + msg))
                        status.update(label="完成", state="complete", expanded=False)
                    answer = res["answer"]
                    sources = res["sources"]
                    steps = res["steps"] if res["multi"] else None
                    grounded = res["grounded"]
                    is_clarify = res.get("is_clarify", False)
                else:
                    with st.spinner(f"{TASK_LABEL[classify(prompt)]}　生成中…"):
                        task = classify(prompt)
                        state = run(prompt, api_key=st.session_state.api_key,
                                    model=st.session_state.get("model"))
                        answer = state["answer"]
                        sources = state.get("sources") if task == "qa" else None
            except MissingAPIKeyError as e:
                answer, sources = str(e), None
                log.warning("缺少 API key：%s", e)
            except Exception as e:  # noqa: BLE001
                answer, sources = f"生成失敗：{e}", None
                log.exception("生成失敗 model=%s q=%r",
                              st.session_state.get("model"), prompt)

            if steps:
                plan_txt = "　".join(f"{i+1}. {s['note']}" for i, s in enumerate(steps))
                st.caption(f"規劃 {len(steps)} 個步驟　·　{plan_txt}")

            st.markdown(answer)

            if is_clarify:
                st.caption("請補充說明後，我再為你回答")
            elif grounded is True:
                st.caption("✓ 已逐句核對教材")
            elif grounded is False:
                st.caption("部分內容教材未涵蓋，已修正")
            if sources and not is_clarify:
                srcs = "、".join(f"{s['source']} 第{s['page']}頁" for s in sources)
                st.caption(f"引用來源：{srcs}")

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
        db.save_message(st.session_state.session_id, "assistant", answer, sources)
