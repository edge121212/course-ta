"""介面樣式（編輯學院風 Editorial Academic）。

米白底 / 墨黑字 / 墨綠點綴，Noto Serif TC 標題、Noto Sans TC 內文。
集中管理 CSS 與版面元件，讓 app.py 專注於邏輯。
"""
from __future__ import annotations

import streamlit as st

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@300;400;500;700&family=Noto+Serif+TC:wght@500;600;700&display=swap');

:root{
  --bg:#FAF8F3; --surface:#FFFFFF; --surface-2:#F2EEE6;
  --ink:#1A1A1A; --muted:#6F6B61; --line:#E5DFD0;
  --accent:#2F4538; --accent-soft:#43604F;
}

/* ---- 整體 ---- */
.stApp{ background:var(--bg); color:var(--ink); }
html, body, .stApp, input, textarea, button, select,
p, li, a, label, h1,h2,h3,h4,h5,h6,
.stMarkdown, [data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"]{
  font-family:'Noto Sans TC', system-ui, -apple-system, sans-serif;
}
/* 還原 Material 圖示字型，避免 expander 箭頭、上傳圖示變成文字疊在一起 */
[data-testid="stIconMaterial"],
.material-icons, .material-icons-outlined,
.material-symbols-outlined, .material-symbols-rounded,
[class^="material-"], [class*=" material-"]{
  font-family:'Material Symbols Rounded','Material Symbols Outlined','Material Icons Outlined','Material Icons' !important;
}

/* 隱藏 Streamlit 預設裝飾，去除工具味 */
#MainMenu, footer, [data-testid="stDecoration"], [data-testid="stStatusWidget"]{ display:none !important; }
[data-testid="stHeader"]{ background:transparent; }

/* 內容寬度與留白 */
[data-testid="stMainBlockContainer"], .block-container{
  max-width:960px; padding-top:2.2rem; padding-bottom:4rem;
}

/* ---- 標題（襯線）---- */
h1,h2,h3,h4{
  font-family:'Noto Serif TC', serif !important;
  color:#22302A; font-weight:600; letter-spacing:.01em;
}

/* ---- Hero ---- */
.hero{ margin:.2rem 0 1.4rem; }
.hero-kicker{ font-size:.72rem; letter-spacing:.30em; color:var(--accent-soft); font-weight:600; }
.hero-title{
  font-family:'Noto Serif TC', serif; font-size:2.5rem; font-weight:700;
  color:#1F2E26; margin:.4rem 0 .55rem; line-height:1.12;
}
.hero-sub{ color:var(--muted); font-size:1rem; max-width:640px; line-height:1.75; margin:0; }
.hero-rule{ height:1px; background:linear-gradient(90deg,var(--accent) 0,var(--line) 55%,transparent); margin-top:1.4rem; }

/* ---- 區段標籤 ---- */
.sec{
  font-family:'Noto Serif TC', serif; font-size:.98rem; font-weight:600; color:#2B3A32;
  letter-spacing:.04em; margin:.4rem 0 .7rem; padding-bottom:.4rem; border-bottom:1px solid var(--line);
}
.doc-row{ font-size:.9rem; color:var(--ink); line-height:1.4; }
.doc-row small{ color:var(--muted); }

/* ---- 側欄 ---- */
[data-testid="stSidebar"]{ background:var(--surface-2); border-right:1px solid var(--line); }
[data-testid="stSidebar"] .block-container{ padding-top:1.3rem; }

/* ---- 按鈕 ---- */
.stButton>button, .stDownloadButton>button, .stFormSubmitButton>button{
  border-radius:8px; border:1px solid var(--line); background:var(--surface);
  color:var(--ink); font-weight:500; padding:.45rem 1.05rem; transition:all .15s ease;
  box-shadow:0 1px 2px rgba(31,46,38,.04);
}
.stButton>button:hover, .stDownloadButton>button:hover, .stFormSubmitButton>button:hover{
  border-color:var(--accent); color:var(--accent);
}
button[kind="primary"], button[kind="primaryFormSubmit"],
button[data-testid="baseButton-primary"], button[data-testid="baseButton-primaryFormSubmit"]{
  background:var(--accent) !important; color:#F6F4EE !important; border:1px solid var(--accent) !important;
}
button[kind="primary"]:hover, button[kind="primaryFormSubmit"]:hover{
  background:#26382E !important; color:#FFFFFF !important;
}

/* ---- 輸入框 / 下拉 ---- */
.stTextInput input, textarea, [data-baseweb="input"], [data-baseweb="select"]>div{
  background:var(--surface) !important; border-radius:8px !important;
}
.stTextInput input:focus{ border-color:var(--accent) !important; }

/* ---- Expander 卡片 ---- */
[data-testid="stExpander"]{
  border:1px solid var(--line); border-radius:12px; background:var(--surface);
  box-shadow:0 1px 3px rgba(31,46,38,.04); overflow:hidden; margin-bottom:1rem;
}
[data-testid="stExpander"] summary{
  font-family:'Noto Serif TC', serif; font-weight:600; font-size:1.04rem; color:#22302A; padding:.5rem .3rem;
}
[data-testid="stExpander"] summary:hover{ color:var(--accent); }

/* ---- Metric 卡片 ---- */
[data-testid="stMetric"]{
  background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:1rem 1.1rem;
}
[data-testid="stMetricValue"]{ font-family:'Noto Serif TC', serif; font-weight:700; color:#1F2E26; }
[data-testid="stMetricLabel"]{ color:var(--muted); }

/* ---- Chat ---- */
[data-testid="stChatMessage"]{
  background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:.3rem 1rem;
}

/* ---- Alerts ---- */
[data-testid="stAlert"]{ border-radius:10px; border:1px solid var(--line); }

/* ---- 其他 ---- */
a{ color:var(--accent) !important; }
hr{ border-color:var(--line); }
[data-testid="stDataFrame"]{ border:1px solid var(--line); border-radius:10px; }

/* 作答對錯標記 */
.mk{ font-weight:700; }
.mk-ok{ color:var(--accent); }
.mk-no{ color:#9A3B36; }
</style>
"""


def inject() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str, kicker: str = "COURSE ASSISTANT") -> None:
    st.markdown(
        f'<div class="hero">'
        f'<div class="hero-kicker">{kicker}</div>'
        f'<div class="hero-title">{title}</div>'
        f'<p class="hero-sub">{subtitle}</p>'
        f'<div class="hero-rule"></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def section(label: str) -> None:
    st.markdown(f'<div class="sec">{label}</div>', unsafe_allow_html=True)
