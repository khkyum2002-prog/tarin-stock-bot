import io, os, re, time, warnings, logging
import numpy as np
import pandas as pd
import yfinance as yf
import requests
import streamlit as st
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
try:
    from streamlit_autorefresh import st_autorefresh
    _HAS_AUTOREFRESH = True
except ImportError:
    _HAS_AUTOREFRESH = False

try:
    from streamlit_extras.colored_header import colored_header as _clr_hdr
    _HAS_CLR_HDR = True
except ImportError:
    _HAS_CLR_HDR = False

try:
    import streamlit_shadcn_ui as _ui
    _HAS_SHADCN = True
except ImportError:
    _HAS_SHADCN = False

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

st.set_page_config(page_title="퇴근길 주식", page_icon="📈", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
/* ── 기본 폰트 / 배경 ── */
*, *::before, *::after { font-family: 'Inter', sans-serif !important; }
.stApp { background: #0d1117; }
#MainMenu, footer { visibility: hidden; }
.block-container { max-width: 1100px; padding: 1.5rem 2rem 3rem; }

/* ── 메트릭 카드 ── */
div[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 2px 0;
}
[data-testid="stMetricValue"] {
    font-size: 1.35rem !important;
    font-weight: 700 !important;
    color: #e6edf3 !important;
    letter-spacing: -0.02em;
}
[data-testid="stMetricLabel"] {
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    color: #7d8590 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
[data-testid="stMetricDelta"] { font-size: 0.82rem !important; }

/* ── 탭 ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 1px solid #21262d;
    background: transparent;
    flex-wrap: wrap;
}
.stTabs [data-baseweb="tab"] {
    font-weight: 600;
    font-size: 0.88rem;
    color: #7d8590;
    padding: 10px 20px;
    background: transparent !important;
    border-bottom: 2px solid transparent;
    white-space: nowrap;
}
.stTabs [aria-selected="true"] {
    color: #e6edf3 !important;
    border-bottom: 2px solid #2f81f7 !important;
}
/* ── 모바일 탭 ── */
@media (max-width: 640px) {
    .stTabs [data-baseweb="tab"] {
        font-size: 0.72rem;
        padding: 8px 10px;
    }
    .block-container { padding: 1rem 0.75rem 2rem; }
    [data-testid="stMetricValue"] { font-size: 1.05rem !important; }
}

/* ── 섹션 헤더 (zone-header) ── */
.zone-header {
    font-size: 0.9rem;
    font-weight: 700;
    letter-spacing: 0.01em;
    color: #c9d1d9;
    margin: 1.8rem 0 0.6rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #21262d;
    display: block;
}

/* ── 신호 카드 ── */
.sig-green {
    background: rgba(63,185,80,0.08);
    border-left: 3px solid #3fb950;
    border-radius: 6px; padding: 9px 14px; margin: 4px 0;
    font-size: 0.875rem; color: #e6edf3;
}
.sig-red {
    background: rgba(248,81,73,0.08);
    border-left: 3px solid #f85149;
    border-radius: 6px; padding: 9px 14px; margin: 4px 0;
    font-size: 0.875rem; color: #e6edf3;
}
.sig-yellow {
    background: rgba(210,153,34,0.1);
    border-left: 3px solid #d29922;
    border-radius: 6px; padding: 9px 14px; margin: 4px 0;
    font-size: 0.875rem; color: #e6edf3;
}

/* ── 버튼 ── */
div[data-testid="stButton"] button[kind="primary"] {
    background: #238636;
    border: 1px solid rgba(240,246,252,0.1);
    border-radius: 6px;
    font-weight: 600;
    font-size: 0.875rem;
    transition: background 0.15s;
}
div[data-testid="stButton"] button[kind="primary"]:hover { background: #2ea043; }
div[data-testid="stButton"] button:not([kind="primary"]) {
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 6px;
    font-size: 0.875rem;
}

/* ── 구분선 ── */
hr { border: none; border-top: 1px solid #21262d; margin: 1.2rem 0; }

/* ── 파일 업로더 ── */
div[data-testid="stFileUploader"] {
    background: #161b22;
    border: 1px dashed #30363d;
    border-radius: 8px;
}
div[data-testid="stFileUploader"] label { font-size: 0.82rem !important; color: #7d8590; }
/* 모바일: 업로더 내부 드래그영역 세로 정렬 */
@media (max-width: 640px) {
    div[data-testid="stFileUploader"] section {
        flex-direction: column !important;
        align-items: flex-start !important;
        gap: 8px !important;
    }
    div[data-testid="stFileUploader"] section > div {
        width: 100% !important;
    }
    div[data-testid="stFileUploaderDropzone"] {
        padding: 12px !important;
    }
    div[data-testid="stFileUploaderDropzoneInstructions"] {
        flex-direction: column !important;
        align-items: center !important;
        text-align: center;
    }
}

/* ── 익스팬더 ── */
div[data-testid="stExpander"] {
    border: 1px solid #21262d !important;
    border-radius: 8px;
    background: #161b22;
}
div[data-testid="stExpander"] summary {
    font-weight: 500;
    font-size: 0.875rem;
    color: #c9d1d9;
}

/* ── 캡션 ── */
.stCaption p { color: #7d8590 !important; font-size: 0.78rem !important; }

/* ── 토글 ── */
div[data-testid="stToggle"] label { font-size: 0.875rem; }

/* ── 스크롤바 ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #484f58; }

/* ── 등급 뱃지 ── */
.badge-bz   { background:#2d1f3d; color:#c792ea; border:1px solid #9c59d1; border-radius:12px; padding:3px 12px; font-size:0.80rem; font-weight:700; margin:2px; display:inline-block; }
.badge-star { background:#1f3a2a; color:#3fb950; border:1px solid #3fb950; border-radius:12px; padding:3px 12px; font-size:0.80rem; font-weight:700; margin:2px; display:inline-block; }
.badge-good { background:#1a2c45; color:#79c0ff; border:1px solid #388bfd; border-radius:12px; padding:3px 12px; font-size:0.80rem; font-weight:600; margin:2px; display:inline-block; }
.badge-watch { background:#2d2a1e; color:#e3b341; border:1px solid #d29922; border-radius:12px; padding:3px 12px; font-size:0.80rem; font-weight:500; margin:2px; display:inline-block; }

/* ── 섹터 태그 ── */
.sector-tag { display:inline-block; background:#21262d; border:1px solid #30363d; border-radius:6px; padding:3px 10px; margin:2px; font-size:0.80rem; color:#8b949e; }
.sector-tag.on { background:#1f3a2a; border-color:#3fb950; color:#3fb950; font-weight:600; }

/* ── 대시보드 ── */
.dash-status-green  { background:rgba(63,185,80,0.10);  border:1.5px solid #3fb950; border-radius:12px; padding:20px 28px; margin:4px 0 16px; text-align:center; }
.dash-status-red    { background:rgba(248,81,73,0.10);   border:1.5px solid #f85149; border-radius:12px; padding:20px 28px; margin:4px 0 16px; text-align:center; }
.dash-status-yellow { background:rgba(210,153,34,0.10);  border:1.5px solid #d29922; border-radius:12px; padding:20px 28px; margin:4px 0 16px; text-align:center; }
.dash-metric { background:#161b22; border:1px solid #21262d; border-radius:8px; padding:12px 8px; text-align:center; min-height:78px; }

/* ── 개선된 zone-header ── */
.zone-header {
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    color: #8b949e !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    margin: 1.6rem 0 0.8rem !important;
    padding: 0 0 0.5rem !important;
    border-bottom: 1px solid #21262d !important;
    display: block !important;
}

/* ── sig 카드 개선 ── */
.sig-green, .sig-red, .sig-yellow {
    border-left-width: 4px !important;
    padding: 10px 16px !important;
    line-height: 1.6 !important;
}

/* ── 메트릭 카드 hover ── */
div[data-testid="metric-container"]:hover {
    border-color: #388bfd;
    transition: border-color 0.2s;
}

/* ── info 박스 compact ── */
div[data-testid="stAlert"] {
    padding: 8px 16px !important;
    font-size: 0.83rem !important;
}

/* ── spinner 텍스트 ── */
div[data-testid="stSpinner"] p { font-size: 0.8rem !important; color: #7d8590 !important; }
</style>""", unsafe_allow_html=True)

_days = ["월","화","수","목","금","토","일"]
import datetime as _dt_hdr
_kst_hdr = _dt_hdr.datetime.utcnow() + _dt_hdr.timedelta(hours=9)
_status_dot = "🟢" if (9 <= _kst_hdr.hour < 15 or (_kst_hdr.hour == 15 and _kst_hdr.minute <= 30)) and _kst_hdr.weekday() < 5 else "⚫"
st.markdown(
    f'<h2 style="margin:0;font-size:1.4rem;font-weight:700;color:#e6edf3;letter-spacing:-0.03em;">퇴근길 주식 {_status_dot}</h2>'
    f'<p style="margin:2px 0 1rem;font-size:0.78rem;color:#7d8590;">'
    f'{_kst_hdr.strftime("%Y-%m-%d")} ({_days[_kst_hdr.weekday()]}요일) &nbsp;·&nbsp; {_kst_hdr.strftime("%H:%M")} KST</p>',
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────────────────────────────────────
# 사이드바 — 로컬 파일 경로 설정
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULT_XL_DIR  = r"C:\Users\khkyu\OneDrive\바탕 화면\태린이주식\20260515"
_DEFAULT_XL_DIR2 = r"C:\Users\khkyu\OneDrive\바탕 화면\태린이주식\20260515 (2)"

with st.sidebar:
    st.header("⚙️ 파일 설정")
    _xl_dir = st.text_input("데이터 폴더", value=_DEFAULT_XL_DIR, key="xl_dir",
                             help="Excel 파일들이 들어있는 최상위 폴더 경로")
    _xl_dir2 = st.text_input("수출데이터 폴더", value=_DEFAULT_XL_DIR2, key="xl_dir2",
                              help="수출데이터 Excel 파일들이 들어있는 폴더")

    _LOCAL_FILES = {
        "fg":        os.path.join(_xl_dir, "한국피어앤그리드오실레이터", "피어앤그리드.xlsx"),
        "rs_stock":  os.path.join(_xl_dir, "한국 개별종목 상대강도", "종목상대강도데이터.xlsx"),
        "rs_etf":    os.path.join(_xl_dir, "한국 etf 활용한 상대강도 추출", "etf상대강도데이터.xlsx"),
        "trend_wk":  os.path.join(_xl_dir, "차트추세판별기", "추세판별기(주간).xlsx"),
        "trend_sup": os.path.join(_xl_dir, "차트추세판별기", "추세판별기(수급까지체크).xlsx"),
        "trading":   os.path.join(_xl_dir, "거래대금 강도를 통해 매수 타이밍 잡기", "국장 거래대금 강도 확인용.xlsx"),
        "consensus": os.path.join(_xl_dir, "컨센 가속 및 수급 가속반영 등", "데이터 정리.xlsx"),
        "exp_may":   os.path.join(_xl_dir2, "주요품목별수출정리 5월 잠정(태린이아빠)(매일).xlsx"),
        "exp_apr":   os.path.join(_xl_dir2, "주요품목별수출정리 4월 확정(태린이아빠)(매일).xlsx"),
    }

    def _fstatus(k):
        p = _LOCAL_FILES[k]
        return ("✅", p) if os.path.exists(p) else ("❌", p)

    def _local_bytes(k):
        p = _LOCAL_FILES[k]
        if os.path.exists(p):
            with open(p, "rb") as f:
                return f.read()
        return None

    st.markdown("**로컬 파일 감지**")
    for _k, _label in [
        ("fg",        "피어앤그리드"),
        ("rs_stock",  "종목 RS"),
        ("rs_etf",    "ETF RS"),
        ("trend_wk",  "추세판별기(주간)"),
        ("trend_sup", "추세판별기(수급)"),
        ("trading",   "거래대금 강도"),
        ("consensus", "컨센데이터"),
        ("exp_may",   "수출(5월잠정)"),
        ("exp_apr",   "수출(4월확정)"),
    ]:
        _ic, _p = _fstatus(_k)
        st.caption(f"{_ic} {_label}")

    st.divider()
    if st.button("🔄 로컬 파일 전체 로드", use_container_width=True, key="load_all_local",
                 help="감지된 모든 Excel 파일을 세션에 로드합니다"):
        _loaded = []
        _map = {
            "fg":        "c_fg_bytes",
            "rs_stock":  "c_rs_bytes",
            "rs_etf":    "c_etf_xl_bytes",
            "trend_wk":  "c_weekly_bytes",
            "trend_sup": "c_supply_bytes",
            "trading":   "c_trading_bytes",
            "consensus": "c_consensus_bytes",
            "exp_may":   "c_exp_may_bytes",
            "exp_apr":   "c_exp_apr_bytes",
        }
        for _k, _sk in _map.items():
            _b = _local_bytes(_k)
            if _b:
                st.session_state[_sk] = _b
                _loaded.append(_k)
        if _loaded:
            st.success(f"{len(_loaded)}개 로드 완료")
        else:
            st.warning("감지된 파일 없음")

    st.divider()
    st.caption("버튼 없이 자동 적용: 탭에서 분석 버튼을 누르면 로컬 파일이 자동으로 사용됩니다.")

# 앱 시작 시 로컬 파일 자동 사전 로드 (세션에 없을 때만)
_AUTO_LOAD_MAP = {
    "fg":        "c_fg_bytes",
    "rs_stock":  "c_rs_bytes",
    "rs_etf":    "c_etf_xl_bytes",
    "trend_wk":  "c_weekly_bytes",
    "trend_sup": "c_supply_bytes",
    "trading":   "c_trading_bytes",
    "consensus": "c_consensus_bytes",
    "exp_may":   "c_exp_may_bytes",
    "exp_apr":   "c_exp_apr_bytes",
}
for _ak, _sk in _AUTO_LOAD_MAP.items():
    if _sk not in st.session_state:
        _b = _local_bytes(_ak)
        if _b:
            st.session_state[_sk] = _b

tab0, tab4, tab2, tab3, tab1, tab5 = st.tabs(["📊 대시보드", "🎯 종목 선정", "🇰🇷 국내 시장", "🔍 종목 분석", "🌎 미국 시장", "📦 수출 데이터"])

# ─────────────────────────────────────────────────────────────────────────────
# 공통 유틸
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def _dl(ticker, period="3y", start=None, end=None, retries=3):
    for i in range(retries):
        try:
            kw = dict(auto_adjust=True, progress=False, threads=False)
            df = yf.download(ticker, start=start, end=end, **kw) if start else yf.download(ticker, period=period, **kw)
            if not df.empty: return df
        except Exception: pass
        time.sleep(2 * (i + 1))
    return pd.DataFrame()

def _close(ticker, period="3y", start=None, end=None):
    df = _dl(ticker, period=period, start=start, end=end)
    if df.empty: return pd.Series(dtype=float)
    if isinstance(df.columns, pd.MultiIndex):
        try: return df["Close"][ticker].dropna()
        except: return df["Close"].iloc[:, 0].dropna()
    return df["Close"].dropna()

def _rsi(series, window=10):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    return 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

def _macd_hist(series, fast=12, slow=26, signal=9):
    m = series.ewm(span=fast, adjust=False).mean() - series.ewm(span=slow, adjust=False).mean()
    return m - m.ewm(span=signal, adjust=False).mean()

def _td_back(n):
    d = datetime.today(); cnt = 0
    while cnt < n:
        d -= timedelta(days=1)
        if d.weekday() < 5: cnt += 1
    return d.strftime("%Y-%m-%d")

def _heat_label(h):
    if h >= 7.5: return "🔴 과열"
    elif h >= 5.0: return "🟠 주의"
    elif h >= 2.5: return "🟡 보통"
    return "🟢 안전"

def _cv(v, fmt=".2f"):
    color = "#00c853" if v >= 0 else "#ff4b4b"
    sign = "+" if v >= 0 else ""
    arrow = "▲" if v >= 0 else "▼"
    return f'<span style="color:{color};font-weight:bold">{arrow} {sign}{v:{fmt}}</span>'

# ─────────────────────────────────────────────────────────────────────────────
# 수출 데이터 파싱 유틸
# ─────────────────────────────────────────────────────────────────────────────
def _parse_exp_date(s):
    """'2022년01월' or '2023년 1월' → pd.Timestamp"""
    m = re.match(r'(\d{4})년\s*(\d{1,2})월', str(s).strip())
    if m:
        try:
            return pd.Timestamp(int(m.group(1)), int(m.group(2)), 1)
        except Exception:
            pass
    return pd.NaT

def _load_exp_sheet(xl_bytes, sheet_name, layout="may"):
    """Parse one sheet from export Excel files.

    layout="may"  → 5월 잠정: date=col0, 일평균=col5, MoM=col6, YoY=col7
    layout="apr"  → 4월 확정: detect date column from '년월' in row2, then infer sibling cols
    Returns DataFrame with columns: 날짜, 금액, 일평균, MoM, YoY, 단가
    """
    try:
        df_raw = pd.read_excel(io.BytesIO(xl_bytes), sheet_name=sheet_name,
                                header=None, engine="openpyxl")
    except Exception:
        return pd.DataFrame()
    if df_raw.shape[0] < 5:
        return pd.DataFrame()

    data = df_raw.iloc[4:].copy().reset_index(drop=True)
    nc = data.shape[1]

    def _col(i):
        return pd.to_numeric(data.iloc[:, i] if nc > i else pd.Series(dtype=float), errors="coerce")

    if layout == "may":
        dates = data.iloc[:, 0].apply(_parse_exp_date)
        valid = dates.notna()
        data = data[valid].reset_index(drop=True)
        dates = dates[valid].reset_index(drop=True)
        nc = data.shape[1]
        result = pd.DataFrame({
            "날짜":   dates.values,
            "금액":   _col(15).values if nc > 15 else _col(5).values,
            "일평균": _col(5).values,
            "MoM":    _col(6).values,
            "YoY":    _col(7).values,
            "단가":   _col(8).values,
        })
    else:  # apr — dynamic column detection via row 2 header
        hdr2 = df_raw.iloc[2].tolist()
        hdr3 = df_raw.iloc[3].tolist()
        # Find date column: first col where hdr2 cell contains '년월'
        date_col = next((i for i, v in enumerate(hdr2) if isinstance(v, str) and "년월" in v), None)
        if date_col is None:
            # Fallback: scan data rows for first col whose values match date pattern
            for ci in range(min(nc, 12)):
                test = data.iloc[:, ci].astype(str).str.match(r"\d{4}년\s*\d{1,2}월")
                if test.sum() >= 3:
                    date_col = ci
                    break
        if date_col is None:
            return pd.DataFrame()

        dates = data.iloc[:, date_col].apply(_parse_exp_date)
        valid = dates.notna()
        data = data[valid].reset_index(drop=True)
        dates = dates[valid].reset_index(drop=True)
        nc = data.shape[1]

        # 금액(달러) is the column right after date_col
        amt_col = date_col + 1

        # Find 일평균수출 / mom / yoy in row3 headers (left block, before date_col)
        def _find_hdr3(keyword):
            for i, v in enumerate(hdr3[:date_col]):
                if isinstance(v, str) and keyword in v:
                    return i
            return None

        avg_col = _find_hdr3("일평균수출") or max(0, date_col - 3)
        mom_col = _find_hdr3("mom") or _find_hdr3("MoM") or (avg_col + 1)
        yoy_col = _find_hdr3("yoy") or _find_hdr3("YoY") or (avg_col + 2)
        # 단가 = 금액/중량 — usually at date_col + 3 or hdr3 position
        dan_col = next((i for i, v in enumerate(hdr3) if isinstance(v, str) and "달러" in v and "중량" in v), date_col + 3)

        result = pd.DataFrame({
            "날짜":   dates.values,
            "금액":   _col(amt_col).values,
            "일평균": _col(avg_col).values,
            "MoM":    _col(mom_col).values,
            "YoY":    _col(yoy_col).values,
            "단가":   _col(dan_col).values,
        })

    return result.dropna(subset=["날짜"]).reset_index(drop=True)

def _exp_chart(df_list, labels, metric="금액", title="수출 트렌드"):
    """Plotly chart: 수출 금액 + YoY bar for multiple categories."""
    n = len(df_list)
    if n == 0:
        return None
    colors = ["#58a6ff", "#3fb950", "#f78166", "#d2a8ff", "#ffa657",
              "#79c0ff", "#56d364", "#ff7b72", "#bc8cff", "#ffa07a"]
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.6, 0.4], vertical_spacing=0.06,
        subplot_titles=(title, "YoY (%)"),
    )
    for i, (df, lbl) in enumerate(zip(df_list, labels)):
        if df.empty or metric not in df.columns:
            continue
        c = colors[i % len(colors)]
        df = df.dropna(subset=["날짜", metric])
        fig.add_trace(go.Scatter(
            x=df["날짜"], y=df[metric], name=lbl,
            line=dict(color=c, width=2), mode="lines",
        ), row=1, col=1)
        if "YoY" in df.columns:
            df_yoy = df.dropna(subset=["YoY"])
            if not df_yoy.empty:
                bar_colors = ["#3fb950" if v >= 0 else "#f85149" for v in df_yoy["YoY"]]
                fig.add_trace(go.Bar(
                    x=df_yoy["날짜"], y=(df_yoy["YoY"] * 100).round(1),
                    name=f"{lbl} YoY", marker_color=bar_colors,
                    showlegend=False, opacity=0.75,
                ), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)", row=2, col=1)
    fig.update_layout(
        height=420, margin=dict(l=10, r=10, t=36, b=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e0e0e0", legend=dict(orientation="h", y=1.06),
        dragmode=False,
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
    return fig

@st.cache_data(ttl=3600, show_spinner=False)
def _scan_exp_anomalies(xl_bytes: bytes, layout: str,
                         yoy_thresh: float = 0.4, mom_thresh: float = 0.25) -> pd.DataFrame:
    """전체 시트를 스캔해 최신월 YoY/MoM 특이점을 DataFrame으로 반환.

    Returns columns: 시트명, 날짜, YoY_pct, MoM_pct, 일평균, 방향, 강도
    """
    _skip = {"반도체 수출 판가 증가율"}
    try:
        sheet_names = pd.ExcelFile(io.BytesIO(xl_bytes), engine="openpyxl").sheet_names
    except Exception:
        return pd.DataFrame()

    rows = []
    for sn in sheet_names:
        if sn in _skip:
            continue
        try:
            df = _load_exp_sheet(xl_bytes, sn, layout)
        except Exception:
            continue
        if df.empty or len(df) < 3:
            continue

        # 최근 3개월 중 유효한 YoY / MoM 데이터가 있는 마지막 행
        recent = df.tail(4).copy()
        # 가장 최근 유효 YoY
        yoy_row = recent.dropna(subset=["YoY"]).tail(1)
        mom_row = recent.dropna(subset=["MoM"]).tail(1)
        if yoy_row.empty and mom_row.empty:
            continue

        yoy_val = float(yoy_row["YoY"].iloc[0]) if not yoy_row.empty else np.nan
        mom_val = float(mom_row["MoM"].iloc[0]) if not mom_row.empty else np.nan
        avg_val = float(yoy_row["일평균"].iloc[0]) if not yoy_row.empty and pd.notna(yoy_row["일평균"].iloc[0]) else np.nan
        date_val = yoy_row["날짜"].iloc[0] if not yoy_row.empty else mom_row["날짜"].iloc[0]

        is_yoy_anomaly = pd.notna(yoy_val) and abs(yoy_val) >= yoy_thresh
        is_mom_anomaly = pd.notna(mom_val) and abs(mom_val) >= mom_thresh

        if not (is_yoy_anomaly or is_mom_anomaly):
            continue

        # 강도 = |YoY| 우선, 없으면 |MoM|
        strength = abs(yoy_val) if pd.notna(yoy_val) else abs(mom_val)
        direction = "▲" if (yoy_val if pd.notna(yoy_val) else mom_val) > 0 else "▼"

        rows.append({
            "시트명": sn,
            "날짜": date_val,
            "YoY_pct": round(yoy_val * 100, 1) if pd.notna(yoy_val) else np.nan,
            "MoM_pct": round(mom_val * 100, 1) if pd.notna(mom_val) else np.nan,
            "일평균": avg_val,
            "방향": direction,
            "강도": strength,
        })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    result = result.sort_values("강도", ascending=False).reset_index(drop=True)
    return result

# ─────────────────────────────────────────────────────────────────────────────
# 티커 이름 사전
# ─────────────────────────────────────────────────────────────────────────────
TICKER_NAMES = {
    "AAPL":"애플","MSFT":"마이크로소프트","NVDA":"엔비디아","AMZN":"아마존","META":"메타",
    "GOOGL":"구글A","GOOG":"구글C","TSLA":"테슬라","LLY":"일라이릴리","AVGO":"브로드컴",
    "JPM":"JP모건","UNH":"유나이티드헬스","XOM":"엑슨모빌","V":"비자","MA":"마스터카드",
    "PG":"P&G","JNJ":"존슨앤존슨","HD":"홈디포","COST":"코스트코","ABBV":"애브비",
    "MRK":"머크","NFLX":"넷플릭스","CVX":"쉐브론","BAC":"뱅크오브아메리카","CRM":"세일즈포스",
    "ORCL":"오라클","AMD":"AMD","WMT":"월마트","PLTR":"팔란티어","IBM":"IBM","CAT":"캐터필러",
    "AMGN":"암젠","NOW":"서비스나우","ISRG":"인튜이티브서지컬","QCOM":"퀄컴","UBER":"우버",
    "GS":"골드만삭스","HON":"허니웰","MS":"모건스탠리","BKNG":"부킹홀딩스","AXP":"아메리칸익스프레스",
    "BLK":"블랙록","GILD":"길리어드","PFE":"화이자","BA":"보잉","PANW":"팔로알토네트웍스",
    "MU":"마이크론","SBUX":"스타벅스","REGN":"리제네론","MELI":"메르카도리브레","MRNA":"모더나",
    "ASML":"ASML","NXPI":"NXP반도체","CRWD":"크라우드스트라이크","DDOG":"데이터독","ZS":"지스케일러",
    "INTU":"인튜이트","AMAT":"어플라이드머티리얼즈","LRCX":"램리서치","KLAC":"KLA","ADI":"아날로그디바이스",
    "INTC":"인텔","TXN":"텍사스인스트루먼트","CDNS":"케이던스","SNPS":"시놉시스","FTNT":"포티넷",
    "TEAM":"아틀라시안","WDAY":"워크데이","ADSK":"오토데스크","APP":"앱러빈","TTD":"더트레이드데스크",
    "SPY":"S&P500 ETF","QQQ":"나스닥100 ETF","IWM":"러셀2000 ETF","TLT":"20년국채 ETF",
    "GLD":"금 ETF","HYG":"하이일드채권 ETF","IEF":"중기국채 ETF","RSP":"동일가중S&P500",
    "XLK":"기술섹터","XLF":"금융섹터","XLV":"헬스케어섹터","XLE":"에너지섹터",
    "XLI":"산업재섹터","XLC":"통신섹터","XLY":"임의소비재섹터","XLP":"필수소비재섹터",
    "XLB":"소재섹터","XLRE":"리츠섹터","XLU":"유틸리티섹터",
    "SOXX":"반도체 ETF","IBB":"바이오테크 ETF","GDX":"금광업 ETF","ARKK":"혁신기업 ETF",
    "CIBR":"사이버보안 ETF","IGV":"소프트웨어 ETF","TAN":"태양광 ETF","URA":"우라늄 ETF",
    "JETS":"항공 ETF","KRE":"지역은행 ETF","ITA":"항공우주방산 ETF","LIT":"리튬배터리 ETF",
    "ACWI":"전세계주식 ETF","EEM":"신흥국 ETF","VEA":"선진국 ETF",
    "BRK-B":"버크셔해서웨이","PEP":"펩시코","ACN":"액센츄어","LIN":"린데","MCD":"맥도날드",
    "CSCO":"시스코","TMO":"써모피셔","ADBE":"어도비","TMUS":"T-모바일","GE":"GE에어로스페이스",
    "PM":"필립모리스","TXN":"텍사스인스트루먼트","RTX":"레이시온","SPGI":"S&P글로벌",
    "DHR":"다나허","NEE":"넥스트에라에너지","LOW":"로우스","UNP":"유니온퍼시픽",
    "SCHW":"찰스슈왑","C":"씨티그룹","SYK":"스트라이커","DE":"존디어","MDT":"메드트로닉",
    "AMAT":"어플라이드머티리얼즈","ETN":"이튼","VRTX":"버텍스파마","SBUX":"스타벅스",
    "CB":"처브","MMC":"마쉬맥레넌","SO":"서던컴퍼니","DUK":"듀크에너지","BSX":"보스턴사이언티픽",
    "PLD":"프롤로지스","CI":"시그나","ZTS":"조에티스","ICE":"인터콘티넨탈익스체인지",
    "CME":"CME그룹","WM":"웨이스트매니지먼트","APH":"암페놀","MCO":"무디스","ITW":"일리노이툴워크",
    "NOC":"노스롭그루먼","EMR":"에머슨일렉트릭",
    # ── 한국 대형주 (KOSPI) ──
    "005930.KS":"삼성전자","000660.KS":"SK하이닉스","005380.KS":"현대차","000270.KS":"기아",
    "005490.KS":"POSCO홀딩스","051910.KS":"LG화학","006400.KS":"삼성SDI","373220.KS":"LG에너지솔루션",
    "207940.KS":"삼성바이오로직스","068270.KS":"셀트리온","035420.KS":"NAVER","035720.KS":"카카오",
    "012330.KS":"현대모비스","066570.KS":"LG전자","028260.KS":"삼성물산","009150.KS":"삼성전기",
    "011070.KS":"LG이노텍","032830.KS":"삼성생명","086790.KS":"하나금융지주","105560.KS":"KB금융",
    "055550.KS":"신한지주","316140.KS":"우리금융지주","138040.KS":"메리츠금융지주",
    "096770.KS":"SK이노베이션","010950.KS":"S-Oil","015760.KS":"한국전력","036460.KS":"한국가스공사",
    "017670.KS":"SK텔레콤","030200.KS":"KT","032640.KS":"LGU+",
    "012450.KS":"한화에어로스페이스","047810.KS":"한국항공우주","079550.KS":"LIG넥스원",
    "009540.KS":"한국조선해양","329180.KS":"HD현대중공업","010140.KS":"삼성중공업","267260.KS":"HD현대일렉트릭",
    "011210.KS":"현대위아","298040.KS":"효성중공업","010120.KS":"LS일렉트릭","028050.KS":"삼성엔지니어링",
    "000720.KS":"현대건설","006360.KS":"GS건설","047040.KS":"대우건설","028260.KS":"삼성물산",
    "010130.KS":"고려아연","011170.KS":"롯데케미칼","009830.KS":"한화솔루션",
    "128940.KS":"한미약품","326030.KS":"SK바이오팜","145020.KS":"휴젤",
    "036570.KS":"엔씨소프트","259960.KS":"크래프톤","251270.KS":"넷마블","352820.KS":"하이브",
    "139480.KS":"이마트","004170.KS":"신세계","023530.KS":"롯데쇼핑","000120.KS":"CJ대한통운",
    "267250.KS":"HD현대","034020.KS":"두산에너빌리티","042660.KS":"한화오션",
    "003670.KS":"포스코퓨처엠","011790.KS":"SKC","096775.KS":"SK엔무브",
    # ── 한국 중소형/코스닥 ──
    "196170.KQ":"알테오젠","086520.KQ":"에코프로","247540.KQ":"에코프로비엠",
    "066970.KQ":"엘앤에프","278280.KQ":"천보","035900.KQ":"JYP Ent.","041510.KQ":"에스엠",
    "277810.KS":"레인보우로보틱스","454910.KS":"두산로보틱스","090360.KS":"로보스타",
    "263750.KQ":"펄어비스","036030.KQ":"KG이니시스","058470.KQ":"리노공업",
    "214150.KQ":"클래시스","091990.KQ":"셀트리온헬스케어","323410.KS":"카카오뱅크",
    # ── 코스닥 반도체/장비 ──
    "042700.KQ":"한미반도체","240810.KQ":"원익IPS","039030.KQ":"이오테크닉스",
    "089030.KQ":"테크윙","095340.KQ":"ISC","319660.KQ":"피에스케이홀딩스",
    "085660.KQ":"넥스틴","064760.KQ":"티씨케이","218410.KQ":"RFHIC",
    "211050.KQ":"이수페타시스","076410.KQ":"티엘비","084370.KQ":"유진테크",
    "036710.KQ":"심텍","183300.KQ":"코미코","008060.KQ":"대덕전자",
    "140860.KQ":"파크시스템스","102710.KQ":"이엔에프테크놀로지",
    "195870.KQ":"해성디에스","067310.KQ":"하나마이크론",
    "007810.KS":"코리아써키트","357780.KQ":"솔브레인",
    # ── 코스닥 바이오/헬스케어 ──
    "028300.KQ":"HLB","141080.KQ":"리가켐바이오","009420.KQ":"한올바이오파마",
    "298380.KQ":"에이비엘바이오","039200.KQ":"오스코텍","214450.KQ":"파마리서치",
    "226950.KQ":"올릭스","115180.KQ":"큐리언트","214270.KQ":"파미셀",
    "195940.KQ":"HK이노엔","241710.KQ":"코스메카코리아","476000.KQ":"달바글로벌",
    "278470.KQ":"에이피알","108490.KQ":"로보티즈",
    "950200.KQ":"프로티나","067630.KQ":"HLB생명과학",
    # ── 코스닥 기타 ──
    "012510.KQ":"더존비즈온","030190.KQ":"NICE평가정보",
    "293490.KQ":"카카오게임즈","122870.KQ":"와이지엔터테인먼트",
    "194370.KQ":"제이에스코퍼레이션","093370.KQ":"후성",
    "077970.KQ":"STX엔진","429530.KQ":"HD현대마린엔진",
    "082740.KQ":"한화엔진","161890.KQ":"한국콜마",
    "032350.KQ":"롯데관광개발","691610.KQ":"지투지바이오",
}

US_SECTOR_ETFS = {
    "XLK":"기술","SOXX":"반도체","IGV":"소프트웨어","CIBR":"사이버보안",
    "XLF":"금융","KRE":"지역은행",
    "XLV":"헬스케어","IBB":"바이오테크",
    "XLE":"에너지","XOP":"석유가스",
    "XLI":"산업재","ITA":"항공우주방산",
    "XLC":"통신서비스",
    "XLY":"임의소비재","JETS":"항공",
    "XLP":"필수소비재",
    "XLB":"소재","GDX":"금광업",
    "XLRE":"리츠","XLU":"유틸리티",
    "TAN":"태양광","URA":"우라늄","LIT":"리튬배터리","ARKK":"혁신기업",
}

KOREA_ETFS = {
    "091170.KS":"KODEX 은행","140700.KS":"KODEX 보험","102970.KS":"KODEX 증권",
    "117700.KS":"KODEX 건설","300950.KS":"KODEX 게임산업","395160.KS":"KODEX 시스템반도체",
    "445290.KS":"KODEX K-로봇","117460.KS":"KODEX 에너지화학","091160.KS":"KODEX 반도체",
    "244580.KS":"KODEX 바이오","228800.KS":"TIGER 여행레저","364970.KS":"TIGER 바이오TOP10",
    "091180.KS":"KODEX 자동차","305540.KS":"TIGER 2차전지테마","266360.KS":"KODEX 미디어엔터",
    "228790.KS":"TIGER 화장품","463250.KS":"TIGER 우주방산","157490.KS":"TIGER 소프트웨어",
    "449450.KS":"PLUS K방산","139230.KS":"TIGER 200중공업","466920.KS":"SOL 조선TOP3",
    "475300.KS":"SOL 반도체전공정","475310.KS":"SOL 반도체후공정","307510.KS":"TIGER 의료기기",
    "433500.KS":"ACE 원자력테마","261070.KS":"TIGER 코스닥바이오","479850.KS":"HANARO K뷰티",
    "381570.KS":"HANARO 친환경에너지","438900.KS":"HANARO FN K-푸드",
}

SECTOR_ETFS = {
    "091160":"반도체","305720":"2차전지","244580":"바이오","091180":"자동차",
    "139270":"금융","266370":"IT","445290":"로봇","139250":"건설기계",
    "139220":"소재/화학","117460":"에너지화학","143860":"헬스케어","139260":"정보기술",
    "466920":"전력기기","449450":"방산","494670":"전력TOP10",
}

_FALLBACK = {
    "반도체":[("005930","삼성전자"),("000660","SK하이닉스"),("042700","한미반도체")],
    "전력기기":[("267260","HD현대일렉트릭"),("298040","효성중공업"),("010120","LS일렉트릭")],
    "자동차":[("005380","현대차"),("000270","기아"),("012330","현대모비스")],
    "방산":[("012450","한화에어로스페이스"),("047810","한국항공우주"),("079550","LIG넥스원")],
    "바이오":[("207940","삼성바이오로직스"),("068270","셀트리온"),("128940","한미약품")],
    "2차전지":[("373220","LG에너지솔루션"),("006400","삼성SDI"),("051910","LG화학")],
    "로봇":[("277810","레인보우로보틱스"),("454910","두산로보틱스"),("090360","로보스타")],
    "IT":[("005930","삼성전자"),("000660","SK하이닉스"),("035420","NAVER"),("035720","카카오")],
    "정보기술":[("005930","삼성전자"),("000660","SK하이닉스"),("035420","NAVER"),("035720","카카오")],
}

NAME_TO_TICKER = {v.lower(): k for k, v in TICKER_NAMES.items()}

@st.cache_data(ttl=timedelta(hours=24), show_spinner=False)
def _load_all_kr_tickers() -> dict:
    """pykrx로 KOSPI/KOSDAQ 전종목 {티커(.KS/.KQ): 한글명} 딕셔너리 (24h 캐시)"""
    try:
        from pykrx import stock as krx
        import datetime as _dt
        today = _dt.datetime.now()
        # 최근 영업일(최대 7일 전까지)
        date_str = today.strftime("%Y%m%d")
        result = {}
        for market, suffix in (("KOSPI", ".KS"), ("KOSDAQ", ".KQ")):
            try:
                codes = krx.get_market_ticker_list(date_str, market=market)
                if not codes:
                    # 주말/공휴일이면 가장 최근 영업일로 재시도
                    for delta in range(1, 8):
                        d = today - _dt.timedelta(days=delta)
                        if d.weekday() < 5:
                            codes = krx.get_market_ticker_list(d.strftime("%Y%m%d"), market=market)
                            if codes:
                                break
                for code in (codes or []):
                    try:
                        name = krx.get_market_ticker_name(code)
                        if name:
                            result[f"{code}{suffix}"] = name
                    except Exception:
                        pass
            except Exception:
                pass
        return result
    except Exception:
        return {}

NAVER_ETF_URL = "https://finance.naver.com/api/sise/etfItemList.nhn?etfType=0"
HEADERS = {"User-Agent":"Mozilla/5.0","Referer":"https://finance.naver.com"}
SP500_TOP100 = ["AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","BRK-B","TSLA","LLY","AVGO","JPM","UNH","XOM","V","MA","PG","JNJ","HD","COST","ABBV","MRK","NFLX","CVX","BAC","CRM","ORCL","AMD","PEP","ACN","WMT","LIN","MCD","CSCO","TMO","ADBE","PLTR","TMUS","INTU","GE","IBM","CAT","PM","AMGN","TXN","NOW","ISRG","QCOM","UBER","GS","VZ","HON","RTX","SPGI","DHR","NEE","MS","LOW","T","UNP","BKNG","AXP","SCHW","C","BLK","SYK","GILD","PFE","DE","MDT","BA","AMAT","ADI","LRCX","PANW","MU","TJX","ETN","VRTX","KLAC","SBUX","CB","MMC","SO","DUK","BSX","REGN","PLD","CI","ZTS","ICE","CME","WM","APH","MCO","SNPS","CDNS","ITW","NOC","EMR"]
NASDAQ100 = ["AAPL","ABNB","ADBE","ADI","ADP","ADSK","AEP","AMAT","AMD","AMGN","AMZN","ANSS","ASML","AVGO","AZN","BIIB","BKNG","BKR","CCEP","CDNS","CDW","CEG","CHTR","CMCSA","COST","CPRT","CRWD","CSCO","CSX","CTAS","CTSH","DASH","DDOG","DLTR","DXCM","EA","EXC","FANG","FAST","FTNT","GEHC","GILD","GOOG","GOOGL","HON","IDXX","INTC","INTU","ISRG","KDP","KHC","KLAC","LIN","LRCX","LULU","MAR","MCHP","MDLZ","MELI","META","MNST","MRNA","MRVL","MSFT","MU","NFLX","NVDA","NXPI","ODFL","ON","ORLY","PANW","PAYX","PCAR","PDD","PEP","PYPL","QCOM","REGN","ROP","ROST","SBUX","SNPS","TEAM","TMUS","TSLA","TTD","TXN","VRSK","VRTX","WDAY","XEL","ZS"]

# ─────────────────────────────────────────────────────────────────────────────
# 미국 지표 함수
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def get_canary_signal():
    try:
        end = datetime.today().strftime("%Y-%m-%d")
        start = (datetime.today() - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
        results = {}
        for t in ["QQQ","TIP"]:
            px = _close(t, start=start, end=end)
            if len(px) < 260: return None
            results[t] = float(((px.iloc[-1]/px.iloc[-22]-1)+(px.iloc[-1]/px.iloc[-63]-1)+(px.iloc[-1]/px.iloc[-126]-1)+(px.iloc[-1]/px.iloc[-252]-1))/4)
        return {"qqq_mom":results["QQQ"],"tip_mom":results["TIP"],"mode":"공격" if all(v>0 for v in results.values()) else "방어"}
    except Exception as e: return {"error":str(e)}

@st.cache_data(ttl=3600, show_spinner=False)
def get_bofa_heat():
    try:
        tm = {"SPY":"SPY","QQQ":"QQQ","RSP":"RSP","VIX":"^VIX","HYG":"HYG","IEF":"IEF","LQD":"LQD"}
        raw = yf.download(list(tm.values()), start="2015-01-01", auto_adjust=True, progress=False)
        if raw.empty: return None
        px = raw["Close"].copy().rename(columns={v:k for k,v in tm.items()})
        def zs(s): mu=s.rolling(252).mean(); sd=s.rolling(252).std(ddof=0); return (s-mu)/sd
        def n01(z,lo=0.0,hi=2.0): return ((z-lo)/(hi-lo)).clip(0,1)
        df = pd.DataFrame(index=px.index)
        if {"HYG","IEF"}.issubset(px.columns): df["h_risk"] = n01(zs(px["HYG"]/px["IEF"]))
        if {"HYG","LQD"}.issubset(px.columns): df["h_credit"] = n01(zs(px["HYG"]/px["LQD"]))
        if {"RSP","SPY"}.issubset(px.columns): df["h_style"] = ((0-zs(px["RSP"]/px["SPY"]))/2).clip(0,1)
        if "SPY" in px.columns:
            spy_ma200 = px["SPY"].rolling(200).mean()
            df["h_spy_ext"] = n01(zs(px["SPY"]/spy_ma200-1),0.5,2.0)
        if "QQQ" in px.columns:
            qqq_ma200 = px["QQQ"].rolling(200).mean()
            df["h_qqq_ext"] = n01(zs(px["QQQ"]/qqq_ma200-1),0.5,2.0)
        hc = [c for c in df.columns if c.startswith("h_")]
        W = pd.Series({"h_risk":1.2,"h_credit":1.0,"h_style":0.8,"h_spy_ext":0.8,"h_qqq_ext":0.8})
        W = W[[c for c in hc if c in W.index]]; W = W/W.sum()
        heat = ((df[hc]*W).sum(axis=1)*10).rolling(10).mean()
        spy_ma200 = px["SPY"].rolling(200).mean()
        trend_on = (px["SPY"]>spy_ma200)&(spy_ma200.diff(20)>0)
        heat = pd.Series(np.maximum(heat.values,np.where(trend_on,2.5,0.0)),index=heat.index)
        shock_vix = bool(px["VIX"].pct_change(3).iloc[-1]>=0.30) if "VIX" in px.columns else False
        shock_credit = bool((px["HYG"]/px["LQD"]).pct_change(5).iloc[-1]<=-0.02) if {"HYG","LQD"}.issubset(px.columns) else False
        return {"heat":round(float(heat.dropna().iloc[-1]),2),"shock":shock_vix or shock_credit,
                "shock_vix":shock_vix,"shock_credit":shock_credit,"trend_on":bool(trend_on.dropna().iloc[-1])}
    except Exception as e: return {"error":str(e)}

@st.cache_data(ttl=3600, show_spinner=False)
def get_blood_indicator():
    try:
        raw = yf.download(["^IRX","^TNX","HYG","IEF"], start="2015-01-01", auto_adjust=True, progress=False)
        px = raw["Close"].copy()
        irx = px["^IRX"]   # Yahoo quotes in %, e.g. 5.0 = 5%
        t10y = px["^TNX"]  # Yahoo quotes in %, e.g. 4.5 = 4.5%
        hyg_ticker = yf.Ticker("HYG")
        hyg_yield_raw = getattr(hyg_ticker.fast_info,"dividend_yield",None) or hyg_ticker.info.get("dividendYield",0.06)
        hyg_yield = hyg_yield_raw * 100 if hyg_yield_raw < 1 else hyg_yield_raw  # convert decimal to %
        # hyg_yield는 현재 스칼라값이므로 blood 시계열 MA는 신뢰도 낮음 → 현재값+단기MA만 사용
        blood = (irx/(hyg_yield - t10y)).dropna()
        cur=float(blood.iloc[-1]); ma20=float(blood.rolling(20).mean().iloc[-1]); ma60=float(blood.rolling(60).mean().iloc[-1]) if len(blood)>=60 else ma20
        return {"value":round(cur,4),"ma20":round(ma20,4),"ma60":round(ma60,4),"vs_ma20":"위" if cur>ma20 else "아래","vs_ma60":"위" if cur>ma60 else "아래"}
    except Exception as e: return {"error":str(e)}

@st.cache_data(ttl=3600, show_spinner=False)
def get_us_fear_greed():
    try:
        tickers = ["^GSPC","^IXIC","^VIX","^TNX","^FVX","HYG","IEF","QQQ","SPY"]
        raw = yf.download(tickers, start="2024-01-01", auto_adjust=True, progress=False)
        data = raw["Close"].rename(columns={"^GSPC":"SP500","^IXIC":"NASDAQ","^VIX":"VIX","^TNX":"T10Y","^FVX":"T5Y"}).dropna(subset=["SP500","NASDAQ","VIX","T10Y","T5Y","HYG","IEF","QQQ","SPY"])
        data["RiskApp"] = data["HYG"]/data["IEF"]
        def calc_fg(df, col, label):
            df[f"{label}_MA125"] = df[col].rolling(125).mean()
            df[f"{label}_Mom"] = (df[col]-df[f"{label}_MA125"])/df[f"{label}_MA125"]*100
            df[f"{label}_RSI"] = _rsi(df[col]); df[f"{label}_BS"] = df["T10Y"]-df["T5Y"]
            df[f"{label}_VIX"] = df["VIX"]; df[f"{label}_RA"] = df["RiskApp"]
            cols = [f"{label}_Mom",f"{label}_RSI",f"{label}_BS",f"{label}_VIX",f"{label}_RA"]
            v = df[cols].dropna()
            if v.empty: return df
            df.loc[v.index, cols] = MinMaxScaler().fit_transform(v)
            df[f"{label}_FGI"] = (df[f"{label}_Mom"]+df[f"{label}_RA"]+(1-df[f"{label}_VIX"])+df[f"{label}_BS"]+df[f"{label}_RSI"])*0.2
            df[f"{label}_Osc"] = _macd_hist(df[f"{label}_FGI"])
            return df
        data = calc_fg(data,"SP500","SPX"); data = calc_fg(data,"NASDAQ","NDX")
        for col, lbl in [("SPY","SPY"),("QQQ","QQQ")]:
            for w in [20,60,120,200]: data[f"{lbl}_MA{w}"] = data[col].rolling(w).mean()
            data[f"{lbl}_SuperMA"] = data[[f"{lbl}_MA{w}" for w in [20,60,120,200]]].mean(axis=1)
            data[f"{lbl}_GapPct"] = (data[col]-data[f"{lbl}_SuperMA"])/data[f"{lbl}_SuperMA"]*100
            data[f"{lbl}_EMA13"] = data[col].ewm(span=13,adjust=False).mean()
            data[f"{lbl}_MACDh"] = _macd_hist(data[col])
        def impulse(df, lbl):
            last = df[[f"{lbl}_EMA13",f"{lbl}_MACDh"]].dropna()
            if len(last)<2: return "알수없음"
            eu = last[f"{lbl}_EMA13"].iloc[-1]>last[f"{lbl}_EMA13"].iloc[-2]
            mu = last[f"{lbl}_MACDh"].iloc[-1]>last[f"{lbl}_MACDh"].iloc[-2]
            return "🟢 강세" if eu and mu else ("🔴 약세" if not eu and not mu else "🔵 중립")
        def td_setup(s):
            p=s.values; sell=np.zeros(len(p)); buy=np.zeros(len(p))
            for i in range(len(p)):
                sell[i]=sell[i-1]+1 if i>=4 and p[i]>p[i-4] else 0
                buy[i]=buy[i-1]+1 if i>=2 and p[i]<p[i-2] else 0
            return int(sell[-1]),int(buy[-1])
        last = data.dropna(subset=["SPX_Osc","NDX_Osc"])
        spy_ts,spy_tb = td_setup(data["SPY"].dropna()); qqq_ts,qqq_tb = td_setup(data["QQQ"].dropna())
        ch = last.tail(180)
        return {"spx_osc":round(float(last["SPX_Osc"].iloc[-1]),4),"ndx_osc":round(float(last["NDX_Osc"].iloc[-1]),4),
                "spx_sentiment":"탐욕" if last["SPX_Osc"].iloc[-1]>0 else "공포",
                "ndx_sentiment":"탐욕" if last["NDX_Osc"].iloc[-1]>0 else "공포",
                "spy_gap":round(float(data["SPY_GapPct"].dropna().iloc[-1]),2),
                "qqq_gap":round(float(data["QQQ_GapPct"].dropna().iloc[-1]),2),
                "spy_impulse":impulse(data,"SPY"),"qqq_impulse":impulse(data,"QQQ"),
                "spy_td_sell":spy_ts,"spy_td_buy":spy_tb,"qqq_td_sell":qqq_ts,"qqq_td_buy":qqq_tb,
                "chart":{"dates":[str(d.date()) for d in ch.index],
                         "spx_osc":ch["SPX_Osc"].tolist(),"ndx_osc":ch["NDX_Osc"].tolist(),
                         "spy":data["SPY"].reindex(ch.index).tolist()}}
    except Exception as e: return {"error":str(e)}

@st.cache_data(ttl=3600, show_spinner=False)
def get_monthly_fear_greed():
    try:
        start = (datetime.today()-timedelta(days=25*365)).strftime('%Y-%m-%d')
        raw = yf.download(['^GSPC','^IXIC','^VIX','^TNX','^FVX','HYG','IEF'], start=start, auto_adjust=True, progress=False)
        data = raw["Close"].rename(columns={'^GSPC':'SP500','^IXIC':'NASDAQ','^VIX':'VIX','^TNX':'10Y','^FVX':'5Y'})
        data = data.resample('ME').ffill().dropna(); data['RA'] = data['HYG']/data['IEF']
        def calc(df, col, lbl):
            df[f'{lbl}_MA125'] = df[col].rolling(60).mean()  # 월간: 60개월=5년 기준
            df[f'{lbl}_Mom'] = (df[col]-df[f'{lbl}_MA125'])/df[f'{lbl}_MA125']*100
            df[f'{lbl}_RSI'] = _rsi(df[col],10); df[f'{lbl}_BS'] = df['10Y']-df['5Y']
            df[f'{lbl}_VIX'] = df['VIX']; df[f'{lbl}_RA'] = df['RA']
            cols=[f'{lbl}_Mom',f'{lbl}_RSI',f'{lbl}_BS',f'{lbl}_VIX',f'{lbl}_RA']
            v=df[cols].dropna()
            if v.empty: return df
            df.loc[v.index,cols]=MinMaxScaler().fit_transform(v)
            df[f'{lbl}_FGI']=(df[f'{lbl}_Mom']+df[f'{lbl}_RA']+(1-df[f'{lbl}_VIX'])+df[f'{lbl}_BS']+df[f'{lbl}_RSI'])*0.2
            ema6=df[f'{lbl}_FGI'].ewm(span=6,adjust=False).mean(); ema19=df[f'{lbl}_FGI'].ewm(span=19,adjust=False).mean()
            macd=ema6-ema19; df[f'{lbl}_Osc']=macd-macd.ewm(span=6,adjust=False).mean()
            return df
        data=calc(data,'SP500','SPX'); data=calc(data,'NASDAQ','NDX')
        r=data.dropna(subset=['SPX_Osc','NDX_Osc'])
        if r.empty: return {"error":"데이터 부족"}
        return {"date":r.index[-1].strftime('%Y-%m'),"spx_osc":round(float(r['SPX_Osc'].iloc[-1]),4),
                "ndx_osc":round(float(r['NDX_Osc'].iloc[-1]),4),"spx_fgi":round(float(r['SPX_FGI'].iloc[-1]),4),
                "spx_sentiment":"탐욕" if r['SPX_Osc'].iloc[-1]>0 else "공포",
                "ndx_sentiment":"탐욕" if r['NDX_Osc'].iloc[-1]>0 else "공포"}
    except Exception as e: return {"error":str(e)}

@st.cache_data(ttl=3600, show_spinner=False)
def get_coppock():
    try:
        results = {}
        for t in ["SPY","QQQ","^GSPC"]:
            px=_close(t,period="5y")
            if px.empty: continue
            mo=px.resample("ME").last(); mo=pd.concat([mo,px.iloc[[-1]]]); mo=mo[~mo.index.duplicated(keep="last")].sort_index()
            cop=(mo.pct_change(14)+mo.pct_change(11)).ewm(span=10,adjust=False).mean()*100
            last=cop.dropna()
            if last.empty: continue
            val=float(last.iloc[-1]); prev=float(last.iloc[-2]) if len(last)>=2 else val
            results[{"SPY":"SPY","QQQ":"QQQ","^GSPC":"S&P500"}.get(t,t)]={"value":round(val,2),"trend":"상승" if val>prev else "하락","pos":val>0}
        return results
    except Exception as e: return {"error":str(e)}

@st.cache_data(ttl=3600, show_spinner=False)
def get_coppock_fast():
    try:
        results = {}
        for t in ["SPY","QQQ","^GSPC"]:
            px=_close(t,period="5y")
            if px.empty: continue
            mo=px.resample("ME").last(); mo=pd.concat([mo,px.iloc[[-1]]]); mo=mo[~mo.index.duplicated(keep="last")].sort_index()
            cop=(mo.pct_change(4)+mo.pct_change(6)).rolling(3).mean()*100
            last=cop.dropna()
            if last.empty: continue
            val=float(last.iloc[-1]); prev=float(last.iloc[-2]) if len(last)>=2 else val
            results[{"SPY":"SPY","QQQ":"QQQ","^GSPC":"S&P500"}.get(t,t)]={"value":round(val,2),"trend":"상승" if val>prev else "하락","pos":val>0}
        return results
    except Exception as e: return {"error":str(e)}

@st.cache_data(ttl=3600, show_spinner=False)
def get_zbt():
    try:
        universe = list(set(NASDAQ100 + SP500_TOP100[:80]))
        end_date = datetime.today(); start_date = end_date - timedelta(days=30)
        raw = yf.download(universe, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"), auto_adjust=True, progress=False)
        if raw.empty: return {"error":"데이터 없음"}
        data = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        data = data.dropna(axis=1, how="all")
        daily_up = (data.pct_change()>0).sum(axis=1)/data.shape[1]
        zbt = daily_up.rolling(10).mean()
        cur = float(zbt.dropna().iloc[-1])
        prev_min = float(zbt.dropna().iloc[-10:-1].min()) if len(zbt.dropna())>=10 else cur
        vix_px = _close("^VIX", period="5d")
        vix_val = float(vix_px.iloc[-1]) if not vix_px.empty else None
        return {"zbt":round(cur,3),"prev_min":round(prev_min,3),
                "signal":cur>0.615 and prev_min<0.40,
                "vix":round(vix_val,1) if vix_val else None,
                "vix_ok":vix_val<20 if vix_val else None}
    except Exception as e: return {"error":str(e)}

@st.cache_data(ttl=3600, show_spinner=False)
def get_sp500_rs(tickers, top_n=10):
    try:
        spy_px=_close("SPY",period="3y")
        if spy_px.empty: return {"error":"SPY 없음"}
        rows=[]
        for i in range(0,len(tickers),50):
            raw=yf.download(tickers[i:i+50],period="3y",auto_adjust=True,progress=False)
            if raw.empty: continue
            px=raw["Close"] if isinstance(raw.columns,pd.MultiIndex) else raw
            if isinstance(px,pd.Series): px=px.to_frame()
            for t in px.columns:
                s=px[t].dropna(); al=pd.concat([s,spy_px],axis=1,join="inner"); al.columns=["stock","spy"]
                if len(al)<60: continue
                rs_vals=[]
                for win in [60,120,250]:
                    if len(al)<win: continue
                    rel=al["stock"]/al["spy"]; ma=rel.rolling(win).mean(); rs=((rel/ma)-1)*100
                    if not rs.dropna().empty: rs_vals.append(float(rs.dropna().iloc[-1]))
                if rs_vals: rows.append({"ticker":t,"rs":np.mean(rs_vals)})
            time.sleep(0.1)
        if not rows: return {"error":"RS 계산 실패"}
        df=pd.DataFrame(rows).sort_values("rs",ascending=False).head(top_n)
        return {"top":[{"ticker":r["ticker"],"rs":round(r["rs"],1)} for _,r in df.iterrows()]}
    except Exception as e: return {"error":str(e)}

@st.cache_data(ttl=3600, show_spinner=False)
def get_nasdaq100_rs(tickers, top_n=10):
    try:
        raw=yf.download(tickers+["SPY"],period="1y",auto_adjust=True,progress=False)
        if raw.empty: return {"error":"데이터 없음"}
        data=raw["Close"].ffill().dropna(axis=1,how="any")
        if "SPY" not in data.columns: return {"error":"SPY 없음"}
        spy_ret=(data["SPY"].pct_change().rolling(63).mean()*0.5+data["SPY"].pct_change().rolling(126).mean()*0.3+data["SPY"].pct_change().rolling(252).mean()*0.2).iloc[-1]
        rs_dict={}
        for t in data.columns:
            if t=="SPY": continue
            mom=(data[t].pct_change().rolling(63).mean()*0.5+data[t].pct_change().rolling(126).mean()*0.3+data[t].pct_change().rolling(252).mean()*0.2).iloc[-1]
            if spy_ret!=0: rs_dict[t]=float(mom/spy_ret)
        top=sorted(rs_dict,key=rs_dict.get,reverse=True)[:top_n]
        return {"top":[{"ticker":t,"rs":round(rs_dict[t],2)} for t in top]}
    except Exception as e: return {"error":str(e)}

@st.cache_data(ttl=3600, show_spinner=False)
def get_us_sector_rs():
    try:
        tks=list(US_SECTOR_ETFS.keys())
        raw=yf.download(tks+["SPY"],period="2y",auto_adjust=True,progress=False)
        if raw.empty: return {"error":"데이터 없음"}
        data=raw["Close"].ffill().dropna(axis=1,how="any")
        if "SPY" not in data.columns: return {"error":"SPY 없음"}
        results=[]
        for t in tks:
            if t not in data.columns: continue
            etf=data[t]; spy=data["SPY"]; rel=etf/spy
            rs_vals=[]
            for win in [60,120,250]:
                if len(rel)<win: continue
                ma=rel.rolling(win).mean(); rs=((rel/ma)-1)*100
                if not rs.dropna().empty: rs_vals.append(float(rs.dropna().iloc[-1]))
            rs_raw=round(np.mean(rs_vals),2) if rs_vals else 0
            norm_rs=round(100*(1/(1+np.exp(-rs_raw/12))),1)
            mom3=float(etf.pct_change().rolling(63).mean().iloc[-1]) if len(etf)>=63 else 0
            vol3=float(etf.pct_change().rolling(63).std().iloc[-1]) if len(etf)>=63 else 1
            risk_adj=round((mom3/vol3)*100,2) if vol3>0 else 0
            results.append({"ticker":t,"name":US_SECTOR_ETFS.get(t,t),"norm_rs":norm_rs,"rs_raw":rs_raw,"risk_adj":risk_adj})
        results.sort(key=lambda x:x["norm_rs"],reverse=True)
        return {"sectors":results}
    except Exception as e: return {"error":str(e)}

# ─────────────────────────────────────────────────────────────────────────────
# 국내 지표 함수
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=timedelta(minutes=5))
def get_market_summary():
    """코스피/코스닥 — 네이버 실시간 polling API (고가/저가/등락률 포함)"""
    hdrs = {"User-Agent":"Mozilla/5.0","Referer":"https://finance.naver.com/"}
    def _fetch_index(idx_code):
        try:
            r = requests.get(f"https://polling.finance.naver.com/api/realtime/domestic/index/{idx_code}",
                             headers=hdrs, timeout=8)
            d = r.json()["datas"][0]
            def _n(k): return float(str(d.get(k,"0")).replace(",","")) if d.get(k) else 0.0
            close = _n("closePrice"); prev = close - _n("compareToPreviousClosePrice")
            high  = _n("highPrice");  low  = _n("lowPrice"); open_ = _n("openPrice")
            pct   = float(d.get("fluctuationsRatio","0") or 0)
            vol_range = round((high - low) / prev * 100, 2) if prev > 0 else 0.0
            return {
                "close": close, "prev": round(prev, 2),
                "chg": round(_n("compareToPreviousClosePrice"), 2),
                "chg_pct": pct,
                "open": open_, "high": high, "low": low,
                "vol_range": vol_range,  # 하루 변동폭 (고가-저가)/전일종가
                "direction": d.get("compareToPreviousPrice",{}).get("text",""),
                "market_status": d.get("marketStatus",""),
            }
        except: return None
    kp = _fetch_index("KOSPI")
    kq = _fetch_index("KOSDAQ")
    if not kp or not kq:
        # fallback: Yahoo Finance
        try:
            start=_td_back(5); end=datetime.today().strftime("%Y-%m-%d")
            kps=_close("^KS11",start=start,end=end); kqs=_close("^KQ11",start=start,end=end)
            if kps.empty or kqs.empty: return {"error":"데이터 없음"}
            kp_l=float(kps.iloc[-1]); kp_p=float(kps.iloc[-2]) if len(kps)>=2 else kp_l
            kq_l=float(kqs.iloc[-1]); kq_p=float(kqs.iloc[-2]) if len(kqs)>=2 else kq_l
            kp={"close":kp_l,"chg_pct":round((kp_l/kp_p-1)*100,2),"chg":round(kp_l-kp_p,2),"high":kp_l,"low":kp_l,"open":kp_l,"vol_range":0,"direction":"","market_status":"CLOSE"}
            kq={"close":kq_l,"chg_pct":round((kq_l/kq_p-1)*100,2),"chg":round(kq_l-kq_p,2),"high":kq_l,"low":kq_l,"open":kq_l,"vol_range":0,"direction":"","market_status":"CLOSE"}
        except Exception as e: return {"error":str(e)}
    import datetime as _dt
    _kst = _dt.datetime.utcnow() + _dt.timedelta(hours=9)
    return {"date": _kst.strftime("%Y-%m-%d %H:%M"), "kospi": kp, "kosdaq": kq}

@st.cache_data(ttl=timedelta(minutes=30))
def get_sector_performance():
    try:
        r=requests.get(NAVER_ETF_URL,headers=HEADERS,timeout=15); etfs=r.json()["result"]["etfItemList"]
        em={e["itemcode"]:e for e in etfs}; sd={}
        for code,name in SECTOR_ETFS.items():
            if code in em: sd[name]={"chg_pct":round(em[code]["changeRate"],2)}
        if not sd: return {"error":"섹터 데이터 없음"}
        ss=sorted(sd.items(),key=lambda x:x[1]["chg_pct"],reverse=True)
        return {"sectors":sd,"top3":[(n,d["chg_pct"]) for n,d in ss[:3]],"bot3":[(n,d["chg_pct"]) for n,d in ss[-3:]]}
    except Exception as e: return {"error":str(e)}

@st.cache_data(ttl=timedelta(minutes=30))
def get_supply_oscillator():
    try:
        start=_td_back(25); end=datetime.today().strftime("%Y-%m-%d")
        kp=_close("^KS11",start=start,end=end)
        if kp.empty or len(kp)<5: return {"error":"데이터 부족"}
        ma5=kp.rolling(5).mean().iloc[-1]; ma20=kp.rolling(20).mean().iloc[-1] if len(kp)>=20 else kp.mean()
        kp_osc=(ma5/ma20-1)*100; results={"kospi_osc":round(kp_osc,2),"sectors":{}}
        for code,name in SECTOR_ETFS.items():
            try:
                cl=_close(f"{code}.KS",start=start,end=end)
                if cl.empty or len(cl)<5: continue
                m5=cl.rolling(5).mean().iloc[-1]; m20=cl.rolling(20).mean().iloc[-1] if len(cl)>=20 else cl.mean()
                results["sectors"][name]={"rel_osc":round((m5/m20-1)*100-kp_osc,2)}
            except: continue
        if results["sectors"]:
            ss=sorted(results["sectors"].items(),key=lambda x:x[1]["rel_osc"],reverse=True)
            results["strong"]=[(n,d["rel_osc"]) for n,d in ss[:3]]; results["weak"]=[(n,d["rel_osc"]) for n,d in ss[-3:]]
        return results
    except Exception as e: return {"error":str(e)}

@st.cache_data(ttl=timedelta(minutes=30))
def get_binzip_stocks(supply_data=None, top_n=5):
    try:
        lead=[n for n,_ in supply_data["strong"]][:2] if supply_data and "error" not in supply_data and supply_data.get("strong") else ["반도체","정보기술"]
        stocks=[]; seen=set()
        for sn in lead:
            for code,name in _FALLBACK.get(sn,[]):
                if code not in seen: seen.add(code); stocks.append({"code":code,"name":name})
        if not stocks: return {"error":"종목 없음","binzip":[],"sectors":lead}
        start=_td_back(65); end=datetime.today().strftime("%Y-%m-%d")
        kp_rs60=kp_rs20=0.0
        try:
            kp=_close("^KS11",start=start,end=end)
            if len(kp)>=21: kp_rs60=(kp.iloc[-1]/kp.iloc[0]-1)*100; kp_rs20=(kp.iloc[-1]/kp.iloc[-21]-1)*100
        except: pass
        cands=[]
        for s in stocks:
            try:
                cl=_close(f"{s['code']}.KS",start=start,end=end)
                if cl.empty or len(cl)<20: continue
                n=len(cl); now=float(cl.iloc[-1]); ma60=float(cl.rolling(min(60,n)).mean().iloc[-1])
                rs60=(now/float(cl.iloc[0])-1)*100-kp_rs60
                rel20=((now/float(cl.iloc[-21])-1)*100 if n>=21 else 0.0)-kp_rs20
                if -20.0<rel20<-2.0 and now>ma60*0.93 and rs60>5.0:
                    cands.append({"name":s["name"],"code":s["code"],"rs60":round(rs60,1),"rel20":round(rel20,1),"price":int(now)})
            except: continue
        cands.sort(key=lambda x:x["rs60"],reverse=True)
        return {"binzip":cands[:top_n],"scanned":len(stocks),"sectors":lead}
    except Exception as e: return {"error":str(e),"binzip":[]}

@st.cache_data(ttl=timedelta(hours=1))
def get_kr_etf_rs():
    try:
        tks=list(KOREA_ETFS.keys())
        raw=yf.download(tks+["^KS11"],period="1y",auto_adjust=True,progress=False)
        if raw.empty: return {"error":"데이터 없음"}
        data=raw["Close"] if isinstance(raw.columns,pd.MultiIndex) else raw
        if "^KS11" not in data.columns: return {"error":"KOSPI 없음"}
        kospi=data["^KS11"].dropna(); results=[]
        for t in tks:
            if t not in data.columns: continue
            etf=data[t].dropna(); common=etf.index.intersection(kospi.index)
            if len(common)<57: continue
            rel=etf.loc[common]/kospi.loc[common]
            ma=rel.rolling(52,min_periods=52).mean()
            rs=((rel/ma)-1)*100
            rs_val=rs.dropna().iloc[-1] if not rs.dropna().empty else np.nan
            if np.isnan(rs_val): continue
            norm_rs=round(100*(1/(1+np.exp(-float(rs_val)/12))),1)
            results.append({"ticker":t,"name":KOREA_ETFS.get(t,t),"norm_rs":norm_rs,"rs_raw":round(float(rs_val),1)})
        results.sort(key=lambda x:x["norm_rs"],reverse=True)
        strong=[r for r in results if r["norm_rs"]>=70]
        return {"all":results,"strong":strong[:10] if strong else results[:5]}
    except Exception as e: return {"error":str(e)}

@st.cache_data(ttl=3600, show_spinner=False)
def get_buy_timing(ticker):
    try:
        t=ticker.strip().upper()
        raw=yf.download(t,period="1y",auto_adjust=True,progress=False)
        if raw.empty: return {"error":f"{t} 데이터 없음"}
        df=raw.copy()
        if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.droplevel(1)
        df['저가종가괴리']=(df['Low']-df['Close'].shift(1))/df['Close'].shift(1)*100
        df['고가종가하락']=(df['Close']-df['High'])/df['High']*100
        df['시가저가괴리']=(df['Low']-df['Open'])/df['Open']*100
        df['전일종가고가']=(df['High']-df['Close'].shift(1))/df['Close'].shift(1)*100
        df['시가고가괴리']=(df['High']-df['Open'])/df['Open']*100
        info=yf.Ticker(t).info; name=info.get('longName') or info.get('shortName',t)
        return {"ticker":t,"name":name,"price":round(float(df['Close'].iloc[-1]),2),
                "고가종가하락":round(float(df['고가종가하락'].mean()),2),
                "저가종가괴리":round(float(df['저가종가괴리'].dropna().mean()),2),
                "시가저가괴리":round(float(df['시가저가괴리'].mean()),2),
                "전일종가고가":round(float(df['전일종가고가'].dropna().mean()),2),
                "시가고가괴리":round(float(df['시가고가괴리'].mean()),2)}
    except Exception as e: return {"error":str(e)}

# ─────────────────────────────────────────────────────────────────────────────
# 한국 F&G 자동 계산 (Yahoo Finance만 사용)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def get_kr_fg_auto():
    """
    Excel 없이 완전 자동 한국 피어앤그리드 오실레이터
    - KOSPI / KOSDAQ : ^KS11, ^KQ11
    - VKOSPI 대체    : KOSPI 20일 실현변동성(annualized)
    - 채권 스프레드  : 148070.KS(10Y ETF) / 365780.KS(5Y ETF) 상대변화
    - 위험선호 대체  : KOSDAQ/KOSPI 상대강도 (P/C 대리)
    """
    try:
        tks = ["^KS11","^KQ11","148070.KS","365780.KS"]
        raw = yf.download(tks, period="3y", auto_adjust=True, progress=False)
        if raw.empty: return {"error":"데이터 없음"}
        px = raw["Close"].copy()
        if isinstance(px.columns, pd.MultiIndex): px.columns = px.columns.get_level_values(0)

        needed = ["^KS11","^KQ11","148070.KS","365780.KS"]
        for c in needed:
            if c not in px.columns: return {"error":f"{c} 데이터 없음"}
        px = px[needed].dropna(how="all").ffill()

        def _rsi10(s): return _rsi(s, 10)

        results = {}
        for idx_col, label in [("^KS11","KOSPI"),("^KQ11","KOSDAQ")]:
            df = pd.DataFrame(index=px.index)
            s = px[idx_col]

            # 1) Momentum vs MA125
            df["MA125"] = s.rolling(125).mean()
            df["Momentum"] = (s - df["MA125"]) / df["MA125"] * 100

            # 2) RSI(10)
            df["RSI_10"] = _rsi10(s)

            # 3) 실현변동성 (VKOSPI 대체, 높을수록 공포 → 오실레이터에서 반전)
            df["RealVol"] = s.pct_change().rolling(20).std() * (252**0.5) * 100

            # 4) 채권 스프레드 (10Y ETF - 5Y ETF 수익률 차이)
            df["Bond10Y"] = px["148070.KS"].pct_change(20).fillna(0)
            df["Bond5Y"]  = px["365780.KS"].pct_change(20).fillna(0)
            df["BondSpread"] = df["Bond10Y"] - df["Bond5Y"]

            # 5) 위험선호 (KOSDAQ/KOSPI 상대강도 — P/C 대리)
            rel = px["^KQ11"] / px["^KS11"]
            df["RiskApp"] = rel.pct_change(20).fillna(0)

            feats = ["Momentum","RSI_10","RealVol","BondSpread","RiskApp"]
            valid = df.dropna(subset=feats).index
            if len(valid) < 60: continue

            df.loc[valid, feats] = MinMaxScaler().fit_transform(df.loc[valid, feats])
            # RealVol은 높을수록 공포 → 반전
            df.loc[valid, "FGI"] = (
                df.loc[valid,"Momentum"]    * 0.20 +
                df.loc[valid,"RSI_10"]      * 0.20 +
                (1 - df.loc[valid,"RealVol"]) * 0.20 +
                df.loc[valid,"BondSpread"]  * 0.20 +
                df.loc[valid,"RiskApp"]     * 0.20
            )
            ema12 = df["FGI"].ewm(span=12,adjust=False).mean()
            ema26 = df["FGI"].ewm(span=26,adjust=False).mean()
            macd  = ema12 - ema26
            df["Oscillator"] = macd - macd.ewm(span=9,adjust=False).mean()

            recent = df.dropna(subset=["Oscillator"])
            if recent.empty: continue
            osc = round(float(recent["Oscillator"].iloc[-1]), 4)

            # 임펄스
            ema13 = s.ewm(span=13,adjust=False).mean()
            mh    = _macd_hist(s)
            if len(ema13.dropna()) >= 2:
                eu = ema13.iloc[-1] > ema13.iloc[-2]
                mu = mh.iloc[-1] > mh.iloc[-2]
                imp = "🟢 강세" if eu and mu else ("🔴 약세" if not eu and not mu else "🔵 중립")
            else:
                imp = "알수없음"

            # TD Setup
            p = s.values; sc = np.zeros(len(p)); bc = np.zeros(len(p))
            for i in range(len(p)):
                sc[i] = sc[i-1]+1 if i>=4 and p[i]>p[i-4] else 0
                bc[i] = bc[i-1]+1 if i>=2 and p[i]<p[i-2] else 0

            # chart series (최근 6개월)
            cutoff = recent.index[-1] - pd.DateOffset(months=6)
            ch = recent[recent.index >= cutoff]
            chart_series = {
                "dates": [str(d.date()) for d in ch.index],
                "osc": ch["Oscillator"].tolist(),
                "price": s.reindex(ch.index).tolist(),
            }
            results[label] = {
                "osc": osc,
                "sentiment": "탐욕" if osc > 0 else "공포",
                "impulse": imp,
                "td_sell": int(sc[-1]), "td_buy": int(bc[-1]),
                "chart": chart_series,
            }

        if not results: return {"error":"계산 실패"}
        date_str = str(px.index[-1].date())
        return {"date": date_str, "results": results,
                "source": "Yahoo Finance (실현변동성·채권ETF·KOSDAQ상대강도 사용)"}
    except Exception as e: return {"error": str(e)}

# ─────────────────────────────────────────────────────────────────────────────
# 신규 함수 - 한국 개별종목 RS / 주간 추세판별기 / 거래대금 강도 / 한국 F&G / 컨센 가속
# ─────────────────────────────────────────────────────────────────────────────
KR_STOCKS = {k: v for k, v in TICKER_NAMES.items() if k.endswith(".KS") or k.endswith(".KQ")}

# 종목 → 섹터 매핑 (섹터 ETF RS로 강한 섹터 필터링에 사용)
STOCK_SECTOR = {
    # 반도체 (KODEX 반도체 091160.KS)
    "005930.KS":"반도체","000660.KS":"반도체","009150.KS":"반도체","011070.KS":"반도체",
    "042700.KQ":"반도체","240810.KQ":"반도체","039030.KQ":"반도체","089030.KQ":"반도체",
    "095340.KQ":"반도체","319660.KQ":"반도체","085660.KQ":"반도체","064760.KQ":"반도체",
    "084370.KQ":"반도체","036710.KQ":"반도체","183300.KQ":"반도체","008060.KQ":"반도체",
    "140860.KQ":"반도체","102710.KQ":"반도체","195870.KQ":"반도체","067310.KQ":"반도체",
    "007810.KS":"반도체","357780.KQ":"반도체","218410.KQ":"반도체","211050.KQ":"반도체",
    "076410.KQ":"반도체","058470.KQ":"반도체",
    # 방산 (TIGER 우주방산 463250.KS / PLUS K방산 449450.KS)
    "012450.KS":"방산","047810.KS":"방산","079550.KS":"방산",
    "082740.KQ":"방산","077970.KQ":"방산","429530.KQ":"방산",
    # 조선/중공업 (SOL 조선TOP3 466920.KS / TIGER 200중공업 139230.KS)
    "009540.KS":"조선","329180.KS":"조선","010140.KS":"조선","042660.KS":"조선",
    "267260.KS":"조선","298040.KS":"조선","010120.KS":"조선","028050.KS":"조선",
    "267250.KS":"조선",
    # 2차전지 (TIGER 2차전지테마 305540.KS)
    "373220.KS":"2차전지","006400.KS":"2차전지","051910.KS":"2차전지",
    "003670.KS":"2차전지","009830.KS":"2차전지","011790.KS":"2차전지",
    "086520.KQ":"2차전지","247540.KQ":"2차전지","066970.KQ":"2차전지","278280.KQ":"2차전지",
    # 바이오 (KODEX 바이오 244580.KS / TIGER 바이오TOP10 364970.KS)
    "207940.KS":"바이오","068270.KS":"바이오","128940.KS":"바이오",
    "326030.KS":"바이오","145020.KS":"바이오",
    "028300.KQ":"바이오","141080.KQ":"바이오","009420.KQ":"바이오",
    "298380.KQ":"바이오","039200.KQ":"바이오","214450.KQ":"바이오",
    "226950.KQ":"바이오","115180.KQ":"바이오","195940.KQ":"바이오",
    "067630.KQ":"바이오","950200.KQ":"바이오","108490.KQ":"바이오",
    "000100.KS":"바이오",
    # K뷰티/소비재 (HANARO K뷰티 479850.KS)
    "090430.KS":"K뷰티","241710.KQ":"K뷰티","476000.KQ":"K뷰티",
    "278470.KQ":"K뷰티","214150.KQ":"K뷰티","161890.KS":"K뷰티","004370.KS":"K뷰티",
    "097950.KS":"K뷰티",
    # 로봇 (KODEX K-로봇 445290.KS)
    "277810.KS":"로봇","454910.KS":"로봇","090360.KS":"로봇","108490.KQ":"로봇",
    # 자동차 (KODEX 자동차 091180.KS)
    "005380.KS":"자동차","000270.KS":"자동차","012330.KS":"자동차",
    "066570.KS":"자동차","011210.KS":"자동차","086280.KS":"자동차",
    # 원전/에너지 (ACE 원자력테마 433500.KS)
    "034020.KS":"원전","015760.KS":"원전","036460.KS":"원전",
    # 게임/엔터 (KODEX 게임산업 300950.KS)
    "036570.KS":"게임/엔터","259960.KS":"게임/엔터","251270.KS":"게임/엔터",
    "352820.KS":"게임/엔터","263750.KQ":"게임/엔터","293490.KQ":"게임/엔터",
    "035900.KQ":"게임/엔터","041510.KQ":"게임/엔터","122870.KQ":"게임/엔터",
    # 금융 (KODEX 은행 091170.KS)
    "105560.KS":"금융","055550.KS":"금융","086790.KS":"금융",
    "316140.KS":"금융","138040.KS":"금융","032830.KS":"금융","323410.KS":"금융",
    # 통신 (기타)
    "017670.KS":"통신","030200.KS":"통신","032640.KS":"통신",
    # IT/플랫폼
    "035420.KS":"IT플랫폼","035720.KS":"IT플랫폼","012510.KQ":"IT플랫폼",
    "030190.KQ":"IT플랫폼",
}

# 섹터 → 대표 ETF ticker (RS 계산용)
SECTOR_ETF_MAP = {
    "반도체":  "091160.KS",   # KODEX 반도체
    "방산":    "463250.KS",   # TIGER 우주방산
    "조선":    "466920.KS",   # SOL 조선TOP3
    "2차전지": "305540.KS",   # TIGER 2차전지테마
    "바이오":  "244580.KS",   # KODEX 바이오
    "K뷰티":   "479850.KS",   # HANARO K뷰티
    "로봇":    "445290.KS",   # KODEX K-로봇
    "자동차":  "091180.KS",   # KODEX 자동차
    "원전":    "433500.KS",   # ACE 원자력테마
    "게임/엔터":"300950.KS",   # KODEX 게임산업
    "금융":    "091170.KS",   # KODEX 은행
    "통신":    "017670.KS",   # SK텔레콤 (대표 종목으로 대체)
    "IT플랫폼":"035420.KS",   # NAVER (대표 종목으로 대체)
}

# 주요 한국 ETF (KRX 코드 → 이름)  yfinance ticker = code + ".KS"
KR_ETF_CODES = {
    "069500":"KODEX 200",           "102110":"TIGER 200",
    "122630":"KODEX 레버리지",       "114800":"KODEX 인버스",
    "229200":"KODEX 코스닥150",      "233740":"KODEX 코스닥150레버리지",
    "251340":"KODEX 코스닥150인버스",
    "379800":"KODEX 미국S&P500",    "360750":"TIGER 미국S&P500",
    "133690":"TIGER 미국나스닥100",  "292190":"KODEX 미국나스닥100",
    "367380":"KBSTAR 미국나스닥100", "411540":"TIGER 미국나스닥100레버리지",
    "261240":"KODEX 미국채울트라30년","308620":"TIGER 미국채30년",
    "143850":"KODEX 국고채3년",      "148070":"KODEX 국고채10년",
    "091160":"KODEX 반도체",         "139220":"KODEX IT",
    "091170":"KODEX 은행",           "102960":"KODEX 미디어엔터",
    "117700":"KODEX 건설",           "117480":"KODEX 자동차",
    "357870":"KODEX 2차전지산업",    "381180":"KODEX K-방산",
    "394400":"KODEX 원자력",         "305720":"KODEX 2차전지핵심소재",
    "381170":"TIGER K방산우주",      "364980":"TIGER 글로벌반도체SOX",
    "337150":"TIGER 미국테크TOP10",  "325010":"TIGER 원자력테마",
    "423160":"KODEX 방산항공우주",   "437080":"KODEX 미국방산항공우주",
    "441680":"TIGER 한국형글로벌방산","462970":"KODEX AI반도체핵심장비",
    "449450":"TIGER 글로벌AI",       "266360":"KODEX 골드선물",
    "261220":"KODEX WTI원유선물",    "411060":"ACE KRX금현물",
    "182480":"TIGER 부동산인프라고배당","195980":"KODEX 베트남VN30",
    "278530":"TIGER 차이나전기차",   "102780":"KODEX 배당성장",
    "152100":"KODEX 선진국MSCI",     "195930":"TIGER 해외리츠부동산",
}

@st.cache_data(ttl=3600, show_spinner=False)
def get_kr_stock_rs(top_n=15):
    try:
        tks = list(KR_STOCKS.keys())
        all_results = []
        for i in range(0, len(tks), 40):
            batch = tks[i:i+40]
            raw = yf.download(batch+["^KS11"], period="1y", auto_adjust=True, progress=False)
            if raw.empty: continue
            data = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
            if "^KS11" not in data.columns: continue
            kospi = data["^KS11"].dropna()
            for t in batch:
                if t not in data.columns: continue
                stock = data[t].dropna(); common = stock.index.intersection(kospi.index)
                if len(common) < 52: continue
                rel = stock.loc[common] / kospi.loc[common]; ma52 = rel.rolling(52).mean()
                rs_s = ((rel/ma52)-1).dropna()
                if rs_s.empty: continue
                rs_raw = float(rs_s.iloc[-1]*100)
                norm_rs = round(100*(1/(1+np.exp(-rs_raw/12))), 1)
                all_results.append({"ticker":t,"name":KR_STOCKS.get(t,t),"norm_rs":norm_rs,"rs_raw":round(rs_raw,1)})
            time.sleep(0.2)
        if not all_results: return {"error":"RS 계산 실패"}
        all_results.sort(key=lambda x: x["norm_rs"], reverse=True)
        strong = [r for r in all_results if r["norm_rs"]>=70]
        return {"all":all_results,"strong":strong[:top_n] if strong else all_results[:top_n]}
    except Exception as e: return {"error":str(e)}

def calc_kr_stock_rs_excel(df_close, top_n=15):
    """종가 시트 → Mansfield RS: rel = stock/KOSPI, RS = (rel/MA52 - 1) × 100"""
    try:
        date_col = df_close.columns[0]
        df = df_close.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)
        bench_col = next((c for c in df.columns if '코스피' in str(c)), None)
        if bench_col is None:
            return {"error": "코스피 벤치마크 컬럼 없음"}
        kospi = pd.to_numeric(df[bench_col], errors='coerce').where(lambda x: x > 0, np.nan)
        name_to_ticker = {v: k for k, v in KR_STOCKS.items()}
        all_results = []
        for col in df.columns:
            if col == bench_col:
                continue
            s = pd.to_numeric(df[col], errors='coerce').where(lambda x: x > 0, np.nan)
            idx = s.dropna().index.intersection(kospi.dropna().index)
            if len(idx) < 57:
                continue
            sc = s.loc[idx]; kc = kospi.loc[idx]
            rel = sc / kc
            ma = rel.rolling(52, min_periods=52).mean()
            rs = ((rel / ma) - 1) * 100
            rs_val = rs.dropna().iloc[-1] if not rs.dropna().empty else np.nan
            if np.isnan(rs_val):
                continue
            norm_rs = round(100 / (1 + np.exp(-float(rs_val) / 12)), 1)
            ticker = name_to_ticker.get(str(col), "")
            all_results.append({"ticker": ticker, "name": str(col),
                                 "norm_rs": norm_rs, "rs_raw": round(float(rs_val), 1)})
        if not all_results:
            return {"error": "RS 계산 결과 없음"}
        all_results.sort(key=lambda x: x["norm_rs"], reverse=True)
        strong = [r for r in all_results if r["norm_rs"] >= 70]
        return {"all": all_results, "strong": strong[:top_n] if strong else all_results[:top_n]}
    except Exception as e:
        return {"error": str(e)}

def calc_kr_etf_rs_excel(df_data, top_n=15):
    """ETF 데이터 시트 → Mansfield RS: rel = ETF/KOSPI, RS = (rel/MA52 - 1) × 100"""
    try:
        date_col = df_data.columns[0]
        df = df_data.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)
        bench_col = next((c for c in df.columns if '코스피' in str(c)), None)
        if bench_col is None:
            return {"error": "코스피 벤치마크 컬럼 없음"}
        kospi = pd.to_numeric(df[bench_col], errors='coerce').where(lambda x: x > 0, np.nan)
        all_results = []
        for col in df.columns:
            if col == bench_col:
                continue
            s = pd.to_numeric(df[col], errors='coerce').where(lambda x: x > 0, np.nan)
            idx = s.dropna().index.intersection(kospi.dropna().index)
            if len(idx) < 57:
                continue
            sc = s.loc[idx]; kc = kospi.loc[idx]
            rel = sc / kc
            ma = rel.rolling(52, min_periods=52).mean()
            rs = ((rel / ma) - 1) * 100
            rs_val = rs.dropna().iloc[-1] if not rs.dropna().empty else np.nan
            if np.isnan(rs_val):
                continue
            norm_rs = round(100 / (1 + np.exp(-float(rs_val) / 12)), 1)
            all_results.append({"name": str(col), "norm_rs": norm_rs, "rs_raw": round(float(rs_val), 1)})
        if not all_results:
            return {"error": "ETF RS 계산 결과 없음"}
        all_results.sort(key=lambda x: x["norm_rs"], reverse=True)
        strong = [r for r in all_results if r["norm_rs"] >= 70]
        return {"all": all_results, "strong": strong[:top_n] if strong else all_results[:top_n]}
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(ttl=timedelta(hours=4))
def get_kr_stock_rs_auto(top_n=15):
    """yfinance 배치 다운로드 + Mansfield RS: rel=stock/KOSPI, RS=(rel/MA52-1)×100"""
    import datetime as _dt
    end   = _dt.date.today()
    start = end - _dt.timedelta(days=520)  # 2년치 → Excel 데이터와 MA52 안정화
    tks   = list(KR_STOCKS.keys())
    all_prices = {}
    try:
        for i in range(0, len(tks), 40):
            batch = tks[i:i+40]
            raw = yf.download(batch + ["^KS11"], start=start, end=end,
                              auto_adjust=True, progress=False)
            if raw.empty:
                continue
            data = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
            if "^KS11" in data.columns:
                all_prices["코스피"] = data["^KS11"]
            for t in batch:
                if t in data.columns:
                    all_prices[KR_STOCKS.get(t, t)] = data[t]
            time.sleep(0.3)
        if "코스피" not in all_prices or len(all_prices) < 3:
            return {"error": "데이터 수집 실패 (yfinance)"}
        df_close = pd.DataFrame(all_prices)
        df_close.index.name = "Date"
        df_close = df_close.reset_index()
        result = calc_kr_stock_rs_excel(df_close, top_n=top_n)
        if "error" not in result:
            result["source"] = f"📡 yfinance 자동 ({len(all_prices)-1}종목)"
        return result
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(ttl=timedelta(hours=4))
def get_kr_etf_rs_auto(top_n=15):
    """yfinance 배치 다운로드 + Mansfield RS: rel=ETF/KOSPI, RS=(rel/MA52-1)×100"""
    import datetime as _dt
    end   = _dt.date.today()
    start = end - _dt.timedelta(days=520)  # 2년치 → Excel 데이터와 MA52 안정화
    codes = list(KR_ETF_CODES.keys())
    tickers_yf = [c + ".KS" for c in codes]
    all_prices = {}
    try:
        for i in range(0, len(tickers_yf), 40):
            batch = tickers_yf[i:i+40]
            raw = yf.download(batch + ["^KS11"], start=start, end=end,
                              auto_adjust=True, progress=False)
            if raw.empty:
                continue
            data = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
            if "^KS11" in data.columns:
                all_prices["코스피"] = data["^KS11"]
            for yf_t in batch:
                code = yf_t.replace(".KS", "")
                if yf_t in data.columns:
                    all_prices[KR_ETF_CODES.get(code, yf_t)] = data[yf_t]
            time.sleep(0.3)
        if "코스피" not in all_prices or len(all_prices) < 5:
            return {"error": "ETF 데이터 수집 실패"}
        df_etf = pd.DataFrame(all_prices)
        df_etf.index.name = "Date"
        df_etf = df_etf.reset_index()
        result = calc_kr_etf_rs_excel(df_etf, top_n=top_n)
        if "error" not in result:
            result["source"] = f"📡 yfinance 자동 ({len(all_prices)-1}개 ETF)"
        return result
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(ttl=timedelta(hours=2))
def get_composite_score(top_n=30):
    """종합 점수: RS(35%) + 수급(35%) + 거래대금강도(30%) → 상위 종목 선별"""
    import datetime as _dt
    import concurrent.futures
    from bs4 import BeautifulSoup as _BS

    today = _dt.date.today()
    end   = today
    start = end - _dt.timedelta(days=520)  # 2년치 → MA52 안정화

    # ── 0. 섹터 ETF RS로 강한 섹터 선별 ──
    sector_etf_tks = list(set(SECTOR_ETF_MAP.values()))
    strong_sectors = set()
    try:
        _etf_raw = yf.download(sector_etf_tks + ["^KS11"], start=start, end=end,
                               auto_adjust=True, progress=False)
        if not _etf_raw.empty:
            _ec = _etf_raw["Close"] if "Close" in _etf_raw.columns else _etf_raw
            _ek = _ec["^KS11"].dropna()
            for sector, etf_tk in SECTOR_ETF_MAP.items():
                if etf_tk not in _ec.columns: continue
                _es = _ec[etf_tk].dropna()
                _idx = _es.index.intersection(_ek.index)
                if len(_idx) < 57: continue
                _rel = _es.loc[_idx] / _ek.loc[_idx]
                _ma  = _rel.rolling(52, min_periods=52).mean()
                _rv  = ((_rel / _ma) - 1) * 100
                _rv  = _rv.dropna()
                if _rv.empty: continue
                _rs_val = float(_rv.iloc[-1])
                if _rs_val > 0:   # KOSPI 대비 초과수익 = 강한 섹터
                    strong_sectors.add(sector)
    except Exception:
        pass

    # 강한 섹터 종목만 스크리닝 (매핑 없는 종목은 제외)
    if strong_sectors:
        tks = [t for t in KR_STOCKS if STOCK_SECTOR.get(t, "") in strong_sectors]
    else:
        tks = list(KR_STOCKS.keys())   # ETF 데이터 실패 시 전체

    # ── 1. yfinance 배치 (가격 + 거래대금) ──
    all_close = {}; all_turnover = {}
    for i in range(0, len(tks), 50):
        batch = tks[i:i+50]
        try:
            raw = yf.download(batch + ["^KS11"], start=start, end=end,
                              auto_adjust=True, progress=False)
            if raw.empty: continue
            if isinstance(raw.columns, pd.MultiIndex):
                lv0 = raw.columns.get_level_values(0).unique().tolist()
                if "Close" in lv0:
                    close_df = raw["Close"]; vol_df = raw["Volume"]
                else:
                    close_df = pd.DataFrame({t: raw[t]["Close"] for t in batch + ["^KS11"] if t in lv0})
                    vol_df   = pd.DataFrame({t: raw[t]["Volume"] for t in batch + ["^KS11"] if t in lv0})
            else:
                close_df = raw; vol_df = raw
            if "^KS11" in close_df.columns:
                all_close["^KS11"] = close_df["^KS11"]
            for t in batch:
                if t in close_df.columns:
                    all_close[t] = close_df[t]
                    if t in vol_df.columns:
                        all_turnover[t] = vol_df[t] * close_df[t]
        except Exception:
            pass
        time.sleep(0.3)

    if "^KS11" not in all_close or len(all_close) < 5:
        return {"error": "가격 데이터 수집 실패 (yfinance)"}

    kospi = all_close["^KS11"].where(lambda x: x > 0, np.nan)

    # ── 2. RS 계산 (Mansfield, 0-100 정규화) ──
    rs_scores = {}
    for t in tks:
        if t not in all_close: continue
        s   = all_close[t].where(lambda x: x > 0, np.nan)
        idx = s.dropna().index.intersection(kospi.dropna().index)
        if len(idx) < 57: continue
        rel = s.loc[idx] / kospi.loc[idx]
        ma  = rel.rolling(52, min_periods=52).mean()
        rs  = ((rel / ma) - 1) * 100
        rv  = rs.dropna()
        if rv.empty: continue
        rs_scores[t] = round(100 / (1 + np.exp(-float(rv.iloc[-1]) / 12)), 1)

    # ── 3. 거래대금 강도 (최근 5일 vs 52일 평균) ──
    vol_raw = {}
    for t in tks:
        if t not in all_turnover: continue
        s = all_turnover[t].replace(0, np.nan).dropna()
        if len(s) < 57: continue
        m5  = float(s.tail(5).mean())
        m52 = float(s.rolling(52, min_periods=52).mean().dropna().iloc[-1])
        if m52 > 0: vol_raw[t] = m5 / m52
    vol_scores = {}
    if vol_raw:
        vol_scores = (pd.Series(vol_raw).rank(pct=True) * 100).round(1).to_dict()

    # ── 4. 모멘텀 (20일·60일 수익률 합산, 가격 데이터 재활용) ──
    mom_raw = {}
    for t in tks:
        if t not in all_close: continue
        s = all_close[t].where(lambda x: x > 0, np.nan).dropna()
        if len(s) < 62: continue
        r20  = float(s.iloc[-1] / s.iloc[-21] - 1) * 100
        r60  = float(s.iloc[-1] / s.iloc[-61] - 1) * 100
        mom_raw[t] = r20 * 0.4 + r60 * 0.6   # 단기보다 중기에 더 가중
    mom_scores = {}
    if mom_raw:
        mom_scores = (pd.Series(mom_raw).rank(pct=True) * 100).round(1).to_dict()

    # ── 5. 52주 신고가 근접도 (현재가 / 52주 최고가) ──
    high52_raw = {}
    for t in tks:
        if t not in all_close: continue
        s = all_close[t].where(lambda x: x > 0, np.nan).dropna()
        if len(s) < 60: continue
        high52 = float(s.rolling(min(252, len(s))).max().iloc[-1])
        if high52 > 0:
            high52_raw[t] = float(s.iloc[-1]) / high52 * 100   # 100 = 신고가
    high52_scores = {}
    if high52_raw:
        high52_scores = (pd.Series(high52_raw).rank(pct=True) * 100).round(1).to_dict()

    # ── 6. 수급 — 네이버 파이낸스 기관+외국인 40일치 수집 (빈집 감지용) ──
    _hdrs = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://finance.naver.com/",
    }

    def _fetch_supply(ticker):
        """40일치 기관+외국인 순매수 수집 → (short5, long40) 반환"""
        code = ticker.replace(".KS", "").replace(".KQ", "")
        daily = []
        try:
            for pg in range(1, 3):
                r = requests.get(
                    f"https://finance.naver.com/item/frgn.naver?code={code}&page={pg}",
                    headers=_hdrs, timeout=10)
                if r.status_code != 200: break
                soup = _BS(r.content, "html.parser", from_encoding="euc-kr")
                tables = soup.find_all("table")
                if len(tables) < 4: break
                for row in tables[3].find_all("tr"):
                    cells = [c.get_text(strip=True) for c in row.find_all("td")]
                    if len(cells) >= 7 and cells[0] and "." in cells[0]:
                        try:
                            cp  = int(cells[1].replace(",", ""))
                            iq  = int(cells[5].replace(",", "").replace("+", "") or "0")
                            fq  = int(cells[6].replace(",", "").replace("+", "") or "0")
                            if cp > 0:
                                daily.append((iq + fq) * cp)   # 순매수대금 (음수=순매도)
                                if len(daily) >= 40: break
                        except Exception:
                            pass
                if len(daily) >= 40: break
        except Exception:
            pass
        short5 = sum(daily[:5])  if len(daily) >= 5  else 0  # 최근 5일 (전환 감지)
        long40 = sum(daily[:40]) if daily else 0              # 40일 누적 (빈집 수준)
        return ticker, short5, long40

    short5_raw = {}; long40_raw = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for tk, s5, l40 in pool.map(_fetch_supply, tks):
            short5_raw[tk] = s5
            long40_raw[tk] = l40

    # 수급 점수: 40일 누적이 낮을수록(빈집) 오를 여력이 크므로 역순 백분위로 높은 점수
    # ETF가 아직 수량을 확보하지 않은 종목 = 이후 ETF 매수로 오를 가능성 높음
    supply_scores = {}
    if long40_raw and any(v != 0 for v in long40_raw.values()):
        supply_scores = ((1 - pd.Series(long40_raw).rank(pct=True)) * 100).round(1).to_dict()
    elif any(v != 0 for v in short5_raw.values()):
        # short5도 낮을수록(빈집) 좋으므로 역순 백분위 적용
        supply_scores = ((1 - pd.Series(short5_raw).rank(pct=True)) * 100).round(1).to_dict()

    # 빈집 감지: 40일 누적 순매수가 0 이하(절대 기준) + 최근 5일 양수(전환 시작)
    # ※ 상대 백분위(하위 35%) 방식은 섹터 전체가 순매수 구간일 때 양수 종목을 빈집으로
    #   오분류하므로 절대값 기준 사용
    binzip_set = set()
    if long40_raw:
        for tk in tks:
            is_empty  = long40_raw.get(tk, 1) <= 0   # 40일 누적 순매수 ≤ 0 = 진짜 빈집
            is_inflow = short5_raw.get(tk, 0) > 0    # 최근 5일 순매수 양수 = 전환 시작
            if is_empty and is_inflow:
                binzip_set.add(tk)

    # ── 7. 종합 점수 합산 ──
    # RS 25% + 수급(빈집여력) 30% + 모멘텀 20% + 거래대금 15% + 신고가 10%
    WEIGHTS = [("rs", 0.25), ("supply", 0.30), ("momentum", 0.20), ("volume", 0.15), ("high52", 0.10)]
    score_maps = {"rs": rs_scores, "supply": supply_scores,
                  "volume": vol_scores, "momentum": mom_scores, "high52": high52_scores}

    all_tks = set(rs_scores) | set(vol_scores) | set(supply_scores)
    results = []
    for t in all_tks:
        nm = KR_STOCKS.get(t, t)
        vals = {k: score_maps[k].get(t) for k, _ in WEIGHTS}
        parts, wts = [], []
        for k, w in WEIGHTS:
            if vals[k] is not None:
                parts.append(vals[k]); wts.append(w)
        if not parts: continue
        score = round(sum(v * w for v, w in zip(parts, wts)) / sum(wts), 1)

        rs_v  = vals["rs"]  or 0
        sup_v = vals["supply"] or 0
        is_bz = t in binzip_set

        # 빈집전환이 최우선 — 수급 바닥에서 막 들어오는 종목
        if is_bz:
            grade = "🏚️ 빈집"
        elif rs_v >= 65 and sup_v >= 65:
            grade = "⭐ 강력"
        elif rs_v >= 55 and sup_v >= 55:
            grade = "✅ 유망"
        elif rs_v >= 45 or sup_v >= 65:
            grade = "👀 관심"
        else:
            grade = "─"

        sector = STOCK_SECTOR.get(t, "기타")
        results.append({"ticker": t, "name": nm, "score": score, "grade": grade,
                         "sector": sector, "binzip": is_bz,
                         "rs": vals["rs"], "supply": vals["supply"],
                         "volume": vals["volume"], "momentum": vals["momentum"],
                         "high52": vals["high52"]})

    # 빈집전환 종목을 최상단으로, 나머지는 점수순
    results.sort(key=lambda x: (not x["binzip"], -x["score"]))
    return {"results": results[:top_n], "total": len(results),
            "has_supply": bool(supply_scores), "strong_sectors": sorted(strong_sectors),
            "binzip_count": len(binzip_set)}

def calc_weekly_trend_excel(df_wk):
    """추세판별기(주간).xlsx DB 시트로 CMF/임펄스/TD 계산 (오프라인)"""
    try:
        date_col  = df_wk.columns[0]
        df = df_wk.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
        open_col  = next((c for c in df.columns if '시가' in str(c)), None)
        high_col  = next((c for c in df.columns if '고가' in str(c)), None)
        low_col   = next((c for c in df.columns if '저가' in str(c)), None)
        close_col = next((c for c in df.columns if '종가' in str(c)), None)
        vol_col   = next((c for c in df.columns if '거래량' in str(c)), None)
        if not all([open_col, high_col, low_col, close_col, vol_col]):
            return {"error": "OHLCV 컬럼 미감지"}
        for c in [open_col, high_col, low_col, close_col, vol_col]:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna(subset=[close_col, vol_col])
        if len(df) < 15:
            return {"error": "주봉 데이터 부족 (15주 미만)"}
        wk = df.rename(columns={open_col:'Open', high_col:'High', low_col:'Low', close_col:'Close', vol_col:'Volume'})
        wk['Prev_High'] = wk['High'].shift(1)
        wk['Prev_Low']  = wk['Low'].shift(1)
        wk['MA10']      = wk['Close'].rolling(10).mean()
        pr  = wk['High'] - wk['Low']
        mfm = ((wk['Close'] - wk['Low']) - (wk['High'] - wk['Close'])) / pr.replace(0, np.nan)
        wk['CMF'] = (mfm * wk['Volume']).rolling(4).sum() / wk['Volume'].rolling(4).sum()
        buy  = (wk['High'] > wk['Prev_High']) & (wk['Close'] > wk['MA10']) & (wk['CMF'] > 0)
        sell = (wk['Low']  < wk['Prev_Low'])  & (wk['Close'] < wk['MA10']) & (wk['CMF'] < 0)
        ema13 = wk['Close'].ewm(span=13, adjust=False).mean()
        macdh = _macd_hist(wk['Close'])
        if len(ema13.dropna()) >= 2:
            eu = ema13.iloc[-1] > ema13.iloc[-2]
            mu = macdh.iloc[-1] > macdh.iloc[-2]
            impulse_w = "🟢 강세" if eu and mu else ("🔴 약세" if not eu and not mu else "🔵 중립")
        else:
            impulse_w = "알수없음"
        def td_cnt(s):
            p = s.values; sc = np.zeros(len(p)); bc = np.zeros(len(p))
            for i in range(len(p)):
                sc[i] = sc[i-1]+1 if i >= 4 and p[i] > p[i-4] else 0
                bc[i] = bc[i-1]+1 if i >= 2 and p[i] < p[i-2] else 0
            return int(sc[-1]), int(bc[-1])
        w_ts, w_tb = td_cnt(wk['Close'].dropna())
        cur_cmf   = float(wk['CMF'].dropna().iloc[-1])
        cur_close = float(wk['Close'].iloc[-1])
        cur_ma10  = float(wk['MA10'].dropna().iloc[-1]) if not wk['MA10'].dropna().empty else None
        ch = wk.tail(26)
        dates = [str(d.date()) for d in df[date_col].tail(26)]
        chart = {"dates": dates, "close": ch['Close'].tolist(), "ma10": ch['MA10'].tolist(),
                 "cmf": ch['CMF'].tolist(), "high": ch['High'].tolist(), "low": ch['Low'].tolist()}
        return {"price": round(cur_close, 2), "ma10": round(cur_ma10, 2) if cur_ma10 else None,
                "cmf": round(cur_cmf, 4),
                "buy_signal": bool(buy.iloc[-1]) if pd.notna(buy.iloc[-1]) else False,
                "sell_signal": bool(sell.iloc[-1]) if pd.notna(sell.iloc[-1]) else False,
                "recent_buy_4w": int(buy.tail(4).sum()), "impulse_weekly": impulse_w,
                "w_td_sell": w_ts, "w_td_buy": w_tb,
                "date_range": f"{dates[0]} ~ {dates[-1]}", "rows": len(df), "chart": chart}
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(ttl=3600, show_spinner=False)
def get_weekly_trend(ticker):
    try:
        t = ticker.strip().upper()
        raw = yf.download(t, period="4y", auto_adjust=True, progress=False)
        if raw.empty: return {"error":f"{t} 데이터 없음"}
        if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.get_level_values(0)
        df = raw[['Open','High','Low','Close','Volume']].dropna()
        if len(df) < 60: return {"error":"데이터 부족"}
        wk = pd.DataFrame({
            'Open': df['Open'].resample('W-FRI').first(),
            'High': df['High'].resample('W-FRI').max(),
            'Low': df['Low'].resample('W-FRI').min(),
            'Close': df['Close'].resample('W-FRI').last(),
            'Volume': df['Volume'].resample('W-FRI').sum(),
        }).dropna()
        if len(wk) < 15: return {"error":"주봉 데이터 부족"}
        wk['Prev_High'] = wk['High'].shift(1); wk['Prev_Low'] = wk['Low'].shift(1)
        wk['MA10'] = wk['Close'].rolling(10).mean()
        pr = wk['High'] - wk['Low']
        mfm = ((wk['Close']-wk['Low'])-(wk['High']-wk['Close'])) / pr.replace(0,np.nan)
        wk['CMF'] = (mfm*wk['Volume']).rolling(4).sum() / wk['Volume'].rolling(4).sum()
        buy = (wk['High']>wk['Prev_High']) & (wk['Close']>wk['MA10']) & (wk['CMF']>0)
        sell = (wk['Low']<wk['Prev_Low']) & (wk['Close']<wk['MA10']) & (wk['CMF']<0)
        ema13 = wk['Close'].ewm(span=13,adjust=False).mean()
        macdh = _macd_hist(wk['Close'])
        if len(ema13.dropna())>=2:
            eu=ema13.iloc[-1]>ema13.iloc[-2]; mu=macdh.iloc[-1]>macdh.iloc[-2]
            impulse_w="🟢 강세" if eu and mu else ("🔴 약세" if not eu and not mu else "🔵 중립")
        else: impulse_w="알수없음"
        def td_cnt(s):
            p=s.values; sc=np.zeros(len(p)); bc=np.zeros(len(p))
            for i in range(len(p)):
                sc[i]=sc[i-1]+1 if i>=4 and p[i]>p[i-4] else 0
                bc[i]=bc[i-1]+1 if i>=2 and p[i]<p[i-2] else 0
            return int(sc[-1]),int(bc[-1])
        w_ts,w_tb = td_cnt(wk['Close'].dropna())
        d_ts,d_tb = td_cnt(df['Close'].dropna())
        mo = df['Close'].resample('ME').last().dropna(); m_ts,m_tb = td_cnt(mo)
        cur_cmf=float(wk['CMF'].dropna().iloc[-1]); cur_close=float(wk['Close'].iloc[-1])
        cur_ma10 = float(wk['MA10'].dropna().iloc[-1]) if not wk['MA10'].dropna().empty else None
        ch_wk = wk.tail(26)
        chart = {"dates":[str(d.date()) for d in ch_wk.index],
                 "close":ch_wk['Close'].tolist(),"ma10":ch_wk['MA10'].tolist(),
                 "cmf":ch_wk['CMF'].tolist(),"high":ch_wk['High'].tolist(),"low":ch_wk['Low'].tolist()}
        return {"ticker":t,"price":round(cur_close,2),"ma10":round(cur_ma10,2) if cur_ma10 else None,
                "cmf":round(cur_cmf,4),"buy_signal":bool(buy.iloc[-1]),"sell_signal":bool(sell.iloc[-1]),
                "recent_buy_4w":int(buy.tail(4).sum()),"impulse_weekly":impulse_w,
                "w_td_sell":w_ts,"w_td_buy":w_tb,"d_td_sell":d_ts,"d_td_buy":d_tb,"m_td_sell":m_ts,"m_td_buy":m_tb,
                "chart":chart}
    except Exception as e: return {"error":str(e)}

@st.cache_data(ttl=3600, show_spinner=False)
def get_trading_intensity(ticker):
    try:
        t = ticker.strip().upper(); tk = yf.Ticker(t)
        shares = None
        try:
            fi = tk.fast_info
            shares = getattr(fi,'shares_outstanding',None) or getattr(fi,'float_shares',None)
        except: pass
        if not shares:
            try:
                info2 = tk.info
                shares = info2.get('floatShares') or info2.get('sharesOutstanding')
            except: pass
        raw = yf.download(t, period="2y", auto_adjust=False, progress=False)
        if raw.empty: return {"error":f"{t} 데이터 없음"}
        if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.get_level_values(0)
        df = raw[['Open','High','Low','Close','Volume']].dropna()
        VOL_MA=20; ZSCORE_WIN=60; ACC_WIN=7
        df['PrevClose']=df['Close'].shift(1)
        df['VolMA']=df['Volume'].rolling(VOL_MA,min_periods=10).mean()
        df['VolSpike']=df['Volume']/df['VolMA']
        if shares and shares>0:
            df['Turnover']=df['Volume']/shares
        else:
            try:
                mcap=getattr(tk.fast_info,'market_cap',None)
                df['Turnover']=(df['Volume']/(mcap/df['Close'].replace(0,np.nan))) if mcap and mcap>0 else df['Volume']/df['Volume'].rolling(60,min_periods=20).mean()
            except: df['Turnover']=df['Volume']/df['Volume'].rolling(60,min_periods=20).mean()
        df['GapTrend']=(df['Open']/df['PrevClose']-1.0)+(df['Close']/df['Open']-1.0)
        def z(s,w=60):
            mu=s.rolling(w,min_periods=20).mean(); sd=s.rolling(w,min_periods=20).std(ddof=0)
            return (s-mu)/sd.replace(0,np.nan)
        df['TI_raw']=0.5*z(df['VolSpike']).fillna(0)+0.3*z(df['Turnover']).fillna(0)+0.2*z(df['GapTrend']).fillna(0)
        df['TI_acc7']=df['TI_raw'].rolling(ACC_WIN,min_periods=ACC_WIN).sum()
        ti_full=df['TI_acc7'].dropna()
        if ti_full.empty: return {"error":"TI 계산 실패"}
        vmin,vmax=ti_full.min(),ti_full.max()
        ti_norm=pd.Series([50.0],index=ti_full.index[-1:]) if vmin==vmax else (ti_full-vmin)*100.0/(vmax-vmin)
        cur=round(float(ti_norm.iloc[-1]),1)
        ma3=round(float(ti_norm.rolling(3).mean().iloc[-1]),1) if len(ti_norm)>=3 else cur
        sig7=round(float(ti_norm.ewm(span=7,adjust=False).mean().iloc[-1]),1)
        n = ti_norm.tail(120); ma3_s = ti_norm.rolling(3).mean().tail(120); sig7_s = ti_norm.ewm(span=7,adjust=False).mean().tail(120)
        return {"ticker":t,"ti":cur,"ti_ma3":ma3,"ti_signal":sig7,
                "signal_text":"🔴 과열" if cur>=75 else ("🟡 중립" if cur>=40 else "🟢 매집"),
                "chart":{"dates":[str(d.date()) for d in n.index],"ti":n.tolist(),"ti_ma3":ma3_s.tolist(),"ti_signal":sig7_s.tolist()}}
    except Exception as e: return {"error":str(e)}

def calc_kr_fg_excel(df_kospi, df_kosdaq, df_call_oi=None, df_put_oi=None):
    try:
        def ensure_date(df):
            if 'Date' not in df.columns: df=df.rename(columns={df.columns[0]:'Date'})
            df['Date']=pd.to_datetime(df['Date'],errors='coerce')
            return df.dropna(subset=['Date']).copy()
        def calc_rsi_local(df,col,win=10):
            d=df[col].diff(); g=d.where(d>0,0).rolling(win).mean(); l=(-d.where(d<0,0)).rolling(win).mean()
            rs=g/l; df['RSI_10']=100-(100/(1+rs)); return df
        def calc_fg_local(df,pc,vc,cc,ptc,b5,b10,frgn_pc_col=None):
            df=df.copy(); df['MA125']=df[pc].rolling(125).mean()
            df['Momentum']=(df[pc]-df['MA125'])/df['MA125']*100
            if frgn_pc_col and frgn_pc_col in df.columns:
                df['PutCall']=df[frgn_pc_col]
            else:
                df['PutCall']=df[ptc]/df[cc].replace(0,np.nan)
            df['Volatility']=df[vc]; df['BondDiff']=df[b10]-df[b5]
            df.replace([np.inf,-np.inf],np.nan,inplace=True)
            feats=['Momentum','PutCall','Volatility','BondDiff','RSI_10']; valid=df.dropna(subset=feats).index
            if len(valid)==0: return df
            df.loc[valid,feats]=MinMaxScaler().fit_transform(df.loc[valid,feats])
            df.loc[valid,'FGI']=(df.loc[valid,'Momentum']*0.2+(1-df.loc[valid,'PutCall'])*0.2+
                                  (1-df.loc[valid,'Volatility'])*0.2+df.loc[valid,'BondDiff']*0.2+df.loc[valid,'RSI_10']*0.2)
            ema12=df['FGI'].ewm(span=12,adjust=False).mean(); ema26=df['FGI'].ewm(span=26,adjust=False).mean()
            macd=ema12-ema26; df['Oscillator']=macd-macd.ewm(span=9,adjust=False).mean()
            return df
        def get_impulse_local(df,pc):
            e=df[pc].ewm(span=13,adjust=False).mean(); mh=_macd_hist(df[pc])
            if len(e.dropna())<2: return "알수없음"
            eu=e.iloc[-1]>e.iloc[-2]; mu=mh.iloc[-1]>mh.iloc[-2]
            return "🟢 강세" if eu and mu else ("🔴 약세" if not eu and not mu else "🔵 중립")
        def td_local(s):
            p=s.values; sc=np.zeros(len(p)); bc=np.zeros(len(p))
            for i in range(len(p)):
                sc[i]=sc[i-1]+1 if i>=4 and p[i]>p[i-4] else 0
                bc[i]=bc[i-1]+1 if i>=2 and p[i]<p[i-2] else 0
            return int(sc[-1]),int(bc[-1])
        kp=ensure_date(df_kospi); kq=ensure_date(df_kosdaq)
        for c in ['5년 국채선물 추종 지수','10년국채선물지수','코스피 200 변동성지수','코스피','최근월물 CALL ATM','최근월물 PUT ATM']:
            kp[c]=pd.to_numeric(kp[c],errors='coerce')
        for c in ['5년 국채선물 추종 지수','10년국채선물지수','코스피 200 변동성지수','코스닥','최근월물 CALL ATM','최근월물 PUT ATM']:
            kq[c]=pd.to_numeric(kq[c],errors='coerce')
        kp=calc_rsi_local(kp,'코스피'); kq=calc_rsi_local(kq,'코스닥')
        # 콜/풋 미결제 시트 → 외국인 P/C 비율 정밀 계산
        frgn_pc_col = None
        if df_call_oi is not None and df_put_oi is not None:
            try:
                def _prep_oi(df):
                    df = df.copy().rename(columns={df.columns[0]: 'Date'})
                    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                    df = df.dropna(subset=['Date'])
                    fc = next((c for c in df.columns if '외국인' in c), None)
                    if not fc: return pd.Series(dtype=float)
                    s = pd.to_numeric(df[fc], errors='coerce')
                    s.index = df['Date'].values
                    return s
                call_s = _prep_oi(df_call_oi); put_s = _prep_oi(df_put_oi)
                oi_df = pd.DataFrame({'call': call_s, 'put': put_s}).dropna()
                oi_df['Frgn_PC_Ratio'] = oi_df['put'] / oi_df['call'].replace(0, np.nan)
                kp = kp.set_index('Date').join(oi_df[['Frgn_PC_Ratio']], how='left').reset_index()
                if kp['Frgn_PC_Ratio'].notna().sum() > 10: frgn_pc_col = 'Frgn_PC_Ratio'
            except: pass
        kp=calc_fg_local(kp,'코스피','코스피 200 변동성지수','최근월물 CALL ATM','최근월물 PUT ATM','5년 국채선물 추종 지수','10년국채선물지수',frgn_pc_col)
        kq=calc_fg_local(kq,'코스닥','코스피 200 변동성지수','최근월물 CALL ATM','최근월물 PUT ATM','5년 국채선물 추종 지수','10년국채선물지수')
        kp_osc=round(float(kp['Oscillator'].dropna().iloc[-1]),4); kq_osc=round(float(kq['Oscillator'].dropna().iloc[-1]),4)
        kp_ts,kp_tb=td_local(kp['코스피'].dropna()); kq_ts,kq_tb=td_local(kq['코스닥'].dropna())
        # SuperMA and GapPct (KOSPI and KOSDAQ)
        def _super_ma_gap(df, price_col):
            try:
                p = pd.to_numeric(df[price_col], errors='coerce')
                sup = pd.concat([p.rolling(w).mean() for w in [20,60,120,200]], axis=1).mean(axis=1)
                gap = (p - sup) / sup * 100
                return round(float(gap.dropna().iloc[-1]), 2), round(float(sup.dropna().iloc[-1]), 2)
            except: return None, None
        kp_gap, kp_sup = _super_ma_gap(kp, '코스피')
        kq_gap, kq_sup = _super_ma_gap(kq, '코스닥')
        cutoff = kp['Date'].max() - pd.DateOffset(months=6)
        ch_kp = kp[kp['Date']>=cutoff].dropna(subset=['Oscillator','코스피'])
        ch_kq = kq[kq['Date']>=kq['Date'].max()-pd.DateOffset(months=6)].dropna(subset=['Oscillator'])
        chart = {"dates":[str(pd.Timestamp(d).date()) for d in ch_kp['Date'].tolist()],
                 "kospi_osc":ch_kp['Oscillator'].tolist(),"kospi_price":ch_kp['코스피'].tolist(),
                 "kosdaq_dates":[str(pd.Timestamp(d).date()) for d in ch_kq['Date'].tolist()],
                 "kosdaq_osc":ch_kq['Oscillator'].tolist()}
        return {"date":kp['Date'].iloc[-1].strftime('%Y-%m-%d'),
                "kospi_osc":kp_osc,"kosdaq_osc":kq_osc,
                "kospi_sentiment":"탐욕" if kp_osc>0 else "공포","kosdaq_sentiment":"탐욕" if kq_osc>0 else "공포",
                "kospi_impulse":get_impulse_local(kp,'코스피'),"kosdaq_impulse":get_impulse_local(kq,'코스닥'),
                "kospi_td_sell":kp_ts,"kospi_td_buy":kp_tb,"kosdaq_td_sell":kq_ts,"kosdaq_td_buy":kq_tb,
                "kospi_gap":kp_gap,"kosdaq_gap":kq_gap,
                "chart":chart}
    except Exception as e: return {"error":str(e)}

def get_naver_realtime_quote(codes):
    """네이버 실시간 주가 (지연 0초) — 단일 또는 복수 종목코드(리스트/문자열)"""
    try:
        if isinstance(codes, list):
            code_str = ",".join(c.replace(".KS","").replace(".KQ","") for c in codes)
        else:
            code_str = codes.replace(".KS","").replace(".KQ","")
        hdrs = {"User-Agent":"Mozilla/5.0","Referer":"https://finance.naver.com/"}
        url = f"https://polling.finance.naver.com/api/realtime/domestic/stock/{code_str}"
        r = requests.get(url, headers=hdrs, timeout=8)
        if not r.ok: return {"error": f"HTTP {r.status_code}"}
        result = {}
        for d in r.json().get("datas", []):
            code = d.get("itemCode","")
            result[code] = {
                "name":       d.get("stockName",""),
                "price":      d.get("closePriceRaw",""),
                "price_fmt":  d.get("closePrice",""),
                "change":     d.get("compareToPreviousClosePriceRaw",""),
                "change_fmt": d.get("compareToPreviousClosePrice",""),
                "pct":        d.get("fluctuationsRatio",""),
                "direction":  d.get("compareToPreviousPrice",{}).get("text",""),
                "volume":     d.get("accumulatedTradingVolumeRaw",""),
                "high":       d.get("highPriceRaw",""),
                "low":        d.get("lowPriceRaw",""),
                "market_status": d.get("marketStatus",""),
            }
        return result if result else {"error": "데이터 없음"}
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(ttl=timedelta(hours=6))
def get_dart_profile(ticker, dart_key):
    """DART Open API — 분기/반기/사업보고서 핵심 파싱 (사업개요, 주요제품, 수주현황, 매출현황)"""
    import zipfile
    from io import BytesIO
    from bs4 import BeautifulSoup
    import datetime as _dt

    if not dart_key:
        return {"error": "DART_API_KEY 미설정"}

    code = ticker.replace(".KS", "").replace(".KQ", "").zfill(6)
    hdrs = {"User-Agent": "Mozilla/5.0"}
    base = "https://opendart.fss.or.kr/api"

    try:
        # 1. 회사 기본정보 → corp_code
        r1 = requests.get(f"{base}/company.json",
                          params={"crtfc_key": dart_key, "stock_code": code},
                          headers=hdrs, timeout=10)
        cj = r1.json()
        if cj.get("status") != "000":
            return {"error": f"회사 조회 실패: {cj.get('message', '')}"}
        corp_code  = cj["corp_code"]
        corp_name  = cj.get("corp_name", "")
        industry   = cj.get("induty_code", "")
        ceo        = cj.get("ceo_nm", "")
        est_dt     = cj.get("est_dt", "")
        empl_no    = cj.get("empl_no", "")
        address    = cj.get("adres", "")

        # 2. 최신 보고서 목록 (1년치)
        today     = _dt.date.today().strftime("%Y%m%d")
        year_ago  = (_dt.date.today() - _dt.timedelta(days=365)).strftime("%Y%m%d")
        r2 = requests.get(f"{base}/list.json",
                          params={"crtfc_key": dart_key, "corp_code": corp_code,
                                  "bgn_de": year_ago, "end_de": today,
                                  "pblntf_ty": "A", "page_count": 10},
                          headers=hdrs, timeout=10)
        filings = r2.json().get("list", [])

        target = None
        for ptype in ["A003", "A002", "A001"]:
            matches = [f for f in filings if f.get("pblntf_detail_ty") == ptype]
            if matches:
                target = sorted(matches, key=lambda x: x.get("rcept_dt", ""), reverse=True)[0]
                break

        base_result = {
            "corp_name": corp_name, "industry": industry, "ceo": ceo,
            "est_dt": est_dt, "empl_no": empl_no, "address": address,
            "overview": None, "products_html": None,
            "orders_html": None, "sales_html": None, "error": None,
        }
        if not target:
            base_result["report_type"] = "보고서 없음"
            base_result["report_date"] = ""
            return base_result

        rcept_no = target["rcept_no"]
        rpt_map  = {"A003": "분기보고서", "A002": "반기보고서", "A001": "사업보고서"}
        base_result["report_type"] = rpt_map.get(target.get("pblntf_detail_ty", ""), "보고서")
        base_result["report_date"] = target.get("rcept_dt", "")

        # 3. 문서 ZIP 다운로드 + 파싱
        r3 = requests.get(f"{base}/document.json",
                          params={"crtfc_key": dart_key, "rcept_no": rcept_no},
                          headers=hdrs, timeout=40)
        try:
            zf = zipfile.ZipFile(BytesIO(r3.content))
            xml_files = [n for n in zf.namelist()
                         if (n.endswith('.xml') or n.endswith('.html')) and not n.startswith('__')]
            if not xml_files:
                return base_result
            raw = zf.read(xml_files[0])
            soup = BeautifulSoup(raw, "lxml")

            def _section_text(keywords, max_chars=700):
                for kw in keywords:
                    for tag in soup.find_all(string=lambda t: t and kw in t):
                        texts = []
                        for el in tag.parent.find_all_next():
                            if el.name in ('p', 'P', 'div', 'span', 'td', 'li'):
                                txt = el.get_text(" ", strip=True)
                                if txt and len(txt) > 8:
                                    texts.append(txt)
                            if sum(len(s) for s in texts) >= max_chars:
                                break
                        result = " ".join(texts)[:max_chars]
                        if len(result) > 30:
                            return result
                return None

            def _table_html(keywords):
                for kw in keywords:
                    for tag in soup.find_all(string=lambda t: t and kw in t):
                        for el in tag.parent.find_all_next():
                            if el.name == 'table':
                                try:
                                    df = pd.read_html(str(el))[0]
                                    if df.shape[0] >= 1 and df.shape[1] >= 2:
                                        style = (
                                            "style='width:100%;border-collapse:collapse;"
                                            "font-size:0.82rem;color:#e6edf3;'"
                                        )
                                        td_style = "style='padding:4px 8px;border:1px solid #21262d;'"
                                        html = df.head(20).to_html(
                                            index=False, border=0, classes="dart-tbl"
                                        ).replace("<table", f"<table {style}")
                                        return html
                                except Exception:
                                    pass
                return None

            base_result["overview"]      = _section_text(["사업의 개요", "회사의 개요", "사업 개요"])
            base_result["products_html"] = _table_html(["주요 제품", "주요제품", "제품 및 서비스"])
            base_result["orders_html"]   = _table_html(["수주현황", "수주 현황"])
            base_result["sales_html"]    = _table_html(["매출현황", "매출 현황", "매출액 현황"])
        except Exception:
            pass

        return base_result

    except Exception as e:
        return {"error": str(e)}

@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_supply_osc(ticker, chart_days=60, agg_days=20):
    """단일 한국 종목 수급 오실레이터 — 네이버 파이낸스 기관+외국인 순매매"""
    from bs4 import BeautifulSoup
    if not (ticker.endswith(".KS") or ticker.endswith(".KQ")):
        return {"error": "한국 종목만 지원 (.KS/.KQ)"}
    code = ticker.replace(".KS","").replace(".KQ","")
    hdrs = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer":"https://finance.naver.com/"}
    def _pn(s):
        try: return int(s.replace(",","").replace("+",""))
        except: return 0
    try:
        rows_data = []
        pages_needed = (chart_days // 20) + 1
        for pg in range(1, pages_needed + 1):
            r = requests.get(f"https://finance.naver.com/item/frgn.naver?code={code}&page={pg}",
                             headers=hdrs, timeout=12)
            if r.status_code != 200: break
            soup = BeautifulSoup(r.content, "html.parser", from_encoding="euc-kr")
            tables = soup.find_all("table")
            if len(tables) < 4: break
            for row in tables[3].find_all("tr"):
                cells = [c.get_text(strip=True) for c in row.find_all("td")]
                # cells[5]=기관 순매매량, cells[6]=외국인 순매매량
                if len(cells) >= 7 and cells[0] and "." in cells[0]:
                    try:
                        close = _pn(cells[1])
                        inst_qty = _pn(cells[5])   # 기관
                        frgn_qty = _pn(cells[6])   # 외국인
                        if close > 0:
                            rows_data.append({"date":cells[0],"close":close,
                                              "inst_qty":inst_qty,"inst_val":inst_qty*close,
                                              "frgn_qty":frgn_qty,"frgn_val":frgn_qty*close})
                    except: pass
            if len(rows_data) >= chart_days: break
        if not rows_data: return {"error":"데이터 없음"}
        df = pd.DataFrame(rows_data[:chart_days]).iloc[::-1].reset_index(drop=True)
        df["inst_bil"] = df["inst_val"] / 100_000_000
        df["frgn_bil"] = df["frgn_val"] / 100_000_000
        df["inst_ma5"]  = df["inst_bil"].rolling(5, min_periods=1).mean()
        df["frgn_ma5"]  = df["frgn_bil"].rolling(5, min_periods=1).mean()
        df["inst_cum"]  = df["inst_bil"].rolling(agg_days, min_periods=1).sum()
        df["frgn_cum"]  = df["frgn_bil"].rolling(agg_days, min_periods=1).sum()
        return {
            "ticker": ticker, "agg_days": agg_days,
            "inst_agg_bil": round(float(df["inst_bil"].tail(agg_days).sum()), 1),
            "frgn_agg_bil": round(float(df["frgn_bil"].tail(agg_days).sum()), 1),
            "latest_close": int(df["close"].iloc[-1]),
            "chart": {
                "dates":    df["date"].tolist(),
                "price":    df["close"].tolist(),
                "inst_daily": [round(v,1) for v in df["inst_bil"].tolist()],
                "inst_ma5":   [round(v,1) for v in df["inst_ma5"].tolist()],
                "inst_cum":   [round(v,1) for v in df["inst_cum"].tolist()],
                "frgn_daily": [round(v,1) for v in df["frgn_bil"].tolist()],
                "frgn_ma5":   [round(v,1) for v in df["frgn_ma5"].tolist()],
                "frgn_cum":   [round(v,1) for v in df["frgn_cum"].tolist()],
            },
        }
    except Exception as e: return {"error": str(e)}

@st.cache_data(ttl=timedelta(minutes=30))
def get_kr_supply_auto(top_n=20, days=20):
    """네이버 파이낸스 외국인 순매수 자동 스크리닝 + 수급 오실레이터 차트"""
    import concurrent.futures
    from bs4 import BeautifulSoup

    CHART_DAYS = 60  # 오실레이터 차트용 60일 데이터 (3페이지)
    hdrs = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://finance.naver.com/"
    }

    def _parse_num(s):
        try: return int(s.replace(",","").replace("+",""))
        except: return 0

    def fetch_stock(args):
        ticker, name = args
        code = ticker.replace(".KS","").replace(".KQ","")
        try:
            rows_data = []
            pages_needed = (CHART_DAYS // 20) + 1
            for pg in range(1, pages_needed + 1):
                url = f"https://finance.naver.com/item/frgn.naver?code={code}&page={pg}"
                r = requests.get(url, headers=hdrs, timeout=12)
                if r.status_code != 200: break
                soup = BeautifulSoup(r.content, "html.parser", from_encoding="euc-kr")
                tables = soup.find_all("table")
                if len(tables) < 4: break
                for row in tables[3].find_all("tr"):
                    cells = [c.get_text(strip=True) for c in row.find_all("td")]
                    # cells[5]=기관 순매매량, cells[6]=외국인 순매매량
                    if len(cells) >= 7 and cells[0] and "." in cells[0]:
                        try:
                            close = _parse_num(cells[1])
                            inst_qty = _parse_num(cells[5])  # 기관
                            frgn_qty = _parse_num(cells[6])  # 외국인
                            if close > 0:
                                rows_data.append({"close": close, "date": cells[0],
                                                  "inst_qty": inst_qty, "inst_val": inst_qty * close,
                                                  "frgn_qty": frgn_qty, "frgn_val": frgn_qty * close})
                        except: pass
                if len(rows_data) >= CHART_DAYS: break

            if not rows_data: return None

            df_s = pd.DataFrame(rows_data[:CHART_DAYS]).iloc[::-1].reset_index(drop=True)
            df_s["inst_bil"] = df_s["inst_val"] / 100_000_000
            df_s["frgn_bil"] = df_s["frgn_val"] / 100_000_000
            df_s["inst_ma5"]  = df_s["inst_bil"].rolling(5, min_periods=1).mean()
            df_s["frgn_ma5"]  = df_s["frgn_bil"].rolling(5, min_periods=1).mean()
            df_s["inst_cum"]  = df_s["inst_bil"].rolling(days, min_periods=1).sum()
            df_s["frgn_cum"]  = df_s["frgn_bil"].rolling(days, min_periods=1).sum()

            frgn_nd_val = int(df_s["frgn_val"].tail(days).sum())  # 외국인 기준 정렬
            inst_nd_val = int(df_s["inst_val"].tail(days).sum())
            latest_close = int(df_s["close"].iloc[-1])

            return {
                "ticker": ticker, "name": name, "code": code,
                "net_nd_val": frgn_nd_val,  # 외국인 기준 정렬용
                "frgn_nd_bil": round(frgn_nd_val / 100_000_000, 1),
                "inst_nd_bil": round(inst_nd_val / 100_000_000, 1),
                "latest_close": latest_close,
                "chart": {
                    "dates":      df_s["date"].tolist(),
                    "price":      df_s["close"].tolist(),
                    "inst_daily": [round(v, 1) for v in df_s["inst_bil"].tolist()],
                    "inst_ma5":   [round(v, 1) for v in df_s["inst_ma5"].tolist()],
                    "inst_cum":   [round(v, 1) for v in df_s["inst_cum"].tolist()],
                    "frgn_daily": [round(v, 1) for v in df_s["frgn_bil"].tolist()],
                    "frgn_ma5":   [round(v, 1) for v in df_s["frgn_ma5"].tolist()],
                    "frgn_cum":   [round(v, 1) for v in df_s["frgn_cum"].tolist()],
                },
            }
        except: return None

    stock_list = [(t, n) for t, n in KR_STOCKS.items()]
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(fetch_stock, s): s for s in stock_list}
        for fut in concurrent.futures.as_completed(futs):
            res = fut.result()
            if res: results.append(res)

    if not results:
        return {"error": "데이터 수집 실패"}

    df_res = pd.DataFrame([{k: v for k, v in r.items() if k != "chart"} for r in results])
    df_res_sorted = df_res.sort_values("net_nd_val", ascending=False).reset_index(drop=True)
    chart_map = {r["ticker"]: r["chart"] for r in results}

    top20_rows  = df_res_sorted.head(top_n).to_dict("records")
    worst20_rows = df_res_sorted.tail(top_n).sort_values("net_nd_val").to_dict("records")

    for row in top20_rows + worst20_rows:
        row["chart"] = chart_map.get(row["ticker"])

    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "scanned": len(results),
        "days": days,
        "top20": top20_rows,
        "worst20": worst20_rows,
        "note": f"외국인 순매수 억원 ({days}거래일 누적) | 기관 데이터 포함 | 출처: 네이버 파이낸스"
    }

@st.cache_data(ttl=3600, show_spinner=False)
def get_krx_inst_market_flow(days=10):
    """KRX 시장 전체 기관/외국인 순매수 (최근 N거래일) — pykrx"""
    try:
        from pykrx import stock as krx
        import datetime as dt
        today = dt.date.today()
        start = today - dt.timedelta(days=days * 2 + 15)
        result = {}
        for mkt, label in [("KOSPI", "코스피"), ("KOSDAQ", "코스닥")]:
            try:
                df = krx.get_market_trading_value_by_investor(
                    start.strftime("%Y%m%d"), today.strftime("%Y%m%d"), market=mkt
                )
                if df.empty: continue
                df = df.tail(days)
                inst_col = next((c for c in df.columns if '기관합계' in str(c)), None)
                frgn_col = next((c for c in df.columns if '외국인합계' in str(c)), None) or \
                           next((c for c in df.columns if '외국인' in str(c) and '기타' not in str(c)), None)
                indv_col = next((c for c in df.columns if '개인' in str(c)), None)
                if not inst_col: continue
                result[label] = {
                    "dates": [str(d.date()) for d in df.index],
                    "inst": [int(v) for v in df[inst_col].tolist()],
                    "frgn": [int(v) for v in df[frgn_col].tolist()] if frgn_col else [],
                    "indv": [int(v) for v in df[indv_col].tolist()] if indv_col else [],
                    "inst_sum_bil": round(float(df[inst_col].sum()) / 1e8, 0),
                    "frgn_sum_bil": round(float(df[frgn_col].sum()) / 1e8, 0) if frgn_col else 0,
                }
            except: continue
        if not result:
            return {"error": "KRX 데이터 수집 실패 (pykrx)"}
        return {"data": result, "days": days, "date": today.strftime("%Y-%m-%d")}
    except ImportError:
        return {"error": "pykrx 미설치"}
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(ttl=3600, show_spinner=False)
def get_krx_volume_strength():
    """KOSPI 거래대금 강도 (RotationRate) 자동 계산 — pykrx"""
    try:
        from pykrx import stock as krx
        import datetime as dt
        today = dt.date.today()
        start = today - dt.timedelta(days=90)
        df = krx.get_index_ohlcv_by_date(start.strftime("%Y%m%d"), today.strftime("%Y%m%d"), "1001")
        if df.empty: return {"error": "KOSPI 인덱스 데이터 없음"}
        tv_col = next((c for c in df.columns if '거래대금' in str(c)), None)
        if not tv_col: return {"error": "거래대금 컬럼 없음"}
        tv = df[tv_col].dropna()
        if len(tv) < 21: return {"error": "데이터 부족 (최소 21일 필요)"}
        ma20 = tv.rolling(20).mean()
        cur = float(tv.iloc[-1]); ma = float(ma20.iloc[-1])
        rate = round(cur / ma * 100, 1) if ma > 0 else 0
        level = "과열" if rate > 150 else ("활발" if rate > 110 else ("보통" if rate > 80 else "저조"))
        tail30 = tv.tail(30); ma20_tail = ma20.tail(30)
        return {
            "date": str(tv.index[-1].date()),
            "today_tril": round(cur / 1e12, 2),
            "ma20_tril": round(ma / 1e12, 2),
            "rotation_rate": rate,
            "level": level,
            "chart": {
                "dates": [str(d.date()) for d in tail30.index],
                "values": [round(v/1e12, 2) for v in tail30.tolist()],
                "ma20": [round(v/1e12, 2) if pd.notna(v) else None for v in ma20_tail.tolist()]
            }
        }
    except ImportError:
        return {"error": "pykrx 미설치"}
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_inst_osc(ticker, days=20):
    """개별 종목 기관/외국인 순매수 오실레이터 — pykrx"""
    try:
        from pykrx import stock as krx
        import datetime as dt
        code = ticker.replace(".KS","").replace(".KQ","")
        today = dt.date.today()
        start = today - dt.timedelta(days=days * 2 + 15)
        df = krx.get_market_trading_value_by_investor(
            start.strftime("%Y%m%d"), today.strftime("%Y%m%d"), code
        )
        if df.empty: return {"error": "기관 데이터 없음"}
        df = df.tail(days)
        inst_col = next((c for c in df.columns if '기관합계' in str(c)), None)
        frgn_col = next((c for c in df.columns if '외국인합계' in str(c)), None) or \
                   next((c for c in df.columns if '외국인' in str(c) and '기타' not in str(c)), None)
        if not inst_col: return {"error": "기관합계 컬럼 없음"}
        inst_vals = [int(v) for v in df[inst_col].tolist()]
        frgn_vals = [int(v) for v in df[frgn_col].tolist()] if frgn_col else []
        inst_s = pd.Series(inst_vals)
        ma5 = inst_s.rolling(5, min_periods=1).mean()
        cum = inst_s.rolling(days, min_periods=1).sum()
        return {
            "ticker": ticker, "days": days,
            "inst_sum_bil": round(sum(inst_vals) / 1e8, 1),
            "frgn_sum_bil": round(sum(frgn_vals) / 1e8, 1) if frgn_vals else 0,
            "chart": {
                "dates": [str(d.date()) for d in df.index],
                "inst_daily": [round(v/1e8, 1) for v in inst_vals],
                "inst_ma5":   [round(v/1e8, 1) for v in ma5.tolist()],
                "inst_cum":   [round(v/1e8, 1) for v in cum.tolist()],
                "frgn_daily": [round(v/1e8, 1) for v in frgn_vals] if frgn_vals else [],
            }
        }
    except ImportError:
        return {"error": "pykrx 미설치"}
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(ttl=timedelta(hours=1), show_spinner=False)
def get_stock_price_chart(ticker: str, days: int = 180):
    """yfinance 일봉 OHLCV + MA(20/60/120/200) 반환"""
    import math
    try:
        t = ticker.strip().upper()
        raw = yf.download(t, period="3y", auto_adjust=True, progress=False)
        if raw.empty:
            return {"error": f"{t} 데이터 없음"}
        df = raw.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df = df.dropna(subset=["Close"])
        df["ma20"]  = df["Close"].rolling(20).mean()
        df["ma60"]  = df["Close"].rolling(60).mean()
        df["ma120"] = df["Close"].rolling(120).mean()
        df["ma200"] = df["Close"].rolling(200).mean()
        df = df.tail(days)
        def _safe(s):
            return [None if (isinstance(v, float) and math.isnan(v)) else round(float(v), 2) for v in s]
        return {
            "dates":  [str(d.date()) for d in df.index],
            "open":   [round(float(v), 2) for v in df["Open"]],
            "high":   [round(float(v), 2) for v in df["High"]],
            "low":    [round(float(v), 2) for v in df["Low"]],
            "close":  [round(float(v), 2) for v in df["Close"]],
            "volume": [int(v) for v in df["Volume"]],
            "ma20":   _safe(df["ma20"]),
            "ma60":   _safe(df["ma60"]),
            "ma120":  _safe(df["ma120"]),
            "ma200":  _safe(df["ma200"]),
        }
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=timedelta(hours=4), show_spinner=False)
def get_stock_financials(ticker: str):
    """분기 영업이익·매출·순이익·EPS + 애널리스트 목표가 (yfinance quarterly_income_stmt)"""
    import math
    t = ticker.strip().upper()
    try:
        tk = yf.Ticker(t)
        qi = None
        for attr in ("quarterly_income_stmt", "quarterly_financials"):
            try:
                val = getattr(tk, attr, None)
                if val is not None and not val.empty:
                    qi = val; break
            except Exception:
                continue
        if qi is None or qi.empty:
            return {"error": "재무 데이터 없음 (yfinance)"}
        qi_t = qi.T.sort_index()
        def _bil(val):
            try:
                v = float(val)
                return round(v / 1e8, 1) if not math.isnan(v) else None
            except Exception:
                return None
        quarters = []
        for dt, row in qi_t.iterrows():
            op  = row.get("Operating Income") or row.get("Ebit")
            rev = row.get("Total Revenue")
            net = row.get("Net Income")
            eps_v = row.get("Basic EPS") or row.get("Diluted EPS")
            try: eps = round(float(eps_v), 0) if eps_v and not math.isnan(float(eps_v)) else None
            except: eps = None
            quarters.append({"date": str(dt.date())[:7],
                              "op_profit": _bil(op), "revenue": _bil(rev),
                              "net_income": _bil(net), "eps": eps})
        if not quarters:
            return {"error": "분기 데이터 없음"}
        target_price = None
        try: target_price = tk.info.get("targetMeanPrice") or tk.info.get("targetMedianPrice")
        except: pass
        return {"ticker": t, "quarters": quarters[-8:], "target_price": target_price}
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=timedelta(hours=2))
def get_kr_consensus_auto(top_n=20):
    """WiseReport(네이버 내장) 컨센서스 자동 스크리닝 — EPS성장·매수비율·TP인상비율 기반"""
    import concurrent.futures
    from bs4 import BeautifulSoup

    hdrs = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://finance.naver.com/"
    }

    def _pn(s):
        try: return int(str(s).replace("원","").replace(",","").strip())
        except: return None

    def fetch_consensus(args):
        ticker, name = args
        code = ticker.replace(".KS","").replace(".KQ","")
        try:
            url = f"https://navercomp.wisereport.co.kr/v2/company/c1010001.aspx?cmp_cd={code}"
            r = requests.get(url, headers=hdrs, timeout=12)
            if r.status_code != 200: return None
            soup = BeautifulSoup(r.content, "html.parser", from_encoding="utf-8")
            tables = soup.find_all("table")
            if len(tables) < 12: return None

            # EPS 실적/추정 (table[5])
            eps_actual = eps_est = None
            for row in tables[5].find_all("tr"):
                cells = [c.get_text(strip=True) for c in row.find_all(["th","td"])]
                if len(cells) >= 3 and "EPS" in cells[0]:
                    eps_actual = _pn(cells[1]); eps_est = _pn(cells[2])
                    break

            # 컨센서스 요약 (table[11]): avg_rating, cons_tp, analyst_count
            cons_tp = analyst_count = avg_rating = None
            rows11 = tables[11].find_all("tr")
            if len(rows11) >= 2:
                vals = [c.get_text(strip=True) for c in rows11[1].find_all(["th","td"])]
                if len(vals) >= 5:
                    try: avg_rating = float(vals[0])
                    except: pass
                    cons_tp = _pn(vals[1])
                    try: analyst_count = int(vals[4].replace(",",""))
                    except: pass

            # 애널리스트 의견 (table[12]): buy%, TP인상%
            buy_n = total_n = raised_n = 0
            for row in tables[12].find_all("tr")[1:]:
                cells = [c.get_text(strip=True) for c in row.find_all(["th","td"])]
                if len(cells) >= 6:
                    try:
                        tp_now = _pn(cells[2]); tp_prev = _pn(cells[3])
                        rating = cells[5].upper()
                        total_n += 1
                        if "BUY" in rating or "매수" in rating: buy_n += 1
                        if tp_now and tp_prev and tp_now > tp_prev: raised_n += 1
                    except: pass

            if analyst_count is None or analyst_count < 2: return None
            buy_pct = round(buy_n / total_n * 100, 1) if total_n > 0 else 0
            tp_raise_pct = round(raised_n / total_n * 100, 1) if total_n > 0 else 0

            eps_growth = None
            if eps_actual and eps_est and eps_actual > 0:
                eps_growth = round((eps_est - eps_actual) / eps_actual * 100, 1)

            # 컨센 스코어: EPS성장(40%) + 매수비율(30%) + TP인상비율(30%)
            score = 0.0
            if eps_growth is not None: score += min(eps_growth, 300) / 300 * 40
            score += buy_pct / 100 * 30
            score += tp_raise_pct / 100 * 30

            return {
                "ticker": ticker, "name": name,
                "eps_actual": eps_actual, "eps_est": eps_est,
                "eps_growth": eps_growth,
                "cons_tp": cons_tp, "analyst_count": analyst_count,
                "avg_rating": avg_rating,
                "buy_pct": buy_pct, "tp_raise_pct": tp_raise_pct,
                "score": round(score, 1),
            }
        except: return None

    stock_list = [(t, n) for t, n in KR_STOCKS.items()]
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(fetch_consensus, s): s for s in stock_list}
        for fut in concurrent.futures.as_completed(futs):
            res = fut.result()
            if res: results.append(res)

    if not results: return {"error": "컨센서스 데이터 수집 실패"}

    df_c = pd.DataFrame(results).sort_values("score", ascending=False).reset_index(drop=True)
    top20 = df_c.head(top_n).to_dict("records")
    # 컨센 가속 필터: EPS성장 양수 + 매수비율≥60% + TP인상비율≥50%
    accel = df_c[(df_c["eps_growth"].notna()) & (df_c["eps_growth"] > 0) &
                 (df_c["buy_pct"] >= 60) & (df_c["tp_raise_pct"] >= 50)].head(top_n).to_dict("records")

    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "scanned": len(results),
        "top20": top20,
        "accel": accel,
        "note": "출처: WiseReport/네이버 | EPS성장(40%)+매수비율(30%)+TP인상비율(30%) 가중합산"
    }

def calc_consensus_excel(df_db):
    """컨센 Excel db 시트 분석 — 빈집여부 기준 3가지로 분류

    빈집 개념: 외국인+기관 합산 순매수가 낮을수록(음수) = 아직 덜 샀음 = 매수 기회
    - 빈집 + EPS가속: 실적 개선 중인데 아직 안 샀음 → 최우선 기회
    - 빈집 전환: 장기 비어있다가 최근 매집 시작 → 전환 신호
    - 수급 유입 중: 이미 사고 있음 → 모멘텀 참고용
    """
    try:
        cols = list(df_db.columns)
        COL_NAME="종목명"; COL_MKT="유동시가총액"
        COL_EPS_1M=next((c for c in cols if "1개월" in c and "EPS" in c),None)
        COL_EPS_3M=next((c for c in cols if "3개월" in c and "EPS" in c),None)
        COL_F_6M=next((c for c in cols if "120" in c and "외국" in c),None)
        COL_F_1M=next((c for c in cols if "20" in c and "외국" in c and "120" not in c),None)
        COL_I_6M=next((c for c in cols if "120" in c and "기관" in c),None)
        COL_I_1M=next((c for c in cols if "20" in c and "기관" in c and "120" not in c),None)
        for col in [COL_EPS_1M,COL_EPS_3M,COL_MKT,COL_F_6M,COL_F_1M,COL_I_6M,COL_I_1M]:
            if col and col in df_db.columns: df_db[col]=pd.to_numeric(df_db[col],errors='coerce')
        if not COL_EPS_1M or not COL_EPS_3M: return {"error":"EPS 컬럼을 찾을 수 없음"}
        # EPS 가속 필터: 1개월 변화 > 3개월 변화 = 최근 들어 더 많이 상향됨
        mask=(df_db[COL_EPS_1M].notna()&df_db[COL_EPS_3M].notna()&
              (df_db[COL_EPS_1M]>0)&(df_db[COL_EPS_3M]>0)&(df_db[COL_EPS_1M]>df_db[COL_EPS_3M]))
        base=df_db[mask].copy()
        binzip_list=[]; turn_list=[]; inflow_list=[]
        if COL_F_1M and COL_I_1M:
            base=base[base[COL_MKT].notna()&(base[COL_MKT]>0)].copy()
            base["합산_1M"]=base[COL_F_1M].fillna(0)+base[COL_I_1M].fillna(0)   # 단기 외국인+기관
            base["합산_6M"]=(base[COL_F_6M].fillna(0)+base[COL_I_6M].fillna(0)) if COL_F_6M and COL_I_6M else base["합산_1M"]
            # 빈집 + EPS가속: 단기 합산 ≤ 0 (수급 비어있음 = 아직 안 삼)
            bz=base[base["합산_1M"]<=0].sort_values("합산_1M")   # 가장 빈집부터
            binzip_list=bz[COL_NAME].tolist()
            # 빈집 전환: 장기 비어있다가 단기 수급 유입 시작
            tr=base[(base["합산_6M"]<=0)&(base["합산_1M"]>0)].sort_values("합산_1M",ascending=False)
            turn_list=tr[COL_NAME].tolist()
            # 수급 유입 중: 단기 합산 > 0 (이미 사는 중 — 빈집전환 종목 제외)
            inf=base[(base["합산_1M"]>0)&(~base[COL_NAME].isin(turn_list))].sort_values("합산_1M",ascending=False).head(20)
            inflow_list=inf[COL_NAME].tolist()
        return {
            "eps_passed": len(base),
            "binzip_list": binzip_list,      # 🏚️ 빈집 + EPS가속 (핵심)
            "turn_list": turn_list,           # 🔄 빈집 전환 (장기빈집→단기 수급유입)
            "inflow_list": inflow_list,       # 📈 수급 유입 중 (모멘텀 참고)
            "has_supply": bool(COL_F_1M and COL_I_1M),
        }
    except Exception as e: return {"error":str(e)}

# ── 공통 차트 헬퍼 ──
def _chart_layout(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10,r=20,t=30,b=10),
                      plot_bgcolor='#0d1117', paper_bgcolor='rgba(0,0,0,0)',
                      font=dict(color='#c9d1d9', family='Inter, sans-serif'),
                      legend=dict(orientation='h', y=1.08, bgcolor='rgba(0,0,0,0)',
                                  font=dict(size=11)),
                      xaxis=dict(showgrid=True, gridcolor='#21262d', gridwidth=1,
                                 zeroline=False, color='#7d8590'),
                      yaxis=dict(showgrid=True, gridcolor='#21262d', gridwidth=1,
                                 zeroline=False, color='#7d8590'),
                      dragmode=False)
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.07)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.07)')
    return fig

def _rs_bar_chart(items, name_key="name", val_key="norm_rs", height=320):
    s = sorted(items, key=lambda x: x[val_key])
    names = [r[name_key] for r in s]; vals = [r[val_key] for r in s]
    colors = ["#00c853" if v>=70 else ("#ffc107" if v>=50 else "#ff4b4b") for v in vals]
    fig = go.Figure(go.Bar(x=vals, y=names, orientation='h', marker_color=colors,
                           text=[f"{v:.1f}" for v in vals], textposition='outside'))
    fig.update_layout(xaxis_range=[0,112], height=max(height, len(names)*26), dragmode=False,
                      margin=dict(l=10,r=50,t=10,b=10),
                      plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#e0e0e0')
    fig.add_vline(x=70, line_dash="dash", line_color="#00c853", opacity=0.5)
    fig.add_vline(x=50, line_dash="dash", line_color="#ffc107", opacity=0.4)
    return fig

def _osc_bar_chart(dates, osc_vals, height=220):
    colors = ["#00c853" if v>0 else "#ff4b4b" for v in osc_vals]
    fig = go.Figure(go.Bar(x=dates, y=osc_vals, marker_color=colors))
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
    return _chart_layout(fig, height)

# ─────────────────────────────────────────────────────────────────────────────
# 탭 1: 미국 지표
# ─────────────────────────────────────────────────────────────────────────────
with tab0:
    # ── 자동 로딩 (버튼 없음 — @st.cache_data 1시간 캐시) ─────────────────────────
    with st.spinner("📡 시장 분석 중... (첫 실행 1~2분, 이후 즉시 표시)"):
        _d_canary = get_canary_signal()
        _d_bofa   = get_bofa_heat()
        _d_blood  = get_blood_indicator()
        _d_fg     = get_us_fear_greed()
        _d_cop    = get_coppock()
        _d_fg_m   = get_monthly_fear_greed()
        _d_kr_fg  = get_kr_fg_auto()
        _d_market = get_market_summary()

    # ── 전체 판단 ────────────────────────────────────────────────────────────────
    _d_attack  = bool(_d_canary and "error" not in _d_canary and _d_canary.get("mode") == "공격")
    _d_heat    = float(_d_bofa.get("heat", 5)) if _d_bofa and "error" not in _d_bofa else 5.0
    _d_spy_cop = (_d_cop.get("SPY", {}) if _d_cop and "error" not in _d_cop else {}) or {}
    _d_cop_up  = bool(_d_spy_cop.get("pos") and _d_spy_cop.get("trend") == "상승")

    if _d_attack and _d_cop_up and _d_heat < 7.5:
        _d_cls, _d_icon, _d_main, _d_sub = (
            "dash-status-green", "🟢",
            "매수 구간 — 지금 주식을 보유해도 됩니다",
            f"카나리아 공격 · 코포크 상승 · 시장 열기 {_d_heat:.1f}/10",
        )
    elif not _d_attack:
        _d_cls, _d_icon, _d_main, _d_sub = (
            "dash-status-red", "🔴",
            "방어 구간 — 현금·채권 비중 확대",
            "카나리아 방어 모드 (QQQ 또는 TIP 모멘텀 음수)",
        )
    elif _d_heat >= 7.5:
        _d_cls, _d_icon, _d_main, _d_sub = (
            "dash-status-yellow", "🟡",
            "과열 주의 — 신규 매수 자제",
            f"시장 열기 {_d_heat:.1f}/10 (7.5↑ 과열)",
        )
    else:
        _d_cls, _d_icon, _d_main, _d_sub = (
            "dash-status-yellow", "🟡",
            "중립 — 신중하게 대응",
            "신호 혼재 · 기존 포지션 유지 권장",
        )

    st.markdown(
        f'<div class="{_d_cls}">'
        f'<div style="font-size:1.6rem;font-weight:800;color:#e6edf3;letter-spacing:-0.02em;">{_d_icon}&nbsp;{_d_main}</div>'
        f'<div style="font-size:0.82rem;color:#8b949e;margin-top:6px;">{_d_sub}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── 미국 핵심 신호 5개 ─────────────────────────────────────────────────────────
    st.markdown('<p class="zone-header">🌎 미국 시장</p>', unsafe_allow_html=True)

    def _dc(col, icon, value, label, color="#e6edf3"):
        col.markdown(
            f'<div class="dash-metric">'
            f'<div style="font-size:1.3rem;line-height:1.1;">{icon}</div>'
            f'<div style="color:{color};font-weight:700;font-size:0.88rem;margin:4px 0 2px;">{value}</div>'
            f'<div style="color:#7d8590;font-size:0.72rem;">{label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    _d_col1, _d_col2, _d_col3, _d_col4, _d_col5 = st.columns(5, gap="small")

    if _d_canary and "error" not in _d_canary:
        _ci = "🟢" if _d_canary["mode"] == "공격" else "🔴"
        _cc = "#3fb950" if _d_canary["mode"] == "공격" else "#f85149"
        _dc(_d_col1, _ci, _d_canary["mode"], "카나리아", _cc)

    if _d_bofa and "error" not in _d_bofa:
        _bi = "🔴" if _d_heat >= 7.5 else ("🟡" if _d_heat >= 5 else "🟢")
        _bc = "#f85149" if _d_heat >= 7.5 else ("#d29922" if _d_heat >= 5 else "#3fb950")
        _dc(_d_col2, _bi, f"{_d_heat:.1f}/10", "BofA 열기", _bc)

    if _d_blood and "error" not in _d_blood:
        _bli = "🟢" if _d_blood["vs_ma20"] == "위" else "🔴"
        _blc = "#3fb950" if _d_blood["vs_ma20"] == "위" else "#f85149"
        _dc(_d_col3, _bli, f"MA20 {_d_blood['vs_ma20']}", "채권 스트레스", _blc)

    if _d_fg and "error" not in _d_fg:
        _fi = "🟢" if _d_fg["spx_osc"] > 0 else "🔴"
        _fc = "#3fb950" if _d_fg["spx_osc"] > 0 else "#f85149"
        _dc(_d_col4, _fi, _d_fg["spx_sentiment"], "공포탐욕(일)", _fc)

    if _d_spy_cop:
        _cop_pos = _d_spy_cop.get("pos"); _cop_trend = _d_spy_cop.get("trend")
        if not _cop_pos and _cop_trend == "상승":
            _copi = "🔔"; _copc = "#f0c040"  # 전환점 — 가장 강력한 신호
        elif _cop_pos and _cop_trend == "상승":
            _copi = "🟢"; _copc = "#3fb950"
        elif _cop_pos and _cop_trend == "하락":
            _copi = "🟡"; _copc = "#d29922"
        else:
            _copi = "🔴"; _copc = "#f85149"
        _dc(_d_col5, _copi, f"{_d_spy_cop.get('value', 0):+.1f}", "코포크(SPY)", _copc)

    # ── 국내 시장 ──────────────────────────────────────────────────────────────────
    st.markdown('<p class="zone-header">🇰🇷 국내 시장</p>', unsafe_allow_html=True)
    _dk_cols = st.columns(4)

    if _d_market and "error" not in _d_market:
        _ks = _d_market.get("kospi", {}) or {}
        if "close" in _ks:
            _ksi = "🟢" if _ks["chg_pct"] >= 0 else "🔴"
            _dc(_dk_cols[0], _ksi,
                f"{float(_ks['close']):,.0f}  ({_ks['chg_pct']:+.2f}%)", "KOSPI",
                "#3fb950" if _ks["chg_pct"] >= 0 else "#f85149")
        _kq = _d_market.get("kosdaq", {}) or {}
        if "close" in _kq:
            _kqi = "🟢" if _kq["chg_pct"] >= 0 else "🔴"
            _dc(_dk_cols[1], _kqi,
                f"{float(_kq['close']):,.0f}  ({_kq['chg_pct']:+.2f}%)", "KOSDAQ",
                "#3fb950" if _kq["chg_pct"] >= 0 else "#f85149")

    if _d_kr_fg and "error" not in _d_kr_fg:
        _kr_ks = (_d_kr_fg.get("results") or {}).get("KOSPI", {})
        if "osc" in _kr_ks:
            _kfi = "🟢" if _kr_ks["osc"] > 0 else "🔴"
            _dc(_dk_cols[2], _kfi,
                f"{_kr_ks['osc']:+.3f}  {_kr_ks.get('sentiment', '')}", "KR FGI",
                "#3fb950" if _kr_ks["osc"] > 0 else "#f85149")

    if _d_fg_m and "error" not in _d_fg_m:
        _mfi = "🟢" if _d_fg_m["spx_osc"] > 0 else "🔴"
        _dc(_dk_cols[3], _mfi,
            f"{_d_fg_m['spx_osc']:+.3f}  {_d_fg_m.get('spx_sentiment', '')}", "월간심리(US)",
            "#3fb950" if _d_fg_m["spx_osc"] > 0 else "#f85149")

    # ── 하단 새로고침 ────────────────────────────────────────────────────────────────
    st.markdown("---")
    import datetime as _dtd0
    _kstd0 = _dtd0.datetime.utcnow() + _dtd0.timedelta(hours=9)
    _dc0_1, _dc0_2 = st.columns([4, 1])
    _dc0_1.caption(f"갱신: {_kstd0.strftime('%Y-%m-%d %H:%M')} KST · 1시간 캐시 · 상세 분석은 각 탭 클릭")
    if _dc0_2.button("🔄 새로고침", key="dash_refresh"):
        st.cache_data.clear()
        st.rerun()

with tab1:
    with st.expander("❓ 어려운 용어 설명"):
        st.markdown("**🐤 카나리아** — 시장 진입 신호. 나스닥·물가채 모멘텀이 모두 양수면 공격(주식), 하나라도 음수면 방어(현금·채권)")
        st.markdown("**🌡️ BOFA 열기** — 시장 과열도 0~10점. 7.5↑ 과열 / 2.5↓ 안전")
        st.markdown("**🩸 블러드** — 채권 스트레스 지수. 60일 평균 위=안전, 아래=주의")
        st.markdown("**📡 ZBT** — 급락 후 반등 신호. 신호 발생 시 단기 저점 가능성")
        st.markdown("**😱 공포·탐욕** — 시장 심리. 양수(+)=탐욕(과열 주의) / 음수(-)=공포(매수 기회)")
        st.markdown("**⚡ 임펄스** — 단기 추세. 🟢강세=주가+MACD 모두 상승 / 🔴약세=모두 하락")
        st.markdown("**📏 이평 위치** — 20·60·120·200일 평균 대비 현재가. 양수=상승세")
        st.markdown("**🔢 TD 카운트** — 패턴 카운트 1~9. 9에 가까울수록 추세 전환 주의")
        st.markdown("**📈 코포크** — 중장기 추세. 양수(+)이고 상승 중이면 매수 유리")
        st.markdown("**💪 상대강도(RS)** — 다른 종목보다 얼마나 더 올랐는지. 높을수록 강세")

    _t1h, _t1b = st.columns([6, 1])
    _t1h.caption("📡 자동 분석 · 1시간 캐시 유지")
    if _t1b.button("🔄", key="us_refresh", help="캐시 초기화 후 재분석"):
        st.cache_data.clear(); st.rerun()
    with st.spinner("데이터 불러오는 중... (첫 실행 약 1~2분, 이후 즉시)"):
        canary=get_canary_signal()
        bofa=get_bofa_heat()
        blood=get_blood_indicator()
        fg=get_us_fear_greed()
        fg_m=get_monthly_fear_greed()
        coppock=get_coppock(); coppock_fast=get_coppock_fast()
        zbt=get_zbt()
        sp500_rs=get_sp500_rs(SP500_TOP100)
        ndx_rs=get_nasdaq100_rs(NASDAQ100)
        us_sector=get_us_sector_rs()
    if True:

        # ── 1. 종합 신호 ──
        st.markdown('<p class="zone-header">📊 지금 시장은?</p>', unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        if canary and "error" not in canary:
            color = "sig-green" if canary["mode"]=="공격" else "sig-red"
            c1.markdown(f'<div class="{color}">🐤 카나리아 신호<br><b>{canary["mode"]} 모드</b><br><span style="font-size:0.78rem">나스닥 모멘텀 {canary["qqq_mom"]*100:+.1f}%</span></div>', unsafe_allow_html=True)
        if bofa and "error" not in bofa:
            heat_color = "sig-red" if bofa["heat"]>=7.5 else ("sig-yellow" if bofa["heat"]>=5 else "sig-green")
            c2.markdown(f'<div class="{heat_color}">🌡️ 시장 열기<br><b>{bofa["heat"]}/10점</b><br><span style="font-size:0.78rem">{_heat_label(bofa["heat"])} · 상승추세{"✅" if bofa["trend_on"] else "❌"}</span></div>', unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        if blood and "error" not in blood:
            blood_color = "sig-green" if blood["vs_ma60"] == "위" else "sig-red"
            c1.markdown(f'<div class="{blood_color}">🩸 채권 스트레스<br><b>60일평균 {blood["vs_ma60"]}</b><br><span style="font-size:0.78rem">수치: {blood["value"]:.4f}</span></div>', unsafe_allow_html=True)
        if zbt and "error" not in zbt:
            zbt_color = "sig-green" if zbt.get("signal") else "sig-yellow"
            c2.markdown(f'<div class="{zbt_color}">📡 반등 신호(ZBT)<br><b>{"🟢 신호 발생!" if zbt.get("signal") else "⏳ 대기 중"}</b><br><span style="font-size:0.78rem">수치: {zbt["zbt"]:.3f}</span></div>', unsafe_allow_html=True)

        st.divider()

        # ── 2. 공포·탐욕 오실레이터 ──
        st.markdown('<p class="zone-header">😱 시장 심리 — 지금 탐욕인가 공포인가</p>', unsafe_allow_html=True)
        st.caption("양수(+) = 탐욕 → 과열 주의 | 음수(-) = 공포 → 매수 기회일 수 있음")
        _fg_r1 = st.columns(3)
        _fg_r2 = st.columns(3)
        if fg and "error" not in fg:
            _fg_r1[0].metric("S&P500 심리", fg["spx_osc"], f"{'🟢' if fg['spx_osc']>0 else '🔴'} {fg['spx_sentiment']}", help="S&P500 기준 공포·탐욕 오실레이터. 양수=탐욕(과열), 음수=공포(매수기회)")
            _fg_r1[1].metric("나스닥 심리", fg["ndx_osc"], f"{'🟢' if fg['ndx_osc']>0 else '🔴'} {fg['ndx_sentiment']}", help="나스닥 기준 공포·탐욕 오실레이터")
            _fg_r1[2].metric("S&P500 추세", fg["spy_impulse"], help="단기 추세 방향. 🟢강세=주가+MACD 모두 상승 / 🔴약세=모두 하락 / 🔵중립")
            _fg_r2[0].metric("나스닥 추세", fg["qqq_impulse"], help="나스닥 단기 추세 방향")
            _fg_r2[1].metric("S&P500 위치", f"{fg['spy_gap']:+.2f}%", help="장기 이평선(20·60·120·200일 평균) 대비 현재가 위치. 양수=상승세")
            _fg_r2[2].metric("나스닥 위치", f"{fg['qqq_gap']:+.2f}%", help="나스닥 장기 이평선 대비 현재가 위치")
            c1,c2 = st.columns(2)
            c1.metric("S&P500 패턴 카운트", f"매도 {fg['spy_td_sell']} / 매수 {fg['spy_td_buy']}", help="TD 카운트. 숫자가 9에 가까울수록 추세 전환 주의")
            c2.metric("나스닥 패턴 카운트", f"매도 {fg['qqq_td_sell']} / 매수 {fg['qqq_td_buy']}", help="TD 카운트. 숫자가 9에 가까울수록 추세 전환 주의")
            if fg_m and "error" not in fg_m:
                c1.metric("S&P500 월간 심리", fg_m["spx_osc"], f"{'🟢' if fg_m['spx_osc']>0 else '🔴'} {fg_m['spx_sentiment']}", help="월봉 기준 공포·탐욕 오실레이터 (중장기 시각)")
                c2.metric("나스닥 월간 심리", fg_m["ndx_osc"], f"{'🟢' if fg_m['ndx_osc']>0 else '🔴'} {fg_m['ndx_sentiment']}", help="월봉 기준 공포·탐욕 오실레이터 (중장기 시각)")
            if fg.get("chart"):
                with st.expander("📈 오실레이터 차트 (최근 6개월)"):
                    ch=fg["chart"]
                    _fig=go.Figure()
                    _fig.add_trace(go.Scatter(x=ch["dates"],y=ch["spx_osc"],name="S&P500",line=dict(color="#6A5ACD",width=2)))
                    _fig.add_trace(go.Scatter(x=ch["dates"],y=ch["ndx_osc"],name="NASDAQ",line=dict(color="#00B3B3",width=2)))
                    _fig.add_trace(go.Scatter(x=ch["dates"],y=ch["spy"],name="SPY가격",yaxis="y2",line=dict(color="#666",width=1.5,dash="dot")))
                    _fig.add_hline(y=0,line_dash="dash",line_color="rgba(255,255,255,0.3)")
                    _fig.update_layout(yaxis2=dict(overlaying='y',side='right',showgrid=False), dragmode=False,
                                        height=320,margin=dict(l=10,r=60,t=20,b=10),
                                        plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',
                                        font_color='#e0e0e0',legend=dict(orientation='h',y=1.08))
                    _fig.update_xaxes(showgrid=True,gridcolor='rgba(255,255,255,0.07)')
                    _fig.update_yaxes(showgrid=True,gridcolor='rgba(255,255,255,0.07)')
                    st.plotly_chart(_fig, use_container_width=True, config={"scrollZoom": False})

        st.divider()

        # ── 3. 코포크 + ZBT ──
        st.markdown('<p class="zone-header">📈 중장기 방향 — 지금 상승세인가</p>', unsafe_allow_html=True)
        st.caption("코포크가 양수(+)이고 상승 중이면 중장기 매수 유리 | 음수에서 반등(전환점) = 강력 매수 기회 | ZBT: 급락 후 반등 신호")
        _cp1, _cp2 = st.columns(2)
        with _cp1:
            st.caption("표준 코포크 (중장기)")
            if coppock and "error" not in coppock:
                _cp_n = min(len(coppock), 2)
                cols=st.columns(_cp_n)
                for i,(lbl,v) in enumerate(coppock.items()):
                    arr="▲" if v["trend"]=="상승" else "▼"
                    # 전환점 감지: 음수에서 상승 전환 = 가장 강력한 매수 신호
                    if not v["pos"] and v["trend"]=="상승":
                        icon="🔔"; _help="⭐ 전환점! 음수에서 반등 시작 — 역사적으로 가장 강력한 중장기 매수 신호"
                    elif v["pos"] and v["trend"]=="상승":
                        icon="🟢"; _help="양수이고 상승 중 — 중장기 매수 유리"
                    elif v["pos"] and v["trend"]=="하락":
                        icon="🟡"; _help="양수지만 하락 전환 — 모멘텀 약화 주의"
                    else:
                        icon="🔴"; _help="음수이고 하락 중 — 중장기 조심"
                    cols[i % _cp_n].metric(lbl, f"{icon} {v['value']}", f"{arr} {v['trend']}", help=_help)
        with _cp2:
            st.caption("빠른 코포크 (단기)")
            if coppock_fast and "error" not in coppock_fast:
                _cpf_n = min(len(coppock_fast), 2)
                cols=st.columns(_cpf_n)
                for i,(lbl,v) in enumerate(coppock_fast.items()):
                    arr="▲" if v["trend"]=="상승" else "▼"
                    if not v["pos"] and v["trend"]=="상승":
                        icon="🔔"; _help="⭐ 단기 전환점! 음수에서 반등 시작"
                    elif v["pos"] and v["trend"]=="상승":
                        icon="🟢"; _help="단기 상승 추세"
                    elif v["pos"] and v["trend"]=="하락":
                        icon="🟡"; _help="단기 모멘텀 약화"
                    else:
                        icon="🔴"; _help="단기 하락 추세"
                    cols[i % _cpf_n].metric(lbl, f"{icon} {v['value']}", f"{arr} {v['trend']}", help=_help)
        if zbt and "error" not in zbt:
            st.caption("반등 신호(ZBT)")
            _zbt_c1, _zbt_c2 = st.columns(2)
            _zbt_c1.metric("ZBT 신호", "🟢 발생!" if zbt.get("signal") else "⏳ 대기", f"수치: {zbt['zbt']:.3f}", help="급락 후 반등 포착 신호. 발생 시 단기 저점 가능성")
            if zbt.get("vix"): _zbt_c2.metric("변동성(VIX)", zbt["vix"], "안정✅" if zbt.get("vix_ok") else "⚠️ 불안", help="VIX: 시장 공포 지수. 20 이하면 안정, 30 이상이면 공포 구간")

        st.divider()

        # ── 4. RS 상위 종목 ──
        st.markdown('<p class="zone-header">🏆 지금 강한 종목 TOP10</p>', unsafe_allow_html=True)
        st.caption("다른 종목보다 더 많이 오른 종목 순위 — 강한 종목이 계속 강한 경향이 있습니다")
        c1,c2 = st.columns(2)
        with c1:
            st.caption("S&P500 Top 10")
            if sp500_rs and "error" not in sp500_rs:
                for i,item in enumerate(sp500_rs["top"],1):
                    t=item["ticker"]; kr=TICKER_NAMES.get(t,""); rs=item.get("rs",0)
                    st.markdown(f"`{i:2d}` **{t}** {kr} &nbsp; {_cv(rs,'1f')}", unsafe_allow_html=True)
            elif sp500_rs: st.error(sp500_rs.get("error"))
        with c2:
            st.caption("나스닥100 Top 10")
            if ndx_rs and "error" not in ndx_rs:
                for i,item in enumerate(ndx_rs["top"],1):
                    t=item["ticker"]; kr=TICKER_NAMES.get(t,""); rs=item.get("rs",0)
                    st.markdown(f"`{i:2d}` **{t}** {kr} &nbsp; {_cv(rs,'2f')}", unsafe_allow_html=True)
            elif ndx_rs: st.error(ndx_rs.get("error"))

        st.divider()

        # ── 5. 섹터 ETF RS ──
        st.markdown('<p class="zone-header">🏭 어떤 업종이 강한가</p>', unsafe_allow_html=True)
        st.caption("🟢 강세(70↑) = 지금 돈이 몰리는 업종 / 🔴 약세(50↓) = 자금이 빠지는 업종")
        if us_sector and "error" not in us_sector:
            df_s=pd.DataFrame(us_sector["sectors"])
            df_s["강도"]=df_s["norm_rs"].apply(lambda x:"🟢 강세" if x>=70 else ("🟡 중립" if x>=50 else "🔴 약세"))
            st.dataframe(df_s.rename(columns={"ticker":"티커","name":"업종","norm_rs":"강도점수(0~100)","risk_adj":"변동성조정","강도":"판정"})[["티커","업종","강도점수(0~100)","판정"]],
                use_container_width=True, hide_index=True,
                column_config={"강도점수(0~100)":st.column_config.ProgressColumn("강도점수(0~100)",min_value=0,max_value=100,format="%.1f")})
            with st.expander("📊 업종별 강도 차트"):
                st.plotly_chart(_rs_bar_chart(us_sector["sectors"], name_key="name"), use_container_width=True, config={"scrollZoom": False})
        elif us_sector: st.error(us_sector.get("error"))

# ─────────────────────────────────────────────────────────────────────────────
# 탭 2: 국내 지표
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    # ── KST 시간 + 자동 업데이트 ──
    import datetime as _dt
    _kst = _dt.datetime.utcnow() + _dt.timedelta(hours=9)
    _is_weekday   = _kst.weekday() < 5
    _is_after_close = _kst.hour >= 16
    _market_status = (
        "🔴 장 마감 후" if _is_after_close else
        ("🟢 장 중" if (_kst.hour >= 9 and (_kst.hour < 15 or (_kst.hour == 15 and _kst.minute <= 30))) else "⚫ 장 외")
    )
    _col_time, _col_toggle = st.columns([3, 1])
    _col_time.caption(f"🕐 현재 KST {_kst.strftime('%H:%M')} | {_market_status} {'(평일)' if _is_weekday else '(주말)'}")

    _auto_refresh_on = False
    if _is_after_close and _is_weekday:
        with _col_toggle:
            _auto_refresh_on = st.toggle("🔄 자동", key="auto_refresh_toggle", value=False,
                                          help="20분마다 자동 새로고침 (오후 4시 이후 평일만 활성화)")
        if _auto_refresh_on and _HAS_AUTOREFRESH:
            _refresh_count = st_autorefresh(interval=20 * 60 * 1000, key="tab2_autorefresh")
            st.caption(f"⏱ 20분마다 자동 새로고침 중 | 총 {_refresh_count}회 갱신")
        elif _auto_refresh_on and not _HAS_AUTOREFRESH:
            st.caption("⚠ streamlit-autorefresh 패키지 설치 중...")
    else:
        _col_toggle.caption("")

    # 4시 이후 오늘 첫 방문 시 자동 실행
    _today_str = _kst.strftime("%Y-%m-%d")
    _auto_ran_key = f"kr_auto_ran_{_today_str}"

    with st.expander("❓ 어려운 용어 설명"):
        st.markdown("**💹 수급 오실레이터** — 낮을수록(빈집) 매집 여력 큼. 음수/낮음=🏚️빈집(좋음) / 높음=이미 매집됨(주의)")
        st.markdown("**💪 상대강도(RS)** — 다른 ETF 대비 강도. 70점↑ 강세 / 50점↓ 약세")
        st.markdown("**🏠 빈집 주도주** — 주도 업종 중 아직 덜 오른 종목. 따라 오를 가능성")
        st.markdown("**⚡ 임펄스** — 단기 추세. 🟢강세=주가+MACD 모두 상승 / 🔴약세=모두 하락")
        st.markdown("**🔢 TD 카운트** — 패턴 카운트 1~9. 9에 가까울수록 추세 전환 주의")
        st.markdown("**💧 CMF(자금흐름)** — 양수=자금 유입(매수 우세) / 음수=자금 유출(매도 우세)")
        st.markdown("**📊 오실레이터** — 기준선(0) 위=상승 추세 / 아래=하락 추세")

    st.markdown('<p class="zone-header">📡 오늘 시장 현황</p>', unsafe_allow_html=True)

    _t2h, _t2b = st.columns([6, 1])
    _t2h.caption(f"📡 자동 분석 · {_kst.strftime('%H:%M')} KST 기준")
    if _t2b.button("🔄", key="kr_refresh", help="캐시 초기화"):
        st.cache_data.clear(); st.rerun()
    with st.spinner("국내 시장 데이터 불러오는 중..."):
        market=get_market_summary()
        sector=get_sector_performance()
        supply=get_supply_oscillator()
        kr_etf=get_kr_etf_rs()
        binzip=get_binzip_stocks(supply_data=supply)
    if True:

        # ── 지수 현황 ──
        st.markdown('<p class="zone-header">📊 코스피·코스닥 현황</p>', unsafe_allow_html=True)
        if market and "error" not in market:
            kp=market["kospi"]; kq=market["kosdaq"]
            st.caption(f"기준: {market['date']} KST | 출처: 네이버 실시간")
            c1,c2=st.columns(2)
            # 등락률을 메인 수치로, 현재가를 보조로
            kp_arrow = "▲" if kp['chg_pct'] >= 0 else "▼"
            kq_arrow = "▲" if kq['chg_pct'] >= 0 else "▼"
            c1.metric(
                f"코스피 {kp_arrow} {kp['chg_pct']:+.2f}%",
                f"{kp['close']:,.2f}",
                f"전일대비 {kp['chg']:+.2f}pt | 변동폭 {kp['vol_range']:.2f}%",
                delta_color="normal"
            )
            c2.metric(
                f"코스닥 {kq_arrow} {kq['chg_pct']:+.2f}%",
                f"{kq['close']:,.2f}",
                f"전일대비 {kq['chg']:+.2f}pt | 변동폭 {kq['vol_range']:.2f}%",
                delta_color="normal"
            )
            # 고가/저가/시가 상세
            c1, c2 = st.columns(2)
            with c1:
                st.caption(f"코스피  시가 {kp['open']:,.2f}  고가 {kp['high']:,.2f}  저가 {kp['low']:,.2f}")
            with c2:
                st.caption(f"코스닥  시가 {kq['open']:,.2f}  고가 {kq['high']:,.2f}  저가 {kq['low']:,.2f}")
        elif market: st.error(market.get("error"))

        st.divider()

        # ── 업종 강세/약세 ──
        st.markdown('<p class="zone-header">🏭 오늘 강한 업종 vs 약한 업종</p>', unsafe_allow_html=True)
        if sector and "error" not in sector:
            c1,c2=st.columns(2)
            with c1:
                st.markdown("**강세 TOP3**")
                for name,chg in sector.get("top3",[]):
                    st.markdown(f"{name}: {_cv(chg)}%", unsafe_allow_html=True)
            with c2:
                st.markdown("**약세 BOT3**")
                for name,chg in sector.get("bot3",[]):
                    st.markdown(f"{name}: {_cv(chg)}%", unsafe_allow_html=True)
        elif sector: st.error(sector.get("error"))

        st.divider()

        # ── 수급 오실레이터 ──
        st.markdown('<p class="zone-header">💹 외국인·기관 매매 동향</p>', unsafe_allow_html=True)
        st.caption("코스피 단기 모멘텀 오실레이터 (MA5/MA20 기준) — 양수=단기 상승 추세 / 음수=단기 하락 추세 | 업종별 상대강도로 자금 흐름 확인")
        if supply and "error" not in supply:
            osc=supply["kospi_osc"]
            st.metric(f"{'🟢' if osc>0 else '🔴'} 코스피 기준 오실레이터", f"{osc:+.2f}")
            c1,c2=st.columns(2)
            with c1:
                st.markdown("**수급 강세**")
                for name,rel in supply.get("strong",[]): st.markdown(f"{name}: {_cv(rel)}", unsafe_allow_html=True)
            with c2:
                st.markdown("**수급 약세**")
                for name,rel in supply.get("weak",[]): st.markdown(f"{name}: {_cv(rel)}", unsafe_allow_html=True)
        elif supply: st.error(supply.get("error"))

        st.divider()

        # ── 한국 ETF RS ──
        st.markdown('<p class="zone-header">🇰🇷 한국 ETF 순위</p>', unsafe_allow_html=True)
        st.caption("🟢 강세(70↑) = 지금 자금이 몰리는 ETF / 70점 이상 ETF 업종 위주로 종목 탐색 추천")
        if kr_etf and "error" not in kr_etf:
            show=kr_etf.get("strong") or kr_etf.get("all",[])[:10]
            if show:
                df_kr=pd.DataFrame(show)
                df_kr["강도"]=df_kr["norm_rs"].apply(lambda x:"🟢 강세" if x>=70 else "🟡 보통")
                st.dataframe(df_kr.rename(columns={"name":"ETF명","norm_rs":"RS(0~100)","rs_raw":"KOSPI대비(%)","강도":"강도"})[["ETF명","RS(0~100)","KOSPI대비(%)","강도"]],
                    use_container_width=True, hide_index=True,
                    column_config={"RS(0~100)":st.column_config.ProgressColumn("RS(0~100)",min_value=0,max_value=100,format="%.1f")})
            with st.expander("📊 한국 ETF RS 차트 (KOSPI 대비 초과수익률)"):
                _all=kr_etf.get("all",[])[:25]; _all_s=sorted(_all,key=lambda x:x.get("rs_raw",0))
                _names=[r["name"] for r in _all_s]; _vals=[r.get("rs_raw",0) for r in _all_s]
                _colors=["#00c853" if v>=0 else "#ff4b4b" for v in _vals]
                _fig=go.Figure(go.Bar(x=_vals,y=_names,orientation='h',marker_color=_colors,
                                      text=[f"{v:+.1f}%" for v in _vals],textposition='outside'))
                _mx=max(abs(min(_vals,default=0)),abs(max(_vals,default=10)),10)
                _fig.update_layout(xaxis_range=[-_mx*1.35,_mx*1.35],
                                    xaxis_title="KOSPI 대비 초과수익률 (%)",
                                    height=max(320,len(_names)*26),
                                    margin=dict(l=10,r=70,t=10,b=10),
                                    plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',font_color='#e0e0e0',
                                    dragmode=False)
                _fig.add_vline(x=0,line_dash="solid",line_color="#888888",opacity=0.8)
                st.plotly_chart(_fig, use_container_width=True, config={"scrollZoom": False})
        elif kr_etf: st.error(kr_etf.get("error"))

        st.divider()

        # ── 빈집 주도주 ──
        bz=binzip or {}; bl=bz.get("binzip",[]); ss=" + ".join(bz.get("sectors",[])) or "주도업종"
        st.markdown(f'<p class="zone-header">🏠 아직 덜 오른 주도주 (빈집) [{ss}]</p>', unsafe_allow_html=True)
        st.caption("주도 업종이 올랐는데 아직 소외된 종목 → '빈집'이 채워지며 따라 오를 가능성 있는 종목")
        if bl:
            df_bz=pd.DataFrame(bl)[["name","code","price","rs60","rel20"]]
            df_bz.columns=["종목명","코드","현재가","60일RS(%)","20일눌림(%)"]
            df_bz["현재가"]=df_bz["현재가"].apply(lambda x:f"{x:,}원")
            df_bz["60일RS(%)"]=df_bz["60일RS(%)"].apply(lambda x:f"+{x:.1f}%")
            df_bz["20일눌림(%)"]=df_bz["20일눌림(%)"].apply(lambda x:f"{x:.1f}%")
            st.dataframe(df_bz, use_container_width=True, hide_index=True)
            st.caption(f"{bz.get('scanned',0)}종목 스캔 / 빈집 {len(bl)}개")
        elif "error" in bz: st.error(f"오류: {bz['error']}")
        else: st.info(f"빈집 조건 충족 종목 없음 ({bz.get('scanned',0)}종목 스캔)")

    st.divider()

    # ── 한국 개별종목 RS ──
    st.markdown('<p class="zone-header">📈 강한 종목 순위</p>', unsafe_allow_html=True)
    st.caption("코스피·코스닥 전체 대비 더 많이 오른 종목 순위 — 70점 이상이면 강세")
    _rs_src_label = "📡 yfinance 자동"
    if "c_rs_bytes" in st.session_state:
        _rs_src_label = f"📂 로컬 파일 {'(사이드바)' if os.path.exists(_LOCAL_FILES['rs_stock']) else '(업로드됨)'}"
    with st.expander("📂 파일 직접 업로드 (선택 — 로컬 파일이 없을 때)", expanded=False):
        rs_xl_file = st.file_uploader(
            "종목상대강도데이터.xlsx", type=["xlsx"], key="rs_xl_file",
            help="업로드 시 Yahoo Finance 대신 로컬 Excel 종가 데이터로 RS 계산 (빠르고 정확)"
        )
        if rs_xl_file:
            rs_xl_file.seek(0); st.session_state["c_rs_bytes"] = rs_xl_file.read()
    st.caption(f"데이터 소스: {_rs_src_label}")
    if st.button("▶ 개별종목 RS 스크리닝", key="kr_stock_rs_run", use_container_width=True):
        if "c_rs_bytes" in st.session_state:
            _xl_rs = pd.ExcelFile(io.BytesIO(st.session_state["c_rs_bytes"]), engine="openpyxl")
            _rs_sn = next((s for s in _xl_rs.sheet_names if '종가' in str(s) or 'close' in str(s).lower()), _xl_rs.sheet_names[0])
            _df_rs_close = pd.read_excel(io.BytesIO(st.session_state["c_rs_bytes"]), sheet_name=_rs_sn, engine="openpyxl")
            with st.spinner(f"Excel 로컬 데이터로 RS 계산 중 ({len(_df_rs_close.columns)-2}종목, 시트:{_rs_sn})..."):
                _tmp_rs = calc_kr_stock_rs_excel(_df_rs_close, top_n=15)
            if "error" not in _tmp_rs:
                st.session_state["c_kr_rs"] = _tmp_rs
                st.session_state["c_kr_rs_src"] = f"📂 종목상대강도데이터.xlsx | {len(_df_rs_close)}행 × {len(_df_rs_close.columns)-2}종목"
            else:
                st.error(_tmp_rs["error"])
        else:
            with st.spinner(f"한국 개별종목 RS 자동 계산 중 ({len(KR_STOCKS)}종목)..."):
                _tmp_rs = get_kr_stock_rs_auto(top_n=15)
            if "error" not in _tmp_rs:
                st.session_state["c_kr_rs"] = _tmp_rs
                st.session_state["c_kr_rs_src"] = _tmp_rs.get("source", "📡 yfinance 자동")
            else:
                st.error(_tmp_rs["error"])
    if "c_kr_rs" in st.session_state:
        kr_rs = st.session_state["c_kr_rs"]
        _is_cached = not rs_xl_file and "c_rs_bytes" in st.session_state
        st.caption(st.session_state.get("c_kr_rs_src", "") + ("  ⚡ 캐시" if _is_cached else ""))
        show = kr_rs.get("strong") or kr_rs.get("all", [])[:15]
        if show:
            df_rs = pd.DataFrame(show)
            st.dataframe(df_rs.rename(columns={"ticker":"티커","name":"종목명","norm_rs":"RS(0~100)","rs_raw":"RS원시값"})[["티커","종목명","RS(0~100)","RS원시값"]],
                use_container_width=True, hide_index=True,
                column_config={"RS(0~100)":st.column_config.ProgressColumn("RS(0~100)",min_value=0,max_value=100,format="%.1f")})
        st.caption(f"전체 {len(kr_rs.get('all',[]))}종목 스캔 / RS≥70 강세 {len(kr_rs.get('strong',[]))}종목")
        with st.expander("📊 개별종목 RS 차트 (KOSPI 대비 초과수익률)"):
            _show=kr_rs.get("strong") or kr_rs.get("all",[])[:20]
            _show_s=sorted(_show,key=lambda x:x["rs_raw"])
            _names=[r["name"] for r in _show_s]; _vals=[r["rs_raw"] for r in _show_s]
            _colors=["#00c853" if v>=0 else "#ff4b4b" for v in _vals]
            _fig=go.Figure(go.Bar(x=_vals,y=_names,orientation='h',marker_color=_colors,
                                  text=[f"{v:+.1f}%" for v in _vals],textposition='outside'))
            _max_abs=max(abs(min(_vals,default=0)),abs(max(_vals,default=10)),10)
            _fig.update_layout(xaxis_range=[-_max_abs*1.35,_max_abs*1.35],
                                xaxis_title="KOSPI 대비 초과수익률 (%)",
                                height=max(300,len(_names)*28),
                                margin=dict(l=10,r=70,t=10,b=10),
                                plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',font_color='#e0e0e0',
                                dragmode=False)
            _fig.add_vline(x=0,line_dash="solid",line_color="#888888",opacity=0.8)
            st.plotly_chart(_fig, use_container_width=True, config={"scrollZoom": False})

    st.divider()

    # ── 한국 ETF RS ──
    st.markdown('<p class="zone-header">📊 ETF 정밀 순위</p>', unsafe_allow_html=True)
    st.caption("Excel 파일 우선 사용 (더 정확) / 없으면 yfinance 자동 수집")
    _etf_src_label = "📡 yfinance 자동"
    if "c_etf_xl_bytes" in st.session_state:
        _etf_src_label = f"📂 로컬 파일 {'(사이드바)' if os.path.exists(_LOCAL_FILES['rs_etf']) else '(업로드됨)'}"
    with st.expander("📂 파일 직접 업로드 (선택 — 로컬 파일이 없을 때)", expanded=False):
        etf_rs_xl_file = st.file_uploader(
            "etf상대강도데이터.xlsx", type=["xlsx"], key="etf_rs_xl_file",
            help="업로드 시 로컬 Excel 데이터 우선 사용 / 없으면 yfinance로 자동 계산"
        )
        if etf_rs_xl_file:
            etf_rs_xl_file.seek(0); st.session_state["c_etf_xl_bytes"] = etf_rs_xl_file.read()
    st.caption(f"데이터 소스: {_etf_src_label}")
    if st.button("▶ ETF RS 스크리닝", key="kr_etf_rs_run", use_container_width=True):
        if "c_etf_xl_bytes" in st.session_state:
            try:
                _etf_xl = pd.ExcelFile(io.BytesIO(st.session_state["c_etf_xl_bytes"]), engine="openpyxl")
                _etf_sn = next((s for s in _etf_xl.sheet_names if '데이터' in s), _etf_xl.sheet_names[0])
                df_etf_rs = pd.read_excel(io.BytesIO(st.session_state["c_etf_xl_bytes"]), sheet_name=_etf_sn, engine="openpyxl")
                with st.spinner(f"ETF RS 계산 중 ({len(df_etf_rs.columns)-2}개 ETF)..."):
                    _tmp_etf = calc_kr_etf_rs_excel(df_etf_rs, top_n=15)
                if "error" not in _tmp_etf:
                    st.session_state["c_etf_rs"] = _tmp_etf
                    st.session_state["c_etf_rs_meta"] = f"📂 시트: {_etf_sn} | {len(df_etf_rs)}행 | {len(_tmp_etf.get('all',[]))}개 ETF | RS≥70: {len([r for r in _tmp_etf.get('all',[]) if r['norm_rs']>=70])}개"
                else:
                    st.error(_tmp_etf["error"])
            except Exception as e:
                st.error(f"ETF RS 파일 읽기 오류: {e}")
        else:
            with st.spinner(f"ETF RS 자동 계산 중 ({len(KR_ETF_CODES)}개 ETF)..."):
                _tmp_etf = get_kr_etf_rs_auto(top_n=15)
            if "error" not in _tmp_etf:
                st.session_state["c_etf_rs"] = _tmp_etf
                st.session_state["c_etf_rs_meta"] = _tmp_etf.get("source", "📡 yfinance 자동") + f" | {len(_tmp_etf.get('all',[]))}개 | RS≥70: {len([r for r in _tmp_etf.get('all',[]) if r['norm_rs']>=70])}개"
            else:
                st.error(_tmp_etf["error"])
    if "c_etf_rs" in st.session_state:
        etf_rs_xl = st.session_state["c_etf_rs"]
        _is_etf_cached = not etf_rs_xl_file
        st.caption(st.session_state.get("c_etf_rs_meta", "") + ("  ⚡ 캐시" if _is_etf_cached else ""))
        show_etf = etf_rs_xl.get("strong") or etf_rs_xl.get("all", [])[:15]
        if show_etf:
            df_etf_show = pd.DataFrame(show_etf)
            st.dataframe(
                df_etf_show.rename(columns={"name":"ETF명","norm_rs":"RS(0~100)","rs_raw":"KOSPI대비(%)"})[["ETF명","RS(0~100)","KOSPI대비(%)"]],
                use_container_width=True, hide_index=True,
                column_config={"RS(0~100)":st.column_config.ProgressColumn("RS(0~100)",min_value=0,max_value=100,format="%.1f")}
            )
        with st.expander("📊 ETF RS 차트 (KOSPI 대비 초과수익률)"):
            _all_etf = etf_rs_xl.get("all", [])[:25]
            _all_etf_s = sorted(_all_etf, key=lambda x: x["rs_raw"])
            _names_e = [r["name"] for r in _all_etf_s]
            _vals_e = [r["rs_raw"] for r in _all_etf_s]
            _colors_e = ["#00c853" if v>=0 else "#ff4b4b" for v in _vals_e]
            _fig_e = go.Figure(go.Bar(x=_vals_e, y=_names_e, orientation='h', marker_color=_colors_e,
                                     text=[f"{v:+.1f}%" for v in _vals_e], textposition='outside'))
            _max_e = max(abs(min(_vals_e,default=0)),abs(max(_vals_e,default=10)),10)
            _fig_e.update_layout(xaxis_range=[-_max_e*1.35,_max_e*1.35],
                                 xaxis_title="KOSPI 대비 초과수익률 (%)",
                                 height=max(320, len(_names_e)*26),
                                 margin=dict(l=10,r=70,t=10,b=10),
                                 plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#e0e0e0',
                                 dragmode=False)
            _fig_e.add_vline(x=0, line_dash="solid", line_color="#888888", opacity=0.8)
            st.plotly_chart(_fig_e, use_container_width=True, config={"scrollZoom": False})

    st.divider()

    # ── 한국 F&G 오실레이터 ──
    st.markdown('<p class="zone-header">😨 한국 시장 심리 — 공포인가 탐욕인가</p>', unsafe_allow_html=True)
    st.caption("양수(+) = 탐욕 → 과열 주의 | 음수(-) = 공포 → 매수 기회일 수 있음")
    st.caption("자동(참고용): 방향성만 확인  |  Excel(정밀): VKOSPI·국채선물 원본 데이터 분석")
    if st.button("▶ 자동 계산 〔참고용〕", key="kr_fg_auto_run", use_container_width=True, type="primary"):
            with st.spinner("한국 F&G 오실레이터 자동 계산 중..."):
                kr_fg_auto = get_kr_fg_auto()
            if "error" in kr_fg_auto:
                st.error(kr_fg_auto["error"])
            else:
                st.caption(f"기준일: {kr_fg_auto['date']}  |  {kr_fg_auto.get('source','')}")
                for label, v in kr_fg_auto["results"].items():
                    st.markdown(f"**{label}**")
                    c1, c2 = st.columns(2)
                    c1.metric("오실레이터", v["osc"], f"{'🟢' if v['osc']>0 else '🔴'} {v['sentiment']}")
                    c2.metric("임펄스", v["impulse"])
                    st.metric("TD 매도/매수", f"{v['td_sell']} / {v['td_buy']}")
                    if v.get("chart"):
                        with st.expander(f"📈 {label} 오실레이터 차트"):
                            ch = v["chart"]
                            _fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.45,0.55],
                                                 vertical_spacing=0.06, subplot_titles=("지수 가격","F&G 오실레이터"))
                            _fig.add_trace(go.Scatter(x=ch["dates"], y=ch["price"], name="지수",
                                                       line=dict(color="#E0E0E0",width=1.5)), row=1, col=1)
                            _osc_colors = ["#00c853" if v2>0 else "#ff4b4b" for v2 in ch["osc"]]
                            _fig.add_trace(go.Bar(x=ch["dates"], y=ch["osc"], name="Oscillator",
                                                   marker_color=_osc_colors), row=2, col=1)
                            _fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)", row=2, col=1)
                            _fig.update_layout(height=400, margin=dict(l=10,r=20,t=40,b=10),
                                               plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                                               font_color='#e0e0e0', showlegend=False, dragmode=False)
                            _fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.08)')
                            _fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.08)')
                            st.plotly_chart(_fig, use_container_width=True, config={"scrollZoom": False})
    _fg_src = "📂 로컬 파일 감지됨" if "c_fg_bytes" in st.session_state else "❌ 파일 없음 (auto만 가능)"
    st.caption(f"정밀 분석 데이터 소스: {_fg_src}")
    with st.expander("📂 파일 직접 업로드 (선택 — 로컬 파일이 없을 때)", expanded=False):
        fg_file = st.file_uploader("피어앤그리드.xlsx", type=["xlsx"], key="kr_fg_file")
    if fg_file:
        fg_file.seek(0); st.session_state["c_fg_bytes"] = fg_file.read()
    _fg_bytes_to_use = st.session_state.get("c_fg_bytes")
    if _fg_bytes_to_use is not None:
        try:
            _fg_bytes = _fg_bytes_to_use
            df_kp = pd.read_excel(io.BytesIO(_fg_bytes), sheet_name="KOSPI", engine="openpyxl")
            df_kq = pd.read_excel(io.BytesIO(_fg_bytes), sheet_name="KOSDAQ", engine="openpyxl")
            _df_call_oi = _df_put_oi = None
            try:
                _xl_sheets = pd.ExcelFile(io.BytesIO(_fg_bytes), engine="openpyxl").sheet_names
                _call_sn = next((s for s in _xl_sheets if '콜' in s), None)
                _put_sn  = next((s for s in _xl_sheets if '풋' in s), None)
                if _call_sn and _put_sn:
                    _df_call_oi = pd.read_excel(io.BytesIO(_fg_bytes), sheet_name=_call_sn, engine="openpyxl")
                    _df_put_oi  = pd.read_excel(io.BytesIO(_fg_bytes), sheet_name=_put_sn,  engine="openpyxl")
                    st.caption(f"✅ 콜/풋 시트 감지 ({_call_sn} / {_put_sn}) → 외국인 P/C 비율 정밀 계산 적용")
            except: pass
            _tmp_fg = calc_kr_fg_excel(df_kp, df_kq, _df_call_oi, _df_put_oi)
            if "error" not in _tmp_fg:
                st.session_state["c_kr_fg"] = _tmp_fg
            else:
                st.error(_tmp_fg["error"])
        except Exception as e:
            st.error(f"파일 읽기 오류: {e}")
    if "c_kr_fg" in st.session_state:
        kr_fg = st.session_state["c_kr_fg"]
        st.markdown(f"**기준일: {kr_fg['date']}**")
        c1,c2 = st.columns(2)
        c1.metric("코스피 오실레이터", kr_fg["kospi_osc"], f"{'🟢' if kr_fg['kospi_osc']>0 else '🔴'} {kr_fg['kospi_sentiment']}")
        c2.metric("코스닥 오실레이터", kr_fg["kosdaq_osc"], f"{'🟢' if kr_fg['kosdaq_osc']>0 else '🔴'} {kr_fg['kosdaq_sentiment']}")
        c1.metric("코스피 임펄스", kr_fg["kospi_impulse"]); c2.metric("코스닥 임펄스", kr_fg["kosdaq_impulse"])
        c1.metric("코스피 TD 매도/매수", f"{kr_fg['kospi_td_sell']} / {kr_fg['kospi_td_buy']}")
        c2.metric("코스닥 TD 매도/매수", f"{kr_fg['kosdaq_td_sell']} / {kr_fg['kosdaq_td_buy']}")
        if kr_fg.get("kospi_gap") is not None:
            c1.metric("코스피 SuperMA 이격", f"{kr_fg['kospi_gap']:+.2f}%", help="SuperMA = (MA20+MA60+MA120+MA200)/4 기준 이격")
        if kr_fg.get("kosdaq_gap") is not None:
            c2.metric("코스닥 SuperMA 이격", f"{kr_fg['kosdaq_gap']:+.2f}%")
        if kr_fg.get("chart"):
            with st.expander("📈 F&G 오실레이터 차트 (최근 6개월)"):
                ch=kr_fg["chart"]
                _fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.5,0.5],vertical_spacing=0.06,
                                   subplot_titles=("코스피 오실레이터","코스닥 오실레이터"))
                _fig.add_trace(go.Scatter(x=ch["dates"],y=ch["kospi_osc"],name="KOSPI Osc",
                                           line=dict(color="#6A5ACD",width=2),fill='tozeroy',
                                           fillcolor='rgba(106,90,205,0.15)'),row=1,col=1)
                _fig.add_trace(go.Scatter(x=ch.get("kosdaq_dates",ch["dates"]),y=ch.get("kosdaq_osc",[]),
                                           name="KOSDAQ Osc",line=dict(color="#00B3B3",width=2),
                                           fill='tozeroy',fillcolor='rgba(0,179,179,0.15)'),row=2,col=1)
                _fig.add_hline(y=0,line_dash="dash",line_color="rgba(255,255,255,0.3)",row=1,col=1)
                _fig.add_hline(y=0,line_dash="dash",line_color="rgba(255,255,255,0.3)",row=2,col=1)
                _fig.update_layout(height=420,margin=dict(l=10,r=20,t=40,b=10),
                                    plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',
                                    font_color='#e0e0e0',showlegend=False,dragmode=False)
                _fig.update_xaxes(showgrid=True,gridcolor='rgba(255,255,255,0.08)')
                _fig.update_yaxes(showgrid=True,gridcolor='rgba(255,255,255,0.08)')
                st.plotly_chart(_fig, use_container_width=True, config={"scrollZoom": False})

    st.divider()

    # ── 수급 자동 스크리닝 ──
    st.markdown('<p class="zone-header">📡 외국인 매매 동향</p>', unsafe_allow_html=True)
    st.caption("외국인 순매수 누적 | 기관 데이터는 무료소스 없음 — 방향성만 참고")
    col_sa, col_sb = st.columns([1,2])
    with col_sa:
        supply_days = st.selectbox("집계 기간", [10,20,40], index=1, key="supply_days_sel")
    with col_sb:
        st.caption(f"최근 {supply_days}거래일 외국인 순매수량×종가 누적 | 네이버 파이낸스")
    if st.button("▶ 수급 자동 스크리닝 〔참고용〕", key="kr_supply_auto_run", use_container_width=True):
        with st.spinner(f"한국 주요 종목 {len(KR_STOCKS)}개 수급 데이터 수집 중... (40~70초)"):
            sup = get_kr_supply_auto(top_n=20, days=supply_days)
        if "error" in sup:
            st.error(f"수집 오류: {sup['error']}")
        else:
            st.caption(f"기준일: {sup['date']} | 스캔 종목: {sup['scanned']}개 | {sup['note']}")

            def _draw_supply_osc(row, label_prefix=""):
                """수급 오실레이터 차트 — 외국인/기관 2행 + 종가(우축)"""
                ch = row.get("chart")
                if not ch: return
                fval = row.get("frgn_nd_bil", row.get("net_nd_bil", 0))
                ival = row.get("inst_nd_bil", 0)
                fs = "+" if fval >= 0 else ""; is_ = "+" if ival >= 0 else ""
                with st.expander(f"📊 {label_prefix}{row['name']} ({row['ticker']}) — 외국인 {fs}{fval:.1f}억 / 기관 {is_}{ival:.1f}억"):
                    fig_s = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                          subplot_titles=("외국인 순매매(억원)","기관 순매매(억원)"),
                                          vertical_spacing=0.1,
                                          specs=[[{"secondary_y":True}],[{"secondary_y":True}]])
                    for row_i, (dk, mk, color, label) in enumerate([
                        ("frgn_daily","frgn_ma5","#26a69a","외국인"),
                        ("inst_daily","inst_ma5","#ef5350","기관"),
                    ], start=1):
                        bar_c = ["#00c853" if v>=0 else "#ff4b4b" for v in ch.get(dk,[])]
                        fig_s.add_trace(go.Bar(x=ch["dates"],y=ch.get(dk,[]),
                            name=f"{label} 일별",marker_color=bar_c,opacity=0.7,
                            showlegend=(row_i==1)),row=row_i,col=1,secondary_y=False)
                        fig_s.add_trace(go.Scatter(x=ch["dates"],y=ch.get(mk,[]),
                            name=f"{label} MA5",line=dict(color=color,width=2),
                            showlegend=(row_i==1)),row=row_i,col=1,secondary_y=False)
                        fig_s.add_trace(go.Scatter(x=ch["dates"],y=ch["price"],
                            name="종가",line=dict(color="#E0E0E0",width=1.5),
                            showlegend=(row_i==1)),row=row_i,col=1,secondary_y=True)
                    fig_s.update_layout(height=480,margin=dict(l=10,r=60,t=40,b=10),
                        plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",
                        font_color="#e0e0e0",barmode="overlay",
                        legend=dict(orientation="h",y=1.05),dragmode=False)
                    fig_s.update_yaxes(showgrid=True,gridcolor="rgba(255,255,255,0.08)")
                    fig_s.update_xaxes(showgrid=True,gridcolor="rgba(255,255,255,0.08)")
                    st.plotly_chart(fig_s, use_container_width=True, config={"scrollZoom": False})

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**🟢 외국인 순매수 TOP20**")
                for i, row in enumerate(sup["top20"], 1):
                    val = row.get("frgn_nd_bil", row.get("net_nd_bil", 0))
                    ival = row.get("inst_nd_bil", 0)
                    sign = "+" if val >= 0 else ""
                    st.markdown(f"`{i:2d}` **{row['name']}** — 외인 {sign}{val:.1f}억 / 기관 {'+' if ival>=0 else ''}{ival:.1f}억")
            with c2:
                st.markdown("**🔴 외국인 순매도 TOP20**")
                for i, row in enumerate(sup["worst20"], 1):
                    val = row.get("frgn_nd_bil", row.get("net_nd_bil", 0))
                    ival = row.get("inst_nd_bil", 0)
                    sign = "+" if val >= 0 else ""
                    st.markdown(f"`{i:2d}` **{row['name']}** — 외인 {sign}{val:.1f}억 / 기관 {'+' if ival>=0 else ''}{ival:.1f}억")

            st.markdown("#### 📈 수급 오실레이터 — 순매수 TOP10")
            for row in sup["top20"][:10]:
                _draw_supply_osc(row, "🟢 ")
            st.markdown("#### 📉 수급 오실레이터 — 순매도 TOP10")
            for row in sup["worst20"][:10]:
                _draw_supply_osc(row, "🔴 ")

    st.divider()

    # ── KRX 기관 순매수 + 거래대금 강도 ──
    st.markdown('<p class="zone-header">🏦 기관 매매 동향</p>', unsafe_allow_html=True)
    st.caption("KOSPI/KOSDAQ 전체 기관·외국인 순매수 + 거래대금 강도 | 출처: KRX")
    if st.button("▶ 기관 수급 + 거래대금 강도 조회 (KRX)", key="kr_inst_flow_run", use_container_width=True):
        with st.spinner("KRX 데이터 수집 중 (10~20초)..."):
            inst_flow = get_krx_inst_market_flow(days=10)
            vol_str = get_krx_volume_strength()
        if "error" not in vol_str:
            c1, c2 = st.columns(2)
            c1.metric("KOSPI 거래대금 (오늘)", f"{vol_str['today_tril']:.2f}조")
            c2.metric("20일 평균", f"{vol_str['ma20_tril']:.2f}조")
            st.metric("거래대금 강도", f"{vol_str['rotation_rate']:.0f}%", vol_str["level"])
            if vol_str.get("chart"):
                ch = vol_str["chart"]
                fig_tv = go.Figure()
                fig_tv.add_trace(go.Bar(x=ch["dates"], y=ch["values"], name="일별 거래대금",
                                        marker_color="#6A5ACD", opacity=0.7))
                fig_tv.add_trace(go.Scatter(x=ch["dates"], y=ch["ma20"], name="MA20",
                                            line=dict(color="#FF8C00", width=2, dash="dot")))
                fig_tv = _chart_layout(fig_tv, height=220)
                fig_tv.update_yaxes(title_text="거래대금(조원)")
                st.plotly_chart(fig_tv, use_container_width=True, config={"scrollZoom": False})
        else:
            st.caption(f"거래대금 강도: {vol_str['error']}")
        if "error" not in inst_flow:
            st.caption(f"기준: {inst_flow['date']} | 최근 {inst_flow['days']}거래일")
            for mkt, d in inst_flow["data"].items():
                ic = "🟢" if d["inst_sum_bil"] >= 0 else "🔴"
                fc = "🟢" if d.get("frgn_sum_bil", 0) >= 0 else "🔴"
                st.markdown(f"**{mkt}** | {ic} 기관합계: {d['inst_sum_bil']:+,.0f}억 | {fc} 외국인: {d.get('frgn_sum_bil', 0):+,.0f}억")
                if d["inst"]:
                    fig_inst = go.Figure()
                    bc_i = ["#00c853" if v >= 0 else "#ff4b4b" for v in d["inst"]]
                    fig_inst.add_trace(go.Bar(x=d["dates"], y=[v/1e8 for v in d["inst"]],
                                              name="기관합계", marker_color=bc_i, opacity=0.8))
                    if d.get("frgn"):
                        bc_f = ["#6A5ACD" if v >= 0 else "#B03A2E" for v in d["frgn"]]
                        fig_inst.add_trace(go.Bar(x=d["dates"], y=[v/1e8 for v in d["frgn"]],
                                                  name="외국인", marker_color=bc_f, opacity=0.6))
                    fig_inst = _chart_layout(fig_inst, height=200)
                    fig_inst.update_layout(barmode="group", yaxis_title="순매수(억원)")
                    fig_inst.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
                    st.plotly_chart(fig_inst, use_container_width=True, config={"scrollZoom": False})
        else:
            st.caption(f"기관 순매수: {inst_flow['error']}")

    st.divider()

    # ── 컨센 가속 자동 ──
    st.markdown('<p class="zone-header">🤖 전문가 전망 순위</p>', unsafe_allow_html=True)
    st.caption("증권사 전망이 좋아지는 종목 순위 — EPS성장률·매수비율·목표가 인상 합산 (참고용)")
    if st.button("▶ 컨센서스 스크리닝 〔참고용〕", key="kr_consensus_auto_run", use_container_width=True):
        with st.spinner(f"한국 주요 종목 {len(KR_STOCKS)}개 컨센서스 수집 중... (60~90초)"):
            cons_auto = get_kr_consensus_auto(top_n=20)
        if "error" in cons_auto:
            st.error(f"수집 오류: {cons_auto['error']}")
        else:
            st.caption(f"기준일: {cons_auto['date']} | 스캔: {cons_auto['scanned']}개 | {cons_auto['note']}")
            if cons_auto.get("accel"):
                st.markdown("#### 🎯 컨센 가속 종목 (EPS성장↑ + 매수↑ + TP인상↑)")
                for i, row in enumerate(cons_auto["accel"], 1):
                    eps_g = f"{row['eps_growth']:+.0f}%" if row['eps_growth'] is not None else "N/A"
                    st.markdown(
                        f"`{i:2d}` **{row['name']}** ({row['ticker']}) — "
                        f"EPS성장:{eps_g} | 매수비율:{row['buy_pct']:.0f}% | "
                        f"TP인상:{row['tp_raise_pct']:.0f}% | 애널:{row['analyst_count']}명"
                    )
            else:
                st.info("컨센 가속 조건(EPS+·매수≥60%·TP인상≥50%) 충족 종목 없음")
            with st.expander("📊 전체 TOP20 스코어 보기"):
                for i, row in enumerate(cons_auto["top20"], 1):
                    eps_g = f"{row['eps_growth']:+.0f}%" if row['eps_growth'] is not None else "N/A"
                    cons_tp_str = f"{row['cons_tp']:,}원" if row['cons_tp'] else "N/A"
                    st.markdown(
                        f"`{i:2d}` **{row['name']}** — 스코어:{row['score']} | "
                        f"EPS성장:{eps_g} | TP:{cons_tp_str} | "
                        f"매수:{row['buy_pct']:.0f}% | TP인상:{row['tp_raise_pct']:.0f}%"
                    )

    st.divider()

    # ── 컨센 가속 & 수급 (Excel 정밀 분석) ──
    st.markdown('<p class="zone-header">📋 Excel 정밀 분석</p>', unsafe_allow_html=True)
    _cons_src = "📂 로컬 파일 감지됨" if "c_consensus_bytes" in st.session_state else "❌ 파일 없음"
    st.caption(f"📂 데이터 정리.xlsx (db 시트) — EPS가속(1M>3M) + 빈집여부(외국인+기관 합산 순매수) | 소스: {_cons_src}")
    with st.expander("📂 파일 직접 업로드 (선택)", expanded=False):
        consensus_file = st.file_uploader("데이터 정리.xlsx", type=["xlsx"], key="consensus_file")
    if consensus_file:
        consensus_file.seek(0); st.session_state["c_consensus_bytes"] = consensus_file.read()
    _cons_bytes = st.session_state.get("c_consensus_bytes")
    if _cons_bytes:
        try:
            df_db = pd.read_excel(io.BytesIO(_cons_bytes), sheet_name="db", engine="openpyxl")
            _tmp_cons = calc_consensus_excel(df_db)
            if "error" not in _tmp_cons:
                st.session_state["c_consensus"] = _tmp_cons
            else:
                st.error(_tmp_cons["error"])
        except Exception as e:
            st.error(f"파일 읽기 오류: {e}")
    if "c_consensus" in st.session_state:
        cons = st.session_state["c_consensus"]
        st.caption("📌 빈집 개념: 외국인+기관 합산 순매수 **낮을수록(음수)** = 아직 덜 샀음 = 매수 기회 ↑")
        st.metric("EPS 가속 통과 종목", f"{cons['eps_passed']}개")
        # 빈집 전환 (최우선 — 장기 빈집인데 최근 매집 시작)
        turn = cons.get("turn_list", [])
        if turn:
            st.markdown("**🔄 빈집 전환 종목** — 장기 비어있다가 최근 매집 시작 (최우선)")
            st.success(" · ".join(turn[:15]) + (f" 외 {len(turn)-15}개" if len(turn)>15 else ""))
        else:
            st.info("빈집 전환 종목 없음")
        c1, c2 = st.columns(2)
        with c1:
            bz = cons.get("binzip_list", [])
            st.markdown(f"**🏚️ 빈집 + EPS가속** ({len(bz)}개) — 수급 빈 채로 실적 개선 중")
            for i, n in enumerate(bz[:20], 1): st.markdown(f"`{i:2d}` {n}")
            if len(bz) > 20: st.caption(f"... 외 {len(bz)-20}개")
        with c2:
            inf = cons.get("inflow_list", [])
            st.markdown(f"**📈 수급 유입 중** (TOP{len(inf)}) — 이미 사는 중 (모멘텀 참고)")
            for i, n in enumerate(inf, 1): st.markdown(f"`{i:2d}` {n}")

# ─────────────────────────────────────────────────────────────────────────────
# 탭 3: 종목 분석
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.info("**관심 종목을 직접 분석합니다.** 아래에 종목명이나 티커(예: 삼성전자, NVDA)를 입력하고 분석 버튼을 누르세요.")
    with st.expander("❓ 어려운 용어 설명"):
        st.markdown("**💧 CMF(자금흐름)** — 양수=자금 유입(매수 우세) / 음수=자금 유출(매도 우세)")
        st.markdown("**⚡ 임펄스** — 단기 추세. 🟢강세=주가+MACD 모두 상승 / 🔴약세=모두 하락")
        st.markdown("**🔢 TD 카운트** — 패턴 카운트 1~9. 9에 가까울수록 추세 전환 주의")
        st.markdown("**📏 MA10(10주 이평)** — 최근 10주 평균가. 현재가가 이 위에 있으면 중기 상승추세")
        st.markdown("**📉 시가→저가 낙폭** — 장 시작 후 최대 낙폭 평균. 이 수치만큼 낮게 지정가 설정 참고")
        st.markdown("**📈 고가→종가 하락** — 장중 최고점→종가 평균 낙폭. 고점 추격매수 위험 정도")
        st.markdown("**💹 수급 오실레이터** — 낮을수록(빈집) 매집 여력 큼. 음수/낮음=🏚️빈집(좋음) / 높음=이미 매집됨(주의)")
        st.markdown("**📋 컨센서스** — 증권사 실적 전망치. 전망이 올라가는 종목이 주가도 오르는 경향")

    st.markdown('<p class="zone-header">🎯 언제 사면 좋을까 — 매수 타점</p>', unsafe_allow_html=True)
    st.caption("1년 데이터 기반 — 시가 대비 얼마나 낮게 지정가를 걸면 체결될지 평균 통계")

    # 전종목 dict 로드 (pykrx 24h 캐시, TICKER_NAMES 포함)
    with st.spinner("종목 목록 로딩 중..."):
        _kr_full = _load_all_kr_tickers()  # {ticker: 한글명}
    _all_tickers: dict = {**TICKER_NAMES, **_kr_full}  # 기존 + 전종목 병합
    _all_name_to_ticker = {v.lower(): k for k, v in _all_tickers.items()}

    _sel_opts = [""] + sorted(
        [f"{kr} ({t})" for t, kr in _all_tickers.items()],
        key=lambda x: x[0]
    )
    sel_stock = st.selectbox(
        f"📋 목록에서 선택 ({len(_all_tickers):,}개 전종목 검색 가능)",
        _sel_opts, index=0, key="bt_sel"
    )
    ticker_input = st.text_input(
        "또는 직접 입력 (한글명·6자리코드·티커 모두 가능, 쉼표로 여러 개)",
        placeholder="삼성전자, 005930, NVDA, 005930.KS",
        key="bt_ticker"
    )

    st.divider()
    # 로컬 파일 상태 표시
    _sup_ok = "✅" if "c_supply_bytes" in st.session_state else "❌"
    _trd_ok = "✅" if "c_trading_bytes" in st.session_state else "❌"
    _wk_ok  = "✅" if "c_weekly_bytes"  in st.session_state else "❌"
    st.caption(f"로컬 파일: {_sup_ok} 추세(수급) &nbsp;|&nbsp; {_trd_ok} 거래대금강도 &nbsp;|&nbsp; {_wk_ok} 추세(주간)  — 사이드바 '전체 로드' 버튼으로 자동 로드")
    with st.expander("📂 파일 직접 업로드 (선택 — 로컬 파일이 없을 때)"):
        trend_supply_file = st.file_uploader(
            "① 추세판별기(수급까지체크).xlsx", type=["xlsx"], key="trend_supply_file",
            help="DB(2) 시트 — 5일간 기관 매수수량 오실레이터 추가"
        )
        trading_xl_file = st.file_uploader(
            "② 국장 거래대금 강도.xlsx", type=["xlsx"], key="trading_xl_file",
            help="_RotationRate_ 컬럼 — 거래대금 강도 비교 표시"
        )
        weekly_xl_file = st.file_uploader(
            "③ 추세판별기(주간).xlsx", type=["xlsx"], key="weekly_xl_file",
            help="DB 시트 — 주간 OHLCV로 CMF/임펄스/TD 계산 (HTS 원천 데이터)"
        )
    if trend_supply_file:
        trend_supply_file.seek(0); st.session_state["c_supply_bytes"] = trend_supply_file.read()
    if trading_xl_file:
        trading_xl_file.seek(0); st.session_state["c_trading_bytes"] = trading_xl_file.read()
    if weekly_xl_file:
        weekly_xl_file.seek(0); st.session_state["c_weekly_bytes"] = weekly_xl_file.read()
    _eff_supply = io.BytesIO(st.session_state["c_supply_bytes"]) if "c_supply_bytes" in st.session_state else None
    _eff_trading = io.BytesIO(st.session_state["c_trading_bytes"]) if "c_trading_bytes" in st.session_state else None
    _eff_weekly = io.BytesIO(st.session_state["c_weekly_bytes"]) if "c_weekly_bytes" in st.session_state else None

    # 분석 결과를 session_state에 저장 — 버튼 클릭 후 리렌더링 시에도 차트 유지
    if "tab3_tickers" not in st.session_state:
        st.session_state["tab3_tickers"] = []
    if "tab3_rt_quotes" not in st.session_state:
        st.session_state["tab3_rt_quotes"] = {}

    if st.button("🔍 분석 시작", key="bt_run", type="primary", use_container_width=True):
        _new_tickers = []
        if sel_stock:
            _t = sel_stock.split("(")[-1].rstrip(")")
            _new_tickers.append(_t.strip())
        if ticker_input:
            for raw in [x.strip() for x in ticker_input.split(",") if x.strip()]:
                resolved = NAME_TO_TICKER.get(raw.lower())
                if not resolved:
                    resolved = _all_name_to_ticker.get(raw.lower())
                if not resolved and raw.isdigit() and len(raw) == 6:
                    if (raw + ".KS") in _all_tickers:
                        resolved = raw + ".KS"
                    elif (raw + ".KQ") in _all_tickers:
                        resolved = raw + ".KQ"
                    else:
                        resolved = raw + ".KS"
                if not resolved:
                    resolved = raw
                _new_tickers.append(resolved)
        if not _new_tickers:
            st.warning("종목을 선택하거나 입력해주세요.")
        else:
            _kr_t = [t for t in _new_tickers if t.endswith(".KS") or t.endswith(".KQ")]
            st.session_state["tab3_tickers"] = _new_tickers
            _rt_raw = get_naver_realtime_quote(_kr_t) if _kr_t else {}
            st.session_state["tab3_rt_quotes"] = _rt_raw if "error" not in _rt_raw else {}

    if st.session_state.get("tab3_tickers"):
        tickers_to_run = st.session_state["tab3_tickers"]
        _rt_quotes = st.session_state["tab3_rt_quotes"]
        kr_tickers = [t for t in tickers_to_run if t.endswith(".KS") or t.endswith(".KQ")]

        # ── DART 기업 프로필 ──────────────────────────────────────────────────
        try: _dart_key = st.secrets.get("DART_API_KEY", "")
        except Exception: _dart_key = ""
        st.markdown('<p class="zone-header">🏢 기업 정보</p>', unsafe_allow_html=True)
        if not _dart_key:
            st.caption("DART API 키가 설정되지 않았습니다. Streamlit Secrets에 DART_API_KEY를 추가하면 분기/사업보고서 핵심 내용(사업개요·주요제품·수주현황)을 자동으로 표시합니다.")
        elif not kr_tickers:
            st.caption("DART 조회는 한국 종목(.KS / .KQ)만 지원합니다.")
        else:
            for t in kr_tickers:
                _kr_name = TICKER_NAMES.get(t, "")
                with st.spinner(f"{t} {_kr_name} — DART 보고서 로딩 중..."):
                    _dp = get_dart_profile(t, _dart_key)
                if _dp.get("error"):
                    st.caption(f"⚠️ {t}: DART 조회 실패 — {_dp['error']}")
                    continue
                _rdate = _dp.get("report_date", "")
                _rdate_fmt = f"{_rdate[:4]}-{_rdate[4:6]}-{_rdate[6:]}" if len(_rdate) == 8 else _rdate
                _est = _dp.get("est_dt", "")
                _est_fmt = f"{_est[:4]}-{_est[4:6]}-{_est[6:]}" if len(_est) == 8 else _est
                st.markdown(
                    f"**{_dp['corp_name']}** &nbsp;|&nbsp; 업종코드: `{_dp.get('industry','')}`"
                    f" &nbsp;|&nbsp; 대표: {_dp.get('ceo','')}"
                    f" &nbsp;|&nbsp; 설립: {_est_fmt}"
                    f" &nbsp;|&nbsp; 직원: {_dp.get('empl_no','')}명",
                    unsafe_allow_html=True
                )
                if _dp.get("address"):
                    st.caption(f"📍 {_dp['address']}")
                st.caption(f"[{_dp.get('report_type','')} {_rdate_fmt} 접수]")
                if _dp.get("overview"):
                    with st.expander("📋 사업의 개요", expanded=True):
                        st.write(_dp["overview"])
                else:
                    st.caption("사업의 개요 섹션을 찾지 못했습니다.")
                if _dp.get("products_html"):
                    st.markdown("**주요 제품 및 서비스**")
                    st.markdown(_dp["products_html"], unsafe_allow_html=True)
                if _dp.get("orders_html"):
                    st.markdown("**수주현황**")
                    st.markdown(_dp["orders_html"], unsafe_allow_html=True)
                if _dp.get("sales_html"):
                    st.markdown("**매출현황**")
                    st.markdown(_dp["sales_html"], unsafe_allow_html=True)
                st.divider()

        # ── 종합 차트 (주가 + 수급 오실레이터 + 영업이익) ─────────────────────────
        st.markdown('<p class="zone-header">📊 종합 차트</p>', unsafe_allow_html=True)
        st.caption("📈 주가(캔들+이동평균) · 🏚️ 수급 오실레이터(외국인·기관) · 💰 분기 영업이익 추종")
        for t in tickers_to_run:
            kr = TICKER_NAMES.get(t, "")
            st.markdown(f"#### 📌 **{t}** {kr}")

            # ── 주가 차트 + 영업이익 (좌우 배치) ───────────────────────────────
            _pc_col, _fp_col = st.columns([3, 2])
            with _pc_col:
                st.caption("📈 주가 차트 (캔들 + MA20/60/120/200 + 거래량)")
                with st.spinner("주가 로딩 중..."):
                    _pc = get_stock_price_chart(t, days=180)
                if "error" not in _pc:
                    _fig_pc = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                            row_heights=[0.72, 0.28], vertical_spacing=0.03,
                                            subplot_titles=("", "거래량"))
                    _fig_pc.add_trace(go.Candlestick(
                        x=_pc["dates"], open=_pc["open"], high=_pc["high"],
                        low=_pc["low"], close=_pc["close"], name="주가",
                        increasing_line_color="#00c853", decreasing_line_color="#ff4b4b",
                        showlegend=False), row=1, col=1)
                    for _mak, _mac, _man in [("ma20","#FFD700","MA20"),("ma60","#FF8C00","MA60"),
                                              ("ma120","#87CEEB","MA120"),("ma200","#9370DB","MA200")]:
                        if any(v is not None for v in _pc.get(_mak,[])):
                            _fig_pc.add_trace(go.Scatter(x=_pc["dates"], y=_pc[_mak], name=_man,
                                line=dict(color=_mac, width=1.2), mode="lines"), row=1, col=1)
                    _vol_c = ["#00c853" if _pc["close"][i] >= _pc["open"][i] else "#ff4b4b"
                              for i in range(len(_pc["close"]))]
                    _fig_pc.add_trace(go.Bar(x=_pc["dates"], y=_pc["volume"], name="거래량",
                        marker_color=_vol_c, opacity=0.6, showlegend=False), row=2, col=1)
                    _fig_pc.update_layout(height=490, margin=dict(l=10,r=10,t=25,b=10),
                        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                        font_color='#e0e0e0', showlegend=True,
                        legend=dict(orientation='h', y=1.05),
                        xaxis_rangeslider_visible=False, dragmode=False)
                    _fig_pc.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.08)')
                    _fig_pc.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.08)')
                    st.plotly_chart(_fig_pc, use_container_width=True, config={"scrollZoom": False})
                else:
                    st.warning(f"주가 차트: {_pc['error']}")

            with _fp_col:
                st.caption("💰 분기 영업이익 & 매출 추종 (가이던스)")
                with st.spinner("실적 로딩 중..."):
                    _fp = get_stock_financials(t)
                if "error" not in _fp and _fp.get("quarters"):
                    _qs = _fp["quarters"]
                    _qdates = [q["date"] for q in _qs]
                    _op_vals = [q.get("op_profit") for q in _qs]
                    _rev_vals = [q.get("revenue") for q in _qs]
                    _op_clrs = []
                    for _i, _v in enumerate(_op_vals):
                        if _v is None: _op_clrs.append("#888888")
                        elif _i == 0 or _op_vals[_i-1] is None: _op_clrs.append("#2f81f7")
                        else: _op_clrs.append("#00c853" if _v >= _op_vals[_i-1] else "#ff4b4b")
                    _has_rev = any(v is not None for v in _rev_vals)
                    _fig_fp = make_subplots(rows=2 if _has_rev else 1, cols=1,
                                            shared_xaxes=True,
                                            subplot_titles=(["영업이익(억)", "매출(억)"] if _has_rev else ["영업이익(억)"]),
                                            vertical_spacing=0.1)
                    _fig_fp.add_trace(go.Bar(x=_qdates, y=_op_vals, name="영업이익",
                                             marker_color=_op_clrs), row=1, col=1)
                    _fig_fp.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)", row=1, col=1)
                    if _has_rev:
                        _fig_fp.add_trace(go.Bar(x=_qdates, y=_rev_vals, name="매출",
                                                  marker_color="#2f81f7", opacity=0.7), row=2, col=1)
                    if _fp.get("target_price"):
                        _fig_fp.add_annotation(xref="paper", yref="paper", x=0.98, y=1.08,
                            text=f"목표가 {_fp['target_price']:,.0f}", showarrow=False,
                            font=dict(color="#FFD700", size=10))
                    _fig_fp.update_layout(height=490, margin=dict(l=10,r=10,t=35,b=10),
                        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                        font_color='#e0e0e0', showlegend=True,
                        legend=dict(orientation='h', y=1.06), dragmode=False)
                    _fig_fp.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.08)')
                    _fig_fp.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.08)')
                    st.plotly_chart(_fig_fp, use_container_width=True, config={"scrollZoom": False})
                    _eps_rows = [(q["date"], q["eps"]) for q in _qs if q.get("eps") is not None]
                    if _eps_rows:
                        st.dataframe(pd.DataFrame(_eps_rows, columns=["분기","EPS(원)"]),
                                     use_container_width=True, hide_index=True)
                else:
                    st.caption(f"실적 데이터: {_fp.get('error','없음')} (국내 종목은 yfinance 제공 한계 있음)")

            # ── 수급 오실레이터 (한국 종목만) ───────────────────────────────────
            if t.endswith(".KS") or t.endswith(".KQ"):
                st.caption("🏚️ 수급 오실레이터 — 기관+외국인 합산 순매매 (낮을수록=빈집=매집 여력 큼)")
                with st.spinner("수급 데이터 로딩 중..."):
                    _so_c = get_stock_supply_osc(t, chart_days=60, agg_days=20)
                if "error" not in _so_c:
                    _so_total = _so_c["frgn_agg_bil"] + _so_c["inst_agg_bil"]
                    if _so_total <= 0:
                        st.markdown('<div class="sig-green">🏚️ 빈집 — 최근 20일 외국인·기관 순매도/미매집. 매집 여력 큼 (좋은 신호)</div>', unsafe_allow_html=True)
                    elif _so_total <= 100:
                        st.markdown('<div class="sig-yellow">🟡 중립 — 수급 보통 구간. 추세 확인 필요</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="sig-red">⚠️ 수급 포화 — 최근 20일 이미 많이 매집됨. 추가 상승 여지 주의</div>', unsafe_allow_html=True)
                    _ch_c = _so_c["chart"]
                    # HTS 스타일: 기관+외국인 합산 오실레이터 (위) + 주가 (아래)
                    _comb_daily = [f + i for f, i in zip(_ch_c["frgn_daily"], _ch_c["inst_daily"])]
                    _comb_ma5   = pd.Series(_comb_daily).rolling(5, min_periods=1).mean().tolist()
                    _comb_cum20 = pd.Series(_comb_daily).rolling(20, min_periods=1).sum().tolist()
                    _sc_bclrs = ["#00c853" if v >= 0 else "#ff4b4b" for v in _comb_daily]
                    _fig_sc = make_subplots(
                        rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=("합산 수급 오실레이터(억원)", "기관 단독", "외국인 단독"),
                        row_heights=[0.45, 0.275, 0.275], vertical_spacing=0.05
                    )
                    # Row1: 합산 오실레이터
                    _fig_sc.add_trace(go.Bar(x=_ch_c["dates"], y=_comb_daily,
                        name="합산 순매수", marker_color=_sc_bclrs, opacity=0.75), row=1, col=1)
                    _fig_sc.add_trace(go.Scatter(x=_ch_c["dates"], y=_comb_ma5,
                        name="MA5", line=dict(color="#FFD700", width=2)), row=1, col=1)
                    _fig_sc.add_trace(go.Scatter(x=_ch_c["dates"], y=_comb_cum20,
                        name="20일누적", line=dict(color="#FF8C00", width=1.5, dash="dot")), row=1, col=1)
                    _fig_sc.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.4)", row=1, col=1)
                    # Row2: 기관 단독
                    _i_bclrs = ["#00c853" if v >= 0 else "#ff4b4b" for v in _ch_c["inst_daily"]]
                    _fig_sc.add_trace(go.Bar(x=_ch_c["dates"], y=_ch_c["inst_daily"],
                        name="기관", marker_color=_i_bclrs, opacity=0.7, showlegend=False), row=2, col=1)
                    _fig_sc.add_trace(go.Scatter(x=_ch_c["dates"], y=_ch_c["inst_ma5"],
                        name="기관MA5", line=dict(color="#ef5350", width=1.5), showlegend=False), row=2, col=1)
                    _fig_sc.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)", row=2, col=1)
                    # Row3: 외국인 단독
                    _f_bclrs = ["#00c853" if v >= 0 else "#ff4b4b" for v in _ch_c["frgn_daily"]]
                    _fig_sc.add_trace(go.Bar(x=_ch_c["dates"], y=_ch_c["frgn_daily"],
                        name="외국인", marker_color=_f_bclrs, opacity=0.7, showlegend=False), row=3, col=1)
                    _fig_sc.add_trace(go.Scatter(x=_ch_c["dates"], y=_ch_c["frgn_ma5"],
                        name="외국인MA5", line=dict(color="#26a69a", width=1.5), showlegend=False), row=3, col=1)
                    _fig_sc.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)", row=3, col=1)
                    _fig_sc.update_layout(height=520, margin=dict(l=10, r=20, t=40, b=10),
                        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                        font_color='#e0e0e0', showlegend=True,
                        legend=dict(orientation='h', y=1.05), dragmode=False)
                    _fig_sc.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.08)')
                    _fig_sc.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.08)')
                    st.plotly_chart(_fig_sc, use_container_width=True, config={"scrollZoom": False})
                else:
                    st.caption(f"수급 데이터: {_so_c.get('error','없음')}")
            st.divider()

        st.divider()
        for t in tickers_to_run:
            with st.spinner(f"{t} 분석 중..."):
                res = get_buy_timing(t)
            if "error" in res:
                st.error(f"❌ {t}: {res['error']}")
            else:
                kr = TICKER_NAMES.get(t, res.get("name", ""))
                _code = t.replace(".KS","").replace(".KQ","")
                _rt = _rt_quotes.get(_code, {})
                _rt_price = _rt.get("price_fmt", "")
                _rt_pct   = _rt.get("pct", "")
                _rt_dir   = _rt.get("direction", "")
                _rt_status = _rt.get("market_status", "")
                _rt_label = (f"🔴 실시간 {_rt_price}원 ({_rt_dir} {_rt_pct}%)" if _rt_price else "")
                st.markdown(f"#### 📌 **{t}** {kr} — 1년 기준가: `{res['price']:,}` {_rt_label}")
                c1,c2 = st.columns(2)
                c1.metric("고가→종가 평균 하락", f"{res['고가종가하락']:+.2f}%", help="장중 고점 대비 종가 평균 낙폭. 지정가보다 높게 올라갔다가 내려오는 정도")
                c2.metric("전일종가→당일저가 괴리", f"{res['저가종가괴리']:+.2f}%", help="전일 종가 대비 당일 저가 평균 괴리. 갭하락 포함")
                c1.metric("시가→저가 평균 낙폭", f"{res['시가저가괴리']:+.2f}%", help="시가 기준 장중 최대 낙폭 평균")
                c2.metric("전일종가→당일고가", f"{res['전일종가고가']:+.2f}%", help="전일 종가 대비 당일 고가 평균 괴리율")
                c1.metric("시가→당일고가 평균", f"{res['시가고가괴리']:+.2f}%", help="시가 대비 장중 고점까지 평균 상승폭")
                st.caption(f"💡 힌트: 시가 대비 저가 괴리({res['시가저가괴리']:+.2f}%) → 시가보다 그 정도 낮게 지정가 설정")
                st.divider()

        st.divider()
        st.markdown('<p class="zone-header">📊 주간 추세 — 자금이 들어오는가</p>', unsafe_allow_html=True)
        for t in tickers_to_run:
            with st.spinner(f"{t} 주간 추세 분석 중..."):
                wt = get_weekly_trend(t)
            kr = TICKER_NAMES.get(t, "")
            if "error" in wt:
                st.error(f"❌ {t} {kr}: {wt['error']}")
            else:
                st.markdown(f"#### 📌 **{t}** {kr}")
                c1,c2 = st.columns(2)
                c1.metric("CMF (4주)", f"{wt['cmf']:.4f}", "🟢 자금유입" if wt['cmf']>0 else "🔴 자금유출")
                c2.metric("주간 임펄스", wt["impulse_weekly"])
                sig = "🟢 매수신호" if wt["buy_signal"] else ("🔴 매도신호" if wt["sell_signal"] else "⏳ 대기")
                st.metric("CMF 신호", sig, f"최근4주 매수: {wt['recent_buy_4w']}회")
                c1.metric("주봉 TD 매도/매수", f"{wt['w_td_sell']} / {wt['w_td_buy']}")
                c2.metric("일봉 TD 매도/매수", f"{wt['d_td_sell']} / {wt['d_td_buy']}")
                st.metric("월봉 TD 매도/매수", f"{wt['m_td_sell']} / {wt['m_td_buy']}")
                if wt.get("ma10"): st.caption(f"현재가: {wt['price']:,} | 주봉 MA10: {wt['ma10']:,}")
                if wt.get("chart"):
                    with st.expander("📊 주봉 차트 보기 (가격 + CMF)"):
                        ch=wt["chart"]
                        _fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.65,0.35],
                                           vertical_spacing=0.05,subplot_titles=("주가","CMF"))
                        _fig.add_trace(go.Scatter(x=ch["dates"],y=ch["close"],name="종가",
                                                   line=dict(color="#E0E0E0",width=2)),row=1,col=1)
                        _fig.add_trace(go.Scatter(x=ch["dates"],y=ch["ma10"],name="MA10",
                                                   line=dict(color="#FF8C00",width=1.5,dash="dot")),row=1,col=1)
                        _cmf_colors=["#00c853" if v>0 else "#ff4b4b" for v in ch["cmf"]]
                        _fig.add_trace(go.Bar(x=ch["dates"],y=ch["cmf"],name="CMF",
                                               marker_color=_cmf_colors),row=2,col=1)
                        _fig.add_hline(y=0,line_dash="dash",line_color="rgba(255,255,255,0.3)",row=2,col=1)
                        _fig.update_layout(height=400,margin=dict(l=10,r=20,t=40,b=10),
                                            plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',
                                            font_color='#e0e0e0',showlegend=True,
                                            legend=dict(orientation='h',y=1.08),dragmode=False)
                        _fig.update_xaxes(showgrid=True,gridcolor='rgba(255,255,255,0.08)')
                        _fig.update_yaxes(showgrid=True,gridcolor='rgba(255,255,255,0.08)')
                        st.plotly_chart(_fig, use_container_width=True, config={"scrollZoom": False})
                st.divider()

        # ── 추세판별기 주간 Excel (선택) ──
        if _eff_weekly:
            try:
                _wk_xl = pd.ExcelFile(_eff_weekly, engine="openpyxl")
                _wk_sn = next((s for s in _wk_xl.sheet_names if 'DB' in s and '2' not in s), _wk_xl.sheet_names[0])
                if hasattr(_eff_weekly, 'seek'): _eff_weekly.seek(0)
                df_wkxl = pd.read_excel(_eff_weekly, sheet_name=_wk_sn, engine="openpyxl")
                wkr = calc_weekly_trend_excel(df_wkxl)
                if "error" in wkr:
                    st.error(f"추세판별기(주간) Excel 오류: {wkr['error']}")
                else:
                    st.divider()
                    st.markdown('<p class="zone-header">📅 주간 추세 (Excel HTS)</p>', unsafe_allow_html=True)
                    st.caption(f"시트: {_wk_sn} | {wkr['rows']}주 | {wkr['date_range']}")
                    c1, c2 = st.columns(2)
                    c1.metric("CMF (4주)", f"{wkr['cmf']:.4f}", "🟢 자금유입" if wkr['cmf'] > 0 else "🔴 자금유출")
                    c2.metric("주간 임펄스", wkr["impulse_weekly"])
                    sig = "🟢 매수신호" if wkr["buy_signal"] else ("🔴 매도신호" if wkr["sell_signal"] else "⏳ 대기")
                    st.metric("CMF 신호", sig, f"최근4주 매수: {wkr['recent_buy_4w']}회")
                    c1.metric("주봉 TD 매도/매수", f"{wkr['w_td_sell']} / {wkr['w_td_buy']}")
                    if wkr.get("ma10"):
                        st.caption(f"현재가: {wkr['price']:,} | 주봉 MA10: {wkr['ma10']:,}")
                    if wkr.get("chart"):
                        with st.expander("📊 주봉 차트 (Excel HTS 원천)"):
                            _ch = wkr["chart"]
                            _fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                                  row_heights=[0.65, 0.35], vertical_spacing=0.05,
                                                  subplot_titles=("주가", "CMF"))
                            _fig2.add_trace(go.Scatter(x=_ch["dates"], y=_ch["close"], name="종가",
                                                       line=dict(color="#E0E0E0", width=2)), row=1, col=1)
                            _fig2.add_trace(go.Scatter(x=_ch["dates"], y=_ch["ma10"], name="MA10",
                                                       line=dict(color="#FF8C00", width=1.5, dash="dot")), row=1, col=1)
                            _cmf_colors2 = ["#00c853" if v and v > 0 else "#ff4b4b" for v in _ch["cmf"]]
                            _fig2.add_trace(go.Bar(x=_ch["dates"], y=_ch["cmf"], name="CMF",
                                                   marker_color=_cmf_colors2), row=2, col=1)
                            _fig2.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)", row=2, col=1)
                            _fig2.update_layout(height=400, margin=dict(l=10, r=20, t=40, b=10),
                                                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                                                font_color='#e0e0e0', showlegend=True,
                                                legend=dict(orientation='h', y=1.08), dragmode=False)
                            _fig2.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.08)')
                            _fig2.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.08)')
                            st.plotly_chart(_fig2, use_container_width=True, config={"scrollZoom": False})
            except Exception as e:
                st.error(f"추세판별기(주간) 파일 읽기 오류: {e}")

        st.divider()
        # ── 추세판별기 수급 Excel (선택) ──
        if _eff_supply:
            try:
                _tsd_xl = pd.ExcelFile(_eff_supply, engine="openpyxl")
                _tsd_sn = next((s for s in _tsd_xl.sheet_names if 'DB' in s and '2' in s), _tsd_xl.sheet_names[0])
                if hasattr(_eff_supply, 'seek'): _eff_supply.seek(0)
                df_tsd = pd.read_excel(_eff_supply, sheet_name=_tsd_sn, engine="openpyxl")
                date_col_tsd = df_tsd.columns[0]
                # 외국인 or 기관 매수 컬럼 탐지
                buy_col = next((c for c in df_tsd.columns if '매수' in str(c) and ('외국' in str(c) or '기관' in str(c))), None)
                sell_col = next((c for c in df_tsd.columns if '매도' in str(c) and ('외국' in str(c) or '기관' in str(c))), None)
                close_col_tsd = next((c for c in df_tsd.columns if '종가' in str(c)), None)
                buy_label = buy_col if buy_col else "매수수량"
                sell_label = sell_col if sell_col else "매도수량"
                st.markdown('<p class="zone-header">🏦 기관 수급 오실레이터 (Excel HTS)</p>', unsafe_allow_html=True)
                if buy_col and close_col_tsd:
                    df_tsd[date_col_tsd] = pd.to_datetime(df_tsd[date_col_tsd], errors='coerce')
                    df_tsd = df_tsd.dropna(subset=[date_col_tsd]).sort_values(date_col_tsd)
                    df_tsd = df_tsd[pd.to_numeric(df_tsd[close_col_tsd], errors='coerce').fillna(0) != 0]
                    df_tsd[buy_col] = pd.to_numeric(df_tsd[buy_col], errors='coerce').fillna(0)
                    df_tsd[close_col_tsd] = pd.to_numeric(df_tsd[close_col_tsd], errors='coerce')
                    dates_tsd = [str(d.date()) for d in df_tsd[date_col_tsd]]
                    buy_vals_tsd = df_tsd[buy_col].tolist()
                    # 순매수 = 매수 - 매도 (오실레이터)
                    if sell_col:
                        df_tsd[sell_col] = pd.to_numeric(df_tsd[sell_col], errors='coerce').fillna(0)
                        net_vals_tsd = [b - s for b, s in zip(buy_vals_tsd, df_tsd[sell_col].tolist())]
                        osc_cap = f"순매수 = {buy_col} − {sell_col}"
                    else:
                        net_vals_tsd = buy_vals_tsd
                        osc_cap = buy_label
                    net_ma5_tsd = pd.Series(net_vals_tsd).rolling(5, min_periods=1).mean().tolist()
                    bar_clr_tsd = ["#00c853" if v >= 0 else "#ff4b4b" for v in net_vals_tsd]
                    fig_tsd = make_subplots(
                        rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("기관 순매수 오실레이터 (수량)", "주가"),
                        row_heights=[0.58, 0.42], vertical_spacing=0.06
                    )
                    fig_tsd.add_trace(go.Bar(x=dates_tsd, y=net_vals_tsd, name="순매수",
                                             marker_color=bar_clr_tsd, opacity=0.75), row=1, col=1)
                    fig_tsd.add_trace(go.Scatter(x=dates_tsd, y=net_ma5_tsd, name="MA5",
                                                 line=dict(color="#FFD700", width=2)), row=1, col=1)
                    fig_tsd.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.4)", row=1, col=1)
                    fig_tsd.add_trace(go.Scatter(x=dates_tsd, y=df_tsd[close_col_tsd].tolist(), name="종가",
                                                 line=dict(color="#E0E0E0", width=1.5)), row=2, col=1)
                    fig_tsd.update_layout(height=480, margin=dict(l=10, r=20, t=40, b=10),
                                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                          font_color="#e0e0e0", showlegend=True,
                                          legend=dict(orientation="h", y=1.05), dragmode=False)
                    fig_tsd.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
                    fig_tsd.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
                    st.plotly_chart(fig_tsd, use_container_width=True, config={"scrollZoom": False})
                    st.caption(f"시트: {_tsd_sn} | {osc_cap}")
                else:
                    st.warning(f"매수 컬럼 미감지. 컬럼 목록: {list(df_tsd.columns)}")
            except Exception as e:
                st.error(f"추세판별기 파일 읽기 오류: {e}")

        st.divider()
        st.markdown('<p class="zone-header">💰 거래 활성도 — 돈이 몰리는가</p>', unsafe_allow_html=True)
        for t in tickers_to_run:
            with st.spinner(f"{t} 거래대금 강도 분석 중..."):
                ti = get_trading_intensity(t)
            kr = TICKER_NAMES.get(t, "")
            if "error" in ti:
                st.error(f"❌ {t} {kr}: {ti['error']}")
            else:
                st.markdown(f"#### 📌 **{t}** {kr}")
                c1,c2 = st.columns(2)
                c1.metric("거래대금 강도 TI", f"{ti['ti']:.1f}", ti["signal_text"])
                c2.metric("TI MA3", f"{ti['ti_ma3']:.1f}")
                st.metric("TI Signal (EMA7)", f"{ti['ti_signal']:.1f}")
                st.caption("TI ≥75: 🔴 과열 | 40~74: 🟡 중립 | <40: 🟢 매집 구간")
                if ti.get("chart"):
                    with st.expander("📊 거래대금 강도 차트 보기"):
                        ch=ti["chart"]
                        _fig=go.Figure()
                        _fig.add_hrect(y0=75,y1=100,fillcolor="rgba(255,75,75,0.08)",line_width=0)
                        _fig.add_hrect(y0=0,y1=40,fillcolor="rgba(0,200,83,0.08)",line_width=0)
                        _fig.add_trace(go.Scatter(x=ch["dates"],y=ch["ti"],name="TI",
                                                   line=dict(color="#00B3B3",width=2)))
                        _fig.add_trace(go.Scatter(x=ch["dates"],y=ch["ti_ma3"],name="MA3",
                                                   line=dict(color="#FF8C00",width=1.5,dash="dot")))
                        _fig.add_trace(go.Scatter(x=ch["dates"],y=ch["ti_signal"],name="Signal(EMA7)",
                                                   line=dict(color="#6A5ACD",width=1.5,dash="dash")))
                        _fig.add_hline(y=75,line_dash="dash",line_color="#ff4b4b",opacity=0.6,
                                        annotation_text="과열(75)",annotation_position="bottom right")
                        _fig.add_hline(y=40,line_dash="dash",line_color="#00c853",opacity=0.6,
                                        annotation_text="매집(40)",annotation_position="top right")
                        _fig.update_layout(yaxis_range=[0,100],height=320,
                                            margin=dict(l=10,r=20,t=20,b=10),
                                            plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',
                                            font_color='#e0e0e0',legend=dict(orientation='h',y=1.1),
                                            dragmode=False)
                        _fig.update_xaxes(showgrid=True,gridcolor='rgba(255,255,255,0.08)')
                        _fig.update_yaxes(showgrid=True,gridcolor='rgba(255,255,255,0.08)')
                        st.plotly_chart(_fig, use_container_width=True, config={"scrollZoom": False})
                st.divider()

        st.divider()
        # ── 거래대금 강도 Excel _RotationRate_ (선택) ──
        if _eff_trading:
            try:
                df_txi = pd.read_excel(_eff_trading, sheet_name=0, engine="openpyxl")
                st.markdown('<p class="zone-header">📊 거래 활성도 (Excel)</p>', unsafe_allow_html=True)
                date_col_xi = df_txi.columns[0]
                rr_col = next((c for c in df_txi.columns if 'Rotation' in str(c) or 'rotation' in str(c) or '회전' in str(c)), None)
                close_col_xi = next((c for c in df_txi.columns if '종가' in str(c)), None)
                if rr_col:
                    df_txi[date_col_xi] = pd.to_datetime(df_txi[date_col_xi], errors='coerce')
                    df_txi = df_txi.dropna(subset=[date_col_xi]).sort_values(date_col_xi)
                    df_txi[rr_col] = pd.to_numeric(df_txi[rr_col], errors='coerce')
                    dates_xi = [str(d.date()) for d in df_txi[date_col_xi]]
                    rr_vals = df_txi[rr_col].tolist()
                    rr_colors = ["#00c853" if (v or 0) >= 0 else "#ff4b4b" for v in rr_vals]
                    fig_rr = make_subplots(specs=[[{"secondary_y": True}]])
                    fig_rr.add_trace(go.Bar(x=dates_xi, y=rr_vals, name="_RotationRate_",
                                            marker_color=rr_colors, opacity=0.7), secondary_y=False)
                    if close_col_xi:
                        df_txi[close_col_xi] = pd.to_numeric(df_txi[close_col_xi], errors='coerce')
                        fig_rr.add_trace(go.Scatter(x=dates_xi, y=df_txi[close_col_xi].tolist(), name="종가",
                                                    line=dict(color="#E0E0E0", width=2)), secondary_y=True)
                    fig_rr.update_layout(height=340, margin=dict(l=10,r=60,t=30,b=10),
                                         plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                         font_color="#e0e0e0", barmode="overlay",
                                         legend=dict(orientation="h", y=1.08), dragmode=False)
                    fig_rr.update_yaxes(title_text="_RotationRate_", secondary_y=False,
                                        showgrid=True, gridcolor="rgba(255,255,255,0.08)")
                    fig_rr.update_yaxes(title_text="종가(원)", secondary_y=True, showgrid=False)
                    fig_rr.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
                    st.plotly_chart(fig_rr, use_container_width=True, config={"scrollZoom": False})
                    st.caption(f"컬럼: {rr_col} | 출처: 국장 거래대금 강도 Excel")
                else:
                    st.warning(f"_RotationRate_ 컬럼 미감지. 컬럼 목록: {list(df_txi.columns)}")
            except Exception as e:
                st.error(f"거래대금 강도 파일 읽기 오류: {e}")

        st.divider()
        st.markdown('<p class="zone-header">📡 외국인 매매</p>', unsafe_allow_html=True)
        st.caption("수급이 낮을수록(비어있을수록) = 🏚️ 빈집 = 매집 여력이 큼 (좋은 신호) | 한국 종목(.KS/.KQ)만 지원")
        for t in tickers_to_run:
            kr = TICKER_NAMES.get(t, "")
            if not (t.endswith(".KS") or t.endswith(".KQ")):
                st.caption(f"⚠️ {t}: 한국 종목(.KS/.KQ)만 수급 오실레이터 지원")
                continue
            with st.spinner(f"{t} {kr} 수급 데이터 수집 중..."):
                so = get_stock_supply_osc(t, chart_days=60, agg_days=20)
            if "error" in so:
                st.error(f"❌ {t} {kr}: {so['error']}")
                continue
            st.markdown(f"#### 📌 **{t}** {kr}")
            f_sign = "+" if so["frgn_agg_bil"] >= 0 else ""
            i_sign = "+" if so["inst_agg_bil"] >= 0 else ""
            total_agg = so["frgn_agg_bil"] + so["inst_agg_bil"]
            if total_agg <= 0:
                st.markdown('<div class="sig-green">🏚️ 빈집 — 외국인·기관이 아직 덜 담음. 매집 여력이 큼 (좋은 신호)</div>', unsafe_allow_html=True)
            elif total_agg <= 100:
                st.markdown('<div class="sig-yellow">🟡 중립 — 수급 보통 구간. 추세 확인 필요</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="sig-red">⚠️ 수급 포화 — 이미 많이 매집됨. 추가 상승 여지 줄어듦</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.metric("20일 외국인", f"{f_sign}{so['frgn_agg_bil']:.1f}억",
                      "🏚️ 빈집" if so["frgn_agg_bil"] <= 0 else "⚠️ 매집 중")
            c2.metric("20일 기관", f"{i_sign}{so['inst_agg_bil']:.1f}억",
                      "🏚️ 빈집" if so["inst_agg_bil"] <= 0 else "⚠️ 매집 중")
            st.metric("현재가", f"{so['latest_close']:,}원")
            ch = so["chart"]
            # 외국인/기관 2행 차트
            fig_so = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                   subplot_titles=("외국인 순매매(억원)", "기관 순매매(억원)"),
                                   vertical_spacing=0.08, specs=[[{"secondary_y": True}],[{"secondary_y": True}]])
            for row_i, (daily_key, ma_key, cum_key, bar_color, label) in enumerate([
                ("frgn_daily","frgn_ma5","frgn_cum","#26a69a","외국인"),
                ("inst_daily","inst_ma5","inst_cum","#ef5350","기관"),
            ], start=1):
                bar_colors_i = ["#00c853" if v >= 0 else "#ff4b4b" for v in ch[daily_key]]
                fig_so.add_trace(go.Bar(x=ch["dates"], y=ch[daily_key],
                    name=f"{label} 일별", marker_color=bar_colors_i, opacity=0.7,
                    showlegend=(row_i==1)), row=row_i, col=1, secondary_y=False)
                fig_so.add_trace(go.Scatter(x=ch["dates"], y=ch[ma_key],
                    name=f"{label} MA5", line=dict(color=bar_color, width=2),
                    showlegend=(row_i==1)), row=row_i, col=1, secondary_y=False)
                fig_so.add_trace(go.Scatter(x=ch["dates"], y=ch["price"],
                    name="종가", line=dict(color="#E0E0E0", width=1.5),
                    showlegend=(row_i==1)), row=row_i, col=1, secondary_y=True)
            fig_so.update_layout(
                height=560, margin=dict(l=10, r=60, t=40, b=10),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e0e0e0", barmode="overlay",
                legend=dict(orientation="h", y=1.05),
                dragmode=False,
            )
            fig_so.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
            fig_so.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
            st.plotly_chart(fig_so, use_container_width=True, config={"scrollZoom": False})
            st.divider()

        st.markdown('<p class="zone-header">🏦 기관 매매</p>', unsafe_allow_html=True)
        st.caption("수급이 낮을수록(비어있을수록) = 🏚️ 빈집 = 매집 여력 큼 (좋은 신호) | 한국 종목(.KS/.KQ)만 지원")
        for t in tickers_to_run:
            kr = TICKER_NAMES.get(t, "")
            if not (t.endswith(".KS") or t.endswith(".KQ")):
                st.caption(f"⚠️ {t}: 한국 종목(.KS/.KQ)만 기관 수급 지원")
                continue
            with st.spinner(f"{t} {kr} 기관 데이터 조회 중..."):
                si = get_stock_inst_osc(t, days=20)
            if "error" in si:
                st.caption(f"❌ {t} {kr}: {si['error']}")
                continue
            st.markdown(f"#### 📌 **{t}** {kr}")
            si_sign = "+" if si["inst_sum_bil"] >= 0 else ""
            sf_sign = "+" if si.get("frgn_sum_bil", 0) >= 0 else ""
            inst_total = si["inst_sum_bil"] + si.get("frgn_sum_bil", 0)
            if inst_total <= 0:
                st.markdown('<div class="sig-green">🏚️ 빈집 — 기관·외국인 수급 비어있음. 매집 여력 큼 (좋은 신호)</div>', unsafe_allow_html=True)
            elif inst_total <= 100:
                st.markdown('<div class="sig-yellow">🟡 중립 — 수급 보통 구간</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="sig-red">⚠️ 수급 포화 — 이미 많이 매집됨</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.metric("20일 기관 순매수", f"{si_sign}{si['inst_sum_bil']:.1f}억",
                      "🏚️ 빈집" if si["inst_sum_bil"] <= 0 else "⚠️ 매집 중")
            c2.metric("20일 외국인 순매수(KRX)", f"{sf_sign}{si.get('frgn_sum_bil', 0):.1f}억",
                      "🏚️ 빈집" if si.get("frgn_sum_bil", 0) <= 0 else "⚠️ 매집 중")
            ch_si = si["chart"]
            bc_si = ["#00c853" if v >= 0 else "#ff4b4b" for v in ch_si["inst_daily"]]
            fig_si = go.Figure()
            fig_si.add_trace(go.Bar(x=ch_si["dates"], y=ch_si["inst_daily"],
                                    name="기관 일별", marker_color=bc_si, opacity=0.7))
            fig_si.add_trace(go.Scatter(x=ch_si["dates"], y=ch_si["inst_ma5"],
                                        name="MA5", line=dict(color="#FF8C00", width=2)))
            fig_si.add_trace(go.Scatter(x=ch_si["dates"], y=ch_si["inst_cum"],
                                        name="누적", line=dict(color="#6A5ACD", width=1.5, dash="dot")))
            if ch_si.get("frgn_daily"):
                fig_si.add_trace(go.Scatter(x=ch_si["dates"], y=ch_si["frgn_daily"],
                                            name="외국인", line=dict(color="#00B3B3", width=1.5)))
            fig_si.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
            fig_si.update_layout(height=320, margin=dict(l=10,r=30,t=30,b=10),
                                  plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                  font_color="#e0e0e0", legend=dict(orientation="h", y=1.08), dragmode=False)
            fig_si.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
            fig_si.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
            st.plotly_chart(fig_si, use_container_width=True, config={"scrollZoom": False})

# ─────────────────────────────────────────────────────────────────────────────
# 탭 4: 종목 선정
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.info("**지금 가장 유망한 종목을 자동으로 골라드립니다.** 버튼을 누르면 강한 섹터 안에서 수급·모멘텀 기준 상위 종목 순위를 보여드립니다. 1~2분 소요됩니다.")
    st.markdown('<p class="zone-header">🎯 지금 담을 종목</p>', unsafe_allow_html=True)

    _c_btn, _c_n = st.columns([5, 1])
    with _c_n:
        _top_n = st.number_input("종목 수", min_value=10, max_value=50, value=20, step=5,
                                  key="comp_topn", label_visibility="collapsed")
    with _c_btn:
        if st.button("▶ 스크리닝 실행", key="composite_run",
                     use_container_width=True, type="primary"):
            with st.spinner("섹터 분석 → 수급 수집 → 종합 점수 계산 중... (1~2분)"):
                _comp_res = get_composite_score(top_n=int(_top_n))
            st.session_state["c_composite"] = _comp_res

    if "c_composite" in st.session_state:
        _comp = st.session_state["c_composite"]
        if "error" in _comp:
            st.error(_comp["error"])
        else:
            _rows     = _comp.get("results", [])
            _has_sup  = _comp.get("has_supply", False)
            _str_secs = _comp.get("strong_sectors", [])

            # ── 섹터 필터 (클릭 가능) ──
            _all_secs = ["반도체","방산","조선","2차전지","바이오","K뷰티","로봇","자동차","원전","게임/엔터","금융"]
            st.caption(f"강한 섹터(초록)가 기본 선택됩니다 · 대상 {_comp.get('total',0)}종목 · 수급 {'✅' if _has_sup else '❌'}")
            if not _has_sup:
                st.warning("⚠️ 수급 컬럼(외국인/기관 순매수 20일)을 찾지 못했습니다. EPS 가속 필터만 적용됩니다. 컬럼명에 '20', '외국', '기관' 키워드가 포함되어 있는지 확인하세요.")
            _sel_secs = st.pills(
                "섹터 필터",
                _all_secs,
                selection_mode="multi",
                default=_str_secs,
                key="sec_filter",
                label_visibility="collapsed",
            )

            # 선택된 섹터로 필터링 (선택 없으면 전체)
            _filtered_rows = [r for r in _rows if r.get("sector") in _sel_secs] if _sel_secs else _rows

            # ── 뱃지 ──
            _bz   = [r for r in _filtered_rows if r.get("grade","").startswith("🏚️")]
            _star = [r for r in _filtered_rows if r.get("grade","").startswith("⭐")]
            _good = [r for r in _filtered_rows if r.get("grade","").startswith("✅")]
            if _bz or _star or _good:
                _bh  = "".join(f'<span class="badge-bz">🏚️ 빈집전환 {r["name"]}</span>' for r in _bz)
                _bh += "".join(f'<span class="badge-star">⭐ {r["name"]} {r["score"]:.0f}</span>' for r in _star)
                _bh += "".join(f'<span class="badge-good">✅ {r["name"]} {r["score"]:.0f}</span>' for r in _good)
                st.markdown(f'<div style="margin:0.6rem 0 0.8rem">{_bh}</div>',
                            unsafe_allow_html=True)
            if _comp.get("binzip_count", 0) == 0:
                st.caption("🏚️ 빈집전환 종목 없음 — 현재 강한 섹터 내 수급 바닥+전환 종목이 없습니다")

            if _filtered_rows:
                _df_comp = pd.DataFrame(_filtered_rows)
                _df_disp = _df_comp.rename(columns={
                    "grade":"등급","sector":"섹터","name":"종목명","score":"종합",
                    "rs":"RS","supply":"빈집여력","momentum":"모멘텀","volume":"거래대금","high52":"신고가"
                })[["등급","섹터","종목명","종합","RS","빈집여력","모멘텀","거래대금","신고가"]]

                st.dataframe(
                    _df_disp, use_container_width=True, hide_index=True,
                    column_config={
                        "종합":    st.column_config.ProgressColumn("종합점수",  min_value=0, max_value=100, format="%.1f"),
                        "RS":      st.column_config.ProgressColumn("RS",        min_value=0, max_value=100, format="%.0f"),
                        "빈집여력":st.column_config.ProgressColumn("빈집여력",  min_value=0, max_value=100, format="%.0f"),
                        "모멘텀":  st.column_config.ProgressColumn("모멘텀",    min_value=0, max_value=100, format="%.0f"),
                        "거래대금":st.column_config.ProgressColumn("거래대금",  min_value=0, max_value=100, format="%.0f"),
                        "신고가":  st.column_config.ProgressColumn("52주신고가",min_value=0, max_value=100, format="%.0f"),
                    }
                )

                with st.expander("📊 신호별 점수 차트"):
                    _top15 = _filtered_rows[:15][::-1]
                    _cn  = [r["name"]     for r in _top15]
                    _sup = [r["supply"]   or 0 for r in _top15]
                    _rs  = [r["rs"]       or 0 for r in _top15]
                    _mom = [r["momentum"] or 0 for r in _top15]
                    _vol = [r["volume"]   or 0 for r in _top15]
                    _h52 = [r["high52"]   or 0 for r in _top15]
                    _fig4 = go.Figure()
                    _fig4.add_trace(go.Bar(name="수급",    x=[v*0.30 for v in _sup], y=_cn, orientation='h', marker_color="#4CAF50"))
                    _fig4.add_trace(go.Bar(name="RS",      x=[v*0.25 for v in _rs],  y=_cn, orientation='h', marker_color="#2196F3"))
                    _fig4.add_trace(go.Bar(name="모멘텀",  x=[v*0.20 for v in _mom], y=_cn, orientation='h', marker_color="#9C27B0"))
                    _fig4.add_trace(go.Bar(name="거래대금",x=[v*0.15 for v in _vol], y=_cn, orientation='h', marker_color="#FF9800"))
                    _fig4.add_trace(go.Bar(name="신고가",  x=[v*0.10 for v in _h52], y=_cn, orientation='h', marker_color="#F44336"))
                    _fig4.update_layout(
                        barmode="stack", height=max(400, len(_cn)*32),
                        margin=dict(l=10, r=10, t=10, b=90),
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        font_color="#e0e0e0",
                        legend=dict(orientation="h", y=-0.18, x=0, xanchor="left",
                                    font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
                        dragmode=False,
                    )
                    st.plotly_chart(_fig4, use_container_width=True, config={"scrollZoom": False})

    st.divider()
    st.caption("⭐ 강력: RS≥65 + 빈집여력≥65  ·  ✅ 유망: RS≥55 + 빈집여력≥55  ·  빈집여력: 높을수록 아직 매집이 덜 된 종목 (좋은 신호)  ·  점수는 강한 섹터 내 백분위")

# ─────────────────────────────────────────────────────────────────────────────
# 탭 5: 수출 데이터
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.info("**주요 품목·업체 수출 트렌드를 분석합니다.** 사이드바에서 파일이 자동으로 로드되거나, 아래에서 직접 업로드하세요.")

    # ── 파일 로드 ─────────────────────────────────────────────────────────────
    _exp_may_bytes = st.session_state.get("c_exp_may_bytes")
    _exp_apr_bytes = st.session_state.get("c_exp_apr_bytes")

    _exp_src_label = []
    if _exp_may_bytes: _exp_src_label.append("✅ 5월 잠정")
    if _exp_apr_bytes: _exp_src_label.append("✅ 4월 확정")

    with st.expander("📂 수출 파일 업로드 / 상태", expanded=not bool(_exp_src_label)):
        if _exp_src_label:
            st.success("로컬 파일 로드됨: " + " | ".join(_exp_src_label))
            st.caption("사이드바 **🔄 로컬 파일 전체 로드** 버튼을 다시 누르면 최신 파일로 갱신됩니다.")
        else:
            st.warning("로컬 파일이 없습니다. 아래에서 직접 업로드하세요.")
        _c1, _c2 = st.columns(2)
        with _c1:
            _up_may = st.file_uploader("5월 잠정 xlsx (품목별 집계)", type=["xlsx"], key="up_exp_may_t5")
        with _c2:
            _up_apr = st.file_uploader("4월 확정 xlsx (개별 업체)", type=["xlsx"], key="up_exp_apr_t5")
        if _up_may:
            _up_may.seek(0); st.session_state["c_exp_may_bytes"] = _up_may.read()
            _exp_may_bytes = st.session_state["c_exp_may_bytes"]
            st.success("5월 잠정 업로드 완료")
        if _up_apr:
            _up_apr.seek(0); st.session_state["c_exp_apr_bytes"] = _up_apr.read()
            _exp_apr_bytes = st.session_state["c_exp_apr_bytes"]
            st.success("4월 확정 업로드 완료")
        # 추가 파일 (6월 잠정 등 미래 파일)
        st.markdown("---")
        st.caption("추가 파일 (6월 잠정 등) — 업로드하면 파일 선택 목록에 자동으로 나타납니다")
        _up_extra = st.file_uploader("추가 수출 xlsx (자유 업로드)", type=["xlsx"], key="up_exp_extra_t5")
        if _up_extra:
            _up_extra.seek(0); st.session_state["c_exp_extra_bytes"] = _up_extra.read()
            st.session_state["c_exp_extra_name"] = _up_extra.name
            st.success(f"{_up_extra.name} 업로드 완료")

    if not (_exp_may_bytes or _exp_apr_bytes or st.session_state.get("c_exp_extra_bytes")):
        st.stop()

    # ── 파일 선택 ─────────────────────────────────────────────────────────────
    _exp_file_opts = {}
    if _exp_may_bytes: _exp_file_opts["5월 잠정 (품목별 집계)"] = (_exp_may_bytes, "may")
    if _exp_apr_bytes: _exp_file_opts["4월 확정 (개별 업체)"]   = (_exp_apr_bytes, "apr")
    if st.session_state.get("c_exp_extra_bytes"):
        _extra_name = st.session_state.get("c_exp_extra_name", "추가 파일")
        _exp_file_opts[_extra_name] = (st.session_state["c_exp_extra_bytes"], "may")

    _exp_file_sel = st.radio("파일 선택", list(_exp_file_opts.keys()),
                              horizontal=True, key="exp_file_sel_t5")
    _cur_bytes, _cur_layout = _exp_file_opts[_exp_file_sel]

    # ── 시트 목록 ─────────────────────────────────────────────────────────────
    _sheet_cache_key = f"_exp_sheets_{_exp_file_sel}"
    if _sheet_cache_key not in st.session_state:
        try:
            _xl_tmp = pd.ExcelFile(io.BytesIO(_cur_bytes), engine="openpyxl")
            _skip_sheets = {"반도체 수출 판가 증가율"}
            st.session_state[_sheet_cache_key] = [s for s in _xl_tmp.sheet_names if s not in _skip_sheets]
        except Exception as _e:
            st.error(f"시트 목록 로드 실패: {_e}")
            st.session_state[_sheet_cache_key] = []
    _all_sheets = st.session_state.get(_sheet_cache_key, [])
    if not _all_sheets:
        st.warning("시트 목록을 읽을 수 없습니다.")
        st.stop()

    st.caption(f"총 **{len(_all_sheets)}개** 시트 감지")

    # ── 섹션 1: 트렌드 차트 ───────────────────────────────────────────────────
    st.markdown('<p class="zone-header">📈 트렌드 차트</p>', unsafe_allow_html=True)

    _default_may = ["총수출", "디램", "낸드", "HBM", "리튬전지", "mlcc"]
    _defaults    = [s for s in _default_may if s in _all_sheets] if _cur_layout == "may" else _all_sheets[:4]

    _col_sel, _col_metric = st.columns([3, 1])
    with _col_sel:
        _selected = st.multiselect(
            "품목/업체 선택",
            options=_all_sheets,
            default=_defaults[:4],
            key="exp_sel_t5",
            help="여러 항목 선택 시 같은 차트에 겹쳐서 표시",
        )
    with _col_metric:
        _metric_opts = {"일평균": "일평균", "월 총액": "금액", "단가": "단가"}
        _metric_sel  = st.selectbox("지표", list(_metric_opts.keys()), key="exp_metric_t5")
        _metric      = _metric_opts[_metric_sel]

    if _selected:
        _dfs, _labels = [], []
        for _sn in _selected:
            _df = _load_exp_sheet(_cur_bytes, _sn, layout=_cur_layout)
            if not _df.empty:
                _dfs.append(_df)
                _labels.append(_sn)

        if _dfs:
            _all_dates_t5 = pd.concat([d["날짜"] for d in _dfs]).dropna()
            _min_yr_t5 = int(_all_dates_t5.min().year)
            _max_yr_t5 = int(_all_dates_t5.max().year)
            _yr_range_t5 = st.slider("기간 (연도)", _min_yr_t5, _max_yr_t5,
                                      (max(_min_yr_t5, _max_yr_t5 - 3), _max_yr_t5),
                                      key="exp_yr_t5")
            _t_s = pd.Timestamp(_yr_range_t5[0], 1, 1)
            _t_e = pd.Timestamp(_yr_range_t5[1], 12, 31)
            _dfs_f = [d[(d["날짜"] >= _t_s) & (d["날짜"] <= _t_e)] for d in _dfs]

            _fig_t5 = _exp_chart(_dfs_f, _labels, metric=_metric,
                                  title=f"{_exp_file_sel} — {_metric_sel}")
            if _fig_t5:
                st.plotly_chart(_fig_t5, use_container_width=True, config={"scrollZoom": False})

            # 최신 수치 요약 카드
            st.markdown("**최신 데이터 요약**")
            _sum_cols_t5 = st.columns(min(len(_dfs_f), 4))
            for _ci, (_df_c, _lbl) in enumerate(zip(_dfs_f, _labels)):
                if _df_c.empty: continue
                _last_t5 = _df_c.dropna(subset=[_metric]).iloc[-1] if not _df_c.dropna(subset=[_metric]).empty else None
                if _last_t5 is None: continue
                with _sum_cols_t5[_ci % 4]:
                    _val_t5 = _last_t5[_metric]
                    _yoy_t5 = _last_t5["YoY"] if "YoY" in _last_t5.index else None
                    _yoy_pct_t5 = f"{_yoy_t5*100:+.1f}%" if pd.notna(_yoy_t5) else "—"
                    _vfmt = f"${_val_t5/1e6:.1f}M" if _metric in ("금액", "일평균") and _val_t5 > 1e5 else f"{_val_t5:,.1f}"
                    st.metric(label=_lbl[:14], value=_vfmt, delta=f"YoY {_yoy_pct_t5}")
        else:
            st.warning("선택한 시트를 파싱할 수 없습니다.")

    # ── 섹션 2: 특이점 자동 감지 ──────────────────────────────────────────────
    st.markdown('<p class="zone-header">📢 수출 특이점 — YoY/MoM 급변 감지</p>', unsafe_allow_html=True)
    st.caption("전체 시트를 자동 스캔해 최신 월 기준 급증·급감 항목을 추출합니다. 새 파일(6월 잠정 등)을 업로드하면 자동 반영됩니다.")

    _thr_c1, _thr_c2 = st.columns(2)
    with _thr_c1:
        _yoy_thr_t5 = st.slider("YoY 임계값 (%)", 10, 200, 40, step=10, key="anom_yoy_t5",
                                  help="이 수치 이상 변화 시 특이점") / 100
    with _thr_c2:
        _mom_thr_t5 = st.slider("MoM 임계값 (%)", 5, 100, 25, step=5, key="anom_mom_t5") / 100

    with st.spinner("전체 시트 스캔 중..."):
        _anom_df_t5 = _scan_exp_anomalies(_cur_bytes, _cur_layout, _yoy_thr_t5, _mom_thr_t5)

    if _anom_df_t5.empty:
        st.info(f"임계값({_yoy_thr_t5*100:.0f}% / {_mom_thr_t5*100:.0f}%) 해당 특이점 없음")
    else:
        _n_up_t5   = int((_anom_df_t5["YoY_pct"].fillna(0) > 0).sum())
        _n_down_t5 = int((_anom_df_t5["YoY_pct"].fillna(0) < 0).sum())
        st.caption(f"총 **{len(_anom_df_t5)}건** — 🔺 급증 {_n_up_t5}건 / 🔻 급감 {_n_down_t5}건")

        _au = _anom_df_t5[_anom_df_t5["YoY_pct"].fillna(0) > 0].head(20)
        _ad = _anom_df_t5[_anom_df_t5["YoY_pct"].fillna(0) < 0].head(15)

        _tcol1, _tcol2 = st.columns(2)
        with _tcol1:
            if not _au.empty:
                st.markdown("**🔺 급증 항목 (YoY 상위)**")
                for _, _row in _au.iterrows():
                    _ys = f"YoY **{_row['YoY_pct']:+.1f}%**" if pd.notna(_row["YoY_pct"]) else ""
                    _ms = f"MoM {_row['MoM_pct']:+.1f}%" if pd.notna(_row["MoM_pct"]) else ""
                    _as = f"  일평균 ${_row['일평균']/1e6:.1f}M" if pd.notna(_row["일평균"]) and _row["일평균"] > 1e5 else ""
                    _ds = _row["날짜"].strftime("%Y-%m") if pd.notna(_row["날짜"]) else ""
                    _pts = " · ".join(p for p in [_ys, _ms] if p)
                    st.markdown(
                        f'<div class="sig-green"><b>{_row["시트명"]}</b>'
                        f' <span style="color:#7d8590;font-size:0.8em">({_ds})</span>'
                        f'&nbsp; {_pts}{_as}</div>',
                        unsafe_allow_html=True,
                    )
        with _tcol2:
            if not _ad.empty:
                st.markdown("**🔻 급감 항목 (YoY 하위)**")
                for _, _row in _ad.iterrows():
                    _ys = f"YoY **{_row['YoY_pct']:+.1f}%**" if pd.notna(_row["YoY_pct"]) else ""
                    _ms = f"MoM {_row['MoM_pct']:+.1f}%" if pd.notna(_row["MoM_pct"]) else ""
                    _as = f"  일평균 ${_row['일평균']/1e6:.1f}M" if pd.notna(_row["일평균"]) and _row["일평균"] > 1e5 else ""
                    _ds = _row["날짜"].strftime("%Y-%m") if pd.notna(_row["날짜"]) else ""
                    _pts = " · ".join(p for p in [_ys, _ms] if p)
                    st.markdown(
                        f'<div class="sig-red"><b>{_row["시트명"]}</b>'
                        f' <span style="color:#7d8590;font-size:0.8em">({_ds})</span>'
                        f'&nbsp; {_pts}{_as}</div>',
                        unsafe_allow_html=True,
                    )
