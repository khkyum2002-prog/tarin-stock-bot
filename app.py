import io, time, warnings, logging
import numpy as np
import pandas as pd
import yfinance as yf
import requests
import streamlit as st
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

st.set_page_config(page_title="퇴근길 주식", page_icon="📈", layout="centered")
st.markdown("""<style>
/* ── 메트릭 카드 ── */
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 10px 14px;
    margin: 3px 0;
}
[data-testid="stMetricValue"] { font-size: 1.05rem !important; }
[data-testid="stMetricLabel"] { font-size: 0.78rem !important; color: rgba(255,255,255,0.6) !important; }
/* ── 탭 ── */
.stTabs [data-baseweb="tab"] { font-weight: 700; font-size: 0.95rem; }
hr { margin: 0.5rem 0; }
/* ── 신호 카드 ── */
.sig-green {
    background: rgba(0,200,83,0.12);
    border-left: 3px solid #00c853;
    border-radius: 6px; padding: 8px 12px; margin: 4px 0;
    font-size: 0.88rem;
}
.sig-red {
    background: rgba(255,75,75,0.12);
    border-left: 3px solid #ff4b4b;
    border-radius: 6px; padding: 8px 12px; margin: 4px 0;
    font-size: 0.88rem;
}
.sig-yellow {
    background: rgba(255,193,7,0.12);
    border-left: 3px solid #ffc107;
    border-radius: 6px; padding: 8px 12px; margin: 4px 0;
    font-size: 0.88rem;
}
/* ── 섹션 헤더 ── */
.zone-header {
    font-size: 0.75rem; font-weight: 600; letter-spacing: 0.08em;
    color: rgba(255,255,255,0.4); text-transform: uppercase;
    margin: 1.2rem 0 0.4rem;
}
/* ── 업로더 영역 ── */
div[data-testid="stFileUploader"] label {
    font-size: 0.82rem !important;
}
/* ── 버튼 ── */
div[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(135deg,#1a73e8,#0d47a1);
    border: none; border-radius: 8px;
}
</style>""", unsafe_allow_html=True)

_days = ["월","화","수","목","금","토","일"]
st.title("📈 퇴근길 주식")
st.caption(f"오늘: {datetime.today().strftime('%Y-%m-%d')} ({_days[datetime.today().weekday()]}요일)")

tab1, tab2, tab3 = st.tabs(["🌎 미국 지표", "🇰🇷 국내 지표", "🔍 종목 분석"])

# ─────────────────────────────────────────────────────────────────────────────
# 공통 유틸
# ─────────────────────────────────────────────────────────────────────────────
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

NAVER_ETF_URL = "https://finance.naver.com/api/sise/etfItemList.nhn?etfType=0"
HEADERS = {"User-Agent":"Mozilla/5.0","Referer":"https://finance.naver.com"}
SP500_TOP100 = ["AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","BRK-B","TSLA","LLY","AVGO","JPM","UNH","XOM","V","MA","PG","JNJ","HD","COST","ABBV","MRK","NFLX","CVX","BAC","CRM","ORCL","AMD","PEP","ACN","WMT","LIN","MCD","CSCO","TMO","ADBE","PLTR","TMUS","INTU","GE","IBM","CAT","PM","AMGN","TXN","NOW","ISRG","QCOM","UBER","GS","VZ","HON","RTX","SPGI","DHR","NEE","MS","LOW","T","UNP","BKNG","AXP","SCHW","C","BLK","SYK","GILD","PFE","DE","MDT","BA","AMAT","ADI","LRCX","PANW","MU","TJX","ETN","VRTX","KLAC","SBUX","CB","MMC","SO","DUK","BSX","REGN","PLD","CI","ZTS","ICE","CME","WM","APH","MCO","SNPS","CDNS","ITW","NOC","EMR"]
NASDAQ100 = ["AAPL","ABNB","ADBE","ADI","ADP","ADSK","AEP","AMAT","AMD","AMGN","AMZN","ANSS","ASML","AVGO","AZN","BIIB","BKNG","BKR","CCEP","CDNS","CDW","CEG","CHTR","CMCSA","COST","CPRT","CRWD","CSCO","CSX","CTAS","CTSH","DASH","DDOG","DLTR","DXCM","EA","EXC","FANG","FAST","FTNT","GEHC","GILD","GOOG","GOOGL","HON","IDXX","INTC","INTU","ISRG","KDP","KHC","KLAC","LIN","LRCX","LULU","MAR","MCHP","MDLZ","MELI","META","MNST","MRNA","MRVL","MSFT","MU","NFLX","NVDA","NXPI","ODFL","ON","ORLY","PANW","PAYX","PCAR","PDD","PEP","PYPL","QCOM","REGN","ROP","ROST","SBUX","SNPS","TEAM","TMUS","TSLA","TTD","TXN","VRSK","VRTX","WDAY","XEL","ZS"]

# ─────────────────────────────────────────────────────────────────────────────
# 미국 지표 함수
# ─────────────────────────────────────────────────────────────────────────────
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

def get_blood_indicator():
    try:
        raw = yf.download(["^IRX","^TNX","HYG","IEF"], start="2015-01-01", auto_adjust=True, progress=False)
        px = raw["Close"].copy()
        irx = px["^IRX"]   # Yahoo quotes in %, e.g. 5.0 = 5%
        t10y = px["^TNX"]  # Yahoo quotes in %, e.g. 4.5 = 4.5%
        hyg_ticker = yf.Ticker("HYG")
        hyg_yield_raw = getattr(hyg_ticker.fast_info,"dividend_yield",None) or hyg_ticker.info.get("dividendYield",0.06)
        hyg_yield = hyg_yield_raw * 100 if hyg_yield_raw < 1 else hyg_yield_raw  # convert decimal to %
        blood = (irx/(hyg_yield - t10y)).dropna()
        cur=float(blood.iloc[-1]); ma20=float(blood.rolling(20).mean().iloc[-1]); ma60=float(blood.rolling(60).mean().iloc[-1])
        return {"value":round(cur,4),"ma20":round(ma20,4),"ma60":round(ma60,4),"vs_ma20":"위" if cur>ma20 else "아래","vs_ma60":"위" if cur>ma60 else "아래"}
    except Exception as e: return {"error":str(e)}

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

def get_monthly_fear_greed():
    try:
        start = (datetime.today()-timedelta(days=25*365)).strftime('%Y-%m-%d')
        raw = yf.download(['^GSPC','^IXIC','^VIX','^TNX','^FVX','HYG','IEF'], start=start, auto_adjust=True, progress=False)
        data = raw["Close"].rename(columns={'^GSPC':'SP500','^IXIC':'NASDAQ','^VIX':'VIX','^TNX':'10Y','^FVX':'5Y'})
        data = data.resample('ME').ffill().dropna(); data['RA'] = data['HYG']/data['IEF']
        def calc(df, col, lbl):
            df[f'{lbl}_MA125'] = df[col].rolling(125).mean()
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
def get_market_summary():
    try:
        start=_td_back(5); end=datetime.today().strftime("%Y-%m-%d")
        kp=_close("^KS11",start=start,end=end); kq=_close("^KQ11",start=start,end=end)
        if kp.empty or kq.empty: return {"error":"데이터 없음"}
        kp_l=float(kp.iloc[-1]); kp_p=float(kp.iloc[-2]) if len(kp)>=2 else kp_l
        kq_l=float(kq.iloc[-1]); kq_p=float(kq.iloc[-2]) if len(kq)>=2 else kq_l
        return {"date":kp.index[-1].strftime("%Y-%m-%d"),
                "kospi":{"close":round(kp_l,2),"chg_pct":round((kp_l/kp_p-1)*100,2),"week_pct":round((kp_l/float(kp.iloc[0])-1)*100,2)},
                "kosdaq":{"close":round(kq_l,2),"chg_pct":round((kq_l/kq_p-1)*100,2),"week_pct":round((kq_l/float(kq.iloc[0])-1)*100,2)}}
    except Exception as e: return {"error":str(e)}

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

def get_kr_etf_rs():
    try:
        tks=list(KOREA_ETFS.keys())
        raw=yf.download(tks+["^KS11"],period="2y",auto_adjust=True,progress=False)
        if raw.empty: return {"error":"데이터 없음"}
        data=raw["Close"] if isinstance(raw.columns,pd.MultiIndex) else raw
        if "^KS11" not in data.columns: return {"error":"KOSPI 없음"}
        kospi=data["^KS11"].dropna(); results=[]
        for t in tks:
            if t not in data.columns: continue
            etf=data[t].dropna(); common=etf.index.intersection(kospi.index)
            if len(common)<60: continue
            rel=etf.loc[common]/kospi.loc[common]
            rs_vals=[]
            for win in [60,120,250]:
                if len(rel)<win: continue
                ma=rel.rolling(win).mean(); rs=((rel/ma)-1)*100
                if not rs.dropna().empty: rs_vals.append(float(rs.dropna().iloc[-1]))
            rs_raw=round(np.mean(rs_vals),2) if rs_vals else 0
            norm_rs=round(100*(1/(1+np.exp(-rs_raw/12))),1)
            mom3=float(etf.pct_change().rolling(63).mean().iloc[-1]) if len(etf)>=63 else 0
            vol3=float(etf.pct_change().rolling(63).std().iloc[-1]) if len(etf)>=63 else 1
            results.append({"ticker":t,"name":KOREA_ETFS.get(t,t),"norm_rs":norm_rs,"risk_adj":round((mom3/vol3)*100,2) if vol3>0 else 0})
        results.sort(key=lambda x:x["norm_rs"],reverse=True)
        strong=[r for r in results if r["norm_rs"]>=70]
        return {"all":results,"strong":strong[:10] if strong else results[:5]}
    except Exception as e: return {"error":str(e)}

def get_buy_timing(ticker):
    try:
        t=ticker.strip().upper()
        raw=yf.download(t,period="1y",auto_adjust=False,progress=False)
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
    """종목상대강도데이터.xlsx 종가 시트로 Mansfield RS 계산 (log-ratio, 60/120/250d)"""
    try:
        date_col = df_close.columns[0]
        df = df_close.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
        df = df.set_index(date_col)
        bench_col = next((c for c in df.columns if '코스피' in str(c)), None)
        if bench_col is None:
            return {"error": "코스피 벤치마크 컬럼 없음"}
        kospi = pd.to_numeric(df[bench_col], errors='coerce').where(lambda x: x > 0, np.nan)
        name_to_ticker = {v: k for k, v in KR_STOCKS.items()}
        stock_cols = [c for c in df.columns if c != bench_col]
        RS_WINDOWS = [60, 120, 250]
        all_results = []
        for col in stock_cols:
            s = pd.to_numeric(df[col], errors='coerce').where(lambda x: x > 0, np.nan)
            common = s.dropna().index.intersection(kospi.dropna().index)
            if len(common) < 63:
                continue
            sc = s.loc[common]; kc = kospi.loc[common]
            rel = sc / kc
            log_rel = np.log(rel.replace(0, np.nan))
            raw_vals = []
            for win in RS_WINDOWS:
                if len(log_rel.dropna()) < win:
                    continue
                ma = log_rel.rolling(win, min_periods=win).mean()
                rs_s = (log_rel - ma).dropna()
                if not rs_s.empty:
                    raw_vals.append(float(rs_s.iloc[-1] * 100))
            if not raw_vals:
                continue
            rs_avg = float(np.mean(raw_vals))
            norm_rs = round(100 * (1 / (1 + np.exp(-rs_avg / 12))), 1)
            # risk-adjusted momentum (3m/6m/12m)
            r3 = sc.pct_change(fill_method=None).rolling(63).mean().iloc[-1]
            v3 = sc.pct_change(fill_method=None).rolling(63).std().iloc[-1]
            r6 = sc.pct_change(fill_method=None).rolling(126).mean().iloc[-1]
            v6 = sc.pct_change(fill_method=None).rolling(126).std().iloc[-1]
            r12 = sc.pct_change(fill_method=None).rolling(252).mean().iloc[-1]
            v12 = sc.pct_change(fill_method=None).rolling(252).std().iloc[-1]
            risk_adj = float(np.nanmean([
                r3/v3 if v3 and v3 > 0 else np.nan,
                r6/v6 if v6 and v6 > 0 else np.nan,
                r12/v12 if v12 and v12 > 0 else np.nan
            ])) * 100
            ticker = name_to_ticker.get(str(col), str(col))
            all_results.append({"ticker": ticker, "name": str(col), "norm_rs": norm_rs,
                                 "rs_raw": round(rs_avg, 1), "risk_adj": round(risk_adj, 2)})
        if not all_results:
            return {"error": "RS 계산 결과 없음"}
        all_results.sort(key=lambda x: x["norm_rs"], reverse=True)
        strong = [r for r in all_results if r["norm_rs"] >= 70]
        return {"all": all_results, "strong": strong[:top_n] if strong else all_results[:top_n]}
    except Exception as e:
        return {"error": str(e)}

def calc_kr_etf_rs_excel(df_data, top_n=15):
    """etf상대강도데이터.xlsx 데이터 시트로 ETF RS 계산 (linear Mansfield RS, 60/120/250d)"""
    try:
        date_col = df_data.columns[0]
        df = df_data.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
        df = df.set_index(date_col)
        bench_col = next((c for c in df.columns if '코스피' in str(c)), None)
        if bench_col is None:
            return {"error": "코스피 벤치마크 컬럼 없음"}
        kospi = pd.to_numeric(df[bench_col], errors='coerce').where(lambda x: x > 0, np.nan)
        etf_cols = [c for c in df.columns if c != bench_col]
        RS_WINDOWS = [60, 120, 250]
        all_results = []
        for col in etf_cols:
            s = pd.to_numeric(df[col], errors='coerce').where(lambda x: x > 0, np.nan)
            common = s.dropna().index.intersection(kospi.dropna().index)
            if len(common) < 60:
                continue
            sc = s.loc[common]; kc = kospi.loc[common]
            rel = sc / kc
            raw_vals = []
            for win in RS_WINDOWS:
                if len(rel.dropna()) < win:
                    continue
                ma = rel.rolling(win, min_periods=win).mean()
                rs_s = ((rel / ma) - 1).dropna()
                if not rs_s.empty:
                    raw_vals.append(float(rs_s.iloc[-1] * 100))
            if not raw_vals:
                continue
            rs_avg = float(np.mean(raw_vals))
            norm_rs = round(100 * (1 / (1 + np.exp(-rs_avg / 12))), 1)
            r3 = sc.pct_change(fill_method=None).rolling(63).mean().iloc[-1]
            v3 = sc.pct_change(fill_method=None).rolling(63).std().iloc[-1]
            r6 = sc.pct_change(fill_method=None).rolling(126).mean().iloc[-1]
            v6 = sc.pct_change(fill_method=None).rolling(126).std().iloc[-1]
            r12 = sc.pct_change(fill_method=None).rolling(252).mean().iloc[-1]
            v12 = sc.pct_change(fill_method=None).rolling(252).std().iloc[-1]
            risk_adj = float(np.nanmean([
                r3/v3 if v3 and v3 > 0 else np.nan,
                r6/v6 if v6 and v6 > 0 else np.nan,
                r12/v12 if v12 and v12 > 0 else np.nan
            ])) * 100
            all_results.append({"name": str(col), "norm_rs": norm_rs,
                                 "rs_raw": round(rs_avg, 1), "risk_adj": round(risk_adj, 2)})
        if not all_results:
            return {"error": "ETF RS 계산 결과 없음"}
        all_results.sort(key=lambda x: x["norm_rs"], reverse=True)
        strong = [r for r in all_results if r["norm_rs"] >= 70]
        return {"all": all_results, "strong": strong[:top_n] if strong else all_results[:top_n]}
    except Exception as e:
        return {"error": str(e)}

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
                "cmf": round(cur_cmf, 4), "buy_signal": bool(buy.iloc[-1]), "sell_signal": bool(sell.iloc[-1]),
                "recent_buy_4w": int(buy.tail(4).sum()), "impulse_weekly": impulse_w,
                "w_td_sell": w_ts, "w_td_buy": w_tb,
                "date_range": f"{dates[0]} ~ {dates[-1]}", "rows": len(df), "chart": chart}
    except Exception as e:
        return {"error": str(e)}

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

def get_stock_supply_osc(ticker, chart_days=60, agg_days=20):
    """단일 한국 종목 수급 오실레이터 — 네이버 파이낸스 외국인 순매수"""
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
                if len(cells) >= 6 and cells[0] and "." in cells[0]:
                    try:
                        close = _pn(cells[1]); net_qty = _pn(cells[5])
                        if close > 0:
                            rows_data.append({"date":cells[0],"close":close,
                                              "net_qty":net_qty,"net_val":net_qty*close})
                    except: pass
            if len(rows_data) >= chart_days: break
        if not rows_data: return {"error":"데이터 없음"}
        df = pd.DataFrame(rows_data[:chart_days]).iloc[::-1].reset_index(drop=True)
        df["net_bil"] = df["net_val"] / 100_000_000
        df["ma5"]    = df["net_bil"].rolling(5, min_periods=1).mean()
        df["cum20"]  = df["net_bil"].rolling(agg_days, min_periods=1).sum()
        net_agg = round(float(df["net_bil"].tail(agg_days).sum()), 1)
        return {
            "ticker": ticker, "agg_days": agg_days,
            "net_agg_bil": net_agg,
            "latest_close": int(df["close"].iloc[-1]),
            "chart": {
                "dates": df["date"].tolist(),
                "price": df["close"].tolist(),
                "daily": [round(v,1) for v in df["net_bil"].tolist()],
                "ma5":   [round(v,1) for v in df["ma5"].tolist()],
                "cum20": [round(v,1) for v in df["cum20"].tolist()],
            },
        }
    except Exception as e: return {"error": str(e)}

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
                    if len(cells) >= 6 and cells[0] and "." in cells[0]:
                        try:
                            close = _parse_num(cells[1])
                            net_qty = _parse_num(cells[5])
                            if close > 0:
                                rows_data.append({"close": close, "net_qty": net_qty,
                                                  "net_val": net_qty * close, "date": cells[0]})
                        except: pass
                if len(rows_data) >= CHART_DAYS: break

            if not rows_data: return None

            # 최신순 → 오래된 순으로 정렬 (차트용)
            df_s = pd.DataFrame(rows_data[:CHART_DAYS]).iloc[::-1].reset_index(drop=True)
            df_s["net_bil"] = df_s["net_val"] / 100_000_000  # 억원

            # 오실레이터: 5일 MA (시그널), 20일 누적합
            df_s["ma5"]   = df_s["net_bil"].rolling(5, min_periods=1).mean()
            df_s["cum20"] = df_s["net_bil"].rolling(days, min_periods=1).sum()

            # 집계 기간(days) 기준 순매수 합계 (최신 N일)
            net_nd_val = int(df_s["net_val"].tail(days).sum())
            latest_close = int(df_s["close"].iloc[-1])

            return {
                "ticker": ticker, "name": name, "code": code,
                "net_nd_val": net_nd_val,
                "net_nd_bil": round(net_nd_val / 100_000_000, 1),
                "latest_close": latest_close,
                "chart": {
                    "dates":  df_s["date"].tolist(),
                    "price":  df_s["close"].tolist(),
                    "daily":  [round(v, 1) for v in df_s["net_bil"].tolist()],
                    "ma5":    [round(v, 1) for v in df_s["ma5"].tolist()],
                    "cum20":  [round(v, 1) for v in df_s["cum20"].tolist()],
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
        "note": f"외국인 순매수량 × 종가 ({days}거래일 누적) | 출처: 네이버 파이낸스"
    }

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
        mask=(df_db[COL_EPS_1M].notna()&df_db[COL_EPS_3M].notna()&
              (df_db[COL_EPS_1M]>0)&(df_db[COL_EPS_3M]>0)&(df_db[COL_EPS_1M]>df_db[COL_EPS_3M]))
        base=df_db[mask].copy()
        f_top20=[]; i_top20=[]; common=[]
        if COL_F_6M and COL_F_1M and COL_I_6M and COL_I_1M and COL_MKT in df_db.columns:
            base["외국인_6M"]=base[COL_F_6M]/base[COL_MKT]; base["외국인_1M"]=base[COL_F_1M]/base[COL_MKT]
            base["기관_6M"]=base[COL_I_6M]/base[COL_MKT]; base["기관_1M"]=base[COL_I_1M]/base[COL_MKT]
            fp=base[(base["외국인_6M"]>0)&(base["외국인_1M"]>0)]
            ip=base[(base["기관_6M"]>0)&(base["기관_1M"]>0)]
            f_top20=fp.sort_values("외국인_1M",ascending=False).head(20)[COL_NAME].tolist()
            i_top20=ip.sort_values("기관_1M",ascending=False).head(20)[COL_NAME].tolist()
            common=sorted(set(f_top20)&set(i_top20))
        else:
            f_top20=base.head(20)[COL_NAME].tolist()
        return {"eps_passed":len(base),"foreign_top20":f_top20,"inst_top20":i_top20,"common":common}
    except Exception as e: return {"error":str(e)}

# ── 공통 차트 헬퍼 ──
def _chart_layout(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10,r=20,t=30,b=10),
                      plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                      font_color='#e0e0e0', legend=dict(orientation='h',y=1.08))
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.07)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.07)')
    return fig

def _rs_bar_chart(items, name_key="name", val_key="norm_rs", height=320):
    s = sorted(items, key=lambda x: x[val_key])
    names = [r[name_key] for r in s]; vals = [r[val_key] for r in s]
    colors = ["#00c853" if v>=70 else ("#ffc107" if v>=50 else "#ff4b4b") for v in vals]
    fig = go.Figure(go.Bar(x=vals, y=names, orientation='h', marker_color=colors,
                           text=[f"{v:.1f}" for v in vals], textposition='outside'))
    fig.update_layout(xaxis_range=[0,112], height=max(height, len(names)*26),
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
with tab1:
    st.caption("미국 장 마감 후 (한국 오전 6~7시) 실행 권장")

    if st.button("▶ 미국 지표 전체 실행", type="primary", use_container_width=True, key="us_run"):
        prog = st.progress(0, text="카나리아 분석 중...")
        canary=get_canary_signal(); prog.progress(10, text="BOFA Heat 분석 중...")
        bofa=get_bofa_heat(); prog.progress(20, text="블러드 인디케이터...")
        blood=get_blood_indicator(); prog.progress(30, text="피어앤그리드 (일간)...")
        fg=get_us_fear_greed(); prog.progress(42, text="피어앤그리드 (월간)...")
        fg_m=get_monthly_fear_greed(); prog.progress(52, text="코포크 지표...")
        coppock=get_coppock(); coppock_fast=get_coppock_fast(); prog.progress(62, text="ZBT 시장 폭...")
        zbt=get_zbt(); prog.progress(72, text="S&P500 RS 상위...")
        sp500_rs=get_sp500_rs(SP500_TOP100); prog.progress(84, text="나스닥100 RS...")
        ndx_rs=get_nasdaq100_rs(NASDAQ100); prog.progress(93, text="미국 섹터 ETF RS...")
        us_sector=get_us_sector_rs(); prog.progress(100, text="완료!")
        prog.empty()

        # ── 1. 종합 신호 ──
        st.markdown('<p class="zone-header">종합 신호</p>', unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        if canary and "error" not in canary:
            color = "sig-green" if canary["mode"]=="공격" else "sig-red"
            c1.markdown(f'<div class="{color}">🐤 카나리아<br><b>{canary["mode"]}</b><br><span style="font-size:0.78rem">QQQ {canary["qqq_mom"]*100:+.1f}%</span></div>', unsafe_allow_html=True)
        if bofa and "error" not in bofa:
            heat_color = "sig-red" if bofa["heat"]>=7.5 else ("sig-yellow" if bofa["heat"]>=5 else "sig-green")
            c2.markdown(f'<div class="{heat_color}">🔥 BOFA Heat<br><b>{bofa["heat"]}/10</b><br><span style="font-size:0.78rem">{_heat_label(bofa["heat"])} · 추세{"✅" if bofa["trend_on"] else "❌"}</span></div>', unsafe_allow_html=True)
        if blood and "error" not in blood:
            c3.markdown(f'<div class="sig-yellow">🩸 블러드<br><b>{blood["value"]:.4f}</b><br><span style="font-size:0.78rem">60MA {blood["vs_ma60"]}</span></div>', unsafe_allow_html=True)
        if zbt and "error" not in zbt:
            zbt_color = "sig-green" if zbt.get("signal") else "sig-yellow"
            c4.markdown(f'<div class="{zbt_color}">📡 ZBT<br><b>{zbt["zbt"]:.3f}</b><br><span style="font-size:0.78rem">{"🟢 반등신호!" if zbt.get("signal") else "⏳ 대기중"}</span></div>', unsafe_allow_html=True)

        st.divider()

        # ── 2. 피어앤그리드 오실레이터 ──
        st.markdown('<p class="zone-header">😱 피어앤그리드 오실레이터</p>', unsafe_allow_html=True)
        _fg_cols = st.columns([1,1,1,1,1,1])
        if fg and "error" not in fg:
            _fg_cols[0].metric("SPX Osc", fg["spx_osc"], f"{'🟢' if fg['spx_osc']>0 else '🔴'} {fg['spx_sentiment']}")
            _fg_cols[1].metric("NDX Osc", fg["ndx_osc"], f"{'🟢' if fg['ndx_osc']>0 else '🔴'} {fg['ndx_sentiment']}")
            _fg_cols[2].metric("SPY 임펄스", fg["spy_impulse"])
            _fg_cols[3].metric("QQQ 임펄스", fg["qqq_impulse"])
            _fg_cols[4].metric("SPY SuperMA", f"{fg['spy_gap']:+.2f}%")
            _fg_cols[5].metric("QQQ SuperMA", f"{fg['qqq_gap']:+.2f}%")
            c1,c2 = st.columns(2)
            c1.metric("SPY TD 매도/매수", f"{fg['spy_td_sell']} / {fg['spy_td_buy']}")
            c2.metric("QQQ TD 매도/매수", f"{fg['qqq_td_sell']} / {fg['qqq_td_buy']}")
            if fg_m and "error" not in fg_m:
                c1.metric("S&P500 월간Osc", fg_m["spx_osc"], f"{'🟢' if fg_m['spx_osc']>0 else '🔴'} {fg_m['spx_sentiment']}")
                c2.metric("NASDAQ 월간Osc", fg_m["ndx_osc"], f"{'🟢' if fg_m['ndx_osc']>0 else '🔴'} {fg_m['ndx_sentiment']}")
            if fg.get("chart"):
                with st.expander("📈 오실레이터 차트 (최근 6개월)"):
                    ch=fg["chart"]
                    _fig=go.Figure()
                    _fig.add_trace(go.Scatter(x=ch["dates"],y=ch["spx_osc"],name="S&P500",line=dict(color="#6A5ACD",width=2)))
                    _fig.add_trace(go.Scatter(x=ch["dates"],y=ch["ndx_osc"],name="NASDAQ",line=dict(color="#00B3B3",width=2)))
                    _fig.add_trace(go.Scatter(x=ch["dates"],y=ch["spy"],name="SPY가격",yaxis="y2",line=dict(color="#666",width=1.5,dash="dot")))
                    _fig.add_hline(y=0,line_dash="dash",line_color="rgba(255,255,255,0.3)")
                    _fig.update_layout(yaxis2=dict(overlaying='y',side='right',showgrid=False),
                                        height=320,margin=dict(l=10,r=60,t=20,b=10),
                                        plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',
                                        font_color='#e0e0e0',legend=dict(orientation='h',y=1.08))
                    _fig.update_xaxes(showgrid=True,gridcolor='rgba(255,255,255,0.07)')
                    _fig.update_yaxes(showgrid=True,gridcolor='rgba(255,255,255,0.07)')
                    st.plotly_chart(_fig,use_container_width=True)

        st.divider()

        # ── 3. 코포크 + ZBT ──
        st.markdown('<p class="zone-header">📊 코포크 · ZBT</p>', unsafe_allow_html=True)
        _cp1, _cp2, _cp3 = st.columns([2,2,1])
        with _cp1:
            st.caption("표준 (ROC 11/14개월, EMA 10)")
            if coppock and "error" not in coppock:
                cols=st.columns(len(coppock))
                for i,(lbl,v) in enumerate(coppock.items()):
                    arr="▲" if v["trend"]=="상승" else "▼"
                    cols[i].metric(lbl,f"{'🟢' if v['pos'] else '🔴'} {v['value']}",f"{arr} {v['trend']}")
        with _cp2:
            st.caption("빠른버전 (ROC 4/6개월, MA 3)")
            if coppock_fast and "error" not in coppock_fast:
                cols=st.columns(len(coppock_fast))
                for i,(lbl,v) in enumerate(coppock_fast.items()):
                    arr="▲" if v["trend"]=="상승" else "▼"
                    cols[i].metric(lbl,f"{'🟢' if v['pos'] else '🔴'} {v['value']}",f"{arr} {v['trend']}")
        with _cp3:
            st.caption("ZBT 시장폭")
            if zbt and "error" not in zbt:
                st.metric("ZBT", f"{zbt['zbt']:.3f}", "🟢 신호!" if zbt.get("signal") else "⏳")
                st.metric("최근최저", f"{zbt['prev_min']:.3f}")
                if zbt.get("vix"): st.metric("VIX", zbt["vix"], "안정✅" if zbt.get("vix_ok") else "⚠️")

        st.divider()

        # ── 4. RS 상위 종목 ──
        st.markdown('<p class="zone-header">🏆 상대강도 상위</p>', unsafe_allow_html=True)
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
        st.markdown('<p class="zone-header">🏭 미국 섹터 ETF 상대강도</p>', unsafe_allow_html=True)
        if us_sector and "error" not in us_sector:
            df_s=pd.DataFrame(us_sector["sectors"])
            df_s["강도"]=df_s["norm_rs"].apply(lambda x:"🟢 강세" if x>=70 else ("🟡 중립" if x>=50 else "🔴 약세"))
            st.dataframe(df_s.rename(columns={"ticker":"티커","name":"섹터","norm_rs":"RS","rs_raw":"RS원시","risk_adj":"변동성모멘텀","강도":"강도"})[["티커","섹터","RS","변동성모멘텀","강도"]],
                use_container_width=True, hide_index=True,
                column_config={"RS":st.column_config.ProgressColumn("RS(0~100)",min_value=0,max_value=100,format="%.1f")})
            with st.expander("📊 섹터 RS 차트"):
                st.plotly_chart(_rs_bar_chart(us_sector["sectors"], name_key="name"), use_container_width=True)
        elif us_sector: st.error(us_sector.get("error"))

# ─────────────────────────────────────────────────────────────────────────────
# 탭 2: 국내 지표
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.caption("장 마감 후 (오후 4시 이후) 실행 권장")
    st.markdown('<p class="zone-header">자동 시장 스캔</p>', unsafe_allow_html=True)
    if st.button("▶ 국내 지표 전체 실행", type="primary", use_container_width=True, key="kr_run"):
        prog=st.progress(0, text="📊 코스피/코스닥 수집 중...")
        market=get_market_summary(); prog.progress(20, text="🏭 업종 ETF 수집 중...")
        sector=get_sector_performance(); prog.progress(40, text="💹 수급 오실레이터 계산 중...")
        supply=get_supply_oscillator(); prog.progress(60, text="🇰🇷 한국 ETF RS 계산 중...")
        kr_etf=get_kr_etf_rs(); prog.progress(80, text="🏠 빈집 스크리닝 중...")
        binzip=get_binzip_stocks(supply_data=supply); prog.progress(100, text="✅ 완료!")
        prog.empty()

        # ── 지수 현황 ──
        st.markdown('<p class="zone-header">📊 지수 현황</p>', unsafe_allow_html=True)
        if market and "error" not in market:
            kp=market["kospi"]; kq=market["kosdaq"]
            c1,c2=st.columns(2)
            c1.metric(f"코스피 ({market['date']})", f"{kp['close']:,.2f}",
                      f"{kp['chg_pct']:+.2f}% · 주간 {kp['week_pct']:+.2f}%",
                      delta_color="normal")
            c2.metric("코스닥", f"{kq['close']:,.2f}",
                      f"{kq['chg_pct']:+.2f}% · 주간 {kq['week_pct']:+.2f}%",
                      delta_color="normal")
        elif market: st.error(market.get("error"))

        st.divider()

        # ── 업종 강세/약세 ──
        st.markdown('<p class="zone-header">🏭 업종 강세 / 약세</p>', unsafe_allow_html=True)
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
        st.markdown('<p class="zone-header">💹 수급 오실레이터</p>', unsafe_allow_html=True)
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
        st.markdown('<p class="zone-header">🇰🇷 한국 ETF 상대강도</p>', unsafe_allow_html=True)
        if kr_etf and "error" not in kr_etf:
            show=kr_etf.get("strong") or kr_etf.get("all",[])[:10]
            if show:
                df_kr=pd.DataFrame(show)
                df_kr["강도"]=df_kr["norm_rs"].apply(lambda x:"🟢 강세" if x>=70 else "🟡 보통")
                st.dataframe(df_kr.rename(columns={"name":"ETF명","norm_rs":"RS(0~100)","risk_adj":"변동성조정모멘텀","강도":"강도"})[["ETF명","RS(0~100)","변동성조정모멘텀","강도"]],
                    use_container_width=True, hide_index=True,
                    column_config={"RS(0~100)":st.column_config.ProgressColumn("RS(0~100)",min_value=0,max_value=100,format="%.1f")})
            with st.expander("📊 한국 ETF RS 차트"):
                _all=kr_etf.get("all",[])[:25]; _all_s=sorted(_all,key=lambda x:x["norm_rs"])
                _names=[r["name"] for r in _all_s]; _vals=[r["norm_rs"] for r in _all_s]
                _colors=["#00c853" if v>=70 else ("#ffc107" if v>=50 else "#ff4b4b") for v in _vals]
                _fig=go.Figure(go.Bar(x=_vals,y=_names,orientation='h',marker_color=_colors,
                                      text=[f"{v:.1f}" for v in _vals],textposition='outside'))
                _fig.update_layout(xaxis_range=[0,110],height=max(320,len(_names)*26),
                                    margin=dict(l=10,r=50,t=10,b=10),
                                    plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',font_color='#e0e0e0')
                _fig.add_vline(x=70,line_dash="dash",line_color="#00c853",opacity=0.5)
                _fig.add_vline(x=50,line_dash="dash",line_color="#ffc107",opacity=0.5)
                st.plotly_chart(_fig,use_container_width=True)
        elif kr_etf: st.error(kr_etf.get("error"))

        st.divider()

        # ── 빈집 주도주 ──
        bz=binzip or {}; bl=bz.get("binzip",[]); ss=" + ".join(bz.get("sectors",[])) or "주도업종"
        st.markdown(f'<p class="zone-header">🏠 빈집 주도주 [{ss}]</p>', unsafe_allow_html=True)
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
    st.markdown('<p class="zone-header">📈 개별종목 RS</p>', unsafe_allow_html=True)
    rs_xl_file = st.file_uploader(
        "종목상대강도데이터.xlsx 업로드 (선택)", type=["xlsx"], key="rs_xl_file",
        help="업로드 시 Yahoo Finance 대신 로컬 Excel 종가 데이터로 RS 계산 (빠르고 정확)"
    )
    if rs_xl_file:
        rs_xl_file.seek(0); st.session_state["c_rs_bytes"] = rs_xl_file.read()
    if st.button("▶ 개별종목 RS 스크리닝", key="kr_stock_rs_run", use_container_width=True):
        if "c_rs_bytes" in st.session_state:
            _df_rs_close = pd.read_excel(io.BytesIO(st.session_state["c_rs_bytes"]), sheet_name=0, engine="openpyxl")
            with st.spinner(f"Excel 로컬 데이터로 RS 계산 중 ({len(_df_rs_close.columns)-2}종목)..."):
                _tmp_rs = calc_kr_stock_rs_excel(_df_rs_close, top_n=15)
            if "error" not in _tmp_rs:
                st.session_state["c_kr_rs"] = _tmp_rs
                st.session_state["c_kr_rs_src"] = f"📂 종목상대강도데이터.xlsx | {len(_df_rs_close)}행 × {len(_df_rs_close.columns)-2}종목"
            else:
                st.error(_tmp_rs["error"])
        else:
            with st.spinner("한국 개별종목 RS 계산 중 (40종목씩 배치)..."):
                _tmp_rs = get_kr_stock_rs(top_n=15)
            if "error" not in _tmp_rs:
                st.session_state["c_kr_rs"] = _tmp_rs
                st.session_state["c_kr_rs_src"] = "📡 Yahoo Finance"
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
        with st.expander("📊 개별종목 RS 차트"):
            _show=kr_rs.get("strong") or kr_rs.get("all",[])[:15]
            _show_s=sorted(_show,key=lambda x:x["norm_rs"])
            _names=[r["name"] for r in _show_s]; _vals=[r["norm_rs"] for r in _show_s]
            _colors=["#00c853" if v>=70 else ("#ffc107" if v>=50 else "#ff4b4b") for v in _vals]
            _fig=go.Figure(go.Bar(x=_vals,y=_names,orientation='h',marker_color=_colors,
                                  text=[f"{v:.1f}" for v in _vals],textposition='outside'))
            _fig.update_layout(xaxis_range=[0,110],height=max(300,len(_names)*28),
                                margin=dict(l=10,r=50,t=10,b=10),
                                plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',font_color='#e0e0e0')
            _fig.add_vline(x=70,line_dash="dash",line_color="#00c853",opacity=0.5)
            st.plotly_chart(_fig,use_container_width=True)

    st.divider()

    # ── 한국 ETF RS (Excel 정밀) ──
    st.markdown('<p class="zone-header">📊 한국 ETF RS 〔Excel〕</p>', unsafe_allow_html=True)
    etf_rs_xl_file = st.file_uploader(
        "etf상대강도데이터.xlsx 업로드 (데이터 시트 포함)", type=["xlsx"], key="etf_rs_xl_file",
        help="데이터 시트: DATE + 코스피 + ETF 종가 컬럼 — linear Mansfield RS 60/120/250d"
    )
    if etf_rs_xl_file:
        try:
            etf_rs_xl_file.seek(0)
            _etf_xl_sheets = pd.ExcelFile(etf_rs_xl_file, engine="openpyxl").sheet_names
            _etf_sn = next((s for s in _etf_xl_sheets if '데이터' in s), _etf_xl_sheets[0])
            etf_rs_xl_file.seek(0)
            df_etf_rs = pd.read_excel(etf_rs_xl_file, sheet_name=_etf_sn, engine="openpyxl")
            with st.spinner(f"ETF RS 계산 중 ({len(df_etf_rs.columns)-2}개 ETF)..."):
                _tmp_etf = calc_kr_etf_rs_excel(df_etf_rs, top_n=15)
            if "error" not in _tmp_etf:
                st.session_state["c_etf_rs"] = _tmp_etf
                st.session_state["c_etf_rs_meta"] = f"📂 시트: {_etf_sn} | {len(df_etf_rs)}행 | 전체 {len(_tmp_etf.get('all',[]))}개 ETF | RS≥70: {len([r for r in _tmp_etf.get('all',[]) if r['norm_rs']>=70])}개"
            else:
                st.error(_tmp_etf["error"])
        except Exception as e:
            st.error(f"ETF RS 파일 읽기 오류: {e}")
    if "c_etf_rs" in st.session_state:
        etf_rs_xl = st.session_state["c_etf_rs"]
        _is_etf_cached = not etf_rs_xl_file
        st.caption(st.session_state.get("c_etf_rs_meta", "") + ("  ⚡ 캐시" if _is_etf_cached else ""))
        show_etf = etf_rs_xl.get("strong") or etf_rs_xl.get("all", [])[:15]
        if show_etf:
            df_etf_show = pd.DataFrame(show_etf)
            st.dataframe(
                df_etf_show.rename(columns={"name":"ETF명","norm_rs":"RS(0~100)","rs_raw":"RS원시","risk_adj":"변동성조정모멘텀"})[["ETF명","RS(0~100)","RS원시","변동성조정모멘텀"]],
                use_container_width=True, hide_index=True,
                column_config={"RS(0~100)":st.column_config.ProgressColumn("RS(0~100)",min_value=0,max_value=100,format="%.1f")}
            )
        with st.expander("📊 ETF RS 차트 (Excel)"):
            _all_etf = etf_rs_xl.get("all", [])[:25]
            _all_etf_s = sorted(_all_etf, key=lambda x: x["norm_rs"])
            _names_e = [r["name"] for r in _all_etf_s]
            _vals_e = [r["norm_rs"] for r in _all_etf_s]
            _colors_e = ["#00c853" if v>=70 else ("#ffc107" if v>=50 else "#ff4b4b") for v in _vals_e]
            _fig_e = go.Figure(go.Bar(x=_vals_e, y=_names_e, orientation='h', marker_color=_colors_e,
                                     text=[f"{v:.1f}" for v in _vals_e], textposition='outside'))
            _fig_e.update_layout(xaxis_range=[0,110], height=max(320, len(_names_e)*26),
                                 margin=dict(l=10,r=50,t=10,b=10),
                                 plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#e0e0e0')
            _fig_e.add_vline(x=70, line_dash="dash", line_color="#00c853", opacity=0.5)
            _fig_e.add_vline(x=50, line_dash="dash", line_color="#ffc107", opacity=0.5)
            st.plotly_chart(_fig_e, use_container_width=True)

    st.divider()

    # ── 한국 F&G 오실레이터 ──
    st.markdown('<p class="zone-header">😨 한국 F&G 오실레이터</p>', unsafe_allow_html=True)
    st.caption("자동(참고용): 무료 대체지표로 방향성만 확인  |  Excel(정밀): 원본 VKOSPI·국채선물·P/C ATM")
    if st.button("▶ 자동 계산 〔참고용〕", key="kr_fg_auto_run", use_container_width=True, type="primary"):
            with st.spinner("한국 F&G 오실레이터 자동 계산 중..."):
                kr_fg_auto = get_kr_fg_auto()
            if "error" in kr_fg_auto:
                st.error(kr_fg_auto["error"])
            else:
                st.caption(f"기준일: {kr_fg_auto['date']}  |  {kr_fg_auto.get('source','')}")
                for label, v in kr_fg_auto["results"].items():
                    st.markdown(f"**{label}**")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("오실레이터", v["osc"], f"{'🟢' if v['osc']>0 else '🔴'} {v['sentiment']}")
                    c2.metric("임펄스", v["impulse"])
                    c3.metric("TD 매도/매수", f"{v['td_sell']} / {v['td_buy']}")
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
                                               font_color='#e0e0e0', showlegend=False)
                            _fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.08)')
                            _fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.08)')
                            st.plotly_chart(_fig, use_container_width=True)
    st.caption("📂 피어앤그리드.xlsx (KOSPI / KOSDAQ 시트) — 정밀 분석")
    fg_file = st.file_uploader("피어앤그리드.xlsx 업로드", type=["xlsx"], key="kr_fg_file")
    if fg_file:
        try:
            fg_file.seek(0)
            _fg_bytes = fg_file.read()
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
        _is_fg_cached = not fg_file
        st.markdown(f"**기준일: {kr_fg['date']}**" + ("  ⚡ 캐시" if _is_fg_cached else ""))
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
                                    font_color='#e0e0e0',showlegend=False)
                _fig.update_xaxes(showgrid=True,gridcolor='rgba(255,255,255,0.08)')
                _fig.update_yaxes(showgrid=True,gridcolor='rgba(255,255,255,0.08)')
                st.plotly_chart(_fig,use_container_width=True)

    st.divider()

    # ── 수급 자동 스크리닝 ──
    st.markdown('<p class="zone-header">📡 외국인 수급 스크리닝 〔참고용〕</p>', unsafe_allow_html=True)
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
                """수급 오실레이터 차트 — 가격(우축) + 일별순매수바+MA5(좌축)"""
                ch = row.get("chart")
                if not ch: return
                val = row["net_nd_bil"]
                sign = "+" if val >= 0 else ""
                with st.expander(f"📊 {label_prefix}{row['name']} ({row['ticker']}) — {sign}{val:.1f}억"):
                    bar_colors = ["#00c853" if v >= 0 else "#ff4b4b" for v in ch["daily"]]
                    fig_s = make_subplots(specs=[[{"secondary_y": True}]])
                    # 일별 순매수 (좌축, 막대)
                    fig_s.add_trace(go.Bar(
                        x=ch["dates"], y=ch["daily"],
                        name="일별 순매수(억)", marker_color=bar_colors, opacity=0.7,
                    ), secondary_y=False)
                    # 5일 MA 시그널 (좌축, 선)
                    fig_s.add_trace(go.Scatter(
                        x=ch["dates"], y=ch["ma5"],
                        name="MA5", line=dict(color="#FF8C00", width=2),
                    ), secondary_y=False)
                    # 20일 누적합 (좌축, 점선)
                    fig_s.add_trace(go.Scatter(
                        x=ch["dates"], y=ch["cum20"],
                        name=f"{supply_days}일 누적", line=dict(color="#6A5ACD", width=1.5, dash="dot"),
                    ), secondary_y=False)
                    # 종가 (우축, 선)
                    fig_s.add_trace(go.Scatter(
                        x=ch["dates"], y=ch["price"],
                        name="종가", line=dict(color="#E0E0E0", width=2),
                    ), secondary_y=True)
                    fig_s.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)",
                                    secondary_y=False)
                    fig_s.update_layout(
                        height=400, margin=dict(l=10, r=60, t=30, b=10),
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        font_color="#e0e0e0", barmode="overlay",
                        legend=dict(orientation="h", y=1.08),
                    )
                    fig_s.update_yaxes(title_text="순매수(억원)", secondary_y=False,
                                       showgrid=True, gridcolor="rgba(255,255,255,0.08)")
                    fig_s.update_yaxes(title_text="종가(원)", secondary_y=True, showgrid=False)
                    fig_s.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
                    st.plotly_chart(fig_s, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**🟢 외국인 순매수 TOP20**")
                for i, row in enumerate(sup["top20"], 1):
                    val = row["net_nd_bil"]
                    sign = "+" if val >= 0 else ""
                    st.markdown(f"`{i:2d}` **{row['name']}** ({row['ticker']}) — {sign}{val:.1f}억")
            with c2:
                st.markdown("**🔴 외국인 순매도 TOP20**")
                for i, row in enumerate(sup["worst20"], 1):
                    val = row["net_nd_bil"]
                    sign = "+" if val >= 0 else ""
                    st.markdown(f"`{i:2d}` **{row['name']}** ({row['ticker']}) — {sign}{val:.1f}억")

            st.markdown("#### 📈 수급 오실레이터 — 순매수 TOP10")
            for row in sup["top20"][:10]:
                _draw_supply_osc(row, "🟢 ")
            st.markdown("#### 📉 수급 오실레이터 — 순매도 TOP10")
            for row in sup["worst20"][:10]:
                _draw_supply_osc(row, "🔴 ")

    st.divider()

    # ── KRX 기관 순매수 + 거래대금 강도 ──
    st.markdown('<p class="zone-header">🏦 기관 순매수 + 거래대금 강도 〔KRX 자동〕</p>', unsafe_allow_html=True)
    st.caption("KOSPI/KOSDAQ 시장 전체 기관·외국인 순매수 + 거래대금 강도 | 출처: KRX (pykrx)")
    if st.button("▶ 기관 수급 + 거래대금 강도 조회 (KRX)", key="kr_inst_flow_run", use_container_width=True):
        with st.spinner("KRX 데이터 수집 중 (10~20초)..."):
            inst_flow = get_krx_inst_market_flow(days=10)
            vol_str = get_krx_volume_strength()
        if "error" not in vol_str:
            c1, c2, c3 = st.columns(3)
            c1.metric("KOSPI 거래대금 (오늘)", f"{vol_str['today_tril']:.2f}조")
            c2.metric("20일 평균", f"{vol_str['ma20_tril']:.2f}조")
            c3.metric("거래대금 강도", f"{vol_str['rotation_rate']:.0f}%", vol_str["level"])
            if vol_str.get("chart"):
                ch = vol_str["chart"]
                fig_tv = go.Figure()
                fig_tv.add_trace(go.Bar(x=ch["dates"], y=ch["values"], name="일별 거래대금",
                                        marker_color="#6A5ACD", opacity=0.7))
                fig_tv.add_trace(go.Scatter(x=ch["dates"], y=ch["ma20"], name="MA20",
                                            line=dict(color="#FF8C00", width=2, dash="dot")))
                fig_tv = _chart_layout(fig_tv, height=220)
                fig_tv.update_yaxes(title_text="거래대금(조원)")
                st.plotly_chart(fig_tv, use_container_width=True)
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
                    st.plotly_chart(fig_inst, use_container_width=True)
        else:
            st.caption(f"기관 순매수: {inst_flow['error']}")

    st.divider()

    # ── 컨센 가속 자동 ──
    st.markdown('<p class="zone-header">🤖 컨센서스 스크리닝 〔참고용〕</p>', unsafe_allow_html=True)
    st.caption("WiseReport: EPS성장률+매수비율+TP인상비율 합산 스코어 (원본과 근사치)")
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
    st.markdown('<p class="zone-header">📋 컨센 가속 & 수급 〔Excel 정밀〕</p>', unsafe_allow_html=True)
    st.caption("📂 데이터 정리.xlsx (db 시트) — 원본 EPS가속(1M>3M) + 외국인/기관 교집합")
    consensus_file = st.file_uploader("데이터 정리.xlsx 업로드", type=["xlsx"], key="consensus_file")
    if consensus_file:
        try:
            df_db = pd.read_excel(consensus_file, sheet_name="db", engine="openpyxl")
            _tmp_cons = calc_consensus_excel(df_db)
            if "error" not in _tmp_cons:
                st.session_state["c_consensus"] = _tmp_cons
            else:
                st.error(_tmp_cons["error"])
        except Exception as e:
            st.error(f"파일 읽기 오류: {e}")
    if "c_consensus" in st.session_state:
        cons = st.session_state["c_consensus"]
        _is_cons_cached = not consensus_file
        if _is_cons_cached:
            st.caption("⚡ 캐시")
        st.metric("EPS 가속 통과 종목", f"{cons['eps_passed']}개")
        c1,c2 = st.columns(2)
        with c1:
            st.markdown("**외국인 수급강도 TOP20**")
            for i,n in enumerate(cons["foreign_top20"],1): st.markdown(f"`{i:2d}` {n}")
        with c2:
            st.markdown("**기관 수급강도 TOP20**")
            for i,n in enumerate(cons["inst_top20"],1): st.markdown(f"`{i:2d}` {n}")
        if cons["common"]:
            st.markdown("**🎯 외국인 & 기관 공통 (교집합)**")
            st.success(" · ".join(cons["common"]))
        else:
            st.info("교집합 없음")

# ─────────────────────────────────────────────────────────────────────────────
# 탭 3: 종목 분석
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<p class="zone-header">🎯 매수 타점 통계 분석</p>', unsafe_allow_html=True)
    st.caption("1년 데이터 기반 — 고가/저가/시가 평균 괴리율 (지정가 매수 참고용)")

    _sel_opts = [""] + sorted([f"{kr} ({t})" for t, kr in TICKER_NAMES.items()], key=lambda x: x[0])
    sel_stock = st.selectbox("📋 목록에서 선택 (한글명 또는 영문 티커로 검색 가능)", _sel_opts, index=0, key="bt_sel")
    ticker_input = st.text_input("또는 직접 입력 (티커·한글명 모두 가능, 쉼표로 여러 개)", placeholder="NVDA, 엔비디아, 005930.KS", key="bt_ticker")

    st.divider()
    st.caption("📂 추가 Excel 업로드 (선택) — 업로드 시 자동 계산과 함께 정밀 비교 표시")
    _col_xu1, _col_xu2, _col_xu3 = st.columns(3)
    with _col_xu1:
        trend_supply_file = st.file_uploader(
            "추세판별기(수급까지체크).xlsx", type=["xlsx"], key="trend_supply_file",
            help="DB(2) 시트 — 5일간 기관 매수수량 오실레이터 추가"
        )
    with _col_xu2:
        trading_xl_file = st.file_uploader(
            "국장 거래대금 강도.xlsx", type=["xlsx"], key="trading_xl_file",
            help="_RotationRate_ 컬럼 — 거래대금 강도 비교 표시"
        )
    with _col_xu3:
        weekly_xl_file = st.file_uploader(
            "추세판별기(주간).xlsx", type=["xlsx"], key="weekly_xl_file",
            help="DB 시트 — 주간 OHLCV로 CMF/임펄스/TD 계산 (HTS 원천 데이터)"
        )
    if trend_supply_file:
        trend_supply_file.seek(0); st.session_state["c_supply_bytes"] = trend_supply_file.read()
    if trading_xl_file:
        trading_xl_file.seek(0); st.session_state["c_trading_bytes"] = trading_xl_file.read()
    if weekly_xl_file:
        weekly_xl_file.seek(0); st.session_state["c_weekly_bytes"] = weekly_xl_file.read()
    _eff_supply = io.BytesIO(st.session_state["c_supply_bytes"]) if not trend_supply_file and "c_supply_bytes" in st.session_state else trend_supply_file
    _eff_trading = io.BytesIO(st.session_state["c_trading_bytes"]) if not trading_xl_file and "c_trading_bytes" in st.session_state else trading_xl_file
    _eff_weekly = io.BytesIO(st.session_state["c_weekly_bytes"]) if not weekly_xl_file and "c_weekly_bytes" in st.session_state else weekly_xl_file

    if st.button("🔍 분석 시작", key="bt_run", type="primary", use_container_width=True):
        tickers_to_run = []
        if sel_stock:
            _t = sel_stock.split("(")[-1].rstrip(")")
            tickers_to_run.append(_t.strip())
        if ticker_input:
            for raw in [x.strip() for x in ticker_input.split(",") if x.strip()]:
                resolved = NAME_TO_TICKER.get(raw.lower(), raw)
                tickers_to_run.append(resolved)
        if not tickers_to_run:
            st.warning("종목을 선택하거나 입력해주세요.")
        for t in tickers_to_run:
            with st.spinner(f"{t} 분석 중..."):
                res = get_buy_timing(t)
            if "error" in res:
                st.error(f"❌ {t}: {res['error']}")
            else:
                kr = TICKER_NAMES.get(t, res.get("name", ""))
                st.markdown(f"#### 📌 **{t}** {kr} — 현재가: `{res['price']:,}`")
                c1,c2 = st.columns(2)
                c1.metric("고가→종가 평균 하락", f"{res['고가종가하락']:+.2f}%", help="장중 고점 대비 종가 평균 낙폭. 지정가보다 높게 올라갔다가 내려오는 정도")
                c2.metric("전일종가→당일저가 괴리", f"{res['저가종가괴리']:+.2f}%", help="전일 종가 대비 당일 저가 평균 괴리. 갭하락 포함")
                c1.metric("시가→저가 평균 낙폭", f"{res['시가저가괴리']:+.2f}%", help="시가 기준 장중 최대 낙폭 평균")
                c2.metric("전일종가→당일고가", f"{res['전일종가고가']:+.2f}%", help="전일 종가 대비 당일 고가 평균 괴리율")
                c1.metric("시가→당일고가 평균", f"{res['시가고가괴리']:+.2f}%", help="시가 대비 장중 고점까지 평균 상승폭")
                st.caption(f"💡 힌트: 시가 대비 저가 괴리({res['시가저가괴리']:+.2f}%) → 시가보다 그 정도 낮게 지정가 설정")
                st.divider()

        st.divider()
        st.markdown('<p class="zone-header">📊 주간 CMF 추세판별기</p>', unsafe_allow_html=True)
        for t in tickers_to_run:
            with st.spinner(f"{t} 주간 추세 분석 중..."):
                wt = get_weekly_trend(t)
            kr = TICKER_NAMES.get(t, "")
            if "error" in wt:
                st.error(f"❌ {t} {kr}: {wt['error']}")
            else:
                st.markdown(f"#### 📌 **{t}** {kr}")
                c1,c2,c3 = st.columns(3)
                c1.metric("CMF (4주)", f"{wt['cmf']:.4f}", "🟢 자금유입" if wt['cmf']>0 else "🔴 자금유출")
                c2.metric("주간 임펄스", wt["impulse_weekly"])
                sig = "🟢 매수신호" if wt["buy_signal"] else ("🔴 매도신호" if wt["sell_signal"] else "⏳ 대기")
                c3.metric("CMF 신호", sig, f"최근4주 매수: {wt['recent_buy_4w']}회")
                c1.metric("주봉 TD 매도/매수", f"{wt['w_td_sell']} / {wt['w_td_buy']}")
                c2.metric("일봉 TD 매도/매수", f"{wt['d_td_sell']} / {wt['d_td_buy']}")
                c3.metric("월봉 TD 매도/매수", f"{wt['m_td_sell']} / {wt['m_td_buy']}")
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
                                            legend=dict(orientation='h',y=1.08))
                        _fig.update_xaxes(showgrid=True,gridcolor='rgba(255,255,255,0.08)')
                        _fig.update_yaxes(showgrid=True,gridcolor='rgba(255,255,255,0.08)')
                        st.plotly_chart(_fig,use_container_width=True)
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
                    st.markdown('<p class="zone-header">📅 주간 추세판별기 〔Excel HTS〕</p>', unsafe_allow_html=True)
                    st.caption(f"시트: {_wk_sn} | {wkr['rows']}주 | {wkr['date_range']}")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("CMF (4주)", f"{wkr['cmf']:.4f}", "🟢 자금유입" if wkr['cmf'] > 0 else "🔴 자금유출")
                    c2.metric("주간 임펄스", wkr["impulse_weekly"])
                    sig = "🟢 매수신호" if wkr["buy_signal"] else ("🔴 매도신호" if wkr["sell_signal"] else "⏳ 대기")
                    c3.metric("CMF 신호", sig, f"최근4주 매수: {wkr['recent_buy_4w']}회")
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
                                                legend=dict(orientation='h', y=1.08))
                            _fig2.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.08)')
                            _fig2.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.08)')
                            st.plotly_chart(_fig2, use_container_width=True)
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
                st.markdown('<p class="zone-header">🏦 추세판별기 수급 〔Excel〕</p>', unsafe_allow_html=True)
                if buy_col and close_col_tsd:
                    df_tsd[date_col_tsd] = pd.to_datetime(df_tsd[date_col_tsd], errors='coerce')
                    df_tsd = df_tsd.dropna(subset=[date_col_tsd]).sort_values(date_col_tsd)
                    # 유효 데이터만 (종가 != 0)
                    df_tsd = df_tsd[pd.to_numeric(df_tsd[close_col_tsd], errors='coerce').fillna(0) != 0]
                    df_tsd[buy_col] = pd.to_numeric(df_tsd[buy_col], errors='coerce').fillna(0)
                    df_tsd[close_col_tsd] = pd.to_numeric(df_tsd[close_col_tsd], errors='coerce')
                    dates_tsd = [str(d.date()) for d in df_tsd[date_col_tsd]]
                    buy_vals_tsd = df_tsd[buy_col].tolist()
                    bar_colors_tsd = ["#00c853" if v >= 0 else "#ff4b4b" for v in buy_vals_tsd]
                    fig_tsd = make_subplots(specs=[[{"secondary_y": True}]])
                    fig_tsd.add_trace(go.Bar(x=dates_tsd, y=buy_vals_tsd, name=buy_label,
                                             marker_color=bar_colors_tsd, opacity=0.8), secondary_y=False)
                    if sell_col:
                        df_tsd[sell_col] = pd.to_numeric(df_tsd[sell_col], errors='coerce').fillna(0)
                        fig_tsd.add_trace(go.Scatter(x=dates_tsd, y=df_tsd[sell_col].tolist(),
                                                     name=sell_label,
                                                     line=dict(color="#FF8C00", width=1.5, dash="dot")), secondary_y=False)
                    fig_tsd.add_trace(go.Scatter(x=dates_tsd, y=df_tsd[close_col_tsd].tolist(), name="종가",
                                                  line=dict(color="#E0E0E0", width=2)), secondary_y=True)
                    fig_tsd.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)", secondary_y=False)
                    fig_tsd.update_layout(height=400, margin=dict(l=10,r=60,t=30,b=10),
                                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                          font_color="#e0e0e0", barmode="overlay",
                                          legend=dict(orientation="h", y=1.08))
                    fig_tsd.update_yaxes(title_text="수량", secondary_y=False,
                                         showgrid=True, gridcolor="rgba(255,255,255,0.08)")
                    fig_tsd.update_yaxes(title_text="종가(원)", secondary_y=True, showgrid=False)
                    fig_tsd.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
                    st.plotly_chart(fig_tsd, use_container_width=True)
                    st.caption(f"시트: {_tsd_sn} | 매수: {buy_col} | 매도: {sell_col}")
                else:
                    st.warning(f"매수 컬럼 미감지. 컬럼 목록: {list(df_tsd.columns)}")
            except Exception as e:
                st.error(f"추세판별기 파일 읽기 오류: {e}")

        st.divider()
        st.markdown('<p class="zone-header">💰 거래대금 강도</p>', unsafe_allow_html=True)
        for t in tickers_to_run:
            with st.spinner(f"{t} 거래대금 강도 분석 중..."):
                ti = get_trading_intensity(t)
            kr = TICKER_NAMES.get(t, "")
            if "error" in ti:
                st.error(f"❌ {t} {kr}: {ti['error']}")
            else:
                st.markdown(f"#### 📌 **{t}** {kr}")
                c1,c2,c3 = st.columns(3)
                c1.metric("거래대금 강도 TI", f"{ti['ti']:.1f}", ti["signal_text"])
                c2.metric("TI MA3", f"{ti['ti_ma3']:.1f}")
                c3.metric("TI Signal (EMA7)", f"{ti['ti_signal']:.1f}")
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
                                            font_color='#e0e0e0',legend=dict(orientation='h',y=1.1))
                        _fig.update_xaxes(showgrid=True,gridcolor='rgba(255,255,255,0.08)')
                        _fig.update_yaxes(showgrid=True,gridcolor='rgba(255,255,255,0.08)')
                        st.plotly_chart(_fig,use_container_width=True)
                st.divider()

        st.divider()
        # ── 거래대금 강도 Excel _RotationRate_ (선택) ──
        if _eff_trading:
            try:
                df_txi = pd.read_excel(_eff_trading, sheet_name=0, engine="openpyxl")
                st.markdown('<p class="zone-header">📊 거래대금 강도 〔Excel RotationRate〕</p>', unsafe_allow_html=True)
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
                                         legend=dict(orientation="h", y=1.08))
                    fig_rr.update_yaxes(title_text="_RotationRate_", secondary_y=False,
                                        showgrid=True, gridcolor="rgba(255,255,255,0.08)")
                    fig_rr.update_yaxes(title_text="종가(원)", secondary_y=True, showgrid=False)
                    fig_rr.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
                    st.plotly_chart(fig_rr, use_container_width=True)
                    st.caption(f"컬럼: {rr_col} | 출처: 국장 거래대금 강도 Excel")
                else:
                    st.warning(f"_RotationRate_ 컬럼 미감지. 컬럼 목록: {list(df_txi.columns)}")
            except Exception as e:
                st.error(f"거래대금 강도 파일 읽기 오류: {e}")

        st.divider()
        st.markdown('<p class="zone-header">📡 외국인 수급</p>', unsafe_allow_html=True)
        st.caption("한국 종목(.KS/.KQ)만 지원 | 네이버 파이낸스 외국인 순매수 60일")
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
            sign = "+" if so["net_agg_bil"] >= 0 else ""
            c1, c2 = st.columns(2)
            c1.metric("20일 외국인 순매수", f"{sign}{so['net_agg_bil']:.1f}억",
                      "🟢 순매수" if so["net_agg_bil"] >= 0 else "🔴 순매도")
            c2.metric("현재가", f"{so['latest_close']:,}원")
            ch = so["chart"]
            bar_colors = ["#00c853" if v >= 0 else "#ff4b4b" for v in ch["daily"]]
            fig_so = make_subplots(specs=[[{"secondary_y": True}]])
            fig_so.add_trace(go.Bar(
                x=ch["dates"], y=ch["daily"], name="일별 순매수(억)",
                marker_color=bar_colors, opacity=0.7,
            ), secondary_y=False)
            fig_so.add_trace(go.Scatter(
                x=ch["dates"], y=ch["ma5"], name="MA5",
                line=dict(color="#FF8C00", width=2),
            ), secondary_y=False)
            fig_so.add_trace(go.Scatter(
                x=ch["dates"], y=ch["cum20"], name="20일 누적",
                line=dict(color="#6A5ACD", width=1.5, dash="dot"),
            ), secondary_y=False)
            fig_so.add_trace(go.Scatter(
                x=ch["dates"], y=ch["price"], name="종가",
                line=dict(color="#E0E0E0", width=2),
            ), secondary_y=True)
            fig_so.add_hline(y=0, line_dash="dash",
                              line_color="rgba(255,255,255,0.3)", secondary_y=False)
            fig_so.update_layout(
                height=420, margin=dict(l=10, r=60, t=30, b=10),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e0e0e0", barmode="overlay",
                legend=dict(orientation="h", y=1.08),
            )
            fig_so.update_yaxes(title_text="순매수(억원)", secondary_y=False,
                                 showgrid=True, gridcolor="rgba(255,255,255,0.08)")
            fig_so.update_yaxes(title_text="종가(원)", secondary_y=True, showgrid=False)
            fig_so.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
            st.plotly_chart(fig_so, use_container_width=True)
            st.divider()

        st.markdown('<p class="zone-header">🏦 기관 수급 〔KRX 자동〕</p>', unsafe_allow_html=True)
        st.caption("개별 종목 기관/외국인 순매수 — KRX (pykrx) | 한국 종목(.KS/.KQ)만 지원")
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
            c1, c2 = st.columns(2)
            c1.metric("20일 기관 순매수", f"{si_sign}{si['inst_sum_bil']:.1f}억",
                      "🟢 순매수" if si["inst_sum_bil"] >= 0 else "🔴 순매도")
            c2.metric("20일 외국인 순매수(KRX)", f"{sf_sign}{si.get('frgn_sum_bil', 0):.1f}억",
                      "🟢 순매수" if si.get("frgn_sum_bil", 0) >= 0 else "🔴 순매도")
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
                                  font_color="#e0e0e0", legend=dict(orientation="h", y=1.08))
            fig_si.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
            fig_si.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
            st.plotly_chart(fig_si, use_container_width=True)
