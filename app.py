import time, warnings, logging
import numpy as np
import pandas as pd
import yfinance as yf
import requests
import streamlit as st
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

st.set_page_config(page_title="퇴근길 주식", page_icon="📈", layout="centered")
st.markdown("""<style>
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px;
    padding: 10px 14px;
    margin: 3px 0;
}
[data-testid="stMetricValue"] { font-size: 1.1rem !important; }
[data-testid="stMetricLabel"] { font-size: 0.82rem !important; }
.stTabs [data-baseweb="tab"] { font-weight: 700; font-size: 0.95rem; }
hr { margin: 0.6rem 0; }
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
        irx = px["^IRX"]/100; t10y = px["^TNX"]/10
        hyg_yield = getattr(yf.Ticker("HYG").fast_info,"dividend_yield",None) or yf.Ticker("HYG").info.get("dividendYield",0.06)
        blood = (irx/(hyg_yield-t10y)).dropna()
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
        return {"spx_osc":round(float(last["SPX_Osc"].iloc[-1]),4),"ndx_osc":round(float(last["NDX_Osc"].iloc[-1]),4),
                "spx_sentiment":"탐욕" if last["SPX_Osc"].iloc[-1]>0 else "공포",
                "ndx_sentiment":"탐욕" if last["NDX_Osc"].iloc[-1]>0 else "공포",
                "spy_gap":round(float(data["SPY_GapPct"].dropna().iloc[-1]),2),
                "qqq_gap":round(float(data["QQQ_GapPct"].dropna().iloc[-1]),2),
                "spy_impulse":impulse(data,"SPY"),"qqq_impulse":impulse(data,"QQQ"),
                "spy_td_sell":spy_ts,"spy_td_buy":spy_tb,"qqq_td_sell":qqq_ts,"qqq_td_buy":qqq_tb}
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
        raw=yf.download(tks+["SPY"],period="1y",auto_adjust=True,progress=False)
        if raw.empty: return {"error":"데이터 없음"}
        data=raw["Close"].ffill().dropna(axis=1,how="any")
        if "SPY" not in data.columns: return {"error":"SPY 없음"}
        results=[]
        for t in tks:
            if t not in data.columns: continue
            etf=data[t]; rel=etf/data["SPY"]; ma52=rel.rolling(52).mean()
            rs_raw=float(((rel/ma52)-1).dropna().iloc[-1]*100) if not ((rel/ma52)-1).dropna().empty else 0
            norm_rs=round(100*(1/(1+np.exp(-rs_raw/12))),1)
            mom3=float(etf.pct_change().rolling(63).mean().iloc[-1]) if len(etf)>=63 else 0
            vol3=float(etf.pct_change().rolling(63).std().iloc[-1]) if len(etf)>=63 else 1
            risk_adj=round((mom3/vol3)*100,2) if vol3>0 else 0
            results.append({"ticker":t,"name":US_SECTOR_ETFS.get(t,t),"norm_rs":norm_rs,"rs_raw":round(rs_raw,1),"risk_adj":risk_adj})
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
        raw=yf.download(tks+["^KS11"],period="1y",auto_adjust=True,progress=False)
        if raw.empty: return {"error":"데이터 없음"}
        data=raw["Close"] if isinstance(raw.columns,pd.MultiIndex) else raw
        if "^KS11" not in data.columns: return {"error":"KOSPI 없음"}
        kospi=data["^KS11"].dropna(); results=[]
        for t in tks:
            if t not in data.columns: continue
            etf=data[t].dropna(); common=etf.index.intersection(kospi.index)
            if len(common)<52: continue
            rel=etf.loc[common]/kospi.loc[common]; ma52=rel.rolling(52).mean()
            rs_raw=float(((rel/ma52)-1).dropna().iloc[-1]*100) if not ((rel/ma52)-1).dropna().empty else 0
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
# 탭 1: 미국 지표
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("🌎 미국 시장 종합 지표")
    st.caption("미국 장 마감 후 (한국 오전 6~7시) 실행 권장")

    if st.button("▶ 미국 지표 전체 실행", type="primary", use_container_width=True, key="us_run"):
        prog = st.progress(0, text="🐤 카나리아 분석 중...")
        canary=get_canary_signal(); prog.progress(10, text="🔥 BOFA Heat 분석 중...")
        bofa=get_bofa_heat(); prog.progress(20, text="🩸 블러드 인디케이터...")
        blood=get_blood_indicator(); prog.progress(30, text="😱 피어앤그리드 (일간)...")
        fg=get_us_fear_greed(); prog.progress(42, text="📅 피어앤그리드 (월간)...")
        fg_m=get_monthly_fear_greed(); prog.progress(52, text="📊 코포크 지표...")
        coppock=get_coppock(); coppock_fast=get_coppock_fast(); prog.progress(62, text="📡 ZBT 시장 폭...")
        zbt=get_zbt(); prog.progress(72, text="🏆 S&P500 RS 상위...")
        sp500_rs=get_sp500_rs(SP500_TOP100); prog.progress(84, text="🏆 나스닥100 RS...")
        ndx_rs=get_nasdaq100_rs(NASDAQ100); prog.progress(93, text="🏭 미국 섹터 ETF RS...")
        us_sector=get_us_sector_rs(); prog.progress(100, text="✅ 완료!")
        prog.empty(); st.success("✅ 분석 완료!")

        # ── 종합 신호 카드 ──
        st.markdown("### 🚦 종합 신호")
        c1,c2,c3,c4 = st.columns(4)
        if canary and "error" not in canary:
            e="🟢" if canary["mode"]=="공격" else "🔴"
            c1.metric("🐤 카나리아 모드",f"{e} {canary['mode']}",f"QQQ {canary['qqq_mom']*100:+.1f}%")
        if bofa and "error" not in bofa:
            c2.metric("🔥 BOFA Heat",f"{bofa['heat']}/10",_heat_label(bofa['heat']))
        if blood and "error" not in blood:
            c3.metric("🩸 블러드",blood["value"],f"60MA {blood['vs_ma60']}")
        if zbt and "error" not in zbt:
            c4.metric("📡 ZBT",f"{zbt['zbt']:.3f}","🟢 반등신호" if zbt.get("signal") else "⏳ 대기중")

        st.divider()

        # ── BOFA 세부 ──
        if bofa and "error" not in bofa:
            with st.expander("🔥 BOFA Heat 세부 보기", expanded=False):
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Heat Score",f"{bofa['heat']}/10",_heat_label(bofa['heat']))
                c2.metric("상승추세","✅ Yes" if bofa['trend_on'] else "❌ No")
                c3.metric("VIX 쇼크","⚠️ ON" if bofa['shock_vix'] else "✅ 정상")
                c4.metric("크레딧 쇼크","⚠️ ON" if bofa['shock_credit'] else "✅ 정상")
                st.caption("Heat ≥7.5: 과열 🔴 / ≥5: 주의 🟠 / ≥2.5: 보통 🟡 / <2.5: 안전 🟢")

        # ── ZBT 세부 ──
        if zbt and "error" not in zbt:
            with st.expander("📡 ZBT (Zweig Breadth Thrust) 세부 보기", expanded=False):
                c1,c2,c3 = st.columns(3)
                c1.metric("ZBT 현재",f"{zbt['zbt']:.3f}","🟢 신호발생!" if zbt["signal"] else "❌ 미발생")
                c2.metric("최근 최저",f"{zbt['prev_min']:.3f}","<40% 필요")
                c3.metric("VIX",str(zbt.get('vix','N/A')),"✅ 안정" if zbt.get('vix_ok') else "⚠️ 높음")
                st.caption("ZBT: 최근10일최저<40% → 현재>61.5% 돌파 시 저점 반등 신호 (나스닥100+S&P500 종목 기준)")

        st.divider()

        # ── 피어앤그리드 ──
        st.markdown("### 😱 피어앤그리드 오실레이터")
        c_d, c_m = st.columns(2)
        with c_d:
            st.markdown("**📅 일간 (단기 추세)**")
            if fg and "error" not in fg:
                c1,c2=st.columns(2)
                c1.metric("S&P500 Osc",fg["spx_osc"],f"{'🟢' if fg['spx_osc']>0 else '🔴'} {fg['spx_sentiment']}")
                c2.metric("NASDAQ Osc",fg["ndx_osc"],f"{'🟢' if fg['ndx_osc']>0 else '🔴'} {fg['ndx_sentiment']}")
                c1.metric("SPY 임펄스",fg["spy_impulse"]); c2.metric("QQQ 임펄스",fg["qqq_impulse"])
                c1.metric("SPY SuperMA 이격",f"{fg['spy_gap']:+.2f}%"); c2.metric("QQQ SuperMA 이격",f"{fg['qqq_gap']:+.2f}%")
                c1.metric("SPY TD 매도/매수",f"{fg['spy_td_sell']} / {fg['spy_td_buy']}")
                c2.metric("QQQ TD 매도/매수",f"{fg['qqq_td_sell']} / {fg['qqq_td_buy']}")
            else: st.error("일간 F&G 오류")
        with c_m:
            st.markdown("**📆 월간 (중장기 추세)**")
            if fg_m and "error" not in fg_m:
                c1,c2=st.columns(2)
                c1.metric("S&P500 월간",fg_m["spx_osc"],f"{'🟢' if fg_m['spx_osc']>0 else '🔴'} {fg_m['spx_sentiment']}")
                c2.metric("NASDAQ 월간",fg_m["ndx_osc"],f"{'🟢' if fg_m['ndx_osc']>0 else '🔴'} {fg_m['ndx_sentiment']}")
                c1.metric("기준월",fg_m["date"]); c2.metric("S&P500 FGI",fg_m["spx_fgi"])
            else: st.error("월간 F&G 오류")

        st.divider()

        # ── 코포크 ──
        st.markdown("### 📊 코포크 지표")
        c_std, c_fast = st.columns(2)
        with c_std:
            st.markdown("**표준 (11/14개월 ROC, 10개월 EMA)**")
            if coppock and "error" not in coppock:
                cols=st.columns(len(coppock))
                for i,(lbl,v) in enumerate(coppock.items()):
                    e="🟢" if v["pos"] else "🔴"; arr="▲" if v["trend"]=="상승" else "▼"
                    cols[i].metric(lbl,f"{e} {v['value']}",f"{arr} {v['trend']}")
        with c_fast:
            st.markdown("**빠른 버전 (4/6개월 ROC, 3개월 MA)**")
            if coppock_fast and "error" not in coppock_fast:
                cols=st.columns(len(coppock_fast))
                for i,(lbl,v) in enumerate(coppock_fast.items()):
                    e="🟢" if v["pos"] else "🔴"; arr="▲" if v["trend"]=="상승" else "▼"
                    cols[i].metric(lbl,f"{e} {v['value']}",f"{arr} {v['trend']}")

        st.divider()

        # ── RS 상위 종목 ──
        st.markdown("### 🏆 상대강도(RS) 상위 종목")
        c1,c2=st.columns(2)
        with c1:
            st.markdown("**📈 S&P500 Top 10**")
            if sp500_rs and "error" not in sp500_rs:
                for i,item in enumerate(sp500_rs["top"],1):
                    t=item["ticker"]; kr=TICKER_NAMES.get(t,""); rs=item.get("rs",0)
                    st.markdown(f"`{i:2d}` **{t}** {kr} &nbsp; {_cv(rs,'1f')}", unsafe_allow_html=True)
            elif sp500_rs: st.error(sp500_rs.get("error"))
        with c2:
            st.markdown("**📈 나스닥100 Top 10**")
            if ndx_rs and "error" not in ndx_rs:
                for i,item in enumerate(ndx_rs["top"],1):
                    t=item["ticker"]; kr=TICKER_NAMES.get(t,""); rs=item.get("rs",0)
                    st.markdown(f"`{i:2d}` **{t}** {kr} &nbsp; {_cv(rs,'2f')}", unsafe_allow_html=True)
            elif ndx_rs: st.error(ndx_rs.get("error"))

        st.divider()

        # ── 섹터 ETF RS ──
        st.markdown("### 🏭 미국 섹터 ETF 상대강도")
        if us_sector and "error" not in us_sector:
            df_s=pd.DataFrame(us_sector["sectors"])
            df_s["강도"]=df_s["norm_rs"].apply(lambda x:"🟢 강세" if x>=70 else ("🟡 중립" if x>=50 else "🔴 약세"))
            df_s.columns=[c if c!="ticker" else "티커" for c in df_s.columns]
            st.dataframe(df_s.rename(columns={"ticker":"티커","name":"섹터","norm_rs":"RS(0~100)","rs_raw":"RS원시","risk_adj":"변동성조정모멘텀"}),
                use_container_width=True, hide_index=True,
                column_config={"RS(0~100)":st.column_config.ProgressColumn("RS(0~100)",min_value=0,max_value=100,format="%.1f")})
        elif us_sector: st.error(us_sector.get("error"))

# ─────────────────────────────────────────────────────────────────────────────
# 탭 2: 국내 지표
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("🇰🇷 국내 시장 지표")
    st.caption("장 마감 후 (오후 4시 이후) 실행 권장")

    if st.button("▶ 국내 지표 전체 실행", type="primary", use_container_width=True, key="kr_run"):
        prog=st.progress(0, text="📊 코스피/코스닥 수집 중...")
        market=get_market_summary(); prog.progress(20, text="🏭 업종 ETF 수집 중...")
        sector=get_sector_performance(); prog.progress(40, text="💹 수급 오실레이터 계산 중...")
        supply=get_supply_oscillator(); prog.progress(60, text="🇰🇷 한국 ETF RS 계산 중...")
        kr_etf=get_kr_etf_rs(); prog.progress(80, text="🏠 빈집 스크리닝 중...")
        binzip=get_binzip_stocks(supply_data=supply); prog.progress(100, text="✅ 완료!")
        prog.empty(); st.success("✅ 분석 완료!")

        # ── 지수 현황 ──
        st.markdown("### 📊 지수 현황")
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
        st.markdown("### 🏭 업종 강세 / 약세")
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
        st.markdown("### 💹 수급 오실레이터")
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
        st.markdown("### 🇰🇷 한국 산업/테마 ETF 상대강도 (vs KOSPI)")
        if kr_etf and "error" not in kr_etf:
            show=kr_etf.get("strong") or kr_etf.get("all",[])[:10]
            if show:
                df_kr=pd.DataFrame(show)
                df_kr["강도"]=df_kr["norm_rs"].apply(lambda x:"🟢 강세" if x>=70 else "🟡 보통")
                st.dataframe(df_kr.rename(columns={"name":"ETF명","norm_rs":"RS(0~100)","risk_adj":"변동성조정모멘텀","강도":"강도"})[["ETF명","RS(0~100)","변동성조정모멘텀","강도"]],
                    use_container_width=True, hide_index=True,
                    column_config={"RS(0~100)":st.column_config.ProgressColumn("RS(0~100)",min_value=0,max_value=100,format="%.1f")})
        elif kr_etf: st.error(kr_etf.get("error"))

        st.divider()

        # ── 빈집 주도주 ──
        bz=binzip or {}; bl=bz.get("binzip",[]); ss=" + ".join(bz.get("sectors",[])) or "주도업종"
        st.markdown(f"### 🏠 수급 빈집 주도주 [{ss}]")
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

# ─────────────────────────────────────────────────────────────────────────────
# 탭 3: 종목 분석
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("🔍 개별 종목 분석")

    st.markdown("### 🎯 매수 타점 통계 분석")
    st.caption("1년 데이터 기반 — 고가/저가/시가 평균 괴리율 (지정가 매수 참고용)")
    ticker_input = st.text_input("티커 입력 (쉼표로 여러 개 가능)", placeholder="NVDA, AAPL, 005930.KS", key="bt_ticker")
    if st.button("🔍 분석 시작", key="bt_run", type="primary", use_container_width=True):
        if ticker_input:
            for t in [x.strip() for x in ticker_input.split(",") if x.strip()]:
                with st.spinner(f"{t} 분석 중..."):
                    res=get_buy_timing(t)
                if "error" in res:
                    st.error(f"❌ {t}: {res['error']}")
                else:
                    kr=TICKER_NAMES.get(t, res.get("name",""))
                    st.markdown(f"#### 📌 **{t}** {kr} — 현재가: `{res['price']:,}`")
                    c1,c2=st.columns(2)
                    c1.metric("고가→종가 평균 하락",f"{res['고가종가하락']:+.2f}%",help="장중 고점 대비 종가 평균 낙폭. 지정가보다 높게 올라갔다가 내려오는 정도")
                    c2.metric("전일종가→당일저가 괴리",f"{res['저가종가괴리']:+.2f}%",help="전일 종가 대비 당일 저가 평균 괴리. 갭하락 포함")
                    c1.metric("시가→저가 평균 낙폭",f"{res['시가저가괴리']:+.2f}%",help="시가 기준 장중 최대 낙폭 평균")
                    c2.metric("전일종가→당일고가",f"{res['전일종가고가']:+.2f}%",help="전일 종가 대비 당일 고가 평균 괴리율")
                    c1.metric("시가→당일고가 평균",f"{res['시가고가괴리']:+.2f}%",help="시가 대비 장중 고점까지 평균 상승폭")
                    st.caption(f"💡 매수 타이밍 힌트: 시가 대비 저가 괴리({res['시가저가괴리']:+.2f}%) 활용 → 시가보다 그 정도 낮게 지정가 설정")
                    st.divider()
        else:
            st.warning("티커를 입력해주세요 (예: NVDA)")
