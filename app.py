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

st.set_page_config(page_title="태린이 주식봇", page_icon="📈", layout="centered")

st.title("📈 태린이 주식봇")
st.caption(f"오늘: {datetime.today().strftime('%Y-%m-%d (%A)').replace('Monday','월요일').replace('Tuesday','화요일').replace('Wednesday','수요일').replace('Thursday','목요일').replace('Friday','금요일').replace('Saturday','토요일').replace('Sunday','일요일')}")

tab1, tab2 = st.tabs(["🌎 미국 지표", "🇰🇷 국내 지표"])

# ─────────────────────────────────────────────────────────────────────────────
# 공통 유틸
# ─────────────────────────────────────────────────────────────────────────────
def _dl(ticker, period="3y", start=None, end=None, retries=3):
    for i in range(retries):
        try:
            if start:
                df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False, threads=False)
            else:
                df = yf.download(ticker, period=period, auto_adjust=True, progress=False, threads=False)
            if not df.empty:
                return df
        except Exception:
            pass
        time.sleep(2 * (i + 1))
    return pd.DataFrame()

def _close(ticker, period="3y", start=None, end=None):
    df = _dl(ticker, period=period, start=start, end=end)
    if df.empty:
        return pd.Series(dtype=float)
    if isinstance(df.columns, pd.MultiIndex):
        return df["Close"][ticker].dropna()
    return df["Close"].dropna()

def _rsi(series, window=10):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _macd_hist(series, fast=12, slow=26, signal=9):
    macd = series.ewm(span=fast, adjust=False).mean() - series.ewm(span=slow, adjust=False).mean()
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd - sig

def _td_back(n):
    d = datetime.today()
    cnt = 0
    while cnt < n:
        d -= timedelta(days=1)
        if d.weekday() < 5:
            cnt += 1
    return d.strftime("%Y-%m-%d")

# ─────────────────────────────────────────────────────────────────────────────
# 미국 지표 함수들
# ─────────────────────────────────────────────────────────────────────────────
def get_canary_signal():
    try:
        end = datetime.today().strftime("%Y-%m-%d")
        start = (datetime.today() - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
        results = {}
        for t in ["QQQ", "TIP"]:
            px = _close(t, start=start, end=end)
            if len(px) < 260:
                return None
            results[t] = float(((px.iloc[-1]/px.iloc[-22]-1) + (px.iloc[-1]/px.iloc[-63]-1) + (px.iloc[-1]/px.iloc[-126]-1) + (px.iloc[-1]/px.iloc[-252]-1)) / 4)
        both_positive = all(v > 0 for v in results.values())
        return {"qqq_mom": results["QQQ"], "tip_mom": results["TIP"], "mode": "공격" if both_positive else "방어"}
    except Exception as e:
        return {"error": str(e)}

def get_bofa_heat():
    try:
        tickers_map = {"SPY":"SPY","QQQ":"QQQ","RSP":"RSP","VIX":"^VIX","HYG":"HYG","IEF":"IEF","LQD":"LQD"}
        raw = yf.download(list(tickers_map.values()), start="2015-01-01", auto_adjust=True, progress=False)
        if raw.empty: return None
        px = raw["Close"].copy().rename(columns={v:k for k,v in tickers_map.items()})
        ZS_WIN = 252
        def zscore(s):
            mu = s.rolling(ZS_WIN).mean(); sd = s.rolling(ZS_WIN).std(ddof=0)
            return (s - mu) / sd
        def norm01(z, lo=0.0, hi=2.0):
            return ((z - lo) / (hi - lo)).clip(0, 1)
        df = pd.DataFrame(index=px.index)
        if {"HYG","IEF"}.issubset(px.columns): df["h_risk"] = norm01(zscore(px["HYG"]/px["IEF"]))
        if {"HYG","LQD"}.issubset(px.columns): df["h_credit"] = norm01(zscore(px["HYG"]/px["LQD"]))
        if {"RSP","SPY"}.issubset(px.columns): df["h_style"] = ((0-zscore(px["RSP"]/px["SPY"]))/2).clip(0,1)
        if "SPY" in px.columns:
            spy_ma200 = px["SPY"].rolling(200).mean()
            df["h_spy_ext"] = norm01(zscore(px["SPY"]/spy_ma200-1), 0.5, 2.0)
        if "QQQ" in px.columns:
            qqq_ma200 = px["QQQ"].rolling(200).mean()
            df["h_qqq_ext"] = norm01(zscore(px["QQQ"]/qqq_ma200-1), 0.5, 2.0)
        heat_cols = [c for c in df.columns if c.startswith("h_")]
        weights = {"h_risk":1.2,"h_credit":1.0,"h_style":0.8,"h_spy_ext":0.8,"h_qqq_ext":0.8}
        W = pd.Series({c: weights.get(c,1.0) for c in heat_cols}); W = W/W.sum()
        heat = ((df[heat_cols]*W).sum(axis=1)*10).rolling(10).mean()
        if "SPY" in px.columns:
            spy_ma200 = px["SPY"].rolling(200).mean()
            trend_on = (px["SPY"]>spy_ma200)&(spy_ma200.diff(20)>0)
            heat = pd.Series(np.maximum(heat.values, np.where(trend_on,2.5,0.0)), index=heat.index)
        shock_vix = bool(px["VIX"].pct_change(3).iloc[-1]>=0.30) if "VIX" in px.columns else False
        shock_credit = bool((px["HYG"]/px["LQD"]).pct_change(5).iloc[-1]<=-0.02) if {"HYG","LQD"}.issubset(px.columns) else False
        trend_val = bool(trend_on.dropna().iloc[-1]) if "SPY" in px.columns else False
        return {"heat":round(float(heat.dropna().iloc[-1]),2),"shock":shock_vix or shock_credit,"trend_on":trend_val}
    except Exception as e:
        return {"error": str(e)}

def get_blood_indicator():
    try:
        raw = yf.download(["^IRX","^TNX","HYG","IEF"], start="2015-01-01", auto_adjust=True, progress=False)
        px = raw["Close"].copy()
        irx = px["^IRX"]/100; t10y = px["^TNX"]/10
        hyg_yield = getattr(yf.Ticker("HYG").fast_info, "dividend_yield", None) or yf.Ticker("HYG").info.get("dividendYield", 0.06)
        blood = (irx/(hyg_yield-t10y)).dropna()
        cur = float(blood.iloc[-1])
        ma20 = float(blood.rolling(20).mean().iloc[-1])
        ma60 = float(blood.rolling(60).mean().iloc[-1])
        return {"value":round(cur,4),"ma20":round(ma20,4),"ma60":round(ma60,4),"vs_ma20":"위" if cur>ma20 else "아래","vs_ma60":"위" if cur>ma60 else "아래"}
    except Exception as e:
        return {"error": str(e)}

def get_us_fear_greed():
    try:
        tickers = ["^GSPC","^IXIC","^VIX","^TNX","^FVX","HYG","IEF","QQQ","SPY"]
        raw = yf.download(tickers, start="2024-01-01", auto_adjust=True, progress=False)
        data = raw["Close"].rename(columns={"^GSPC":"SP500","^IXIC":"NASDAQ","^VIX":"VIX","^TNX":"T10Y","^FVX":"T5Y"}).dropna(subset=["SP500","NASDAQ","VIX","T10Y","T5Y","HYG","IEF","QQQ","SPY"])
        data["RiskApp"] = data["HYG"]/data["IEF"]
        def calc_fg(df, col, label):
            df[f"{label}_MA125"] = df[col].rolling(125).mean()
            df[f"{label}_Mom"] = (df[col]-df[f"{label}_MA125"])/df[f"{label}_MA125"]*100
            df[f"{label}_RSI"] = _rsi(df[col])
            df[f"{label}_BondSpread"] = df["T10Y"]-df["T5Y"]
            df[f"{label}_VIX"] = df["VIX"]; df[f"{label}_RA"] = df["RiskApp"]
            cols = [f"{label}_Mom",f"{label}_RSI",f"{label}_BondSpread",f"{label}_VIX",f"{label}_RA"]
            valid = df[cols].dropna()
            if valid.empty: return df
            df.loc[valid.index, cols] = MinMaxScaler().fit_transform(valid)
            df[f"{label}_FGI"] = (df[f"{label}_Mom"]+df[f"{label}_RA"]+(1-df[f"{label}_VIX"])+df[f"{label}_BondSpread"]+df[f"{label}_RSI"])*0.2
            df[f"{label}_Osc"] = _macd_hist(df[f"{label}_FGI"])
            return df
        data = calc_fg(data,"SP500","SPX"); data = calc_fg(data,"NASDAQ","NDX")
        for col, label in [("SPY","SPY"),("QQQ","QQQ")]:
            ma_cols = [f"{label}_MA{w}" for w in [20,60,120,200]]
            for w in [20,60,120,200]: data[f"{label}_MA{w}"] = data[col].rolling(w).mean()
            data[f"{label}_SuperMA"] = data[ma_cols].mean(axis=1)
            data[f"{label}_GapPct"] = (data[col]-data[f"{label}_SuperMA"])/data[f"{label}_SuperMA"]*100
        def impulse(df, col, label):
            df[f"{label}_EMA13"] = df[col].ewm(span=13,adjust=False).mean()
            df[f"{label}_MACDh"] = _macd_hist(df[col])
            last = df[[f"{label}_EMA13",f"{label}_MACDh"]].dropna()
            if len(last)<2: return "알수없음"
            eu = last[f"{label}_EMA13"].iloc[-1]>last[f"{label}_EMA13"].iloc[-2]
            mu = last[f"{label}_MACDh"].iloc[-1]>last[f"{label}_MACDh"].iloc[-2]
            return "초록(강세)" if eu and mu else ("빨강(약세)" if not eu and not mu else "파랑(중립)")
        def td_setup(s):
            p=s.values; sell=np.zeros(len(p)); buy=np.zeros(len(p))
            for i in range(len(p)):
                sell[i]=sell[i-1]+1 if i>=4 and p[i]>p[i-4] else 0
                buy[i]=buy[i-1]+1 if i>=2 and p[i]<p[i-2] else 0
            return int(sell[-1]),int(buy[-1])
        last = data.dropna(subset=["SPX_Osc","NDX_Osc"])
        spy_ts, spy_tb = td_setup(data["SPY"].dropna())
        qqq_ts, qqq_tb = td_setup(data["QQQ"].dropna())
        return {"spx_osc":round(float(last["SPX_Osc"].iloc[-1]),4),"ndx_osc":round(float(last["NDX_Osc"].iloc[-1]),4),
                "spx_sentiment":"탐욕" if last["SPX_Osc"].iloc[-1]>0 else "공포",
                "ndx_sentiment":"탐욕" if last["NDX_Osc"].iloc[-1]>0 else "공포",
                "spy_gap":round(float(data["SPY_GapPct"].dropna().iloc[-1]),2),
                "qqq_gap":round(float(data["QQQ_GapPct"].dropna().iloc[-1]),2),
                "spy_impulse":impulse(data,"SPY","SPY"),"qqq_impulse":impulse(data,"QQQ","QQQ"),
                "spy_td_sell":spy_ts,"spy_td_buy":spy_tb,"qqq_td_sell":qqq_ts,"qqq_td_buy":qqq_tb}
    except Exception as e:
        return {"error": str(e)}

def get_coppock():
    try:
        results = {}
        for t in ["SPY","QQQ","^GSPC"]:
            px = _close(t, period="5y")
            if px.empty: continue
            monthly = px.resample("ME").last()
            monthly = pd.concat([monthly,px.iloc[[-1]]])
            monthly = monthly[~monthly.index.duplicated(keep="last")].sort_index()
            coppock = (monthly.pct_change(14)+monthly.pct_change(11)).ewm(span=10,adjust=False).mean()*100
            last = coppock.dropna()
            if last.empty: continue
            val=float(last.iloc[-1]); prev=float(last.iloc[-2]) if len(last)>=2 else val
            results[{"SPY":"SPY","QQQ":"QQQ","^GSPC":"S&P500"}.get(t,t)] = {"value":round(val,2),"trend":"상승" if val>prev else "하락"}
        return results
    except Exception as e:
        return {"error": str(e)}

def get_sp500_rs(tickers, top_n=10):
    try:
        spy_px = _close("SPY", period="3y")
        if spy_px.empty: return {"error":"SPY 없음"}
        rows = []
        for i in range(0, len(tickers), 50):
            raw = yf.download(tickers[i:i+50], period="3y", auto_adjust=True, progress=False)
            if raw.empty: continue
            px_batch = raw["Close"] if not isinstance(raw.columns, pd.MultiIndex) else raw["Close"]
            if isinstance(px_batch, pd.Series): px_batch = px_batch.to_frame()
            for t in px_batch.columns:
                s = px_batch[t].dropna()
                aligned = pd.concat([s,spy_px],axis=1,join="inner"); aligned.columns=["stock","spy"]
                if len(aligned)<60: continue
                rs_vals = []
                for win in [60,120,250]:
                    if len(aligned)<win: continue
                    rel=aligned["stock"]/aligned["spy"]; ma=rel.rolling(win).mean(); rs=((rel/ma)-1)*100
                    if not rs.dropna().empty: rs_vals.append(float(rs.dropna().iloc[-1]))
                if rs_vals: rows.append({"ticker":t,"rs_avg":np.mean(rs_vals)})
            time.sleep(0.1)
        if not rows: return {"error":"RS 계산 실패"}
        return {"top": pd.DataFrame(rows).sort_values("rs_avg",ascending=False).head(top_n)["ticker"].tolist()}
    except Exception as e:
        return {"error": str(e)}

def get_nasdaq100_rs(tickers, top_n=10):
    try:
        raw = yf.download(tickers+["SPY"], period="1y", auto_adjust=True, progress=False)
        if raw.empty: return {"error":"데이터 없음"}
        data = raw["Close"].ffill().dropna(axis=1, how="any")
        if "SPY" not in data.columns: return {"error":"SPY 없음"}
        spy_ret = (data["SPY"].pct_change().rolling(63).mean()*0.5+data["SPY"].pct_change().rolling(126).mean()*0.3+data["SPY"].pct_change().rolling(252).mean()*0.2).iloc[-1]
        rs_dict = {}
        for t in data.columns:
            if t=="SPY": continue
            mom=(data[t].pct_change().rolling(63).mean()*0.5+data[t].pct_change().rolling(126).mean()*0.3+data[t].pct_change().rolling(252).mean()*0.2).iloc[-1]
            if spy_ret!=0: rs_dict[t]=float(mom/spy_ret)
        return {"top": sorted(rs_dict,key=rs_dict.get,reverse=True)[:top_n]}
    except Exception as e:
        return {"error": str(e)}

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
}

SP500_TOP100 = ["AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","BRK-B","TSLA","LLY","AVGO","JPM","UNH","XOM","V","MA","PG","JNJ","HD","COST","ABBV","MRK","NFLX","CVX","BAC","CRM","ORCL","AMD","PEP","ACN","WMT","LIN","MCD","CSCO","TMO","ADBE","PLTR","TMUS","INTU","GE","IBM","CAT","PM","AMGN","TXN","NOW","ISRG","QCOM","UBER","GS","VZ","HON","RTX","SPGI","DHR","NEE","MS","LOW","T","UNP","BKNG","AXP","SCHW","C","BLK","SYK","GILD","PFE","DE","MDT","BA","AMAT","ADI","LRCX","PANW","MU","TJX","ETN","VRTX","KLAC","SBUX","CB","MMC","SO","DUK","BSX","REGN","PLD","CI","ZTS","ICE","CME","WM","APH","MCO","SNPS","CDNS","ITW","NOC","EMR"]
NASDAQ100 = ["AAPL","ABNB","ADBE","ADI","ADP","ADSK","AEP","AMAT","AMD","AMGN","AMZN","ANSS","ASML","AVGO","AZN","BIIB","BKNG","BKR","CCEP","CDNS","CDW","CEG","CHTR","CMCSA","COST","CPRT","CRWD","CSCO","CSX","CTAS","CTSH","DASH","DDOG","DLTR","DXCM","EA","EXC","FANG","FAST","FTNT","GEHC","GILD","GOOG","GOOGL","HON","IDXX","INTC","INTU","ISRG","KDP","KHC","KLAC","LIN","LRCX","LULU","MAR","MCHP","MDLZ","MELI","META","MNST","MRNA","MRVL","MSFT","MU","NFLX","NVDA","NXPI","ODFL","ON","ORLY","PANW","PAYX","PCAR","PDD","PEP","PYPL","QCOM","REGN","ROP","ROST","SBUX","SNPS","TEAM","TMUS","TSLA","TTD","TXN","VRSK","VRTX","WDAY","XEL","ZS"]

# ─────────────────────────────────────────────────────────────────────────────
# 국내 지표 함수들
# ─────────────────────────────────────────────────────────────────────────────
NAVER_ETF_URL = "https://finance.naver.com/api/sise/etfItemList.nhn?etfType=0"
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.naver.com"}
SECTOR_ETFS = {"091160":"반도체","305720":"2차전지","244580":"바이오","091180":"자동차","139270":"금융","266370":"IT","445290":"로봇","139250":"건설기계","139220":"소재/화학","117460":"에너지화학","143860":"헬스케어","139260":"정보기술","466920":"전력기기","449450":"방산","494670":"전력TOP10"}
_FALLBACK = {
    "반도체":[("005930","삼성전자"),("000660","SK하이닉스"),("042700","한미반도체"),("240810","원익IPS"),("058470","리노공업")],
    "전력기기":[("267260","HD현대일렉트릭"),("298040","효성중공업"),("010120","LS일렉트릭"),("103590","일진전기"),("033100","제룡전기")],
    "자동차":[("005380","현대차"),("000270","기아"),("012330","현대모비스")],
    "방산":[("012450","한화에어로스페이스"),("047810","한국항공우주"),("079550","LIG넥스원")],
    "바이오":[("207940","삼성바이오로직스"),("068270","셀트리온"),("128940","한미약품")],
    "2차전지":[("373220","LG에너지솔루션"),("006400","삼성SDI"),("051910","LG화학")],
    "로봇":[("277810","레인보우로보틱스"),("454910","두산로보틱스"),("090360","로보스타")],
    "IT":[("005930","삼성전자"),("000660","SK하이닉스"),("035420","NAVER"),("035720","카카오"),("251270","넷마블")],
    "정보기술":[("005930","삼성전자"),("000660","SK하이닉스"),("035420","NAVER"),("035720","카카오")],
}

def get_market_summary():
    try:
        start=_td_back(5); end=datetime.today().strftime("%Y-%m-%d")
        kp_close=_close("^KS11",start=start,end=end); kq_close=_close("^KQ11",start=start,end=end)
        if kp_close.empty or kq_close.empty: return {"error":"데이터 없음"}
        kp_last=float(kp_close.iloc[-1]); kp_prev=float(kp_close.iloc[-2]) if len(kp_close)>=2 else kp_last
        kq_last=float(kq_close.iloc[-1]); kq_prev=float(kq_close.iloc[-2]) if len(kq_close)>=2 else kq_last
        return {"date":kp_close.index[-1].strftime("%Y-%m-%d"),
                "kospi":{"close":round(kp_last,2),"chg_pct":round((kp_last/kp_prev-1)*100,2),"week_pct":round((kp_last/float(kp_close.iloc[0])-1)*100,2)},
                "kosdaq":{"close":round(kq_last,2),"chg_pct":round((kq_last/kq_prev-1)*100,2),"week_pct":round((kq_last/float(kq_close.iloc[0])-1)*100,2)}}
    except Exception as e: return {"error":str(e)}

def get_sector_performance():
    try:
        r=requests.get(NAVER_ETF_URL,headers=HEADERS,timeout=15); etfs=r.json()["result"]["etfItemList"]
        etf_map={e["itemcode"]:e for e in etfs}; sector_data={}
        for code,name in SECTOR_ETFS.items():
            if code in etf_map: sector_data[name]={"chg_pct":round(etf_map[code]["changeRate"],2)}
        if not sector_data: return {"error":"섹터 데이터 없음"}
        sorted_s=sorted(sector_data.items(),key=lambda x:x[1]["chg_pct"],reverse=True)
        return {"sectors":sector_data,"top3":[(n,d["chg_pct"]) for n,d in sorted_s[:3]],"bot3":[(n,d["chg_pct"]) for n,d in sorted_s[-3:]]}
    except Exception as e: return {"error":str(e)}

def get_supply_oscillator():
    try:
        start=_td_back(25); end=datetime.today().strftime("%Y-%m-%d")
        kp_close=_close("^KS11",start=start,end=end)
        if kp_close.empty or len(kp_close)<5: return {"error":"데이터 부족"}
        kp_ma5=kp_close.rolling(5).mean().iloc[-1]
        kp_ma20=kp_close.rolling(20).mean().iloc[-1] if len(kp_close)>=20 else kp_close.mean()
        kp_osc=(kp_ma5/kp_ma20-1)*100
        results={"kospi_osc":round(kp_osc,2),"sectors":{}}
        for code,name in SECTOR_ETFS.items():
            try:
                close=_close(f"{code}.KS",start=start,end=end)
                if close.empty or len(close)<5: continue
                ma5=close.rolling(5).mean().iloc[-1]
                ma20=close.rolling(20).mean().iloc[-1] if len(close)>=20 else close.mean()
                osc=(ma5/ma20-1)*100; rel=osc-kp_osc
                results["sectors"][name]={"rel_osc":round(rel,2)}
            except: continue
        if results["sectors"]:
            sorted_s=sorted(results["sectors"].items(),key=lambda x:x[1]["rel_osc"],reverse=True)
            results["strong"]=[(n,d["rel_osc"]) for n,d in sorted_s[:3]]
            results["weak"]=[(n,d["rel_osc"]) for n,d in sorted_s[-3:]]
        return results
    except Exception as e: return {"error":str(e)}

def get_binzip_stocks(supply_data=None, top_n=5):
    try:
        lead_sectors=[n for n,_ in supply_data["strong"]][:2] if supply_data and "error" not in supply_data and supply_data.get("strong") else ["반도체","정보기술"]
        all_stocks=[]; seen=set()
        for sn in lead_sectors:
            for code,name in _FALLBACK.get(sn,[]):
                if code not in seen: seen.add(code); all_stocks.append({"code":code,"name":name})
        if not all_stocks: return {"error":"종목 없음","binzip":[],"sectors":lead_sectors}
        start=_td_back(65); end=datetime.today().strftime("%Y-%m-%d")
        kp_rs60=0.0; kp_rs20=0.0
        try:
            kp=_close("^KS11",start=start,end=end)
            if len(kp)>=21: kp_rs60=(kp.iloc[-1]/kp.iloc[0]-1)*100; kp_rs20=(kp.iloc[-1]/kp.iloc[-21]-1)*100
        except: pass
        candidates=[]
        for s in all_stocks:
            try:
                close=_close(f"{s['code']}.KS",start=start,end=end)
                if close.empty or len(close)<20: continue
                n=len(close); now=float(close.iloc[-1])
                ma60=float(close.rolling(min(60,n)).mean().iloc[-1])
                rs60=(now/float(close.iloc[0])-1)*100-kp_rs60
                rel20=((now/float(close.iloc[-21])-1)*100 if n>=21 else 0.0)-kp_rs20
                if -20.0<rel20<-2.0 and now>ma60*0.93 and rs60>5.0:
                    candidates.append({"name":s["name"],"code":s["code"],"rs60":round(rs60,1),"rel20":round(rel20,1),"price":int(now)})
            except: continue
        candidates.sort(key=lambda x:x["rs60"],reverse=True)
        return {"binzip":candidates[:top_n],"scanned":len(all_stocks),"sectors":lead_sectors}
    except Exception as e: return {"error":str(e),"binzip":[]}

# ─────────────────────────────────────────────────────────────────────────────
# 탭 1: 미국 지표
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("🌎 미국 시장 지표")
    st.caption("매일 아침 (미국 장 마감 후) 실행하세요")

    if st.button("▶ 미국 지표 실행", type="primary", use_container_width=True, key="us_run"):
        with st.spinner("데이터 수집 중... (2~5분 소요)"):

            with st.spinner("🐤 카나리아 자산 계산 중..."):
                canary = get_canary_signal()
            with st.spinner("🔥 BOFA Heat 계산 중..."):
                bofa = get_bofa_heat()
            with st.spinner("🩸 블러드 인디케이터 계산 중..."):
                blood = get_blood_indicator()
            with st.spinner("😱 피어앤그리드 계산 중..."):
                fg = get_us_fear_greed()
            with st.spinner("📊 코포크 지표 계산 중..."):
                coppock = get_coppock()
            with st.spinner("🏆 S&P500 RS 상위 계산 중 (시간 소요)..."):
                sp500_rs = get_sp500_rs(SP500_TOP100)
            with st.spinner("🏆 나스닥100 RS 계산 중..."):
                ndx_rs = get_nasdaq100_rs(NASDAQ100)

        st.success("✅ 완료!")

        # 카나리아
        if canary and "error" not in canary:
            mode_color = "🟢" if canary["mode"] == "공격" else "🔴"
            st.metric("🐤 카나리아 모드", f"{mode_color} {canary['mode']} 모드")
            col1, col2 = st.columns(2)
            col1.metric("QQQ 모멘텀", f"{canary['qqq_mom']*100:+.2f}%")
            col2.metric("TIP 모멘텀", f"{canary['tip_mom']*100:+.2f}%")

        st.divider()

        # BOFA Heat
        if bofa and "error" not in bofa:
            heat = bofa["heat"]
            heat_label = "🔴 과열" if heat >= 7.5 else ("🟠 주의" if heat >= 5 else "🟢 안전")
            col1, col2, col3 = st.columns(3)
            col1.metric("🔥 BOFA Heat", f"{heat}/10", heat_label)
            col2.metric("Shock", "⚠️ ON" if bofa["shock"] else "정상")
            col3.metric("상승추세", "예 ✅" if bofa["trend_on"] else "아니오")

        st.divider()

        # 블러드
        if blood and "error" not in blood:
            col1, col2, col3 = st.columns(3)
            col1.metric("🩸 블러드", blood["value"])
            col2.metric("20MA", blood["ma20"], blood["vs_ma20"])
            col3.metric("60MA", blood["ma60"], blood["vs_ma60"])

        st.divider()

        # 피어앤그리드
        if fg and "error" not in fg:
            st.subheader("😱 피어앤그리드 오실레이터")
            col1, col2 = st.columns(2)
            col1.metric("S&P500", fg["spx_osc"], fg["spx_sentiment"])
            col2.metric("NASDAQ", fg["ndx_osc"], fg["ndx_sentiment"])
            col1.metric("SPY 임펄스", fg["spy_impulse"])
            col2.metric("QQQ 임펄스", fg["qqq_impulse"])
            col1.metric("SPY SuperMA 이격", f"{fg['spy_gap']:+.2f}%")
            col2.metric("QQQ SuperMA 이격", f"{fg['qqq_gap']:+.2f}%")
            col1.metric("SPY TD 매도/매수", f"{fg['spy_td_sell']} / {fg['spy_td_buy']}")
            col2.metric("QQQ TD 매도/매수", f"{fg['qqq_td_sell']} / {fg['qqq_td_buy']}")

        st.divider()

        # 코포크
        if coppock and "error" not in coppock:
            st.subheader("📊 코포크 지표 (월간)")
            cols = st.columns(len(coppock))
            for i, (label, v) in enumerate(coppock.items()):
                arrow = "▲" if v["trend"] == "상승" else "▼"
                cols[i].metric(label, v["value"], f"{arrow} {v['trend']}")

        st.divider()

        # RS 상위
        st.subheader("🏆 RS 상위 종목")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**S&P500 상위 10**")
            if sp500_rs and "error" not in sp500_rs:
                for i, t in enumerate(sp500_rs["top"], 1):
                    st.write(f"{i}. **{t}** {TICKER_NAMES.get(t,'')}")
        with col2:
            st.write("**나스닥100 상위 10**")
            if ndx_rs and "error" not in ndx_rs:
                for i, t in enumerate(ndx_rs["top"], 1):
                    st.write(f"{i}. **{t}** {TICKER_NAMES.get(t,'')}")

# ─────────────────────────────────────────────────────────────────────────────
# 탭 2: 국내 지표
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("🇰🇷 국내 시장 지표")
    st.caption("장 마감 후 (16:10 이후) 실행하세요")

    if st.button("▶ 국내 지표 실행", type="primary", use_container_width=True, key="kr_run"):
        with st.spinner("데이터 수집 중..."):
            with st.spinner("📊 코스피/코스닥 수집 중..."): market = get_market_summary()
            with st.spinner("🏭 업종 ETF 수집 중..."): sector = get_sector_performance()
            with st.spinner("💹 수급 오실레이터 계산 중..."): supply = get_supply_oscillator()
            with st.spinner("🏠 빈집 스크리닝 중..."): binzip = get_binzip_stocks(supply_data=supply)

        st.success("✅ 완료!")

        # 지수 현황
        if market and "error" not in market:
            st.subheader(f"📊 지수 현황 ({market['date']})")
            kp = market["kospi"]; kq = market["kosdaq"]
            col1, col2 = st.columns(2)
            col1.metric("코스피", f"{kp['close']:,.2f}", f"{kp['chg_pct']:+.2f}% (주간 {kp['week_pct']:+.2f}%)")
            col2.metric("코스닥", f"{kq['close']:,.2f}", f"{kq['chg_pct']:+.2f}% (주간 {kq['week_pct']:+.2f}%)")

        st.divider()

        # 업종 강세/약세
        if sector and "error" not in sector:
            st.subheader("🏭 업종 강세/약세")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**강세 TOP3** 🟢")
                for name, chg in sector.get("top3", []):
                    st.write(f"▲ {name}: **{chg:+.2f}%**")
            with col2:
                st.write("**약세 BOT3** 🔴")
                for name, chg in sector.get("bot3", []):
                    st.write(f"▼ {name}: **{chg:+.2f}%**")

        st.divider()

        # 수급 오실레이터
        if supply and "error" not in supply:
            st.subheader("💹 수급 오실레이터")
            st.metric("코스피 기준 오실레이터", f"{supply['kospi_osc']:+.2f}")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**수급 강세** 🟢")
                for name, rel in supply.get("strong", []):
                    st.write(f"▲ {name}: +{rel:.2f}")
            with col2:
                st.write("**수급 약세** 🔴")
                for name, rel in supply.get("weak", []):
                    st.write(f"▼ {name}: {rel:.2f}")

        st.divider()

        # 빈집 주도주
        bz = binzip or {}
        binzip_list = bz.get("binzip", [])
        sector_str = " + ".join(bz.get("sectors", [])) or "주도업종"
        st.subheader(f"🏠 수급 빈집 주도주 [{sector_str}]")
        if binzip_list:
            df_bz = pd.DataFrame(binzip_list)[["name","code","price","rs60","rel20"]]
            df_bz.columns = ["종목명","코드","현재가","60일RS(%)","20일눌림(%)"]
            df_bz["현재가"] = df_bz["현재가"].apply(lambda x: f"{x:,}원")
            df_bz["60일RS(%)"] = df_bz["60일RS(%)"].apply(lambda x: f"+{x:.1f}%")
            df_bz["20일눌림(%)"] = df_bz["20일눌림(%)"].apply(lambda x: f"{x:.1f}%")
            st.dataframe(df_bz, use_container_width=True, hide_index=True)
            st.caption(f"{bz.get('scanned',0)}종목 스캔 / 빈집 {len(binzip_list)}개")
        elif "error" in bz:
            st.error(f"오류: {bz['error']}")
        else:
            st.info(f"빈집 조건 충족 종목 없음 ({bz.get('scanned',0)}종목 스캔)")
