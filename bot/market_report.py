# -*- coding: utf-8 -*-
import os, sys, time
import requests, pandas as pd, numpy as np, yfinance as yf
import holidays
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kr_screening import check_kr_screening

TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
    print("TELEGRAM_TOKEN / TELEGRAM_CHAT_ID 환경변수 없음")
    sys.exit(1)


def _retry(fn, attempts=3, delay=20):
    """함수를 최대 attempts번 재시도. 모두 실패하면 None 반환."""
    for i in range(attempts):
        try:
            result = fn()
            if result is not None:
                return result
        except Exception as e:
            print(f"  재시도 {i+1}/{attempts}: {e}")
            if i < attempts - 1:
                time.sleep(delay)
    return None


def _yf_direct(tickers, period_days=60):
    """yfinance 완전 실패 시 Yahoo Finance API 직접 호출 (종가만)."""
    import urllib.request, json
    from datetime import timezone as tz
    end_ts = int(datetime.now(tz.utc).timestamp())
    start_ts = end_ts - period_days * 86400
    single = isinstance(tickers, str)
    ticker_list = [tickers] if single else list(tickers)
    dfs = {}
    for t in ticker_list:
        for host in ("query1", "query2"):
            try:
                url = (f"https://{host}.finance.yahoo.com/v8/finance/chart/{t}"
                       f"?period1={start_ts}&period2={end_ts}&interval=1d")
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=12) as r:
                    data = json.loads(r.read())
                res = data["chart"]["result"][0]
                closes = res["indicators"]["quote"][0]["close"]
                idx = pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_localize(None)
                dfs[t] = pd.Series(closes, index=idx, name=t)
                break
            except Exception:
                continue
    if not dfs:
        return None
    df = pd.DataFrame(dfs).dropna(how="all")
    return df if not df.empty else None


def _yf_download(tickers, **kwargs):
    """yfinance download → 실패 시 직접 API 폴백."""
    def _dl():
        df = yf.download(tickers, progress=False, **kwargs)
        if df.empty:
            raise ValueError("빈 데이터")
        return df
    result = _retry(_dl, attempts=3, delay=15)
    if result is not None:
        return result
    # yfinance 완전 실패 → 직접 API로 폴백 (종가 데이터만)
    period = kwargs.get("period", "")
    days = {"5d":5,"1mo":30,"3mo":90,"6mo":180,"1y":365,"2y":730}.get(period, 60)
    start = kwargs.get("start", "")
    if start:
        try:
            days = (datetime.today() - datetime.strptime(start, "%Y-%m-%d")).days
        except Exception:
            days = 365
    print(f"  yfinance 실패 → 직접 API 폴백 시도: {tickers}")
    return _yf_direct(tickers, period_days=days)


def send_telegram(message: str, retries: int = 3) -> bool:
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    if len(message) > 4000:
        message = message[:3990] + "\n...(생략)"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, data=payload, timeout=15)
            if resp.ok:
                return True
            print(f"텔레그램 오류(시도{attempt}): {resp.text}")
        except Exception as e:
            print(f"텔레그램 전송 실패(시도{attempt}): {e}")
        if attempt < retries:
            time.sleep(5)
    return False


def check_macro() -> str:
    print("  [1/10] 글로벌 매크로 분석 중...")
    try:
        ticker_map = {
            "DX-Y.NYB": "달러(DXY)", "GC=F": "금", "CL=F": "WTI원유",
            "^TNX": "미국10Y", "^IRX": "미국3M", "^KS11": "KOSPI", "^N225": "니케이",
        }
        # 합리적 값 범위 (min, max) — 벗어나면 yfinance 반환 오류로 판단하고 건너뜀
        SANITY = {
            "DX-Y.NYB": (70, 130), "GC=F": (500, 8000), "CL=F": (5, 200),
            "^TNX": (0, 20), "^IRX": (0, 20),
            "^KS11": (500, 10000),    # KOSPI: 역대 최고 3,316 → 상한 10,000
            "^N225": (5000, 150000),  # 니케이: 역대 최고 ~42,000 → 상한 150,000
        }
        raw = _yf_download(list(ticker_map.keys()), period="5d", auto_adjust=False)
        if raw is None: return "🌍 글로벌 매크로: 데이터 일시 불가 (잠시 후 재시도)"
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        lines = []
        t10_val, t3m_val = None, None
        for ticker, name in ticker_map.items():
            if ticker not in close.columns:
                continue
            s = close[ticker].dropna()
            if len(s) < 2:
                continue
            val = float(s.iloc[-1])
            lo, hi = SANITY.get(ticker, (None, None))
            if lo is not None and not (lo <= val <= hi):
                lines.append(f"  {name}: ⚠️ 이상값({val:,.2f}) — yfinance 데이터 오류")
                continue
            chg = float(s.pct_change().iloc[-1]) * 100
            arrow = "▲" if chg >= 0 else "▼"
            if ticker in ("^TNX", "^IRX"):
                lines.append(f"  {name}: {val:.2f}%  {arrow}{abs(chg):.2f}%p")
                if ticker == "^TNX":
                    t10_val = val
                else:
                    t3m_val = val
            else:
                lines.append(f"  {name}: {val:,.2f}  {arrow}{abs(chg):.2f}%")
        yc_str = ""
        if t10_val and t3m_val:
            spread = t10_val - t3m_val
            if spread < 0:
                yc_str = f"\n  금리역전: {spread:+.2f}%p  경기침체 경보"
            elif spread < 0.5:
                yc_str = f"\n  장단기스프레드: {spread:+.2f}%p (주의)"
            else:
                yc_str = f"\n  장단기스프레드: {spread:+.2f}%p (정상)"
        return "🌍 <b>글로벌 매크로</b>\n" + "\n".join(lines) + yc_str
    except Exception as e:
        return f"🌍 매크로: 오류 ({e})"


def check_fear_greed() -> str:
    print("  [2/10] 공포탐욕 지수 분석 중...")

    def _cnn_fg():
        # Referer 헤더 필수 — 없으면 418 반환
        hdrs = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "*/*",
            "Referer": "https://www.cnn.com/markets/fear-and-greed",
        }
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        r = requests.get(url, headers=hdrs, timeout=12)
        if not r.ok:
            raise ValueError(f"CNN API {r.status_code}")
        data = r.json()
        fg = data.get("fear_and_greed", {})
        score = float(fg.get("score", 0))
        if score <= 0:
            raise ValueError("score=0")
        return score, float(fg.get("previous_close", score)), float(fg.get("previous_1_week", score))

    result = _retry(_cnn_fg, attempts=3, delay=15)

    if result:
        score, prev_close, prev_week = result
        if score <= 25:   emoji, label = "😱", "극도의 공포"
        elif score <= 45: emoji, label = "😨", "공포"
        elif score <= 55: emoji, label = "😐", "중립"
        elif score <= 75: emoji, label = "😊", "탐욕"
        else:             emoji, label = "🤑", "극도의 탐욕"
        bar = "█" * int(score / 10) + "░" * (10 - int(score / 10))
        return (f"😨 <b>CNN 공포탐욕 지수</b>\n{emoji} <b>{score:.0f}/100</b>  {label}\n"
                f"[{bar}]\n전일대비: {score-prev_close:+.1f}  1주대비: {score-prev_week:+.1f}")

    # CNN API 완전 실패 시 → VIX로 대체 산출
    try:
        vix = _yf_download("^VIX", period="10d", auto_adjust=False)
        if vix is not None:
            vix_c = vix["Close"] if isinstance(vix.columns, pd.MultiIndex) else vix
            if isinstance(vix_c, pd.DataFrame):
                vix_c = vix_c.iloc[:, 0]
            v = float(vix_c.dropna().iloc[-1])
            score = max(0, min(100, 100 - (v - 10) * 2.5))
            if score <= 25:   emoji, label = "😱", "극도의 공포"
            elif score <= 45: emoji, label = "😨", "공포"
            elif score <= 55: emoji, label = "😐", "중립"
            elif score <= 75: emoji, label = "😊", "탐욕"
            else:             emoji, label = "🤑", "극도의 탐욕"
            return (f"😨 <b>공포탐욕 (VIX 대체값)</b>\n{emoji} <b>{score:.0f}/100</b>  {label}\n"
                    f"  VIX={v:.1f} 기반 추정치")
    except Exception:
        pass
    return "😨 공포탐욕: 데이터 일시 불가"


def check_blood() -> str:
    print("  [3/10] BLOOD 인디케이터 분석 중...")
    try:
        raw = _yf_download(["^IRX", "^TNX", "HYG"], period="2y", auto_adjust=False)
        if raw is None: return "🩸 BLOOD: 데이터 일시 불가"
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        irx = close["^IRX"].dropna() / 100
        t10 = close["^TNX"].dropna()
        try:
            dy = yf.Ticker("HYG").info.get("dividendYield") or 0
            hyg_yield = float(dy) * 100 if dy and float(dy) > 0.01 else 6.5
        except Exception:
            hyg_yield = 6.5
        hy_spread = max(hyg_yield - float(t10.iloc[-1]), 0.01)
        blood_now = float(irx.iloc[-1]) / (hy_spread / 100)
        blood_series = irx.reindex(t10.index, method="ffill").dropna() / (hy_spread / 100)
        ma20 = float(blood_series.rolling(20).mean().dropna().iloc[-1])
        ma60 = float(blood_series.rolling(60).mean().dropna().iloc[-1])
        if blood_now > ma20 > ma60:   status = "🟢 상승추세"
        elif blood_now < ma20 < ma60: status = "🔴 하락추세"
        else:                         status = "🟡 혼조"
        return (f"🩸 <b>BLOOD 인디케이터</b>  단기금리÷하이일드스프레드\n"
                f"  높을수록 위험선호↑ | 낮을수록 안전자산 도피\n"
                f"현재: {blood_now:.3f}  20일선: {ma20:.3f}  60일선: {ma60:.3f}\n{status}")
    except Exception as e:
        return f"🩸 BLOOD: 오류 ({e})"


def check_canary() -> str:
    print("  [4/10] 카나리아 자산 분석 중...")
    try:
        start = (datetime.today() - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
        raw = _yf_download(["QQQ", "TIP", "AGG", "GLD", "BIL"], start=start, auto_adjust=False)
        if raw is None: return "📡 카나리아: 데이터 일시 불가"
        close = (raw["Adj Close"] if isinstance(raw.columns, pd.MultiIndex) and "Adj Close" in raw.columns.get_level_values(0)
                 else raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw).ffill().dropna()
        if len(close) < 252:
            return "📡 카나리아: 데이터 부족"
        def mom4(s): return (s.iloc[-1]/s.iloc[-22]-1 + s.iloc[-1]/s.iloc[-63]-1 + s.iloc[-1]/s.iloc[-126]-1 + s.iloc[-1]/s.iloc[-252]-1) / 4
        results = {col: mom4(close[col].dropna()) for col in close.columns if len(close[col].dropna()) >= 252}
        qqq_m, tip_m = results.get("QQQ", 0), results.get("TIP", 0)
        if qqq_m > 0 and tip_m > 0:   signal = "🚀 <b>공격 모드</b>"
        elif qqq_m <= 0 and tip_m <= 0: signal = "🛡️ <b>방어 모드</b>"
        else: signal = f"⚠️ <b>주의 모드</b>"
        name_map = {"QQQ":"나스닥100", "TIP":"물가채(TIPS)", "AGG":"채권", "GLD":"금", "BIL":"단기채"}
        # 신호 판단 기준인 QQQ·TIP를 먼저 표시
        priority = ["QQQ", "TIP"]
        ordered_keys = [k for k in priority if k in results] + [k for k in results if k not in priority]
        lines = [f"  {name_map.get(k,k)}: {results[k]:+.2%}" for k in ordered_keys]
        return ("📡 <b>카나리아 자산</b>  공격/방어 모드 판단\n"
                "  QQQ↑+TIP↑=공격 | 둘다↓=방어\n" + signal + "\n" + "\n".join(lines))
    except Exception as e:
        return f"📡 카나리아: 오류 ({e})"


def check_heat() -> str:
    print("  [5/10] Heat 인디케이터 분석 중...")
    try:
        raw = _yf_download(["SPY","QQQ","RSP","HYG","IEF","LQD","^VIX"], start="2015-01-01", auto_adjust=False)
        if raw is None: return "🌡️ Heat: 데이터 일시 불가"
        px = (raw["Adj Close"] if isinstance(raw.columns, pd.MultiIndex) else raw).rename(columns={"^VIX":"VIX"}).sort_index()
        def zs(s): return (s - s.rolling(252).mean()) / s.rolling(252).std(ddof=0)
        def n01(z, lo, hi): return z.clip(lo, hi).sub(lo).div(hi - lo)
        df = pd.DataFrame(index=px.index)
        if {"HYG","IEF"}.issubset(px.columns): df["h1"] = n01(zs(px["HYG"]/px["IEF"]), 0, 2)
        if {"HYG","LQD"}.issubset(px.columns): df["h2"] = n01(zs(px["HYG"]/px["LQD"]), 0, 2)
        if {"RSP","SPY"}.issubset(px.columns): df["h3"] = ((-zs(px["RSP"]/px["SPY"]))/2).clip(0, 1)
        if "SPY" in px.columns: df["h4"] = n01(zs(px["SPY"]/px["SPY"].rolling(200).mean()-1), 0.5, 2)
        if "VIX" in px.columns: df["h5"] = n01(-zs(px["VIX"]), 0, 2)  # VIX 낮을수록=과열, 높을수록=냉각
        df["heat"] = (df[[c for c in df.columns if c.startswith("h")]].mean(axis=1) * 10).rolling(10).mean()
        heat_val = float(df["heat"].dropna().iloc[-1])
        if heat_val >= 7.5:   status = "🔴 과열"
        elif heat_val <= 2.5: status = "🟢 냉각 — 매수 기회"
        else:                 status = "🟡 정상"
        return (f"🌡️ <b>시장과열 지수 (BofA Heat)</b>\n"
                f"  7.5↑=과열주의 | 2.5↓=냉각(매수기회)\n"
                f"점수: <b>{heat_val:.1f}</b>/10  {status}")
    except Exception as e:
        return f"🌡️ Heat: 오류 ({e})"


def check_sector_rotation() -> str:
    print("  [6/10] 섹터 로테이션 분석 중...")
    try:
        sectors = {"XLK":"기술","XLC":"통신","XLY":"경기소비재","XLP":"필수소비재","XLI":"산업재",
                   "XLB":"소재","XLE":"에너지","XLF":"금융","XLV":"헬스케어","XLU":"유틸리티","XLRE":"리츠","SPY":"S&P500"}
        raw = _yf_download(list(sectors.keys()), period="1y", auto_adjust=False)
        if raw is None: return "🔄 섹터: 데이터 일시 불가"
        cl = (raw["Adj Close"] if isinstance(raw.columns, pd.MultiIndex) and "Adj Close" in raw.columns.get_level_values(0)
              else raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw).ffill().dropna(axis=1, how="any")
        r1m = cl.pct_change(21).iloc[-1]
        adj_mom = (r1m*0.5 + cl.pct_change(63).iloc[-1]*0.3 + cl.pct_change(126).iloc[-1]*0.2) / cl.pct_change().rolling(63).std().iloc[-1].replace(0, np.nan)
        # 절대비율(÷SPY) 대신 섹터 간 백분위 순위로 표시
        sectors_mom = adj_mom.drop("SPY", errors="ignore").dropna()
        if sectors_mom.empty: return "🔄 섹터: 데이터 없음"
        ranked = sectors_mom.rank(pct=True) * 100  # 0~100, 높을수록 강세
        sorted_ranked = ranked.sort_values(ascending=False)
        fmt = lambda t, v: f"  {sectors.get(t,t)}({t}): 상위{max(1, round(100-v)):.0f}%  {float(r1m.get(t,0)):+.1%}/1M"
        return ("🔄 <b>섹터 로테이션</b>\n<b>▲ 강세</b>\n" + "\n".join(fmt(t,v) for t,v in sorted_ranked.head(3).items()) +
                "\n<b>▼ 약세</b>\n" + "\n".join(fmt(t,v) for t,v in sorted_ranked.tail(3).items()))
    except Exception as e:
        return f"🔄 섹터: 오류 ({e})"


def check_coppock() -> str:
    print("  [7/10] 코폭 지표 분석 중...")
    results = []
    for ticker, name in [("SPY","S&P500"),("QQQ","NASDAQ"),("^KS11","KOSPI")]:
        try:
            data = _yf_download(ticker, start="2000-01-01", auto_adjust=False)
            if data is None:
                results.append(f"  {name}: 데이터 없음")
                continue
            # 단일 티커 MultiIndex 대응
            if isinstance(data.columns, pd.MultiIndex):
                lvl0 = data.columns.get_level_values(0).unique()
                col = "Adj Close" if "Adj Close" in lvl0 else "Close"
                close = data[col].squeeze()
            else:
                close = data["Adj Close"] if "Adj Close" in data.columns else data["Close"]
            close = pd.Series(close.values, index=close.index, dtype=float).dropna()
            monthly = close.resample("ME").last().dropna()
            # 미완월(현재 월)의 최신 가격이 월말 데이터에 없으면 추가
            if close.index[-1] > monthly.index[-1]:
                monthly.loc[close.index[-1]] = float(close.iloc[-1])
                monthly = monthly.sort_index()
            # 전통 코폭 공식: WMA(ROC(14)+ROC(11), 10)  — EWM 대신 WMA 사용
            roc = (monthly.pct_change(14) + monthly.pct_change(11)) * 100
            roc = roc.clip(-200, 200).dropna()  # 데이터 이상치 방지
            _w = np.arange(1, 11, dtype=float)
            _norm = _w.sum()
            coppock = roc.rolling(10).apply(lambda x: (x * _w).sum() / _norm, raw=True).dropna()
            if len(coppock) < 2:
                results.append(f"  {name}: 데이터 부족")
                continue
            val, prev = float(coppock.iloc[-1]), float(coppock.iloc[-2])
            if val > 0 and prev <= 0:   sig = "골든크로스! 🎯"
            elif val < 0 and prev >= 0: sig = "데드크로스! ⚠️"
            elif val > 0:               sig = "양수권 ✅"
            else:                       sig = "음수권 ⚠️"
            results.append(f"  {name}: {val:.1f} {'↑' if val>prev else '↓'}  ({sig})")
        except Exception:
            results.append(f"  {name}: 계산 실패")
    return ("📈 <b>코폭 지표 (월간)</b>  장기 추세 전환 신호\n"
            "  0선 상향돌파=골든크로스(매수신호)\n" + "\n".join(results))


def check_breadth() -> str:
    print("  [8/10] ZBT 시장 폭 분석 중...")
    try:
        sample = ["AAPL","MSFT","NVDA","AMZN","META","GOOGL","BRK-B","LLY","JPM","V",
                  "UNH","XOM","TSLA","MA","AVGO","PG","COST","HD","MRK","ABBV",
                  "CVX","KO","PEP","WMT","ADBE","CRM","BAC","TMO","ACN","AMD",
                  "NFLX","TXN","NEE","QCOM","DHR","LIN","AMGN","ORCL","MDT","HON",
                  "RTX","LOW","SPGI","INTU","ISRG","SBUX","CAT","GS","AXP","BLK",
                  "DE","ELV","GILD","SYK","REGN","ZTS","VRTX","PANW","LRCX","KLAC",
                  "AMAT","MU","MRVL","SNPS","CDNS","PAYX","ADP","MSI","ADI","CRWD",
                  "WFC","USB","PNC","MS","C","SLB","EOG","COP","MPC","VLO"]
        end_dt = datetime.today()
        cl_short = _yf_download(sample, start=(end_dt-timedelta(days=45)).strftime("%Y-%m-%d"), auto_adjust=False)
        if cl_short is None: return "⚡ ZBT: 데이터 일시 불가"
        cl_short = (cl_short["Close"] if isinstance(cl_short.columns, pd.MultiIndex) else cl_short).ffill().dropna(axis=1, how="any")
        zbt = ((cl_short.pct_change() > 0).sum(axis=1) / cl_short.shape[1]).rolling(10).mean().dropna()
        zbt_now = float(zbt.iloc[-1])
        cl_long = _yf_download(sample, period="1y", auto_adjust=False)
        if cl_long is None: return f"⚡ ZBT: {zbt_now:.1%} (MA 데이터 없음)"
        cl_long = (cl_long["Close"] if isinstance(cl_long.columns, pd.MultiIndex) else cl_long).ffill().dropna(axis=1, how="any")
        above50 = sum(1 for col in cl_long.columns if len(cl_long[col].dropna()) >= 50 and cl_long[col].dropna().iloc[-1] > cl_long[col].dropna().rolling(50).mean().iloc[-1])
        above200 = sum(1 for col in cl_long.columns if len(cl_long[col].dropna()) >= 200 and cl_long[col].dropna().iloc[-1] > cl_long[col].dropna().rolling(200).mean().iloc[-1])
        n = len(cl_long.columns)
        if zbt_now > 0.615:         sig = "🟢 ZBT 강한 반등!"
        elif zbt_now > 0.55:        sig = "🟡 ZBT 반등 조짐"
        elif zbt_now < 0.45:        sig = "🔴 ZBT 약세"
        else:                       sig = "⚪ ZBT 중립"
        return (f"⚡ <b>ZBT + 시장 폭</b>  상승종목 비율 급반등 신호\n"
                f"  61.5%↑=강한반등 | 45%↓=약세\n"
                f"ZBT: <b>{zbt_now:.1%}</b>  {sig}\n"
                f"50일선 위: {above50/n*100:.0f}%  200일선 위: {above200/n*100:.0f}%")
    except Exception as e:
        return f"⚡ ZBT: 오류 ({e})"


def check_momentum_stocks() -> str:
    print("  [9/10] 모멘텀 + 거래대금 분석 중...")
    NAMES = {
        "AAPL":"애플","ADBE":"어도비","ADI":"아날로그디바이시스","ADP":"ADP",
        "ADSK":"오토데스크","AMAT":"어플라이드머티리얼즈","AMD":"AMD","AMGN":"암젠",
        "AMZN":"아마존","ANF":"아베크롬비","ASML":"ASML","AVGO":"브로드컴",
        "BIIB":"바이오젠","BKNG":"부킹홀딩스","CDNS":"케이던스","COST":"코스트코",
        "CRWD":"크라우드스트라이크","CSCO":"시스코","DDOG":"데이터독","EA":"EA게임즈",
        "FAST":"패스널","FTNT":"포티넷","GILD":"길리어드","GOOG":"구글",
        "HON":"허니웰","INTC":"인텔","INTU":"인튜이트","ISRG":"인튜이티브서지컬",
        "KLAC":"KLA","LIN":"린데","LRCX":"램리서치","MAR":"메리어트",
        "META":"메타","MRNA":"모더나","MRVL":"마벨테크","MSFT":"마이크로소프트",
        "MU":"마이크론","NFLX":"넷플릭스","NVDA":"엔비디아","PANW":"팔로알토",
        "PAYX":"페이첵스","PEP":"펩시코","PYPL":"페이팔","QCOM":"퀄컴",
        "REGN":"리제네론","ROST":"로스스토어","SBUX":"스타벅스","SNPS":"시놉시스",
        "TEAM":"아틀라시안","TMUS":"T모바일","TSLA":"테슬라","TXN":"텍사스인스트루먼트",
        "VRTX":"버텍스파마","WDAY":"워크데이","ZS":"지스케일러",
    }
    try:
        tickers = ["AAPL","ADBE","ADI","ADP","ADSK","AMAT","AMD","AMGN","AMZN","ANF",
                   "ASML","AVGO","BIIB","BKNG","CDNS","COST","CRWD","CSCO","DDOG","EA",
                   "FAST","FTNT","GILD","GOOG","HON","INTC","INTU","ISRG","KLAC","LIN",
                   "LRCX","MAR","META","MRNA","MRVL","MSFT","MU","NFLX","NVDA","PANW",
                   "PAYX","PEP","PYPL","QCOM","REGN","ROST","SBUX","SNPS","TEAM","TMUS",
                   "TSLA","TXN","VRTX","WDAY","ZS","SPY"]
        data = _yf_download(tickers, period="1y", auto_adjust=False)
        if data is None: return "📊 모멘텀: 데이터 일시 불가"
        if not isinstance(data.columns, pd.MultiIndex):
            return "📊 모멘텀: 데이터 구조 오류"
        close = (data["Adj Close"] if "Adj Close" in data.columns.get_level_values(0) else data["Close"]).ffill().dropna(axis=1, how="any")
        volume, close_raw = data["Volume"], data["Close"]
        if "SPY" not in close.columns: return "📊 모멘텀: SPY 없음"
        # 보유 행수에 맞게 룩백 상한 설정 (pct_change(n)은 n+1행 필요)
        n = len(close)
        p63  = min(63,  n - 2)
        p126 = min(126, n - 2)
        p252 = min(252, n - 2)
        mom = (close.pct_change(p63).iloc[-1]  * 0.5 +
               close.pct_change(p126).iloc[-1] * 0.3 +
               close.pct_change(p252).iloc[-1] * 0.2)
        # 절대비율(÷SPY) 대신 유니버스 내 백분위 순위로 표시
        stocks_mom = mom.drop("SPY", errors="ignore").dropna()
        if stocks_mom.empty:
            return "📊 모멘텀: 데이터 부족"
        ranked = stocks_mom.rank(pct=True) * 100   # 0~100, 높을수록 강한 모멘텀
        top8 = ranked.nlargest(8)
        r1m = close.pct_change(21).iloc[-1]
        rs_lines = []
        for i, (t, pct_rank) in enumerate(top8.items()):
            r1m_val = float(r1m.get(t, np.nan))
            r1m_str = f"{r1m_val:+.1%}" if not np.isnan(r1m_val) else "N/A"
            name = NAMES.get(t, t)
            rs_lines.append(f"  {i+1}. {t}({name})  상위{max(1, round(100-pct_rank)):.0f}%  {r1m_str}/1M")
        vol_rows = []
        for t in tickers:
            if t not in close_raw.columns or t not in volume.columns: continue
            dv = (close_raw[t] * volume[t]).dropna()
            if len(dv) >= 6:
                pct = (float(dv.iloc[-1]) - float(dv.iloc[-6:-1].mean())) / float(dv.iloc[-6:-1].mean())
                vol_rows.append({"t": t, "pct": pct})
        result = "📊 <b>모멘텀 상위 8종목</b>  유니버스 내 상대 순위\n" + "\n".join(rs_lines)
        if vol_rows:
            top_vol = sorted(vol_rows, key=lambda x: x["pct"], reverse=True)[:5]
            result += "\n\n💰 <b>거래대금 스파이크 Top 5</b>\n" + "\n".join(
                f"  {i+1}. {r['t']}({NAMES.get(r['t'], r['t'])})  {r['pct']:+.0%}" for i,r in enumerate(top_vol))
        return result
    except Exception as e:
        return f"📊 모멘텀: 오류 ({e})"


def _cboe_latest(index_name: str) -> float | None:
    """CBOE 직접 다운로드로 최신 지수값 조회 (^SKEW, ^VVIX 대체)."""
    url = f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{index_name}_History.csv"
    hdrs = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.cboe.com/"}
    try:
        r = requests.get(url, headers=hdrs, timeout=15)
        if not r.ok:
            return None
        lines = [l for l in r.text.strip().split("\n") if l.strip()]
        for line in reversed(lines[1:]):
            parts = line.split(",")
            if len(parts) >= 2:
                val_str = parts[-1].strip().strip('"')
                if val_str and val_str not in ("null", ""):
                    try:
                        return float(val_str)
                    except ValueError:
                        continue
    except Exception:
        pass
    return None


def check_tail_risk() -> str:
    print("  [10/10] 꼬리리스크 분석 중...")
    try:
        # VIX는 yfinance로 (안정적)
        vix_raw = _yf_download("^VIX", period="5d", auto_adjust=False)
        if vix_raw is not None:
            vix_s = (vix_raw["Close"] if isinstance(vix_raw.columns, pd.MultiIndex) else vix_raw).dropna()
            if isinstance(vix_s, pd.DataFrame):
                vix_s = vix_s.iloc[:, 0]
            vix_now = float(vix_s.dropna().iloc[-1]) if not vix_s.dropna().empty else None
        else:
            vix_now = None

        # SKEW / VVIX: yfinance 시도 → 실패 시 CBOE 직접
        def _get_index(yf_ticker, cboe_name):
            raw = _yf_download(yf_ticker, period="5d", auto_adjust=False)
            if raw is not None:
                s = (raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw).dropna()
                if isinstance(s, pd.DataFrame): s = s.iloc[:, 0]
                if not s.dropna().empty:
                    return float(s.dropna().iloc[-1])
            print(f"  yfinance {yf_ticker} 실패 → CBOE 직접 조회")
            return _cboe_latest(cboe_name)

        skew_now = _get_index("^SKEW", "SKEW")
        vvix_now = _get_index("^VVIX", "VVIX")

        if vix_now is None:
            return "🎯 꼬리리스크: VIX 데이터 없음"

        if skew_now is None or vvix_now is None:
            if vix_now >= 30:   regime = "🚨 CRITICAL (VIX 급등)"
            elif vix_now >= 22: regime = "⚠️ ELEVATED"
            elif vix_now >= 15: regime = "🟡 MODERATE"
            else:               regime = "🟢 LOW"
            missing = []
            if skew_now is None: missing.append("SKEW")
            if vvix_now is None: missing.append("VVIX")
            return (f"🎯 <b>꼬리리스크</b>\n"
                    f"  SKEW=기관풋옵션강도  VIX=공포지수\n"
                    f"VIX(공포): {vix_now:.1f}  {regime}\n"
                    f"  ({'/'.join(missing)} 오늘 미제공)")

        skew_score = np.clip((skew_now - 100) / 50 * 100, 0, 100)
        vvix_score = np.clip((vvix_now - 70) / 80 * 100, 0, 100)
        composite = skew_score * 0.50 + vvix_score * 0.50
        if composite >= 75:   regime = "🚨 CRITICAL"
        elif composite >= 55: regime = "⚠️ ELEVATED"
        elif composite >= 35: regime = "🟡 MODERATE"
        else:                 regime = "🟢 LOW"
        alerts = []
        if skew_now > 135 and vix_now < 20:
            alerts.append("🚨 VIX 안정 속 SKEW 급등 — 기관 풋옵션 대량 매수")
        if vvix_now > 110 and vix_now < 18:
            alerts.append("🚨 VVIX/VIX 발산 — VIX 급등 선행 신호")
        result = (f"🎯 <b>꼬리리스크 트라이앵글</b>\n"
                  f"  SKEW=기관풋옵션강도  VVIX=공포의공포  VIX=공포지수\n"
                  f"복합: <b>{composite:.0f}/100</b>  {regime}\n"
                  f"SKEW: {skew_now:.1f}  VVIX: {vvix_now:.1f}  VIX(공포): {vix_now:.1f}")
        if alerts:
            result += "\n" + "\n".join(alerts)
        return result
    except Exception as e:
        return f"🎯 꼬리리스크: 오류 ({e})"


def check_tga() -> str:
    print("  [11/11] TGA 잔고 분석 중...")
    try:
        records = _fetch_tga_records()
        if len(records) < 2:
            raise ValueError("데이터 부족")

        bal = records[-1]           # 십억달러 (B)
        chg_w = bal - records[-2]   # 전주 대비
        chg_m = bal - records[max(0, len(records) - 5)]  # ~1달전 대비

        if chg_w < -100:   sig = "🟢 급감 — 시중 유동성 대량 공급 (호재)"
        elif chg_w < -30:  sig = "🟢 감소 — 유동성 소폭 공급"
        elif chg_w > 100:  sig = "🔴 급증 — 시중 유동성 대량 흡수 (악재)"
        elif chg_w > 30:   sig = "🟡 증가 — 유동성 소폭 흡수"
        else:              sig = "⚪ 보합"

        return (f"🏦 <b>TGA 잔고</b>  미국 재무부 당좌계좌\n"
                f"  잔고↓=시중에 돈 풀림(호재) | 잔고↑=흡수(악재)\n"
                f"현재: <b>${bal:,.0f}B</b>  전주: {chg_w:+,.0f}B  전월: {chg_m:+,.0f}B\n{sig}")
    except Exception as e:
        return f"🏦 TGA: 데이터 일시 불가 ({e})"


def _fetch_tga_records() -> list:
    """TGA 잔고(십억달러) 리스트 반환. 재무부 공식 API → FRED CSV 순으로 시도."""
    hdrs = {"User-Agent": "Mozilla/5.0"}

    # ── ① 미국 재무부 공식 API (fiscaldata.treasury.gov) ──────────────
    # close_today_bal은 null, 최신 구조에서는 open_today_bal에 전일 마감잔고가 들어있음
    try:
        url = ("https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/"
               "accounting/dts/operating_cash_balance"
               "?fields=record_date,account_type,open_today_bal"
               "&filter=account_type:eq:Treasury General Account (TGA) Closing Balance"
               "&sort=-record_date&page[size]=40")
        r = requests.get(url, headers=hdrs, timeout=20)
        if r.ok:
            data = r.json().get("data", [])
            vals = []
            for item in reversed(data):  # 오래된 순으로 쌓기
                try:
                    v = float(item["open_today_bal"])
                    if v > 0:
                        vals.append(v / 1000)  # 백만→십억달러(B)
                except (ValueError, KeyError, TypeError):
                    pass
            if len(vals) >= 2:
                print("  TGA: 재무부 공식 API 성공")
                return vals
    except Exception as e:
        print(f"  TGA 재무부 API 실패: {e}")

    # ── ② FRED CSV 폴백 ──────────────────────────────────────────────
    for attempt in range(3):
        try:
            r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=WTREGEN",
                             headers=hdrs, timeout=20)
            if r.ok:
                vals = []
                for line in r.text.strip().split("\n")[1:]:
                    parts = line.strip().split(",")
                    if len(parts) == 2 and parts[1] not in (".", "", "nan"):
                        try:
                            vals.append(float(parts[1]) / 1000)
                        except ValueError:
                            pass
                if len(vals) >= 2:
                    print("  TGA: FRED 폴백 성공")
                    return vals
        except Exception as e:
            print(f"  TGA FRED 시도 {attempt+1}/3: {e}")
            if attempt < 2:
                time.sleep(10)

    return []


def _is_trading_day() -> bool:
    # FORCE_RUN=true 이면 주말/공휴일도 강제 발송
    if os.environ.get("FORCE_RUN", "false").lower() == "true":
        return True
    # GitHub Actions는 UTC로 실행 → KST(UTC+9)로 변환해서 판단
    from datetime import timezone
    kst_now = datetime.now(timezone.utc) + timedelta(hours=9)
    today = kst_now.date()
    if today.weekday() >= 5:  # 토(5) 일(6)
        return False
    kr_holidays = holidays.country_holidays("KR", years=today.year)
    return today not in kr_holidays


def main():
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"\n[{now}] ====== 시장 리포트 생성 시작 ======")

    if not _is_trading_day():
        print(f"[{now}] 오늘은 주말/공휴일 — 발송 건너뜀")
        sys.exit(0)

    # 시작 핑 — 이게 안 오면 GitHub Actions 자체 문제
    if not send_telegram(f"🔔 <b>리포트 시작</b>  {now}"):
        print("[FATAL] 텔레그램 연결 실패 — 토큰/Chat ID 확인 필요")
        sys.exit(1)

    header = (f"📋 <b>아침주식</b>\n🕐 {now} (KST)\n{'─'*26}\n"
              f"① 매크로  ② 공포탐욕  ③ BLOOD\n④ 카나리아  ⑤ Heat  ⑥ 섹터\n"
              f"⑦ 코폭  ⑧ ZBT  ⑨ RS+거래대금\n⑩ 꼬리리스크  ⑪ TGA잔고  🇰🇷 KR종목선정")

    sections = [check_macro, check_fear_greed, check_blood, check_canary, check_heat,
                check_sector_rotation, check_coppock, check_breadth, check_momentum_stocks,
                check_tail_risk, check_tga, check_kr_screening]

    send_telegram(header)
    time.sleep(0.3)

    failed = []
    for fn in sections:
        try:
            msg = fn()
        except Exception as e:
            msg = f"⚠️ {fn.__name__} 오류: {e}"
            failed.append(fn.__name__)
        # 첫 줄만 로그에 출력 (오류/정상 여부 확인용)
        first_line = msg.split('\n')[0][:80].replace('<b>','').replace('</b>','')
        print(f"  → {fn.__name__}: {first_line}", flush=True)
        # 전송 실패 시 30초 후 1회 재시도
        if not send_telegram(msg):
            time.sleep(30)
            if not send_telegram(msg):
                failed.append(f"{fn.__name__}(전송실패)")
        time.sleep(0.5)

    finish = datetime.now().strftime('%H:%M:%S')
    send_telegram(f"{'⚠️' if failed else '✅'} <b>리포트 완료</b>  {finish}" + (f"\n실패: {', '.join(failed)}" if failed else ""))
    print(f"[{finish}] 완료 / 실패: {failed or '없음'}")
    # 실패 섹션이 절반 이상이면 비정상 종료 → GitHub Actions 실패로 기록
    if len(failed) > len(sections) // 2:
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"[FATAL]\n{traceback.format_exc()}")
        try:
            send_telegram(f"🚨 <b>리포트 오류</b>\n{str(e)[:300]}")
        except Exception:
            pass
        sys.exit(1)
