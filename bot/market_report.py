# -*- coding: utf-8 -*-
import os, sys, time
import requests, pandas as pd, numpy as np, yfinance as yf
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kr_screening import check_kr_screening

TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
    print("TELEGRAM_TOKEN / TELEGRAM_CHAT_ID 환경변수 없음")
    sys.exit(1)


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
        raw = yf.download(list(ticker_map.keys()), period="5d", auto_adjust=False, progress=False)
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
    try:
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        resp.raise_for_status()
        fg = resp.json()["fear_and_greed"]
        score = float(fg["score"])
        prev_close = float(fg["previous_close"])
        prev_week = float(fg["previous_1_week"])
        if score <= 25:   emoji, label = "😱", "극도의 공포"
        elif score <= 45: emoji, label = "😨", "공포"
        elif score <= 55: emoji, label = "😐", "중립"
        elif score <= 75: emoji, label = "😊", "탐욕"
        else:             emoji, label = "🤑", "극도의 탐욕"
        bar = "█" * int(score / 10) + "░" * (10 - int(score / 10))
        return (f"😨 <b>CNN 공포탐욕 지수</b>\n{emoji} <b>{score:.0f}/100</b>  {label}\n"
                f"[{bar}]\n전일대비: {score-prev_close:+.1f}  1주대비: {score-prev_week:+.1f}")
    except Exception:
        return "😨 공포탐욕: API 오류"


def check_blood() -> str:
    print("  [3/10] BLOOD 인디케이터 분석 중...")
    try:
        raw = yf.download(["^IRX", "^TNX", "HYG"], period="2y", auto_adjust=False, progress=False)
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        irx = close["^IRX"].dropna() / 100
        t10 = close["^TNX"].dropna()
        hyg_yield = yf.Ticker("HYG").info.get("dividendYield", 0.065) * 100
        hy_spread = max(hyg_yield - float(t10.iloc[-1]), 0.01)
        blood_now = float(irx.iloc[-1]) / (hy_spread / 100)
        blood_series = irx.reindex(t10.index, method="ffill").dropna() / (hy_spread / 100)
        ma20 = float(blood_series.rolling(20).mean().dropna().iloc[-1])
        ma60 = float(blood_series.rolling(60).mean().dropna().iloc[-1])
        if blood_now > ma20 > ma60:   status = "🟢 상승추세"
        elif blood_now < ma20 < ma60: status = "🔴 하락추세"
        else:                         status = "🟡 혼조"
        return (f"🩸 <b>BLOOD 인디케이터</b>\n현재: {blood_now:.3f}  MA20: {ma20:.3f}  MA60: {ma60:.3f}\n{status}")
    except Exception as e:
        return f"🩸 BLOOD: 오류 ({e})"


def check_canary() -> str:
    print("  [4/10] 카나리아 자산 분석 중...")
    try:
        start = (datetime.today() - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
        raw = yf.download(["QQQ", "TIP", "AGG", "GLD", "BIL"], start=start, auto_adjust=False, progress=False)
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
        lines = [f"  {k}: {v:+.2%}" for k, v in results.items()]
        return "📡 <b>카나리아 자산</b>\n" + signal + "\n" + "\n".join(lines)
    except Exception as e:
        return f"📡 카나리아: 오류 ({e})"


def check_heat() -> str:
    print("  [5/10] Heat 인디케이터 분석 중...")
    try:
        raw = yf.download(["SPY","QQQ","RSP","HYG","IEF","LQD","^VIX"], start="2015-01-01", auto_adjust=False, progress=False)
        px = (raw["Adj Close"] if isinstance(raw.columns, pd.MultiIndex) else raw).rename(columns={"^VIX":"VIX"}).sort_index()
        def zs(s): return (s - s.rolling(252).mean()) / s.rolling(252).std(ddof=0)
        def n01(z, lo, hi): return z.clip(lo, hi).sub(lo).div(hi - lo)
        df = pd.DataFrame(index=px.index)
        if {"HYG","IEF"}.issubset(px.columns): df["h1"] = n01(zs(px["HYG"]/px["IEF"]), 0, 2)
        if {"HYG","LQD"}.issubset(px.columns): df["h2"] = n01(zs(px["HYG"]/px["LQD"]), 0, 2)
        if {"RSP","SPY"}.issubset(px.columns): df["h3"] = ((-zs(px["RSP"]/px["SPY"]))/2).clip(0, 1)
        if "SPY" in px.columns: df["h4"] = n01(zs(px["SPY"]/px["SPY"].rolling(200).mean()-1), 0.5, 2)
        df["heat"] = (df[[c for c in df.columns if c.startswith("h")]].mean(axis=1) * 10).rolling(10).mean()
        heat_val = float(df["heat"].dropna().iloc[-1])
        if heat_val >= 7.5:   status = "🔴 과열"
        elif heat_val <= 2.5: status = "🟢 냉각 — 매수 기회"
        else:                 status = "🟡 정상"
        return f"🌡️ <b>BofA Heat</b>\n점수: <b>{heat_val:.1f}</b>/10  {status}"
    except Exception as e:
        return f"🌡️ Heat: 오류 ({e})"


def check_sector_rotation() -> str:
    print("  [6/10] 섹터 로테이션 분석 중...")
    try:
        sectors = {"XLK":"기술","XLC":"통신","XLY":"경기소비재","XLP":"필수소비재","XLI":"산업재",
                   "XLB":"소재","XLE":"에너지","XLF":"금융","XLV":"헬스케어","XLU":"유틸리티","XLRE":"리츠","SPY":"S&P500"}
        raw = yf.download(list(sectors.keys()), period="1y", auto_adjust=False, progress=False)
        cl = (raw["Adj Close"] if isinstance(raw.columns, pd.MultiIndex) and "Adj Close" in raw.columns.get_level_values(0)
              else raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw).ffill().dropna(axis=1, how="any")
        r1m = cl.pct_change(21).iloc[-1]
        adj_mom = (r1m*0.5 + cl.pct_change(63).iloc[-1]*0.3 + cl.pct_change(126).iloc[-1]*0.2) / cl.pct_change().rolling(63).std().iloc[-1].replace(0, np.nan)
        spy_score = adj_mom.get("SPY", None)
        if not spy_score: return "🔄 섹터: SPY 없음"
        rs = adj_mom.drop("SPY", errors="ignore") / abs(spy_score)
        sorted_rs = rs.sort_values(ascending=False)
        fmt = lambda t, v: f"  {sectors.get(t,t)}({t}): {v:.2f}x  {float(r1m.get(t,0)):+.1%}/1M"
        return ("🔄 <b>섹터 로테이션</b>\n<b>▲ 강세</b>\n" + "\n".join(fmt(t,v) for t,v in sorted_rs.head(3).items()) +
                "\n<b>▼ 약세</b>\n" + "\n".join(fmt(t,v) for t,v in sorted_rs.tail(3).items()))
    except Exception as e:
        return f"🔄 섹터: 오류 ({e})"


def check_coppock() -> str:
    print("  [7/10] 코폭 지표 분석 중...")
    results = []
    for ticker, name in [("SPY","S&P500"),("QQQ","NASDAQ"),("^KS11","KOSPI")]:
        try:
            data = yf.download(ticker, start="2000-01-01", auto_adjust=False, progress=False)
            close = (data["Adj Close"] if "Adj Close" in data.columns else data["Close"]).squeeze()
            monthly = close.resample("ME").last()
            monthly.loc[close.index[-1]] = float(close.iloc[-1])
            coppock = ((monthly.pct_change(14) + monthly.pct_change(11)) * 100).ewm(span=10, adjust=False).mean().dropna()
            val, prev = float(coppock.iloc[-1]), float(coppock.iloc[-2])
            if val > 0 and prev <= 0:   sig = "골든크로스!"
            elif val < 0 and prev >= 0: sig = "데드크로스!"
            elif val > 0:               sig = "양수권 ✅"
            else:                       sig = "음수권 ⚠️"
            results.append(f"  {name}: {val:.1f} {'↑' if val>prev else '↓'}  ({sig})")
        except Exception:
            results.append(f"  {name}: 계산 실패")
    return "📈 <b>코폭 지표 (월간)</b>\n" + "\n".join(results)


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
        cl_short = yf.download(sample, start=(end_dt-timedelta(days=45)).strftime("%Y-%m-%d"),
                               auto_adjust=False, progress=False)
        cl_short = (cl_short["Close"] if isinstance(cl_short.columns, pd.MultiIndex) else cl_short).ffill().dropna(axis=1, how="any")
        zbt = ((cl_short.pct_change() > 0).sum(axis=1) / cl_short.shape[1]).rolling(10).mean().dropna()
        zbt_now = float(zbt.iloc[-1])
        cl_long = yf.download(sample[:60], period="1y", auto_adjust=False, progress=False)
        cl_long = (cl_long["Close"] if isinstance(cl_long.columns, pd.MultiIndex) else cl_long).ffill().dropna(axis=1, how="any")
        above50 = sum(1 for col in cl_long.columns if len(cl_long[col].dropna()) >= 50 and cl_long[col].dropna().iloc[-1] > cl_long[col].dropna().rolling(50).mean().iloc[-1])
        above200 = sum(1 for col in cl_long.columns if len(cl_long[col].dropna()) >= 200 and cl_long[col].dropna().iloc[-1] > cl_long[col].dropna().rolling(200).mean().iloc[-1])
        n = len(cl_long.columns)
        if zbt_now > 0.615:         sig = "🟢 ZBT 강한 반등!"
        elif zbt_now > 0.55:        sig = "🟡 ZBT 반등 조짐"
        elif zbt_now < 0.45:        sig = "🔴 ZBT 약세"
        else:                       sig = "⚪ ZBT 중립"
        return (f"⚡ <b>ZBT + 시장 폭</b>\nZBT: <b>{zbt_now:.1%}</b>  {sig}\n"
                f"MA50 상회: {above50/n*100:.0f}%  MA200 상회: {above200/n*100:.0f}%")
    except Exception as e:
        return f"⚡ ZBT: 오류 ({e})"


def check_momentum_stocks() -> str:
    print("  [9/10] 모멘텀 + 거래대금 분석 중...")
    try:
        tickers = ["AAPL","ADBE","ADI","ADP","ADSK","AMAT","AMD","AMGN","AMZN","ANF",
                   "ASML","AVGO","BIIB","BKNG","CDNS","COST","CRWD","CSCO","DDOG","EA",
                   "FAST","FTNT","GILD","GOOG","HON","INTC","INTU","ISRG","KLAC","LIN",
                   "LRCX","MAR","META","MRNA","MRVL","MSFT","MU","NFLX","NVDA","PANW",
                   "PAYX","PEP","PYPL","QCOM","REGN","ROST","SBUX","SNPS","TEAM","TMUS",
                   "TSLA","TXN","VRTX","WDAY","ZS","SPY"]
        data = yf.download(tickers, period="1y", auto_adjust=False, progress=False)
        if not isinstance(data.columns, pd.MultiIndex):
            return "📊 모멘텀: 데이터 구조 오류"
        close = (data["Adj Close"] if "Adj Close" in data.columns.get_level_values(0) else data["Close"]).ffill().dropna(axis=1, how="any")
        volume, close_raw = data["Volume"], data["Close"]
        if "SPY" not in close.columns: return "📊 모멘텀: SPY 없음"
        avg = close.pct_change(63).iloc[-1]*0.5 + close.pct_change(126).iloc[-1]*0.3 + close.pct_change(252).iloc[-1]*0.2
        spy_m = avg.get("SPY", 0)
        if spy_m == 0: return "📊 모멘텀: SPY 0"
        rs = avg.drop("SPY", errors="ignore") / spy_m
        top8 = rs.nlargest(8)
        rs_lines = [f"  {i+1}. {t}  (RS {v:.2f}x  {float(close.pct_change(21).iloc[-1].get(t,0)):+.1%}/1M)" for i,(t,v) in enumerate(top8.items())]
        vol_rows = []
        for t in tickers:
            if t not in close_raw.columns or t not in volume.columns: continue
            dv = (close_raw[t] * volume[t]).dropna()
            if len(dv) >= 6:
                pct = (float(dv.iloc[-1]) - float(dv.iloc[-6:-1].mean())) / float(dv.iloc[-6:-1].mean())
                vol_rows.append({"t": t, "pct": pct})
        result = "📊 <b>NASDAQ RS 상위 8</b>\n" + "\n".join(rs_lines)
        if vol_rows:
            top_vol = sorted(vol_rows, key=lambda x: x["pct"], reverse=True)[:5]
            result += "\n\n💰 <b>거래대금 스파이크 Top 5</b>\n" + "\n".join(f"  {i+1}. {r['t']}  ({r['pct']:+.0%})" for i,r in enumerate(top_vol))
        return result
    except Exception as e:
        return f"📊 모멘텀: 오류 ({e})"


def check_tail_risk() -> str:
    print("  [10/10] 꼬리리스크 분석 중...")
    try:
        raw = yf.download(["^SKEW","^VVIX","^VIX"], period="1y", progress=False, auto_adjust=False)
        close = (raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw).ffill().dropna()
        if not all(t in close.columns for t in ["^SKEW","^VVIX","^VIX"]):
            return "🎯 꼬리리스크: 데이터 누락"
        skew_now = float(close["^SKEW"].iloc[-1])
        vvix_now = float(close["^VVIX"].iloc[-1])
        vix_now  = float(close["^VIX"].iloc[-1])
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
        result = (f"🎯 <b>꼬리리스크 트라이앵글</b>\n복합: <b>{composite:.0f}/100</b>  {regime}\n"
                  f"SKEW: {skew_now:.1f}  VVIX: {vvix_now:.1f}  VIX: {vix_now:.1f}")
        if alerts:
            result += "\n" + "\n".join(alerts)
        return result
    except Exception as e:
        return f"🎯 꼬리리스크: 오류 ({e})"


def main():
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"\n[{now}] ====== 시장 리포트 생성 시작 ======")

    # 시작 핑 — 이게 안 오면 GitHub Actions 자체 문제
    if not send_telegram(f"🔔 <b>리포트 시작</b>  {now}"):
        print("[FATAL] 텔레그램 연결 실패 — 토큰/Chat ID 확인 필요")
        sys.exit(1)

    header = (f"📋 <b>태린이아빠 시장 리포트</b>\n🕐 {now} (KST)\n{'─'*26}\n"
              f"① 매크로  ② 공포탐욕  ③ BLOOD\n④ 카나리아  ⑤ Heat  ⑥ 섹터\n"
              f"⑦ 코폭  ⑧ ZBT  ⑨ RS+거래대금\n⑩ 꼬리리스크  🇰🇷 KR종목선정")

    sections = [check_macro, check_fear_greed, check_blood, check_canary, check_heat,
                check_sector_rotation, check_coppock, check_breadth, check_momentum_stocks,
                check_tail_risk, check_kr_screening]

    send_telegram(header)
    time.sleep(0.3)

    failed = []
    for fn in sections:
        try:
            msg = fn()
        except Exception as e:
            msg = f"⚠️ {fn.__name__} 오류: {e}"
            failed.append(fn.__name__)
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
