# -*- coding: utf-8 -*-
"""
장마감 수급 오실레이터 — 텔레그램 발송
평일 16:10 KST 실행 (장 마감 후 네이버 수급 데이터 반영 시점)

계산식은 `외국인기관수급오실레이터 (700)(태린이아빠)(매일).xlsm`의 시트 수식을 그대로 이식했다.

    시기외(t)   = (기관순매수(t) + 외인순매수(t)) / 시가총액(t)
    시기외12(t) = 시기외(t)*(2/13) + 시기외12(t-1)*(11/13)      # 12일 EMA
    시기외26(t) = 시기외(t)*(2/27) + 시기외26(t-1)*(25/27)      # 26일 EMA
    MACD(t)     = 시기외12(t) - 시기외26(t)
    시그널(t)   = MACD(t)*(2/10) + 시그널(t-1)*(8/10)           # 9일 EMA
    오실(t)     = MACD(t) - 시그널(t)                           # 최종 출력값

시가총액 = 종가 x 상장주식수, 순매수금액 = 순매매량 x 종가 이므로 종가가 약분되어
시기외(t) = 순매매량(t) / 상장주식수 로 계산한다. 상장주식수는 네이버 수급 페이지의
외국인보유주수 / 외국인보유율로 역산한다(삼성전자 기준 네이버 공시값과 0.0001% 일치).
"""

import os
import sys
import time
import concurrent.futures
from datetime import datetime, timedelta, timezone

import requests
import holidays
from bs4 import BeautifulSoup as BS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kr_screening import KR_STOCKS, HDRS, _find_supply_table, _parse_supply_int

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
    print("TELEGRAM_TOKEN / TELEGRAM_CHAT_ID 환경변수 없음")
    sys.exit(1)

PAGES = 5           # 네이버 수급 페이지 수 (페이지당 20일 → 최대 100일)
MIN_DAYS = 40       # EMA26이 수렴하려면 최소 이 정도는 필요
A_FAST = 2 / 13     # 12일 EMA
A_SLOW = 2 / 27     # 26일 EMA
A_SIGNAL = 2 / 10   # 9일 시그널


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


def _fetch_frgn(code: str) -> list[dict]:
    """네이버 수급 페이지 파싱. 최신일이 앞. 컬럼: 날짜0 종가1 전일비2 등락률3 거래량4
    기관순매매량5 외국인순매매량6 외국인보유주수7 외국인보유율8"""
    rows = []
    for pg in range(1, PAGES + 1):
        for attempt in range(2):
            try:
                url = f"https://finance.naver.com/item/frgn.naver?code={code}&page={pg}"
                r = requests.get(url, headers=HDRS, timeout=15)
                if not r.ok:
                    break
                table = _find_supply_table(BS(r.content, "html.parser", from_encoding="euc-kr"))
                if table is None:
                    break
                found = 0
                for tr in table.find_all("tr"):
                    c = [x.get_text(strip=True) for x in tr.find_all("td")]
                    if len(c) < 9 or not c[0] or "." not in c[0]:
                        continue
                    try:
                        held = int(c[7].replace(",", ""))
                        rate = float(c[8].replace("%", "").strip())
                    except (ValueError, AttributeError):
                        continue
                    rows.append({
                        "date": c[0],
                        "inst": _parse_supply_int(c[5]),
                        "forgn": _parse_supply_int(c[6]),
                        "held": held,
                        "rate": rate,
                    })
                    found += 1
                if found == 0 and pg == 1 and attempt == 0:
                    time.sleep(3)
                    continue
                break
            except requests.RequestException:
                if attempt == 0:
                    time.sleep(3)
                else:
                    break
        time.sleep(0.15)
    return rows


def _shares_outstanding(rows: list[dict]) -> float | None:
    """외국인보유주수 / 외국인보유율로 상장주식수 역산.
    보유율이 낮으면 반올림 오차가 커지므로 가장 높은 보유율 행을 쓴다."""
    best = max((r for r in rows if r["rate"] > 0), key=lambda r: r["rate"], default=None)
    if best is None or best["rate"] < 0.5:
        return None
    return best["held"] / (best["rate"] / 100)


def _ema(series: list[float], alpha: float) -> list[float]:
    """엑셀과 동일: 첫 값을 시드로 두고 v*alpha + prev*(1-alpha)."""
    out = [series[0]]
    for v in series[1:]:
        out.append(v * alpha + out[-1] * (1 - alpha))
    return out


def _oscillator(ticker: str) -> dict | None:
    code = ticker.replace(".KS", "").replace(".KQ", "")
    try:
        rows = _fetch_frgn(code)
    except Exception:
        return None
    if len(rows) < MIN_DAYS:
        return None

    shares = _shares_outstanding(rows)
    if not shares:
        return None

    rows = rows[::-1]  # 오래된 날짜부터 (EMA 진행 방향)
    sigiwe = [(r["inst"] + r["forgn"]) / shares for r in rows]

    ema12 = _ema(sigiwe, A_FAST)
    ema26 = _ema(sigiwe, A_SLOW)
    macd = [f - s for f, s in zip(ema12, ema26)]
    signal = _ema(macd, A_SIGNAL)
    osc = [m - s for m, s in zip(macd, signal)]

    now, prev = osc[-1], osc[-2]

    # 0선 돌파가 최우선 신호, 그다음이 진행 방향
    if now > 0 and prev <= 0:
        trend, emoji, rank = "매수전환", "🎯", 0
    elif now > 0 and now > prev:
        trend, emoji, rank = "매수가속", "🔥", 1
    elif now > 0:
        trend, emoji, rank = "매수우위", "🟢", 2
    elif now < 0 and prev >= 0:
        trend, emoji, rank = "매도전환", "⚠️", 3
    elif now < 0 and now < prev:
        trend, emoji, rank = "매도가속", "🔴", 5
    else:
        trend, emoji, rank = "매도우위", "🟡", 4

    return {
        "ticker": ticker,
        "name": KR_STOCKS[ticker],
        "osc": now,
        "prev": prev,
        "macd": macd[-1],
        "signal": signal[-1],
        "days": len(rows),
        "trend": trend,
        "emoji": emoji,
        "rank": rank,
    }


def _bp(v: float) -> str:
    """오실 값은 발행주식수 대비 비율이고 0.0003%~0.06% 범위라 bp로 표시한다."""
    return f"{v * 10000:+.2f}"


def build_message() -> str:
    tickers = list(KR_STOCKS.keys())
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(_oscillator, t): t for t in tickers}
        for fut in concurrent.futures.as_completed(futures):
            try:
                r = fut.result()
            except Exception:
                r = None
            if r:
                results.append(r)

    if not results:
        return "📊 <b>수급 오실레이터</b>\n데이터 수집 실패 — 네이버 응답 없음"

    kst = datetime.now(timezone.utc) + timedelta(hours=9)
    header = (f"📊 <b>수급 오실레이터</b>\n"
              f"🕐 {kst.strftime('%Y-%m-%d %H:%M')} (KST)\n"
              f"  외국인+기관 순매수 ÷ 시총 → MACD(12,26,9)\n"
              f"  오실 = MACD − 시그널  (발행주식수 대비 bp)\n"
              f"  🎯매수전환 🔥매수가속 🟢매수우위\n"
              f"  ⚠️매도전환 🟡매도우위 🔴매도가속\n"
              f"{'─' * 26}")

    results.sort(key=lambda r: (r["rank"], -r["osc"]))

    groups: dict[str, list] = {}
    for r in results:
        groups.setdefault(r["trend"], []).append(r)

    body = []
    for trend in ("매수전환", "매수가속", "매수우위", "매도전환", "매도우위", "매도가속"):
        rows = groups.get(trend)
        if not rows:
            continue
        body.append(f"\n{rows[0]['emoji']} <b>{trend}</b> ({len(rows)}종목)")
        for r in rows:
            body.append(f"  {r['name']}  오실 {_bp(r['prev'])} → {_bp(r['osc'])}bp")

    missing = len(tickers) - len(results)
    footer = f"\n\n※ 데이터 부족/실패 {missing}종목" if missing else ""

    return header + "\n".join(body) + footer


def _is_trading_day() -> bool:
    if os.environ.get("FORCE_RUN", "false").lower() == "true":
        return True
    today = (datetime.now(timezone.utc) + timedelta(hours=9)).date()
    if today.weekday() >= 5:
        return False
    return today not in holidays.country_holidays("KR", years=today.year)


def main():
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"\n[{now}] ====== 수급 오실레이터 시작 ======")

    if not _is_trading_day():
        print(f"[{now}] 오늘은 주말/공휴일 — 발송 건너뜀")
        sys.exit(0)

    msg = build_message()
    print(msg.replace("<b>", "").replace("</b>", ""))

    if not send_telegram(msg):
        print("[FATAL] 텔레그램 발송 실패")
        sys.exit(1)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 발송 완료")


if __name__ == "__main__":
    main()
