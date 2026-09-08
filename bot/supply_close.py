# -*- coding: utf-8 -*-
"""
장마감 수급 오실레이터 — KR_STOCKS 전 종목의 외국인+기관 순매수 추세를 텔레그램 발송
평일 15:35 KST 실행 (장 마감 후 네이버 수급 데이터 반영 시점)
"""

import os
import sys
import time
import concurrent.futures
from datetime import datetime, timedelta, timezone

import requests
import holidays

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kr_screening import KR_STOCKS, _naver_supply_single

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
    print("TELEGRAM_TOKEN / TELEGRAM_CHAT_ID 환경변수 없음")
    sys.exit(1)

DAYS = 20
HALF = DAYS // 2


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


def _oscillator(ticker: str) -> dict | None:
    """단일 종목 수급 오실레이터. daily는 최신일이 인덱스 0, 단위는 억원."""
    code = ticker.replace(".KS", "").replace(".KQ", "")
    try:
        daily = _naver_supply_single(code, days=DAYS)
    except Exception:
        return None
    if len(daily) < HALF + 2:
        return None

    latter = sum(daily[:HALF])              # 최근 10일
    former = sum(daily[HALF:DAYS])          # 그 이전 10일
    recent5 = sum(daily[:min(5, len(daily))])
    total = sum(daily)

    # 빈집전환이 최우선: 20일 누적 순매도(빈집) 상태에서 최근 5일 순매수로 돌아선 종목
    # (binzip_alert.yml이 삼성전자 단일 종목에 쓰는 것과 동일한 기준)
    if total < 0 and recent5 > 0:
        trend, emoji, rank = "빈집전환", "🎯", 0
    elif latter > 0 and latter > abs(former) * 1.5:
        trend, emoji, rank = "유입가속", "🔥", 1
    elif latter > 0:
        trend, emoji, rank = "유입", "🟢", 2
    elif latter < 0 and abs(latter) > abs(former) * 1.5:
        trend, emoji, rank = "이탈가속", "🔴", 5
    elif latter < 0:
        trend, emoji, rank = "이탈", "🟡", 4
    else:
        trend, emoji, rank = "횡보", "⚪", 3

    return {
        "ticker": ticker,
        "name": KR_STOCKS[ticker],
        "recent5": recent5,
        "total": total,
        "latter": latter,
        "former": former,
        "trend": trend,
        "emoji": emoji,
        "rank": rank,
    }


def _fmt(v: float) -> str:
    if abs(v) >= 10000:
        return f"{v / 10000:+,.1f}조"
    return f"{v:+,.0f}억"


def build_message() -> str:
    tickers = list(KR_STOCKS.keys())
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(_oscillator, t): t for t in tickers}
        for fut in concurrent.futures.as_completed(futures):
            r = fut.result()
            if r:
                results.append(r)

    if not results:
        return "📊 <b>장마감 수급 오실레이터</b>\n데이터 수집 실패 — 네이버 응답 없음"

    kst = datetime.now(timezone.utc) + timedelta(hours=9)
    header = (f"📊 <b>장마감 수급 오실레이터</b>\n"
              f"🕐 {kst.strftime('%Y-%m-%d %H:%M')} (KST)\n"
              f"  외국인+기관 순매수 | 최근10일 vs 이전10일\n"
              f"  🎯빈집전환(20일↓ + 5일↑) = 바닥 매수유입\n"
              f"  🔥유입가속 🟢유입 ⚪횡보 🟡이탈 🔴이탈가속\n"
              f"{'─' * 26}")

    # 유입 강한 순 → 이탈 강한 순
    results.sort(key=lambda r: (r["rank"], -r["latter"]))

    groups: dict[str, list] = {}
    for r in results:
        groups.setdefault(r["trend"], []).append(r)

    body = []
    for trend in ("빈집전환", "유입가속", "유입", "횡보", "이탈", "이탈가속"):
        rows = groups.get(trend)
        if not rows:
            continue
        emoji = rows[0]["emoji"]
        body.append(f"\n{emoji} <b>{trend}</b> ({len(rows)}종목)")
        for r in rows:
            if trend == "빈집전환":
                # 판정 근거가 5일·20일이므로 그대로 노출
                body.append(f"  {r['name']}  20일{_fmt(r['total'])} → 5일{_fmt(r['recent5'])}")
            else:
                # 판정 근거인 최근10일 vs 이전10일을 노출 (표시값-분류 일치)
                body.append(f"  {r['name']}  이전10일{_fmt(r['former'])} → 최근10일{_fmt(r['latter'])}")

    # 최근 10일 순매수 상위 3종목을 주목 종목으로 별도 표기
    top = sorted(results, key=lambda r: -r["latter"])[:3]
    if top and top[0]["latter"] > 0:
        body.append(f"\n{'─' * 26}\n⭐ <b>수급 유입 TOP 3</b>")
        for i, r in enumerate(top, 1):
            if r["latter"] <= 0:
                break
            body.append(f"  {i}. {r['name']}  최근10일 {_fmt(r['latter'])}")

    missing = len(tickers) - len(results)
    footer = f"\n\n※ 데이터 수집 실패 {missing}종목" if missing else ""

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
    print(f"\n[{now}] ====== 장마감 수급 오실레이터 시작 ======")

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
