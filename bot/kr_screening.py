# -*- coding: utf-8 -*-
"""
KR 종목 선정 — 빈집여력 + RS 스크리닝
market_report.py에서 호출해 텔레그램으로 발송
"""

import os
import sys
import time
import concurrent.futures
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from bs4 import BeautifulSoup as BS

HDRS = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.naver.com/"}

KR_STOCKS = {
    # ── 반도체 ──────────────────────────────────────────────────
    "005930.KS": "삼성전자",
    "000660.KS": "SK하이닉스",
    "042700.KS": "한미반도체",
    "011070.KS": "LG이노텍",
    "009150.KS": "삼성전기",
    # ── 항공·물류 ────────────────────────────────────────────────
    "003490.KS": "대한항공",
    # ── IT·플랫폼 ───────────────────────────────────────────────
    "035420.KS": "NAVER",
    "035720.KS": "카카오",
    # ── 화학·소재 ───────────────────────────────────────────────
    "051910.KS": "LG화학",
    "006400.KS": "삼성SDI",
    # ── 바이오 ──────────────────────────────────────────────────
    "196170.KS": "알테오젠",
    "207940.KS": "삼성바이오로직스",
    "068270.KS": "셀트리온",
    "326030.KS": "SK바이오팜",
    # ── 건설·중공업 ─────────────────────────────────────────────
    "028260.KS": "삼성물산",
    "010140.KS": "삼성중공업",
    "000720.KS": "현대건설",
    # ── 에너지 ──────────────────────────────────────────────────
    "010950.KS": "S-Oil",
    "047050.KS": "포스코인터내셔널",
    # ── 방산 ────────────────────────────────────────────────────
    "012450.KS": "한화에어로스페이스",
    "064350.KS": "현대로템",
    "272210.KS": "한화시스템",
    # ── 조선 ────────────────────────────────────────────────────
    "009540.KS": "HD한국조선해양",
    "010620.KS": "HD현대미포",
    "042660.KS": "한화오션",
    "329180.KS": "HD현대중공업",
    # ── 2차전지 ─────────────────────────────────────────────────
    "247540.KS": "에코프로비엠",
    "086520.KS": "에코프로",
    "373220.KS": "LG에너지솔루션",
    "305720.KS": "롯데에너지머티리얼즈",
    # ── K뷰티·식품 ──────────────────────────────────────────────
    "090430.KS": "아모레퍼시픽",
    "161890.KS": "한국콜마",
    "003230.KS": "삼양식품",
    # ── 자동차 ──────────────────────────────────────────────────
    "005380.KS": "현대차",
    "000270.KS": "기아",
    "012330.KS": "현대모비스",
    # ── 금융 ────────────────────────────────────────────────────
    "105560.KS": "KB금융",
    "055550.KS": "신한지주",
    "086790.KS": "하나금융지주",
    # ── 로봇·AI ─────────────────────────────────────────────────
    "277810.KS": "레인보우로보틱스",
    "348210.KS": "넥스틴",
    # ── 게임·엔터 ───────────────────────────────────────────────
    "036570.KS": "엔씨소프트",
    "251270.KS": "넷마블",
    "035900.KS": "JYP엔터테인먼트",
}


def _parse_supply_int(s: str) -> int:
    s = s.replace(",", "").replace("+", "").strip()
    if not s or s == "-" or s == "－":
        return 0
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return 0


def _naver_supply_single(code: str, days: int = 40) -> list:
    daily = []
    for pg in range(1, 5):
        if len(daily) >= days:
            break
        try:
            url = f"https://finance.naver.com/item/frgn.naver?code={code}&page={pg}"
            r = requests.get(url, headers=HDRS, timeout=12)
            soup = BS(r.content, "html.parser", from_encoding="euc-kr")
            tables = soup.find_all("table")
            if len(tables) < 4:
                break
            for row in tables[3].find_all("tr"):
                cells = [c.get_text(strip=True) for c in row.find_all("td")]
                if len(cells) >= 7 and cells[0] and "." in cells[0]:
                    try:
                        cp = int(cells[1].replace(",", ""))
                        iq = _parse_supply_int(cells[5])
                        fq = _parse_supply_int(cells[6])
                        if cp > 0:
                            daily.append((iq + fq) * cp / 1e8)
                            if len(daily) >= days:
                                break
                    except Exception:
                        pass
        except Exception:
            break
        time.sleep(0.1)
    return daily


def _fetch_supply_worker(ticker: str) -> tuple:
    code = ticker.replace(".KS", "").replace(".KQ", "")
    daily = _naver_supply_single(code, days=40)
    if not daily:
        return ticker, 0.0, 0.0, True
    return ticker, sum(daily[:min(5, len(daily))]), sum(daily[:min(40, len(daily))]), False


def check_kr_screening() -> str:
    print("  [KR] 국내 종목 빈집여력 스크리닝 중...")
    rows = []
    tickers = list(KR_STOCKS.keys())

    try:
        bench_raw = yf.download(tickers + ["^KS11"], period="6mo", auto_adjust=True, progress=False)
        close_all = bench_raw["Close"].dropna(how="all") if isinstance(bench_raw.columns, pd.MultiIndex) else bench_raw.dropna(how="all")
    except Exception:
        close_all = pd.DataFrame()

    bench_ret = None
    if "^KS11" in close_all.columns and len(close_all) >= 20:
        bx = close_all["^KS11"].squeeze().dropna()
        n_b = min(66, len(bx) - 1)
        if n_b > 0 and float(bx.iloc[-n_b]) > 0:
            bench_ret = float(bx.iloc[-1] / bx.iloc[-n_b] - 1)

    rs_map = {}
    for t in tickers:
        if t in close_all.columns:
            s = close_all[t].squeeze().dropna()
            if len(s) >= 20:
                n = min(66, len(s) - 1)
                base = float(s.iloc[-n])
                if base > 0:
                    rs_map[t] = float(s.iloc[-1] / base - 1) - (bench_ret or 0.0)
                else:
                    rs_map[t] = 0.0
            else:
                rs_map[t] = 0.0
        else:
            rs_map[t] = 0.0

    rs_pct = (pd.Series(rs_map).rank(pct=True) * 100).round(1)

    long40_map, short5_map, fetch_failed = {}, {}, set()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for t, s5, l40, failed in pool.map(_fetch_supply_worker, tickers):
            if failed:
                fetch_failed.add(t)
            else:
                short5_map[t] = s5
                long40_map[t] = l40

    if long40_map:
        bz_pct = ((1 - pd.Series(long40_map).rank(pct=True, ascending=True)) * 100).round(1)
    else:
        bz_pct = pd.Series({t: 50.0 for t in tickers})

    binzip_set = set()
    for t in tickers:
        if t not in fetch_failed and long40_map.get(t, 1) <= 0 and short5_map.get(t, 0) > 0:
            binzip_set.add(t)

    for t in tickers:
        rs = float(rs_pct.get(t, 50))
        bz = float(bz_pct.get(t, 50))
        score = rs * 0.40 + bz * 0.60
        is_bz = t in binzip_set

        if is_bz:
            grade = "🏚️빈집전환"
        elif rs >= 65 and bz >= 65:
            grade = "⭐강력"
        elif rs >= 55 and bz >= 55:
            grade = "✅유망"
        elif rs >= 45 or bz >= 55:
            grade = "👀관심"
        else:
            grade = None

        if grade:
            rows.append({"ticker": t, "name": KR_STOCKS[t], "rs": rs, "bz": bz, "score": score, "grade": grade, "binzip": is_bz})

    if not rows:
        return f"🇰🇷 <b>KR 종목 선정</b>\n  해당 종목 없음"

    rows.sort(key=lambda x: (not x["binzip"], -x["score"]))

    lines = []
    for r in rows[:12]:
        l40_val = long40_map.get(r["ticker"], 0)
        s5_val = short5_map.get(r["ticker"], 0)
        lines.append(f"  {r['grade']} {r['name']}\n    RS {r['rs']:.0f}  빈집여력 {r['bz']:.0f}  (40일:{l40_val:+.0f}억/5일:{s5_val:+.0f}억)")

    failed_note = f"\n  ⚠️ 수급조회실패: {len(fetch_failed)}종목" if fetch_failed else ""
    today = datetime.now().strftime("%m/%d")
    return f"🇰🇷 <b>KR 종목 선정 ({today})</b>\n빈집여력↑=아직 덜 매집=기회\n{'─'*22}\n" + "\n".join(lines) + failed_note
