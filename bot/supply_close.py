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
                    try:
                        close = int(c[1].replace(",", ""))
                    except ValueError:
                        continue
                    rows.append({
                        "date": c[0],
                        "close": close,
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
        # 차트용 시계열 (엑셀 수급오실레이터 시트의 G열=시가총액, H열=오실)
        "dates": [r["date"] for r in rows],
        "mktcap": [r["close"] * shares for r in rows],
        "osc_series": osc,
    }


def _bp(v: float) -> str:
    """오실 값은 발행주식수 대비 비율이고 0.0003%~0.06% 범위라 bp로 표시한다."""
    return f"{v * 10000:+.2f}"


def send_photo(png: bytes, caption: str = "", retries: int = 3) -> bool:
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption[:1000], "parse_mode": "HTML"}
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, data=data,
                                 files={"photo": ("chart.png", png, "image/png")},
                                 timeout=60)
            if resp.ok:
                return True
            print(f"사진 전송 오류(시도{attempt}): {resp.text[:200]}")
        except Exception as e:
            print(f"사진 전송 실패(시도{attempt}): {e}")
        if attempt < retries:
            time.sleep(5)
    return False


def _setup_font() -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager as fm
    have = {f.name for f in fm.fontManager.ttflist}
    for cand in ("NanumGothic", "NanumBarunGothic", "Noto Sans CJK KR",
                 "Noto Sans KR", "Malgun Gothic", "AppleGothic"):
        if cand in have:
            matplotlib.rcParams["font.family"] = cand
            break
    else:
        print("  [경고] 한글 폰트 없음 — 라벨이 깨질 수 있음")
    matplotlib.rcParams["axes.unicode_minus"] = False


def _chart_single(r: dict) -> bytes:
    """엑셀 수급오실레이터 차트와 동일 구성: 시가총액(좌축) + 수급오실레이터(우축) 이중축 선그래프."""
    import matplotlib.pyplot as plt
    from io import BytesIO

    n = len(r["dates"])
    x = list(range(n))
    osc_bp = [v * 10000 for v in r["osc_series"]]

    fig, ax1 = plt.subplots(figsize=(9, 4.2))
    ax1.plot(x, [v / 1e12 for v in r["mktcap"]], color="#1F4E79", lw=1.4)
    ax1.set_ylabel("시가총액 (조원)", color="#1F4E79", fontsize=9)
    ax1.tick_params(axis="y", labelcolor="#1F4E79", labelsize=8)

    ax2 = ax1.twinx()
    ax2.plot(x, osc_bp, color="#FF0000", lw=1.4)
    ax2.axhline(0, color="gray", lw=0.9, ls="--")
    ax2.set_ylabel("수급오실레이터 (bp)", color="#FF0000", fontsize=9)
    ax2.tick_params(axis="y", labelcolor="#FF0000", labelsize=8)

    step = max(1, n // 8)
    ax1.set_xticks(x[::step])
    ax1.set_xticklabels([r["dates"][i][5:] for i in x[::step]], fontsize=8, rotation=45)
    # 이모지는 한글 폰트에 글리프가 없어 두부로 깨지므로 차트에는 쓰지 않는다
    ax1.set_title(f"{r['name']}  수급오실레이터  [{r['trend']}]  ({_bp(r['osc'])}bp)",
                  fontsize=11, color=("#C00000" if r["osc"] < 0 else "#1F4E79"))
    ax1.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return buf.getvalue()


def _chart_grid(results: list[dict]) -> bytes:
    """전 종목을 한 장에. 각 칸은 오실(적색)과 시가총액(회색, 정규화)을 겹쳐 그린다."""
    import matplotlib.pyplot as plt
    from io import BytesIO

    n = len(results)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 1.9))
    axes = axes.ravel() if n > 1 else [axes]

    for ax, r in zip(axes, results):
        osc_bp = [v * 10000 for v in r["osc_series"]]
        x = list(range(len(osc_bp)))
        mc = r["mktcap"]
        lo, hi = min(mc), max(mc)
        span = (hi - lo) or 1
        olo, ohi = min(osc_bp), max(osc_bp)
        ospan = (ohi - olo) or 1
        # 시가총액을 오실 축 범위로 정규화해 겹쳐 그림
        mc_scaled = [(v - lo) / span * ospan + olo for v in mc]
        ax.plot(x, mc_scaled, color="#B0B0B0", lw=0.9)
        ax.plot(x, osc_bp, color="#FF0000", lw=1.1)
        ax.axhline(0, color="gray", lw=0.7, ls="--")
        # NanumGothic에 bold 웨이트가 없어 굵기 대신 * 표시로 전환 종목을 구분한다
        mark = "*" if r["trend"] in ("매수전환", "매도전환") else ""
        ax.set_title(f"{mark}{r['name']}  {_bp(r['osc'])}", fontsize=8,
                     color=("#C00000" if r["osc"] < 0 else "#1F4E79"))
        ax.tick_params(labelsize=6)
        ax.set_xticks([])

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("수급오실레이터 (적색) vs 시가총액 (회색)  —  MACD(12,26,9), 단위 bp"
                 "   ※ * 표시 = 0선 돌파(전환)",
                 fontsize=11, y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.985))

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=105)
    plt.close(fig)
    return buf.getvalue()


def collect() -> list[dict]:
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(_oscillator, t): t for t in KR_STOCKS}
        for fut in concurrent.futures.as_completed(futures):
            try:
                r = fut.result()
            except Exception:
                r = None
            if r:
                results.append(r)
    results.sort(key=lambda r: (r["rank"], -r["osc"]))
    return results


def build_message(results: list[dict]) -> str:
    tickers = list(KR_STOCKS.keys())

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

    results = collect()
    msg = build_message(results)
    print(msg.replace("<b>", "").replace("</b>", ""))

    if not send_telegram(msg):
        print("[FATAL] 텔레그램 발송 실패")
        sys.exit(1)

    if not results:
        return

    # 차트: 전 종목 그리드 1장 + 전환 신호 종목만 개별 차트
    try:
        _setup_font()
        if not send_photo(_chart_grid(results), "📈 전 종목 수급오실레이터"):
            print("  그리드 차트 전송 실패")

        signals = [r for r in results if r["trend"] in ("매수전환", "매도전환")]
        for r in signals:
            cap = f"{r['emoji']} <b>{r['name']}</b> {r['trend']}  오실 {_bp(r['prev'])} → {_bp(r['osc'])}bp"
            if not send_photo(_chart_single(r), cap):
                print(f"  {r['name']} 차트 전송 실패")
            time.sleep(0.5)
        print(f"  차트 발송: 그리드 1장 + 개별 {len(signals)}장")
    except Exception as e:
        print(f"  차트 생성 실패 (텍스트는 발송됨): {e}")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] 발송 완료")


if __name__ == "__main__":
    main()
