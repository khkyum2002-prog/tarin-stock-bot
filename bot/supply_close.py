# -*- coding: utf-8 -*-
"""
장마감 수급 오실레이터 — 텔레그램 발송
평일 18:30 KST 실행 (16시대에는 투자자별 수급이 아직 잠정치라 확정 이후로 잡는다)

계산식과 판정조건 모두 `외국인기관수급오실레이터 (700)(태린이아빠)(매일).xlsm`에서 가져왔다.

[계산]  원본은 FnGuide DataGuide 항목을 쓴다 (단위 KRW bil)
    U510320 = 5일누적 기관 순매수대금        U530320 = 5일누적 외국인총합계 순매수대금
    S102100 = 시가총액 (보통주만, 우선주 제외)

    시기외(t)   = (U510320 + U530320) / S102100
    시기외12(t) = 시기외(t)*(2/13) + 시기외12(t-1)*(11/13)      # 12일 EMA
    시기외26(t) = 시기외(t)*(2/27) + 시기외26(t-1)*(25/27)      # 26일 EMA
    MACD(t)     = 시기외12(t) - 시기외26(t)
    시그널(t)   = MACD(t)*(2/10) + 시그널(t-1)*(8/10)           # 9일 EMA
    오실(t)     = MACD(t) - 시그널(t)                           # 최종 출력값

[판정]  수급오실레이터 시트 L7:L12 + 조건부서식(cellIs lessThan $L$12)
    P90/P75/AVG/P25/P10 = 오실 자기 이력 77일의 백분위·평균
    현재값보다 작은 임계치 개수(0~5)로 구간을 정한다. 부호가 아니라 자기 이력 대비 위치다.

[전략]  일관성 시트에 적힌 원본 방법론 — "높은 확률의 종목군에서 수급 빈집만 공략함"
    RS가 강한 종목은 수급이 일시적으로 약해져도 재차 강해질 확률이 높으므로
    오실이 하위 구간(빈집)일 때가 관심 대상이다. 높다고 좋은 게 아니다.

FnGuide는 유료라 네이버로 대체한다. 순매수대금은 (순매매량 x 종가)로,
시가총액은 (종가 x 상장주식수)로 근사한다.
엑셀 내 FnGuide 실측값과 14종목 280표본 대조 결과 시총 오차 0.15% 이내, 시기외 평균오차 0.42bp.

[데이터 소스] 2026-09 네이버가 finance.naver.com을 SPA로 개편해 HTML 표 파싱이 죽었다
(테이블 0개, EUC-KR→UTF-8). 모바일 증권 API로 전환:
    m.stock.naver.com/api/stock/{code}/trend         일별 기관·외국인 순매매량 (10건씩, bizdate 커서)
    m.stock.naver.com/api/stock/{code}/integration   시가총액 (한글 문자열, 예 "1,517조 1,093억")
상장주식수 = 시가총액 / 종가 (삼성전자 검증 시 공시값과 오차 0.0000%)
"""

import os
import sys
import time
import concurrent.futures
from datetime import datetime, timedelta, timezone

import json
import re

import requests
import holidays

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kr_screening import KR_STOCKS

API_BASE = "https://m.stock.naver.com/api/stock"
API_HDRS = {
    "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15",
    "Referer": "https://m.stock.naver.com/",
}

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
    print("TELEGRAM_TOKEN / TELEGRAM_CHAT_ID 환경변수 없음")
    sys.exit(1)

PAGES = 5           # 네이버 수급 페이지 수 (페이지당 20일 → 최대 100일)
CUM_DAYS = 5        # 엑셀 입력이 "5일누적 순매수대금"
MIN_DAYS = 45       # EMA26 수렴분 + 누적으로 잃는 앞쪽 4일
A_FAST = 2 / 13     # 12일 EMA
A_SLOW = 2 / 27     # 26일 EMA
A_SIGNAL = 2 / 10   # 9일 시그널
PCT_WINDOW = 77     # 백분위 산출 기간 (엑셀 H8:H84 = 77일)
RS_DAYS = 66        # RS 기간 — kr_screening.py의 기존 정의와 동일
RS_STRONG = 65      # RS 백분위 강세 기준 — kr_screening.py의 "강력" 기준과 동일

# 현재값보다 작은 임계치 개수(P10·P25·평균·P75·P90 중) → 구간명
BANDS = {
    0: ("빈집", "🎯"),        # 하위 10% 이하 — 원본 전략의 공략 대상
    1: ("빈집근접", "🔵"),    # P10 ~ P25
    2: ("평균이하", "⚪"),
    3: ("평균이상", "🟡"),
    4: ("상위권", "🟠"),      # P75 ~ P90
    5: ("과열", "🔴"),        # P90 초과
}


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


def _int(s) -> int:
    """'+3,332,528' / '-2,208,594' / '259,500' → int"""
    try:
        return int(str(s).replace(",", "").replace("+", "").strip())
    except (ValueError, AttributeError):
        return 0


def _api(code: str, path: str):
    r = requests.get(f"{API_BASE}/{code}/{path}", headers=API_HDRS, timeout=15)
    return r.json() if r.ok else None


def _fetch_frgn(code: str) -> list[dict]:
    """모바일 증권 API로 일별 수급 수집. 최신일이 앞.

    2026-09 네이버가 finance.naver.com을 SPA로 개편하면서 기존 HTML 표 파싱이 죽었다
    (테이블 0개, 인코딩도 EUC-KR→UTF-8). trend API는 한 번에 10건만 주므로
    bizdate를 커서로 넘겨 과거로 거슬러 올라간다(해당 날짜 '이전' 10건을 반환)."""
    rows, seen, cursor = [], set(), None
    for _ in range(PAGES * 2 + 4):          # 10건씩 → 100일 확보에 여유분
        path = "trend" + (f"?bizdate={cursor}" if cursor else "")
        try:
            batch = _api(code, path)
        except requests.RequestException:
            break
        if not batch:
            break
        added = 0
        for e in batch:
            d = e.get("bizdate")
            close = _int(e.get("closePrice"))
            if not d or d in seen or close <= 0:
                continue
            seen.add(d)
            rows.append({
                "date": f"{d[:4]}.{d[4:6]}.{d[6:]}",
                "close": close,
                "inst": _int(e.get("organPureBuyQuant")),
                "forgn": _int(e.get("foreignerPureBuyQuant")),
            })
            added += 1
        if added == 0:
            break
        cursor = batch[-1].get("bizdate")   # 배치의 가장 오래된 날 → 다음 호출은 그 이전
        if len(rows) >= PAGES * 20 or not cursor:
            break
        time.sleep(0.1)
    return rows


def _won(s: str) -> float:
    """'1,517조 1,093억' → 1517109300000000.0"""
    t = str(s).replace(",", "").replace(" ", "")
    total = 0.0
    for unit, mul in (("조", 10 ** 12), ("억", 10 ** 8), ("만", 10 ** 4)):
        m = re.search(rf"(\d+){unit}", t)
        if m:
            total += int(m.group(1)) * mul
    if total == 0:
        try:
            total = float(re.sub(r"[^\d.]", "", t))
        except ValueError:
            return 0.0
    return total


def _shares_outstanding(code: str, latest_close: int) -> float | None:
    """상장주식수 = 시가총액 / 종가.

    개편된 사이트에는 외국인보유주수가 없어 기존 역산(보유주수/보유율)을 못 쓴다.
    integration API의 시총 문자열을 파싱해 나눈다. 삼성전자로 검증 시 공시값과 오차 0.0000%."""
    try:
        d = _api(code, "integration")
        mv = next((x["value"] for x in (d or {}).get("totalInfos", [])
                   if x.get("code") == "marketValue"), None)
    except requests.RequestException:
        return None
    if not mv or latest_close <= 0:
        return None
    won = _won(mv)
    return won / latest_close if won > 0 else None


def _kospi_return() -> float | None:
    """KOSPI의 RS_DAYS 수익률. kr_screening.py가 ^KS11을 벤치마크로 쓰는 것과 같은 역할."""
    url = ("https://api.finance.naver.com/siseJson.naver?symbol=KOSPI&requestType=1"
           f"&startTime={(datetime.now() - timedelta(days=200)).strftime('%Y%m%d')}"
           f"&endTime={datetime.now().strftime('%Y%m%d')}&timeframe=day")
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        rows = json.loads(r.text.replace("'", '"'))[1:]
        closes = [float(x[4]) for x in rows if x[4]]
        n = min(RS_DAYS, len(closes) - 1)
        if n <= 0 or closes[-1 - n] <= 0:
            return None
        return closes[-1] / closes[-1 - n] - 1
    except Exception as e:
        print(f"  KOSPI 조회 실패: {e}")
        return None


def _rank_pct(values: dict) -> dict:
    """pandas rank(pct=True)*100 과 동일 (동점은 평균 순위)."""
    items = sorted(values.items(), key=lambda kv: kv[1])
    n = len(items)
    out, i = {}, 0
    while i < n:
        j = i
        while j + 1 < n and items[j + 1][1] == items[i][1]:
            j += 1
        pct = ((i + j) / 2 + 1) / n * 100
        for k in range(i, j + 1):
            out[items[k][0]] = pct
        i = j + 1
    return out


def _percentile(values: list[float], p: float) -> float:
    """엑셀 PERCENTILE(=PERCENTILE.INC)과 동일: rank = p*(n-1), 선형보간.
    엑셀 캐시값과 대조해 오차 0 확인."""
    s = sorted(values)
    k = p * (len(s) - 1)
    f = int(k)
    return s[f] + (k - f) * (s[f + 1] - s[f]) if f + 1 < len(s) else s[f]


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

    shares = _shares_outstanding(code, rows[0]["close"])   # rows[0] = 최신 거래일
    if not shares:
        return None

    rows = rows[::-1]  # 오래된 날짜부터 (EMA 진행 방향)

    # 엑셀 외인!C14 / 기관!C14 헤더가 "5일누적 ... 순매수대금(일간)" 이므로
    # 오실레이터 입력은 일별이 아니라 5일 누적 순매수대금이다.
    amount = [(r["inst"] + r["forgn"]) * r["close"] for r in rows]   # 일별 순매수대금(원)
    dates, mktcap, sigiwe = [], [], []
    for i in range(CUM_DAYS - 1, len(rows)):
        mc = rows[i]["close"] * shares
        dates.append(rows[i]["date"])
        mktcap.append(mc)
        sigiwe.append(sum(amount[i - CUM_DAYS + 1:i + 1]) / mc)

    ema12 = _ema(sigiwe, A_FAST)
    ema26 = _ema(sigiwe, A_SLOW)
    macd = [f - s for f, s in zip(ema12, ema26)]
    signal = _ema(macd, A_SIGNAL)
    osc = [m - s for m, s in zip(macd, signal)]

    now, prev = osc[-1], osc[-2]

    # 판정: 수급오실레이터 시트 L7:L12 + 조건부서식(cellIs lessThan $L$12).
    # 오실 자기 이력의 백분위 5개를 현재값과 비교해 구간을 정한다. 부호 기준이 아니다.
    hist = osc[-PCT_WINDOW:]
    th = [_percentile(hist, p) for p in (0.10, 0.25)]
    th.append(sum(hist) / len(hist))                    # L9 = AVERAGE
    th += [_percentile(hist, p) for p in (0.75, 0.90)]
    rank = sum(1 for t in th if t < now)                # 강조되는 임계치 개수 0~5

    # 일관성 시트: "높은 확률의 종목군에서 수급 빈집만 공략함" → 낮을수록 관심 대상
    trend, emoji = BANDS[rank]

    # RS용 수익률 (벤치마크 차감과 백분위 환산은 collect()에서)
    closes = [r["close"] for r in rows]
    n_rs = min(RS_DAYS, len(closes) - 1)
    ret = closes[-1] / closes[-1 - n_rs] - 1 if n_rs > 0 and closes[-1 - n_rs] > 0 else 0.0

    return {
        "ticker": ticker,
        "name": KR_STOCKS[ticker],
        "osc": now,
        "prev": prev,
        "macd": macd[-1],
        "signal": signal[-1],
        "days": len(rows),
        "ret": ret,
        "trend": trend,
        "emoji": emoji,
        "rank": rank,
        # 차트용 시계열 (엑셀 수급오실레이터 시트의 G열=시가총액, H열=오실)
        "dates": dates,
        "mktcap": mktcap,
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
        # 엑셀과 동일하게 이중축으로 겹쳐 그린다 (각 계열이 자기 범위로 자동 스케일)
        ax.plot(x, [v / 1e12 for v in r["mktcap"]], color="#8FAADC", lw=0.9)
        ax.tick_params(axis="y", labelsize=5, colors="#4472C4")
        ax2 = ax.twinx()
        ax2.plot(x, osc_bp, color="#FF0000", lw=1.1)
        ax2.axhline(0, color="gray", lw=0.7, ls="--")
        ax2.tick_params(axis="y", labelsize=5, colors="#C00000")
        # NanumGothic에 bold 웨이트가 없어 굵기 대신 * 표시로 전환 종목을 구분한다
        mark = "*" if r.get("target") else ""
        ax.set_title(f"{mark}{r['name']}  {_bp(r['osc'])}", fontsize=8,
                     color=("#C00000" if r["osc"] < 0 else "#1F4E79"))
        ax.tick_params(labelsize=6)
        ax.set_xticks([])

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("수급오실레이터 (적색, 우축 bp) vs 시가총액 (청색, 좌축 조원)"
                 "  —  5일누적 순매수 ÷ 시총 → MACD(12,26,9)   ※ * = 수급 빈집(하위 25% 이하)",
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
    # RS = 66일 초과수익률(종목 - KOSPI)의 유니버스 내 백분위 (kr_screening.py와 동일 정의)
    bench = _kospi_return() or 0.0
    rs_pct = _rank_pct({r["ticker"]: r["ret"] - bench for r in results})
    for r in results:
        r["rs"] = rs_pct.get(r["ticker"], 50.0)
        # 원본 전략의 타깃: RS가 강한 종목 중의 수급 빈집
        r["target"] = r["rank"] <= 1 and r["rs"] >= RS_STRONG

    # 구간 순(빈집 먼저), 구간 안에서는 오실 낮은 순 → 가장 깊은 빈집이 맨 위
    results.sort(key=lambda r: (r["rank"], r["osc"]))
    return results


def build_message(results: list[dict]) -> str:
    tickers = list(KR_STOCKS.keys())

    if not results:
        return "📊 <b>수급 오실레이터</b>\n데이터 수집 실패 — 네이버 응답 없음"

    kst = datetime.now(timezone.utc) + timedelta(hours=9)
    header = (f"📊 <b>수급 오실레이터</b>\n"
              f"🕐 {kst.strftime('%Y-%m-%d %H:%M')} (KST)\n"
              f"  5일누적 외국인+기관 순매수 ÷ 시총 → MACD(12,26,9)\n"
              f"  오실 = MACD − 시그널  (시총 대비 bp)\n"
              f"  자기 이력 {PCT_WINDOW}일 백분위 구간으로 판정\n"
              f"  🎯빈집 🔵빈집근접 ⚪평균이하 🟡평균이상 🟠상위권 🔴과열\n"
              f"  RS = KOSPI 대비 {RS_DAYS}일 초과수익 백분위(0~100)\n"
              f"  ⭐ = 빈집 + RS {RS_STRONG}↑ → 원본 전략의 공략 대상\n"
              f"{'─' * 26}")

    groups: dict[str, list] = {}
    for r in results:
        groups.setdefault(r["trend"], []).append(r)

    body = []
    for trend, _ in (BANDS[i] for i in range(6)):
        rows = groups.get(trend)
        if not rows:
            continue
        body.append(f"\n{rows[0]['emoji']} <b>{trend}</b> ({len(rows)}종목)")
        for r in rows:
            star = "⭐" if r.get("target") else "  "
            body.append(f"{star}{r['name']}  오실 {_bp(r['prev'])} → {_bp(r['osc'])}bp"
                        f"  RS {r.get('rs', 50):.0f}")

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

    # 수집이 통째로 실패하면 워크플로도 실패로 남겨야 한다.
    # 2026-09-10~11 네이버 개편으로 0종목이 수집됐는데 exit 0이라 success로 찍혀 이틀간 모르고 지나갔다.
    if len(results) < len(KR_STOCKS) // 2:
        send_telegram(f"🚨 <b>수급 오실레이터 수집 실패</b>\n"
                      f"{len(results)}/{len(KR_STOCKS)}종목만 수집됨 — 데이터 소스 점검 필요")
        print(f"[FATAL] 수집 {len(results)}/{len(KR_STOCKS)} — 실패 처리")
        sys.exit(1)
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

        # 전략 타깃(빈집 + RS강) 우선, 없으면 빈집 구간만이라도 보낸다
        signals = [r for r in results if r.get("target")]
        if not signals:
            signals = [r for r in results if r["trend"] == "빈집"]
        for r in signals:
            cap = (f"{r['emoji']} <b>{r['name']}</b> {r['trend']}"
                   f"  오실 {_bp(r['prev'])} → {_bp(r['osc'])}bp  RS {r.get('rs', 50):.0f}")
            if not send_photo(_chart_single(r), cap):
                print(f"  {r['name']} 차트 전송 실패")
            time.sleep(0.5)
        print(f"  차트 발송: 그리드 1장 + 개별 {len(signals)}장")
    except Exception as e:
        print(f"  차트 생성 실패 (텍스트는 발송됨): {e}")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] 발송 완료")


if __name__ == "__main__":
    main()
