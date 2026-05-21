# -*- coding: utf-8 -*-
"""
삼성전자 빈집전환 감지 스크립트
조건: 40일 누적 기관+외국인 순매수 < 0 (빈집) AND 최근 5일 > 0 (전환)
"""
import sys, os, requests
from bs4 import BeautifulSoup as BS

TARGET = "005930"  # 삼성전자
HDRS = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.naver.com/"}


def fetch_daily_supply(code: str, days: int = 40) -> list[float]:
    daily = []
    for pg in range(1, 4):
        if len(daily) >= days:
            break
        try:
            r = requests.get(
                f"https://finance.naver.com/item/frgn.naver?code={code}&page={pg}",
                headers=HDRS, timeout=15,
            )
            if r.status_code != 200:
                break
            soup = BS(r.content, "html.parser", from_encoding="euc-kr")
            tables = soup.find_all("table")
            if len(tables) < 4:
                break
            for row in tables[3].find_all("tr"):
                cells = [c.get_text(strip=True) for c in row.find_all("td")]
                if len(cells) >= 7 and cells[0] and "." in cells[0]:
                    try:
                        cp = int(cells[1].replace(",", ""))
                        iq = int(cells[5].replace(",", "").replace("+", "") or "0")
                        fq = int(cells[6].replace(",", "").replace("+", "") or "0")
                        if cp > 0:
                            daily.append((iq + fq) * cp)
                            if len(daily) >= days:
                                break
                    except Exception:
                        pass
        except Exception:
            break
    return daily


def main():
    daily = fetch_daily_supply(TARGET, days=40)

    if len(daily) < 5:
        print("데이터 부족 — 스킵")
        sys.exit(0)

    short5 = sum(daily[:5])
    long40 = sum(daily[:40]) if len(daily) >= 40 else sum(daily)

    short5_억 = short5 / 1e8
    long40_억 = long40 / 1e8

    print(f"삼성전자 | 5일 수급: {short5_억:+.1f}억  40일 누적: {long40_억:+.1f}억")

    is_empty = long40 < 0
    is_inflow = short5 > 0
    is_binzip = is_empty and is_inflow

    if is_binzip:
        print("🚨 빈집전환 신호 감지!")
        # GitHub Actions 환경: 출력 파일에 신호 기록 (workflow에서 읽음)
        signal_file = os.environ.get("GITHUB_OUTPUT", "")
        if signal_file:
            with open(signal_file, "a", encoding="utf-8") as f:
                f.write(f"binzip=true\n")
                f.write(f"short5={short5_억:.1f}\n")
                f.write(f"long40={long40_억:.1f}\n")
        sys.exit(0)
    else:
        status = []
        if not is_empty:
            status.append(f"40일 누적 양수({long40_억:+.1f}억) — 아직 빈집 아님")
        if not is_inflow:
            status.append(f"5일 수급 음수({short5_억:+.1f}억) — 아직 전환 안됨")
        print("정상: " + " / ".join(status))

        signal_file = os.environ.get("GITHUB_OUTPUT", "")
        if signal_file:
            with open(signal_file, "a", encoding="utf-8") as f:
                f.write("binzip=false\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
