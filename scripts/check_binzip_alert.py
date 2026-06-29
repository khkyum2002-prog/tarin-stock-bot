# -*- coding: utf-8 -*-
"""
삼성전자 빈집전환 감지 + Windows 팝업 알림
조건: 40일 누적 기관+외국인 순매수 < 0 (빈집) AND 최근 5일 > 0 (전환)
"""
import sys, os, subprocess, datetime, requests
from bs4 import BeautifulSoup as BS

TARGET = "005930"
HDRS = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.naver.com/"}
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "binzip_log.txt")


def fetch_daily_supply(code: str, days: int = 40) -> list:
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


def windows_notify(title: str, msg: str):
    ps_code = f"""
Add-Type -AssemblyName System.Windows.Forms
$n = New-Object System.Windows.Forms.NotifyIcon
$n.Icon = [System.Drawing.SystemIcons]::Warning
$n.BalloonTipIcon = [System.Windows.Forms.ToolTipIcon]::Warning
$n.BalloonTipTitle = '{title}'
$n.BalloonTipText = '{msg}'
$n.Visible = $true
$n.ShowBalloonTip(15000)
Start-Sleep 16
$n.Dispose()
"""
    subprocess.Popen(
        ["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", ps_code],
        creationflags=0x08000000,
    )


def log(msg: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    log("=== 삼성전자 빈집전환 체크 시작 ===")
    daily = fetch_daily_supply(TARGET, days=40)

    if len(daily) < 5:
        log("데이터 부족 -- 스킵")
        sys.exit(0)

    short5 = sum(daily[:5])
    long40 = sum(daily[:40]) if len(daily) >= 40 else sum(daily)
    short5_uk = short5 / 1e8
    long40_uk = long40 / 1e8

    log(f"5일 수급: {short5_uk:+.1f}억  |  40일 누적: {long40_uk:+.1f}억")

    is_empty  = long40 < 0
    is_inflow = short5 > 0

    if is_empty and is_inflow:
        log("!!! 빈집전환 신호 감지 !!!")
        windows_notify(
            "삼성전자 빈집전환!",
            f"5일: {short5_uk:+.1f}억 / 40일누적: {long40_uk:+.1f}억\n수급 바닥 반등 시작 패턴"
        )
    else:
        reasons = []
        if not is_empty:
            reasons.append(f"40일 누적 양수({long40_uk:+.1f}억)")
        if not is_inflow:
            reasons.append(f"5일 수급 음수({short5_uk:+.1f}억)")
        log("신호 없음: " + " / ".join(reasons))

    sys.exit(0)


if __name__ == "__main__":
    main()
