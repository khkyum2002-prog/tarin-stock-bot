# -*- coding: utf-8 -*-
"""
삼성전자 빈집전환 감지 + Windows 팝업 알림
조건: 40일 누적 기관+외국인 순매수 < 0 (빈집) AND 최근 5일 > 0 (전환)
"""
import sys, os, subprocess, datetime, time, requests

TARGET = "005930"
# 2026-09 네이버가 finance.naver.com을 SPA로 개편해 HTML 표 파싱이 죽었다(테이블 0개).
# 모바일 증권 API로 교체. trend는 10건씩 주므로 bizdate를 커서로 과거로 거슬러 올라간다.
API = "https://m.stock.naver.com/api/stock/{code}/trend"
HDRS = {
    "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15",
    "Referer": "https://m.stock.naver.com/",
}
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "binzip_log.txt")


def _int(s) -> int:
    try:
        return int(str(s).replace(",", "").replace("+", "").strip())
    except (ValueError, AttributeError):
        return 0


def fetch_daily_supply(code: str, days: int = 40) -> list:
    """일별 (기관+외국인) 순매수대금. 최신일이 앞."""
    daily, seen, cursor = [], set(), None
    for _ in range(days // 10 + 3):
        url = API.format(code=code) + (f"?bizdate={cursor}" if cursor else "")
        try:
            r = requests.get(url, headers=HDRS, timeout=15)
            if not r.ok:
                break
            batch = r.json()
        except Exception:
            break
        if not batch:
            break
        added = 0
        for e in batch:
            d = e.get("bizdate")
            cp = _int(e.get("closePrice"))
            if not d or d in seen or cp <= 0:
                continue
            seen.add(d)
            daily.append((_int(e.get("organPureBuyQuant")) + _int(e.get("foreignerPureBuyQuant"))) * cp)
            added += 1
            if len(daily) >= days:
                return daily
        if added == 0:
            break
        cursor = batch[-1].get("bizdate")
        if not cursor:
            break
        time.sleep(0.1)
    return daily


def gh_output(**kw):
    """워크플로가 steps.check.outputs.* 로 읽는 값. 이걸 안 쓰면 메일 단계가 영영 실행되지 않는다."""
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as f:
        for k, v in kw.items():
            f.write(f"{k}={v}\n")


def windows_notify(title: str, msg: str):
    if os.name != "nt":          # GitHub Actions(ubuntu)에서는 건너뛴다
        return
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
        # 조용히 넘어가면 데이터 소스가 죽어도 워크플로가 계속 success로 찍힌다
        log(f"데이터 수집 실패({len(daily)}일) -- 소스 점검 필요")
        sys.exit(1)

    short5 = sum(daily[:5])
    long40 = sum(daily[:40]) if len(daily) >= 40 else sum(daily)
    short5_uk = short5 / 1e8
    long40_uk = long40 / 1e8

    log(f"5일 수급: {short5_uk:+.1f}억  |  40일 누적: {long40_uk:+.1f}억")

    is_empty  = long40 < 0
    is_inflow = short5 > 0

    gh_output(binzip=str(is_empty and is_inflow).lower(),
              short5=f"{short5_uk:+.1f}", long40=f"{long40_uk:+.1f}")

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
