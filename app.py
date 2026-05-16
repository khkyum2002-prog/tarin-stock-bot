import time, warnings, logging
import numpy as np
import pandas as pd
import yfinance as yf
import requests
import streamlit as st
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

st.set_page_config(page_title="?닿렐湲?二쇱떇", page_icon="?뱢", layout="centered")
st.markdown("""<style>
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px;
    padding: 10px 14px;
    margin: 3px 0;
}
[data-testid="stMetricValue"] { font-size: 1.1rem !important; }
[data-testid="stMetricLabel"] { font-size: 0.82rem !important; }
.stTabs [data-baseweb="tab"] { font-weight: 700; font-size: 0.95rem; }
hr { margin: 0.6rem 0; }
</style>""", unsafe_allow_html=True)

_days = ["??,"??,"??,"紐?,"湲?,"??,"??]
st.title("?뱢 ?닿렐湲?二쇱떇")
st.caption(f"?ㅻ뒛: {datetime.today().strftime('%Y-%m-%d')} ({_days[datetime.today().weekday()]}?붿씪)")

tab1, tab2, tab3 = st.tabs(["?뙉 誘멸뎅 吏??, "?눖?눟 援?궡 吏??, "?뵇 醫낅ぉ 遺꾩꽍"])

# ?????????????????????????????????????????????????????????????????????????????
# 怨듯넻 ?좏떥
# ?????????????????????????????????????????????????????????????????????????????
def _dl(ticker, period="3y", start=None, end=None, retries=3):
    for i in range(retries):
        try:
            kw = dict(auto_adjust=True, progress=False, threads=False)
            df = yf.download(ticker, start=start, end=end, **kw) if start else yf.download(ticker, period=period, **kw)
            if not df.empty: return df
        except Exception: pass
        time.sleep(2 * (i + 1))
    return pd.DataFrame()

def _close(ticker, period="3y", start=None, end=None):
    df = _dl(ticker, period=period, start=start, end=end)
    if df.empty: return pd.Series(dtype=float)
    if isinstance(df.columns, pd.MultiIndex):
        try: return df["Close"][ticker].dropna()
        except: return df["Close"].iloc[:, 0].dropna()
    return df["Close"].dropna()

def _rsi(series, window=10):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    return 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

def _macd_hist(series, fast=12, slow=26, signal=9):
    m = series.ewm(span=fast, adjust=False).mean() - series.ewm(span=slow, adjust=False).mean()
    return m - m.ewm(span=signal, adjust=False).mean()

def _td_back(n):
    d = datetime.today(); cnt = 0
    while cnt < n:
        d -= timedelta(days=1)
        if d.weekday() < 5: cnt += 1
    return d.strftime("%Y-%m-%d")

def _heat_label(h):
    if h >= 7.5: return "?뵶 怨쇱뿴"
    elif h >= 5.0: return "?윝 二쇱쓽"
    elif h >= 2.5: return "?윞 蹂댄넻"
    return "?윟 ?덉쟾"

def _cv(v, fmt=".2f"):
    color = "#00c853" if v >= 0 else "#ff4b4b"
    sign = "+" if v >= 0 else ""
    arrow = "?? if v >= 0 else "??
    return f'<span style="color:{color};font-weight:bold">{arrow} {sign}{v:{fmt}}</span>'

# ?????????????????????????????????????????????????????????????????????????????
# ?곗빱 ?대쫫 ?ъ쟾
# ?????????????????????????????????????????????????????????????????????????????
TICKER_NAMES = {
    "AAPL":"?좏뵆","MSFT":"留덉씠?щ줈?뚰봽??,"NVDA":"?붾퉬?붿븘","AMZN":"?꾨쭏議?,"META":"硫뷀?",
    "GOOGL":"援ш?A","GOOG":"援ш?C","TSLA":"?뚯뒳??,"LLY":"?쇰씪?대┫由?,"AVGO":"釉뚮줈?쒖뺨",
    "JPM":"JP紐④굔","UNH":"?좊굹?댄떚?쒗뿬??,"XOM":"?묒뒯紐⑤퉴","V":"鍮꾩옄","MA":"留덉뒪?곗뭅??,
    "PG":"P&G","JNJ":"議댁뒯?ㅼ〈??,"HD":"?덈뵒??,"COST":"肄붿뒪?몄퐫","ABBV":"?좊툕鍮?,
    "MRK":"癒명겕","NFLX":"?룻뵆由?뒪","CVX":"?먮툕濡?,"BAC":"諭낇겕?ㅻ툕?꾨찓由ъ뭅","CRM":"?몄씪利덊룷??,
    "ORCL":"?ㅻ씪??,"AMD":"AMD","WMT":"?붾쭏??,"PLTR":"?붾??곗뼱","IBM":"IBM","CAT":"罹먰꽣?꾨윭",
    "AMGN":"?붿젨","NOW":"?쒕퉬?ㅻ굹??,"ISRG":"?명뒠?댄떚釉뚯꽌吏而?,"QCOM":"?꾩뺨","UBER":"?곕쾭",
    "GS":"怨⑤뱶留뚯궘??,"HON":"?덈땲??,"MS":"紐④굔?ㅽ깲由?,"BKNG":"遺?뱁??⑹뒪","AXP":"?꾨찓由ъ뭏?듭뒪?꾨젅??,
    "BLK":"釉붾옓濡?,"GILD":"湲몃━?대뱶","PFE":"?붿씠??,"BA":"蹂댁엵","PANW":"?붾줈?뚰넗?ㅽ듃?띿뒪",
    "MU":"留덉씠?щ줎","SBUX":"?ㅽ?踰낆뒪","REGN":"由ъ젣?ㅻ줎","MELI":"硫붾Ⅴ移대룄由щ툕??,"MRNA":"紐⑤뜑??,
    "ASML":"ASML","NXPI":"NXP諛섎룄泥?,"CRWD":"?щ씪?곕뱶?ㅽ듃?쇱씠??,"DDOG":"?곗씠?곕룆","ZS":"吏?ㅼ??쇰윭",
    "INTU":"?명뒠?댄듃","AMAT":"?댄뵆?쇱씠?쒕㉧?곕━?쇱쫰","LRCX":"?⑤━?쒖튂","KLAC":"KLA","ADI":"?꾨궇濡쒓렇?붾컮?댁뒪",
    "INTC":"?명뀛","TXN":"?띿궗?ㅼ씤?ㅽ듃猷⑤㉫??,"CDNS":"耳?대뜕??,"SNPS":"?쒕냹?쒖뒪","FTNT":"?ы떚??,
    "TEAM":"?꾪??쇱떆??,"WDAY":"?뚰겕?곗씠","ADSK":"?ㅽ넗?곗뒪??,"APP":"?깅윭鍮?,"TTD":"?뷀듃?덉씠?쒕뜲?ㅽ겕",
    "SPY":"S&P500 ETF","QQQ":"?섏뒪??00 ETF","IWM":"?ъ?2000 ETF","TLT":"20?꾧뎅梨?ETF",
    "GLD":"湲?ETF","HYG":"?섏씠?쇰뱶梨꾧텒 ETF","IEF":"以묎린援?콈 ETF","RSP":"?숈씪媛以멣&P500",
    "XLK":"湲곗닠?뱁꽣","XLF":"湲덉쑖?뱁꽣","XLV":"?ъ뒪耳?댁꽮??,"XLE":"?먮꼫吏?뱁꽣",
    "XLI":"?곗뾽?ъ꽮??,"XLC":"?듭떊?뱁꽣","XLY":"?꾩쓽?뚮퉬?ъ꽮??,"XLP":"?꾩닔?뚮퉬?ъ꽮??,
    "XLB":"?뚯옱?뱁꽣","XLRE":"由ъ툩?뱁꽣","XLU":"?좏떥由ы떚?뱁꽣",
    "SOXX":"諛섎룄泥?ETF","IBB":"諛붿씠?ㅽ뀒??ETF","GDX":"湲덇킅??ETF","ARKK":"?곸떊湲곗뾽 ETF",
    "CIBR":"?ъ씠踰꾨낫??ETF","IGV":"?뚰봽?몄썾??ETF","TAN":"?쒖뼇愿?ETF","URA":"?곕씪??ETF",
    "JETS":"??났 ETF","KRE":"吏?????ETF","ITA":"??났?곗＜諛⑹궛 ETF","LIT":"由ы뒳諛고꽣由?ETF",
    "ACWI":"?꾩꽭怨꾩＜??ETF","EEM":"?좏씎援?ETF","VEA":"?좎쭊援?ETF",
    "BRK-B":"踰꾪겕?뷀빐?쒖썾??,"PEP":"?⑹떆肄?,"ACN":"?≪꽱痢꾩뼱","LIN":"由곕뜲","MCD":"留λ룄?좊뱶",
    "CSCO":"?쒖뒪肄?,"TMO":"?⑤え?쇱뀛","ADBE":"?대룄鍮?,"TMUS":"T-紐⑤컮??,"GE":"GE?먯뼱濡쒖뒪?섏씠??,
    "PM":"?꾨┰紐⑤━??,"TXN":"?띿궗?ㅼ씤?ㅽ듃猷⑤㉫??,"RTX":"?덉씠?쒖삩","SPGI":"S&P湲濡쒕쾶",
    "DHR":"?ㅻ굹??,"NEE":"?μ뒪?몄뿉?쇱뿉?덉?","LOW":"濡쒖슦??,"UNP":"?좊땲?⑦띁?쒗뵿",
    "SCHW":"李곗뒪?덉솑","C":"?⑦떚洹몃９","SYK":"?ㅽ듃?쇱씠而?,"DE":"議대뵒??,"MDT":"硫붾뱶?몃줈??,
    "AMAT":"?댄뵆?쇱씠?쒕㉧?곕━?쇱쫰","ETN":"?댄듉","VRTX":"踰꾪뀓?ㅽ뙆留?,"SBUX":"?ㅽ?踰낆뒪",
    "CB":"泥섎툕","MMC":"留덉돩留λ젅??,"SO":"?쒕뜕而댄띁??,"DUK":"??ъ뿉?덉?","BSX":"蹂댁뒪?댁궗?댁뼵?고뵿",
    "PLD":"?꾨·濡쒖???,"CI":"?쒓렇??,"ZTS":"議곗뿉?곗뒪","ICE":"?명꽣肄섑떚?⑦깉?듭뒪泥댁씤吏",
    "CME":"CME洹몃９","WM":"?⑥씠?ㅽ듃留ㅻ땲吏癒쇳듃","APH":"?뷀럹?","MCO":"臾대뵒??,"ITW":"?쇰━?몄씠?댁썙??,
    "NOC":"?몄뒪濡?렇猷⑤㉫","EMR":"?먮㉧?⑥씪?됲듃由?,
    # ?? ?쒓뎅 ??뺤＜ (KOSPI) ??
    "005930.KS":"?쇱꽦?꾩옄","000660.KS":"SK?섏씠?됱뒪","005380.KS":"?꾨?李?,"000270.KS":"湲곗븘",
    "005490.KS":"POSCO??⑹뒪","051910.KS":"LG?뷀븰","006400.KS":"?쇱꽦SDI","373220.KS":"LG?먮꼫吏?붾（??,
    "207940.KS":"?쇱꽦諛붿씠?ㅻ줈吏곸뒪","068270.KS":"??몃━??,"035420.KS":"NAVER","035720.KS":"移댁뭅??,
    "012330.KS":"?꾨?紐⑤퉬??,"066570.KS":"LG?꾩옄","028260.KS":"?쇱꽦臾쇱궛","009150.KS":"?쇱꽦?꾧린",
    "011070.KS":"LG?대끂??,"032830.KS":"?쇱꽦?앸챸","086790.KS":"?섎굹湲덉쑖吏二?,"105560.KS":"KB湲덉쑖",
    "055550.KS":"?좏븳吏二?,"316140.KS":"?곕━湲덉쑖吏二?,"138040.KS":"硫붾━痢좉툑?듭?二?,
    "096770.KS":"SK?대끂踰좎씠??,"010950.KS":"S-Oil","015760.KS":"?쒓뎅?꾨젰","036460.KS":"?쒓뎅媛?ㅺ났??,
    "017670.KS":"SK?붾젅肄?,"030200.KS":"KT","032640.KS":"LGU+",
    "012450.KS":"?쒗솕?먯뼱濡쒖뒪?섏씠??,"047810.KS":"?쒓뎅??났?곗＜","079550.KS":"LIG?μ뒪??,
    "009540.KS":"?쒓뎅議곗꽑?댁뼇","329180.KS":"HD?꾨?以묎났??,"010140.KS":"?쇱꽦以묎났??,"267260.KS":"HD?꾨??쇰젆?몃┃",
    "011210.KS":"?꾨??꾩븘","298040.KS":"?⑥꽦以묎났??,"010120.KS":"LS?쇰젆?몃┃","028050.KS":"?쇱꽦?붿??덉뼱留?,
    "000720.KS":"?꾨?嫄댁꽕","006360.KS":"GS嫄댁꽕","047040.KS":"??곌굔??,"028260.KS":"?쇱꽦臾쇱궛",
    "010130.KS":"怨좊젮?꾩뿰","011170.KS":"濡?뜲耳誘몄뭡","009830.KS":"?쒗솕?붾（??,
    "128940.KS":"?쒕??쏀뭹","326030.KS":"SK諛붿씠?ㅽ뙗","145020.KS":"?댁젮",
    "036570.KS":"?붿뵪?뚰봽??,"259960.KS":"?щ옒?꾪넠","251270.KS":"?룸쭏釉?,"352820.KS":"?섏씠釉?,
    "139480.KS":"?대쭏??,"004170.KS":"?좎꽭怨?,"023530.KS":"濡?뜲?쇳븨","000120.KS":"CJ??쒗넻??,
    "267250.KS":"HD?꾨?","034020.KS":"?먯궛?먮꼫鍮뚮━??,"042660.KS":"?쒗솕?ㅼ뀡",
    "003670.KS":"?ъ뒪肄뷀벂泥섏뿞","011790.KS":"SKC","096775.KS":"SK?붾Т釉?,
    # ?? ?쒓뎅 以묒냼??肄붿뒪????
    "196170.KQ":"?뚰뀒?ㅼ젨","086520.KQ":"?먯퐫?꾨줈","247540.KQ":"?먯퐫?꾨줈鍮꾩뿞",
    "066970.KQ":"?섏븻?먰봽","278280.KQ":"泥쒕낫","035900.KQ":"JYP Ent.","041510.KQ":"?먯뒪??,
    "277810.KS":"?덉씤蹂댁슦濡쒕낫?깆뒪","454910.KS":"?먯궛濡쒕낫?깆뒪","090360.KS":"濡쒕낫?ㅽ?",
    "263750.KQ":"?꾩뼱鍮꾩뒪","036030.KQ":"KG?대땲?쒖뒪","058470.KQ":"由щ끂怨듭뾽",
    "214150.KQ":"?대옒?쒖뒪","091990.KQ":"??몃━?⑦뿬?ㅼ???,"323410.KS":"移댁뭅?ㅻ콉??,
}

US_SECTOR_ETFS = {
    "XLK":"湲곗닠","SOXX":"諛섎룄泥?,"IGV":"?뚰봽?몄썾??,"CIBR":"?ъ씠踰꾨낫??,
    "XLF":"湲덉쑖","KRE":"吏?????,
    "XLV":"?ъ뒪耳??,"IBB":"諛붿씠?ㅽ뀒??,
    "XLE":"?먮꼫吏","XOP":"?앹쑀媛??,
    "XLI":"?곗뾽??,"ITA":"??났?곗＜諛⑹궛",
    "XLC":"?듭떊?쒕퉬??,
    "XLY":"?꾩쓽?뚮퉬??,"JETS":"??났",
    "XLP":"?꾩닔?뚮퉬??,
    "XLB":"?뚯옱","GDX":"湲덇킅??,
    "XLRE":"由ъ툩","XLU":"?좏떥由ы떚",
    "TAN":"?쒖뼇愿?,"URA":"?곕씪??,"LIT":"由ы뒳諛고꽣由?,"ARKK":"?곸떊湲곗뾽",
}

KOREA_ETFS = {
    "091170.KS":"KODEX ???,"140700.KS":"KODEX 蹂댄뿕","102970.KS":"KODEX 利앷텒",
    "117700.KS":"KODEX 嫄댁꽕","300950.KS":"KODEX 寃뚯엫?곗뾽","395160.KS":"KODEX ?쒖뒪?쒕컲?꾩껜",
    "445290.KS":"KODEX K-濡쒕큸","117460.KS":"KODEX ?먮꼫吏?뷀븰","091160.KS":"KODEX 諛섎룄泥?,
    "244580.KS":"KODEX 諛붿씠??,"228800.KS":"TIGER ?ы뻾?덉?","364970.KS":"TIGER 諛붿씠?짽OP10",
    "091180.KS":"KODEX ?먮룞李?,"305540.KS":"TIGER 2李⑥쟾吏?뚮쭏","266360.KS":"KODEX 誘몃뵒?댁뿏??,
    "228790.KS":"TIGER ?붿옣??,"463250.KS":"TIGER ?곗＜諛⑹궛","157490.KS":"TIGER ?뚰봽?몄썾??,
    "449450.KS":"PLUS K諛⑹궛","139230.KS":"TIGER 200以묎났??,"466920.KS":"SOL 議곗꽑TOP3",
    "475300.KS":"SOL 諛섎룄泥댁쟾怨듭젙","475310.KS":"SOL 諛섎룄泥댄썑怨듭젙","307510.KS":"TIGER ?섎즺湲곌린",
    "433500.KS":"ACE ?먯옄?ν뀒留?,"261070.KS":"TIGER 肄붿뒪?λ컮?댁삤","479850.KS":"HANARO K酉고떚",
    "381570.KS":"HANARO 移쒗솚寃쎌뿉?덉?","438900.KS":"HANARO FN K-?몃뱶",
}

SECTOR_ETFS = {
    "091160":"諛섎룄泥?,"305720":"2李⑥쟾吏","244580":"諛붿씠??,"091180":"?먮룞李?,
    "139270":"湲덉쑖","266370":"IT","445290":"濡쒕큸","139250":"嫄댁꽕湲곌퀎",
    "139220":"?뚯옱/?뷀븰","117460":"?먮꼫吏?뷀븰","143860":"?ъ뒪耳??,"139260":"?뺣낫湲곗닠",
    "466920":"?꾨젰湲곌린","449450":"諛⑹궛","494670":"?꾨젰TOP10",
}

_FALLBACK = {
    "諛섎룄泥?:[("005930","?쇱꽦?꾩옄"),("000660","SK?섏씠?됱뒪"),("042700","?쒕?諛섎룄泥?)],
    "?꾨젰湲곌린":[("267260","HD?꾨??쇰젆?몃┃"),("298040","?⑥꽦以묎났??),("010120","LS?쇰젆?몃┃")],
    "?먮룞李?:[("005380","?꾨?李?),("000270","湲곗븘"),("012330","?꾨?紐⑤퉬??)],
    "諛⑹궛":[("012450","?쒗솕?먯뼱濡쒖뒪?섏씠??),("047810","?쒓뎅??났?곗＜"),("079550","LIG?μ뒪??)],
    "諛붿씠??:[("207940","?쇱꽦諛붿씠?ㅻ줈吏곸뒪"),("068270","??몃━??),("128940","?쒕??쏀뭹")],
    "2李⑥쟾吏":[("373220","LG?먮꼫吏?붾（??),("006400","?쇱꽦SDI"),("051910","LG?뷀븰")],
    "濡쒕큸":[("277810","?덉씤蹂댁슦濡쒕낫?깆뒪"),("454910","?먯궛濡쒕낫?깆뒪"),("090360","濡쒕낫?ㅽ?")],
    "IT":[("005930","?쇱꽦?꾩옄"),("000660","SK?섏씠?됱뒪"),("035420","NAVER"),("035720","移댁뭅??)],
    "?뺣낫湲곗닠":[("005930","?쇱꽦?꾩옄"),("000660","SK?섏씠?됱뒪"),("035420","NAVER"),("035720","移댁뭅??)],
}

NAME_TO_TICKER = {v.lower(): k for k, v in TICKER_NAMES.items()}

NAVER_ETF_URL = "https://finance.naver.com/api/sise/etfItemList.nhn?etfType=0"
HEADERS = {"User-Agent":"Mozilla/5.0","Referer":"https://finance.naver.com"}
SP500_TOP100 = ["AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","BRK-B","TSLA","LLY","AVGO","JPM","UNH","XOM","V","MA","PG","JNJ","HD","COST","ABBV","MRK","NFLX","CVX","BAC","CRM","ORCL","AMD","PEP","ACN","WMT","LIN","MCD","CSCO","TMO","ADBE","PLTR","TMUS","INTU","GE","IBM","CAT","PM","AMGN","TXN","NOW","ISRG","QCOM","UBER","GS","VZ","HON","RTX","SPGI","DHR","NEE","MS","LOW","T","UNP","BKNG","AXP","SCHW","C","BLK","SYK","GILD","PFE","DE","MDT","BA","AMAT","ADI","LRCX","PANW","MU","TJX","ETN","VRTX","KLAC","SBUX","CB","MMC","SO","DUK","BSX","REGN","PLD","CI","ZTS","ICE","CME","WM","APH","MCO","SNPS","CDNS","ITW","NOC","EMR"]
NASDAQ100 = ["AAPL","ABNB","ADBE","ADI","ADP","ADSK","AEP","AMAT","AMD","AMGN","AMZN","ANSS","ASML","AVGO","AZN","BIIB","BKNG","BKR","CCEP","CDNS","CDW","CEG","CHTR","CMCSA","COST","CPRT","CRWD","CSCO","CSX","CTAS","CTSH","DASH","DDOG","DLTR","DXCM","EA","EXC","FANG","FAST","FTNT","GEHC","GILD","GOOG","GOOGL","HON","IDXX","INTC","INTU","ISRG","KDP","KHC","KLAC","LIN","LRCX","LULU","MAR","MCHP","MDLZ","MELI","META","MNST","MRNA","MRVL","MSFT","MU","NFLX","NVDA","NXPI","ODFL","ON","ORLY","PANW","PAYX","PCAR","PDD","PEP","PYPL","QCOM","REGN","ROP","ROST","SBUX","SNPS","TEAM","TMUS","TSLA","TTD","TXN","VRSK","VRTX","WDAY","XEL","ZS"]

# ?????????????????????????????????????????????????????????????????????????????
# 誘멸뎅 吏???⑥닔
# ?????????????????????????????????????????????????????????????????????????????
def get_canary_signal():
    try:
        end = datetime.today().strftime("%Y-%m-%d")
        start = (datetime.today() - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
        results = {}
        for t in ["QQQ","TIP"]:
            px = _close(t, start=start, end=end)
            if len(px) < 260: return None
            results[t] = float(((px.iloc[-1]/px.iloc[-22]-1)+(px.iloc[-1]/px.iloc[-63]-1)+(px.iloc[-1]/px.iloc[-126]-1)+(px.iloc[-1]/px.iloc[-252]-1))/4)
        return {"qqq_mom":results["QQQ"],"tip_mom":results["TIP"],"mode":"怨듦꺽" if all(v>0 for v in results.values()) else "諛⑹뼱"}
    except Exception as e: return {"error":str(e)}

def get_bofa_heat():
    try:
        tm = {"SPY":"SPY","QQQ":"QQQ","RSP":"RSP","VIX":"^VIX","HYG":"HYG","IEF":"IEF","LQD":"LQD"}
        raw = yf.download(list(tm.values()), start="2015-01-01", auto_adjust=True, progress=False)
        if raw.empty: return None
        px = raw["Close"].copy().rename(columns={v:k for k,v in tm.items()})
        def zs(s): mu=s.rolling(252).mean(); sd=s.rolling(252).std(ddof=0); return (s-mu)/sd
        def n01(z,lo=0.0,hi=2.0): return ((z-lo)/(hi-lo)).clip(0,1)
        df = pd.DataFrame(index=px.index)
        if {"HYG","IEF"}.issubset(px.columns): df["h_risk"] = n01(zs(px["HYG"]/px["IEF"]))
        if {"HYG","LQD"}.issubset(px.columns): df["h_credit"] = n01(zs(px["HYG"]/px["LQD"]))
        if {"RSP","SPY"}.issubset(px.columns): df["h_style"] = ((0-zs(px["RSP"]/px["SPY"]))/2).clip(0,1)
        if "SPY" in px.columns:
            spy_ma200 = px["SPY"].rolling(200).mean()
            df["h_spy_ext"] = n01(zs(px["SPY"]/spy_ma200-1),0.5,2.0)
        if "QQQ" in px.columns:
            qqq_ma200 = px["QQQ"].rolling(200).mean()
            df["h_qqq_ext"] = n01(zs(px["QQQ"]/qqq_ma200-1),0.5,2.0)
        hc = [c for c in df.columns if c.startswith("h_")]
        W = pd.Series({"h_risk":1.2,"h_credit":1.0,"h_style":0.8,"h_spy_ext":0.8,"h_qqq_ext":0.8})
        W = W[[c for c in hc if c in W.index]]; W = W/W.sum()
        heat = ((df[hc]*W).sum(axis=1)*10).rolling(10).mean()
        spy_ma200 = px["SPY"].rolling(200).mean()
        trend_on = (px["SPY"]>spy_ma200)&(spy_ma200.diff(20)>0)
        heat = pd.Series(np.maximum(heat.values,np.where(trend_on,2.5,0.0)),index=heat.index)
        shock_vix = bool(px["VIX"].pct_change(3).iloc[-1]>=0.30) if "VIX" in px.columns else False
        shock_credit = bool((px["HYG"]/px["LQD"]).pct_change(5).iloc[-1]<=-0.02) if {"HYG","LQD"}.issubset(px.columns) else False
        return {"heat":round(float(heat.dropna().iloc[-1]),2),"shock":shock_vix or shock_credit,
                "shock_vix":shock_vix,"shock_credit":shock_credit,"trend_on":bool(trend_on.dropna().iloc[-1])}
    except Exception as e: return {"error":str(e)}

def get_blood_indicator():
    try:
        raw = yf.download(["^IRX","^TNX","HYG","IEF"], start="2015-01-01", auto_adjust=True, progress=False)
        px = raw["Close"].copy()
        irx = px["^IRX"]   # Yahoo quotes in %, e.g. 5.0 = 5%
        t10y = px["^TNX"]  # Yahoo quotes in %, e.g. 4.5 = 4.5%
        hyg_ticker = yf.Ticker("HYG")
        hyg_yield_raw = getattr(hyg_ticker.fast_info,"dividend_yield",None) or hyg_ticker.info.get("dividendYield",0.06)
        hyg_yield = hyg_yield_raw * 100 if hyg_yield_raw < 1 else hyg_yield_raw  # convert decimal to %
        blood = (irx/(hyg_yield - t10y)).dropna()
        cur=float(blood.iloc[-1]); ma20=float(blood.rolling(20).mean().iloc[-1]); ma60=float(blood.rolling(60).mean().iloc[-1])
        return {"value":round(cur,4),"ma20":round(ma20,4),"ma60":round(ma60,4),"vs_ma20":"?? if cur>ma20 else "?꾨옒","vs_ma60":"?? if cur>ma60 else "?꾨옒"}
    except Exception as e: return {"error":str(e)}

def get_us_fear_greed():
    try:
        tickers = ["^GSPC","^IXIC","^VIX","^TNX","^FVX","HYG","IEF","QQQ","SPY"]
        raw = yf.download(tickers, start="2024-01-01", auto_adjust=True, progress=False)
        data = raw["Close"].rename(columns={"^GSPC":"SP500","^IXIC":"NASDAQ","^VIX":"VIX","^TNX":"T10Y","^FVX":"T5Y"}).dropna(subset=["SP500","NASDAQ","VIX","T10Y","T5Y","HYG","IEF","QQQ","SPY"])
        data["RiskApp"] = data["HYG"]/data["IEF"]
        def calc_fg(df, col, label):
            df[f"{label}_MA125"] = df[col].rolling(125).mean()
            df[f"{label}_Mom"] = (df[col]-df[f"{label}_MA125"])/df[f"{label}_MA125"]*100
            df[f"{label}_RSI"] = _rsi(df[col]); df[f"{label}_BS"] = df["T10Y"]-df["T5Y"]
            df[f"{label}_VIX"] = df["VIX"]; df[f"{label}_RA"] = df["RiskApp"]
            cols = [f"{label}_Mom",f"{label}_RSI",f"{label}_BS",f"{label}_VIX",f"{label}_RA"]
            v = df[cols].dropna()
            if v.empty: return df
            df.loc[v.index, cols] = MinMaxScaler().fit_transform(v)
            df[f"{label}_FGI"] = (df[f"{label}_Mom"]+df[f"{label}_RA"]+(1-df[f"{label}_VIX"])+df[f"{label}_BS"]+df[f"{label}_RSI"])*0.2
            df[f"{label}_Osc"] = _macd_hist(df[f"{label}_FGI"])
            return df
        data = calc_fg(data,"SP500","SPX"); data = calc_fg(data,"NASDAQ","NDX")
        for col, lbl in [("SPY","SPY"),("QQQ","QQQ")]:
            for w in [20,60,120,200]: data[f"{lbl}_MA{w}"] = data[col].rolling(w).mean()
            data[f"{lbl}_SuperMA"] = data[[f"{lbl}_MA{w}" for w in [20,60,120,200]]].mean(axis=1)
            data[f"{lbl}_GapPct"] = (data[col]-data[f"{lbl}_SuperMA"])/data[f"{lbl}_SuperMA"]*100
            data[f"{lbl}_EMA13"] = data[col].ewm(span=13,adjust=False).mean()
            data[f"{lbl}_MACDh"] = _macd_hist(data[col])
        def impulse(df, lbl):
            last = df[[f"{lbl}_EMA13",f"{lbl}_MACDh"]].dropna()
            if len(last)<2: return "?뚯닔?놁쓬"
            eu = last[f"{lbl}_EMA13"].iloc[-1]>last[f"{lbl}_EMA13"].iloc[-2]
            mu = last[f"{lbl}_MACDh"].iloc[-1]>last[f"{lbl}_MACDh"].iloc[-2]
            return "?윟 媛뺤꽭" if eu and mu else ("?뵶 ?쎌꽭" if not eu and not mu else "?뵷 以묐┰")
        def td_setup(s):
            p=s.values; sell=np.zeros(len(p)); buy=np.zeros(len(p))
            for i in range(len(p)):
                sell[i]=sell[i-1]+1 if i>=4 and p[i]>p[i-4] else 0
                buy[i]=buy[i-1]+1 if i>=2 and p[i]<p[i-2] else 0
            return int(sell[-1]),int(buy[-1])
        last = data.dropna(subset=["SPX_Osc","NDX_Osc"])
        spy_ts,spy_tb = td_setup(data["SPY"].dropna()); qqq_ts,qqq_tb = td_setup(data["QQQ"].dropna())
        return {"spx_osc":round(float(last["SPX_Osc"].iloc[-1]),4),"ndx_osc":round(float(last["NDX_Osc"].iloc[-1]),4),
                "spx_sentiment":"?먯슃" if last["SPX_Osc"].iloc[-1]>0 else "怨듯룷",
                "ndx_sentiment":"?먯슃" if last["NDX_Osc"].iloc[-1]>0 else "怨듯룷",
                "spy_gap":round(float(data["SPY_GapPct"].dropna().iloc[-1]),2),
                "qqq_gap":round(float(data["QQQ_GapPct"].dropna().iloc[-1]),2),
                "spy_impulse":impulse(data,"SPY"),"qqq_impulse":impulse(data,"QQQ"),
                "spy_td_sell":spy_ts,"spy_td_buy":spy_tb,"qqq_td_sell":qqq_ts,"qqq_td_buy":qqq_tb}
    except Exception as e: return {"error":str(e)}

def get_monthly_fear_greed():
    try:
        start = (datetime.today()-timedelta(days=25*365)).strftime('%Y-%m-%d')
        raw = yf.download(['^GSPC','^IXIC','^VIX','^TNX','^FVX','HYG','IEF'], start=start, auto_adjust=True, progress=False)
        data = raw["Close"].rename(columns={'^GSPC':'SP500','^IXIC':'NASDAQ','^VIX':'VIX','^TNX':'10Y','^FVX':'5Y'})
        data = data.resample('ME').ffill().dropna(); data['RA'] = data['HYG']/data['IEF']
        def calc(df, col, lbl):
            df[f'{lbl}_MA125'] = df[col].rolling(125).mean()
            df[f'{lbl}_Mom'] = (df[col]-df[f'{lbl}_MA125'])/df[f'{lbl}_MA125']*100
            df[f'{lbl}_RSI'] = _rsi(df[col],10); df[f'{lbl}_BS'] = df['10Y']-df['5Y']
            df[f'{lbl}_VIX'] = df['VIX']; df[f'{lbl}_RA'] = df['RA']
            cols=[f'{lbl}_Mom',f'{lbl}_RSI',f'{lbl}_BS',f'{lbl}_VIX',f'{lbl}_RA']
            v=df[cols].dropna()
            if v.empty: return df
            df.loc[v.index,cols]=MinMaxScaler().fit_transform(v)
            df[f'{lbl}_FGI']=(df[f'{lbl}_Mom']+df[f'{lbl}_RA']+(1-df[f'{lbl}_VIX'])+df[f'{lbl}_BS']+df[f'{lbl}_RSI'])*0.2
            ema6=df[f'{lbl}_FGI'].ewm(span=6,adjust=False).mean(); ema19=df[f'{lbl}_FGI'].ewm(span=19,adjust=False).mean()
            macd=ema6-ema19; df[f'{lbl}_Osc']=macd-macd.ewm(span=6,adjust=False).mean()
            return df
        data=calc(data,'SP500','SPX'); data=calc(data,'NASDAQ','NDX')
        r=data.dropna(subset=['SPX_Osc','NDX_Osc'])
        if r.empty: return {"error":"?곗씠??遺議?}
        return {"date":r.index[-1].strftime('%Y-%m'),"spx_osc":round(float(r['SPX_Osc'].iloc[-1]),4),
                "ndx_osc":round(float(r['NDX_Osc'].iloc[-1]),4),"spx_fgi":round(float(r['SPX_FGI'].iloc[-1]),4),
                "spx_sentiment":"?먯슃" if r['SPX_Osc'].iloc[-1]>0 else "怨듯룷",
                "ndx_sentiment":"?먯슃" if r['NDX_Osc'].iloc[-1]>0 else "怨듯룷"}
    except Exception as e: return {"error":str(e)}

def get_coppock():
    try:
        results = {}
        for t in ["SPY","QQQ","^GSPC"]:
            px=_close(t,period="5y")
            if px.empty: continue
            mo=px.resample("ME").last(); mo=pd.concat([mo,px.iloc[[-1]]]); mo=mo[~mo.index.duplicated(keep="last")].sort_index()
            cop=(mo.pct_change(14)+mo.pct_change(11)).ewm(span=10,adjust=False).mean()*100
            last=cop.dropna()
            if last.empty: continue
            val=float(last.iloc[-1]); prev=float(last.iloc[-2]) if len(last)>=2 else val
            results[{"SPY":"SPY","QQQ":"QQQ","^GSPC":"S&P500"}.get(t,t)]={"value":round(val,2),"trend":"?곸듅" if val>prev else "?섎씫","pos":val>0}
        return results
    except Exception as e: return {"error":str(e)}

def get_coppock_fast():
    try:
        results = {}
        for t in ["SPY","QQQ","^GSPC"]:
            px=_close(t,period="5y")
            if px.empty: continue
            mo=px.resample("ME").last(); mo=pd.concat([mo,px.iloc[[-1]]]); mo=mo[~mo.index.duplicated(keep="last")].sort_index()
            cop=(mo.pct_change(4)+mo.pct_change(6)).rolling(3).mean()*100
            last=cop.dropna()
            if last.empty: continue
            val=float(last.iloc[-1]); prev=float(last.iloc[-2]) if len(last)>=2 else val
            results[{"SPY":"SPY","QQQ":"QQQ","^GSPC":"S&P500"}.get(t,t)]={"value":round(val,2),"trend":"?곸듅" if val>prev else "?섎씫","pos":val>0}
        return results
    except Exception as e: return {"error":str(e)}

def get_zbt():
    try:
        universe = list(set(NASDAQ100 + SP500_TOP100[:80]))
        end_date = datetime.today(); start_date = end_date - timedelta(days=30)
        raw = yf.download(universe, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"), auto_adjust=True, progress=False)
        if raw.empty: return {"error":"?곗씠???놁쓬"}
        data = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        data = data.dropna(axis=1, how="all")
        daily_up = (data.pct_change()>0).sum(axis=1)/data.shape[1]
        zbt = daily_up.rolling(10).mean()
        cur = float(zbt.dropna().iloc[-1])
        prev_min = float(zbt.dropna().iloc[-10:-1].min()) if len(zbt.dropna())>=10 else cur
        vix_px = _close("^VIX", period="5d")
        vix_val = float(vix_px.iloc[-1]) if not vix_px.empty else None
        return {"zbt":round(cur,3),"prev_min":round(prev_min,3),
                "signal":cur>0.615 and prev_min<0.40,
                "vix":round(vix_val,1) if vix_val else None,
                "vix_ok":vix_val<20 if vix_val else None}
    except Exception as e: return {"error":str(e)}

def get_sp500_rs(tickers, top_n=10):
    try:
        spy_px=_close("SPY",period="3y")
        if spy_px.empty: return {"error":"SPY ?놁쓬"}
        rows=[]
        for i in range(0,len(tickers),50):
            raw=yf.download(tickers[i:i+50],period="3y",auto_adjust=True,progress=False)
            if raw.empty: continue
            px=raw["Close"] if isinstance(raw.columns,pd.MultiIndex) else raw
            if isinstance(px,pd.Series): px=px.to_frame()
            for t in px.columns:
                s=px[t].dropna(); al=pd.concat([s,spy_px],axis=1,join="inner"); al.columns=["stock","spy"]
                if len(al)<60: continue
                rs_vals=[]
                for win in [60,120,250]:
                    if len(al)<win: continue
                    rel=al["stock"]/al["spy"]; ma=rel.rolling(win).mean(); rs=((rel/ma)-1)*100
                    if not rs.dropna().empty: rs_vals.append(float(rs.dropna().iloc[-1]))
                if rs_vals: rows.append({"ticker":t,"rs":np.mean(rs_vals)})
            time.sleep(0.1)
        if not rows: return {"error":"RS 怨꾩궛 ?ㅽ뙣"}
        df=pd.DataFrame(rows).sort_values("rs",ascending=False).head(top_n)
        return {"top":[{"ticker":r["ticker"],"rs":round(r["rs"],1)} for _,r in df.iterrows()]}
    except Exception as e: return {"error":str(e)}

def get_nasdaq100_rs(tickers, top_n=10):
    try:
        raw=yf.download(tickers+["SPY"],period="1y",auto_adjust=True,progress=False)
        if raw.empty: return {"error":"?곗씠???놁쓬"}
        data=raw["Close"].ffill().dropna(axis=1,how="any")
        if "SPY" not in data.columns: return {"error":"SPY ?놁쓬"}
        spy_ret=(data["SPY"].pct_change().rolling(63).mean()*0.5+data["SPY"].pct_change().rolling(126).mean()*0.3+data["SPY"].pct_change().rolling(252).mean()*0.2).iloc[-1]
        rs_dict={}
        for t in data.columns:
            if t=="SPY": continue
            mom=(data[t].pct_change().rolling(63).mean()*0.5+data[t].pct_change().rolling(126).mean()*0.3+data[t].pct_change().rolling(252).mean()*0.2).iloc[-1]
            if spy_ret!=0: rs_dict[t]=float(mom/spy_ret)
        top=sorted(rs_dict,key=rs_dict.get,reverse=True)[:top_n]
        return {"top":[{"ticker":t,"rs":round(rs_dict[t],2)} for t in top]}
    except Exception as e: return {"error":str(e)}

def get_us_sector_rs():
    try:
        tks=list(US_SECTOR_ETFS.keys())
        raw=yf.download(tks+["SPY"],period="2y",auto_adjust=True,progress=False)
        if raw.empty: return {"error":"?곗씠???놁쓬"}
        data=raw["Close"].ffill().dropna(axis=1,how="any")
        if "SPY" not in data.columns: return {"error":"SPY ?놁쓬"}
        results=[]
        for t in tks:
            if t not in data.columns: continue
            etf=data[t]; spy=data["SPY"]; rel=etf/spy
            rs_vals=[]
            for win in [60,120,250]:
                if len(rel)<win: continue
                ma=rel.rolling(win).mean(); rs=((rel/ma)-1)*100
                if not rs.dropna().empty: rs_vals.append(float(rs.dropna().iloc[-1]))
            rs_raw=round(np.mean(rs_vals),2) if rs_vals else 0
            norm_rs=round(100*(1/(1+np.exp(-rs_raw/12))),1)
            mom3=float(etf.pct_change().rolling(63).mean().iloc[-1]) if len(etf)>=63 else 0
            vol3=float(etf.pct_change().rolling(63).std().iloc[-1]) if len(etf)>=63 else 1
            risk_adj=round((mom3/vol3)*100,2) if vol3>0 else 0
            results.append({"ticker":t,"name":US_SECTOR_ETFS.get(t,t),"norm_rs":norm_rs,"rs_raw":rs_raw,"risk_adj":risk_adj})
        results.sort(key=lambda x:x["norm_rs"],reverse=True)
        return {"sectors":results}
    except Exception as e: return {"error":str(e)}

# ?????????????????????????????????????????????????????????????????????????????
# 援?궡 吏???⑥닔
# ?????????????????????????????????????????????????????????????????????????????
def get_market_summary():
    try:
        start=_td_back(5); end=datetime.today().strftime("%Y-%m-%d")
        kp=_close("^KS11",start=start,end=end); kq=_close("^KQ11",start=start,end=end)
        if kp.empty or kq.empty: return {"error":"?곗씠???놁쓬"}
        kp_l=float(kp.iloc[-1]); kp_p=float(kp.iloc[-2]) if len(kp)>=2 else kp_l
        kq_l=float(kq.iloc[-1]); kq_p=float(kq.iloc[-2]) if len(kq)>=2 else kq_l
        return {"date":kp.index[-1].strftime("%Y-%m-%d"),
                "kospi":{"close":round(kp_l,2),"chg_pct":round((kp_l/kp_p-1)*100,2),"week_pct":round((kp_l/float(kp.iloc[0])-1)*100,2)},
                "kosdaq":{"close":round(kq_l,2),"chg_pct":round((kq_l/kq_p-1)*100,2),"week_pct":round((kq_l/float(kq.iloc[0])-1)*100,2)}}
    except Exception as e: return {"error":str(e)}

def get_sector_performance():
    try:
        r=requests.get(NAVER_ETF_URL,headers=HEADERS,timeout=15); etfs=r.json()["result"]["etfItemList"]
        em={e["itemcode"]:e for e in etfs}; sd={}
        for code,name in SECTOR_ETFS.items():
            if code in em: sd[name]={"chg_pct":round(em[code]["changeRate"],2)}
        if not sd: return {"error":"?뱁꽣 ?곗씠???놁쓬"}
        ss=sorted(sd.items(),key=lambda x:x[1]["chg_pct"],reverse=True)
        return {"sectors":sd,"top3":[(n,d["chg_pct"]) for n,d in ss[:3]],"bot3":[(n,d["chg_pct"]) for n,d in ss[-3:]]}
    except Exception as e: return {"error":str(e)}

def get_supply_oscillator():
    try:
        start=_td_back(25); end=datetime.today().strftime("%Y-%m-%d")
        kp=_close("^KS11",start=start,end=end)
        if kp.empty or len(kp)<5: return {"error":"?곗씠??遺議?}
        ma5=kp.rolling(5).mean().iloc[-1]; ma20=kp.rolling(20).mean().iloc[-1] if len(kp)>=20 else kp.mean()
        kp_osc=(ma5/ma20-1)*100; results={"kospi_osc":round(kp_osc,2),"sectors":{}}
        for code,name in SECTOR_ETFS.items():
            try:
                cl=_close(f"{code}.KS",start=start,end=end)
                if cl.empty or len(cl)<5: continue
                m5=cl.rolling(5).mean().iloc[-1]; m20=cl.rolling(20).mean().iloc[-1] if len(cl)>=20 else cl.mean()
                results["sectors"][name]={"rel_osc":round((m5/m20-1)*100-kp_osc,2)}
            except: continue
        if results["sectors"]:
            ss=sorted(results["sectors"].items(),key=lambda x:x[1]["rel_osc"],reverse=True)
            results["strong"]=[(n,d["rel_osc"]) for n,d in ss[:3]]; results["weak"]=[(n,d["rel_osc"]) for n,d in ss[-3:]]
        return results
    except Exception as e: return {"error":str(e)}

def get_binzip_stocks(supply_data=None, top_n=5):
    try:
        lead=[n for n,_ in supply_data["strong"]][:2] if supply_data and "error" not in supply_data and supply_data.get("strong") else ["諛섎룄泥?,"?뺣낫湲곗닠"]
        stocks=[]; seen=set()
        for sn in lead:
            for code,name in _FALLBACK.get(sn,[]):
                if code not in seen: seen.add(code); stocks.append({"code":code,"name":name})
        if not stocks: return {"error":"醫낅ぉ ?놁쓬","binzip":[],"sectors":lead}
        start=_td_back(65); end=datetime.today().strftime("%Y-%m-%d")
        kp_rs60=kp_rs20=0.0
        try:
            kp=_close("^KS11",start=start,end=end)
            if len(kp)>=21: kp_rs60=(kp.iloc[-1]/kp.iloc[0]-1)*100; kp_rs20=(kp.iloc[-1]/kp.iloc[-21]-1)*100
        except: pass
        cands=[]
        for s in stocks:
            try:
                cl=_close(f"{s['code']}.KS",start=start,end=end)
                if cl.empty or len(cl)<20: continue
                n=len(cl); now=float(cl.iloc[-1]); ma60=float(cl.rolling(min(60,n)).mean().iloc[-1])
                rs60=(now/float(cl.iloc[0])-1)*100-kp_rs60
                rel20=((now/float(cl.iloc[-21])-1)*100 if n>=21 else 0.0)-kp_rs20
                if -20.0<rel20<-2.0 and now>ma60*0.93 and rs60>5.0:
                    cands.append({"name":s["name"],"code":s["code"],"rs60":round(rs60,1),"rel20":round(rel20,1),"price":int(now)})
            except: continue
        cands.sort(key=lambda x:x["rs60"],reverse=True)
        return {"binzip":cands[:top_n],"scanned":len(stocks),"sectors":lead}
    except Exception as e: return {"error":str(e),"binzip":[]}

def get_kr_etf_rs():
    try:
        tks=list(KOREA_ETFS.keys())
        raw=yf.download(tks+["^KS11"],period="2y",auto_adjust=True,progress=False)
        if raw.empty: return {"error":"?곗씠???놁쓬"}
        data=raw["Close"] if isinstance(raw.columns,pd.MultiIndex) else raw
        if "^KS11" not in data.columns: return {"error":"KOSPI ?놁쓬"}
        kospi=data["^KS11"].dropna(); results=[]
        for t in tks:
            if t not in data.columns: continue
            etf=data[t].dropna(); common=etf.index.intersection(kospi.index)
            if len(common)<60: continue
            rel=etf.loc[common]/kospi.loc[common]
            rs_vals=[]
            for win in [60,120,250]:
                if len(rel)<win: continue
                ma=rel.rolling(win).mean(); rs=((rel/ma)-1)*100
                if not rs.dropna().empty: rs_vals.append(float(rs.dropna().iloc[-1]))
            rs_raw=round(np.mean(rs_vals),2) if rs_vals else 0
            norm_rs=round(100*(1/(1+np.exp(-rs_raw/12))),1)
            mom3=float(etf.pct_change().rolling(63).mean().iloc[-1]) if len(etf)>=63 else 0
            vol3=float(etf.pct_change().rolling(63).std().iloc[-1]) if len(etf)>=63 else 1
            results.append({"ticker":t,"name":KOREA_ETFS.get(t,t),"norm_rs":norm_rs,"risk_adj":round((mom3/vol3)*100,2) if vol3>0 else 0})
        results.sort(key=lambda x:x["norm_rs"],reverse=True)
        strong=[r for r in results if r["norm_rs"]>=70]
        return {"all":results,"strong":strong[:10] if strong else results[:5]}
    except Exception as e: return {"error":str(e)}

def get_buy_timing(ticker):
    try:
        t=ticker.strip().upper()
        raw=yf.download(t,period="1y",auto_adjust=False,progress=False)
        if raw.empty: return {"error":f"{t} ?곗씠???놁쓬"}
        df=raw.copy()
        if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.droplevel(1)
        df['?媛醫낃?愿대━']=(df['Low']-df['Close'].shift(1))/df['Close'].shift(1)*100
        df['怨좉?醫낃??섎씫']=(df['Close']-df['High'])/df['High']*100
        df['?쒓??媛愿대━']=(df['Low']-df['Open'])/df['Open']*100
        df['?꾩씪醫낃?怨좉?']=(df['High']-df['Close'].shift(1))/df['Close'].shift(1)*100
        df['?쒓?怨좉?愿대━']=(df['High']-df['Open'])/df['Open']*100
        info=yf.Ticker(t).info; name=info.get('longName') or info.get('shortName',t)
        return {"ticker":t,"name":name,"price":round(float(df['Close'].iloc[-1]),2),
                "怨좉?醫낃??섎씫":round(float(df['怨좉?醫낃??섎씫'].mean()),2),
                "?媛醫낃?愿대━":round(float(df['?媛醫낃?愿대━'].dropna().mean()),2),
                "?쒓??媛愿대━":round(float(df['?쒓??媛愿대━'].mean()),2),
                "?꾩씪醫낃?怨좉?":round(float(df['?꾩씪醫낃?怨좉?'].dropna().mean()),2),
                "?쒓?怨좉?愿대━":round(float(df['?쒓?怨좉?愿대━'].mean()),2)}
    except Exception as e: return {"error":str(e)}

# ?????????????????????????????????????????????????????????????????????????????
# ??1: 誘멸뎅 吏??# ?????????????????????????????????????????????????????????????????????????????
with tab1:
    st.subheader("?뙉 誘멸뎅 ?쒖옣 醫낇빀 吏??)
    st.caption("誘멸뎅 ??留덇컧 ??(?쒓뎅 ?ㅼ쟾 6~7?? ?ㅽ뻾 沅뚯옣")

    if st.button("??誘멸뎅 吏???꾩껜 ?ㅽ뻾", type="primary", use_container_width=True, key="us_run"):
        prog = st.progress(0, text="?맍 移대굹由ъ븘 遺꾩꽍 以?..")
        canary=get_canary_signal(); prog.progress(10, text="?뵦 BOFA Heat 遺꾩꽍 以?..")
        bofa=get_bofa_heat(); prog.progress(20, text="?㈇ 釉붾윭???몃뵒耳?댄꽣...")
        blood=get_blood_indicator(); prog.progress(30, text="?삺 ?쇱뼱?ㅺ렇由щ뱶 (?쇨컙)...")
        fg=get_us_fear_greed(); prog.progress(42, text="?뱟 ?쇱뼱?ㅺ렇由щ뱶 (?붽컙)...")
        fg_m=get_monthly_fear_greed(); prog.progress(52, text="?뱤 肄뷀룷??吏??..")
        coppock=get_coppock(); coppock_fast=get_coppock_fast(); prog.progress(62, text="?뱻 ZBT ?쒖옣 ??..")
        zbt=get_zbt(); prog.progress(72, text="?룇 S&P500 RS ?곸쐞...")
        sp500_rs=get_sp500_rs(SP500_TOP100); prog.progress(84, text="?룇 ?섏뒪??00 RS...")
        ndx_rs=get_nasdaq100_rs(NASDAQ100); prog.progress(93, text="?룺 誘멸뎅 ?뱁꽣 ETF RS...")
        us_sector=get_us_sector_rs(); prog.progress(100, text="???꾨즺!")
        prog.empty(); st.success("??遺꾩꽍 ?꾨즺!")

        # ?? 醫낇빀 ?좏샇 移대뱶 ??
        st.markdown("### ?슗 醫낇빀 ?좏샇")
        c1,c2,c3,c4 = st.columns(4)
        if canary and "error" not in canary:
            e="?윟" if canary["mode"]=="怨듦꺽" else "?뵶"
            c1.metric("?맍 移대굹由ъ븘 紐⑤뱶",f"{e} {canary['mode']}",f"QQQ {canary['qqq_mom']*100:+.1f}%")
        if bofa and "error" not in bofa:
            c2.metric("?뵦 BOFA Heat",f"{bofa['heat']}/10",_heat_label(bofa['heat']))
        if blood and "error" not in blood:
            c3.metric("?㈇ 釉붾윭??,blood["value"],f"60MA {blood['vs_ma60']}")
        if zbt and "error" not in zbt:
            c4.metric("?뱻 ZBT",f"{zbt['zbt']:.3f}","?윟 諛섎벑?좏샇" if zbt.get("signal") else "???湲곗쨷")

        st.divider()

        # ?? BOFA ?몃? ??
        if bofa and "error" not in bofa:
            with st.expander("?뵦 BOFA Heat ?몃? 蹂닿린", expanded=False):
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Heat Score",f"{bofa['heat']}/10",_heat_label(bofa['heat']))
                c2.metric("?곸듅異붿꽭","??Yes" if bofa['trend_on'] else "??No")
                c3.metric("VIX ?쇳겕","?좑툘 ON" if bofa['shock_vix'] else "???뺤긽")
                c4.metric("?щ젅???쇳겕","?좑툘 ON" if bofa['shock_credit'] else "???뺤긽")
                st.caption("Heat ??.5: 怨쇱뿴 ?뵶 / ??: 二쇱쓽 ?윝 / ??.5: 蹂댄넻 ?윞 / <2.5: ?덉쟾 ?윟")

        # ?? ZBT ?몃? ??
        if zbt and "error" not in zbt:
            with st.expander("?뱻 ZBT (Zweig Breadth Thrust) ?몃? 蹂닿린", expanded=False):
                c1,c2,c3 = st.columns(3)
                c1.metric("ZBT ?꾩옱",f"{zbt['zbt']:.3f}","?윟 ?좏샇諛쒖깮!" if zbt["signal"] else "??誘몃컻??)
                c2.metric("理쒓렐 理쒖?",f"{zbt['prev_min']:.3f}","<40% ?꾩슂")
                c3.metric("VIX",str(zbt.get('vix','N/A')),"???덉젙" if zbt.get('vix_ok') else "?좑툘 ?믪쓬")
                st.caption("ZBT: 理쒓렐10?쇱턀?<40% ???꾩옱>61.5% ?뚰뙆 ?????諛섎벑 ?좏샇 (?섏뒪??00+S&P500 醫낅ぉ 湲곗?)")

        st.divider()

        # ?? ?쇱뼱?ㅺ렇由щ뱶 ??
        st.markdown("### ?삺 ?쇱뼱?ㅺ렇由щ뱶 ?ㅼ떎?덉씠??)
        c_d, c_m = st.columns(2)
        with c_d:
            st.markdown("**?뱟 ?쇨컙 (?④린 異붿꽭)**")
            if fg and "error" not in fg:
                c1,c2=st.columns(2)
                c1.metric("S&P500 Osc",fg["spx_osc"],f"{'?윟' if fg['spx_osc']>0 else '?뵶'} {fg['spx_sentiment']}")
                c2.metric("NASDAQ Osc",fg["ndx_osc"],f"{'?윟' if fg['ndx_osc']>0 else '?뵶'} {fg['ndx_sentiment']}")
                c1.metric("SPY ?꾪럡??,fg["spy_impulse"]); c2.metric("QQQ ?꾪럡??,fg["qqq_impulse"])
                c1.metric("SPY SuperMA ?닿꺽",f"{fg['spy_gap']:+.2f}%"); c2.metric("QQQ SuperMA ?닿꺽",f"{fg['qqq_gap']:+.2f}%")
                c1.metric("SPY TD 留ㅻ룄/留ㅼ닔",f"{fg['spy_td_sell']} / {fg['spy_td_buy']}")
                c2.metric("QQQ TD 留ㅻ룄/留ㅼ닔",f"{fg['qqq_td_sell']} / {fg['qqq_td_buy']}")
            else: st.error("?쇨컙 F&G ?ㅻ쪟")
        with c_m:
            st.markdown("**?뱠 ?붽컙 (以묒옣湲?異붿꽭)**")
            if fg_m and "error" not in fg_m:
                c1,c2=st.columns(2)
                c1.metric("S&P500 ?붽컙",fg_m["spx_osc"],f"{'?윟' if fg_m['spx_osc']>0 else '?뵶'} {fg_m['spx_sentiment']}")
                c2.metric("NASDAQ ?붽컙",fg_m["ndx_osc"],f"{'?윟' if fg_m['ndx_osc']>0 else '?뵶'} {fg_m['ndx_sentiment']}")
                c1.metric("湲곗???,fg_m["date"]); c2.metric("S&P500 FGI",fg_m["spx_fgi"])
            else: st.error("?붽컙 F&G ?ㅻ쪟")

        st.divider()

        # ?? 肄뷀룷????
        st.markdown("### ?뱤 肄뷀룷??吏??)
        c_std, c_fast = st.columns(2)
        with c_std:
            st.markdown("**?쒖? (11/14媛쒖썡 ROC, 10媛쒖썡 EMA)**")
            if coppock and "error" not in coppock:
                cols=st.columns(len(coppock))
                for i,(lbl,v) in enumerate(coppock.items()):
                    e="?윟" if v["pos"] else "?뵶"; arr="?? if v["trend"]=="?곸듅" else "??
                    cols[i].metric(lbl,f"{e} {v['value']}",f"{arr} {v['trend']}")
        with c_fast:
            st.markdown("**鍮좊Ⅸ 踰꾩쟾 (4/6媛쒖썡 ROC, 3媛쒖썡 MA)**")
            if coppock_fast and "error" not in coppock_fast:
                cols=st.columns(len(coppock_fast))
                for i,(lbl,v) in enumerate(coppock_fast.items()):
                    e="?윟" if v["pos"] else "?뵶"; arr="?? if v["trend"]=="?곸듅" else "??
                    cols[i].metric(lbl,f"{e} {v['value']}",f"{arr} {v['trend']}")

        st.divider()

        # ?? RS ?곸쐞 醫낅ぉ ??
        st.markdown("### ?룇 ?곷?媛뺣룄(RS) ?곸쐞 醫낅ぉ")
        c1,c2=st.columns(2)
        with c1:
            st.markdown("**?뱢 S&P500 Top 10**")
            if sp500_rs and "error" not in sp500_rs:
                for i,item in enumerate(sp500_rs["top"],1):
                    t=item["ticker"]; kr=TICKER_NAMES.get(t,""); rs=item.get("rs",0)
                    st.markdown(f"`{i:2d}` **{t}** {kr} &nbsp; {_cv(rs,'1f')}", unsafe_allow_html=True)
            elif sp500_rs: st.error(sp500_rs.get("error"))
        with c2:
            st.markdown("**?뱢 ?섏뒪??00 Top 10**")
            if ndx_rs and "error" not in ndx_rs:
                for i,item in enumerate(ndx_rs["top"],1):
                    t=item["ticker"]; kr=TICKER_NAMES.get(t,""); rs=item.get("rs",0)
                    st.markdown(f"`{i:2d}` **{t}** {kr} &nbsp; {_cv(rs,'2f')}", unsafe_allow_html=True)
            elif ndx_rs: st.error(ndx_rs.get("error"))

        st.divider()

        # ?? ?뱁꽣 ETF RS ??
        st.markdown("### ?룺 誘멸뎅 ?뱁꽣 ETF ?곷?媛뺣룄")
        if us_sector and "error" not in us_sector:
            df_s=pd.DataFrame(us_sector["sectors"])
            df_s["媛뺣룄"]=df_s["norm_rs"].apply(lambda x:"?윟 媛뺤꽭" if x>=70 else ("?윞 以묐┰" if x>=50 else "?뵶 ?쎌꽭"))
            df_s.columns=[c if c!="ticker" else "?곗빱" for c in df_s.columns]
            st.dataframe(df_s.rename(columns={"ticker":"?곗빱","name":"?뱁꽣","norm_rs":"RS(0~100)","rs_raw":"RS?먯떆","risk_adj":"蹂?숈꽦議곗젙紐⑤찘?"}),
                use_container_width=True, hide_index=True,
                column_config={"RS(0~100)":st.column_config.ProgressColumn("RS(0~100)",min_value=0,max_value=100,format="%.1f")})
        elif us_sector: st.error(us_sector.get("error"))

# ?????????????????????????????????????????????????????????????????????????????
# ??2: 援?궡 吏??# ?????????????????????????????????????????????????????????????????????????????
with tab2:
    st.subheader("?눖?눟 援?궡 ?쒖옣 吏??)
    st.caption("??留덇컧 ??(?ㅽ썑 4???댄썑) ?ㅽ뻾 沅뚯옣")

    if st.button("??援?궡 吏???꾩껜 ?ㅽ뻾", type="primary", use_container_width=True, key="kr_run"):
        prog=st.progress(0, text="?뱤 肄붿뒪??肄붿뒪???섏쭛 以?..")
        market=get_market_summary(); prog.progress(20, text="?룺 ?낆쥌 ETF ?섏쭛 以?..")
        sector=get_sector_performance(); prog.progress(40, text="?뮰 ?섍툒 ?ㅼ떎?덉씠??怨꾩궛 以?..")
        supply=get_supply_oscillator(); prog.progress(60, text="?눖?눟 ?쒓뎅 ETF RS 怨꾩궛 以?..")
        kr_etf=get_kr_etf_rs(); prog.progress(80, text="?룧 鍮덉쭛 ?ㅽ겕由щ떇 以?..")
        binzip=get_binzip_stocks(supply_data=supply); prog.progress(100, text="???꾨즺!")
        prog.empty(); st.success("??遺꾩꽍 ?꾨즺!")

        # ?? 吏???꾪솴 ??
        st.markdown("### ?뱤 吏???꾪솴")
        if market and "error" not in market:
            kp=market["kospi"]; kq=market["kosdaq"]
            c1,c2=st.columns(2)
            c1.metric(f"肄붿뒪??({market['date']})", f"{kp['close']:,.2f}",
                      f"{kp['chg_pct']:+.2f}% 쨌 二쇨컙 {kp['week_pct']:+.2f}%",
                      delta_color="normal")
            c2.metric("肄붿뒪??, f"{kq['close']:,.2f}",
                      f"{kq['chg_pct']:+.2f}% 쨌 二쇨컙 {kq['week_pct']:+.2f}%",
                      delta_color="normal")
        elif market: st.error(market.get("error"))

        st.divider()

        # ?? ?낆쥌 媛뺤꽭/?쎌꽭 ??
        st.markdown("### ?룺 ?낆쥌 媛뺤꽭 / ?쎌꽭")
        if sector and "error" not in sector:
            c1,c2=st.columns(2)
            with c1:
                st.markdown("**媛뺤꽭 TOP3**")
                for name,chg in sector.get("top3",[]):
                    st.markdown(f"{name}: {_cv(chg)}%", unsafe_allow_html=True)
            with c2:
                st.markdown("**?쎌꽭 BOT3**")
                for name,chg in sector.get("bot3",[]):
                    st.markdown(f"{name}: {_cv(chg)}%", unsafe_allow_html=True)
        elif sector: st.error(sector.get("error"))

        st.divider()

        # ?? ?섍툒 ?ㅼ떎?덉씠????
        st.markdown("### ?뮰 ?섍툒 ?ㅼ떎?덉씠??)
        if supply and "error" not in supply:
            osc=supply["kospi_osc"]
            st.metric(f"{'?윟' if osc>0 else '?뵶'} 肄붿뒪??湲곗? ?ㅼ떎?덉씠??, f"{osc:+.2f}")
            c1,c2=st.columns(2)
            with c1:
                st.markdown("**?섍툒 媛뺤꽭**")
                for name,rel in supply.get("strong",[]): st.markdown(f"{name}: {_cv(rel)}", unsafe_allow_html=True)
            with c2:
                st.markdown("**?섍툒 ?쎌꽭**")
                for name,rel in supply.get("weak",[]): st.markdown(f"{name}: {_cv(rel)}", unsafe_allow_html=True)
        elif supply: st.error(supply.get("error"))

        st.divider()

        # ?? ?쒓뎅 ETF RS ??
        st.markdown("### ?눖?눟 ?쒓뎅 ?곗뾽/?뚮쭏 ETF ?곷?媛뺣룄 (vs KOSPI)")
        if kr_etf and "error" not in kr_etf:
            show=kr_etf.get("strong") or kr_etf.get("all",[])[:10]
            if show:
                df_kr=pd.DataFrame(show)
                df_kr["媛뺣룄"]=df_kr["norm_rs"].apply(lambda x:"?윟 媛뺤꽭" if x>=70 else "?윞 蹂댄넻")
                st.dataframe(df_kr.rename(columns={"name":"ETF紐?,"norm_rs":"RS(0~100)","risk_adj":"蹂?숈꽦議곗젙紐⑤찘?","媛뺣룄":"媛뺣룄"})[["ETF紐?,"RS(0~100)","蹂?숈꽦議곗젙紐⑤찘?","媛뺣룄"]],
                    use_container_width=True, hide_index=True,
                    column_config={"RS(0~100)":st.column_config.ProgressColumn("RS(0~100)",min_value=0,max_value=100,format="%.1f")})
        elif kr_etf: st.error(kr_etf.get("error"))

        st.divider()

        # ?? 鍮덉쭛 二쇰룄二???
        bz=binzip or {}; bl=bz.get("binzip",[]); ss=" + ".join(bz.get("sectors",[])) or "二쇰룄?낆쥌"
        st.markdown(f"### ?룧 ?섍툒 鍮덉쭛 二쇰룄二?[{ss}]")
        if bl:
            df_bz=pd.DataFrame(bl)[["name","code","price","rs60","rel20"]]
            df_bz.columns=["醫낅ぉ紐?,"肄붾뱶","?꾩옱媛","60?퍻S(%)","20?쇰닃由?%)"]
            df_bz["?꾩옱媛"]=df_bz["?꾩옱媛"].apply(lambda x:f"{x:,}??)
            df_bz["60?퍻S(%)"]=df_bz["60?퍻S(%)"].apply(lambda x:f"+{x:.1f}%")
            df_bz["20?쇰닃由?%)"]=df_bz["20?쇰닃由?%)"].apply(lambda x:f"{x:.1f}%")
            st.dataframe(df_bz, use_container_width=True, hide_index=True)
            st.caption(f"{bz.get('scanned',0)}醫낅ぉ ?ㅼ틪 / 鍮덉쭛 {len(bl)}媛?)
        elif "error" in bz: st.error(f"?ㅻ쪟: {bz['error']}")
        else: st.info(f"鍮덉쭛 議곌굔 異⑹” 醫낅ぉ ?놁쓬 ({bz.get('scanned',0)}醫낅ぉ ?ㅼ틪)")

# ?????????????????????????????????????????????????????????????????????????????
# ??3: 醫낅ぉ 遺꾩꽍
# ?????????????????????????????????????????????????????????????????????????????
with tab3:
    st.subheader("?뵇 媛쒕퀎 醫낅ぉ 遺꾩꽍")

    st.markdown("### ?렞 留ㅼ닔 ????듦퀎 遺꾩꽍")
    st.caption("1???곗씠??湲곕컲 ??怨좉?/?媛/?쒓? ?됯퇏 愿대━??(吏?뺢? 留ㅼ닔 李멸퀬??")

    _sel_opts = [""] + sorted([f"{kr} ({t})" for t, kr in TICKER_NAMES.items()], key=lambda x: x[0])
    sel_stock = st.selectbox("?뱥 紐⑸줉?먯꽌 ?좏깮 (?쒓?紐??먮뒗 ?곷Ц ?곗빱濡?寃??媛??", _sel_opts, index=0, key="bt_sel")
    ticker_input = st.text_input("?먮뒗 吏곸젒 ?낅젰 (?곗빱쨌?쒓?紐?紐⑤몢 媛?? ?쇳몴濡??щ윭 媛?", placeholder="NVDA, ?붾퉬?붿븘, 005930.KS", key="bt_ticker")

    if st.button("?뵇 遺꾩꽍 ?쒖옉", key="bt_run", type="primary", use_container_width=True):
        tickers_to_run = []
        if sel_stock:
            _t = sel_stock.split("(")[-1].rstrip(")")
            tickers_to_run.append(_t.strip())
        if ticker_input:
            for raw in [x.strip() for x in ticker_input.split(",") if x.strip()]:
                resolved = NAME_TO_TICKER.get(raw.lower(), raw)
                tickers_to_run.append(resolved)
        if not tickers_to_run:
            st.warning("醫낅ぉ???좏깮?섍굅???낅젰?댁＜?몄슂.")
        for t in tickers_to_run:
            with st.spinner(f"{t} 遺꾩꽍 以?.."):
                res = get_buy_timing(t)
            if "error" in res:
                st.error(f"??{t}: {res['error']}")
            else:
                kr = TICKER_NAMES.get(t, res.get("name", ""))
                st.markdown(f"#### ?뱦 **{t}** {kr} ???꾩옱媛: `{res['price']:,}`")
                c1,c2 = st.columns(2)
                c1.metric("怨좉??믪쥌媛 ?됯퇏 ?섎씫", f"{res['怨좉?醫낃??섎씫']:+.2f}%", help="?μ쨷 怨좎젏 ?鍮?醫낃? ?됯퇏 ?숉룺. 吏?뺢?蹂대떎 ?믨쾶 ?щ씪媛붾떎媛 ?대젮?ㅻ뒗 ?뺣룄")
                c2.metric("?꾩씪醫낃??믩떦?쇱?媛 愿대━", f"{res['?媛醫낃?愿대━']:+.2f}%", help="?꾩씪 醫낃? ?鍮??뱀씪 ?媛 ?됯퇏 愿대━. 媛?븯???ы븿")
                c1.metric("?쒓??믪?媛 ?됯퇏 ?숉룺", f"{res['?쒓??媛愿대━']:+.2f}%", help="?쒓? 湲곗? ?μ쨷 理쒕? ?숉룺 ?됯퇏")
                c2.metric("?꾩씪醫낃??믩떦?쇨퀬媛", f"{res['?꾩씪醫낃?怨좉?']:+.2f}%", help="?꾩씪 醫낃? ?鍮??뱀씪 怨좉? ?됯퇏 愿대━??)
                c1.metric("?쒓??믩떦?쇨퀬媛 ?됯퇏", f"{res['?쒓?怨좉?愿대━']:+.2f}%", help="?쒓? ?鍮??μ쨷 怨좎젏源뚯? ?됯퇏 ?곸듅??)
                st.caption(f"?뮕 ?뚰듃: ?쒓? ?鍮??媛 愿대━({res['?쒓??媛愿대━']:+.2f}%) ???쒓?蹂대떎 洹??뺣룄 ??쾶 吏?뺢? ?ㅼ젙")
                st.divider()
