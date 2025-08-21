import os, time, math, datetime as dt
import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
import pandas_ta as ta
import requests
import feedparser

# ------------------ Seiteneinstellungen ------------------
st.set_page_config(page_title="Aktien-Radar", layout="wide")
st.title("📈 Aktien-Radar – Fundamentaldaten • Technik • News (Ampelsystem)")
st.caption("Weltweit • Kostenlos • Telegram-Push optional • Empfehlung: Kaufen / Halten / Verkaufen")

# ------------------ Secrets / Umgebungsvariablen ------------------
TELEGRAM_BOT_TOKEN = st.secrets.get("TELEGRAM_BOT_TOKEN", os.getenv("TELEGRAM_BOT_TOKEN", ""))
TELEGRAM_CHAT_ID   = st.secrets.get("TELEGRAM_CHAT_ID", os.getenv("TELEGRAM_CHAT_ID", ""))

# ------------------ Ampel-Helfer ------------------
def ampel(color, text=""):
    colors = {"green":"#16a34a","yellow":"#eab308","red":"#dc2626","grey":"#9ca3af"}
    c = colors.get(color, "#9ca3af")
    label = f"&nbsp;{text}" if text else ""
    return f"""
    <span style="display:inline-flex;align-items:center">
      <span style="width:10px;height:10px;border-radius:9999px;background:{c};display:inline-block;margin-right:6px;border:1px solid rgba(0,0,0,.05)"></span>
      <span>{text}</span>
    </span>
    """

# ------------------ Bewertungsregeln (evidenzbasiert, robust) ------------------
def grade_pe(pe):
    if not pe or pe<=0 or math.isnan(pe): return "grey"
    if pe<=15: return "green"
    if pe<=25: return "yellow"
    return "red"

def grade_pb(pb):
    if not pb or pb<=0 or math.isnan(pb): return "grey"
    if pb<=2.5: return "green"
    if pb<=4.0: return "yellow"
    return "red"

def grade_ev_ebitda(x):
    if not x or x<=0 or math.isnan(x): return "grey"
    if x<=10: return "green"
    if x<=14: return "yellow"
    return "red"

def grade_roe(roe):
    if roe is None or (isinstance(roe,float) and math.isnan(roe)): return "grey"
    roe = roe*100 if roe<2 else roe
    if roe>=15: return "green"
    if roe>=8: return "yellow"
    return "red"

def grade_margin(m):
    if m is None or (isinstance(m,float) and math.isnan(m)): return "grey"
    m = m*100 if m<2 else m
    if m>=10: return "green"
    if m>=5: return "yellow"
    return "red"

def grade_growth(g):
    if g is None or (isinstance(g,float) and math.isnan(g)): return "grey"
    g = g*100 if g<2 else g
    if g>=10: return "green"
    if g>=0: return "yellow"
    return "red"

def grade_nd_ebitda(x):
    if x is None or (isinstance(x,float) and math.isnan(x)): return "grey"
    if x<=0: return "green"  # Net Cash
    if x<=2.0: return "green"
    if x<=3.0: return "yellow"
    return "red"

def grade_fcf_yield(x):
    if x is None or (isinstance(x,float) and math.isnan(x)): return "grey"
    x = x*100 if abs(x)<2 else x
    if x>=5: return "green"
    if x>=2: return "yellow"
    return "red"

def grade_beta(b):
    if b is None or (isinstance(b,float) and math.isnan(b)): return "grey"
    if b<=1.2: return "green"
    if b<=1.6: return "yellow"
    return "red"

def grade_trend_sma(sma50, sma200):
    if not sma50 or not sma200 or sma200==0: return "grey"
    if sma50 > sma200: return "green"
    if abs(sma50 - sma200)/sma200 <= 0.01: return "yellow"
    return "red"

def grade_price_vs_200d(price, sma200):
    if not price or not sma200 or sma200==0: return "grey"
    diff = (price - sma200)/sma200 * 100
    if diff >= 0: return "green"
    if -1 <= diff < 0: return "yellow"
    return "red"

def grade_mom_12_1(ret):
    if ret is None or (isinstance(ret,float) and math.isnan(ret)): return "grey"
    ret = ret*100 if abs(ret)<2 else ret
    if ret > 10: return "green"
    if ret >= -5: return "yellow"
    return "red"

def grade_rsi(r):
    if r is None or (isinstance(r,float) and math.isnan(r)): return "grey"
    if 40 <= r <= 70: return "green"
    if (30 <= r < 40) or (70 < r <= 80): return "yellow"
    return "red"

def grade_52w_near(pct_from_high):
    if pct_from_high is None or (isinstance(pct_from_high,float) and math.isnan(pct_from_high)): return "grey"
    pct = abs(pct_from_high)*100 if abs(pct_from_high)<2 else pct_from_high
    if pct <= 5: return "green"
    if pct <= 15: return "yellow"
    return "red"

def grade_news_sentiment(s):
    if s is None or (isinstance(s,float) and math.isnan(s)): return "grey"
    if s >= 0.2: return "green"
    if s > -0.2: return "yellow"
    return "red"

WEIGHTS = dict(value=0.40, quality=0.20, growth=0.10, technical=0.20, news=0.10)

# ------------------ Daten-Lader ------------------
@st.cache_data(ttl=3600)
def load_universe():
    try:
        df = pd.read_csv("universe.csv").dropna()
        return sorted(list(set(df["symbol"].astype(str).str.strip())))
    except Exception:
        return ["AAPL","MSFT","NVDA","AMZN","META","GOOGL"]

@st.cache_data(ttl=3600)
def load_watchlist():
    if os.path.exists("watchlist.txt"):
        with open("watchlist.txt","r",encoding="utf-8") as f:
            return list(sorted(set([x.strip() for x in f if x.strip()])))
    return []

UNIVERSE = load_universe()
WATCHLIST = load_watchlist()

@st.cache_data(ttl=900)
def get_prices(tickers, period="2y", interval="1d"):
    if not tickers: return pd.DataFrame()
    return yf.download(tickers=tickers, period=period, interval=interval, auto_adjust=True, group_by='ticker', threads=True, progress=False)

@st.cache_data(ttl=86400)
def get_fundamentals(tickers):
    rows = []
    for t in tickers:
        info = {}
        try:
            info = yf.Ticker(t).get_info()
        except Exception:
            try:
                info = yf.Ticker(t).info
            except Exception:
                info = {}
        # FCF-Yield
        market_cap = info.get("marketCap")
        free_cf = info.get("freeCashflow")
        fcf_yield = (free_cf/market_cap) if (market_cap and free_cf) else None
        # Net Debt / EBITDA
        net_debt = info.get("netDebt")
        if net_debt is None and (info.get("totalDebt") is not None or info.get("totalCash") is not None):
            td = info.get("totalDebt") or 0
            cash = info.get("totalCash") or 0
            net_debt = td - cash
        ebitda = info.get("ebitda")
        nd_ebitda = (net_debt/ebitda) if (net_debt is not None and ebitda not in (None,0)) else (0 if (net_debt is not None and net_debt<=0 and ebitda) else None)

        rows.append(dict(
            symbol=t,
            longName=info.get("longName") or info.get("shortName"),
            sector=info.get("sector"),
            trailingPE=info.get("trailingPE"),
            forwardPE=info.get("forwardPE"),
            priceToBook=info.get("priceToBook"),
            enterpriseToEbitda=info.get("enterpriseToEbitda"),
            returnOnEquity=info.get("returnOnEquity"),
            profitMargins=info.get("profitMargins"),
            grossMargins=info.get("grossMargins"),
            revenueGrowth=info.get("revenueGrowth"),
            beta=info.get("beta"),
            netDebtToEbitda=nd_ebitda,
            fcfYield=fcf_yield
        ))
        time.sleep(0.05)
    return pd.DataFrame(rows)

def compute_technicals(price_df_for_symbol):
    close = price_df_for_symbol.get("Close", pd.Series(dtype=float)).dropna()
    if close.empty:
        return dict(price=None, sma50=None, sma200=None, rsi=None, mom_12_1=None, pct_from_52w_high=None)
    sma50 = close.rolling(50).mean().iloc[-1] if len(close)>=50 else None
    sma200 = close.rolling(200).mean().iloc[-1] if len(close)>=200 else None
    rsi = ta.rsi(close, length=14).iloc[-1] if len(close)>=14 else None
    # Momentum 12-1 (letzter Monat ausgeschlossen)
    try:
        monthly = close.resample("M").last()
        r_12 = monthly.iloc[-2]/monthly.iloc[-13] - 1.0 if len(monthly)>=13 else None
    except Exception:
        r_12 = None
    pct_from_high = None
    if len(close)>=252:
        high_52w = close[-252:].max()
        pct_from_high = (high_52w - close.iloc[-1]) / high_52w * 100.0
    return dict(price=close.iloc[-1], sma50=sma50, sma200=sma200, rsi=rsi, mom_12_1=r_12, pct_from_52w_high=pct_from_high)

# ------------------ News (kostenlos via Google News RSS) ------------------
def fetch_news_google_rss(query, limit=3):
    from urllib.parse import quote
    url = f"https://news.google.com/rss/search?q={quote(query)}&hl=de&gl=DE&ceid=DE:de"
    try:
        feed = feedparser.parse(url)
        items = []
        for e in feed.entries[:limit]:
            items.append(dict(title=e.title, url=e.link, published_at=getattr(e, "published", None), sentiment=None))
        return items
    except Exception:
        return []

def simple_headline_sentiment(title: str):
    if not title: return 0.0
    t = title.lower()
    pos_kw = ["beat", "beats", "above expectations", "upgrade", "raises guidance", "record", "surge", "strong growth", "wins"]
    neg_kw = ["miss", "misses", "below expectations", "downgrade", "profit warning", "lawsuit", "scandal", "fraud", "recall"]
    score = 0
    for w in pos_kw:
        if w in t: score += 1
    for w in neg_kw:
        if w in t: score -= 1
    return max(-1.0, min(1.0, score/3.0))  # -1..+1

# ------------------ Scoring ------------------
def score_row(f_row, t_row, news_items):
    pe = f_row.get("forwardPE") or f_row.get("trailingPE")
    pb = f_row.get("priceToBook")
    ev = f_row.get("enterpriseToEbitda")
    roe = f_row.get("returnOnEquity")
    pm = f_row.get("profitMargins")
    gm = f_row.get("grossMargins")
    rg = f_row.get("revenueGrowth")
    nd = f_row.get("netDebtToEbitda")
    fcfy = f_row.get("fcfYield")
    beta = f_row.get("beta")

    g_value = np.nanmean([
        {"green":1,"yellow":0,"red":-1,"grey":0}[grade_pe(pe)],
        {"green":1,"yellow":0,"red":-1,"grey":0}[grade_pb(pb)],
        {"green":1,"yellow":0,"red":-1,"grey":0}[grade_ev_ebitda(ev)],
        {"green":1,"yellow":0,"red":-1,"grey":0}[grade_fcf_yield(fcfy)]
    ])

    g_quality = np.nanmean([
        {"green":1,"yellow":0,"red":-1,"grey":0}[grade_roe(roe)],
        {"green":1,"yellow":0,"red":-1,"grey":0}[grade_margin(pm)],
        {"green":1,"yellow":0,"red":-1,"grey":0}[grade_nd_ebitda(nd)],
        {"green":1,"yellow":0,"red":-1,"grey":0}[grade_beta(beta)]
    ])

    g_growth = {"green":1,"yellow":0,"red":-1,"grey":0}[grade_growth(rg)]

    g_trend = np.nanmean([
        {"green":1,"yellow":0,"red":-1,"grey":0}[grade_trend_sma(t_row.get("sma50"), t_row.get("sma200"))],
        {"green":1,"yellow":0,"red":-1,"grey":0}[grade_price_vs_200d(t_row.get("price"), t_row.get("sma200"))],
        {"green":1,"yellow":0,"red":-1,"grey":0}[grade_mom_12_1(t_row.get("mom_12_1"))],
        {"green":1,"yellow":0,"red":-1,"grey":0}[grade_rsi(t_row.get("rsi"))],
        {"green":1,"yellow":0,"red":-1,"grey":0}[grade_52w_near(t_row.get("pct_from_52w_high"))]
    ])

    # News-Sentiment (einfach): Mittel der Regel-Keywords
    if news_items:
        sentiments = [simple_headline_sentiment(x["title"]) for x in news_items if x.get("title")]
        s_avg = float(np.mean(sentiments)) if sentiments else 0.0
    else:
        s_avg = 0.0
    g_news = {"green":1,"yellow":0,"red":-1,"grey":0}[grade_news_sentiment(s_avg)]

    total = (WEIGHTS["value"]*g_value + WEIGHTS["quality"]*g_quality +
             WEIGHTS["growth"]*g_growth + WEIGHTS["technical"]*g_trend +
             WEIGHTS["news"]*g_news)

    rec = "Kaufen" if total>=0.5 else ("Verkaufen" if total<=-0.5 else "Halten")

    return dict(g_value=g_value, g_quality=g_quality, g_growth=g_growth,
                g_technical=g_trend, g_news=g_news, total=total,
                recommendation=rec, news_sentiment=s_avg)

# ------------------ Sidebar ------------------
with st.sidebar:
    st.header("⚙️ Einstellungen")
    top_n = st.slider("Top-Ergebnisse (Ranking)", 10, 200, 50, step=10)
    custom_symbols = st.text_area("Eigene Symbole (kommasepariert)", "")
    auto_refresh = st.checkbox("Alle 5 Min. automatisch aktualisieren", value=True)
    test_push = st.button("🔔 Telegram-Test senden")

    if test_push:
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            try:
                requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                              json={"chat_id": TELEGRAM_CHAT_ID, "text":"Test: Hallo vom Aktien-Radar 👋"},
                              timeout=10)
                st.success("Test gesendet – schau in Telegram nach!")
            except Exception as e:
                st.error(f"Telegram-Fehler: {e}")
        else:
            st.error("Bitte TELEGRAM_BOT_TOKEN & TELEGRAM_CHAT_ID als Secrets setzen (siehe Anleitung).")

if auto_refresh:
    st.experimental_set_query_params(refresh=str(int(time.time())//300))

# ------------------ Universe erstellen ------------------
symbols = load_universe()
if custom_symbols.strip():
    extra = [s.strip() for s in custom_symbols.split(",") if s.strip()]
    symbols = sorted(list(set(symbols + extra)))

st.write(f"**Universe geladen:** {len(symbols)} Symbole")

# ------------------ Daten abrufen ------------------
prices = get_prices(symbols, period="2y", interval="1d")
fundamentals = get_fundamentals(symbols).set_index("symbol")

# Technische Kennzahlen
rows = []
for t in symbols:
    try:
        df_t = prices[t]
    except Exception:
        df_t = pd.DataFrame()
    tech = compute_technicals(df_t) if not df_t.empty else dict(price=None,sma50=None,sma200=None,rsi=None,mom_12_1=None,pct_from_52w_high=None)
    rows.append(pd.Series(tech, name=t))
technicals = pd.DataFrame(rows)

# News (kostenlos, 3 Schlagzeilen)
news_map = {}
for t in symbols:
    items = fetch_news_google_rss(f'"{t}" stock OR Aktie', limit=3)
    news_map[t] = items

# Scoring + Tabelle
records = []
for t in symbols:
    f_row = fundamentals.loc[t].to_dict() if t in fundamentals.index else {}
    t_row = technicals.loc[t].to_dict() if t in technicals.index else {}
    n_items = news_map.get(t, [])
    sc = score_row(f_row, t_row, n_items)
    records.append(dict(
        symbol=t, name=f_row.get("longName",""), sector=f_row.get("sector",""),
        price=t_row.get("price"),
        f_pe=f_row.get("forwardPE") or f_row.get("trailingPE"),
        f_pb=f_row.get("priceToBook"),
        f_ev_ebitda=f_row.get("enterpriseToEbitda"),
        f_roe=f_row.get("returnOnEquity"),
        f_gm=f_row.get("grossMargins"),
        f_nm=f_row.get("profitMargins"),
        f_rg=f_row.get("revenueGrowth"),
        f_ndebitda=f_row.get("netDebtToEbitda"),
        f_fcfy=f_row.get("fcfYield"),
        f_beta=f_row.get("beta"),
        t_sma50=t_row.get("sma50"), t_sma200=t_row.get("sma200"),
        t_rsi=t_row.get("rsi"), t_mom12_1=t_row.get("mom_12_1"),
        t_52w_gap=t_row.get("pct_from_52w_high"),
        news_sent=sc["news_sentiment"],
        score_total=sc["total"], recommendation=sc["recommendation"],
        news_items=n_items
    ))

df = pd.DataFrame(records).sort_values("score_total", ascending=False).head(top_n)

def fmt_num(x, pct=False, digits=2):
    if x is None or (isinstance(x,float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    if pct:
        v = x*100 if abs(x)<2 else x
        return f"{v:,.1f}%"
    return f"{x:,.{digits}f}"

display = pd.DataFrame({
    "Symbol": df["symbol"],
    "Name": df["name"].fillna(""),
    "Preis": [fmt_num(x) for x in df["price"]],
    "P/E": [ampel(grade_pe(x), fmt_num(x)) for x in df["f_pe"]],
    "P/B": [ampel(grade_pb(x), fmt_num(x)) for x in df["f_pb"]],
    "EV/EBITDA": [ampel(grade_ev_ebitda(x), fmt_num(x)) for x in df["f_ev_ebitda"]],
    "ROE": [ampel(grade_roe(x), fmt_num(x, pct=True)) for x in df["f_roe"]],
    "Gross %": [ampel(grade_margin(x), fmt_num(x, pct=True)) for x in df["f_gm"]],
    "Net %": [ampel(grade_margin(x), fmt_num(x, pct=True)) for x in df["f_nm"]],
    "Umsatz YoY": [ampel(grade_growth(x), fmt_num(x, pct=True)) for x in df["f_rg"]],
    "NetDebt/EBITDA": [ampel(grade_nd_ebitda(x), fmt_num(x)) for x in df["f_ndebitda"]],
    "FCF-Yield": [ampel(grade_fcf_yield(x), fmt_num(x, pct=True)) for x in df["f_fcfy"]],
    "Beta": [ampel(grade_beta(x), fmt_num(x)) for x in df["f_beta"]],
    "Trend 50/200": [ampel(grade_trend_sma(a,b)) for a,b in zip(df["t_sma50"], df["t_sma200"])],
    "Preis vs 200d": [ampel(grade_price_vs_200d(a,b)) for a,b in zip(df["price"], df["t_sma200"])],
    "Momentum 12-1": [ampel(grade_mom_12_1(x), fmt_num(x, pct=True)) for x in df["t_mom12_1"]],
    "RSI(14)": [ampel(grade_rsi(x), f"{x:.0f}" if pd.notna(x) else "—") for x in df["t_rsi"]],
    "52W-Hoch Dist.": [ampel(grade_52w_near(x), fmt_num(x, pct=True)) for x in df["t_52w_gap"]],
    "News-Sent.": [ampel(grade_news_sentiment(x), f"{x:+.2f}") for x in df["news_sent"]],
    "Gesamt": df["score_total"].map(lambda v: f"{v:+.2f}"),
    "Empfehlung": df["recommendation"]
})

st.markdown("### 🔎 Ranking & Ampeln (Top)")
st.write(display.to_html(escape=False, index=False), unsafe_allow_html=True)

st.markdown("### 📰 Neueste Nachrichten pro Titel")
for _, row in df.iterrows():
    st.markdown(f"**{row['symbol']} — {row['name'] or ''}**  | Empfehlung: **{row['recommendation']}**  | Gesamt: **{row['score_total']:+.2f}**", unsafe_allow_html=True)
    items = row["news_items"]
    if not items:
        st.write("Keine News gefunden.")
    else:
        for it in items:
            ts = it.get("published_at","")
            st.markdown(f"- [{it['title']}]({it['url']}) — {ts}")
    st.divider()
