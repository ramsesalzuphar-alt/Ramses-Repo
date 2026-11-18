"""Own trades: practice strategies that decide trades themselves, from market data and the news, instead of copying
a trader. Each check, entries() says which coins each signal would trade now; shadow.open_signals practices them in
their own books ($100 a trade, held a fixed time), graded and shown in the tournament like every other strategy.
Never raises: a source that fails just gives no signals this check.

Backtests where history exists: shorting very high funding won in 2 of 3 periods (worth testing live); Fear & Greed
extremes had no edge over 5 years (left out). Open-interest surges and news have no free history, so they can only
prove themselves here, live.
"""
import re, time, traceback
from copier import get
import crypto_copier as cc

OI_MOVE, OI_LAG = 0.02, 3600  # open interest compared hour on hour; the price must have moved 2%+ too
NEWS_EVERY = 15 * 60
FEEDS = ("https://www.coindesk.com/arc/outboundfeeds/rss/", "https://cointelegraph.com/rss", "https://decrypt.co/feed")
# a grid of variants, all practiced at once: which threshold and which hold work is what the tournament finds out
# (backtest: shorts above +0.02%/h funding won 64%; more variants also means more lucky ones, which graduation filters)
FUNDING = [(t, h) for t in (0.0001, 0.0002, 0.0005) for h in (8, 24)]
OI = [(j, h) for j in (0.05, 0.10, 0.20) for h in (4, 12)]
NEWS_HOLDS = (6, 24, 72)
fund_name = lambda side, t, h: f"Own: {side} when funding {'above +' if side == 'short' else 'below -'}{t * 100:.2f}%/h, {h}h"
oi_name = lambda way, j, h: f"Own: open interest +{j:.0%} in 1h, {way}, {h}h"
news_name = lambda h: f"Own: news sentiment, {h}h"
SIGNALS = {  # practice book -> hours each trade is held
    **{fund_name(side, t, h): h for t, h in FUNDING for side in ("short", "long")},
    **{oi_name(way, j, h): h for j, h in OI for way in ("with the move", "against the move")},
    **{news_name(h): h for h in NEWS_HOLDS},
}
NAMES = {"BTC": r"bitcoin|btc", "ETH": r"ethereum|ether\b|eth\b", "SOL": r"solana|\bsol\b", "XRP": r"\bxrp\b|ripple",
         "DOGE": r"dogecoin|\bdoge\b", "ADA": r"cardano", "AVAX": r"avalanche|\bavax\b", "LINK": r"chainlink",
         "DOT": r"polkadot", "LTC": r"litecoin", "BNB": r"\bbnb\b|binance coin", "SUI": r"\bsui\b", "HYPE": r"hyperliquid|\bhype\b",
         "TON": r"toncoin|\bton\b", "NEAR": r"near protocol", "UNI": r"uniswap", "AAVE": r"\baave\b", "TRX": r"\btron\b|\btrx\b"}
UP = r"surge|soar|rall(y|ies)|jump|record high|all-time high|approv|etf inflow|partnership|adopt|upgrade|bullish|breakout|buy(s|ing)? \$"
DOWN = r"hack|exploit|drain|lawsuit|sues|charged|ban(s|ned)?\b|crash|plunge|tumble|delist|outage|halt|bearish|liquidat|selloff|sell-off|fraud"
_state = {"oi": [], "news_at": 0.0, "seen": set()}


def news_score(title):
    """(coin, +1 bullish / -1 bearish) for each listed coin a headline names with a clear tone, else []."""
    t = title.lower()
    tone = bool(re.search(UP, t)) - bool(re.search(DOWN, t))
    return [(c, tone) for c, pat in NAMES.items() if tone and re.search(pat, t)]


def from_news(titles):
    """Headline tones summed per coin -> [(book, coin, side)]; mixed news (net 0) trades nothing."""
    net = {}
    for title in titles:
        for coin, tone in news_score(title):
            net[coin] = net.get(coin, 0) + tone
    return [(news_name(h), c, "long" if n > 0 else "short") for c, n in net.items() if n for h in NEWS_HOLDS]


def from_market(ctxs, now, snaps):
    """Funding and open-interest signals from Hyperliquid's market snapshot: ctxs = {coin: ctx}. snaps is the list of
    earlier (time, {coin: (open interest, price)}) snapshots, kept here for the hour-on-hour comparison."""
    out = []
    for coin, c in ctxs.items():
        f = float(c.get("funding") or 0)
        for t, h in FUNDING:
            if f >= t:
                out.append((fund_name("short", t, h), coin, "short"))
            elif f <= -t:
                out.append((fund_name("long", t, h), coin, "long"))
    snap = {coin: (float(c.get("openInterest") or 0), float(c.get("markPx") or 0)) for coin, c in ctxs.items()}
    old = next((s for t, s in snaps if now - t >= OI_LAG), None)  # snaps: newest first
    if old:
        for coin, (oi, px) in snap.items():
            o = old.get(coin)
            if o and o[0] > 0 and o[1] > 0 and abs(move := px / o[1] - 1) >= OI_MOVE:
                up = "long" if move > 0 else "short"
                for j, h in OI:
                    if oi / o[0] - 1 >= j:  # momentum and its mirror (fading the move), so the tournament tells which
                        out += [(oi_name("with the move", j, h), coin, up),
                                (oi_name("against the move", j, h), coin, "short" if up == "long" else "long")]
    snaps.insert(0, (now, snap))
    snaps[:] = [s for s in snaps if now - s[0] <= 2 * OI_LAG]  # two hours of 5-minute snapshots is plenty
    return out


def entries(now=None):
    """[(book, coin, side)] every signal would trade right now."""
    now = now or time.time()
    out = []
    try:
        meta, ctxs = cc.hl({"type": "metaAndAssetCtxs"})
        live = {u["name"]: c for u, c in zip(meta["universe"], ctxs) if not u.get("isDelisted") and float(c.get("dayNtlVlm") or 0) > 1e6}
        out += from_market(live, now, _state["oi"])
    except Exception:
        traceback.print_exc()
    if now - _state["news_at"] >= NEWS_EVERY:
        _state["news_at"] = now
        new = []
        for url in FEEDS:
            try:
                xml = get(url).decode("utf-8", "replace")
                for t in re.findall(r"<title>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</title>", xml, re.S)[1:]:
                    if t not in _state["seen"]:
                        _state["seen"].add(t)
                        new.append(t)
            except Exception:
                traceback.print_exc()
        if len(_state["seen"]) > 5000:
            _state["seen"] = set(list(_state["seen"])[-2000:])
        if _state.get("primed"):  # the first read only learns what's old news
            out += from_news(new)
        _state["primed"] = True
    return out
