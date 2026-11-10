"""Crypto copier logic with a built-in paper account (no exchange, no keys, works in the US). Used by app.py.
Watches the top Hyperliquid traders. A new position becomes a proposal you approve; once copied, your position is
closed automatically as soon as THAT trader no longer holds it, even if they've dropped out of the top list.
Optional .env: CRYPTO_DOLLARS_PER_TRADE (default 100), CRYPTO_TOP_N (default 5), CRYPTO_START_CASH (default 10000)
"""
import json, os, statistics, time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from copier import load_env, http, load_model, predict_for_trader, verdict, daily_closes, global_features, GLOBAL_SYMBOLS

HERE = Path(__file__).parent
STATE_FILE = HERE / "crypto_state.json"
LEDGER_FILE = HERE / "paper_ledger.json"
EVENTS_FILE = HERE / "crypto_events.jsonl"  # every new position seen, so learn.py can learn from them later
SCOUT_FILE = HERE / "scout.json"  # written by scout.py: every scanned trader's consistency verdict
SCOUT_VERSION = 2  # bump when the scout's rules change: older verdicts are ignored and rescanned first
WATCH_MAX = 30
MAJORS = {"BTC", "ETH", "SOL"}
FEE = 0.00045  # Hyperliquid base taker fee per side
STOP_LOSS = 0.02  # return on the position (1x, so = price move); autopilot trades only. If changed, update index.html's text
TAKE_PROFIT = 3 * STOP_LOSS  # 3:1; applies to every copy, yours too
# shortcut: ignores funding payments and slippage, so results look slightly better than real; add them before trusting P&L


def short_addr(a):
    return a[:6] + ".." + a[-4:]


def money(x):
    return f"{'+' if x >= 0 else '-'}${abs(x):,.2f}"


def leaderboard():
    """Every Hyperliquid leaderboard row. Retries: this network drops Hyperliquid connections now and then."""
    for wait in (5, 20, 60, 0):
        try:
            st, d = http("GET", "https://stats-data.hyperliquid.xyz/Mainnet/leaderboard", {})
            if st == 200:
                return d["leaderboardRows"]
            err = RuntimeError(f"leaderboard error {st}")
        except OSError as e:
            err = e
        time.sleep(wait)
    raise err


def top_traders(n, window="month"):
    """[(profit, addr, name)] of the n highest earners over `window` ("month" or "allTime"), active this month."""
    rows = []
    for r in leaderboard():
        w = dict(r["windowPerformances"])
        # vlm>0 drops idle accounts that "earn" by deposits; allTime>0 drops one lucky month after big losses
        if float(w["month"]["vlm"]) > 0 and float(w["allTime"]["pnl"]) > 0:
            rows.append((float(w[window]["pnl"]), r["ethAddress"], r.get("displayName")))
    return sorted(rows, reverse=True)[:n]


def edge_of(t):
    """How consistently copying this scouted trader won: worst-period win rate of the trader or of their best pattern."""
    return max([t["worst"]] * bool(t.get("consistent")) + [p["worst"] for p in t.get("patterns", [])], default=0)


def win_low(p, n, z=2.33):
    """Lower 99% confidence bound of a win rate p over n trades: 45 of 60 -> ~0.60; 9 of 12 -> ~0.44."""
    return (p + z * z / (2 * n) - z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5) / (1 + z * z / n)


def high_winners(k=30, min_n=30):
    """[(addr, name, scout record)] of the scanned traders whose copies won most reliably (ranked by the 99% lower
    bound of their win rate, so a lucky 5-for-5 doesn't rank) and made money on average."""
    try:
        traders = json.loads(SCOUT_FILE.read_text())["traders"]
    except (OSError, json.JSONDecodeError, KeyError):
        return []
    ok = [(win_low(t["win"], t["n"]), a, t) for a, t in traders.items()
          if t.get("v") == SCOUT_VERSION and t.get("n", 0) >= min_n and (t.get("avg") or 0) > 0]
    return [(a, t.get("name"), t) for _, a, t in sorted(ok, key=lambda x: -x[0])[:k]]


def watchlist(k=WATCH_MAX):
    """[(addr, name, scout record)] of the scout's most consistent finds, best first."""
    try:
        traders = json.loads(SCOUT_FILE.read_text())["traders"]
    except (OSError, json.JSONDecodeError, KeyError):
        return []
    found = [(edge_of(t), t["n"], a, t) for a, t in traders.items() if t.get("v") == SCOUT_VERSION and edge_of(t)]
    return [(a, t.get("name"), t) for _, _, a, t in sorted(found, reverse=True)[:k]]


def hl(body):
    """One Hyperliquid request. Retries dropped connections (this network drops them now and then) and rate limits
    (429, when the scout or a relearn is busy too), then gives up."""
    for wait in (1, 3, 10, None):
        try:
            st, d = http("POST", "https://api.hyperliquid.xyz/info", {}, body)
            if st != 429 or wait is None:
                break
        except OSError:
            if wait is None:
                raise
        time.sleep(wait)
    if st != 200:
        raise RuntimeError(f"Hyperliquid error {st}: {d}")
    return d


def positions_full(addr):
    """{coin: (side, position value in $)} of everything the trader holds."""
    return {p["position"]["coin"]: ("long" if float(p["position"]["szi"]) > 0 else "short", float(p["position"].get("positionValue") or 0))
            for p in hl({"type": "clearinghouseState", "user": addr})["assetPositions"]}


def positions(addr):
    return {c: side for c, (side, _) in positions_full(addr).items()}


def market_stats():
    """{coin: {"markPx", "prevDayPx", "funding" (hourly rate)}, ...} for every perp."""
    meta, ctxs = hl({"type": "metaAndAssetCtxs"})
    return {u["name"]: c for u, c in zip(meta["universe"], ctxs)}


def diff(old, new):
    """Return (event, addr, coin) for every position opened or closed. A side flip counts as close + open."""
    events = []
    for addr, pos in new.items():
        before = old.get(addr, {})
        for coin, side in before.items():
            if pos.get(coin) != side:
                events.append((f"closed {side}", addr, coin))
        for coin, side in pos.items():
            if before.get(coin) != side:
                events.append((f"opened {side}", addr, coin))
    return events


def crypto_features(side, coin, addr, move):
    """Traits learn.py learns from and the live copier scores on. move = price change over the 24h before the trade."""
    with_trend = move if side == "long" else -move
    return {"side": side,
            "trend": "with 24h trend" if with_trend > 0.02 else "against 24h trend" if with_trend < -0.02 else "flat 24h",
            "coin": "BTC/ETH/SOL" if coin in MAJORS else "altcoin",
            "trader": "trader " + short_addr(addr)}


HOUR = 3_600_000
_fng = {"at": 0, "days": {}}


def fear_greed():
    """{YYYY-MM-DD: 0-100} Crypto Fear & Greed Index, every day since 2018 (alternative.me; built daily from news and
    social sentiment, volatility, momentum and dominance). The real-world mood the learner can learn from. Cached 6h."""
    if time.time() - _fng["at"] > 6 * 3600:
        st, d = http("GET", "https://api.alternative.me/fng/?limit=0&format=json", {})
        if st == 200 and isinstance(d, dict) and d.get("data"):
            _fng.update(at=time.time(), days={datetime.fromtimestamp(int(x["timestamp"]), timezone.utc).strftime("%Y-%m-%d"): int(x["value"])
                                              for x in d["data"]})
    return _fng["days"]


def sentiment_feature(fng, t):
    """The market mood on the day of time t (ms): extreme fear .. extreme greed. {} if unknown."""
    v = fng.get(datetime.fromtimestamp(t / 1000, timezone.utc).strftime("%Y-%m-%d"))
    if v is None:
        return {}
    return {"sentiment": "extreme fear" if v < 25 else "fear" if v < 45 else "neutral mood" if v <= 55
            else "greed" if v <= 75 else "extreme greed"}


def regime_feature(btc_closes, t, side):
    """Whole-market regime: BTC's move over the 7 days before t, relative to the trade's side (a short in a falling
    market is 'with trade'). btc_closes = {hour start ms: close}, only finished hours used. {} if missing."""
    px = lambda ts: btc_closes.get(ts - ts % HOUR)
    now, week = px(t - HOUR), px(t - 7 * 24 * HOUR)
    if not (now and week):
        return {}
    rel = (now / week - 1) * (1 if side == "long" else -1)
    return {"regime": "BTC 7d: with trade" if rel > 0.05 else "BTC 7d: against trade" if rel < -0.05 else "BTC 7d: flat"}


def btc_hourly(start, end):
    return {k["t"]: float(k["c"]) for k in hl({"type": "candleSnapshot", "req": {
        "coin": "BTC", "interval": "1h", "startTime": start, "endTime": end}})}


M5 = 5 * 60_000  # five minutes in ms


def candles_5m(coin, start, end):
    """5-minute candles (Hyperliquid keeps ~17 days of them)."""
    return [{"t": k["t"], "h": float(k["h"]), "l": float(k["l"]), "c": float(k["c"]), "v": float(k["v"])}
            for k in hl({"type": "candleSnapshot", "req": {"coin": coin, "interval": "5m", "startTime": start, "endTime": end}})]


def minute_features(candles, t, side):
    """Market conditions in the 4 hours before time t, from 5-minute candles that had already closed by t
    (so nothing after the trade leaks in). Side-relative: 'its way' = the direction the trade profits. {} if too little data."""
    past = [k for k in candles if k["t"] + M5 <= t][-48:]
    if len(past) < 48:
        return {}
    d = 1 if side == "long" else -1
    c = [k["c"] for k in past]
    m15, m60 = (c[-1] / c[-4] - 1) * d, (c[-1] / c[-13] - 1) * d
    vol = statistics.pstdev([c[i] / c[i - 1] - 1 for i in range(-12, 0)])
    recent_v, base_v = sum(k["v"] for k in past[-3:]) / 3, sum(k["v"] for k in past[:-3]) / 45
    hi, lo = max(k["h"] for k in past), min(k["l"] for k in past)
    pos = (c[-1] - lo) / (hi - lo) if hi > lo else 0.5
    pos = pos if side == "long" else 1 - pos  # 1 = at the 4h extreme in the trade's direction
    return {
        "move15": "15m before: moving its way" if m15 > 0.003 else "15m before: moving against" if m15 < -0.003 else "15m before: flat",
        "move60": "1h before: moving its way" if m60 > 0.01 else "1h before: moving against" if m60 < -0.01 else "1h before: flat",
        "volatility": "calm last hour" if vol < 0.0015 else "wild last hour" if vol > 0.004 else "normal volatility last hour",
        "volume": ("volume surge" if base_v and recent_v > 2 * base_v else
                   "volume dried up" if base_v and recent_v < 0.5 * base_v else "normal volume"),
        "range": ("chasing: at 4h extreme its way" if pos > 0.8 else
                  "fading: at opposite 4h extreme" if pos < 0.2 else "mid 4h range"),
    }


def score_crypto(side, holders, stats, churn):
    """Fallback rule-based 0-100 score (used until learn.py has trained a model)."""
    pts, why = 50, []
    agree, against = holders.count(side) - 1, sum(s != side for s in holders)
    pts += 15 * agree - 15 * against
    why.append(f"{agree} other top trader(s) agree, {against} on the other side" if agree or against else "no other top trader holds it")
    if stats:
        move = float(stats["markPx"]) / float(stats["prevDayPx"]) - 1
        with_trend = move if side == "long" else -move
        if with_trend > 0.02:
            pts += 15; why.append(f"24h trend agrees ({move:+.1%})")
        elif with_trend < -0.02:
            pts -= 15; why.append(f"against 24h trend ({move:+.1%})")
        else:
            why.append(f"flat 24h ({move:+.1%})")
        cost = float(stats["funding"]) * (1 if side == "long" else -1)  # longs pay positive funding, shorts receive it
        if cost > 0.0001:
            pts -= 10; why.append(f"pricey to hold ({cost * 24 * 365:.0%}/yr funding)")
        elif cost < -0.00005:
            pts += 5; why.append("funding pays you to hold")
    if churn >= 6:
        pts -= 20; why.append(f"trader made {churn} changes since last check, likely a bot")
    return max(0, min(100, pts)), why


def load_ledger():
    if LEDGER_FILE.exists():
        return json.loads(LEDGER_FILE.read_text())
    start = float(os.environ.get("CRYPTO_START_CASH", "10000"))
    return {"start": start, "cash": start, "positions": {}, "history": []}


def save_ledger(ledger):
    LEDGER_FILE.write_text(json.dumps(ledger, indent=1))


def open_pos(ledger, coin, side, dollars, px, source=None, feats=None, auto=False, exit=None):
    fee = dollars * FEE
    ledger["cash"] -= dollars + fee  # 1x: the full position size is set aside as margin
    ledger["positions"][coin] = {"side": side, "qty": dollars / px, "entry": px, "margin": dollars, "source": source,
                                 "feats": feats, "auto": auto,  # feats: conditions at entry, learned from when it closes
                                 "exit": exit, "opened": datetime.now().isoformat(timespec="seconds")}
    ledger["history"].append({"time": datetime.now().isoformat(timespec="seconds"), "coin": coin,
                              "action": f"open {side}", "price": px, "dollars": dollars, "fee": fee, "source": source})


def pnl_of(p, px):
    return (px - p["entry"]) * p["qty"] * (1 if p["side"] == "long" else -1)


def close_pos(ledger, coin, px):
    p = ledger["positions"].pop(coin)
    pnl, fee = pnl_of(p, px), p["qty"] * px * FEE
    ledger["cash"] += p["margin"] + pnl - fee
    ledger["history"].append({"time": datetime.now().isoformat(timespec="seconds"), "coin": coin,
                              "action": f"close {p['side']}", "price": px, "pnl": pnl - fee, "fee": fee,
                              "ret": (pnl - fee) / p["margin"], "feats": p.get("feats"), "auto": p.get("auto", False),
                              "source": p.get("source")})
    return pnl - fee


def equity(ledger, mids):
    return ledger["cash"] + sum(p["margin"] + pnl_of(p, float(mids[c])) for c, p in ledger["positions"].items() if c in mids)


def auto_exits(ledger, holdings, mids, log):
    """Close every copy whose source trader no longer holds that coin on that side. Catches missed exits too."""
    for coin, p in list(ledger["positions"].items()):
        src = p.get("source")
        if src in holdings and holdings[src].get(coin) != p["side"] and coin in mids:
            pnl = close_pos(ledger, coin, float(mids[coin]))
            log(f"Auto-closed {p['side']} {coin}: {short_addr(src)} exited ({money(pnl)} after fees)")


def is_new_position(fill):
    """A fill that opens a fresh position (from zero) or flips long<->short. Shared with learn.py."""
    d = fill["dir"]
    return (d.startswith("Open") and float(fill["startPosition"]) == 0) or d in ("Long > Short", "Short > Long")


def entry_times(addr):
    """{coin: (time ms, price)} of the trader's latest new position per coin, from their recent fills."""
    out = {}
    for x in hl({"type": "userFills", "user": addr, "aggregateByTime": True}):
        if is_new_position(x) and x["time"] > out.get(x["coin"], (0, 0))[0]:
            out[x["coin"]] = (x["time"], float(x["px"]))
    return out


def rule_of(p):
    """(take profit, stop loss, max hours) of a copy. Autopilot copies use the close rule the scout learned for their
    trader, if it found a consistent one; otherwise the 3:1 default. Your own copies only take profit."""
    if not p.get("auto"):
        return TAKE_PROFIT, None, None
    r = p.get("exit")
    return (r.get("tp"), r.get("sl"), r.get("max_h")) if r else (TAKE_PROFIT, STOP_LOSS, None)


def target_exits(ledger, mids, log):
    """Close copies that hit their take profit, stop loss or time limit. (Following the trader's exit is auto_exits.)"""
    now = datetime.now()
    for coin, p in list(ledger["positions"].items()):
        if coin not in mids:
            continue
        tp, sl, mh = rule_of(p)
        r = pnl_of(p, float(mids[coin])) / p["margin"]
        why = ("Take profit" if tp and r >= tp else "Stop loss" if sl and r <= -sl else
               "Time limit" if mh and p.get("opened") and now - datetime.fromisoformat(p["opened"]) >= timedelta(hours=mh) else None)
        if why:
            pnl = close_pos(ledger, coin, float(mids[coin]))
            log(f"{why} hit ({r:+.1%}): closed {p['side']} {coin} ({money(pnl)} after fees)")


def rule_text(p):
    tp, sl, mh = rule_of(p)
    learned = (p.get("exit") or {}).get("name") if p.get("auto") else None
    return "follow trader" + "".join(x for x in (tp and f" · TP {tp:.1%}", sl and f" · SL {sl:.1%}", mh and f" · max {mh}h") if x) + (
        " (learned)" if learned else "")


def scout_exit(addr):
    """The close rule the scout learned for this trader, or None."""
    try:
        t = json.loads(SCOUT_FILE.read_text())["traders"].get(addr, {})
        return t.get("exit") if t.get("v") == SCOUT_VERSION else None
    except (OSError, json.JSONDecodeError, KeyError):
        return None


def exit_check(log):
    """Quick check between full checks (every minute): take profit / stop loss, then exits of the traders you copied."""
    ledger = load_ledger()
    before = len(ledger["positions"])
    mids = hl({"type": "allMids"})
    target_exits(ledger, mids, log)
    copied = {p["source"] for p in ledger["positions"].values() if p.get("source")}
    if copied:
        auto_exits(ledger, {a: positions(a) for a in copied}, mids, log)
    if len(ledger["positions"]) != before:
        save_ledger(ledger)


def check_crypto(pending, log):
    """One check: auto-exit copies whose trader left, drop stale proposals, add new positions to `pending`.
    Returns the top-trader list for the UI."""
    load_env()
    n = int(os.environ.get("CRYPTO_TOP_N", "5"))
    dollars = float(os.environ.get("CRYPTO_DOLLARS_PER_TRADE", "100"))
    ledger = load_ledger()
    top = top_traders(n)
    tops = [a for _, a, _ in top]
    names = {a: nm or short_addr(a) for _, a, nm in top}
    scouted = {}
    for a, nm, t in watchlist():  # plus the scout's consistent finds, whatever their size
        scouted[a] = t
        if a not in names:
            tops.append(a)
            names[a] = nm or short_addr(a)
    mids, stats, model = hl({"type": "allMids"}), market_stats(), load_model("crypto")
    copied = {p["source"] for p in ledger["positions"].values() if p.get("source")}
    full = {a: positions_full(a) for a in set(tops) | copied}  # keep watching traders you copied, even off the top list
    new = {a: {c: side for c, (side, _) in f.items()} for a, f in full.items()}

    target_exits(ledger, mids, log)
    auto_exits(ledger, new, mids, log)
    pending[:] = [x for x in pending if x["kind"] != "crypto" or new.get(x["addr"], {}).get(x["coin"]) == x["side"]]

    old = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
    known = [a for a in tops if a in old]  # a trader just entering the top list is only recorded, not treated as 26 "new" trades
    events = diff({a: old[a] for a in known}, {a: new[a] for a in known})
    churn = Counter(a for _, a, _ in events)
    waiting = {(x.get("addr"), x.get("coin")) for x in pending}
    entries = {}  # trader -> entry_times(), fetched only for traders who opened something
    c5 = {}  # coin -> recent 5-minute candles, fetched once per coin per check
    try:  # market-wide conditions, once per check: BTC's week, and the real-world mood
        now0 = int(datetime.now().timestamp() * 1000)
        btc, fng = btc_hourly(now0 - 8 * 24 * HOUR, now0), fear_greed()
    except Exception:
        btc, fng = {}, {}
    try:  # the world's markets, daily (Alpaca): dollar, rates, oil, gold, world stocks, fear gauge
        world = daily_closes(GLOBAL_SYMBOLS, (datetime.now() - timedelta(days=45)).date().isoformat(), datetime.now().date().isoformat())
    except Exception:
        world = {}
    with EVENTS_FILE.open("a") as f:
        for event, addr, coin in events:
            verb, side = event.split()
            if verb != "opened":
                continue
            now_ms = int(datetime.now().timestamp() * 1000)
            if addr not in entries:
                try:
                    entries[addr] = entry_times(addr)
                except RuntimeError:
                    entries[addr] = {}  # no entry time is better than no proposal
            when, their_px = entries[addr].get(coin, (None, None))
            s = stats.get(coin)
            holders = [p[coin] for a, p in new.items() if a in tops and coin in p]
            learned, feats = bool(model and s), None
            if learned:
                feats = crypto_features(side, coin, addr, float(s["markPx"]) / float(s["prevDayPx"]) - 1)
                if coin not in c5:
                    try:
                        c5[coin] = candles_5m(coin, now_ms - 8 * 3_600_000, now_ms)
                    except RuntimeError:
                        c5[coin] = []
                feats.update(minute_features(c5[coin], when or now_ms, side))  # conditions up to the trader's actual entry
                feats.update(regime_feature(btc, when or now_ms, side))
                feats.update(sentiment_feature(fng, when or now_ms))
                feats.update(global_features(world, datetime.fromtimestamp((when or now_ms) / 1000, timezone.utc).date().isoformat(), side))
                score, why = predict_for_trader(model, feats)
                agree = holders.count(side) - 1
                why.append(f"{agree} other top trader(s) agree, {len(holders) - holders.count(side)} opposite")
                if addr in scouted:
                    t = scouted[addr]
                    hit = [p for p in t["patterns"] if p["when"] in feats.values()]
                    why.insert(1, f"scout: {', '.join(p['when'] for p in hit)} is their consistent pattern ({hit[0]['win']:.0%} won)"
                               if hit else f"scout: copies won {t['worst']:.0%}+ in every period (n={t['n']})"
                               if t["consistent"] else "scout: this trade is NOT one of their consistent patterns")
            else:
                score, why = score_crypto(side, holders, s, churn[addr])
            # every new position is logged with its learned score, copied or not, so learn.py can grade it 24h later
            f.write(json.dumps({"t": now_ms, "addr": addr, "coin": coin, "side": side, "entered_at": when,
                                "score": score if learned else None}) + "\n")
            if coin in ledger["positions"] or (addr, coin) in waiting or coin not in mids:
                continue
            pending.append({"id": f"c:{addr}:{coin}:{side}", "kind": "crypto", "addr": addr, "coin": coin, "side": side,
                            "label": f"{side.title()} ${dollars:.0f} of {coin}",
                            "title": f"{names[addr]} opened {side} {coin}",
                            "detail": "1x leverage · paper",
                            "entered_at": when, "entry_px": their_px, "price_now": float(mids[coin]),
                            "volume": full[addr][coin][1],  # the trader's position size in $
                            "seen_at": int(datetime.now().timestamp() * 1000),
                            "score": score, "verdict": verdict(score, 60 if model else 70), "why": why, "feats": feats})
    entered = [names[a] for a in tops if a not in old]
    if old and entered:
        log(f"Now watching: {', '.join(entered)} (recorded their current positions)")
    STATE_FILE.write_text(json.dumps(new))
    save_ledger(ledger)
    return [{"name": names[a], "pnl": pnl, "longs": sum(s == "long" for s in new[a].values()),
             "shorts": sum(s == "short" for s in new[a].values())} for pnl, a, _ in top]


def decide_crypto(item, auto=False, stake=1):
    """Copy an approved proposal at the live price, if the trader still holds it. stake multiplies the trade size
    (autopilot earns more than x1 by being consistently profitable). Returns a message."""
    load_env()
    dollars = float(os.environ.get("CRYPTO_DOLLARS_PER_TRADE", "100")) * stake
    ledger, coin, side = load_ledger(), item["coin"], item["side"]
    if coin in ledger["positions"]:
        return f"You already hold {coin}, not copied"
    if positions(item["addr"]).get(coin) != side:
        return f"{short_addr(item['addr'])} already exited {coin}, not copied"
    if ledger["cash"] < dollars * (1 + FEE):
        return f"Not enough paper cash ({money(ledger['cash'])})"
    px = float(hl({"type": "allMids"})[coin])
    rule = scout_exit(item["addr"]) if auto else None  # autopilot closes each trader's copies by that trader's learned rule
    open_pos(ledger, coin, side, dollars, px, item["addr"], item.get("feats"), auto, rule)
    save_ledger(ledger)
    return f"Copied: {side} ${dollars:.0f} of {coin} at ${px:,.6g}" + (f" · close rule: {rule['name']}" if rule else "")


def close_manual(coin):
    ledger = load_ledger()
    if coin not in ledger["positions"]:
        return f"No {coin} position"
    pnl = close_pos(ledger, coin, float(hl({"type": "allMids"})[coin]))
    save_ledger(ledger)
    return f"Closed {coin} ({money(pnl)} after fees)"


_last_mids = {}


def crypto_account():
    """If Hyperliquid can't be reached, still show the positions, priced at the last prices seen."""
    global _last_mids
    ledger, stale = load_ledger(), None
    try:
        mids = _last_mids = hl({"type": "allMids"})
    except Exception as e:
        stale = f"Can't reach Hyperliquid ({e}). Prices and P&L are frozen at the last ones seen (or entry price); auto-exits wait until it's back."
        mids = {**{c: p["entry"] for c, p in ledger["positions"].items()}, **_last_mids}
    eq = equity(ledger, mids)
    invested = sum(p["margin"] for p in ledger["positions"].values())  # what you put into the open copies (1x: their size)
    value = invested + sum(pnl_of(p, float(mids.get(c, p["entry"]))) for c, p in ledger["positions"].items())
    return {"stale": stale, "equity": eq, "change": eq - ledger["start"], "cash": ledger["cash"], "invested": invested, "value": value,
            "positions": [{"coin": c, "side": p["side"], "entry": p["entry"], "now": float(mids.get(c, p["entry"])),
                           "pnl": pnl_of(p, float(mids.get(c, p["entry"]))), "source": short_addr(p["source"]) if p.get("source") else "?",
                           "auto": bool(p.get("auto")), "rule": rule_text(p)}
                          for c, p in sorted(ledger["positions"].items())]}
