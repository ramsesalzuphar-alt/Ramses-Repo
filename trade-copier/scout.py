"""Edge scout: a nonstop scan of every active, profitable Hyperliquid trader (thousands, not just the top earners),
looking for traders whose trades are CONSISTENTLY worth copying: copies won in every period of their history,
not just on average, or a pattern of theirs (e.g. their shorts against the 24h trend) won in both halves of it.
Each copy is graded like learn.py: enter 30 min after them, hold 24h, after fees.
Runs as a background thread in the app; writes scout.json. The copier watches the best finds (crypto_copier.watchlist)
and learn.py builds them personal models. Run alone: python scout.py
"""
import json, os, statistics, time
from array import array
from datetime import datetime
import learn
from crypto_copier import SCOUT_FILE, SCOUT_VERSION, FEE, win_low, leaderboard, fear_greed, sentiment_feature, regime_feature, crypto_features, is_new_position

H = learn.H
MIN_TRADES, PARTS, MIN_WIN = 30, 3, 0.55  # consistent trader: 30+ graded copies; each third of their history won 55%+ and made money
PAT_MIN, PAT_WIN = 15, 0.60               # consistent pattern: 15+ copies; both halves won 60%+ and made money
PAGE_PAUSE = 10  # extra seconds per older page of fills read
PAUSE = 8          # seconds between traders: well under Hyperliquid's rate limit, leaving room for the copier's own checks
SAVE_EVERY = 120   # seconds between saves, so a restart loses at most ~2 minutes of scanning
PRICES_FOR = 3600  # seconds hourly prices are reused before refetching


def candidates():
    """[(addr, name)] of every account that traded $100k+ this month and is profitable all-time, any size."""
    out = []
    for r in leaderboard():
        w = dict(r["windowPerformances"])
        if float(w["month"]["vlm"]) > 100_000 and float(w["allTime"]["pnl"]) > 0:
            out.append((r["ethAddress"], r.get("displayName")))
    return out


HISTORY_DAYS, MAX_PAGES = 200, 4  # Hyperliquid serves a trader's 10000 latest fills: the latest 2000 + up to 4 older pages


def fills(addr):
    """Up to the trader's 10000 latest fills, oldest first. userFillsByTime returns the OLDEST 2000 in a window,
    so older pages are walked forward from 200 days ago up to where the latest 2000 begin. Returns (fills, extra pages)."""
    latest = learn.hl({"type": "userFills", "user": addr, "aggregateByTime": True}) or []
    older, pages = [], 0
    if len(latest) >= 2000:
        first, start = min(x["time"] for x in latest), int(time.time() * 1000) - HISTORY_DAYS * 24 * H
        while pages < MAX_PAGES and start < first:
            page = learn.hl({"type": "userFillsByTime", "user": addr, "startTime": start, "endTime": first - 1,
                             "aggregateByTime": True}) or []
            pages += 1
            older += page
            if len(page) < 2000:
                break
            start = max(x["time"] for x in page) + 1
    return sorted(older + latest, key=lambda x: x["time"]), pages


def opens(addr, per_trader=300):
    """[(time ms, coin, side, their close time or None, their return or None)] of the trader's new positions in their
    latest fills, one per coin+side per half hour. Their close = the fill that took that position to zero or flipped it.
    Also returns how many extra pages of history were read (each one costs rate limit)."""
    seen, out, waiting = set(), [], {}
    fs, pages = fills(addr)
    for x in fs:
        coin = x["coin"]
        if ":" in coin or coin.startswith("@"):
            continue  # spot and side-exchange markets
        start = float(x["startPosition"])
        after = start + float(x["sz"]) * (1 if x["side"] == "B" else -1)
        if start and (abs(after) <= abs(start) * 1e-9 or (after > 0) != (start > 0)):  # closed out, or flipped
            for rec in waiting.pop(coin, []):
                rec[3], rec[4] = x["time"], (float(x["px"]) / rec[5] - 1) * (1 if rec[2] == "long" else -1)
        if is_new_position(x):
            side = "long" if x["dir"] in ("Open Long", "Short > Long") else "short"
            key = (coin, side, x["time"] // (H // 2))
            if key not in seen:
                seen.add(key)
                rec = [x["time"], coin, side, None, None, float(x["px"])]
                out.append(rec)
                waiting.setdefault(coin, []).append(rec)
    return [tuple(r[:5]) for r in out[-per_trader:]], pages


def hourly(coins, since, now):
    """{coin: (first hour ms, highs, lows, closes)} in compact arrays (thousands of coins x months of hours would
    otherwise eat hundreds of MB). An hour with no trades repeats the last close."""
    out = {}
    for c in sorted(coins):
        ks = learn.hl({"type": "candleSnapshot", "req": {"coin": c, "interval": "1h", "startTime": since, "endTime": now}}) or []
        if ks:
            start, hi, lo, cl = ks[0]["t"], array("d"), array("d"), array("d")
            for k in ks:
                while len(cl) < (k["t"] - start) // H:
                    hi.append(cl[-1]); lo.append(cl[-1]); cl.append(cl[-1])
                hi.append(float(k["h"])); lo.append(float(k["l"])); cl.append(float(k["c"]))
            out[c] = (start, hi, lo, cl)
        time.sleep(0.5)
    return out


def at(cs, ts):
    """(high, low, close) of the hourly candle containing ts, or None."""
    i = (ts - cs[0]) // H if cs else -1
    return (cs[1][i], cs[2][i], cs[3][i]) if 0 <= i < len(cs[3]) else None


M5 = 5 * 60_000
_fine = {}  # (coin, hour ms) -> [(high, low)] of its 5-minute candles, or [] if older than Hyperliquid keeps (~17 days)


def five(coin, hour):
    """The 5-minute candles inside one hour, to tell whether the take profit or the stop loss came first."""
    key = (coin, hour - hour % H)
    if key not in _fine:
        ks = learn.hl({"type": "candleSnapshot", "req": {"coin": coin, "interval": "5m", "startTime": key[1],
                                                          "endTime": key[1] + H - 1}}) or []
        _fine[key] = [(float(k["h"]), float(k["l"])) for k in sorted(ks, key=lambda k: k["t"])]
        if len(_fine) > 20000:
            _fine.clear()  # bounded memory; refetching is cheap
        time.sleep(0.3)
    return _fine[key]


MAX_HOLD, MIN_EXIT = 7 * 24, 20  # a copy still open after a week is closed; a rule needs 20+ replayed copies to be judged


def first_hit(bars, entry, d, tp, sl):
    """'tp', 'sl' or None: which level the bars (high, low) reached first. Both inside one bar -> 'sl' (the cautious guess)."""
    for hi, lo in bars:
        moves = ((hi / entry - 1) * d, (lo / entry - 1) * d)
        if sl and min(moves) <= -sl:
            return "sl"
        if tp and max(moves) >= tp:
            return "tp"
    return None


def simulate(cs, t, side, close_t, rule, now, fine=None):
    """Return of one copy under a close rule, replayed on hourly candles: enter 30 min after the trader, close at whichever
    comes first: take profit, stop loss, time limit, the trader's own close, or a week. None if the outcome isn't known yet.
    fine(hour) -> that hour's 5-minute (high, low) bars: used when one hour reached both the take profit and the stop loss."""
    entry = at(cs, t + H // 2)
    if not entry or (close_t and close_t < t + H // 2):
        return None  # no prices, or they closed before a copy would have entered (the copier wouldn't copy it)
    entry, d = entry[2], 1 if side == "long" else -1
    limit = t + (rule.get("max_h") or MAX_HOLD) * H
    end = min(close_t or limit, limit)
    if end > now:
        return None  # still open
    tp, sl = rule.get("tp"), rule.get("sl")
    h = t + H // 2 + H
    while h <= end + H - 1:  # every hourly candle after entry, up to the one holding the close
        c = at(cs, h)
        hit = c and first_hit([c[:2]], entry, d, tp, sl)
        if hit == "sl" and tp and first_hit([c[:2]], entry, d, tp, None) == "tp":  # this hour reached both levels
            hit = first_hit(fine(h) if fine else [], entry, d, tp, sl) or "sl"  # 5-minute order if known, else cautious
        if hit:
            return (tp if hit == "tp" else -sl) - 2 * FEE
        h += H
    last = at(cs, end)
    return (last[2] / entry - 1) * d - 2 * FEE if last else None


def rules(copies):
    """Close rules to try. All of them still close when the trader closes; they differ in what else can close a copy first:
    nothing, a 3:1 take profit / stop loss at a few sizes, or the trader's own habit (their typical win and loss, and
    how long they usually hold)."""
    out = [{"name": "follow trader"}] + [{"name": f"TP {3 * s:.0%} / SL {s:.0%}", "tp": 3 * s, "sl": s} for s in (0.01, 0.02, 0.03, 0.05)]
    rets = [c[4] for c in copies if c[4] is not None]
    wins, losses = [r for r in rets if r > 0], [-r for r in rets if r < 0]
    holds = sorted((c[3] - c[0]) / H for c in copies if c[3])
    if len(wins) >= 5 and len(losses) >= 5:
        tp, sl, mh = statistics.median(wins), statistics.median(losses), max(1, round(holds[int(len(holds) * 0.9)]))
        out.append({"name": f"their habit: TP {tp:.1%} / SL {sl:.1%} / max {mh}h", "tp": tp, "sl": sl, "max_h": mh})
    return out


def learn_exit(copies, candles, now):
    """Pick the close rule that did best on the OLDER half of the trader's copies (rules are built from that half too),
    then test it on the newer half it never saw. Adopted only if it made money there; else None (the default applies).
    Picking the best of 6 rules on all the data would flatter whichever rule got lucky."""
    half = len(copies) // 2
    older, newer = copies[:half], copies[half:]
    replay = lambda rule, cps: [r for t, coin, side, c, _ in cps if (r := simulate(
        candles.get(coin), t, side, c, rule, now, lambda h, coin=coin: five(coin, h))) is not None]
    tried = [(rule, replay(rule, older)) for rule in rules(older)]
    tried = [(statistics.fmean(rs), rule) for rule, rs in tried if len(rs) >= MIN_EXIT // 2]
    if not tried:
        return None
    train, rule = max(tried, key=lambda x: x[0])
    test = replay(rule, newer)
    if train <= 0 or len(test) < MIN_EXIT // 2 or statistics.fmean(test) <= 0:
        return None
    return {**rule, "train": train, "avg": statistics.fmean(test), "win": win(test), "n": len(test)}  # avg/win/n: on unseen copies


def wilson_low(rs, z=2.33):
    """Lower 99% confidence bound of the win rate: 45 wins of 60 -> ~0.60; 9 of 12 -> ~0.44. Small lucky samples fall below 0.5."""
    return win_low(win(rs), len(rs), z)


def win(rs):
    return sum(r > 0 for r in rs) / len(rs)


def good(rs, need):
    return bool(rs) and win(rs) >= need and statistics.fmean(rs) > 0


def judge(graded):
    """graded: [(time, features, return)] oldest first -> this trader's consistency verdict and consistent patterns."""
    rs = [r for _, _, r in graded]
    n = len(rs)
    out = {"n": n, "win": win(rs) if rs else None, "avg": statistics.fmean(rs) if rs else None,
           "worst": None, "consistent": False, "patterns": []}
    if n >= MIN_TRADES:
        parts = [rs[i * n // PARTS:(i + 1) * n // PARTS] for i in range(PARTS)]
        out["worst"] = min(win(p) for p in parts)  # win rate in their weakest period: the consistency number
        out["consistent"] = all(good(p, MIN_WIN) for p in parts) and wilson_low(rs) > 0.5
    by = {}
    for _, f, r in graded:
        for k in ("side", "trend", "coin", "regime", "sentiment"):
            if k in f:  # regime/sentiment can be missing (no BTC prices or mood for that day)
                by.setdefault(f[k], []).append(r)
    for v, prs in by.items():
        half = len(prs) // 2
        if len(prs) >= PAT_MIN and good(prs[:half], PAT_WIN) and good(prs[half:], PAT_WIN) and wilson_low(prs) > 0.5:
            out["patterns"].append({"when": v, "n": len(prs), "win": win(prs), "avg": statistics.fmean(prs),
                                    "worst": min(win(prs[:half]), win(prs[half:]))})
    return out


def scan(addr, prices):
    """Grade every copy of this trader, and learn their close rule. prices: {coin: (fetched at, candles)}, shared, refreshed hourly."""
    now = int(time.time() * 1000)
    os_, pages = opens(addr)
    os_ = [o for o in os_ if o[0] < now - H]  # fast bots' last-minutes trades can't be copied or graded: skip fetching prices for them
    coins = {o[1] for o in os_} | ({"BTC"} if os_ else set())  # BTC: the market's week, for the regime
    stale = {c for c in coins if time.time() - prices.get(c, (0, None))[0] > PRICES_FOR}
    for c, cs in hourly(stale, now - 202 * 24 * H, now).items():
        prices[c] = (time.time(), cs)
    candles = {c: prices[c][1] for c in coins if c in prices}
    closes = {c: {cs[0] + i * H: v for i, v in enumerate(cs[3])} for c, cs in candles.items()}  # the form learn.copy_result reads
    try:
        mood = fear_greed()
    except Exception:
        mood = {}
    graded = []
    for t, coin, side, *_ in os_:
        res = learn.copy_result(closes, coin, t, side)
        if res:
            f = crypto_features(side, coin, addr, res[0])
            f.update(regime_feature(closes.get("BTC", {}), t, side))
            f.update(sentiment_feature(mood, t))
            graded.append((t, f, res[1]))
    return {**judge(graded), "exit": learn_exit(os_, candles, now), "pages": pages}


def load():
    try:
        return json.loads(SCOUT_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {"traders": {}, "candidates": 0}


def save(state):
    tmp = SCOUT_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, separators=(",", ":")))
    os.replace(tmp, SCOUT_FILE)  # the copier reads this file while the scout runs


def run(log=print):
    """Forever: scan never-scanned traders first, then the longest-unscanned. Rereads the leaderboard every 200."""
    state, prices, saved_at = load(), {}, time.time()
    found = {a for a, t in state["traders"].items() if t.get("v") == SCOUT_VERSION and (t.get("consistent") or t.get("patterns"))}
    while True:
        try:
            cands = candidates()
            names = dict(cands)
            state["candidates"], state["error"] = len(names), None
            last = lambda a: (t := state["traders"].get(a, {})).get("scanned", "") if t.get("v") == SCOUT_VERSION else ""
            todo = sorted(names, key=last)[:200]  # never scanned (or under older rules) first, then the oldest scan
            for i, addr in enumerate(todo, 1):
                try:
                    r = scan(addr, prices)
                except Exception:
                    time.sleep(60)  # Hyperliquid unreachable: try this trader again later, don't record an empty result
                    continue
                state["traders"][addr] = {"v": SCOUT_VERSION, "name": names[addr], "scanned": datetime.now().isoformat(timespec="seconds"), **r}
                if (r["consistent"] or r["patterns"]) and addr not in found:
                    found.add(addr)
                    what = (f"won {r['worst']:.0%}+ in every period" if r["consistent"]
                            else "; ".join(f"{p['when']}: {p['win']:.0%} won (n={p['n']})" for p in r["patterns"]))
                    log(f"Scout found a consistent edge: {names[addr] or addr[:6] + '..' + addr[-4:]} ({what})")
                time.sleep(PAGE_PAUSE * r["pages"])  # each older page of fills is heavy on the rate limit
                if time.time() - saved_at > SAVE_EVERY:
                    save(state)
                    saved_at = time.time()
                time.sleep(PAUSE)
            save(state)
        except Exception as e:
            state["error"] = str(e)
            time.sleep(120)


if __name__ == "__main__":
    learn.load_env()
    run()
