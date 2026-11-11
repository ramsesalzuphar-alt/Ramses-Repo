"""Learns which copied trades paid off, from history, and writes model.json for both copiers' scores.
  Congress: every stock trade any House member filed since 2022 -> stock vs S&P 500 (SPY) over the month after
          the filing date, with a personal model for each member with enough trades.
  Crypto: new positions opened by the top 60 Hyperliquid traders, plus every one the app logged ->
          return of copying 30 min later (your check delay) and holding 24h, after fees.
Two honest tests:
  backtest - trained on the oldest 70%, scored on the newest 30% it never saw.
  live     - every trade the app scored, graded 24h later. The model scored these BEFORE knowing the outcome,
             so this is the test that decides whether autopilot may unlock.
Run: python learn.py   (the app also runs it daily; takes ~10 min)
"""
from http.client import HTTPException
import json, os, statistics, sys, time
from datetime import date, datetime, timedelta, timezone
from copier import (http, load_env, fetch_trades, stock_features, record_feature, predict, predict_for_trader, daily_closes, stock_history,
                    global_features, GLOBAL_SYMBOLS,
                    MODEL_FILE, CONGRESS_CACHE, STOCK_EVENTS)
from crypto_copier import (top_traders, crypto_features, is_new_position, candles_5m, minute_features, load_ledger,
                           watchlist, high_winners, fear_greed, sentiment_feature, regime_feature, EVENTS_FILE, FEE)

G, R, B, D, X = "\033[32m", "\033[31m", "\033[1m", "\033[2m", "\033[0m"  # terminal colors
H = 3_600_000  # one hour in ms
STRONG, WEAK, MIN_EACH = 60, 45, 20  # live test: STRONG trades must beat WEAK ones, with at least 20 of each


STABLE_N = 10  # examples a trait needs in every third of history to count


def fit(samples):
    """samples: [(traits, outcome)]. Model = for every trait value: [count, average outcome, share that made money]."""
    def stat(rs):
        return [len(rs), statistics.fmean(rs), sum(r > 0 for r in rs) / len(rs)]
    def raw(ss):
        by = {}
        for f, r in ss:
            for k, v in f.items():
                by.setdefault(k, {}).setdefault(v, []).append(r)
        return {"base": stat([r for _, r in ss]), "features": {k: {v: stat(rs) for v, rs in d.items()} for k, d in by.items()}}
    m = raw(samples)
    # Keep only traits that pushed the same way in each third of history (samples are oldest first). A trait that
    # helped in one stretch and hurt in another is luck or a passing market mood, and it made high scores lose.
    thirds = [raw(samples[i * len(samples) // 3:(i + 1) * len(samples) // 3]) for i in range(3)] if len(samples) >= 30 else []
    for k, vals in m["features"].items():
        for v in list(vals):
            d = vals[v][1] - m["base"][1]
            if not all((s := t["features"].get(k, {}).get(v)) and s[0] >= STABLE_N and (s[1] - t["base"][1]) * d > 0 for t in thirds):
                del vals[v]
    return m


def summ(rs):
    return f"{sum(r > 0 for r in rs) / len(rs):.0%} made money, avg {statistics.fmean(rs):+.2%} (n={len(rs)})" if rs else "none yet"


MIN_EDGE = 0.005  # high-scored trades must average +0.5% or more: a +0.02% "edge" is noise, not something worth trading


def has_edge(hi, lo):
    """Profit first: high-scored trades must make a meaningful profit on average (+0.5%+) AND beat low-scored ones,
    with enough of each. Decides the backtest verdicts and whether an autopilot unlocks."""
    return (len(hi) >= MIN_EACH and len(lo) >= MIN_EACH and statistics.fmean(hi) >= MIN_EDGE
            and statistics.fmean(hi) > statistics.fmean(lo))


MIN_PERSONAL = 40  # trades a trader needs before they get their own model


def fit_personal(samples):
    """Pooled model plus one model per trader with enough history, each learned only from that trader's trades."""
    m, by = fit(samples), {}
    for f, r in samples:
        by.setdefault(f.get("trader"), []).append(({k: v for k, v in f.items() if k != "trader"}, r))
    m["traders"] = {t: fit(ss) for t, ss in by.items() if t and len(ss) >= MIN_PERSONAL}
    return m


def profiles(samples, m, k=10):
    """Each modelled trader's method: conditions they trade in far more than everyone else does,
    plus the conditions where copying them worked best and worst (at least k trades each)."""
    total, overall = len(samples), {}
    for f, _ in samples:
        for key, v in f.items():
            if key != "trader":
                overall[(key, v)] = overall.get((key, v), 0) + 1
    out = []
    for t, tm in m["traders"].items():
        n, mean, win = tm["base"]
        style = []
        for key, vals in tm["features"].items():
            for v, s in vals.items():
                share, usual = s[0] / n, overall[(key, v)] / total
                if share >= 0.3 and share / usual >= 1.3:  # a habit: common for them AND more common than for others
                    style.append((share / usual, f"{v} ({share:.0%} of entries vs {usual:.0%} overall)"))
        cond = sorted(((s[2], v, s[0]) for vals in tm["features"].values() for v, s in vals.items() if s[0] >= k), reverse=True)
        out.append({"trader": t, "n": n, "win": win, "avg": mean,
                    "style": [s for _, s in sorted(style, reverse=True)[:3]],
                    "best": [{"when": v, "win": w, "n": c} for w, v, c in cond[:2]],
                    "worst": [{"when": v, "win": w, "n": c} for w, v, c in cond[-2:][::-1]] if len(cond) > 2 else []})
    return sorted(out, key=lambda p: -p["n"])


def validate(samples, fit_fn=fit, predict_fn=predict):
    """Train on the oldest 70%, then check the newest 30%: do high scores actually do better than low ones?"""
    cut = int(len(samples) * 0.7)
    m, test = fit_fn(samples[:cut]), samples[cut:]
    scored = [(predict_fn(m, f)[0], r) for f, r in test]
    hi, lo = [r for s, r in scored if s >= 55], [r for s, r in scored if s < 45]
    return {"all": summ([r for _, r in scored]), "high": summ(hi), "low": summ(lo), "edge": has_edge(hi, lo),
            "spread": statistics.fmean(hi) - statistics.fmean(lo) if hi and lo else 0.0}  # how well it ranks trades


HOLD_DAYS = 21  # trading days a congressional copy is judged over (about a month)


def excess_after(bars, spy, spy_day, sym, day, sale):
    """The stock's return minus SPY's over HOLD_DAYS trading days, entering the first trading day after `day`.
    A sale 'wins' if the stock then lags. None if prices are missing or it's too recent to know."""
    sb = bars.get(sym)
    if not sb:
        return None  # renamed or delisted ticker
    i = next((k for k, (d, _) in enumerate(sb) if d > day), None)
    if i is None or i + HOLD_DAYS >= len(sb) or sb[i][0] not in spy_day:
        return None
    j = spy_day[sb[i][0]]
    if j + HOLD_DAYS >= len(spy):
        return None
    ex = (sb[i + HOLD_DAYS][1] / sb[i][1] - 1) - (spy[j + HOLD_DAYS][1] / spy[j][1] - 1)
    return -ex if sale else ex


def congress_samples(since_year=2022):
    """Every House member's stock trades since 2022, each graded as a copy. First run downloads ~1500 filings (~20 min),
    later runs only new ones (cached in congress_cache.json)."""
    try:
        cache = json.loads(CONGRESS_CACHE.read_text())
    except (OSError, json.JSONDecodeError):
        cache = {}
    print(f"Congress: reading every member's filings {since_year}-now ({len(cache)} already cached)...", flush=True)
    try:
        trades = fetch_trades(None, years=range(since_year, date.today().year + 1), max_age=10 ** 6, cache=cache)
    finally:
        tmp = CONGRESS_CACHE.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache))
        os.replace(tmp, CONGRESS_CACHE)
    trades = [t for t in trades if t["AssetType"] == "ST" and t["Ticker"] and t["Transaction"] != "Exchange"]
    print(f"  {len(trades)} stock trades by {len({t['Representative'] for t in trades})} members; fetching prices...", flush=True)
    bars = daily_closes({t["Ticker"] for t in trades} | {"SPY"} | GLOBAL_SYMBOLS, f"{since_year - 1}-11-01",
                        (date.today() - timedelta(days=1)).isoformat())
    spy = bars["SPY"]
    spy_day = {d: i for i, (d, _) in enumerate(spy)}
    out = []
    for t in trades:
        filed = datetime.strptime(t["ReportDate"], "%m/%d/%Y").date().isoformat()
        r = excess_after(bars, spy, spy_day, t["Ticker"], filed, t["Transaction"] != "Purchase")
        if r is not None:
            out.append((filed, stock_features(t, bars, filed), r))
    out.sort(key=lambda x: x[0])
    add_records(out)
    own = [(h["feats"], h["ret"]) for h in stock_history() if h.get("feats") and "ret" in h]
    print(f"  {len(out)} graded; + {len(own)} of its own closed stock trades (real wins and losses)", flush=True)
    return [(f, r) for _, f, r in out] + own


RECORDS = {}  # member -> [n, wins, total] of every graded copy, saved in the model for live scoring


def add_records(dated):
    """Give each trade its member's track record AS IT WAS when filed: only copies whose month had already ended
    (filed 35+ days before), so the model never peeks at outcomes it couldn't have known."""
    RECORDS.clear()
    known, j = {}, 0
    for filed, f, r in dated:
        cutoff = (date.fromisoformat(filed) - timedelta(days=35)).isoformat()
        while j < len(dated) and dated[j][0] <= cutoff:
            k = known.setdefault(dated[j][1]["trader"], [0, 0, 0.0])
            k[0], k[1], k[2] = k[0] + 1, k[1] + (dated[j][2] > 0), k[2] + dated[j][2]
            j += 1
        f.update(record_feature(known.get(f["trader"])))
    for _, f, r in dated:
        k = RECORDS.setdefault(f["trader"].replace("member ", ""), [0, 0, 0.0])
        k[0], k[1], k[2] = k[0] + 1, k[1] + (r > 0), k[2] + r


def hl(body):
    err = None
    for wait in (0, 10, 30, 60):  # back off if Hyperliquid rate-limits us or the connection drops (flaky DNS/resets here)
        time.sleep(wait)
        try:
            st, d = http("POST", "https://api.hyperliquid.xyz/info", {}, body)
        except (OSError, HTTPException) as e:  # HTTPException: a reply cut off mid-way
            err = e
            continue
        if st == 200:
            return d
        if st != 429:
            return None
        err = None
    if err:
        raise err
    return None


def hourly_closes(coins, since, now):
    """{coin: {hour start ms: close}} for every coin that has candles."""
    out = {}
    for c in sorted(c for c in coins if ":" not in c and not c.startswith("@")):  # skip spot and side-exchange markets
        out[c] = {k["t"]: float(k["c"]) for k in hl({"type": "candleSnapshot", "req": {
            "coin": c, "interval": "1h", "startTime": since, "endTime": now}}) or []}
        time.sleep(0.5)
    return out


def copy_result(closes, coin, t, side):
    """(24h move before t, return of copying 30 min after t and holding 24h after fees), or None if prices are missing."""
    px = lambda ts: closes.get(coin, {}).get(ts - ts % H)  # close of the hourly candle containing ts
    before, before24 = px(t - H), px(t - 25 * H)  # last finished candle, so no peeking ahead
    entry, exit_ = px(t + H // 2), px(t + H // 2 + 24 * H)
    if not (before and before24 and entry and exit_):
        return None
    return before / before24 - 1, (exit_ / entry - 1) * (1 if side == "long" else -1) - 2 * FEE


def logged_events():
    return [json.loads(l) for l in EVENTS_FILE.read_text().splitlines() if l.strip()] if EVENTS_FILE.exists() else []


def crypto_samples(n_traders=60, per_trader=300):
    now = int(time.time() * 1000)
    since = now - 200 * 24 * H
    addrs = [a for _, a, _ in top_traders(n_traders)]  # highest earners this month
    for a in ([a for _, a, _ in top_traders(20, "allTime")]  # highest earners of all time
              + [a for a, _, _ in high_winners()]           # highest, most reliable win rates the scout has measured
              + [a for a, _, _ in watchlist()]):            # the scout's consistent finds
        if a not in addrs:
            addrs.append(a)
    n_traders = len(addrs)
    print(f"Crypto: reading trade history of {n_traders} traders (top earners this month and all time, top win rates, "
          f"scout finds; about 10-15 min)...", flush=True)
    opens = []  # (time ms, addr, coin, side)
    for i, addr in enumerate(addrs, 1):
        mine, seen = [], set()
        for x in hl({"type": "userFillsByTime", "user": addr, "startTime": since, "aggregateByTime": True}) or []:
            if is_new_position(x):
                side = "long" if x["dir"] in ("Open Long", "Short > Long") else "short"
                key = (x["coin"], side, x["time"] // (H // 2))  # one per half hour, like a 30-min check would see
                if key not in seen:
                    seen.add(key)
                    mine.append((x["time"], addr, x["coin"], side, x["time"]))
        opens += sorted(mine)[-per_trader:]  # cap so one bot doesn't drown out everyone else
        print(f"\r  {i}/{n_traders} traders, {len(opens)} new positions", end="", flush=True)
        time.sleep(5)
    logged = logged_events()
    # (time the outcome is measured from, trader, coin, side, time the trader actually entered)
    opens += [(e["t"], e["addr"], e["coin"], e["side"], e.get("entered_at") or e["t"]) for e in logged]
    coins = {o[2] for o in opens} | {"BTC"}  # BTC: the whole-market regime for every trade
    try:
        fng = fear_greed()
    except Exception:
        fng = {}  # mood unavailable: learn without it this time
    try:  # the world's markets (dollar, rates, oil, gold, world stocks, fear gauge) over the same period, daily
        world = daily_closes(GLOBAL_SYMBOLS, datetime.fromtimestamp(since / 1000).date().isoformat(), date.today().isoformat())
    except Exception:
        world = {}
    print(f"\n  + {len(logged)} positions the app logged; fetching hourly prices...", flush=True)
    closes = hourly_closes(coins, since - 2 * 24 * H, now)
    print(f"  fetching 5-minute candles (last ~17 days) for the minutes-before patterns...", flush=True)
    five = {}
    for c in coins:
        if ":" not in c and not c.startswith("@"):
            try:
                five[c] = candles_5m(c, now - 18 * 24 * H, now)
            except Exception:  # one coin's candles failing just loses its minute patterns
                five[c] = []
            time.sleep(0.5)
    out = []
    for t, addr, coin, side, entered in sorted(opens):
        res = copy_result(closes, coin, t, side)
        if res:
            feats = crypto_features(side, coin, addr, res[0])
            feats.update(minute_features(five.get(coin, []), entered, side))  # {} when older than the 5m candles reach
            feats.update(regime_feature(closes.get("BTC", {}), entered, side))
            feats.update(sentiment_feature(fng, entered))
            feats.update(global_features(world, datetime.fromtimestamp(entered / 1000, timezone.utc).date().isoformat(), side))
            out.append((feats, res[1]))
    # its own closed paper trades, with their REAL results: what it actually won and lost
    own = [(h["feats"], h["ret"]) for h in load_ledger()["history"] if h.get("feats") and "ret" in h]
    print(f"  + {len(own)} of its own closed trades (real wins and losses)", flush=True)
    return out + own


def top_patterns(m, k=20, n=8):
    """The trait values whose win rate differs most from average, with at least k examples. Skips individual traders."""
    _, _, win0 = m["base"]
    rows = [(f, v, s) for f, vals in m["features"].items() if f != "trader" for v, s in vals.items() if s[0] >= k]
    rows.sort(key=lambda r: -abs(r[2][2] - win0))
    return [{"pattern": v, "n": s[0], "win": s[2], "avg": s[1]} for _, v, s in rows[:n]]


def live_stock_record():
    """Grade every congressional trade the app scored, once a month of trading has passed."""
    events = [json.loads(l) for l in STOCK_EVENTS.read_text().splitlines() if l.strip()] if STOCK_EVENTS.exists() else []
    ready = [e for e in events if e.get("score") is not None and e["day"] < (date.today() - timedelta(days=35)).isoformat()]
    if not ready:
        return {"graded": 0, "n_high": 0, "n_low": 0, "high": "none yet", "low": "none yet", "edge": False}
    bars = daily_closes({e["sym"] for e in ready} | {"SPY"}, min(e["day"] for e in ready),
                        (date.today() - timedelta(days=1)).isoformat())
    spy = bars.get("SPY", [])
    spy_day = {d: i for i, (d, _) in enumerate(spy)}
    graded = [(e["score"], r) for e in ready
              if (r := excess_after(bars, spy, spy_day, e["sym"], e["day"], e["kind"] != "Purchase")) is not None]
    hi, lo = [r for s_, r in graded if s_ >= STRONG], [r for s_, r in graded if s_ < WEAK]
    return {"graded": len(graded), "n_high": len(hi), "n_low": len(lo), "high": summ(hi), "low": summ(lo), "edge": has_edge(hi, lo)}


def live_record():
    """Grade every trade the app scored with a learned model, once 24h (+ a little) have passed."""
    now = int(time.time() * 1000)
    ready = [e for e in logged_events() if e.get("score") is not None and e["t"] < now - 26 * H]
    if not ready:
        return {"graded": 0, "n_high": 0, "n_low": 0, "high": "none yet", "low": "none yet", "edge": False}
    closes = hourly_closes({e["coin"] for e in ready}, min(e["t"] for e in ready) - 26 * H, now)
    graded = [(e["score"], res[1]) for e in ready if (res := copy_result(closes, e["coin"], e["t"], e["side"]))]
    hi, lo = [r for s, r in graded if s >= STRONG], [r for s, r in graded if s < WEAK]
    return {"graded": len(graded), "n_high": len(hi), "n_low": len(lo), "high": summ(hi), "low": summ(lo), "edge": has_edge(hi, lo)}


def keep_on_error(fn, previous):
    try:
        return fn()
    except Exception as e:
        print(f"  {R}{fn.__name__} failed ({e}); keeping the previous one{X}", flush=True)
        return previous or {"graded": 0, "n_high": 0, "n_low": 0, "high": "none yet", "low": "none yet", "edge": False}


def main():
    load_env()
    model = {"trained": datetime.now().isoformat(timespec="minutes")}
    try:
        prev = json.loads(MODEL_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        prev = {}
    for name, build in (("congress", congress_samples), ("crypto", crypto_samples)):
        try:
            samples = build()
        except Exception as e:  # one market's data failing (e.g. a dropped connection) mustn't throw away the other's
            print(f"  {R}{name}: couldn't read its data ({e}); keeping its previous model{X}", flush=True)
            if name in prev:
                model[name] = prev[name]
            continue
        if len(samples) < 30:
            print(f"  {R}only {len(samples)} usable trades, not enough to learn from; keeping the fixed rules{X}")
            continue
        v = validate(samples)
        # per-trader models: each crypto trader / each House member, learned only from their own trades
        vp = validate(samples, fit_personal, predict_for_trader)
        model[name] = {**fit_personal(samples), "validation": v, "validation_personal": vp, "n": len(samples),
                       # score with per-trader models only while they rank trades better than the pooled one
                       "use_personal": vp["spread"] > v["spread"]}
        print(f"  scoring with {'per-trader' if model[name]['use_personal'] else 'pooled'} models (better backtest)")
        model[name]["profiles"] = profiles(samples, model[name])
        if name == "congress":
            model[name]["records"] = dict(RECORDS)
        print(f"  per-trader models for {len(model[name]['traders'])} traders. Backtest, per-trader vs pooled:")
        print(f"    per-trader scored 55+: {vp['high']}\n    per-trader under 45:  {vp['low']}")
        print(f"  {G + 'per-trader models show some edge' if vp['edge'] else R + 'per-trader models: no proven edge yet'}{X}")
        for p in model[name]["profiles"]:
            print(f"    {p['trader']}: {p['n']} trades, {p['win']:.0%} won · " + "; ".join(p["style"]))
        model[name]["patterns"] = top_patterns(model[name])
        for p in model[name]["patterns"]:
            print(f"    pattern: {p['pattern']:38} {p['win']:.0%} won, avg {p['avg']:+.2%} (n={p['n']})")
        print(f"\n{B}{name}: learned from {len(samples)} past trades{X}")
        print(f"  Backtest on newest 30% (never seen in training):")
        print(f"    all trades:      {v['all']}")
        print(f"    scored 55+:      {v['high']}")
        print(f"    scored under 45: {v['low']}")
        print(f"  {G + 'some edge' if v['edge'] else R + 'no proven edge yet'}{X}")
    model["live_stocks"] = ls = keep_on_error(live_stock_record, prev.get("live_stocks"))
    print(f"\n{B}Live track record, stocks{X} ({ls['graded']} graded): STRONG {ls['high']} | WEAK {ls['low']}")
    model["live"] = lv = keep_on_error(live_record, prev.get("live"))
    print(f"\n{B}Live track record{X} ({lv['graded']} scored trades graded)")
    print(f"    STRONG ({STRONG}+): {lv['high']}\n    WEAK (<{WEAK}):   {lv['low']}")
    print(f"  {G + 'live edge proven - autopilot can unlock' if lv['edge'] else R + 'no live edge yet - autopilot stays locked'}{X}")
    tmp = MODEL_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(model, indent=1))
    os.replace(tmp, MODEL_FILE)  # the app reads this file while running: swap it in whole, never half-written
    print(f"\n{D}Saved {MODEL_FILE.name}.{X}")


if __name__ == "__main__":
    main()
