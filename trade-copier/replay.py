"""Blind replay: every tournament strategy traded over the last ~32 days of real top-trader positions, so each starts
with a month of history instead of waiting a month. It is blind: each day's trades are scored by a model fitted only
on the 21 days before that day, from trades whose 24h result was already known then. No day is graded by a model that
saw it. Writes replay.json {made, strategies: {name: [closed trades]}}. The app runs it once a day; or: python replay.py

Differences from live practice (approximations): prices are hourly closes; no minute-before patterns (5m candles only
reach ~17 days back); no per-trader models or learned exits (today's scout saw these days, so it isn't blind); 'Scout
finds only' isn't replayed for the same reason.
"""
import json, os, sys, time
from datetime import datetime, timezone
import learn, shadow, crypto_copier as cc
from copier import predict

FILE = cc.HERE / "replay.json"
H = learn.H
DAYS, TRAIN_DAYS, TAIL_H = 32, learn.CRYPTO_DAYS, 50  # entries from 32 days ago up to 50h ago (so 48h holds finish)
MAX_PAGES = 6  # pages of 2000 fills per trader: a very busy bot's month doesn't fit, its latest pages do


def fills_since(addr, since):
    """Every fill of the trader since `since` (ms), oldest first, paging forward 2000 at a time."""
    out, start = [], since
    for _ in range(MAX_PAGES):
        page = learn.hl({"type": "userFillsByTime", "user": addr, "startTime": start, "aggregateByTime": True}) or []
        out += page
        if len(page) < 2000:
            break
        start = page[-1]["time"] + 1
        time.sleep(1)
    return out


def positions_of(addr, fills, keep_from):
    """[(entered ms, addr, coin, side, exited ms or None)]: each new position from keep_from on, with when the trader
    fully closed or flipped it."""
    out, opened, seen = [], {}, set()
    for x in fills:
        coin, d, start, sz = x["coin"], x["dir"], float(x["startPosition"]), float(x["sz"])
        if coin in opened and (d in ("Long > Short", "Short > Long") or (d.startswith("Close") and abs(abs(start) - sz) < 1e-9 * max(1, sz))):
            out[opened.pop(coin)][4] = x["time"]
        if cc.is_new_position(x):
            side = "long" if d in ("Open Long", "Short > Long") else "short"
            key = (coin, side, x["time"] // (H // 2))
            if x["time"] >= keep_from and key not in seen:
                seen.add(key)
                opened[coin] = len(out)
                out.append([x["time"], addr, coin, side, None])
    return [tuple(p) for p in out]


def simulate(px, kind, t, side, exited):
    """(exit ms, return after fees) of copying 30 min after t under exit kind, from hourly closes; None if prices miss."""
    entry_t = t + H // 2
    entry = px(entry_t)
    if not entry or (exited == "?" and kind in (True, False)):  # trader's exit unknown: can't replay following it
        return None
    r = lambda p: (p / entry - 1) * (1 if side == "long" else -1)
    if isinstance(kind, str) and kind != "tp":  # exact hold
        end = entry_t + int(kind[:-1]) * H
        return (end, r(px(end)) - 2 * cc.FEE) if px(end) else None
    limit = 24 if kind == "tp" else shadow.MAX_HOURS
    for h in range(1, limit + 1):
        ts = entry_t + h * H
        p = px(ts)
        if not p:
            return None
        ret = r(p)
        if kind == "tp" and ret >= cc.TAKE_PROFIT:
            return ts, ret - 2 * cc.FEE
        if kind is False and (ret >= cc.TAKE_PROFIT or ret <= -cc.STOP_LOSS):
            return ts, ret - 2 * cc.FEE
        if kind in (True, False) and exited and exited <= ts:
            return ts, ret - 2 * cc.FEE
    return ts, ret - 2 * cc.FEE


def run(n_traders=100, now=None):
    now = now or int(time.time() * 1000)
    first, last = now - DAYS * 24 * H, now - TAIL_H * H
    since = first - TRAIN_DAYS * 24 * H
    opens = []
    addrs = [a for _, a, _ in cc.top_traders(n_traders)]
    for i, addr in enumerate(addrs, 1):
        try:
            opens += positions_of(addr, fills_since(addr, since), since)
        except Exception as e:  # one trader failing just leaves them out
            print(f"  {addr[:8]}: {e}")
        print(f"\r  {i}/{len(addrs)} traders, {len(opens)} positions", end="", flush=True)
        time.sleep(2)
    # positions the app logged live (busy bots' fills only reach back a day or two); their exits weren't logged ('?'),
    # so they count only for hold and quick strategies
    have = {(o[1], o[2], o[3], o[0] // H) for o in opens}
    for e in learn.logged_events():
        t = e.get("entered_at") or e["t"]
        if since <= t and (e["addr"], e["coin"], e["side"], t // H) not in have:
            have.add((e["addr"], e["coin"], e["side"], t // H))
            opens.append((t, e["addr"], e["coin"], e["side"], "?"))
    closes = learn.hourly_closes({o[2] for o in opens} | {"BTC"}, since - 2 * 24 * H, now)
    try:
        fng = cc.fear_greed()
    except Exception:
        fng = {}
    rows = []  # (entered, coin, side, exited, feats, 24h result, px)
    for t, addr, coin, side, exited in sorted(opens, key=lambda o: o[0]):
        res = learn.copy_result(closes, coin, t, side)
        if not res:
            continue
        feats = cc.crypto_features(side, coin, addr, res[0])
        feats.update(cc.regime_feature(closes.get("BTC", {}), t, side))
        feats.update(cc.sentiment_feature(fng, t))
        rows.append((t, coin, side, exited, feats, res[1]))
    print(f"\n  {len(rows)} positions with prices; replaying day by day...", flush=True)
    names = [n for n, s in shadow.STRATEGIES.items() if not s[2]]  # scout-only: today's scout isn't blind to the past
    out = {n: [] for n in names}
    for day in range(first, last, 24 * H):
        train = [(f, r) for t, _, _, _, f, r in rows if day - TRAIN_DAYS * 24 * H <= t and t + 25 * H <= day]
        today = [x for x in rows if day <= x[0] < min(day + 24 * H, last)]
        if len(train) < 200 or not today:
            continue
        m = learn.fit(train)
        weak, strong = learn.score_bars([predict(m, f)[0] for f, _ in train])
        for t, coin, side, exited, feats, _ in today:
            s = predict(m, feats)[0]
            x = {"kind": "crypto", "addr": "", "coin": coin, "side": side, "feats": feats, "score": s,
                 "tier": "strong" if s >= strong else "weak" if s <= weak else "mid"}
            when = datetime.fromtimestamp(t / 1000, timezone.utc)
            px = lambda ts, c=coin: closes.get(c, {}).get(ts - ts % H)
            for n in names:
                if shadow.takes(n, x, set(), when):
                    sim = simulate(px, shadow.STRATEGIES[n][1], t, side, exited)
                    if sim:
                        out[n].append({"time": datetime.fromtimestamp(sim[0] / 1000).isoformat(timespec="seconds"),
                                       "coin": coin, "side": side, "ret": sim[1], "pnl": shadow.SIZE * sim[1]})
    for n in out:
        out[n].sort(key=lambda h: h["time"])
    return {"made": datetime.now().isoformat(timespec="seconds"), "positions": len(rows), "strategies": out}


_cache = {"mtime": None, "data": None}


def load():
    """replay.json, re-read only when it changes (the page and every check ask for it)."""
    try:
        mtime = FILE.stat().st_mtime
        if mtime != _cache["mtime"]:
            _cache.update(mtime=mtime, data=json.loads(FILE.read_text()))
        return _cache["data"]
    except (OSError, json.JSONDecodeError):
        return {"made": "", "strategies": {}}


if __name__ == "__main__":
    data = run(int(sys.argv[1]) if len(sys.argv) > 1 else 100)
    tmp = FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    os.replace(tmp, FILE)
    for n, t in sorted(data["strategies"].items(), key=lambda kv: -sum(h["pnl"] for h in kv[1])):
        print(f"  {n:55s} n={len(t):5d} net ${sum(h['pnl'] for h in t):9.2f}")
