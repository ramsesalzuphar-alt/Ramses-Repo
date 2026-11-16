"""Blind replay: positions and the trader's exits from fills, exit simulation per strategy kind, and the fast track."""
import replay, app, crypto_copier as cc
H = replay.H
f = lambda t, coin, d, start, sz=1.0: {"time": t, "coin": coin, "dir": d, "startPosition": str(start), "sz": str(sz)}
fills = [f(1 * H, "ETH", "Open Long", 0), f(2 * H, "ETH", "Close Long", 1, 0.5), f(3 * H, "ETH", "Close Long", 0.5, 0.5),
         f(4 * H, "BTC", "Open Short", 0), f(6 * H, "BTC", "Short > Long", -1, 2), f(0, "SOL", "Open Long", 0)]
ps = replay.positions_of("A", sorted(fills, key=lambda x: x["time"]), keep_from=H)
assert ps == [(1 * H, "A", "ETH", "long", 3 * H), (4 * H, "A", "BTC", "short", 6 * H), (6 * H, "A", "BTC", "long", None)], ps

# prices: entry 100 at t+30min; +7% at hour 2, then back to +1%
path = {0: 100, 1: 103, 2: 107, 3: 101}
px = lambda ts: {H * k: v for k, v in path.items()}.get(ts - ts % H) if ts - ts % H <= 3 * H else 101
t = -H // 2   # so the entry candle is hour 0
assert abs(replay.simulate(px, "tp", t, "long", None)[1] - (0.07 - 2 * cc.FEE)) < 1e-9        # quick: out at the TP
assert abs(replay.simulate(px, "12h", t, "long", None)[1] - (0.01 - 2 * cc.FEE)) < 1e-9       # held 12h
assert abs(replay.simulate(px, True, t, "long", 1 * H)[1] - (0.03 - 2 * cc.FEE)) < 1e-9       # followed the trader out
assert replay.simulate(px, True, t, "long", "?") is None                                       # exit unknown: not replayed
assert abs(replay.simulate(px, False, t, "short", None)[1] - (-0.03 - 2 * cc.FEE)) < 1e-9     # fixed exits: short stopped out
print("replay tests pass")

# fast track: blind replay (from before live practice began) + live practice, as one record, pass every test,
# and live confirms for a week -> promoted to the paper autopilot
past = [{"time": f"2026-10-{d:02d}T12:00:00", "ret": 0.01, "pnl": 1.0} for d in range(1, 31) for _ in range(2)]
now = lambda n, d0=1, ret=0.01: [{"time": f"2026-11-{d0 + i % 8:02d}T13:00:00", "ret": ret, "pnl": 100 * ret} for i in range(n)]
replay.load = lambda: {"made": "x", "strategies": {"S": past}}
assert app.fast_track("S", sorted(now(25), key=lambda h: h["time"]))["fast"]
assert not app.fast_track("S", now(10))["fast"]                       # too few live trades
assert not app.fast_track("S", now(25, ret=0.001))["fast"]            # live average too small
assert not app.fast_track("T", now(25))["fast"]                       # never replayed
replay.load = lambda: {"made": "x", "strategies": {"S": past[-10:]}}  # replay covers only the last days of October
assert not app.fast_track("S", sorted(now(25), key=lambda h: h["time"]))["fast"]   # record spans < 30 days
print("fast track tests pass")
