"""Scout: consistency verdicts and the watchlist the copier follows."""
import json, tempfile, pathlib, scout, crypto_copier as cc

f = lambda side="short", trend="flat 24h", coin="altcoin": {"side": side, "trend": trend, "coin": coin, "trader": "x"}

# consistent: wins in every third of their history
steady = [(i, f(), 0.01 if i % 4 else -0.005) for i in range(60)]          # 75% won, all the way through
r = scout.judge(steady)
assert r["consistent"] and r["worst"] >= 0.7 and r["n"] == 60, r

# same overall win rate, but all the wins came early: NOT consistent
lucky = [(i, f(), 0.01 if i < 45 else -0.01) for i in range(60)]
r = scout.judge(lucky)
assert not r["consistent"] and r["worst"] == 0.25, r

# not consistent overall, but their shorts win in both halves: a consistent pattern
mixed = [(i, f("short" if i % 2 else "long"), 0.01 if i % 2 else -0.02) for i in range(60)]
r = scout.judge(mixed)
assert not r["consistent"] and [p["when"] for p in r["patterns"]] == ["short"], r

# too few copies: no verdict
assert not scout.judge(steady[:10])["consistent"]

# watchlist: only finds, most consistent first
cc.SCOUT_FILE = pathlib.Path(tempfile.mkdtemp()) / "scout.json"
assert cc.watchlist() == []
cc.SCOUT_FILE.write_text(json.dumps({"traders": {
    "a": {"v": cc.SCOUT_VERSION, "name": "A", **scout.judge(mixed)},
    "b": {"v": cc.SCOUT_VERSION, "name": None, **scout.judge(steady)},
    "old": {"name": "old rules", **scout.judge(mixed)},   # scanned under older rules: ignored
    "c": {"v": cc.SCOUT_VERSION, "name": "C", **scout.judge(lucky)}}}))
assert [a for a, _, _ in cc.watchlist()] == ["a", "b"], cc.watchlist()   # a's short pattern won 100% in both halves; b 75%
assert scout.wilson_low([1] * 9 + [-1] * 3) < 0.5 < scout.wilson_low([1] * 45 + [-1] * 15)   # 9/12 could be luck; 45/60 isn't
print("scout tests pass")

# opens: finds each position's close (to zero, or a flip) and the trader's own return
H = scout.H
fills = [  # newest first, like the API
    {"coin": "ETH", "dir": "Close Long", "startPosition": "2", "sz": "2", "side": "A", "px": "2200", "time": 5 * H},
    {"coin": "SOL", "dir": "Long > Short", "startPosition": "10", "sz": "20", "side": "A", "px": "90", "time": 4 * H},
    {"coin": "ETH", "dir": "Open Long", "startPosition": "1", "sz": "1", "side": "B", "px": "2100", "time": 3 * H},  # add, not new
    {"coin": "SOL", "dir": "Open Long", "startPosition": "0", "sz": "10", "side": "B", "px": "100", "time": 2 * H},
    {"coin": "ETH", "dir": "Open Long", "startPosition": "0", "sz": "1", "side": "B", "px": "2000", "time": 1 * H}]
scout.learn.hl = lambda body: fills
o, pages = scout.opens("x")
assert pages == 0   # under 2000 fills: no older pages needed
r5 = lambda t: t[:4] + (round(t[4], 9),)
assert r5(o[0]) == (H, "ETH", "long", 5 * H, 0.1) and r5(o[1]) == (2 * H, "SOL", "long", 4 * H, -0.1), o
assert o[2][:3] == (4 * H, "SOL", "short") and o[2][3] is None, o   # the flip opened a short, still open
print("close detection tests pass")

# simulate: candles as (start, highs, lows, closes); entry = close of the candle 30 min after t
from array import array
def cs(rows): return (0, array("d", [r[0] for r in rows]), array("d", [r[1] for r in rows]), array("d", [r[2] for r in rows]))
up = cs([(100, 100, 100), (101, 99, 100), (107, 100, 106), (108, 105, 106)])        # long rises 7% in hour 2
fee2 = 2 * scout.FEE
assert abs(scout.simulate(up, 0, "long", 3 * H, {"tp": 0.06, "sl": 0.02}, 10 * H) - (0.06 - fee2)) < 1e-12   # take profit
assert abs(scout.simulate(up, 0, "long", 3 * H, {}, 10 * H) - (0.06 - fee2)) < 1e-12                        # follow: exit at their close
assert abs(scout.simulate(up, 0, "short", 3 * H, {"tp": 0.06, "sl": 0.02}, 10 * H) - (-0.02 - fee2)) < 1e-12  # short stopped out
assert abs(scout.simulate(up, 0, "long", None, {"max_h": 1}, 10 * H) - (0.0 - fee2)) < 1e-12                # time limit, 1h
assert scout.simulate(up, 0, "long", None, {}, 3 * H) is None                                                # still open: unknown
assert scout.simulate(up, 0, "long", 10 * 60_000, {}, 10 * H) is None                                        # closed before a copy could enter
print("replay tests pass")

# learn_exit: picks a rule only if it made money in both halves
spike = cs([(100, 100, 100), (101, 99.5, 100), (107, 100, 101), (101, 99, 100)])   # +7% spike, back to flat by their close
best = scout.learn_exit([(0, "X", "long", 3 * H, 0.0)] * 30, {"X": spike}, 10 * H)
assert best["name"] == "TP 6% / SL 2%", best   # following them earns ~0; a 6% take profit catches the spike
down = cs([(100, 100, 100), (100, 95, 96), (100, 90, 92), (100, 90, 92)])
assert scout.learn_exit([(0, "X", "long", 3 * H, -0.08)] * 30, {"X": down}, 10 * H) is None
print("close rule tests pass")

# the copier applies it: autopilot copies by their learned rule, yours only take profit
from datetime import datetime, timedelta
p = {"auto": True, "exit": {"name": "x", "tp": 0.03, "sl": 0.01}}
assert cc.rule_of(p) == (0.03, 0.01, None) and cc.rule_of({"auto": True}) == (cc.TAKE_PROFIT, cc.STOP_LOSS, None)
assert cc.rule_of({"auto": False, "exit": p["exit"]}) == (cc.TAKE_PROFIT, None, None)
L = {"cash": 1000, "positions": {}, "history": []}
cc.open_pos(L, "A", "long", 100, 100, "x", auto=True, exit={"name": "follow trader"})            # no TP/SL: stays at -5%
cc.open_pos(L, "B", "long", 100, 100, "x", auto=True, exit={"name": "h", "max_h": 2})
L["positions"]["B"]["opened"] = (datetime.now() - timedelta(hours=3)).isoformat(timespec="seconds")  # past its time limit
msgs = []; cc.target_exits(L, {"A": "95", "B": "100"}, msgs.append)
assert list(L["positions"]) == ["A"] and "Time limit" in msgs[0], (msgs, L["positions"])
print("apply close rule tests pass")

# deeper history: walks older pages forward from 200 days ago until it reaches the latest 2000
import time as _t
T0 = int(_t.time() * 1000) - 10 * 24 * H   # the latest 2000 fills start 10 days ago
calls = []
def fake_hl(body):
    calls.append(body)
    if body["type"] == "userFills":
        return [{"time": T0 + i} for i in range(2000)]
    start = body["startTime"]
    return [{"time": start + i} for i in range(2000 if len(calls) < 4 else 5)]   # 2 full pages, then a short one
scout.learn.hl = fake_hl
fs, pages = scout.fills("x")
assert pages == 3 and len(fs) == 6005 and [x["time"] for x in fs] == sorted(x["time"] for x in fs), (pages, len(fs))
assert all(b["endTime"] == T0 - 1 for b in calls[1:])   # never overlaps the latest page
print("history paging tests pass")

# same hour reached both TP and SL: 5-minute bars decide; without them, the cautious stop
both = cs([(100, 100, 100), (107, 97, 100), (100, 100, 100)])
rule = {"tp": 0.06, "sl": 0.02}
assert abs(scout.simulate(both, 0, "long", 2 * H, rule, 10 * H) - (-0.02 - fee2)) < 1e-12
assert abs(scout.simulate(both, 0, "long", 2 * H, rule, 10 * H, lambda h: [(101, 99.5), (107, 99), (100, 97)]) - (0.06 - fee2)) < 1e-12
assert abs(scout.simulate(both, 0, "long", 2 * H, rule, 10 * H, lambda h: [(100, 97), (107, 99)]) - (-0.02 - fee2)) < 1e-12
print("5-minute resolution tests pass")

# holdout: a rule that only worked in the older half is NOT adopted
flip = [(i * 10 * H, "X", "long", i * 10 * H + 3 * H, 0.0) for i in range(40)]
scout.five = lambda coin, h: []
# older half: price spikes +7% after entry (TP 6% wins); newer half: it drops 8% (everything loses)
rows = []
for i in range(40):
    rows += ([(100, 100, 100), (101, 99.5, 100), (107, 100, 101), (101, 99, 100)] if i < 20 else
             [(100, 100, 100), (100, 92, 93), (100, 92, 93), (100, 92, 93)]) + [(100, 100, 100)] * 6
regime = {"X": cs(rows)}
assert scout.learn_exit(flip, regime, 1000 * H) is None
steady_spikes = {"X": cs(([(100, 100, 100), (101, 99.5, 100), (107, 100, 101), (101, 99, 100)] + [(100, 100, 100)] * 6) * 40)}
r = scout.learn_exit(flip, steady_spikes, 1000 * H)
assert r and r["name"] == "TP 6% / SL 2%" and r["n"] == 20, r   # judged on the 20 newer copies it never saw
print("holdout tests pass")

# end to end: an autopilot copy of a scouted trader carries that trader's learned rule, and target_exits uses it
cc.SCOUT_FILE.write_text(json.dumps({"traders": {"T": {"v": cc.SCOUT_VERSION, "exit": {"name": "their habit", "tp": 0.04, "sl": 0.01, "max_h": 5}}}}))
L = {"cash": 1000, "positions": {}, "history": [], "start": 1000}
cc.load_ledger, cc.save_ledger = (lambda: L), (lambda l: None)
cc.positions = lambda a: {"ETH": "long"}
cc.hl = lambda body: {"ETH": "100"}
msg = cc.decide_crypto({"coin": "ETH", "side": "long", "addr": "T"}, auto=True)
assert "close rule: their habit" in msg and L["positions"]["ETH"]["exit"]["tp"] == 0.04, msg
msgs = []; cc.target_exits(L, {"ETH": "104.5"}, msgs.append)
assert "Take profit" in msgs[0] and not L["positions"], msgs     # +4.5% closes under ITS rule (the default needs +6%)
cc.decide_crypto({"coin": "ETH", "side": "long", "addr": "T"})    # your own copy: no learned rule
assert L["positions"]["ETH"]["exit"] is None
print("end-to-end close rule tests pass")

# high winners: ranked by the 99% lower bound of win rate (not raw win rate); must make money; old-rule scans ignored
V = cc.SCOUT_VERSION
cc.SCOUT_FILE.write_text(json.dumps({"traders": {
    "lucky":   {"v": V, "n": 30, "win": 0.80, "avg": 0.01},    # high rate, few trades
    "steady":  {"v": V, "n": 200, "win": 0.75, "avg": 0.01},   # slightly lower rate, far more evidence -> ranks first
    "loser":   {"v": V, "n": 200, "win": 0.90, "avg": -0.01},  # wins often but loses money overall
    "tiny":    {"v": V, "n": 10, "win": 1.0, "avg": 0.05},     # too few
    "oldrule": {"n": 200, "win": 0.95, "avg": 0.02}}}))
assert [a for a, _, _ in cc.high_winners()] == ["steady", "lucky"], cc.high_winners()
assert abs(scout.wilson_low([1] * 45 + [-1] * 15) - cc.win_low(0.75, 60)) < 1e-12
print("high winner tests pass")

# patterns can be market conditions, and trades missing a condition still count for the others
cond = [(i, {**f(), "regime": "BTC 7d: with trade"} if i % 2 else f(), 0.01 if i % 2 else -0.02) for i in range(60)]
r = scout.judge(cond)
assert "BTC 7d: with trade" in [p["when"] for p in r["patterns"]], r["patterns"]
print("condition pattern tests pass")
