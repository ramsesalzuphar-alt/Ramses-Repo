"""Autopilot per market: lock, experiment mode, profit-ranked copying, loss guard (pause -> relearn -> resume; 3 trips -> stop)."""
import tempfile, pathlib, app, copier, crypto_copier
app.APP_FILE = pathlib.Path(tempfile.mkdtemp()) / "state.json"   # never touch the running app's real state
copied, bought, retrains = [], [], []
stakes = []
crypto_copier.decide_crypto = lambda item, auto=False, stake=1: copied.append((item["coin"], auto)) or stakes.append(stake) or f"Copied {item['coin']}"
copier.decide_pelosi = lambda item, auto=False, stake=1: bought.append((item["title"], auto)) or stakes.append(stake) or f"Bought {item['title']}"
app.threading.Thread = lambda target, daemon: type("T", (), {"start": lambda self: retrains.append(target)})()
ledger = {"start": 10000, "cash": 10000, "positions": {}, "history": []}
crypto_copier.load_ledger = lambda: ledger
stock_hist = []
copier.stock_history = lambda: stock_hist
copier.SOURCES_FILE = pathlib.Path(tempfile.mkdtemp()) / "src.json"

from datetime import datetime as _dt
class Clock:  # controllable clock: guard windows are judged by time
    t = "2099-01-01T00:00:00"
    @classmethod
    def now(cls): return _dt.fromisoformat(cls.t)
    fromisoformat = staticmethod(_dt.fromisoformat)
    fromtimestamp = staticmethod(_dt.fromtimestamp)
app.datetime = Clock

rule = {"strong": 60, "weak": 45, "min_each": 20}
app.S.update(pending=[], log=[], ap={m: dict(app.NEW_AP, guard_trips=[]) for m in app.MARKETS})  # not the real app's saved state
app.S["model"] = {"trained": "2026-11-09T10:00", "rule": rule, "crypto": {"live": {"edge": False}}, "stocks": {"live": {"edge": False}}}
A = app.ap("crypto")

# locked: plain "on" refused; the user's experiment is allowed and labelled
assert "locked" in app.set_autopilot(True) and A["on"] is False
app.set_autopilot(True, force=True)
assert A["on"] and A["forced"] and "EXPERIMENT" in app.S["log"][0]["msg"]
assert not app.ap("stocks")["on"]   # markets are independent

# copies only STRONG crypto, most profitable first, max 5, marked as autopilot trades
app.S["pending"] = [{"kind": "crypto", "coin": f"C{i}", "title": f"C{i}", "score": 61 + i} for i in range(7)] + \
                   [{"kind": "crypto", "coin": "WEAK1", "title": "WEAK1", "score": 40},
                    {"kind": "pelosi", "coin": "NVDA", "title": "NVDA", "side": "buy", "score": 90}]
app.run_autopilot("crypto")
assert copied == [(f"C{i}", True) for i in (6, 5, 4, 3, 2)], copied
assert {x["coin"] for x in app.S["pending"]} == {"C0", "C1", "WEAK1", "NVDA"}

# loss guard: 10 closed autopilot trades losing $15 total -> pause and relearn
def close_trades(n, pnl, t):
    ledger["history"] += [{"time": t, "action": "close short", "auto": True, "pnl": pnl, "ret": pnl / 100, "feats": {}}] * n
close_trades(9, -1.5, "2099-01-01T00:00:01")
app.guard("crypto"); assert A["on"]                                  # only 9: not judged yet
close_trades(1, -1.5, "2099-01-01T00:00:01")
Clock.t = "2099-01-01T00:00:02"
app.guard("crypto")
assert not A["on"] and A["resume"] and retrains == [app.retrain], app.S["log"][0]
assert app.S["auto"]["crypto"]["recent_pnl"] == -15.0

# relearning resumes it (experiment still on), and the old losses no longer count against it
app.learn.main = lambda: None
app.model_summary = lambda: app.S["model"]
app.retrain()
assert A["on"] and "resumed" in app.S["log"][0]["msg"]
app.guard("crypto"); assert A["on"] and app.S["auto"]["crypto"]["recent_n"] == 0   # fresh window after the trip

# winning trades don't trip it
close_trades(10, +2.0, "2099-01-02T00:00:00"); app.guard("crypto"); assert A["on"]

# three trips in 24h -> stop for good and wait for the user
A["guard_trips"] = [__import__("time").time()] * 2
close_trades(10, -2.0, "2099-01-03T00:00:00"); app.guard("crypto")
assert not A["on"] and not A["forced"] and "STOPPED" in app.S["log"][0]["msg"]

# a proven (unlocked) model doesn't need force, and isn't marked as an experiment
app.S["model"]["crypto"]["live"]["edge"] = True
app.set_autopilot(True); assert A["on"] and not A["forced"]
print("crypto autopilot tests pass")

# stocks: its own switch; buys STRONG congressional purchases only (no sells, no crypto)
B = app.ap("stocks")
assert "locked" in app.set_autopilot(True, m="stocks")
app.set_autopilot(True, force=True, m="stocks"); assert B["on"] and B["forced"]
app.S["pending"] = [{"kind": "pelosi", "title": "BUY1", "side": "buy", "score": 70}, {"kind": "pelosi", "title": "BUY2", "side": "buy", "score": 80},
                    {"kind": "pelosi", "title": "SELL", "side": "sell", "score": 95}, {"kind": "pelosi", "title": "MEH", "side": "buy", "score": 50},
                    {"kind": "crypto", "coin": "X", "title": "X", "score": 99}]
copied.clear(); A["on"] = False
app.run_autopilot("stocks")
assert bought == [("BUY2", True), ("BUY1", True)] and copied == [], (bought, copied)
# stock loss guard: its own history and its own (bigger) limit
stock_hist += [{"time": "2099-01-04T00:00:00", "auto": True, "pnl": -15.0, "ret": -0.015}] * 10
app.guard("stocks"); assert not B["on"] and B["resume"] and app.S["auto"]["stocks"]["recent_pnl"] == -150.0
assert "Stocks loss guard" in app.S["log"][0]["msg"]
print("stock autopilot tests pass")

# earned stake: x1 until its latest blocks of 10 closed trades each made money; x2/x3/x4 for 2/3/4+ blocks; a losing latest block -> x1
def hist(*blocks):  # each block: total P&L of 10 trades, oldest block first
    ledger["history"] = [{"time": "2099-02-01", "auto": True, "pnl": b / 10} for b in blocks for _ in range(10)]
hist(+5); assert app.stake("crypto") == 1                      # one good block isn't "consistent"
hist(-9, +5, +5); assert app.stake("crypto") == 2               # latest 2 blocks profitable
hist(+5, +5, +5, +5, +5, +5); assert app.stake("crypto") == 4   # capped at x4
hist(+5, +5, +5, -1); assert app.stake("crypto") == 1           # latest block lost: back to x1
ledger["history"] += [{"time": "2099-02-02", "auto": True, "pnl": 0.1}] * 3; assert app.stake("crypto") == 1   # partial block doesn't count
hist(+5, +5, +5)
app.S["pending"] = [{"kind": "crypto", "coin": "Z", "title": "Z", "score": 90}]
A["on"], A["forced"] = True, True; stakes.clear()
app.run_autopilot("crypto")
assert stakes == [3] and "stake x3" in app.S["log"][0]["msg"], (stakes, app.S["log"][0])
assert app.auto_stats("crypto")["size"] == 300.0
print("stake tests pass")

# expire: crypto trades older than 10 min disappear; congressional trades (no seen_at) stay
import time as _time
now_ms = _time.time() * 1000
app.S["pending"] = [{"title": "old", "seen_at": now_ms - 11 * 60_000}, {"title": "fresh", "seen_at": now_ms - 60_000},
                    {"title": "pelosi"}]
app.expire()
assert [x["title"] for x in app.S["pending"]] == ["fresh", "pelosi"] and "Expired 1" in app.S["log"][0]["msg"], app.S["pending"]
print("expire tests pass")
