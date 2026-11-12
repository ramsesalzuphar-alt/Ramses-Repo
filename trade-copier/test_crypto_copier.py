from crypto_copier import diff, open_pos, close_pos, equity, FEE

old = {"a": {"BTC": "long", "ETH": "short"}, "b": {"SOL": "long"}}
new = {"a": {"BTC": "short", "DOGE": "long"}, "b": {"SOL": "long"}}
got = sorted(diff(old, new))
assert got == sorted([("closed long", "a", "BTC"), ("closed short", "a", "ETH"),
                      ("opened short", "a", "BTC"), ("opened long", "a", "DOGE")]), got
assert diff(new, new) == []

close = lambda a, b: abs(a - b) < 1e-9
fresh = lambda: {"start": 1000, "cash": 1000, "positions": {}, "history": []}

# long: price +10% -> +$10 minus two fees
L = fresh(); open_pos(L, "BTC", "long", 100, 50000)
assert close(L["cash"], 900 - 100 * FEE)
assert close(equity(L, {"BTC": "55000"}), 1000 - 100 * FEE + 10)
assert close(close_pos(L, "BTC", 55000), 10 - 110 * FEE)
assert close(L["cash"], 1010 - 100 * FEE - 110 * FEE) and L["positions"] == {}

# short: price -10% -> +$10; price +10% -> -$10
S = fresh(); open_pos(S, "ETH", "short", 100, 2000)
assert close(close_pos(S, "ETH", 1800), 10 - 90 * FEE)
S = fresh(); open_pos(S, "ETH", "short", 100, 2000)
assert close(close_pos(S, "ETH", 2200), -10 - 110 * FEE)
assert len(S["history"]) == 2
print("all tests pass")

from crypto_copier import score_crypto
up = {"markPx": "105", "prevDayPx": "100", "funding": "0.00001"}
assert score_crypto("long", ["long", "long", "long"], up, 1)[0] == 50 + 30 + 15          # consensus + trend
assert score_crypto("short", ["short", "long"], up, 1)[0] == 50 - 15 - 15                # opposed + against trend
assert score_crypto("long", ["long"], None, 8)[0] == 30                                   # lone bot trade
pricey = {"markPx": "100", "prevDayPx": "100", "funding": "0.0002"}
assert score_crypto("long", ["long"], pricey, 1)[0] == 40 and score_crypto("short", ["short"], pricey, 1)[0] == 55
print("score tests pass")

from crypto_copier import auto_exits
L = fresh()
open_pos(L, "BTC", "short", 100, 50000, "A")   # copied from A
open_pos(L, "ETH", "long", 100, 2000, "A")     # A flips ETH to short -> must close
open_pos(L, "SOL", "short", 100, 100, "B")     # B still holds -> stays
open_pos(L, "DOGE", "long", 100, 0.1, None)    # unknown source -> never auto-closed
msgs = []
auto_exits(L, {"A": {"ETH": "short"}, "B": {"SOL": "short"}}, {"BTC": "49000", "ETH": "2000", "SOL": "100", "DOGE": "0.1"}, msgs.append)
assert sorted(L["positions"]) == ["DOGE", "SOL"], L["positions"]
assert len(msgs) == 2 and all(m.startswith("Auto-closed") for m in msgs), msgs
auto_exits(L, {"B": {}}, {"SOL": "100"}, msgs.append)                 # B exits SOL later
assert sorted(L["positions"]) == ["DOGE"]
print("auto-exit tests pass")

# exit_check: closes only the copy whose trader left, and saves it
import tempfile, pathlib, crypto_copier as cc
cc.LEDGER_FILE = pathlib.Path(tempfile.mkdtemp()) / "ledger.json"
L = fresh(); open_pos(L, "BTC", "short", 100, 50000, "A"); open_pos(L, "ETH", "long", 100, 2000, "B"); cc.save_ledger(L)
cc.positions = lambda a: {"A": {}, "B": {"ETH": "long"}}[a]          # A exited BTC, B still holds ETH
cc.hl = lambda body: {"BTC": "49000", "ETH": "2000"}
msgs = []; cc.exit_check(msgs.append)
assert list(cc.load_ledger()["positions"]) == ["ETH"] and len(msgs) == 1 and "BTC" in msgs[0], (msgs, cc.load_ledger())
print("exit_check tests pass")

# take profit (+6%) closes any copy; stop loss (-2%) only autopilot's own
L = fresh()
open_pos(L, "A", "long", 100, 100, "x")             # yours, +7%  -> take profit
open_pos(L, "B", "short", 100, 100, "x", auto=True)  # auto, -3%   -> stop loss
open_pos(L, "C", "long", 100, 100, "x")             # yours, -10% -> stays (no stop loss on your copies)
open_pos(L, "D", "long", 100, 100, "x", auto=True)  # auto, +5%   -> stays
msgs = []; cc.target_exits(L, {"A": "107", "B": "103", "C": "90", "D": "105"}, msgs.append)
assert sorted(L["positions"]) == ["C", "D"] and "Take profit" in msgs[0] and "Stop loss" in msgs[1], (msgs, L["positions"])
print("take profit / stop loss tests pass")

# crypto_account: Hyperliquid down -> positions still shown, at entry price, with a warning
cc._last_mids = {}
def down(body): raise RuntimeError("reset")
cc.hl = down
a = cc.crypto_account()
assert a["stale"] and [p["coin"] for p in a["positions"]] == ["ETH"] and a["positions"][0]["pnl"] == 0, a
assert abs(a["equity"] - (cc.load_ledger()["cash"] + 100)) < 1e-9, a   # ETH margin still counted
print("crypto_account offline tests pass")

# entry_times: latest fresh open per coin, ignoring adds and closes
cc.hl = lambda body: [
    {"coin": "SUI", "dir": "Open Short", "startPosition": "0", "time": 1000, "px": "0.80"},
    {"coin": "SUI", "dir": "Open Short", "startPosition": "-50", "time": 2000, "px": "0.81"},   # adding, not new
    {"coin": "SUI", "dir": "Close Short", "startPosition": "-80", "time": 3000, "px": "0.82"},
    {"coin": "SUI", "dir": "Open Short", "startPosition": "0", "time": 4000, "px": "0.83"},    # latest new one
    {"coin": "ETH", "dir": "Long > Short", "startPosition": "2", "time": 1500, "px": "2100"}]  # flip counts
assert cc.entry_times("x") == {"SUI": (4000, 0.83), "ETH": (1500, 2100.0)}, cc.entry_times("x")
print("entry_times tests pass")

# minute_features: only candles closed before t count; side-relative labels
from crypto_copier import minute_features, M5
base = 1_000_000_000
def mk(prices, vols=None):
    return [{"t": base + i * M5, "h": p * 1.001, "l": p * 0.999, "c": p, "v": (vols or [10] * len(prices))[i]} for i, p in enumerate(prices)]
rising = [100 * (1.001 ** i) for i in range(60)]                      # steady climb, ~+1.2%/h
t = base + 48 * M5                                                     # candles 0..47 have closed by t
f = minute_features(mk(rising), t, "long")
assert f["move60"] == "1h before: moving its way" and f["range"].startswith("chasing"), f
g = minute_features(mk(rising), t, "short")
assert g["move60"] == "1h before: moving against" and g["range"].startswith("fading"), g
spike = mk(rising, [10] * 45 + [50] * 3 + [10] * 12)
assert minute_features(spike, t, "long")["volume"] == "volume surge"
assert minute_features(mk(rising), base + 30 * M5, "long") == {}       # not 4h of history yet
# a huge candle AFTER t must not change anything (no peeking ahead)
later = mk(rising); later[48]["c"] = 1e9; later[48]["h"] = 1e9
assert minute_features(later, t, "long") == f
print("minute feature tests pass")

# closing a position records its real return, entry conditions and autopilot flag for learning
L = fresh(); open_pos(L, "BTC", "long", 100, 50000, "A", {"side": "long", "range": "mid 4h range"}, True)
close_pos(L, "BTC", 55000)
h = L["history"][-1]
assert h["auto"] and h["feats"]["range"] == "mid 4h range" and abs(h["ret"] - (10 - 110 * FEE) / 100) < 1e-9, h
print("feedback tests pass")

# positions_full: each holding's side and $ size (shown as the trade's volume); positions() keeps just the sides
cc.hl = lambda body: {"assetPositions": [{"position": {"coin": "BTC", "szi": "-0.5", "positionValue": "45000.0"}},
                                         {"position": {"coin": "ETH", "szi": "10", "positionValue": "20000"}}]}
assert cc.positions_full("x") == {"BTC": ("short", 45000.0), "ETH": ("long", 20000.0)}
import importlib; real = importlib.reload(cc)   # an earlier test replaced positions() with a fake
real.hl = (lambda body: {"assetPositions": [{"position": {"coin": "BTC", "szi": "-0.5", "positionValue": "45000.0"}},
                                                              {"position": {"coin": "ETH", "szi": "10", "positionValue": "20000"}}]})
assert real.positions("x") == {"BTC": "short", "ETH": "long"}
print("position size tests pass")

# account totals: entered = what's in open copies, worth now = that plus their live P&L
import importlib, tempfile as _tf2, pathlib as _pl2
ac = importlib.reload(cc)
ac.LEDGER_FILE = _pl2.Path(_tf2.mkdtemp()) / "ledger.json"
L = fresh(); open_pos(L, "BTC", "long", 100, 50000); open_pos(L, "ETH", "short", 200, 2000)
ac.save_ledger(L)
ac.hl = lambda body: {"BTC": "55000", "ETH": "1900"}     # BTC +10% (+$10), ETH short -5% price (+$10)
a = ac.crypto_account()
assert a["invested"] == 300 and abs(a["value"] - 320) < 1e-9, (a["invested"], a["value"])
print("account total tests pass")

# market conditions: whole-market regime (BTC's week, side-relative) and the real-world mood (Fear & Greed)
mc = importlib.reload(cc)
H1 = mc.HOUR
t0 = 1_000 * H1
btc = {t0 - H1: 110.0, t0 - 7 * 24 * H1: 100.0}                  # BTC +10% over the week
assert mc.regime_feature(btc, t0 + 5, "long") == {"regime": "BTC 7d: with trade"}
assert mc.regime_feature(btc, t0 + 5, "short") == {"regime": "BTC 7d: against trade"}
assert mc.regime_feature({}, t0, "long") == {}
mc.http = lambda m, url, h, body=None: (200, {"data": [{"value": "12", "timestamp": "1794268800"}, {"value": "80", "timestamp": "1794182400"}]})
mc._fng["at"] = 0
fng = mc.fear_greed()
assert fng == {"2026-11-10": 12, "2026-11-09": 80}, fng
assert mc.sentiment_feature(fng, 1794268800 * 1000 + 3600_000) == {"sentiment": "extreme fear"}
assert mc.sentiment_feature(fng, 1794182400 * 1000) == {"sentiment": "extreme greed"}
assert mc.sentiment_feature(fng, 0) == {}
print("market condition tests pass")

# hl: retries rate limits (429) and dropped connections, then gives up
rq = importlib.reload(cc); rq.time.sleep = lambda s: None
calls = []
def flaky(m, url, h, body=None):
    calls.append(1)
    if len(calls) == 1: raise OSError("reset")
    if len(calls) == 2: return 429, None
    return 200, {"ok": True}
rq.http = flaky
assert rq.hl({}) == {"ok": True} and len(calls) == 3
rq.http = lambda m, url, h, body=None: (429, None)
try:
    rq.hl({}); assert False, "should give up"
except RuntimeError as e:
    assert "429" in str(e)
print("request retry tests pass")

# your own price: fills now if the market is already there, otherwise waits and fills (at your price) when reached
import importlib, tempfile as _tf3, pathlib as _pl3
cc = importlib.reload(cc)
cc.LEDGER_FILE = _pl3.Path(_tf3.mkdtemp()) / "ledger.json"
cc.positions = lambda a: {"ETH": "long", "SOL": "short"}
mids = {"ETH": "2000", "SOL": "100"}
cc.hl = lambda body: mids
item = lambda coin, side: {"coin": coin, "side": side, "addr": "0xabc", "feats": None}
msg = cc.decide_crypto(item("ETH", "long"), limit=1900, qty=0.1)            # long below the market: waits
assert msg.startswith("Order placed") and "ETH" in cc.load_ledger()["orders"], msg
assert "already" in cc.decide_crypto(item("ETH", "long"), limit=1900, qty=0.1)   # one order or position per coin
msg = cc.decide_crypto(item("SOL", "short"), limit=95, qty=2)               # short below the market: already there, fills now
assert msg.startswith("Copied") and cc.load_ledger()["positions"]["SOL"]["entry"] == 100, msg
mids["ETH"] = "1890"
log = []; cc.exit_check(log.append)
L = cc.load_ledger()
assert "ETH" in L["positions"] and L["positions"]["ETH"]["entry"] == 1900 and not L["orders"], (log, L)
assert abs(L["positions"]["ETH"]["margin"] - 190) < 1e-9 and "filled" in log[0]
cc.decide_crypto(item("SOL", "short"), limit=1, qty=1)                      # already held
L = cc.load_ledger(); L["orders"]["BTC"] = {"side": "long", "limit": 1, "qty": 1, "source": "0xabc", "feats": None, "placed": "2020-01-01T00:00:00"}
mids["BTC"] = "50000"; log = []; cc.fill_orders(L, mids, log.append)
assert "BTC" not in L["orders"] and "not reached" in log[0]                  # expired after 24h
L["orders"]["BTC"] = {**{"side": "long", "limit": 1, "qty": 1, "source": "0xabc", "feats": None}, "placed": "2099-01-01T00:00:00"}
cc.save_ledger(L)
assert cc.close_manual("BTC") == "Cancelled your BTC order" and "BTC" not in cc.load_ledger()["orders"]
print("your-price order tests pass")

# trader gate: needs 20+ graded copies; losing -> dropped, +0.5%+ avg -> boosted
import crypto_copier as _cc, learn as _ln
assert _cc.track_gate(None) is None and _cc.track_gate({"n": 19, "avg": -0.05, "win": 0.1}) is None
assert _cc.track_gate({"n": 20, "avg": -0.0001, "win": 0.5}) == "drop"
assert _cc.track_gate({"n": 20, "avg": 0.004, "win": 0.6}) is None and _cc.track_gate({"n": 20, "avg": 0.005, "win": 0.6}) == "boost"
H = _ln.H
_closes = {"X": {i * H: 100 + i for i in range(60)}}  # rising every hour
_ev = [{"addr": "A", "coin": "X", "side": "long", "t": 30 * H}, {"addr": "B", "coin": "X", "side": "short", "t": 30 * H},
       {"addr": "B", "coin": "Y", "side": "long", "t": 30 * H}]  # no prices for Y: not graded
_r = _ln.trader_records(_closes, _ev)
assert set(_r) == {"A", "B"} and _r["A"]["n"] == 1 and _r["A"]["win"] == 1 and _r["A"]["avg"] > 0 > _r["B"]["avg"], _r
print("trader gate tests pass")
