"""Practice strategies: who takes which proposal, exits by rule / trader exit / 3-day limit, fees, and grading."""
import tempfile, pathlib
from datetime import datetime, timedelta
import shadow, crypto_copier as cc, app

shadow.FILE = pathlib.Path(tempfile.mkdtemp()) / "shadow.json"   # never the real books
app.shadow.FILE = shadow.FILE
cc.scout_exit = lambda addr: {"name": "habit", "tp": 0.03, "sl": 0.01} if addr == "S" else None
t0 = datetime(2026, 11, 1, 12)
item = lambda addr, coin, side, score, px=100.0: {"kind": "crypto", "addr": addr, "coin": coin, "side": side, "score": score, "price_now": px}
shadow.open_new([item("A", "ETH", "long", 58), item("S", "SOL", "short", 62), item("A", "BTC", "long", 30),
                 {"kind": "pelosi", "score": 99}], scouted={"S"}, now=t0)
b = shadow.load()
assert set(b["Every proposal"]["open"]) == {"A:ETH:long", "S:SOL:short", "A:BTC:long"}
assert set(b["Score 55+"]["open"]) == {"A:ETH:long", "S:SOL:short"} and set(b["Score 60+"]["open"]) == {"S:SOL:short"}
assert set(b["Scout finds only, 55+"]["open"]) == {"S:SOL:short"}
assert b["Score 55+"]["open"]["S:SOL:short"]["exit"]["tp"] == 0.03 and b["Score 55+, fixed exits"]["open"]["S:SOL:short"]["exit"] is None
shadow.open_new([item("A", "ETH", "long", 58, px=200)], scouted=set(), now=t0)   # same trade again: not doubled
assert shadow.load()["Score 55+"]["open"]["A:ETH:long"]["entry"] == 100.0

# SOL falls 3.5%: a short's learned TP (3%) closes it; the fixed-exit book waits for 6%. ETH +1%: nothing yet.
shadow.update({"SOL": "96.5", "ETH": "101", "BTC": "100"}, now=t0 + timedelta(hours=1))
tr = shadow.trades()
c = tr["Score 60+"][0]
assert len(c) == 1 and c[0]["why"] == "take profit" and abs(c[0]["ret"] - (0.035 - 2 * cc.FEE)) < 1e-9 and abs(c[0]["pnl"] - 100 * c[0]["ret"]) < 1e-9
assert tr["Score 55+, fixed exits"] == ([], 2)
# the full check shows A left ETH -> closed; BTC (still held) only by the 3-day limit
shadow.update({"ETH": "101", "BTC": "100", "SOL": "96.5"}, holdings={"A": {"BTC": "long"}}, now=t0 + timedelta(hours=2))
assert [x["why"] for x in shadow.trades()["Every proposal"][0]] == ["take profit", "trader exited"]
shadow.update({"BTC": "100"}, holdings={"A": {"BTC": "long"}}, now=t0 + timedelta(hours=73))
assert shadow.trades()["Every proposal"] == ([*shadow.trades()["Every proposal"][0]], 0) and shadow.trades()["Every proposal"][0][-1]["why"] == "3-day limit"

# graduation lists every practice strategy, graded like the autopilot
app.S["graduates"] = []
app.closed_trades = lambda m: ([], 0)
app.graduation()
names = {r["name"]: r for r in app.S["grad"]}
assert {f"Practice: {k}" for k in shadow.STRATEGIES} <= set(names) and names["Practice: Score 60+"]["n"] == 1, list(names)
print("practice strategy tests pass")
