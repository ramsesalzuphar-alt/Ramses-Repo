"""Own-trade signals: funding, open-interest surges, news tone; and their practice books (open once, close after the hold)."""
import tempfile, pathlib
from datetime import datetime, timedelta
import signals, shadow, app

# funding: +0.03%/h -> short, -0.03%/h -> long, +0.01%/h -> nothing
ctx = lambda f, oi=1000, px=10: {"funding": str(f), "openInterest": str(oi), "markPx": str(px)}
snaps = []
got = signals.from_market({"AAA": ctx(0.0003), "BBB": ctx(-0.0003), "CCC": ctx(0.0001)}, 0, snaps)
assert sorted(got) == [("Own: long very negative funding", "BBB", "long"), ("Own: short very high funding", "AAA", "short")], got

# open interest: +12% with price +3% an hour later -> long; +12% with price flat -> nothing; +5% -> nothing
snaps = []
signals.from_market({"X": ctx(0, 1000, 10), "Y": ctx(0, 1000, 10), "Z": ctx(0, 1000, 10)}, 0, snaps)
signals.from_market({"X": ctx(0, 1050, 10.1)}, 1800, snaps)   # half an hour: too soon to compare with
got = signals.from_market({"X": ctx(0, 1120, 10.3), "Y": ctx(0, 1120, 10.05), "Z": ctx(0, 1050, 10.5)}, 3700, snaps)
assert got == [("Own: open-interest surge (momentum)", "X", "long")], got
assert all(3700 - t <= 2 * signals.OI_LAG for t, _ in snaps)

# news tone: bullish / bearish per coin named; mixed or toneless headlines trade nothing
assert signals.news_score("Bitcoin surges to record high as ETF inflows jump") == [("BTC", 1)]
assert signals.news_score("Solana DeFi protocol drained in $40M exploit") == [("SOL", -1)]
assert signals.news_score("Ethereum developers meet to discuss roadmap") == []
assert signals.news_score("Nvidia shares rise on earnings") == []          # not a listed coin
got = signals.from_news(["Bitcoin rallies past $120k", "Exchange hack hits bitcoin holders", "XRP soars after court approval"])
assert got == [("Own: news sentiment", "XRP", "long")], got             # BTC: one up, one down -> no trade
print("signal tests pass")

# own-trade books: open once per coin/side at the current price, close only after their hold, graded like the rest
shadow.FILE = pathlib.Path(tempfile.mkdtemp()) / "shadow.json"
app.shadow.FILE = shadow.FILE
t0 = datetime(2026, 11, 18, 12)
shadow.open_signals([("Own: short very high funding", "AAA", "short"), ("Not a book", "AAA", "short")], {"AAA": "10"}, now=t0)
shadow.open_signals([("Own: short very high funding", "AAA", "short")], {"AAA": "12"}, now=t0 + timedelta(minutes=5))  # still hot: not doubled
b = shadow.load()["Own: short very high funding"]
assert list(b["open"]) == ["own:AAA:short"] and b["open"]["own:AAA:short"]["entry"] == 10.0
shadow.update({"AAA": "9"}, holdings={}, now=t0 + timedelta(hours=23))
assert shadow.trades()["Own: short very high funding"] == ([], 1)          # no trader to follow, no early exit
shadow.update({"AAA": "9"}, now=t0 + timedelta(hours=24))
c = shadow.trades()["Own: short very high funding"][0]
assert len(c) == 1 and c[0]["why"] == "24h hold" and abs(c[0]["ret"] - (0.10 - 2 * shadow.cc.FEE)) < 1e-9
board = {x["name"]: x for x in app.practice_board()["strategies"]}
assert board["Own: short very high funding"]["filter"] == "its own trades (no trader copied)" and board["Own: news sentiment"]["exit"] == "hold 24h"
print("own-trade book tests pass")
