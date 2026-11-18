"""Own-trade signals: funding, open-interest surges, news tone; and their practice books (open once, close after the hold)."""
import tempfile, pathlib
from datetime import datetime, timedelta
import signals, shadow, app

# funding: +0.03%/h -> short, -0.03%/h -> long, +0.01%/h -> nothing
ctx = lambda f, oi=1000, px=10: {"funding": str(f), "openInterest": str(oi), "markPx": str(px)}
snaps = []
got = signals.from_market({"AAA": ctx(0.0003), "BBB": ctx(-0.0003), "CCC": ctx(0.00005)}, 0, snaps)
F = signals.fund_name
assert sorted(got) == sorted([(F("short", t, h), "AAA", "short") for t in (0.0001, 0.0002) for h in (8, 24)]
                             + [(F("long", t, h), "BBB", "long") for t in (0.0001, 0.0002) for h in (8, 24)]), got   # 0.05%/h variants: not hot enough

# open interest: +12% with price +3% an hour later -> long; +12% with price flat -> nothing; +5% -> nothing
snaps = []
signals.from_market({"X": ctx(0, 1000, 10), "Y": ctx(0, 1000, 10), "Z": ctx(0, 1000, 10)}, 0, snaps)
signals.from_market({"X": ctx(0, 1050, 10.1)}, 1800, snaps)   # half an hour: too soon to compare with
got = signals.from_market({"X": ctx(0, 1120, 10.3), "Y": ctx(0, 1120, 10.05), "Z": ctx(0, 1030, 10.5)}, 3700, snaps)
O = signals.oi_name   # X: +12% open interest and +3% price -> the 5% and 10% variants, each with the move and against it
assert sorted(got) == sorted([(O("with the move", j, h), "X", "long") for j in (0.05, 0.10) for h in (4, 12)]
                             + [(O("against the move", j, h), "X", "short") for j in (0.05, 0.10) for h in (4, 12)]), got
assert all(3700 - t <= 2 * signals.OI_LAG for t, _ in snaps)

# news tone: bullish / bearish per coin named; mixed or toneless headlines trade nothing
assert signals.news_score("Bitcoin surges to record high as ETF inflows jump") == [("BTC", 1)]
assert signals.news_score("Solana DeFi protocol drained in $40M exploit") == [("SOL", -1)]
assert signals.news_score("Ethereum developers meet to discuss roadmap") == []
assert signals.news_score("Nvidia shares rise on earnings") == []          # not a listed coin
got = signals.from_news(["Bitcoin rallies past $120k", "Exchange hack hits bitcoin holders", "XRP soars after court approval"])
assert got == [(signals.news_name(h), "XRP", "long") for h in (6, 24, 72)], got             # BTC: one up, one down -> no trade
print("signal tests pass")

# own-trade books: open once per coin/side at the current price, close only after their hold, graded like the rest
BOOK = signals.fund_name("short", 0.0002, 24)
shadow.FILE = pathlib.Path(tempfile.mkdtemp()) / "shadow.json"
app.shadow.FILE = shadow.FILE
t0 = datetime(2026, 11, 18, 12)
shadow.open_signals([(BOOK, "AAA", "short"), ("Not a book", "AAA", "short")], {"AAA": "10"}, now=t0)
shadow.open_signals([(BOOK, "AAA", "short")], {"AAA": "12"}, now=t0 + timedelta(minutes=5))  # still hot: not doubled
b = shadow.load()[BOOK]
assert list(b["open"]) == ["own:AAA:short"] and b["open"]["own:AAA:short"]["entry"] == 10.0
shadow.update({"AAA": "9"}, holdings={}, now=t0 + timedelta(hours=23))
assert shadow.trades()[BOOK] == ([], 1)          # no trader to follow, no early exit
shadow.update({"AAA": "9"}, now=t0 + timedelta(hours=24))
c = shadow.trades()[BOOK][0]
assert len(c) == 1 and c[0]["why"] == "24h hold" and abs(c[0]["ret"] - (0.10 - 2 * shadow.cc.FEE)) < 1e-9
board = {x["name"]: x for x in app.practice_board()["strategies"]}
assert board[BOOK]["filter"] == "its own trades (no trader copied)" and board[signals.news_name(24)]["exit"] == "hold 24h"
print("own-trade book tests pass")
