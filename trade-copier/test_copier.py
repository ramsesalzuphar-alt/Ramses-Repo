from copier import parse_ptr, plan_order, trade_key

# Real rows from Pelosi's Jan 2026 filing (pdftotext -layout), including the wrapped-[ST] case
SAMPLE = """
ID       Owner Asset                  Transaction Date  Notification Amount
         SP  Alphabet Inc. - Class A Common          P            01/16/2026 01/16/2026 $500,001 -
             Stock (GOOGL) [ST]
         SP  Amazon.com, Inc. - Common Stock P                    12/30/2025 12/30/2025 $100,001 -
             (AMZN) [OP]
    SP  Apple Inc. - Common Stock (AAPL) S (partial)   12/24/2025 12/24/2025 $5,000,001 -
        [ST]                                                                $25,000,000
    SP      Versant Media Group, Inc. - Class A E         01/02/2026 01/02/2026 $15.00
            Common Stock (VSNT) [ST]
         SP    REOF XXX, LLC [AB]     P  09/08/2026 09/08/2026 $500,001 -
"""
rows = parse_ptr(SAMPLE)
got = [(r["Transaction"], r["Ticker"], r["AssetType"]) for r in rows]
assert got == [("Purchase", "GOOGL", "ST"), ("Purchase", "AMZN", "OP"), ("Sale (partial)", "AAPL", "ST"),
               ("Exchange", "VSNT", "ST"), ("Purchase", "", "AB")], got

acts = [plan_order(r, 500)[1] for r in rows]
assert acts[0][0] == "POST" and acts[0][2]["symbol"] == "GOOGL" and acts[0][2]["notional"] == "500.00"
assert acts[1] is None  # option: skipped
assert acts[2] == ("DELETE", "/v2/positions/AAPL", None)
assert acts[3] is None and acts[4] is None  # exchange, private LLC
assert trade_key(rows[0]) != trade_key({**rows[0], "Transaction": "Sale"})
print("all tests pass")

from datetime import date
from copier import score_pelosi
s, why = score_pelosi({"TransactionDate": "11/01/2026", "Range": "$1,000,001 -", "Transaction": "Purchase"}, date(2026, 11, 8))
assert s == 90, (s, why)
s, why = score_pelosi({"TransactionDate": "08/01/2026", "Range": "$15,001 -", "Transaction": "Sale (partial)"}, date(2026, 11, 8))
assert s == 5, (s, why)
print("score tests pass")

from copier import predict, pelosi_features
from learn import fit, validate
samples = [({"kind": "Purchase", "size": "$1M+"}, 0.05), ({"kind": "Sale", "size": "$1M+"}, -0.02)] * 30  # mixed through time
m = fit(samples)
assert m["base"][0] == 60 and abs(m["base"][2] - 0.5) < 1e-9
buy, why = predict(m, {"kind": "Purchase", "size": "$1M+"})
sell, _ = predict(m, {"kind": "Sale", "size": "$1M+"})
assert buy >= 60 and sell < 45 and "Purchase: avg +5.00%" in why[1], (buy, sell, why)   # profit-first: +3.6% expected vs -0.6%
assert predict(m, {"kind": "never seen"})[0] == 65   # unknown trait -> overall average profit (+1.5% -> 65)
f = pelosi_features({"ReportDate": "1/23/2026", "TransactionDate": "01/16/2026", "Range": "$1,000,001 -", "Transaction": "Purchase"})
assert f == {"kind": "Purchase", "size": "$1M+", "lag": "filed within 14d"}, f
import random; random.seed(1)
v = validate([({"kind": k}, (0.03 if k == "good" else -0.03) + random.uniform(-0.02, 0.02)) for k in random.choices(["good", "bad"], k=200)])
assert v["edge"], v
print("learn tests pass")

# live grading: copy 30 min after the event, hold 24h, minus fees; only scored events count
import learn
H = learn.H
t0 = 100 * H
closes = {"BTC": {t0 - H: 100.0, t0 - 25 * H: 90.0, t0: 100.0, t0 + 24 * H: 110.0}}
before, r = learn.copy_result(closes, "BTC", t0, "long")
assert abs(before - (100 / 90 - 1)) < 1e-9 and abs(r - (0.10 - 2 * learn.FEE)) < 1e-9, (before, r)
assert abs(learn.copy_result(closes, "BTC", t0, "short")[1] - (-0.10 - 2 * learn.FEE)) < 1e-9
assert learn.copy_result(closes, "ETH", t0, "long") is None
import time as _t
old = int(_t.time() * 1000) - 30 * H
V = learn.SCORING
learn.logged_events = lambda: [{"t": old, "coin": "BTC", "side": "long", "score": 70, "v": V}] * 25 + \
                              [{"t": old, "coin": "BTC", "side": "short", "score": 30, "v": V}] * 25 + \
                              [{"t": old, "coin": "BTC", "side": "short", "score": 90}] * 40 + \
                              [{"t": old, "coin": "BTC", "side": "long", "score": None}] * 50   # unscored, or scored the old way: ignored
learn.hourly_closes = lambda coins, since, now: {"BTC": {old - H - old % H: 1, old - 25 * H - old % H: 1,
                                                         (old + H // 2) - (old + H // 2) % H: 100.0,
                                                         (old + H // 2 + 24 * H) - (old + H // 2 + 24 * H) % H: 105.0}}
lv = learn.live_record()
assert lv["graded"] == 50 and lv["n_high"] == 25 and lv["n_low"] == 25 and lv["edge"], lv
print("live record tests pass")

m = {"base": [100, 0.0, 0.5], "features": {"trader": {"t x": [500, 0, 0.9]},
     "move15": {"a": [30, 0.01, 0.7], "b": [5, -0.1, 0.0], "c": [40, -0.01, 0.45]}}}
assert [p["pattern"] for p in learn.top_patterns(m)] == ["a", "c"]   # needs n>=20, skips traders, biggest gap first
print("pattern tests pass")

# per-trader models: A wins when chasing, B wins when fading -> each scored by its own method
from copier import predict_for_trader
def trades(trader, good, bad, n):
    return [({"trader": trader, "range": good}, 0.02), ({"trader": trader, "range": bad}, -0.02)] * n
S = [x for pair in zip(trades("A", "chasing", "fading", 30), trades("B", "fading", "chasing", 30)) for x in pair] \
    + trades("C", "chasing", "fading", 5)
pm = learn.fit_personal(S)
assert set(pm["traders"]) == {"A", "B"}                                     # C has only 10 trades: no personal model
a, why = predict_for_trader(pm, {"trader": "A", "range": "chasing"})
b, _ = predict_for_trader(pm, {"trader": "B", "range": "chasing"})
assert a >= 60 and b < 45 and "personal model" in why[0], (a, b, why)       # same conditions, opposite verdicts
assert predict_for_trader(pm, {"trader": "C", "range": "chasing"})[0] == predict(pm, {"trader": "C", "range": "chasing"})[0]
# pooled model can't tell them apart: chasing is 50/50 overall
assert 40 <= predict(fit_pool := learn.fit([(dict((k, v) for k, v in f.items() if k != "trader"), r) for f, r in S]), {"range": "chasing"})[0] <= 60

# habits: A chases 80% of the time vs 50% overall -> listed; best/worst conditions reported
H2 = ([({"trader": "A", "range": "chasing"}, 0.01)] * 4 + [({"trader": "A", "range": "fading"}, -0.01)]) * 10 + \
     ([({"trader": "B", "range": "fading"}, 0.01)] * 4 + [({"trader": "B", "range": "chasing"}, -0.01)]) * 10
prof = {p["trader"]: p for p in learn.profiles(H2, learn.fit_personal(H2))}
assert prof["A"]["style"] and prof["A"]["style"][0].startswith("chasing (80%"), prof["A"]
assert prof["B"]["style"][0].startswith("fading (80%") and prof["B"]["best"][0]["when"] == "fading", prof["B"]
print("per-trader tests pass")

# every House member: urgency by size, biggest first, auto-sell only when the member you copied from sells
import json as _json, tempfile as _tf, pathlib as _pl, copier as cp
d = _pl.Path(_tf.mkdtemp())
cp.SEEN_FILE, cp.DOCS_FILE, cp.SOURCES_FILE = d / "seen.json", d / "docs.json", d / "src.json"
cp.STOCK_EVENTS, cp.STOCK_HISTORY, cp.MY_ORDERS = d / "events.jsonl", d / "hist.json", d / "orders.json"   # never the real files
cp.load_model = lambda name: None
mk = lambda rep, sym, kind, rng: {"Representative": rep, "Last": rep.split()[-1], "Ticker": sym, "AssetType": "ST",
                                  "Transaction": kind, "TransactionDate": "11/01/2026", "ReportDate": "11/05/2026", "Range": rng}
filings = [mk("Ann Small", "AAA", "Purchase", "$1,001 - $15,000"), mk("Bob Big", "BBB", "Purchase", "$1,000,001 - $5,000,000"),
           mk("Cy Mid", "CCC", "Purchase", "$250,001 - $500,000"),
           mk("Dee Seller", "HELD", "Sale", "$15,001 - $50,000"),   # sells a stock you hold, but you copied it from Eve
           mk("Eve Source", "MINE", "Sale", "$15,001 - $50,000")]    # sells the stock you copied from her
asked = []
cp.fetch_trades = lambda who, skip_docs=None, **k: asked.append(who) or filings
cp.SOURCES_FILE.write_text(_json.dumps({"HELD": "Eve Source", "MINE": "Eve Source"}))
orders = []
cp.alpaca = lambda m, path, body=None: (orders.append((m, path)) or (200, {}))
pending, logs = [], []
cp.check_pelosi(pending, logs.append)
assert asked == [""], asked   # no POLITICIAN set: every member
assert [p["trade"]["Ticker"] for p in pending] == ["BBB", "CCC", "HELD", "AAA"], [p["title"] for p in pending]
assert [p["urgency_label"] for p in pending] == ["URGENT", "HIGH", "LOW", "LOW"]
assert ("DELETE", "/v2/positions/MINE") in orders and ("DELETE", "/v2/positions/HELD") not in orders, orders
assert "MINE" not in _json.loads(cp.SOURCES_FILE.read_text()) and any("Eve Source" in m for m in logs), logs
pending.clear(); cp.check_pelosi(pending, logs.append)
assert pending == []   # seen: not proposed twice
print("congress tests pass")

# member gate: a member whose 20+ graded copies lost isn't proposed (but is still logged); a +0.5%+ one gets +5
cp.SEEN_FILE.write_text("[]"); cp.STOCK_EVENTS.write_text("")
_m = {"base": [100, 0.0, 0.5], "features": {}, "traders": {}, "use_personal": False,
      "records": {"Ann Small": [30, 10, -0.6], "Bob Big": [30, 20, 0.6], "Cy Mid": [5, 0, -1.0]}}
cp.load_model = lambda name: _m
cp.daily_closes = lambda *a, **k: {}
pending.clear(); cp.check_pelosi(pending, logs.append)
got = {p["trade"]["Ticker"]: p for p in pending}
assert set(got) == {"BBB"}, list(got)   # Ann's copies lost; Cy has too few to be proven: only proven members are proposed
assert "beat the market by +2.00%" in got["BBB"]["why"][0] and "Cy Mid" in cp.STOCK_EVENTS.read_text(), got["BBB"]["why"]
assert "Ann Small" in cp.STOCK_EVENTS.read_text()   # still logged, so she can earn her way back
cp.load_model = lambda name: None
print("member gate tests pass")

# trend before filing: stock vs SPY over 20 trading days, side-relative, only days before `day`
days = [f"2026-10-{i:02d}" for i in range(1, 31)]
spy = [(d, 100.0) for d in days]
rising = [(d, 100 * 1.005 ** i) for i, d in enumerate(days)]
assert cp.trend_feature(rising, spy, "2026-10-25", "Purchase") == {"trend": "with 20d trend"}
assert cp.trend_feature(rising, spy, "2026-10-25", "Sale") == {"trend": "against 20d trend"}
assert cp.trend_feature(rising, spy, "2026-10-10", "Purchase") == {}   # not 21 days of history yet
f = cp.stock_features(mk("Bob Big", "BBB", "Purchase", "$1,000,001 - $5,000,000"), {"BBB": rising, "SPY": spy}, "2026-10-25")
assert f["trader"] == "member Bob Big" and f["size"] == "$1M+" and f["trend"] == "with 20d trend", f
print("stock feature tests pass")

# exits: every copy takes profit at +9%; autopilot's also stop at -3% and end after 30 days; results are recorded
from datetime import datetime as _d, timedelta as _td
cp.STOCK_HISTORY.write_text("[]")
old_open = (_d.now() - _td(days=31)).isoformat(timespec="seconds")
cp.SOURCES_FILE.write_text(_json.dumps({
    "TPX": {"rep": "A", "auto": False, "opened": old_open},      # yours, +10%: take profit
    "SLX": {"rep": "A", "auto": True, "opened": _d.now().isoformat(timespec="seconds"), "feats": {"size": "x"}},  # auto, -4%
    "YOURS": {"rep": "A", "auto": False, "opened": old_open},    # yours, -10% and old: no stop, no time limit
    "OLD": {"rep": "A", "auto": True, "opened": old_open}})) # auto, flat, 31 days: time limit
pos = {"TPX": 0.10, "SLX": -0.04, "YOURS": -0.10, "OLD": 0.001}
sold = []
def fake_alpaca(m, path, body=None):
    if m == "GET" and path == "/v2/positions":
        return 200, [{"symbol": k, "unrealized_plpc": str(v), "unrealized_pl": str(v * 1000)} for k, v in pos.items()]
    if m == "DELETE":
        sold.append(path.rsplit("/", 1)[1]); return 200, {}
    return 200, {}
cp.alpaca = fake_alpaca
msgs = []; cp.stock_exits(msgs.append)
assert sorted(sold) == ["OLD", "SLX", "TPX"], (sold, msgs)
assert list(_json.loads(cp.SOURCES_FILE.read_text())) == ["YOURS"]
h = {x["sym"]: x for x in cp.stock_history()}
assert h["SLX"]["auto"] and h["SLX"]["ret"] == -0.04 and h["SLX"]["feats"] == {"size": "x"} and h["SLX"]["pnl"] == -40.0, h
assert not h["TPX"]["auto"] and any("Take profit" in m for m in msgs) and any("Time limit" in m for m in msgs)
print("stock exit tests pass")

# grading a congressional copy: stock minus SPY over 21 trading days, entering the day after; sales flipped
import learn
bars = {"SPY": [(f"d{i:03d}", 100.0) for i in range(40)], "UP": [(f"d{i:03d}", 100.0 + i) for i in range(40)]}
sd = {d: i for i, (d, _) in enumerate(bars["SPY"])}
r = learn.excess_after(bars, bars["SPY"], sd, "UP", "d004", False)   # enters d005 at 105, exits d026 at 126
assert abs(r - (126 / 105 - 1)) < 1e-12 and abs(learn.excess_after(bars, bars["SPY"], sd, "UP", "d004", True) + r) < 1e-12
assert learn.excess_after(bars, bars["SPY"], sd, "UP", "d030", False) is None   # too recent
assert learn.excess_after(bars, bars["SPY"], sd, "GONE", "d004", False) is None
print("stock grading tests pass")

# learn.main: one market failing keeps its previous model; the other's fresh results are still saved
import learn as L2
L2.MODEL_FILE = d / "model.json"
PREV_LIVE = {"graded": 7, "n_high": 0, "n_low": 0, "high": "-", "low": "-", "edge": False}
L2.MODEL_FILE.write_text(_json.dumps({"crypto": {"old": True}, "live": PREV_LIVE}))
L2.load_env = lambda: None
def boom(): raise OSError("connection reset")
samples = [({"kind": "Purchase", "trader": f"member {i % 3}"}, 0.01 * ((i % 5) - 2)) for i in range(120)]
L2.congress_samples, L2.crypto_samples = (lambda: samples), boom
L2.live_stock_record, L2.live_record = (lambda: {"graded": 1, "n_high": 0, "n_low": 0, "high": "-", "low": "-", "edge": False}), boom
L2.main()
saved = _json.loads(L2.MODEL_FILE.read_text())
assert saved["crypto"] == {"old": True} and saved["congress"]["n"] == 120 and saved["live"] == PREV_LIVE, saved.keys()
print("learn resilience tests pass")

# a meaningful edge: high-scored trades must average +0.5% or more, not just "above zero"
assert not learn.has_edge([0.0002] * 30, [-0.0005] * 30)   # the stock model's +0.02% vs -0.05%: noise
assert learn.has_edge([0.006] * 30, [-0.01] * 30) and not learn.has_edge([0.006] * 30, [0.01] * 30)
print("edge threshold tests pass")

# breach watch: the copier's own orders are remembered; any other order in the account is flagged; reset clears it
cp.MY_ORDERS.unlink(missing_ok=True)
account = []
def fake_alpaca2(m, path, body=None):
    if m == "POST" or m == "DELETE":
        o = {"id": f"mine{len(account)}", "side": "buy" if m == "POST" else "sell", "symbol": "AAA", "created_at": "2026-11-10T17:00:00Z"}
        account.append(o); return 200, o
    if path.startswith("/v2/orders"):
        assert "after=" in path; return 200, account
    return 200, {}
cp.alpaca, cp.load_env = fake_alpaca2, (lambda: None)
cp.place("POST", "/v2/orders", {}); cp.place("DELETE", "/v2/positions/AAA")
assert cp.foreign_orders() == []                       # only the copier traded: nothing flagged
account.append({"id": "stranger", "side": "buy", "symbol": "TSLA", "created_at": "2026-11-10T18:05:00Z"})
assert cp.foreign_orders() == ["buy TSLA at 2026-11-10 18:05 UTC"], cp.foreign_orders()
cp.reset_orders(); account[:] = []
assert cp.foreign_orders() == []
print("breach watch tests pass")

# stock market conditions: SPY's 20-day trend relative to the trade, and calm vs volatile
spy_up = [(f"2026-09-{i:02d}", 100 * 1.003 ** i) for i in range(1, 30)]     # steady climb, low swings
assert cp.market_features(spy_up, "2026-09-28", "Purchase") == {"market": "market 20d: with trade", "volatility": "calm market"}
assert cp.market_features(spy_up, "2026-09-28", "Sale")["market"] == "market 20d: against trade"
choppy = [(f"2026-09-{i:02d}", 100 * (1.03 if i % 2 else 0.97)) for i in range(1, 30)]
assert cp.market_features(choppy, "2026-09-28", "Purchase")["volatility"] == "volatile market"
assert cp.market_features(spy_up, "2026-09-10", "Purchase") == {}
assert "volatility" in cp.stock_features(mk("Bob Big", "BBB", "Purchase", "$1,001 -"), {"SPY": spy_up}, "2026-09-28")
print("stock market condition tests pass")

# global influences: each fund's 20-day move -> rising / falling / flat, rates inverted from bond prices, tagged by side
days20 = [f"2026-09-{i:02d}" for i in range(1, 30)]
world = {"UUP": [(d, 100 * 1.002 ** i) for i, d in enumerate(days20)],    # dollar +4%: rising
         "TLT": [(d, 100 * 0.997 ** i) for i, d in enumerate(days20)],    # bonds -6%: rates rising
         "USO": [(d, 100.0) for d in days20]}                             # oil flat; gold etc. missing: left out
g = cp.global_features(world, "2026-09-28", "long")
assert g == {"dollar": "dollar rising · long", "rates": "rates rising · long", "oil": "oil flat · long"}, g
assert cp.global_features(world, "2026-09-10", "sell") == {}             # not 20 trading days of history yet
assert cp.stock_features(mk("Bob Big", "BBB", "Sale", "$1,001 -"), world, "2026-09-28")["dollar"] == "dollar rising · sell"
print("global influence tests pass")

# win chance: from the trader's own model when they have one, else the pooled one
pool = {"base": [100, 0.0, 0.5], "features": {"side": {"long": [100, 0.0, 0.5]}},
        "traders": {"trader A": {"base": [50, 0.01, 0.8], "features": {"side": {"long": [50, 0.01, 0.8]}}}}}
assert abs(cp.win_chance(pool, {"trader": "trader A", "side": "long"}) - 0.8) < 1e-9
assert abs(cp.win_chance(pool, {"trader": "trader B", "side": "long"}) - 0.5) < 1e-9
assert cp.predict(pool, {"side": "long"})[0] == 50      # predict unchanged by the split
print("win chance tests pass")

# member track record: only copies whose month had ended count, so learning never peeks at the future
import learn as L3
rows = [(f"2024-01-{d:02d}", {"trader": "member A"}, 0.02) for d in range(1, 26)] + [("2024-03-01", {"trader": "member A"}, -0.5)]
L3.add_records(rows)
assert rows[0][1]["record"] == "record: under 20 known trades"     # nothing known yet on day 1
assert rows[-1][1]["record"] == "record: high win rate (55%+)"      # by March, all 25 January copies were known winners
assert rows[-1][1]["gross"] == "record: high-grossing (+1%+ avg)"   # and its own -50% isn't counted
assert L3.RECORDS["A"][0] == 26                                     # live scoring sees every graded copy
assert cp.record_feature(None) == {"record": "record: under 20 known trades"}
print("member record tests pass")

# stable traits only: a trait that won early and lost later is dropped, one that held every third is kept
mix = [({"t": "steady"}, 0.02), ({"t": "steady2"}, -0.02)] * 45
flip = [({"t": "flip"}, 0.03 if i < 45 else -0.03) for i in range(90)]
sm = fit([x for pair in zip(mix, flip) for x in pair])
assert "steady" in sm["features"]["t"] and "flip" not in sm["features"]["t"], sm["features"]["t"]
print("stable trait tests pass")

assert predict_for_trader({**pm, "use_personal": False}, {"trader": "A", "range": "chasing"})[0] == predict(pm, {"trader": "A", "range": "chasing"})[0]
print("personal-or-pooled tests pass")

# your own price on a stock buy: a limit order for that many shares
import copier as cp2, tempfile as _tf4, pathlib as _pl4
cp2.SOURCES_FILE = _pl4.Path(_tf4.mkdtemp()) / "sources.json"
sent = []
cp2.alpaca = lambda m, p, b=None: (200, {})
cp2.place = lambda m, p, b=None: (sent.append(b), (200, {"id": "x"}))[1]
it = {"trade": {"Ticker": "DDOG", "Transaction": "Purchase", "AssetType": "ST", "Representative": "A"}}
msg = cp2.decide_pelosi(it, limit=180.5, qty=3)
assert sent[-1] == {"symbol": "DDOG", "qty": "3", "side": "buy", "type": "limit", "limit_price": "180.50", "time_in_force": "day"}, sent
assert "Limit buy 3 DDOG at $180.50 ($541.50)" in msg, msg
cp2.decide_pelosi(it)
assert sent[-1]["type"] == "market"                                          # no price given: as before
print("stock limit order tests pass")
