"""Practice strategies: several versions of the crypto autopilot run side by side on every crypto proposal, each in
its own practice book ($100 a trade, real prices and fees, no account money). Each builds its own 30-day track record
at once, graded like the autopilot (app.grade), so you learn which rules work without testing them one after another.
Every version closes when the trader closes, or by its exit rule. Writes shadow.json.
"""
import json, os
from datetime import datetime, timedelta
import crypto_copier as cc

FILE = cc.HERE / "shadow.json"
SIZE = 100.0
MAX_HOURS = 72  # shortcut: a trader who leaves the watched list can't be seen exiting; close after 3 days instead
STRATEGIES = {  # name -> (tiers taken: each model's top 20% = strong, bottom 20% = weak; learned exits?; scout finds only?)
    "Every proposal": (("strong", "mid", "weak", None), True, False),
    "Strong (the autopilot's rule)": (("strong",), True, False),
    "Not weak (top 80%)": (("strong", "mid"), True, False),
    "Strong, fixed exits": (("strong",), False, False),
    "Scout finds only, strong": (("strong",), True, True),
    "Strong, held 24h (as graded)": (("strong",), "24h", False),
    "Strong, weekdays only": (("strong",), True, False, "weekdays"),  # backtest: weekend copies were weaker every period  # the model learned 24h holds; copies of fast traders
}                                                                  # close within minutes when they exit: which pays?
# the tournament: every mix of which picks, how long to hold, and the market's direction, practiced side by side;
# one that passes every graduation test is promoted to the crypto autopilot (app.graduation)
for _t, _tn in ((("strong",), "strong"), (("strong", "mid"), "not weak")):
    for _e, _en in ((True, "follow trader"), ("12h", "12h"), ("24h", "24h"), ("48h", "48h"), ("tp", "quick take profit")):
        for _d, _dn in ((None, "any market"), ("with", "with BTC's week"), ("against", "against BTC's week")):
            if (_t, _e, False, *([_d] if _d else [])) not in STRATEGIES.values():
                STRATEGIES[f"Tourney: {_tn}, {_en}, {_dn}"] = (_t, _e, False, *([_d] if _d else []))


def hold_of(name):
    """Hours a strategy holds each copy exactly (no exit rule, no following the trader), or None."""
    e = STRATEGIES[name][1]
    return 24 if e == "tp" else int(e[:-1]) if isinstance(e, str) else None


def quick(name):
    """Quick trades: closed the minute the take profit (the trader's learned one, else the default) is met, any age;
    still out after 24h if it never is."""
    return STRATEGIES[name][1] == "tp"


def takes(name, x, scouted, now):
    """Would this strategy copy proposal x?"""
    tiers, _, scout_only, *only = STRATEGIES[name]
    regime = (x.get("feats") or {}).get("regime", "")
    return (x.get("tier") in tiers and not (scout_only and x["addr"] not in scouted)
            and not (only == ["weekdays"] and now.weekday() >= 5)
            and not (only == ["with"] and "with" not in regime) and not (only == ["against"] and "against" not in regime))


def load():
    try:
        book = json.loads(FILE.read_text())
    except (OSError, json.JSONDecodeError):
        book = {}
    return {name: book.get(name) or {"open": {}, "closed": []} for name in STRATEGIES}


def save(book):
    tmp = FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(book))
    os.replace(tmp, FILE)


def open_new(items, scouted, now=None):
    """Each strategy 'copies' the new crypto proposals it would take, at the price when they were spotted."""
    now = now or datetime.now()
    book = load()
    for x in items:
        if x.get("kind") != "crypto" or not x.get("price_now"):
            continue
        key = f"{x['addr']}:{x['coin']}:{x['side']}"
        for name, (_, learned, *_) in STRATEGIES.items():
            if not takes(name, x, scouted, now) or key in book[name]["open"]:
                continue
            book[name]["open"][key] = {"addr": x["addr"], "coin": x["coin"], "side": x["side"], "entry": x["price_now"],
                                       "opened": now.isoformat(timespec="seconds"), "score": x["score"],
                                       "exit": cc.scout_exit(x["addr"]) if learned in (True, "tp") else None, "quick": quick(name),
                                       "hold": hold_of(name)}
    save(book)


def update(mids, holdings=None, now=None):
    """Close practice copies that hit their rule (the autopilot's: learned, else TP 6% / SL 2%), whose trader exited
    (holdings = {addr: {coin: side}}, from the full check), or that are past MAX_HOURS."""
    now = now or datetime.now()
    book = load()
    for b in book.values():
        for key, p in list(b["open"].items()):
            if p["coin"] not in mids:
                continue
            px = float(mids[p["coin"]])
            r = (px / p["entry"] - 1) * (1 if p["side"] == "long" else -1)
            tp, sl, mh = cc.rule_of({"auto": True, "exit": p.get("exit")})  # quick books: TP only, no hold (that's p's)
            age = now - datetime.fromisoformat(p["opened"])
            if p.get("hold"):  # held exactly as the model was graded: no rule, no following the trader
                why = ("quick take profit" if p.get("quick") and tp and r >= tp else
                       f"{p['hold']}h hold" if age >= timedelta(hours=p["hold"]) else None)
            else:
                why = ("take profit" if tp and r >= tp else "stop loss" if sl and r <= -sl else
                       "time limit" if mh and age >= timedelta(hours=mh) else
                       "trader exited" if holdings and p["addr"] in holdings and holdings[p["addr"]].get(p["coin"]) != p["side"] else
                       "3-day limit" if age >= timedelta(hours=MAX_HOURS) else None)
            if why:
                ret = r - 2 * cc.FEE
                b["closed"].append({"time": now.isoformat(timespec="seconds"), "coin": p["coin"], "side": p["side"],
                                    "addr": p["addr"], "ret": ret, "pnl": SIZE * ret, "why": why})
                del b["open"][key]
    save(book)


def trades():
    """{strategy: closed practice trades, oldest first}, plus how many each has open."""
    book = load()
    return {name: (b["closed"], len(b["open"])) for name, b in book.items()}
