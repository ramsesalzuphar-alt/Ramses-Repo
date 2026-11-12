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
    "Strong, held 24h (as graded)": (("strong",), "24h", False),  # the model learned 24h holds; copies of fast traders
}                                                                  # close within minutes when they exit: which pays?


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
        for name, (tiers, learned, scout_only) in STRATEGIES.items():
            if x.get("tier") not in tiers or (scout_only and x["addr"] not in scouted) or key in book[name]["open"]:
                continue
            book[name]["open"][key] = {"addr": x["addr"], "coin": x["coin"], "side": x["side"], "entry": x["price_now"],
                                       "opened": now.isoformat(timespec="seconds"), "score": x["score"],
                                       "exit": cc.scout_exit(x["addr"]) if learned is True else None,
                                       "hold": 24 if learned == "24h" else None}
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
            tp, sl, mh = cc.rule_of({"auto": True, "exit": p.get("exit")})
            age = now - datetime.fromisoformat(p["opened"])
            if p.get("hold"):  # held exactly as the model was graded: no rule, no following the trader
                why = "24h hold" if age >= timedelta(hours=p["hold"]) else None
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
