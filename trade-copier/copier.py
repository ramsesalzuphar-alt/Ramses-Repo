"""Pelosi copier logic: reads her official House trade filings, scores each new trade, and places Alpaca PAPER orders.
Used by app.py (the button UI). Full sales of stocks you hold are sold automatically; buys wait for your approval.
Keys are read from .mcp.json (alpaca env block). Optional settings in .env: DOLLARS_PER_TRADE, POLITICIAN (one member's last name; default: every House member), MAX_FILING_AGE_DAYS.
"""
import io, json, os, re, shutil, subprocess, tempfile, urllib.request, urllib.error, zipfile
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).parent
SEEN_FILE = HERE / "seen.json"
DOCS_FILE = HERE / "seen_docs.json"  # filing PDFs already read, so each is downloaded once
SOURCES_FILE = HERE / "stock_sources.json"  # {symbol: {rep, auto, opened, feats}}: who you copied each stock from
STOCK_HISTORY = HERE / "stock_history.json"  # closed stock copies with their real results: learned from, and the loss guard
MY_ORDERS = HERE / "my_orders.json"  # IDs of every order the copier placed: any other order means someone else has your keys
STOCK_EVENTS = HERE / "stock_events.jsonl"  # every congressional trade scored, copied or not: graded ~30 days later
CONGRESS_CACHE = HERE / "congress_cache.json"  # parsed filings by ID, so learning reads each PDF once
STOCK_STOP, STOCK_MAX_DAYS = 0.03, 30  # autopilot stock copies: stop loss -3%, close after 30 days (the learned horizon)
STOCK_TP = 3 * STOCK_STOP  # 3:1 take profit, +9%: applies to every stock copy, yours too
MODEL_FILE = HERE / "model.json"
ALPACA = "https://paper-api.alpaca.markets"  # paper only on purpose; never point this at live without a code review
CLERK = "https://disclosures-clerk.house.gov/public_disc"
PDFTOTEXT = shutil.which("pdftotext") or r"C:\Program Files\Git\mingw64\bin\pdftotext.exe"


def load_env():
    env = HERE / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    mcp = HERE / ".mcp.json"
    if mcp.exists():
        e = json.loads(mcp.read_text())["mcpServers"]["alpaca"]["env"]
        os.environ.setdefault("ALPACA_KEY", e.get("ALPACA_API_KEY", ""))
        os.environ.setdefault("ALPACA_SECRET", e.get("ALPACA_SECRET_KEY", ""))


def http(method, url, headers, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={**headers, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read()
            return r.status, json.loads(raw) if raw else None
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")


def alpaca(method, path, body=None):
    h = {"APCA-API-KEY-ID": os.environ["ALPACA_KEY"], "APCA-API-SECRET-KEY": os.environ["ALPACA_SECRET"]}
    return http(method, ALPACA + path, h, body)


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "trade-copier"}), timeout=60) as r:
        return r.read()


def verdict(score, strong=70):
    """Learned scores are win chances, so callers pass strong=60."""
    return "STRONG" if score >= strong else "MIXED" if score >= 45 else "WEAK"


def load_model(name):
    """The section of model.json that learn.py wrote for 'pelosi' or 'crypto', or None if not trained yet."""
    try:
        return json.loads(MODEL_FILE.read_text())[name]
    except (FileNotFoundError, KeyError):
        return None


def predict(m, feats, k=20):
    """Profit-first score (0-100) and reasons, from how past trades with the same traits turned out.
    Score = 50 + expected profit after fees x 1000, so 60 = +1% expected, 45 = -0.5%, 100 = +5% or more.
    Starts at the overall average and nudges it per trait; k shrinks traits with few examples toward the average."""
    _, mean0, win0 = m["base"]
    win, mean, parts = win0, mean0, []
    for f, v in feats.items():
        s = m["features"].get(f, {}).get(v)
        if not s:
            continue
        n, mu, w = s
        wt = n / (n + k)
        win += wt * (w - win0)
        mean += wt * (mu - mean0)
        parts.append((abs(wt * (mu - mean0)), f"{v}: avg {mu:+.2%}, {w:.0%} won (n={n})"))  # ranked by profit impact
    win = min(0.99, max(0.01, win))
    why = [f"similar past trades: expected {mean:+.2%} after fees, {win:.0%} won"] + [p[1] for p in sorted(parts, reverse=True)[:3]]
    return round(min(100, max(0, 50 + mean * 1000))), why


def predict_for_trader(m, feats):
    """Score with the trader's own model (learned only from their trades) when they have one, else the pooled model."""
    own = (m.get("traders") or {}).get(feats.get("trader"))
    if not own:
        return predict(m, feats)
    score, why = predict(own, {k: v for k, v in feats.items() if k != "trader"})
    why[0] = why[0].replace("similar past trades", f"this trader's similar past trades (personal model, n={own['base'][0]})")
    return score, why


ROW = re.compile(r"\s(P|S|S \(partial\)|E)\s+(\d\d/\d\d/\d{4})\s+(\d\d/\d\d/\d{4})\s+(\$[\d,]+(?: -)?)")
TICKER, ATYPE = re.compile(r"\(([A-Z][A-Z.]{0,5})\)"), re.compile(r"\[([A-Z]{2})\]")  # separate: the PDF often wraps [ST] onto the next line
KINDS = {"P": "Purchase", "S": "Sale", "S (partial)": "Sale (partial)", "E": "Exchange"}


def parse_ptr(text):
    """Turn pdftotext -layout output of one House PTR into trade dicts. Each row line starts a record; wrapped lines follow it."""
    lines, trades, cur = text.splitlines(), [], None
    for line in lines:
        m = ROW.search(line)
        if m:
            cur = {"Transaction": KINDS[m[1]], "TransactionDate": m[2], "Range": m[4], "_text": line}
            trades.append(cur)
        elif cur is not None and line.strip():
            cur["_text"] += " " + line.strip()
    for t in trades:
        text = t.pop("_text")
        tk, ty = TICKER.search(text), ATYPE.search(text)
        t["Ticker"], t["AssetType"] = (tk[1] if tk else ""), (ty[1] if ty else "")
    return trades


def fetch_trades(who, years=None, max_age=None, skip_docs=None, cache=None):
    """Stock trades from House periodic transaction reports. who = a last name, or None/"" for every member.
    skip_docs: a set of filing IDs already read; they're skipped and new ones are added to it.
    cache: {filing ID: parsed rows}, read and filled in, so a filing's PDF is downloaded only once."""
    if max_age is None:
        max_age = int(os.environ.get("MAX_FILING_AGE_DAYS", "30"))  # skip stale filings so the first run doesn't ask about January
    out, today = [], date.today()
    for year in years or (today.year - 1, today.year):
        z = zipfile.ZipFile(io.BytesIO(get(f"{CLERK}/financial-pdfs/{year}FD.zip")))
        for m in ET.fromstring(z.read(f"{year}FD.xml")):
            d = {c.tag: (c.text or "") for c in m}
            if d["FilingType"] != "P" or (who and d["Last"].lower() != who.lower()):
                continue
            if skip_docs is not None:
                if d["DocID"] in skip_docs:
                    continue
                skip_docs.add(d["DocID"])
            filed = datetime.strptime(d["FilingDate"], "%m/%d/%Y").date()
            if (today - filed).days > max_age:
                continue
            rows = (cache or {}).get(d["DocID"])
            if rows is None:
                with tempfile.TemporaryDirectory() as tmp:
                    pdf = Path(tmp) / "ptr.pdf"
                    try:
                        pdf.write_bytes(get(f"{CLERK}/ptr-pdfs/{year}/{d['DocID']}.pdf"))
                    except urllib.error.HTTPError:
                        continue  # a few filings are scanned paper with no PDF at this path
                    text = subprocess.run([PDFTOTEXT, "-layout", str(pdf), "-"], capture_output=True, text=True).stdout
                rows = parse_ptr(text)
                if cache is not None:
                    cache[d["DocID"]] = rows
            for t in rows:
                out.append({**t, "Representative": f"{d['First']} {d['Last']}".strip(), "Last": d["Last"],
                            "ReportDate": d["FilingDate"], "DocID": d["DocID"]})
    return out


def trade_key(t):
    return f"{t.get('Representative')}|{t.get('Ticker')}|{t.get('TransactionDate')}|{t.get('Transaction')}|{t.get('Range')}"


def old_key(t):  # the key from before every member was tracked (Pelosi only): keeps her already-seen trades seen
    return f"{t.get('Ticker')}|{t.get('TransactionDate')}|{t.get('Transaction')}|{t.get('Range')}"


def amount(t):
    """Low end of the disclosed dollar range, e.g. '$1,000,001 -' -> 1000001."""
    return int(re.sub(r"\D", "", t["Range"].split("-")[0]) or 0)


URGENCY = ((1_000_000, 3, "URGENT"), (250_000, 2, "HIGH"), (50_000, 1, "MEDIUM"), (0, 0, "LOW"))


def urgency(t):
    """(level 0-3, label) from the trade's size: bigger trades = more conviction, so they're shown first."""
    return next((lvl, name) for floor, lvl, name in URGENCY if amount(t) >= floor)


def score_pelosi(t, today):
    """Rule-based 0-100 score: for members with no learned model, and for Pelosi until learn.py has trained one."""
    pts, why = 50, []
    age = (today - datetime.strptime(t["TransactionDate"], "%m/%d/%Y").date()).days
    if age <= 14:
        pts += 20; why.append(f"fresh ({age}d since their trade)")
    elif age <= 30:
        why.append(f"{age}d since their trade")
    else:
        pts -= 25; why.append(f"stale ({age}d since their trade, move may be over)")
    low = int(re.sub(r"\D", "", t["Range"].split("-")[0]) or 0)
    if low >= 1_000_000:
        pts += 20; why.append("very large ($1M+), high conviction")
    elif low >= 250_000:
        pts += 10; why.append("large ($250k+)")
    else:
        pts -= 5; why.append("small")
    if "partial" in t["Transaction"].lower():
        pts -= 15; why.append("partial sale = trimming, not exiting")
    return max(0, min(100, pts)), why


def pelosi_features(t):
    """Traits learn.py learns from and the live copier scores on. Same function for both so they always match."""
    lag = (datetime.strptime(t["ReportDate"], "%m/%d/%Y") - datetime.strptime(t["TransactionDate"], "%m/%d/%Y")).days
    low = int(re.sub(r"\D", "", t["Range"].split("-")[0]) or 0)
    return {"kind": t["Transaction"],
            "size": "$1M+" if low >= 1_000_000 else "$250k-1M" if low >= 250_000 else "under $250k",
            "lag": "filed within 14d" if lag <= 14 else "filed 15-30d after" if lag <= 30 else "filed 31d+ after"}


def daily_closes(symbols, start, end):
    """{symbol: [(YYYY-MM-DD, split-adjusted close), ...]} from Alpaca market data (your paper keys work), 100 symbols a call."""
    h = {"APCA-API-KEY-ID": os.environ["ALPACA_KEY"], "APCA-API-SECRET-KEY": os.environ["ALPACA_SECRET"]}
    out, syms = {}, sorted(symbols)
    for i in range(0, len(syms), 100):
        token = None
        while True:
            url = (f"https://data.alpaca.markets/v2/stocks/bars?symbols={','.join(syms[i:i + 100])}&timeframe=1Day"
                   f"&start={start}&end={end}&adjustment=all&limit=10000" + (f"&page_token={token}" if token else ""))
            st, d = http("GET", url, h)
            if st != 200:
                raise RuntimeError(f"Alpaca data error {st}: {d}")
            for sym, bars in (d.get("bars") or {}).items():
                out.setdefault(sym, []).extend((b["t"][:10], b["c"]) for b in bars)
            token = d.get("next_page_token")
            if not token:
                break
    return out


def trend_feature(sym_closes, spy_closes, day, kind):
    """The stock vs the market (SPY) over the 20 trading days before `day` (YYYY-MM-DD), only days already closed.
    Side-relative: for a sale, a falling stock is 'with trend'. {} if there isn't enough price history."""
    a, m = [c for d, c in sym_closes if d < day][-21:], [c for d, c in spy_closes if d < day][-21:]
    if len(a) < 21 or len(m) < 21:
        return {}
    rel = (a[-1] / a[0] - 1) - (m[-1] / m[0] - 1)
    rel = -rel if "sale" in kind.lower() else rel
    return {"trend": "with 20d trend" if rel > 0.03 else "against 20d trend" if rel < -0.03 else "flat 20d vs market"}


def stock_features(t, closes, day):
    """Traits learn.py learns from and the live copier scores on (same function for both): size, kind, filing lag,
    the member (so each member gets their own model), and the stock's trend vs the market before `day`."""
    return {**pelosi_features(t), "trader": "member " + t["Representative"],
            **trend_feature(closes.get(t["Ticker"], []), closes.get("SPY", []), day, t["Transaction"])}


def is_auto(src):
    return isinstance(src, dict) and bool(src.get("auto"))


def rep_of(src):
    """Who a stock was copied from (older entries are just the name)."""
    return src if isinstance(src, str) else (src or {}).get("rep")


def plan_order(t, notional):
    """Return (description, action) or (reason_skipped, None). action = (method, path, body)."""
    sym, kind = (t.get("Ticker") or "").upper(), str(t.get("Transaction", "")).lower()
    if t.get("AssetType", "ST") != "ST":
        return f"skip: not a plain stock (type {t.get('AssetType') or '?'}, e.g. options or a private fund)", None
    if not sym or not sym.replace(".", "").isalpha():
        return f"skip: no usable ticker ({sym!r})", None
    if "purchase" in kind:
        return f"Buy ${notional:.0f} of {sym}", ("POST", "/v2/orders",
                {"symbol": sym, "notional": str(notional), "side": "buy", "type": "market", "time_in_force": "day"})
    if "sale" in kind:
        return f"Sell all {sym} you hold", ("DELETE", f"/v2/positions/{sym}", None)
    return f"skip: unknown transaction type {kind!r}", None


def check_pelosi(pending, log):
    """Look for new House trade filings from every member (or just POLITICIAN in .env, e.g. Pelosi).
    A full sale by the member you copied a stock from sells it automatically (that's their exit); other trades
    go to `pending` for your approval, rated by urgency (size). log(msg) records what happened."""
    load_env()
    notional = float(os.environ.get("DOLLARS_PER_TRADE", "1000"))
    who = os.environ.get("POLITICIAN", "")
    seen = set(json.loads(SEEN_FILE.read_text())) if SEEN_FILE.exists() else set()
    docs = set(json.loads(DOCS_FILE.read_text())) if DOCS_FILE.exists() else set()
    sources = json.loads(SOURCES_FILE.read_text()) if SOURCES_FILE.exists() else {}
    model = load_model("congress")  # learned from every member's trades, with personal models for active members
    try:
        trades = fetch_trades(who, skip_docs=docs)
    finally:
        DOCS_FILE.write_text(json.dumps(sorted(docs)))
    today = date.today().isoformat()
    try:  # prices for the trend-before-filing feature, all symbols in one go
        closes = daily_closes({t["Ticker"] for t in trades if t["Ticker"]} | {"SPY"},
                              (date.today() - timedelta(days=45)).isoformat(), today) if trades and model else {}
    except Exception:
        closes = {}
    events = []
    for t in sorted(trades, key=amount, reverse=True):
        k, name = trade_key(t), t["Representative"]
        if k in seen or old_key(t) in seen:
            continue
        seen.add(k)
        desc, action = plan_order(t, notional)
        sym = t["Ticker"]
        if action is None:
            continue  # options, funds, bonds...: not copyable as a plain stock
        if action[0] == "DELETE":
            if alpaca("GET", f"/v2/positions/{sym}")[0] != 200:
                continue  # you don't hold it
            if t["Transaction"] == "Sale" and rep_of(sources.get(sym)) == name:
                if close_stock(sym, f"{name}, who you copied it from, sold their whole position", sources, log) is None:
                    seen.discard(k)  # retry next check
                continue
        feats = stock_features(t, closes, today)
        if model:
            score, why = predict_for_trader(model, feats)
        else:
            score, why = score_pelosi(t, date.today())
            why.append("rule-based: relearn to learn from every member's history")
        events.append({"t": int(datetime.now().timestamp() * 1000), "day": today, "rep": name, "sym": sym,
                       "kind": t["Transaction"], "score": score if model else None})
        lvl, label = urgency(t)
        pending.append({"id": "p:" + k, "kind": "pelosi", "trade": t, "label": desc,
                        "side": "buy" if action[0] == "POST" else "sell",
                        "title": f"{name} {t['Transaction'].lower()} {sym}",
                        "detail": "Alpaca paper",
                        "urgency": lvl, "urgency_label": label,
                        "entered_at": int(datetime.strptime(t["TransactionDate"], "%m/%d/%Y").timestamp() * 1000),
                        "filed_at": int(datetime.strptime(t["ReportDate"], "%m/%d/%Y").timestamp() * 1000),
                        "score": score, "verdict": verdict(score, 60 if model else 70), "why": why, "feats": feats})
    SEEN_FILE.write_text(json.dumps(sorted(seen)))
    SOURCES_FILE.write_text(json.dumps(sources))
    with STOCK_EVENTS.open("a") as f:  # every scored trade, copied or not, so learn.py can grade it a month later
        f.writelines(json.dumps(e) + "\n" for e in events)


def decide_pelosi(item, auto=False):
    """Place the approved order (auto=True: autopilot's own). Returns a short message for the activity log."""
    load_env()
    _, action = plan_order(item["trade"], float(os.environ.get("DOLLARS_PER_TRADE", "1000")))
    if action[0] == "POST" and alpaca("GET", f"/v2/assets/{action[2]['symbol']}")[0] != 200:
        return f"Alpaca can't trade {action[2]['symbol']}, skipped"
    st, r = place(*action)
    if st < 300 and action[0] == "POST":  # remember who you copied it from: their full sale sells it automatically
        sources = json.loads(SOURCES_FILE.read_text()) if SOURCES_FILE.exists() else {}
        sources[action[2]["symbol"]] = {"rep": item["trade"].get("Representative"), "auto": auto,
                                         "opened": datetime.now().isoformat(timespec="seconds"), "feats": item.get("feats")}
        SOURCES_FILE.write_text(json.dumps(sources))
    return f"Done: {item['label']}" if st < 300 else f"Failed ({st}): {r}"


def my_orders():
    try:
        return json.loads(MY_ORDERS.read_text())
    except (OSError, json.JSONDecodeError):  # first run: watch from now on (older orders can't be told apart)
        mine = {"since": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "ids": []}
        MY_ORDERS.write_text(json.dumps(mine))
        return mine


def reset_orders():
    """After you rotate your keys: start watching again from now."""
    MY_ORDERS.unlink(missing_ok=True)
    my_orders()


def place(method, path, body=None):
    """Send an order to Alpaca and remember its ID as the copier's own (POST /orders and DELETE /positions both return one)."""
    st, r = alpaca(method, path, body)
    if st < 300 and isinstance(r, dict) and r.get("id"):
        mine = my_orders()
        mine["ids"].append(r["id"])
        MY_ORDERS.write_text(json.dumps(mine))
    return st, r


def foreign_orders():
    """Orders in your Alpaca paper account that the copier didn't place, since it started watching.
    Only the copier should trade there, so any of these means someone else may be using your keys
    (or you traded on the Alpaca website yourself)."""
    load_env()
    mine = my_orders()
    st, orders = alpaca("GET", f"/v2/orders?status=all&after={mine['since']}&limit=500&direction=desc")
    if st != 200:
        return []
    known = set(mine["ids"])
    return [f"{o.get('side', '?')} {o.get('symbol', '?')} at {o.get('created_at', '?')[:16].replace('T', ' ')} UTC"
            for o in orders if o.get("id") not in known]


def stock_history():
    try:
        return json.loads(STOCK_HISTORY.read_text())
    except (OSError, json.JSONDecodeError):
        return []


def close_stock(sym, why, sources, log, pos=None):
    """Sell the whole position, and record its real result (learned from; judged by the loss guard).
    Returns the P&L, or None if the order failed. `sources` is updated; the caller saves it."""
    pos = pos or alpaca("GET", f"/v2/positions/{sym}")[1]
    st, r = place("DELETE", f"/v2/positions/{sym}")
    if st >= 300:
        log(f"Selling {sym} failed ({st}), will retry")
        return None
    src = sources.pop(sym, None)
    pnl = float(pos.get("unrealized_pl", 0)) if isinstance(pos, dict) else 0.0
    ret = float(pos.get("unrealized_plpc", 0)) if isinstance(pos, dict) else 0.0
    src = src if isinstance(src, dict) else {"rep": src}
    h = stock_history() + [{"time": datetime.now().isoformat(timespec="seconds"), "sym": sym, "pnl": pnl, "ret": ret,
                            "rep": src.get("rep"), "auto": bool(src.get("auto")), "feats": src.get("feats")}]
    STOCK_HISTORY.write_text(json.dumps(h[-2000:]))
    log(f"Sold {sym} ({'+' if pnl >= 0 else '-'}${abs(pnl):,.2f}): {why}")
    return pnl


def stock_rule(src):
    """(take profit, stop loss, max days): every copy takes profit at +9%; autopilot's also stop at -3% and end after 30 days."""
    return (STOCK_TP, STOCK_STOP, STOCK_MAX_DAYS) if is_auto(src) else (STOCK_TP, None, None)


def stock_exits(log):
    """Every minute: take profit / stop loss / time limit on stock copies. Only calls Alpaca if you hold copies."""
    sources = json.loads(SOURCES_FILE.read_text()) if SOURCES_FILE.exists() else {}
    if not sources:
        return
    load_env()
    st, ps = alpaca("GET", "/v2/positions")
    if st != 200:
        return
    held = {p["symbol"]: p for p in ps}
    now, changed = datetime.now(), False
    for sym, src in list(sources.items()):
        p = held.get(sym)
        if not p:
            continue
        tp, sl, days = stock_rule(src)
        r = float(p["unrealized_plpc"])
        why = ("Take profit" if r >= tp else "Stop loss" if sl and r <= -sl else
               "Time limit" if days and now - datetime.fromisoformat(src["opened"]) >= timedelta(days=days) else None)
        if why:
            changed |= close_stock(sym, f"{why} hit ({r:+.1%})", sources, log, p) is not None
    if changed:
        SOURCES_FILE.write_text(json.dumps(sources))


def stock_account():
    """Alpaca paper account summary and positions for the UI."""
    load_env()
    st, a = alpaca("GET", "/v2/account")
    _, ps = alpaca("GET", "/v2/positions")
    if st != 200:
        return {"error": f"Alpaca error {st}"}
    src = json.loads(SOURCES_FILE.read_text()) if SOURCES_FILE.exists() else {}
    return {"equity": float(a["equity"]), "change": float(a["equity"]) - float(a["last_equity"]), "cash": float(a["cash"]),
            "positions": [{"symbol": p["symbol"], "qty": float(p["qty"]), "entry": float(p["avg_entry_price"]),
                           "now": float(p["current_price"]), "pnl": float(p["unrealized_pl"]),
                           "source": rep_of(src.get(p["symbol"])) or "", "auto": is_auto(src.get(p["symbol"]))}
                          for p in (ps or [])]}
