"""Trade Copier app: one page with buttons for everything. Run: python app.py, then open http://127.0.0.1:8765
Checks every House member's trade filings and the top crypto traders every 30 minutes (or when you press Check now).
New trades wait for your Copy/Skip; exits of copied trades happen automatically. Paper money only.
Relearns daily. Each market (crypto, stocks) has its own autopilot that copies STRONG trades by itself; it only unlocks
once that market's live track record proves an edge, or runs as a paper experiment if you start it.
"""
import json, threading, time, traceback
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import copier, crypto_copier, learn, scout

HERE = Path(__file__).parent
APP_FILE = HERE / "app_state.json"  # waiting proposals + activity log, so a restart loses nothing
PORT, CHECK_EVERY, RETRY_AFTER = 8765, 30 * 60, 2 * 60  # a failed check is retried after 2 min
RETRAIN_EVERY, AUTO_MAX_PER_CHECK = 24 * 3600, 5
EXPIRE_AFTER = 10 * 60  # seconds a crypto trade waits for you
MARKETS = ("crypto", "stocks")
# loss guard: last 10 autopilot trades losing more than ~1% each on average -> pause + relearn ($100 crypto, $1000 stock copies)
GUARD_LAST, GUARD_MAX_TRIPS, GUARD_LOSS = 10, 3, {"crypto": -10.0, "stocks": -100.0}
KEEP = ("pending", "log", "traders", "last_check", "ap", "security")
NEW_AP = {"on": False, "forced": False, "guard_since": "", "guard_trips": [], "resume": False}

lock = threading.RLock()
wake = threading.Event()
saved = json.loads(APP_FILE.read_text()) if APP_FILE.exists() else {}
if "ap" not in saved and "autopilot" in saved:  # state saved before stocks had an autopilot: it was crypto's
    saved["ap"] = {"crypto": {"on": saved["autopilot"], "forced": saved.get("forced", False), "guard_since": saved.get("guard_since", ""),
                              "guard_trips": saved.get("guard_trips", []), "resume": saved.get("resume_after_retrain", False)}}
S = {"pending": [], "log": [], "traders": [], "last_check": None, "security": [], **{k: v for k, v in saved.items() if k in KEEP},
     "crypto": None, "stocks": None, "next_check": None, "checking": False, "error": None, "training": False,
     "model": None, "auto": {}, "scout": None}
S["ap"] = {m: {**NEW_AP, **S.get("ap", {}).get(m, {})} for m in MARKETS}
snapshot = b"{}"


def ap(m):
    """That market's autopilot state: on, forced (a paper experiment), guard_since, guard_trips, resume."""
    return S["ap"][m]


def closed_trades(m):
    """(autopilot's closed trades with real P&L, how many of its copies are open) for one market."""
    if m == "crypto":
        ledger = crypto_copier.load_ledger()
        return ([h for h in ledger["history"] if h.get("auto") and "pnl" in h],
                sum(bool(p.get("auto")) for p in ledger["positions"].values()))
    sources = json.loads(copier.SOURCES_FILE.read_text()) if copier.SOURCES_FILE.exists() else {}
    return [h for h in copier.stock_history() if h.get("auto")], sum(copier.is_auto(v) for v in sources.values())


def auto_stats(m):
    """How a market's autopilot is doing. 'recent' = its trades closed since it was last (re)started."""
    closes, n_open = closed_trades(m)
    recent = [h for h in closes if h["time"] > ap(m)["guard_since"]][-GUARD_LAST:]
    return {"open": n_open, "closed": len(closes), "pnl": sum(h["pnl"] for h in closes),
            "won": sum(h["pnl"] > 0 for h in closes) / len(closes) if closes else None,
            "recent_n": len(recent), "recent_pnl": sum(h["pnl"] for h in recent)}


def guard(m):
    """Learn from losses to keep winning: if a market's last 10 autopilot trades lost too much, pause it, relearn
    (those losses are now training data), then resume. Three trips in 24h -> stop and wait for the user."""
    a, st = ap(m), auto_stats(m)
    S["auto"][m] = st
    if not a["on"] or st["recent_n"] < GUARD_LAST or st["recent_pnl"] >= GUARD_LOSS[m]:
        return
    now = time.time()
    a["guard_trips"] = [t for t in a["guard_trips"] if now - t < 86400] + [now]
    a["on"] = False
    a["guard_since"] = datetime.now().isoformat(timespec="seconds")  # judge the relearned model on fresh trades only
    if len(a["guard_trips"]) >= GUARD_MAX_TRIPS:
        a["forced"] = a["resume"] = False
        log(f"{m.title()} autopilot STOPPED: loss guard tripped {GUARD_MAX_TRIPS} times in 24h. Review before turning it back on.")
        return
    a["resume"] = True
    log(f"{m.title()} loss guard: last {GUARD_LAST} autopilot trades lost {crypto_copier.money(st['recent_pnl'])}. "
        "Paused; relearning from them, then resuming.")
    threading.Thread(target=retrain, daemon=True).start()


def model_summary():
    """What the page shows about learning, per market: backtests, personal models, patterns, live track record."""
    try:
        m = json.loads(copier.MODEL_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    def sec(x, live):
        if not x:
            return {"live": live}
        return {"n": x.get("n"), "base_win": x["base"][2], "patterns": x.get("patterns", []), "backtest": x["validation"],
                "personal": x.get("validation_personal"), "profiles": x.get("profiles", [])[:24], "live": live}
    return {"trained": m["trained"], "rule": {"strong": learn.STRONG, "weak": learn.WEAK, "min_each": learn.MIN_EACH},
            "crypto": sec(m.get("crypto"), m.get("live")), "stocks": sec(m.get("congress"), m.get("live_stocks"))}


def unlocked(m):
    return bool((((S["model"] or {}).get(m) or {}).get("live") or {}).get("edge"))


def retrain():
    """Runs learn.py in the background. Turns a market's autopilot off if its edge is gone; resumes guard pauses."""
    with lock:
        if S["training"]:
            return
        S["training"] = True
        publish()
    try:
        learn.main()
        result = "Relearned from all trades so far"
    except (Exception, SystemExit) as e:
        traceback.print_exc()
        result = f"Relearning failed ({e}); keeping the previous model"
    with lock:
        S["training"] = False
        S["model"] = model_summary()
        log(result)
        for m in MARKETS:
            a = ap(m)
            if a["on"] and not a["forced"] and not unlocked(m):
                a["on"] = False
                log(f"{m.title()} autopilot turned OFF: the live track record no longer shows an edge")
            if a["resume"]:
                a["resume"] = False
                if a["forced"] or unlocked(m):
                    a["on"] = True
                    log(f"{m.title()} autopilot resumed with the relearned model")
        publish()


def retrain_due():
    m = S["model"]
    return not S["training"] and (not m or time.time() - datetime.fromisoformat(m["trained"]).timestamp() > RETRAIN_EVERY)


def run_autopilot(m):
    """Copy a market's most profitable-looking STRONG proposals by itself, at most a few per check. Stocks: buys only.
    Only while switched on, and unlocked by a live edge or started as an experiment by the user."""
    a = ap(m)
    if not (a["on"] and (a["forced"] or unlocked(m))):
        return
    mine = (lambda x: x["kind"] == "crypto") if m == "crypto" else (lambda x: x["kind"] == "pelosi" and x["side"] == "buy")
    strong = sorted((x for x in S["pending"] if mine(x) and x["score"] >= learn.STRONG), key=lambda x: -x["score"])
    for item in strong[:AUTO_MAX_PER_CHECK]:
        S["pending"].remove(item)
        try:
            msg = crypto_copier.decide_crypto(item, auto=True) if m == "crypto" else copier.decide_pelosi(item, auto=True)
            log(f"{m.title()} autopilot ({item['score']}): {msg}")
        except Exception as e:
            log(f"{m.title()} autopilot failed on {item['title']}: {e}")


def expire():
    """Crypto trades waiting for you disappear 10 minutes after they were spotted: after that the copy price is stale."""
    cutoff = (time.time() - EXPIRE_AFTER) * 1000
    old = [x for x in S["pending"] if x.get("seen_at") and x["seen_at"] < cutoff]
    for x in old:
        S["pending"].remove(x)
    if old:
        log(f"Expired {len(old)} waiting trade(s) after {EXPIRE_AFTER // 60} min: {', '.join(x['title'] for x in old)}")


def log(msg):
    S["log"].insert(0, {"time": datetime.now().strftime("%b %d %H:%M"), "msg": msg})
    del S["log"][200:]


def publish():
    """Save to disk and refresh the JSON the page reads (so page reads never wait on a running check)."""
    global snapshot
    APP_FILE.write_text(json.dumps({k: S[k] for k in KEEP}, indent=1))
    snapshot = json.dumps(S).encode()


def security_check():
    """Alpaca orders the copier didn't place mean someone else may have your keys: flag each new one once."""
    try:
        foreign = copier.foreign_orders()
    except Exception:
        return  # can't reach Alpaca: check again next time
    new = [o for o in foreign if o not in S["security"]]
    if new:
        log(f"SECURITY: {len(new)} order(s) in your Alpaca paper account the copier didn't place: {'; '.join(new)}. "
            "If you didn't place them, rotate your Alpaca keys now.")
    S["security"] = foreign


def scout_summary():
    """What the page shows about the edge scout: progress through all traders, and its most consistent finds."""
    try:
        st = json.loads(crypto_copier.SCOUT_FILE.read_text())
    except (OSError, json.JSONDecodeError):  # OSError: missing, or locked mid-replace by the scout / OneDrive
        return S.get("scout")
    finds = [{"name": nm or crypto_copier.short_addr(a), "addr": a, "watched": True, **{k: t[k] for k in
              ("n", "win", "avg", "worst", "consistent", "patterns")}, "exit": t.get("exit")} for a, nm, t in crypto_copier.watchlist()]
    return {"scanned": sum(t.get("v") == crypto_copier.SCOUT_VERSION for t in st["traders"].values()), "of": st.get("candidates", 0), "error": st.get("error"), "finds": finds}


def scout_log(msg):
    with lock:
        log(msg)
        publish()


def refresh_accounts():
    S["scout"] = scout_summary()
    for key, fn in (("crypto", crypto_copier.crypto_account), ("stocks", copier.stock_account)):
        try:
            S[key] = fn()
        except Exception as e:
            S[key] = {"error": str(e)}


def run_check():
    with lock:
        S["checking"] = True
        publish()
        try:
            expire()
            before = len(S["pending"])
            copier.check_pelosi(S["pending"], log)
            S["traders"] = crypto_copier.check_crypto(S["pending"], log)
            for m in MARKETS:
                guard(m)  # judge the trades that just closed before opening new ones
                run_autopilot(m)
            security_check()
            added = len(S["pending"]) - before
            log(f"Checked: {added} new trade(s) to review" if added > 0 else "Checked: nothing new")
            S["error"] = None
        except Exception as e:
            traceback.print_exc()
            S["error"] = f"Last check failed: {e}"
        S["checking"] = False
        S["last_check"] = datetime.now().strftime("%b %d %H:%M")
        wait = RETRY_AFTER if S["error"] else CHECK_EVERY
        S["next_check"] = datetime.fromtimestamp(time.time() + wait).strftime("%H:%M")
        refresh_accounts()
        publish()
        return wait


def loop():
    """Never dies: an unexpected error in one pass is logged and the next pass runs anyway."""
    while True:
        try:
            loop_once()
        except Exception as e:
            traceback.print_exc()
            with lock:
                S["checking"] = False
                log(f"Internal error, recovering: {e}")
                publish()
            time.sleep(60)


def loop_once():
    """One check, then a quick exit check every minute until the next one."""
    deadline = time.time() + run_check()
    if retrain_due():
        threading.Thread(target=retrain, daemon=True).start()
    while time.time() < deadline:
        if wake.wait(60):  # "Check now" pressed
            wake.clear()
            break
        with lock:  # every minute: close copies whose trader exited or that hit their exit rule; fresh prices and P&L
            expire()
            for name, fn in (("crypto", crypto_copier.exit_check), ("stocks", copier.stock_exits)):
                try:
                    fn(log)
                    guard(name)
                except Exception:
                    traceback.print_exc()  # a failed quick check is retried next minute; the full check also catches exits
            refresh_accounts()
            publish()


def decide(item_id, approve):
    with lock:
        item = next((x for x in S["pending"] if x["id"] == item_id), None)
        if not item:
            return "That trade is no longer waiting"
        S["pending"].remove(item)
        if not approve:
            msg = f"Skipped: {item['title']}"
        else:
            try:
                msg = (copier.decide_pelosi if item["kind"] == "pelosi" else crypto_copier.decide_crypto)(item)
            except Exception as e:
                msg = f"Failed: {e}"
        log(msg)
        refresh_accounts()
        publish()
        return msg


def close(coin):
    with lock:
        try:
            msg = crypto_copier.close_manual(coin)
        except Exception as e:
            msg = f"Failed: {e}"
        log(msg)
        refresh_accounts()
        publish()
        return msg


def set_autopilot(on, force=False, m="crypto"):
    """force = the user's own choice to run an unproven model as a paper experiment."""
    if m not in MARKETS:
        return "Unknown market"
    with lock:
        a = ap(m)
        if on and not (force or unlocked(m)):
            return (f"{m.title()} autopilot is locked until STRONG trades average +0.5% or more and beat WEAK ones live. "
                    "Use Run experiment to try it anyway.")
        a.update(on=on, forced=bool(on and not unlocked(m)), guard_since=datetime.now().isoformat(timespec="seconds"),
                 guard_trips=[], resume=False)
        what = ("copies STRONG crypto trades by expected profit" if m == "crypto"
                else "buys STRONG congressional trades by expected profit")
        log(f"{m.title()} autopilot OFF" if not on else
            f"{m.title()} autopilot ON{' as an EXPERIMENT (model not proven yet)' if a['forced'] else ''}: {what}, "
            f"max {AUTO_MAX_PER_CHECK} per check, loss guard active (paper)")
        publish()
        return f"{m.title()} autopilot {'on' if on else 'off'}"


class Handler(BaseHTTPRequestHandler):
    def _ok_origin(self):
        # only this computer's own page may use the app: blocks other websites and DNS-rebinding tricks
        return self.headers.get("Host") in (f"127.0.0.1:{PORT}", f"localhost:{PORT}")

    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self._ok_origin():
            return self._send(403, b"forbidden", "text/plain")
        if self.path == "/":
            return self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        if self.path == "/api/state":
            return self._send(200, snapshot)
        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if not self._ok_origin() or self.headers.get("X-App") != "1":  # custom header = cross-site forms can't send it
            return self._send(403, b"forbidden", "text/plain")
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        if self.path == "/api/decide":
            msg = decide(str(body.get("id")), bool(body.get("approve")))
        elif self.path == "/api/close":
            msg = close(str(body.get("coin")))
        elif self.path == "/api/check":
            wake.set()
            msg = "Checking now..."
        elif self.path == "/api/autopilot":
            msg = set_autopilot(bool(body.get("on")), bool(body.get("force")), str(body.get("market") or "crypto"))
        elif self.path == "/api/security_reset":
            with lock:
                copier.reset_orders()
                S["security"] = []
                log("Security alert cleared (keys rotated); watching for unknown orders from now on")
                publish()
            msg = "Alert cleared"
        elif self.path == "/api/retrain":
            if S["training"]:
                msg = "Already relearning"
            else:
                threading.Thread(target=retrain, daemon=True).start()
                msg = "Relearning started (about 10 min; the first time with every member's filings, ~30 min)"
        else:
            return self._send(404, b"not found", "text/plain")
        self._send(200, json.dumps({"msg": msg}).encode())

    def log_message(self, *a):
        pass  # keep the console quiet; the page has its own activity log


if __name__ == "__main__":
    import sys
    if sys.stderr is None:  # started windowless (pythonw, at login): keep error traces in a file instead of losing them
        sys.stderr = sys.stdout = open(HERE / "app_errors.log", "a", buffering=1, encoding="utf-8")
    S["model"] = model_summary()
    for m in MARKETS:
        if ap(m)["on"] and not ap(m)["forced"] and not unlocked(m):
            ap(m)["on"] = False  # never resume autopilot on an unproven model unless the user started it as an experiment
        S["auto"][m] = auto_stats(m)
    publish()
    threading.Thread(target=loop, daemon=True).start()
    threading.Thread(target=scout.run, args=(scout_log,), daemon=True).start()  # nonstop search for consistent traders
    print(f"Trade Copier running at http://127.0.0.1:{PORT}  (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
