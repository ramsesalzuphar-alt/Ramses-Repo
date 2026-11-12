"""Company insider trades (SEC Form 4). Executives, directors and 10% owners must report buying or selling their own
company's stock within 2 business days, weeks sooner than House members. Their open-market buys go through the same
stock pipeline as House trades (scoring, gate, autopilot, exits), and are graded over 10 trading days, not a month.
History: the SEC's quarterly data sets (for learn.py). Live: EDGAR's latest-filings feed, every check.
"""
import csv, io, json, os, re, time, urllib.error, urllib.request, zipfile
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
from pathlib import Path

HERE = Path(__file__).parent
CACHE_FILE = HERE / "insider_cache.json"   # parsed quarters, so each is downloaded once
SEEN_FILE = HERE / "insider_seen.json"     # live filings already read, and recent buys (for 'other insiders bought too')
# the SEC refuses automated requests that don't give a contact: SEC_CONTACT in .env (kept out of the public repo)
UA = lambda: f"TradeCopier {os.environ.get('SEC_CONTACT', '')}".strip()
MIN_VALUE, MIN_PRICE = 25_000, 1.0  # skip token buys and penny stocks (both live and in learning)
HOLD = 10            # trading days an insider copy is graded over
CLUSTER_DAYS = 30    # other insiders buying the same stock within this many days before = a cluster
FIRST_QUARTER = (2024, 1)


def get(url, tries=4):
    """GET from the SEC: under its 10-requests-a-second limit, retrying throttling and outages."""
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA()}), timeout=90) as r:
                data = r.read()
            time.sleep(0.15)
            return data
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503) or i == tries - 1:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if i == tries - 1:
                raise
        time.sleep(5 * (i + 1))


def role(director, officer, ten, title):
    t = (title or "").lower()
    if officer and re.search(r"chief executive|\bceo\b|president", t):
        return "CEO/president"
    if officer and re.search(r"chief financial|\bcfo\b", t):
        return "CFO"
    return "other officer" if officer else "director" if director else "10% owner" if ten else "other"


def trade(owner, sym, code, traded, filed, shares, value, after, rl):
    """A Form 4 trade in the same shape as a House trade, so the stock pipeline handles it unchanged."""
    us = lambda d: datetime.strptime(d, "%Y-%m-%d").strftime("%m/%d/%Y")
    kind = "Purchase" if code == "P" else "Sale (Full)" if not after else "Sale (Partial)"
    return {"Representative": " ".join(w.capitalize() for w in owner.split()), "Ticker": sym.upper().strip(), "AssetType": "ST",
            "Transaction": kind, "TransactionDate": us(traded), "ReportDate": us(filed), "Range": f"${value:,.0f} - ${value:,.0f}",
            "Source": "insider", "Role": rl, "Stake": shares / (after - shares) if code == "P" and after > shares else None}


def combine(rows):
    """One trade per filing and kind: [(code, date, shares, price, owned after)] -> {code: (first date, shares, value, after)}."""
    out = {}
    for code, d, sh, px, after in rows:
        o = out.setdefault(code, [d, 0.0, 0.0, 0.0])
        o[0], o[1], o[2], o[3] = min(o[0], d), o[1] + sh, o[2] + sh * px, after
    return out


def keep(code, shares, value):
    return code == "S" or (value >= MIN_VALUE and shares and value / shares >= MIN_PRICE)


def quarter(y, q):
    """[trade] of a published quarter's insider buys (sales aren't learned from). None if not published yet."""
    for folder in ("datastandardsinnovation", "structureddata"):  # the SEC moved newer quarters to a new folder
        try:
            z = zipfile.ZipFile(io.BytesIO(get(f"https://www.sec.gov/files/{folder}/data/insider-transactions-data-sets/{y}q{q}_form345.zip")))
            break
        except urllib.error.HTTPError as e:
            if e.code not in (403, 404):
                raise
    else:
        return None
    tsv = lambda name: csv.DictReader(io.TextIOWrapper(z.open(name), encoding="utf-8", errors="replace"), delimiter="\t", quoting=csv.QUOTE_NONE)
    sub = {r["ACCESSION_NUMBER"]: r for r in tsv("SUBMISSION.tsv") if r["DOCUMENT_TYPE"] == "4" and r["ISSUERTRADINGSYMBOL"]}
    own = {}
    for r in tsv("REPORTINGOWNER.tsv"):
        own.setdefault(r["ACCESSION_NUMBER"], r)  # first owner of a joint filing
    rows = {}
    for r in tsv("NONDERIV_TRANS.tsv"):
        if r["TRANS_CODE"] == "P" and r["ACCESSION_NUMBER"] in sub:
            try:
                d = datetime.strptime(r["TRANS_DATE"], "%d-%b-%Y").strftime("%Y-%m-%d")
                rows.setdefault(r["ACCESSION_NUMBER"], []).append(("P", d, float(r["TRANS_SHARES"]), float(r["TRANS_PRICEPERSHARE"] or 0),
                                                                   float(r["SHRS_OWND_FOLWNG_TRANS"] or 0)))
            except ValueError:
                continue
    out = []
    for acc, rs in rows.items():
        s, o = sub[acc], own.get(acc)
        if not o:
            continue
        rel = o["RPTOWNER_RELATIONSHIP"]
        rl = role("Director" in rel, "Officer" in rel, "TenPercentOwner" in rel, o["RPTOWNER_TITLE"])
        filed = datetime.strptime(s["FILING_DATE"], "%d-%b-%Y").strftime("%Y-%m-%d")
        for code, (d, sh, val, after) in combine(rs).items():
            if keep(code, sh, val):
                out.append(trade(o["RPTOWNERNAME"], s["ISSUERTRADINGSYMBOL"], code, d, filed, sh, val, after, rl))
    return out


def history(today=None):
    """Every insider buy since FIRST_QUARTER, oldest quarter first, each with 'Cluster' (how many other insiders
    bought the same stock in the 30 days before it was filed). Completed quarters are cached."""
    today = today or date.today()
    try:
        cache = json.loads(CACHE_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        cache = {}
    out, (y, q) = [], FIRST_QUARTER
    while (y, q) <= (today.year, (today.month - 1) // 3 + 1):
        key = f"{y}q{q}"
        if key not in cache:
            got = quarter(y, q)
            if got is None:
                break  # the SEC publishes a quarter some weeks after it ends; newer buys come from the live feed
            cache[key] = got
            tmp = CACHE_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps(cache))
            os.replace(tmp, CACHE_FILE)
        out += cache[key]
        y, q = (y + 1, 1) if q == 4 else (y, q + 1)
    out = [t for t in out if tradable(t["Ticker"])]
    add_clusters(out)
    return out


def tradable(sym):
    """A plain stock ticker (AAPL, BRK.B), not a note, unit or private-fund name like AB-LEND that Alpaca refuses."""
    return bool(re.fullmatch(r"[A-Z]{1,5}(\.[A-Z])?", sym)) and sym not in ("NONE", "NA", "N")  # 'no ticker' placeholders


def iso(us):
    return datetime.strptime(us, "%m/%d/%Y").date()


def add_clusters(trades, recent=()):
    """t['Cluster'] = other insiders' buys of the same stock filed in the 30 days before t (recent: earlier live buys)."""
    buys = sorted([(iso(t["ReportDate"]), t["Ticker"], t["Representative"]) for t in trades if t["Transaction"] == "Purchase"]
                  + [(date.fromisoformat(d), s, n) for d, s, n in recent])
    by = {}
    for d, s, n in buys:
        by.setdefault(s, []).append((d, n))
    for t in trades:
        if t["Transaction"] == "Purchase":
            f = iso(t["ReportDate"])
            t["Cluster"] = len({n for d, n in by.get(t["Ticker"], []) if f - timedelta(days=CLUSTER_DAYS) <= d < f and n != t["Representative"]})


def parse_filing(text, filed):
    """[trade] from one Form 4's full text (its <XML> part): the open-market buys (P) and sales (S)."""
    m = re.search(r"<XML>\s*(.*?)\s*</XML>", text, re.S)
    if not m:
        return []
    doc = ET.fromstring(m.group(1).strip())
    v = lambda node, path: (node.findtext(path) or "").strip()
    if v(doc, "documentType") != "4" or not v(doc, "issuer/issuerTradingSymbol"):
        return []
    o = doc.find("reportingOwner")
    flag = lambda tag: v(o, f"reportingOwnerRelationship/{tag}") in ("1", "true")
    rl = role(flag("isDirector"), flag("isOfficer"), flag("isTenPercentOwner"), v(o, "reportingOwnerRelationship/officerTitle"))
    rows = []
    for x in doc.iter("nonDerivativeTransaction"):
        code = v(x, "transactionCoding/transactionCode")
        if code in ("P", "S"):
            try:
                rows.append((code, v(x, "transactionDate/value")[:10], float(v(x, "transactionAmounts/transactionShares/value")),
                             float(v(x, "transactionAmounts/transactionPricePerShare/value") or 0),
                             float(v(x, "postTransactionAmounts/sharesOwnedFollowingTransaction/value") or 0)))
            except ValueError:
                continue
    return [trade(v(o, "reportingOwnerId/rptOwnerName"), v(doc, "issuer/issuerTradingSymbol"), code, d, filed, sh, val, after, rl)
            for code, (d, sh, val, after) in combine(rows).items() if keep(code, sh, val)]


def live(log=print):
    """New insider buys and sales since the last call, from EDGAR's latest 100 Form 4 filings."""
    try:
        seen = json.loads(SEEN_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        seen = {"filings": [], "recent": []}
    done = set(seen["filings"])
    feed = get("https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=4&count=100&output=atom").decode("latin-1")
    out = []
    for link, filed in re.findall(r'href="(https://www\.sec\.gov/Archives/edgar/data/\d+/\d+/[\d-]+)-index\.htm".*?Filed:&lt;/b&gt; (\d{4}-\d\d-\d\d)', feed, re.S):
        acc = link.rsplit("/", 1)[1]
        if acc in done:
            continue
        done.add(acc)
        seen["filings"].append(acc)
        try:
            out += parse_filing(get(link + ".txt").decode("utf-8", "replace"), filed)
        except (urllib.error.URLError, ET.ParseError, TimeoutError, ConnectionError) as e:
            log(f"Couldn't read insider filing {acc}: {e}")
    out = [t for t in out if tradable(t["Ticker"])]
    cutoff = (date.today() - timedelta(days=CLUSTER_DAYS + 5)).isoformat()
    recent = [r for r in seen["recent"] if r[0] >= cutoff]
    add_clusters(out, recent)
    seen["recent"] = recent + [[iso(t["ReportDate"]).isoformat(), t["Ticker"], t["Representative"]] for t in out if t["Transaction"] == "Purchase"]
    seen["filings"] = seen["filings"][-3000:]
    SEEN_FILE.write_text(json.dumps(seen))
    return out
