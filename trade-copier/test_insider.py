"""Company insider trades (SEC Form 4): filing parsing, one trade per filing, filters, roles, clusters, features."""
import copier, insider

def filing(code="P", shares="10000", price="12.50", after="40000", title="Chief Executive Officer", officer="1", sym="ABC"):
    tx = lambda c, sh, px, af, d: f"""<nonDerivativeTransaction><transactionDate><value>{d}</value></transactionDate>
      <transactionCoding><transactionCode>{c}</transactionCode></transactionCoding><transactionAmounts>
      <transactionShares><value>{sh}</value></transactionShares><transactionPricePerShare><value>{px}</value></transactionPricePerShare>
      </transactionAmounts><postTransactionAmounts><sharesOwnedFollowingTransaction><value>{af}</value></sharesOwnedFollowingTransaction>
      </postTransactionAmounts></nonDerivativeTransaction>"""
    return f"""<SEC-DOCUMENT><XML>
<?xml version="1.0"?><ownershipDocument><documentType>4</documentType>
<issuer><issuerTradingSymbol>{sym}</issuerTradingSymbol></issuer>
<reportingOwner><reportingOwnerId><rptOwnerName>DOE JANE A</rptOwnerName></reportingOwnerId>
<reportingOwnerRelationship><isDirector>0</isDirector><isOfficer>{officer}</isOfficer><isTenPercentOwner>0</isTenPercentOwner>
<officerTitle>{title}</officerTitle></reportingOwnerRelationship></reportingOwner>
<nonDerivativeTable>{tx(code, shares, price, after, "2026-11-05")}{tx(code, shares, price, int(after) + (int(shares) if code == "P" else -int(shares)), "2026-11-06")}
{tx("A", "500", "0", "1", "2026-11-06")}</nonDerivativeTable></ownershipDocument>
</XML></SEC-DOCUMENT>"""

t, = insider.parse_filing(filing(), "2026-11-07")   # two buys in one filing -> one trade; the grant (A) is ignored
assert t["Representative"] == "Doe Jane A" and t["Ticker"] == "ABC" and t["Transaction"] == "Purchase" and t["Role"] == "CEO/president"
assert t["TransactionDate"] == "11/05/2026" and t["ReportDate"] == "11/07/2026" and t["Range"] == "$250,000 - $250,000"
assert abs(t["Stake"] - 20000 / 30000) < 1e-9 and copier.amount(t) == 250000   # bought 20k on top of 30k
assert insider.parse_filing(filing(shares="900"), "2026-11-07") == []              # $22.5k: a token buy, skipped
assert insider.parse_filing(filing(shares="900000", price="0.50"), "2026-11-07") == []   # penny stock, skipped
s_, = insider.parse_filing(filing(code="S", shares="10", after="10"), "2026-11-07")   # sales are kept (they're exits)
assert s_["Transaction"] == "Sale (Full)" and s_["Stake"] is None
assert insider.parse_filing("<html>no xml</html>", "2026-11-07") == []
assert [insider.role(0, 1, 0, x) for x in ("CFO", "President & COO", "VP Sales")] == ["CFO", "CEO/president", "other officer"]
assert insider.role(1, 0, 1, "") == "director" and insider.role(0, 0, 1, "") == "10% owner"

# clusters: other insiders' buys of the same stock filed in the 30 days before (not their own, not later ones)
mk = lambda who, sym, filed: {"Representative": who, "Ticker": sym, "Transaction": "Purchase", "ReportDate": filed}
ts = [mk("A", "X", "11/01/2026"), mk("B", "X", "11/10/2026"), mk("A", "X", "11/12/2026"), mk("C", "Y", "11/12/2026"), mk("D", "X", "12/30/2026")]
insider.add_clusters(ts, recent=[("2026-10-20", "Y", "E")])
assert [t["Cluster"] for t in ts] == [0, 1, 1, 1, 0], [t["Cluster"] for t in ts]

# features: insiders say who's buying; House trades just their source; separate per-trader names
f = copier.stock_features({**t, "Cluster": 2}, {}, "2026-11-07")
assert f["trader"] == "insider Doe Jane A" and f["source"] == "company insider" and f["role"] == "insider: CEO/president"
assert f["cluster"] == "other insiders bought too (30d)" and f["stake"] == "stake up 20%+"
h = copier.stock_features({"Representative": "Bob", "Ticker": "Z", "Transaction": "Purchase", "TransactionDate": "11/01/2026",
                           "ReportDate": "11/05/2026", "Range": "$1,001 - $15,000"}, {}, "2026-11-05")
assert h["trader"] == "member Bob" and h["source"] == "House member" and "role" not in h

# gate: an insider with no record is still proposed (they rarely trade 20 times); a losing record keeps them out
m = {"records": {"Loser": [25, 5, -1.0]}}
assert copier.member_gate(m, "New", 50, [], proven_only=False) == 50 and copier.member_gate(m, "New", 50, []) is None
assert copier.member_gate(m, "Loser", 50, [], proven_only=False) is None
print("insider tests pass")
assert insider.tradable("AAPL") and insider.tradable("BRK.B") and not insider.tradable("AB-LEND") and not insider.tradable("") and not insider.tradable("TOOLONG")
print("ticker filter tests pass")

# exits: insider autopilot copies are held 2 weeks (no take profit, -15% safety stop); House and your own copies unchanged
assert copier.stock_rule({"auto": True, "feats": {"source": "company insider"}}) == (None, 0.15, 14)
assert copier.stock_rule({"auto": True, "feats": {"source": "House member"}}) == (copier.STOCK_TP, copier.STOCK_STOP, copier.STOCK_MAX_DAYS)
assert copier.stock_rule({"auto": False, "feats": {"source": "company insider"}}) == (copier.STOCK_TP, None, None)
print("insider exit rule tests pass")

# the autopilot buys a stock once, even when several insiders file buys of it; you can still add more by hand
import tempfile as _tf, pathlib as _pl, json as _js
copier.SOURCES_FILE = _pl.Path(_tf.mkdtemp()) / "src.json"
copier.alpaca = lambda *a, **k: (200, {})
_placed = []
copier.place = lambda *a: _placed.append(a) or (200, {})
_buy = {"trade": {"Representative": "Doe Jane A", "Ticker": "DOX", "AssetType": "ST", "Transaction": "Purchase", "Range": "$50,000 - $50,000"},
        "feats": {"source": "company insider"}}
assert copier.decide_pelosi(_buy, auto=True).startswith("Done") and len(_placed) == 1
assert "not bought again" in copier.decide_pelosi({**_buy, "trade": {**_buy["trade"], "Representative": "Roe Ann"}}, auto=True) and len(_placed) == 1
assert copier.decide_pelosi(_buy).startswith("Done") and len(_placed) == 2
print("buy-once tests pass")
assert not insider.tradable("NONE") and insider.tradable("NOW")
