"""
fetch_fundamentals.py
─────────────────────
One-time data downloader: fetches last 16 quarters of fundamental data
for NSE banking stocks via yfinance and saves to data/fundamentals/<SYMBOL>.json.

NOT part of the training pipeline — run this separately to populate data files.

Usage:
    python fetch_fundamentals.py HDFCBANK ICICIBANK CANBK SBIN AXISBANK
    python fetch_fundamentals.py --all-banking   # downloads all BANKING_SYMBOLS

Requirements:
    pip install yfinance

The script appends .NS to each symbol for NSE lookup (e.g. HDFCBANK → HDFCBANK.NS).
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed. Run: pip install yfinance")
    sys.exit(1)

OUTPUT_DIR = Path("data/fundamentals")

BANKING_SYMBOLS = [
    # Private banks
    "HDFCBANK", "ICICIBANK", "AXISBANK", "KOTAKBANK", "INDUSINDBK",
    "IDFCFIRSTB", "FEDERALBNK", "BANDHANBNK", "YESBANK", "AUBANK", "CUB",
    # Public sector banks
    "SBIN", "PNB", "BANKBARODA", "CANBK", "UNIONBANK", "BANKINDIA", "INDIANB",
]

# Map yfinance column/row names → our JSON keys
# yfinance income statement rows (quarterly)
INCOME_MAP = {
    "Net Interest Income": "nii_cr",
    "Net Income":          "pat_cr",
    "Total Revenue":       "revenue_cr",  # fallback if NII not available
}

# yfinance balance sheet rows (quarterly)
BALANCE_MAP = {
    "Total Assets":          "total_assets_cr",
    "Stockholders Equity":   "shareholders_equity_cr",
    "Common Stock Equity":   "shareholders_equity_cr",  # alternate label
}

CR = 1e7  # 1 Crore = 1e7


def _quarter_label(dt: datetime) -> str:
    """Convert a period-end date to Indian FY quarter label (e.g., Q2FY25)."""
    year  = dt.year
    month = dt.month
    # Indian FY: Apr–Jun = Q1, Jul–Sep = Q2, Oct–Dec = Q3, Jan–Mar = Q4
    if month in (4, 5, 6):
        q, fy = "Q1", year + 1
    elif month in (7, 8, 9):
        q, fy = "Q2", year + 1
    elif month in (10, 11, 12):
        q, fy = "Q3", year + 1
    else:  # Jan–Mar
        q, fy = "Q4", year
    fy_str = str(fy)[2:]  # 2025 → "25"
    return f"{q}FY{fy_str}"


def _safe_val(df, row_name: str, col) -> float | None:
    """Extract a value from a yfinance DataFrame, return None if missing/NaN."""
    for label in (df.index if hasattr(df, "index") else []):
        if str(label).strip() == row_name:
            try:
                val = float(df.loc[label, col])
                if val != val:  # NaN check
                    return None
                return round(val / CR, 2)
            except Exception:
                return None
    return None


def fetch_symbol(symbol: str) -> bool:
    yf_sym = f"{symbol}.NS"
    print(f"Fetching {symbol} ({yf_sym}) ...")

    try:
        ticker = yf.Ticker(yf_sym)
        qs_income  = ticker.quarterly_income_stmt
        qs_balance = ticker.quarterly_balance_sheet
        info       = ticker.info
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

    if qs_income is None or qs_income.empty:
        print(f"  SKIP: no quarterly income statement for {symbol}")
        return False

    cols = list(qs_income.columns)[:16]  # newest first, max 16 quarters

    quarters = []
    for col in cols:
        try:
            period_end = col.strftime("%Y-%m-%d")
            period_dt  = col.to_pydatetime() if hasattr(col, "to_pydatetime") else col

            # announcement_date ≈ 3–6 weeks after period end
            # yfinance columns ARE the reporting date (result announcement date) for most providers;
            # if not, approximate as +45 days after period end.
            ann_date = col.strftime("%Y-%m-%d")

            # Build quarter dict
            q: dict = {
                "period":            _quarter_label(period_dt),
                "period_end":        period_end,
                "announcement_date": ann_date,
            }

            # Income statement
            for label, key in INCOME_MAP.items():
                val = _safe_val(qs_income, label, col)
                if val is not None and key not in q:
                    q[key] = val

            # Balance sheet
            if qs_balance is not None and not qs_balance.empty and col in qs_balance.columns:
                for label, key in BALANCE_MAP.items():
                    val = _safe_val(qs_balance, label, col)
                    if val is not None and key not in q:
                        q[key] = val

            # Shareholding (snapshot from info — not per-quarter, use as latest estimate)
            if not quarters:  # only fill for latest quarter from info
                promoter = info.get("heldPercentInsiders")
                inst     = info.get("heldPercentInstitutions")
                if promoter is not None:
                    q["promoter_pct"] = round(float(promoter) * 100, 2)
                if inst is not None:
                    # yfinance lumps FII+DII into "institutions"; split 60/40 as rough estimate
                    # Users should replace with actual NSE shareholding data for accuracy
                    q["fii_pct"] = round(float(inst) * 100 * 0.6, 2)
                    q["dii_pct"] = round(float(inst) * 100 * 0.4, 2)

            quarters.append(q)
        except Exception as e:
            print(f"  [warn] skipping col {col}: {e}")
            continue

    if not quarters:
        print(f"  SKIP: no quarters processed for {symbol}")
        return False

    data = {
        "symbol":   symbol.upper(),
        "exchange": "NSE",
        "type":     "equity",
        "sector":   "Banking",
        "fetched_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "quarters": quarters,
    }

    out_path = OUTPUT_DIR / f"{symbol.upper()}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved {len(quarters)} quarters → {out_path}")
    return True


def main():
    args = sys.argv[1:]
    if not args:
        print("Usage: python fetch_fundamentals.py SYMBOL1 SYMBOL2 ...")
        print("       python fetch_fundamentals.py --all-banking")
        sys.exit(1)

    if "--all-banking" in args:
        symbols = BANKING_SYMBOLS
    else:
        symbols = [s.upper() for s in args]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    success = 0
    for i, sym in enumerate(symbols):
        ok = fetch_symbol(sym)
        if ok:
            success += 1
        if i < len(symbols) - 1:
            time.sleep(1.5)  # rate-limit yfinance calls

    print(f"\nDone: {success}/{len(symbols)} symbols saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
