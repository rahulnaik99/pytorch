"""
fundamentals.py
───────────────
Banking-sector fundamental data layer for the trading LLM pipeline.

Loads pre-cached quarterly balance sheet / income data from
data/fundamentals/<SYMBOL>.json and summarizes it as % QoQ changes + trend
labels — no raw ₹ values ever reach the prompt.

JSON schema (one file per symbol):
{
  "symbol": "HDFCBANK",
  "type": "equity",
  "sector": "Banking",
  "quarters": [
    {
      "period":            "Q2FY25",
      "period_end":        "2024-09-30",
      "announcement_date": "2024-10-19",   <- look-ahead guard
      "nii_cr":            31500,          <- Net Interest Income (₹ Cr)
      "pat_cr":            16820,          <- Profit After Tax (₹ Cr)
      "total_assets_cr":   3850000,        <- Total Assets (₹ Cr)
      "shareholders_equity_cr": 340000,
      "promoter_pct":      26.2,           <- % holding (from SEBI quarterly filing)
      "fii_pct":           32.5,
      "dii_pct":           18.3
    },
    ...  (up to 16 quarters)
  ]
}

Quarters must be ordered newest-first (index 0 = most recent quarter).
"""
from __future__ import annotations

import json
import os
from pathlib import Path


_INDEX_SYMBOLS: frozenset[str] = frozenset({
    "BANKNIFTY", "NIFTYBANK", "NIFTY50", "SENSEX",
    "NIFTY_BANK", "NIFTY_50", "INDIA50", "NIFTYNEXT50",
})


class FundamentalAnalyzer:
    """Load and summarize banking-sector quarterly fundamentals."""

    def __init__(self, fundamentals_dir: str = "data/fundamentals"):
        self._dir = Path(fundamentals_dir)
        self._cache: dict[str, dict] = {}

    # ── Public API ────────────────────────────────────────────────────

    def is_index(self, symbol: str) -> bool:
        return symbol.upper() in _INDEX_SYMBOLS

    def _load(self, symbol: str) -> dict | None:
        key = symbol.upper()
        if key in self._cache:
            return self._cache[key]
        path = self._dir / f"{key}.json"
        if not path.exists():
            self._cache[key] = None
            return None
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            self._cache[key] = data
            return data
        except Exception as e:
            print(f"  [warn] fundamentals: failed to load {path}: {e}")
            self._cache[key] = None
            return None

    def get_quarters(self, symbol: str, as_of: str, n: int = 4) -> list[dict]:
        """Return last n quarters with announcement_date < as_of (look-ahead safe)."""
        if self.is_index(symbol):
            return []
        data = self._load(symbol)
        if not data:
            return []
        visible = [
            q for q in data.get("quarters", [])
            if q.get("announcement_date", "9999") < as_of
        ]
        # quarters should be newest-first; take at most n
        return visible[:n]

    def summarize(self, symbol: str, as_of: str) -> dict | None:
        """
        Return compact summary dict for prompt, or None if index / no data.

        Keys:
          symbol, as_of_quarter,
          nii_qoq_pct, nii_trend, nii_streak,
          pat_qoq_pct, pat_trend, pat_streak,
          assets_qoq_pct, assets_trend,
          promoter_pct, promoter_delta_pp,
          fii_pct, fii_delta_pp,
          dii_pct, dii_delta_pp
        """
        if self.is_index(symbol):
            return None

        # Get 5 quarters: 4 for trend + 1 for prior QoQ baseline
        quarters = self.get_quarters(symbol, as_of, n=5)
        if len(quarters) < 2:
            return None

        latest = quarters[0]

        def _qoq_pct(key: str, idx_new: int, idx_old: int) -> float | None:
            v_new = latest[key] if idx_new == 0 else quarters[idx_new].get(key)
            q_old = quarters[idx_old] if idx_old < len(quarters) else None
            if q_old is None:
                return None
            v_old = q_old.get(key)
            if not v_old or v_old == 0:
                return None
            v_new = v_new or 0
            return round((v_new - v_old) / abs(v_old) * 100, 1)

        # QoQ % changes for latest quarter
        nii_qoq   = _qoq_pct("nii_cr",                0, 1)
        pat_qoq   = _qoq_pct("pat_cr",                0, 1)
        assets_qoq = _qoq_pct("total_assets_cr",      0, 1)

        # Build QoQ series for trend (newest first, up to 4 entries)
        def _series(key: str) -> list[float]:
            vals = [q.get(key) for q in quarters]
            changes = []
            for i in range(len(vals) - 1):
                if vals[i] and vals[i + 1] and vals[i + 1] != 0:
                    changes.append(round((vals[i] - vals[i + 1]) / abs(vals[i + 1]) * 100, 1))
            return changes

        nii_series    = _series("nii_cr")
        pat_series    = _series("pat_cr")
        assets_series = _series("total_assets_cr")

        nii_trend,    nii_streak    = _classify_trend(nii_series)
        pat_trend,    pat_streak    = _classify_trend(pat_series)
        assets_trend, _             = _classify_trend(assets_series)
        # Rename assets trend to be more descriptive
        assets_trend = "growing" if assets_trend == "rising" else (
                        "shrinking" if assets_trend == "falling" else assets_trend)

        # Holding deltas (pp = percentage points)
        def _delta_pp(key: str) -> float | None:
            v_new = latest.get(key)
            v_old = quarters[1].get(key) if len(quarters) > 1 else None
            if v_new is None or v_old is None:
                return None
            return round(v_new - v_old, 2)

        return {
            "symbol":         symbol.upper(),
            "as_of_quarter":  latest.get("period", "—"),
            "nii_qoq_pct":    nii_qoq,
            "nii_trend":      nii_trend,
            "nii_streak":     nii_streak,
            "pat_qoq_pct":    pat_qoq,
            "pat_trend":      pat_trend,
            "pat_streak":     pat_streak,
            "assets_qoq_pct": assets_qoq,
            "assets_trend":   assets_trend,
            "promoter_pct":   latest.get("promoter_pct"),
            "promoter_delta_pp": _delta_pp("promoter_pct"),
            "fii_pct":        latest.get("fii_pct"),
            "fii_delta_pp":   _delta_pp("fii_pct"),
            "dii_pct":        latest.get("dii_pct"),
            "dii_delta_pp":   _delta_pp("dii_pct"),
        }


# ── Trend classification ──────────────────────────────────────────────

def _classify_trend(pct_changes: list[float]) -> tuple[str, int]:
    """
    Classify a list of QoQ % changes (newest first) into trend + streak.

    Returns:
      ("rising",  n) — n consecutive quarters with QoQ > +0.5%
      ("falling", n) — n consecutive quarters with QoQ < -0.5%
      ("stable",  0) — all changes within ±0.5%
      ("mixed",   0) — no consistent pattern
    """
    if not pct_changes:
        return "mixed", 0

    THRESHOLD = 0.5

    pos = [x > THRESHOLD  for x in pct_changes]
    neg = [x < -THRESHOLD for x in pct_changes]

    if all(pos):
        return "rising", len(pos)
    if all(neg):
        return "falling", len(neg)

    # Count leading streak
    pos_streak = 0
    for v in pct_changes:
        if v > THRESHOLD:
            pos_streak += 1
        else:
            break
    neg_streak = 0
    for v in pct_changes:
        if v < -THRESHOLD:
            neg_streak += 1
        else:
            break

    if pos_streak >= 2:
        return "rising", pos_streak
    if neg_streak >= 2:
        return "falling", neg_streak

    if all(abs(x) <= THRESHOLD for x in pct_changes):
        return "stable", 0

    return "mixed", 0


if __name__ == "__main__":
    import sys
    symbol = sys.argv[1] if len(sys.argv) > 1 else "CANBK"
    as_of  = sys.argv[2] if len(sys.argv) > 2 else "2024-08-19"
    fa = FundamentalAnalyzer()
    fd = fa.summarize(symbol, as_of)
    if fd:
        print(json.dumps(fd, indent=2))
    else:
        print(f"No fundamental data for {symbol} as of {as_of}")
