"""
technicals.py
─────────────
TechnicalsAnalyzer: load candles, compute indicators, detect zones.

Usage:
    ta = TechnicalsAnalyzer()
    df = ta.load_candles("CANBK", "2024-01-10", lookback_days=30)
    tech = ta.compute(df)
    vc   = ta.vix_context("2024-01-10")
    zc   = ta.compute_zones("CANBK", "2024-01-10")

DATA_SOURCE env var: "csv" (default data/prices/) or "yfinance"
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env", override=False)
except ImportError:
    pass

DATA_SOURCE = os.environ.get("DATA_SOURCE", "yfinance").strip().lower()
CSV_DIR     = os.environ.get("CSV_DIR", "data/prices")
VIX_CSV     = os.environ.get("VIX_CSV", "")
VIX_LOW      = float(os.environ.get("VIX_LOW", 13))
VIX_NORMAL   = float(os.environ.get("VIX_NORMAL", 18))
VIX_ELEVATED = float(os.environ.get("VIX_ELEVATED", 25))


class TechnicalsAnalyzer:
    def __init__(self, csv_dir: str = CSV_DIR, vix_csv: str = ""):
        self.csv_dir = csv_dir
        self.vix_csv = vix_csv or VIX_CSV or str(Path(csv_dir) / "INDIAVIX.csv")
        self._raw_cache: dict[str, pd.DataFrame] = {}
        self._vix_cache: pd.DataFrame | None = None
        self._vix_loaded = False

    # ── Candle loading ────────────────────────────────────────────────

    def _load_raw(self, symbol: str) -> pd.DataFrame:
        if symbol in self._raw_cache:
            return self._raw_cache[symbol]

        if DATA_SOURCE == "csv":
            path = Path(self.csv_dir) / f"{symbol}.csv"
            raw = self._load_csv(path)
        else:
            import yfinance as yf
            raw = yf.download(f"{symbol}.NS", start="2015-01-01",
                              end=(pd.Timestamp.today() + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                              interval="1d", progress=False, auto_adjust=True)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [c[0].lower() for c in raw.columns]
            else:
                raw.columns = [c.lower() for c in raw.columns]
            raw.index = pd.to_datetime(raw.index)

        self._raw_cache[symbol] = raw
        return raw

    def _load_csv(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")
        df = pd.read_csv(path)
        df.columns = [str(c).strip().lower() for c in df.columns]

        # Detect datetime column.
        # CANBK format:    date, open, high, low, close, volume
        # INDIAVIX format: Ticker, Date/Time, Ticker New, Open, High, Low, Close, Volume, Open Interest
        _EXACT = ("datetime", "date", "time", "timestamp", "date/time")
        dt_col = next((c for c in df.columns if c in _EXACT),
                      next((c for c in df.columns if "date" in c or "time" in c), df.columns[0]))

        # Parse dates — handles both ISO (2024-01-10) and Indian dayfirst (30-12-2022)
        try:
            df[dt_col] = pd.to_datetime(df[dt_col], dayfirst=True, format="mixed", errors="coerce")
        except TypeError:
            df[dt_col] = pd.to_datetime(df[dt_col], dayfirst=True, errors="coerce")

        df = df.dropna(subset=[dt_col]).set_index(dt_col).sort_index()
        df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close",
                                 "v": "volume", "vol": "volume",
                                 "adj close": "close", "adj_close": "close"})
        need = ["open", "high", "low", "close"]
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise ValueError(f"CSV {path} missing columns {missing}")
        if "volume" not in df.columns:
            df["volume"] = 0.0
        df = df[need + ["volume"]].apply(pd.to_numeric, errors="coerce").dropna(subset=need)
        df["volume"] = df["volume"].fillna(0.0)
        # Resample to daily if minute data
        if len(df) > 0:
            span = (df.index[-1] - df.index[0]).days + 1
            if len(df) / max(span, 1) > 2:
                df = self._resample_ohlcv(df, "1D")
        df.index = pd.to_datetime(df.index.date)
        return df

    def _resample_ohlcv(self, df: pd.DataFrame, rule: str) -> pd.DataFrame:
        return df.resample(rule).agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna(subset=["open", "high", "low", "close"])

    def load_candles(self, symbol: str, as_of: str, lookback_days: int = 90) -> pd.DataFrame:
        """Last `lookback_days` daily bars on or before `as_of`."""
        hist = self._load_raw(symbol)
        df = hist[hist.index <= pd.Timestamp(as_of)].tail(lookback_days).copy()
        if df.empty:
            raise ValueError(f"No candle data for {symbol} on/before {as_of}")
        return df

    def load_forward(self, symbol: str, from_date: str, n: int = 5) -> pd.DataFrame:
        """Next `n` daily bars strictly after `from_date` (for outcome simulation)."""
        hist = self._load_raw(symbol)
        return hist[hist.index > pd.Timestamp(from_date)].head(n).copy()

    def load_intraday(self, symbol: str, as_of: str, rule: str, n_bars: int = 160) -> pd.DataFrame:
        """Resample minute data to `rule` (e.g. '15min', '1h'), up to `as_of`."""
        raw = self._raw_cache.get(symbol)
        if raw is None:
            if DATA_SOURCE != "csv":
                return pd.DataFrame()
            path = Path(self.csv_dir) / f"{symbol}.csv"
            if not path.exists():
                return pd.DataFrame()
            raw = self._load_csv_raw_minutely(path)
            if raw is None or raw.empty:
                return pd.DataFrame()
        cutoff = pd.Timestamp(as_of) + pd.Timedelta(days=1)
        sliced = raw[raw.index < cutoff]
        if sliced.empty:
            return pd.DataFrame()
        return self._resample_ohlcv(sliced, rule).tail(n_bars)

    def _load_csv_raw_minutely(self, path: Path) -> pd.DataFrame | None:
        """Load CSV without daily resampling — used for zone detection."""
        if not path.exists():
            return None
        df = pd.read_csv(path)
        df.columns = [str(c).strip().lower() for c in df.columns]
        _EXACT = ("datetime", "date", "time", "timestamp", "date/time")
        dt_col = next((c for c in df.columns if c in _EXACT),
                      next((c for c in df.columns if "date" in c or "time" in c), df.columns[0]))
        try:
            df[dt_col] = pd.to_datetime(df[dt_col], dayfirst=True, format="mixed", errors="coerce")
        except TypeError:
            df[dt_col] = pd.to_datetime(df[dt_col], dayfirst=True, errors="coerce")
        df = df.dropna(subset=[dt_col]).set_index(dt_col).sort_index()
        df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close",
                                 "v": "volume", "vol": "volume"})
        need = ["open", "high", "low", "close"]
        if any(c not in df.columns for c in need):
            return None
        if "volume" not in df.columns:
            df["volume"] = 0.0
        return df[need + ["volume"]].apply(pd.to_numeric, errors="coerce").dropna(subset=need)

    # ── Technicals ────────────────────────────────────────────────────

    def compute(self, df: pd.DataFrame) -> dict:
        """Compute all technical indicators from a daily OHLCV DataFrame."""
        close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]
        cur  = float(close.iloc[-1])
        e50  = float(close.ewm(span=50,  adjust=False).mean().iloc[-1])
        e100 = float(close.ewm(span=100, adjust=False).mean().iloc[-1])
        e200 = float(close.ewm(span=200, adjust=False).mean().iloc[-1])
        tp   = (high + low + close) / 3
        vwap = float((tp * vol).cumsum().iloc[-1] / vol.cumsum().iloc[-1])

        pivots = self._detect_pivots(high) + self._detect_pivots(low)
        trend, structure = self._classify_structure(pivots)
        sup = [(l, h, m) for l, h, m in self._cluster_zones(pivots, "low")  if m < cur]
        res = [(l, h, m) for l, h, m in self._cluster_zones(pivots, "high") if m > cur]
        ns  = max(sup, key=lambda z: z[2]) if sup else None
        nr  = min(res, key=lambda z: z[2]) if res else None

        def zf(z): return {"low": z[0], "high": z[1], "mid": z[2]} if z else None

        stack = ("bullish (50>100>200)" if e50 > e100 > e200 else
                 "bearish (50<100<200)" if e50 < e100 < e200 else "mixed")
        recent = (df.tail(5).reset_index()
                  .rename(columns={"Date": "date", "index": "date"})
                  [["date", "open", "high", "low", "close", "volume"]]
                  .assign(date=lambda d: d["date"].astype(str)).round(2)
                  .to_dict(orient="records"))

        return {
            "close": cur,
            "ema_50": round(e50, 2), "ema_100": round(e100, 2), "ema_200": round(e200, 2),
            "vwap": round(vwap, 2),
            "ema_stack": stack, "above_vwap": cur > vwap, "above_ema200": cur > e200,
            "trend": trend, "structure": structure,
            "nearest_support": zf(ns), "nearest_resistance": zf(nr),
            "support_zones":    [zf(z) for z in sup[-3:]],
            "resistance_zones": [zf(z) for z in res[:3]],
            "swing_highs": [{"date": str(d.date()), "price": round(p, 2)}
                            for t, d, p in pivots if t == "high"][-3:],
            "swing_lows":  [{"date": str(d.date()), "price": round(p, 2)}
                            for t, d, p in pivots if t == "low"][-3:],
            "levels": self._level_breaks(df, cur),
            "recent_candles": recent,
        }

    def _detect_pivots(self, series: pd.Series, order: int = 2):
        pivots = []
        for i in range(order, len(series) - order):
            win = series.iloc[i - order: i + order + 1]
            if series.iloc[i] == win.max():
                pivots.append(("high", series.index[i], float(series.iloc[i])))
            elif series.iloc[i] == win.min():
                pivots.append(("low", series.index[i], float(series.iloc[i])))
        return pivots

    def _classify_structure(self, pivots):
        sh = sorted([(d, p) for t, d, p in pivots if t == "high"])
        sl = sorted([(d, p) for t, d, p in pivots if t == "low"])
        if len(sh) < 2 or len(sl) < 2:
            return "ranging", []
        hp = [p for _, p in sh[-3:]]; lp = [p for _, p in sl[-3:]]
        hh = all(a < b for a, b in zip(hp, hp[1:]))
        hl = all(a < b for a, b in zip(lp, lp[1:]))
        lh = all(a > b for a, b in zip(hp, hp[1:]))
        ll = all(a > b for a, b in zip(lp, lp[1:]))
        if hh and hl: return "uptrend",    ["HH", "HL"]
        if lh and ll: return "downtrend",  ["LH", "LL"]
        if hh and ll: return "expanding",  ["HH", "LL"]
        if lh and hl: return "contracting",["LH", "HL"]
        return "ranging", []

    def _cluster_zones(self, pivots, kind, bw=0.008):
        prices = sorted([p for t, _, p in pivots if t == kind])
        if not prices:
            return []
        zones, c = [], [prices[0]]
        for p in prices[1:]:
            if (p - c[-1]) / c[-1] < bw:
                c.append(p)
            else:
                zones.append((round(min(c), 2), round(max(c), 2), round(float(np.mean(c)), 2)))
                c = [p]
        zones.append((round(min(c), 2), round(max(c), 2), round(float(np.mean(c)), 2)))
        return zones

    def _level_breaks(self, df: pd.DataFrame, cur: float) -> dict:
        out = {}
        if len(df) < 2:
            return out
        today = df.iloc[-1]
        th, tl = float(today["high"]), float(today["low"])
        pdh = float(df.iloc[-2]["high"]); pdl = float(df.iloc[-2]["low"])
        out["pdh"] = round(pdh, 2); out["pdl"] = round(pdl, 2)
        out["pdh_state"] = ("broke"    if cur > pdh else
                            "rejected" if th >= pdh and cur < pdh else "below")
        out["pdl_state"] = ("lost"     if cur < pdl else
                            "held"     if tl <= pdl and cur > pdl else "above")
        if len(df) >= 3:
            w2 = df.iloc[-3:-1]
            d2h = float(w2["high"].max()); d2l = float(w2["low"].min())
            out["two_day_high"] = round(d2h, 2); out["two_day_low"] = round(d2l, 2)
            out["two_day_high_broken"] = cur > d2h
            out["two_day_low_broken"]  = cur < d2l
        if len(df) >= 4:
            w3 = df.iloc[-4:-1]
            d3h = float(w3["high"].max()); d3l = float(w3["low"].min())
            out["three_day_high"] = round(d3h, 2); out["three_day_low"] = round(d3l, 2)
            out["three_day_high_broken"] = cur > d3h
            out["three_day_low_broken"]  = cur < d3l
        try:
            wk = df.resample("1W").agg({"high": "max", "low": "min"}).dropna()
            if len(wk) >= 2:
                pwh = float(wk.iloc[-2]["high"]); pwl = float(wk.iloc[-2]["low"])
                out["week_high"] = round(pwh, 2); out["week_low"] = round(pwl, 2)
                out["week_high_state"] = ("broke"    if cur > pwh else
                                          "rejected" if th >= pwh and cur < pwh else "below")
                out["week_low_state"]  = ("lost"     if cur < pwl else
                                          "held"     if tl <= pwl and cur > pwl else "above")
        except Exception:
            pass
        return out

    # ── VIX context ───────────────────────────────────────────────────

    def vix_context(self, as_of: str) -> dict:
        """VIX regime as of `as_of` (no look-ahead). Returns {} if no VIX file."""
        if not self._vix_loaded:
            self._vix_loaded = True
            p = Path(self.vix_csv)
            if p.exists():
                try:
                    self._vix_cache = self._load_csv(p)
                except Exception:
                    self._vix_cache = None
        hist = self._vix_cache
        if hist is None or hist.empty:
            return {}
        s = hist["close"][hist.index <= pd.Timestamp(as_of)]
        if len(s) < 2:
            return {}
        level = float(s.iloc[-1])
        yr = s.tail(252)
        pctile = round(float((yr < level).mean()) * 100)
        win20 = s.tail(20)
        lo, hi = float(win20.min()), float(win20.max())
        pos20 = ("top"    if hi > lo and (level - lo) / (hi - lo) > 0.66 else
                 "bottom" if hi > lo and (level - lo) / (hi - lo) < 0.33 else "mid")
        chg5 = level - float(s.iloc[-6]) if len(s) >= 6 else 0.0
        direction = "rising" if chg5 > 0.3 else "falling" if chg5 < -0.3 else "flat"
        regime = ("LOW"      if level < VIX_LOW      else
                  "NORMAL"   if level < VIX_NORMAL   else
                  "ELEVATED" if level < VIX_ELEVATED else "EXTREME")
        return {"vix": round(level, 2), "pctile_1y": pctile, "regime": regime,
                "dir_5d": direction, "pos_20d": pos20}

    # ── Demand/supply zones ───────────────────────────────────────────

    def compute_zones(self, symbol: str, as_of: str) -> dict:
        """5min + 1h demand/supply zones with confluence. Returns {} for daily-only data."""
        tf5m = self.load_intraday(symbol, as_of, "5min", 200)
        tf1h = self.load_intraday(symbol, as_of, "1h",   160)
        if tf5m.empty and tf1h.empty:
            return {}
        price = float((tf5m if not tf5m.empty else tf1h)["close"].iloc[-1])
        near5m = self._nearest_zones(self._detect_zones(tf5m), price) if not tf5m.empty else {}
        near1h = self._nearest_zones(self._detect_zones(tf1h), price) if not tf1h.empty else {}
        return {"tf_5min": near5m, "tf_1h": near1h,
                "confluence": self._confluence(near5m, near1h)}

    def _detect_zones(self, df: pd.DataFrame, base_max: int = 3,
                      impulse_mult: float = 3.0, scan: int = 120,
                      allow_retest: bool = True) -> list[dict]:
        if df is None or len(df) < base_max + 3:
            return []
        d = df.tail(scan).reset_index(drop=True)
        highs, lows = d["high"].values, d["low"].values
        opens, closes = d["open"].values, d["close"].values
        n = len(d)
        zones = []
        for i in range(base_max, n - 1):
            impulse_rng = highs[i] - lows[i]
            if impulse_rng <= 0:
                continue
            for nb in range(1, base_max + 1):
                base = slice(i - nb, i)
                base_hi = highs[base].max(); base_lo = lows[base].min()
                base_rng = base_hi - base_lo
                if base_rng <= 0 or impulse_rng < impulse_mult * base_rng:
                    continue
                bullish = closes[i] > opens[i] and closes[i] > base_hi
                bearish = closes[i] < opens[i] and closes[i] < base_lo
                if not (bullish or bearish):
                    continue
                kind = "demand" if bullish else "supply"
                lo, hi = float(base_lo), float(base_hi)
                touches, mitigated = 0, False
                for j in range(i + 1, n):
                    if lows[j] <= hi and highs[j] >= lo:
                        touches += 1
                    if kind == "demand" and closes[j] < lo:
                        mitigated = True; break
                    if kind == "supply" and closes[j] > hi:
                        mitigated = True; break
                valid = (not mitigated) and (touches <= (1 if allow_retest else 0))
                zones.append({"kind": kind, "low": round(lo, 2), "high": round(hi, 2),
                              "formed": i, "touches": touches,
                              "fresh": touches == 0, "valid": valid})
                break
        return zones

    def _nearest_zones(self, zones: list[dict], price: float) -> dict:
        demands  = [z for z in zones if z["kind"] == "demand" and z["valid"] and z["high"] <= price * 1.002]
        supplies = [z for z in zones if z["kind"] == "supply" and z["valid"] and z["low"]  >= price * 0.998]
        nd = max(demands,  key=lambda z: z["high"]) if demands  else None
        ns = min(supplies, key=lambda z: z["low"])  if supplies else None
        out = {}
        if nd:
            nd = dict(nd); nd["dist_pct"] = round((nd["high"] / price - 1) * 100, 2)
            out["demand"] = nd
        if ns:
            ns = dict(ns); ns["dist_pct"] = round((ns["low"]  / price - 1) * 100, 2)
            out["supply"] = ns
        return out

    def _confluence(self, near_short: dict, near_big: dict) -> dict:
        def overlap(a, b):
            if not a or not b: return False
            return a["low"] <= b["high"] and b["low"] <= a["high"]
        return {
            "demand": overlap(near_short.get("demand"), near_big.get("demand")),
            "supply": overlap(near_short.get("supply"), near_big.get("supply")),
        }
