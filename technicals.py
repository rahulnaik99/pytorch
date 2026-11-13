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
        self._raw_cache:    dict[str, pd.DataFrame] = {}   # symbol → daily bars
        self._minute_cache: dict[str, pd.DataFrame] = {}   # symbol → minute bars
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

        # Parse dates — try ISO format first (NSE default), then Indian dayfirst, then mixed.
        # Avoids dayfirst=True with format="mixed" which swaps D/M on pandas 3.x when day ≤ 12.
        _dt_parsed = None
        for _fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S",
                     "%d-%m-%Y", "%d-%m-%Y %H:%M", "%d-%m-%Y %H:%M:%S"):
            try:
                _p = pd.to_datetime(df[dt_col], format=_fmt)
                if not _p.isna().all():
                    _dt_parsed = _p
                    break
            except (ValueError, TypeError):
                pass
        if _dt_parsed is None:
            _dt_parsed = pd.to_datetime(df[dt_col], dayfirst=False, format="mixed", errors="coerce")
        df[dt_col] = _dt_parsed

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
        """Next `n` trading days of minute bars strictly after `from_date`.

        Returns minute-level bars so simulate_outcome() can resolve intraday
        SL/TP order correctly. Falls back to daily bars if minute data unavailable.
        """
        minute = self._load_minute(symbol)
        if not minute.empty:
            fwd = minute[minute.index.normalize() > pd.Timestamp(from_date)]
            # Take bars from the next n unique trading days
            trading_days = sorted(fwd.index.normalize().unique())[:n]
            if trading_days:
                cutoff = trading_days[-1] + pd.Timedelta(days=1)
                return fwd[fwd.index.normalize() <= trading_days[-1]].copy()
        # Fallback: daily bars
        hist = self._load_raw(symbol)
        return hist[hist.index > pd.Timestamp(from_date)].head(n).copy()

    def _load_minute(self, symbol: str) -> pd.DataFrame:
        """Return cached minute-level DataFrame for symbol (CSV mode only)."""
        if symbol in self._minute_cache:
            return self._minute_cache[symbol]
        if DATA_SOURCE != "csv":
            self._minute_cache[symbol] = pd.DataFrame()
            return self._minute_cache[symbol]
        path = Path(self.csv_dir) / f"{symbol}.csv"
        raw = self._load_csv_raw_minutely(path)
        self._minute_cache[symbol] = raw if raw is not None else pd.DataFrame()
        return self._minute_cache[symbol]

    def load_intraday(self, symbol: str, as_of: str, rule: str, n_bars: int = 160) -> pd.DataFrame:
        """Resample minute data to `rule` (e.g. '15min', '1h'), up to `as_of`."""
        raw = self._load_minute(symbol)
        if raw.empty:
            return pd.DataFrame()
        cutoff = pd.Timestamp(as_of) + pd.Timedelta(days=1)
        sliced = raw[raw.index < cutoff]
        if sliced.empty:
            return pd.DataFrame()
        return self._resample_ohlcv(sliced, rule).tail(n_bars)

    def load_intraday_for_slot(self, symbol: str, date: str, rule: str,
                               n_bars: int, slot: str = "close") -> pd.DataFrame:
        """Like load_intraday but with a sub-day cutoff matching the decision slot.

        open  → bars strictly before 09:15 on `date` (no same-day bars at open)
        mid   → bars up to 12:00 on `date` (morning session only)
        close → bars up to 15:30 on `date` (full session)
        """
        CUTOFFS = {"open": "09:15:00", "mid": "12:00:00", "close": "15:30:00"}
        cutoff = pd.Timestamp(f"{date} {CUTOFFS.get(slot, '15:30:00')}")
        raw = self._load_minute(symbol)
        if raw.empty:
            return pd.DataFrame()
        sliced = raw[raw.index < cutoff]
        if sliced.empty:
            return pd.DataFrame()
        return self._resample_ohlcv(sliced, rule).tail(n_bars)

    def compute_tf(self, df: pd.DataFrame, spans: tuple = (9, 21, 50),
                   label: str = "1h", pivot_order: int = 5) -> dict:
        """Compute intraday technicals for a single timeframe.

        Designed for 1H or 5min bars loaded via load_intraday_for_slot.
        Uses session VWAP (resets daily), skips _level_breaks (meaningless
        on intraday bars; 1H gets prev-session H/L instead).
        """
        if df.empty or len(df) < max(spans):
            raise ValueError(
                f"compute_tf({label}): need >= {max(spans)} bars, got {len(df)}")

        close = df["close"].astype(float)
        high  = df["high"].astype(float)
        low   = df["low"].astype(float)
        vol   = df["volume"].astype(float)
        cur   = float(close.iloc[-1])

        # EMAs with provided spans
        e_fast = float(close.ewm(span=spans[0], adjust=False).mean().iloc[-1])
        e_mid  = float(close.ewm(span=spans[1], adjust=False).mean().iloc[-1])
        e_slow = float(close.ewm(span=spans[2], adjust=False).mean().iloc[-1])
        if e_fast > e_mid > e_slow:
            ema_stack = f"bullish ({spans[0]}>{spans[1]}>{spans[2]})"
        elif e_fast < e_mid < e_slow:
            ema_stack = f"bearish ({spans[0]}<{spans[1]}<{spans[2]})"
        else:
            ema_stack = "mixed"

        # Session VWAP — resets daily; use only bars from the last session date
        last_date = df.index.normalize()[-1]
        sess = df[df.index.normalize() == last_date]
        sess_tp  = (sess["high"] + sess["low"] + sess["close"]) / 3
        sess_vol = sess["volume"].astype(float)
        vwap = float((sess_tp * sess_vol).sum() / max(float(sess_vol.sum()), 1e-9))

        # Structure from full window (type-split pivot detection avoids cross-series mislabels)
        pivots_long = (self._detect_pivots(high, order=pivot_order, pivot_type="high") +
                       self._detect_pivots(low,  order=pivot_order, pivot_type="low"))
        trend, structure = self._classify_structure(pivots_long)

        sup_zones_all = self._cluster_zones(pivots_long, "low")
        res_zones_all = self._cluster_zones(pivots_long, "high")
        def zf(z): return {"low": z[0], "high": z[1], "mid": z[2]} if z else None
        sup = [zf(z) for z in sup_zones_all if z[2] < cur]
        res = [zf(z) for z in res_zones_all if z[2] > cur]
        ns  = sup[-1] if sup else None
        nr  = res[0]  if res else None

        # Recent swings (last 30 bars, order=2 for granular short-term pivots)
        df30 = df.iloc[-30:]
        pivots_short = (self._detect_pivots(df30["high"].astype(float), order=2, pivot_type="high") +
                        self._detect_pivots(df30["low"].astype(float),  order=2, pivot_type="low"))
        swing_highs = [{"date": str(d.date()), "price": round(p, 2)}
                       for t, d, p in sorted(pivots_long, key=lambda x: x[1]) if t == "high"]
        swing_lows  = [{"date": str(d.date()), "price": round(p, 2)}
                       for t, d, p in sorted(pivots_long, key=lambda x: x[1]) if t == "low"]

        # Recent 5 bars for vol analysis
        rc_df = df.iloc[-5:].reset_index().rename(columns={"index": "date"})
        recent_candles = [
            {"date": str(r["date"].date() if hasattr(r["date"], "date") else r["date"]),
             "open": round(float(r["open"]), 2), "high": round(float(r["high"]), 2),
             "low":  round(float(r["low"]),  2), "close": round(float(r["close"]), 2),
             "volume": float(r["volume"])}
            for _, r in rc_df.iterrows()
        ]

        # Volume analysis
        vol_series = df["volume"].astype(float)
        avg5   = float(vol_series.iloc[-5:].mean())
        avg20  = float(vol_series.iloc[-20:].mean() if len(vol_series) >= 20 else vol_series.mean())
        v_ratio = round(float(vol_series.iloc[-1]) / max(avg20, 1), 2)
        p5_chg  = float(close.iloc[-1]) - float(close.iloc[-6] if len(close) >= 6 else close.iloc[0])
        p_dir   = "up" if p5_chg > 0 else "down"
        v_trend = "up" if avg5 / max(avg20, 1) > 1.10 else "down" if avg5 / max(avg20, 1) < 0.90 else "flat"
        vol_sentiment = {
            ("up",   "up"):   "bullish_momentum",
            ("up",   "down"): "exhaustion",
            ("down", "up"):   "bearish_pressure",
            ("down", "down"): "potential_base",
        }.get((p_dir, v_trend), "neutral")
        volume_analysis = {
            "avg_5d": round(avg5), "avg_20d": round(avg20), "ratio": v_ratio,
            "vol_trend": v_trend, "price_dir_5d": p_dir, "sentiment": vol_sentiment,
        }

        # Previous session H/L (1H only; 5min bars span too short to be meaningful)
        levels: dict = {}
        if label == "1h":
            sessions = sorted(df.index.normalize().unique())
            if len(sessions) >= 2:
                prev_sess = df[df.index.normalize() == sessions[-2]]
                pdh = round(float(prev_sess["high"].max()), 2)
                pdl = round(float(prev_sess["low"].min()),  2)
                levels = {
                    "pdh": pdh, "pdl": pdl,
                    "pdh_state": "above" if cur > pdh else "below",
                    "pdl_state": "above" if cur > pdl else "below",
                }

        return {
            "close":    round(cur, 2),
            "ema_fast": round(e_fast, 2),
            "ema_mid":  round(e_mid,  2),
            "ema_slow": round(e_slow, 2),
            "ema_stack": ema_stack,
            "vwap":     round(vwap,  2),
            "trend":    trend,
            "structure": structure,
            "support_zones":      sup[-3:],
            "resistance_zones":   res[:3],
            "nearest_support":    ns,
            "nearest_resistance": nr,
            "swing_highs": swing_highs,
            "swing_lows":  swing_lows,
            "swing_highs_30d": [{"date": str(d.date()), "price": round(p, 2)}
                                 for t, d, p in sorted(pivots_short, key=lambda x: x[1])
                                 if t == "high"],
            "swing_lows_30d":  [{"date": str(d.date()), "price": round(p, 2)}
                                 for t, d, p in sorted(pivots_short, key=lambda x: x[1])
                                 if t == "low"],
            "recent_candles":  recent_candles,
            "volume_analysis": volume_analysis,
            "round_levels":    self._round_levels(cur),
            "levels":          levels,
            "label":           label,
            "n_bars":          len(df),
            "spans":           list(spans),
        }

    def _load_csv_raw_minutely(self, path: Path) -> pd.DataFrame | None:
        """Load CSV without daily resampling — used for zone detection."""
        if not path.exists():
            return None
        df = pd.read_csv(path)
        df.columns = [str(c).strip().lower() for c in df.columns]
        _EXACT = ("datetime", "date", "time", "timestamp", "date/time")
        dt_col = next((c for c in df.columns if c in _EXACT),
                      next((c for c in df.columns if "date" in c or "time" in c), df.columns[0]))
        # Parse dates — ISO first (NSE exports), then Indian dayfirst, then mixed fallback.
        # Avoids dayfirst=True with format="mixed" which swaps D/M on pandas 3.x when day ≤ 12.
        _dt_parsed2 = None
        for _fmt2 in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d",
                      "%d-%m-%Y %H:%M:%S", "%d-%m-%Y %H:%M", "%d-%m-%Y"):
            try:
                _p2 = pd.to_datetime(df[dt_col], format=_fmt2)
                if not _p2.isna().all():
                    _dt_parsed2 = _p2
                    break
            except (ValueError, TypeError):
                pass
        if _dt_parsed2 is None:
            _dt_parsed2 = pd.to_datetime(df[dt_col], dayfirst=False, format="mixed", errors="coerce")
        df[dt_col] = _dt_parsed2
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

    def compute(self, df: pd.DataFrame, decision_slot: str = "close") -> dict:
        """Compute all technical indicators from a daily OHLCV DataFrame.

        decision_slot: which intraday price to use as 'current' price —
            "close" (default), "open", or "mid" ((high+low)/2).
        """
        last = df.iloc[-1]

        # Split to avoid look-ahead: for open/mid the day's close/high/low are unknown
        if decision_slot in ("open", "mid"):
            if len(df) < 2:
                raise ValueError("Need at least 2 bars for open/mid slot computation")
            ind_df = df.iloc[:-1]   # indicators use only data through previous close
            cur = float(last["open"]) if decision_slot == "open" \
                  else (float(last["high"]) + float(last["low"])) / 2
        else:
            ind_df = df
            cur = float(last["close"])

        close = ind_df["close"]
        high  = ind_df["high"]
        low   = ind_df["low"]
        vol   = ind_df["volume"]

        e50  = float(close.ewm(span=50,  adjust=False).mean().iloc[-1])
        e100 = float(close.ewm(span=100, adjust=False).mean().iloc[-1])
        e200 = float(close.ewm(span=200, adjust=False).mean().iloc[-1])
        tp   = (high + low + close) / 3
        vwap = float((tp * vol).cumsum().iloc[-1] / max(float(vol.cumsum().iloc[-1]), 1e-9))

        # ── Two-TF pivot detection (type-restricted to prevent cross-series mislabels) ──
        # Long TF (full ind_df, order=5): only clear major structural pivots
        pivots_long  = (self._detect_pivots(high, order=5, pivot_type="high") +
                        self._detect_pivots(low,  order=5, pivot_type="low"))
        # Short TF (last 30 bars of ind_df, order=2): granular recent swings
        df30 = ind_df.iloc[-30:]
        pivots_short = (self._detect_pivots(df30["high"], order=2, pivot_type="high") +
                        self._detect_pivots(df30["low"],  order=2, pivot_type="low"))

        trend, structure = self._classify_structure(pivots_long)
        sup = [(l, h, m) for l, h, m in self._cluster_zones(pivots_long, "low")  if m < cur]
        res = [(l, h, m) for l, h, m in self._cluster_zones(pivots_long, "high") if m > cur]
        ns  = max(sup, key=lambda z: z[2]) if sup else None
        nr  = min(res, key=lambda z: z[2]) if res else None

        def zf(z): return {"low": z[0], "high": z[1], "mid": z[2]} if z else None

        stack = ("bullish (50>100>200)" if e50 > e100 > e200 else
                 "bearish (50<100<200)" if e50 < e100 < e200 else "mixed")
        recent = (ind_df.tail(5).reset_index()
                  .rename(columns={"Date": "date", "index": "date"})
                  [["date", "open", "high", "low", "close", "volume"]]
                  .assign(date=lambda d: d["date"].astype(str)).round(2)
                  .to_dict(orient="records"))

        # ── Volume trend analysis (uses ind_df to avoid look-ahead) ───
        volume_analysis: dict = {}
        if len(ind_df) >= 6:
            avg_5d  = float(vol.iloc[-5:].mean())
            avg_20d = float(vol.iloc[-20:].mean()) if len(ind_df) >= 20 else avg_5d
            v_ratio = round(avg_5d / avg_20d, 2) if avg_20d > 0 else 1.0
            v_trend = "up"   if v_ratio > 1.10 else "down" if v_ratio < 0.90 else "flat"
            p5_chg  = float(close.iloc[-1]) - float(close.iloc[-6])
            p_dir   = "up" if p5_chg > 0 else "down"
            sentiment = {
                ("up",   "up"):   "bullish_momentum",
                ("up",   "down"): "exhaustion",
                ("down", "up"):   "bearish_pressure",
                ("down", "down"): "potential_base",
            }.get((p_dir, v_trend), "neutral")
            volume_analysis = {
                "avg_5d":       round(avg_5d),
                "avg_20d":      round(avg_20d),
                "ratio":        v_ratio,
                "vol_trend":    v_trend,
                "price_dir_5d": p_dir,
                "sentiment":    sentiment,
            }

        return {
            "close": round(cur, 2),
            "ema_50": round(e50, 2), "ema_100": round(e100, 2), "ema_200": round(e200, 2),
            "vwap": round(vwap, 2),
            "ema_stack": stack, "above_vwap": cur > vwap, "above_ema200": cur > e200,
            "trend": trend, "structure": structure,
            "nearest_support": zf(ns), "nearest_resistance": zf(nr),
            "support_zones":    [zf(z) for z in sup[-3:]],
            "resistance_zones": [zf(z) for z in res[:3]],
            # Long TF (90d, order=5): ALL major pivots — no cap
            "swing_highs": [{"date": str(d.date()), "price": round(p, 2)}
                            for t, d, p in pivots_long  if t == "high"],
            "swing_lows":  [{"date": str(d.date()), "price": round(p, 2)}
                            for t, d, p in pivots_long  if t == "low"],
            # Short TF (30d, order=2): granular recent pivots
            "swing_highs_30d": [{"date": str(d.date()), "price": round(p, 2)}
                                for t, d, p in pivots_short if t == "high"],
            "swing_lows_30d":  [{"date": str(d.date()), "price": round(p, 2)}
                                for t, d, p in pivots_short if t == "low"],
            "round_levels":    self._round_levels(cur),
            "volume_analysis": volume_analysis,
            "levels":          self._level_breaks(ind_df, cur),
            "recent_candles":  recent,
        }

    def _detect_pivots(self, series: pd.Series, order: int = 2,
                       pivot_type: str = "both") -> list:
        """Detect pivot points in a price series.

        pivot_type: "high" returns only local maxima (use with the high series),
                    "low" returns only local minima (use with the low series),
                    "both" returns both types (default, for series that mix them).

        Passing pivot_type="high" with a high-column series and pivot_type="low"
        with a low-column series avoids mislabelling: a local minimum of the high
        column would have a high-column price but be tagged as "low", corrupting
        support/resistance clustering.
        """
        pivots: list = []
        seen: set = set()   # deduplicate flat tops / double bottoms at same price
        for i in range(order, len(series) - order):
            win = series.iloc[i - order: i + order + 1]
            v   = series.iloc[i]
            idx = series.index[i]
            if pivot_type in ("high", "both") and v == win.max():
                key = ("high", round(float(v), 4))
                if key not in seen:
                    seen.add(key)
                    pivots.append(("high", idx, float(v)))
            elif pivot_type in ("low", "both") and v == win.min():
                key = ("low", round(float(v), 4))
                if key not in seen:
                    seen.add(key)
                    pivots.append(("low", idx, float(v)))
        return pivots

    def _round_levels(self, price: float, window_pct: float = 0.08) -> list[dict]:
        """Return psychologically significant round-number price levels within ±window_pct.

        Adaptive step size by price magnitude. Each entry:
            level (float), type ("major"|"minor"), dist_pct (float), side ("support"|"resistance")
        Sorted by abs(dist_pct), nearest first.
        """
        if price <= 0:
            return []
        step = (5   if price <   50 else
                25  if price <  200 else
                50  if price <  500 else
                100 if price < 2000 else 250)
        # A "major" level is a multiple of 4× step (e.g. step=25 → every 100 is major)
        major_step = step * 4
        lo = price * (1 - window_pct)
        hi = price * (1 + window_pct)
        first_mult = int(lo / step)
        if first_mult * step < lo:
            first_mult += 1
        levels = []
        mult = first_mult
        while mult * step <= hi:
            lvl = mult * step
            if abs(lvl - price) / price > 0.001:   # skip trivially close levels
                dist_pct = round((lvl - price) / price * 100, 2)
                levels.append({
                    "level":    float(lvl),
                    "type":     "major" if lvl % major_step == 0 else "minor",
                    "dist_pct": dist_pct,
                    "side":     "support" if lvl < price else "resistance",
                })
            mult += 1
        return sorted(levels, key=lambda x: abs(x["dist_pct"]))

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

    def _load_vix(self) -> None:
        """Load VIX CSV into cache (idempotent, safe to call from the main thread before forking)."""
        if self._vix_loaded:
            return
        p = Path(self.vix_csv)
        if p.exists():
            try:
                self._vix_cache = self._load_csv(p)
            except Exception:
                self._vix_cache = None
        # Mark loaded only after the attempt completes — a thread that reads the flag
        # before this line will retry (benign) rather than see a half-loaded cache.
        self._vix_loaded = True

    def vix_context(self, as_of: str) -> dict:
        """VIX regime as of `as_of` (no look-ahead). Returns {} if no VIX file."""
        if not self._vix_loaded:
            self._load_vix()
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
        # Trading-intuition regime (user thresholds: <12 consolidation, 12-15 sideways, ≥15 volatile)
        regime_trading = ("consolidation" if level < 12  else
                          "volatile"      if level >= 15 else "sideways")
        chg10 = level - float(s.iloc[-11]) if len(s) >= 11 else 0.0
        trend_10d = "rising" if chg10 > 0.5 else "falling" if chg10 < -0.5 else "flat"
        quantity_bias = ("fewer shares, bigger point targets" if level >= 15 else
                         "more shares, tighter targets"       if level <  12 else
                         "standard sizing")
        return {"vix": round(level, 2), "pctile_1y": pctile, "regime": regime,
                "dir_5d": direction, "pos_20d": pos20,
                "regime_trading": regime_trading,
                "trend_10d":      trend_10d,
                "quantity_bias":  quantity_bias}

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
