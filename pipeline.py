"""
pipeline.py
───────────
Stateless trading data generation pipeline.

For each trading date between START_DATE and END_DATE:
  1. Skip if (symbol, date) already in DB
  2. Load 30 days of candles
  3. Compute technicals + VIX + zones
  4. Build prompt → call LLM (or rule-based fallback)
  5. Simulate outcome against next 5 forward candles
  6. Store to SQLite

Run:
    python pipeline.py
    DATA_SOURCE=csv python pipeline.py
    SYMBOL=RELIANCE START_DATE=2023-01-01 END_DATE=2023-12-31 python pipeline.py
"""
from __future__ import annotations

import json
import os
import random
import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env", override=False)
except ImportError:
    pass

from technicals import TechnicalsAnalyzer
from prompts import SYSTEM_PROMPT as _SYSTEM, build_prompt

# ── Config ───────────────────────────────────────────────────────────
SYMBOL        = os.environ.get("SYMBOL",        "CANBK")
# Multiple symbols: SYMBOLS=CANBK,HDFC,INFY overrides SYMBOL
_SYMBOLS_ENV  = os.environ.get("SYMBOLS",       "")
SYMBOLS       = [s.strip() for s in _SYMBOLS_ENV.split(",") if s.strip()] or [SYMBOL]
START_DATE    = os.environ.get("START_DATE",    "2024-01-10")
END_DATE      = os.environ.get("END_DATE",      "2024-12-31")
LOOKBACK_DAYS = int(os.environ.get("LOOKBACK_DAYS", 90))
MAX_HOLD_DAYS = int(os.environ.get("MAX_HOLD_DAYS",  5))
COUNT         = int(os.environ.get("COUNT",          10))   # random dates to generate
WORKERS       = int(os.environ.get("WORKERS",         1))    # parallel workers
POST_REASONING    = os.environ.get("POST_REASONING",    "0").strip().lower() in ("1","true","yes","on")
BACKFILL_REASONING = os.environ.get("BACKFILL_REASONING","0").strip().lower() in ("1","true","yes","on")
REQUIRE_LLM       = os.environ.get("REQUIRE_LLM",       "0").strip().lower() in ("1","true","yes","on")
OLLAMA_URL    = os.environ.get("OLLAMA_URL",    "http://localhost:11434/api/chat")
OLLAMA_MODEL  = os.environ.get("OLLAMA_MODEL",  "nemotron-bolt-e4b-16k")
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", 300))  # seconds; Nemotron needs ~170s on CPU
# HuggingFace fine-tuned model (overrides Ollama when set)
# e.g. HF_MODEL=output/nemotron-lora/fused_model
#      HF_MODEL=nvidia/Nemotron-Mini-4B-Instruct HF_ADAPTER=output/nemotron-lora
HF_MODEL      = os.environ.get("HF_MODEL",      "")
HF_ADAPTER    = os.environ.get("HF_ADAPTER",   "")
DB_PATH       = os.environ.get("DB_PATH",       "trading_agent.db")
RISK_PER_TRADE = float(os.environ.get("RISK_PER_TRADE", 0.02))
CAPITAL_FOR_QTY = float(os.environ.get("CAPITAL_FOR_QTY", 10_000))

# Indian equity transaction costs
INCLUDE_COSTS  = os.environ.get("INCLUDE_COSTS", "1").strip().lower() in ("1","true","yes","on")
BROKERAGE_PCT  = float(os.environ.get("BROKERAGE_PCT", 0.0003))
BROKERAGE_MAX  = float(os.environ.get("BROKERAGE_MAX", 20.0))
STT_PCT        = float(os.environ.get("STT_PCT",        0.001))
EXCH_TXN_PCT   = float(os.environ.get("EXCH_TXN_PCT",   0.0000297))
SEBI_PCT       = float(os.environ.get("SEBI_PCT",       0.000001))
GST_PCT        = float(os.environ.get("GST_PCT",        0.18))
STAMP_PCT      = float(os.environ.get("STAMP_PCT",      0.00015))


# ── NSE market holidays (weekday dates with no trading) ──────────────
NSE_HOLIDAYS: set[str] = {
    # 2022
    "2022-01-26", "2022-03-18", "2022-04-14", "2022-04-15", "2022-04-18",
    "2022-05-03", "2022-08-09", "2022-08-15", "2022-10-02", "2022-10-05",
    "2022-10-24", "2022-10-26", "2022-11-08",
    # 2023
    "2023-01-26", "2023-03-07", "2023-03-30", "2023-04-04", "2023-04-07",
    "2023-04-14", "2023-05-01", "2023-06-28", "2023-08-15", "2023-09-19",
    "2023-10-02", "2023-10-24", "2023-11-14", "2023-11-27", "2023-12-25",
    # 2024
    "2024-01-22", "2024-01-26", "2024-03-25", "2024-03-29", "2024-04-11",
    "2024-04-14", "2024-04-17", "2024-04-21", "2024-05-23", "2024-06-17",
    "2024-07-17", "2024-08-15", "2024-10-02", "2024-10-12", "2024-10-14",
    "2024-11-01", "2024-11-15", "2024-12-25",
    # 2025
    "2025-02-26", "2025-03-14", "2025-03-31", "2025-04-10", "2025-04-14",
    "2025-04-18", "2025-05-01", "2025-08-15", "2025-08-27", "2025-10-02",
    "2025-10-21", "2025-10-22", "2025-11-05", "2025-12-25",
    # 2026 (official NSE schedule — verify at nseindia.com before each year)
    "2026-01-26", "2026-02-18", "2026-03-20", "2026-04-02", "2026-04-03",
    "2026-04-14", "2026-05-01", "2026-07-20", "2026-08-15", "2026-08-27",
    "2026-10-01", "2026-10-02", "2026-11-13", "2026-12-25",
    # 2027 (preliminary — NSE publishes final list ~Nov of prior year)
    "2027-01-26", "2027-03-11", "2027-03-26", "2027-03-29", "2027-04-14",
    "2027-08-15", "2027-10-21", "2027-10-22", "2027-11-05", "2027-12-25",
}

# ── Database ──────────────────────────────────────────────────────────

_REQUIRED_COLS = {"symbol", "decision_date", "reasoning", "confidence", "features_used"}


def _db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, timeout=30)
    con.execute("PRAGMA busy_timeout=30000")
    con.execute("PRAGMA journal_mode=WAL")
    return con


def init_db() -> None:
    with _db() as con:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA busy_timeout=30000")
        cols = {r[1] for r in con.execute("PRAGMA table_info(trade_log)")}

        if cols and not _REQUIRED_COLS.issubset(cols):
            con.execute("ALTER TABLE trade_log RENAME TO trade_log_old")
            print("  Migrated old trade_log → trade_log_old")
            cols = set()

        if not cols:
            # Fresh table with full schema
            con.executescript("""
                CREATE TABLE IF NOT EXISTS trade_log (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol          TEXT,
                    decision_date   TEXT,
                    action          TEXT,
                    entry_price     REAL,
                    stop_loss       REAL,
                    take_profit     REAL,
                    exit_price      REAL,
                    exit_date       TEXT,
                    exit_reason     TEXT,
                    pnl             REAL,
                    reasoning       TEXT,
                    confidence      REAL,
                    features_used   TEXT,
                    technicals_json TEXT,
                    prompt_text     TEXT,
                    decision_slot   TEXT DEFAULT 'close',
                    planned_entry   REAL,
                    post_reasoning  TEXT,
                    is_mock         INTEGER DEFAULT 0,
                    key_risk        TEXT,
                    created_at      TEXT,
                    UNIQUE(symbol, decision_date, decision_slot)
                );
            """)
        elif "decision_slot" not in cols:
            # Existing table — migrate to new schema with 3-column UNIQUE key
            print("  Migrating trade_log schema (adding decision_slot, planned_entry, post_reasoning)…")
            con.executescript("""
                CREATE TABLE trade_log_new (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol          TEXT,
                    decision_date   TEXT,
                    action          TEXT,
                    entry_price     REAL,
                    stop_loss       REAL,
                    take_profit     REAL,
                    exit_price      REAL,
                    exit_date       TEXT,
                    exit_reason     TEXT,
                    pnl             REAL,
                    reasoning       TEXT,
                    confidence      REAL,
                    features_used   TEXT,
                    technicals_json TEXT,
                    prompt_text     TEXT,
                    decision_slot   TEXT DEFAULT 'close',
                    planned_entry   REAL,
                    post_reasoning  TEXT,
                    created_at      TEXT,
                    UNIQUE(symbol, decision_date, decision_slot)
                );
                INSERT INTO trade_log_new
                    (id, symbol, decision_date, action, entry_price, stop_loss, take_profit,
                     exit_price, exit_date, exit_reason, pnl, reasoning, confidence,
                     features_used, technicals_json, prompt_text, decision_slot,
                     planned_entry, post_reasoning, created_at)
                SELECT id, symbol, decision_date, action, entry_price, stop_loss, take_profit,
                       exit_price, exit_date, exit_reason, pnl, reasoning, confidence,
                       features_used, technicals_json,
                       COALESCE(prompt_text, NULL),
                       'close', NULL, NULL, created_at
                FROM trade_log;
                DROP TABLE trade_log;
                ALTER TABLE trade_log_new RENAME TO trade_log;
            """)
            print("  Migration complete — all existing rows set to decision_slot='close'")
        else:
            # Table exists with new schema; add any still-missing columns
            for col, defn in [("planned_entry", "REAL"), ("post_reasoning", "TEXT"),
                               ("is_mock", "INTEGER DEFAULT 0"), ("key_risk", "TEXT")]:
                if col not in cols:
                    con.execute(f"ALTER TABLE trade_log ADD COLUMN {col} {defn}")
            # Idempotent: tag rows written by mock_data.py that pre-date the is_mock column
            con.execute("""UPDATE trade_log SET is_mock=1
                           WHERE is_mock=0
                             AND (reasoning LIKE 'Bullish EMA stack%'
                               OR reasoning LIKE 'Bearish EMA stack%'
                               OR reasoning LIKE 'Conflicting signals%')""")

    print(f"DB ready: {DB_PATH}")


def _already_exists(symbol: str, date: str, slot: str = "close") -> bool:
    with _db() as con:
        row = con.execute(
            "SELECT 1 FROM trade_log WHERE symbol=? AND decision_date=? AND decision_slot=? LIMIT 1",
            (symbol, date, slot)
        ).fetchone()
    return row is not None


def _store(row: dict) -> None:
    with _db() as con:
        con.execute("""
            INSERT OR IGNORE INTO trade_log
              (symbol, decision_date, action, entry_price, stop_loss, take_profit,
               exit_price, exit_date, exit_reason, pnl, reasoning, confidence,
               features_used, technicals_json, prompt_text,
               decision_slot, planned_entry, post_reasoning,
               is_mock, key_risk, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            row["symbol"], row["decision_date"], row["action"],
            row["entry_price"], row["stop_loss"], row["take_profit"],
            row["exit_price"], row["exit_date"], row["exit_reason"], row["pnl"],
            row["reasoning"], row["confidence"],
            json.dumps(row.get("features_used", [])),
            json.dumps(row.get("technicals_json", {})),
            row.get("prompt_text"),
            row.get("decision_slot", "close"),
            row.get("planned_entry"),
            row.get("post_reasoning"),
            0,  # is_mock=0 for all real pipeline rows
            row.get("key_risk") or "",
            datetime.now().isoformat(),
        ))


# ── LLM ──────────────────────────────────────────────────────────────

# _SYSTEM is imported from prompts.py as _SYSTEM


def _llm_generate(user: str) -> str | None:
    # HuggingFace fine-tuned model (primary when configured)
    if HF_MODEL:
        try:
            from inference import generate_hf
            return generate_hf(HF_MODEL, _SYSTEM, user,
                               adapter_path=HF_ADAPTER or None)
        except Exception as e:
            print(f"  [HF error: {e}] → Ollama fallback")

    try:
        import requests
        r = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "messages": [{"role": "system", "content": _SYSTEM},
                         {"role": "user",   "content": user}],
            "stream": False,
            "options": {"temperature": 0.2, "num_ctx": 4096},
        }, timeout=OLLAMA_TIMEOUT)
        r.raise_for_status()
        return r.json()["message"]["content"]
    except Exception as e:
        print(f"  [LLM error: {e}] → fallback")
        return None


def _llm_post_reasoning(symbol: str, date: str, action: str,
                        planned_entry: float, sl: float, tp: float,
                        exit_reason: str, exit_p: float, pnl: float) -> str | None:
    """Second LLM call: explain why the trade succeeded or failed."""
    pnl_str = f"{pnl:+.0f}"
    post_prompt = (
        f"Trade outcome for {symbol} {date}:\n"
        f"Plan: {action} limit entry @ {planned_entry:.2f}, SL={sl:.2f}, TP={tp:.2f}\n"
        f"Outcome: {exit_reason} @ {exit_p:.2f} (P&L: {pnl_str})\n\n"
        "In 2-3 sentences: what price action caused this outcome? "
        "What did the setup get right or wrong? Be specific about levels and structure."
    )
    system = "You are a trading analyst. Explain what happened in a trade briefly and precisely."
    try:
        if HF_MODEL:
            from inference import generate_hf
            return generate_hf(HF_MODEL, system, post_prompt, adapter_path=HF_ADAPTER or None)
        import requests
        r = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "messages": [{"role": "system", "content": system},
                         {"role": "user",   "content": post_prompt}],
            "stream": False,
            "options": {"temperature": 0.3, "num_ctx": 2048},
        }, timeout=OLLAMA_TIMEOUT)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()
    except Exception as e:
        print(f"  [post_reasoning error: {e}]")
        return None


def _parse_json(text: str | None) -> dict:
    if not text:
        return {}
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1)
    try:
        return json.loads(text.strip())
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return {}


# ── Decision (LLM or rule-based fallback) ────────────────────────────

def _safe_num(x, default=0.0):
    try:
        v = float(x)
        return v if v == v else default
    except (TypeError, ValueError):
        return default


def _validate_trade(action: str, entry: float, sl: float, tp: float) -> bool:
    """Returns True only when SL/TP are on the correct side of entry."""
    if action == "BUY":  return sl < entry < tp
    if action == "SELL": return tp < entry < sl
    return True  # HOLD always valid


def decide(symbol: str, date: str, tech: dict, vc: dict, zc: dict,
           decision_slot: str = "close") -> tuple[dict, str]:
    """Returns (decision_dict, prompt_text) so the caller can store the exact prompt."""
    user = build_prompt(symbol, date, tech, vc, zc, decision_slot)
    dec  = _parse_json(_llm_generate(user))
    if dec.get("action") in ("BUY", "SELL", "HOLD"):
        return dec, user

    # Rule-based fallback — skipped when REQUIRE_LLM=1
    if REQUIRE_LLM:
        return {"action": "HOLD", "stop_loss": 0, "take_profit": 0, "risk_reward": 0,
                "confidence": 0.0, "reasoning": "__llm_failed__",
                "features_used": []}, user
    ns, nr = tech.get("nearest_support"), tech.get("nearest_resistance")
    cur = tech["close"]
    max_risk = CAPITAL_FOR_QTY * RISK_PER_TRADE
    if tech["ema_stack"].startswith("bullish") and tech["above_vwap"] and ns:
        sl = round(ns["low"] * 0.998, 2)
        tp = round(nr["mid"] if nr else cur * 1.02, 2)
        rp = max(cur - sl, 0.01)
        qty = min(max(1, int(max_risk / rp)), max(1, int(CAPITAL_FOR_QTY // cur)))
        rr = round((tp - cur) / rp, 2)
        if rr >= 1.5:
            return {"action": "BUY", "quantity": qty, "stop_loss": sl, "take_profit": tp,
                    "risk_reward": rr, "confidence": 0.55,
                    "reasoning": f"Rule: bullish EMA stack, above VWAP, support {ns['mid']:.1f}",
                    "features_used": ["ema_stack", "above_vwap", "nearest_support"]}, user
    if tech["ema_stack"].startswith("bearish") and not tech["above_vwap"] and nr:
        sl = round(nr["high"] * 1.002, 2)
        tp = round(ns["mid"] if ns else cur * 0.98, 2)
        rp = max(sl - cur, 0.01)
        qty = min(max(1, int(max_risk / rp)), max(1, int(CAPITAL_FOR_QTY // cur)))
        rr = round((cur - tp) / rp, 2)
        if rr >= 1.5:
            return {"action": "SELL", "quantity": qty, "stop_loss": sl, "take_profit": tp,
                    "risk_reward": rr, "confidence": 0.55,
                    "reasoning": f"Rule: bearish EMA stack, below VWAP, resistance {nr['mid']:.1f}",
                    "features_used": ["ema_stack", "above_vwap", "nearest_resistance"]}, user
    return {"action": "HOLD", "stop_loss": 0, "take_profit": 0, "risk_reward": 0,
            "confidence": 0.4, "reasoning": "Rule: no clear setup.",
            "features_used": ["ema_stack", "above_vwap"]}, user


# ── Outcome simulation ────────────────────────────────────────────────

def simulate_outcome(action: str, entry: float, sl: float, tp: float,
                     fwd_df, max_hold: int = MAX_HOLD_DAYS,
                     planned_entry: float | None = None):
    """Simulate trade outcome against forward price bars.

    Accepts both minute-bar and daily-bar DataFrames. `max_hold` is treated as
    the number of *trading days*, not bars — so this works correctly after
    `load_forward()` switched to returning minute data.

    Returns: (exit_reason, exit_price, exit_date, days_held)
    exit_reason: 'TP_HIT' | 'SL_HIT' | 'TIME_EXIT' | 'ENTRY_NOT_FILLED' | 'HOLD'
    """
    if action == "HOLD" or not sl or not tp or fwd_df.empty:
        return "HOLD", entry, None, 0

    # Build a N-trading-day window — works for both minute and daily bars
    trading_days = sorted(fwd_df.index.normalize().unique())
    n_days = min(max_hold, len(trading_days))
    if not trading_days:
        return "TIME_EXIT", round(float(fwd_df.iloc[-1]["close"]), 2), \
               str(fwd_df.index[-1].date()), 0
    cutoff = trading_days[n_days - 1]
    window = fwd_df[fwd_df.index.normalize() <= cutoff]

    bars = list(window.iterrows())
    fill_price = entry   # default: immediate market fill

    # Phase 1 — wait for limit fill (only when planned_entry differs from current price)
    if planned_entry and abs(planned_entry - entry) > 0.01:
        filled = False
        for i, (idx, row) in enumerate(bars):
            lo, hi = float(row["low"]), float(row["high"])
            if action == "BUY"  and lo <= planned_entry:
                fill_price = planned_entry; bars = bars[i:]; filled = True; break
            if action == "SELL" and hi >= planned_entry:
                fill_price = planned_entry; bars = bars[i:]; filled = True; break
        if not filled:
            last = window.iloc[-1]
            return "ENTRY_NOT_FILLED", round(float(last["close"]), 2), \
                   str(window.index[-1].date()), n_days

    # Phase 2 — track SL/TP from fill bar onward
    for i, (idx, row) in enumerate(bars):
        lo, hi = float(row["low"]), float(row["high"])
        ds = str(idx.date())
        if action == "BUY":
            if lo <= sl: return "SL_HIT", round(sl, 2), ds, i + 1
            if hi >= tp: return "TP_HIT", round(tp, 2), ds, i + 1
        elif action == "SELL":
            if hi >= sl: return "SL_HIT", round(sl, 2), ds, i + 1
            if lo <= tp: return "TP_HIT", round(tp, 2), ds, i + 1

    last = window.iloc[-1]
    return "TIME_EXIT", round(float(last["close"]), 2), str(window.index[-1].date()), n_days


def compute_pnl(action: str, entry: float, exit_p: float, qty: int) -> float:
    if action == "BUY":  return round((exit_p - entry) * qty, 2)
    if action == "SELL": return round((entry - exit_p) * qty, 2)
    return 0.0


def trade_costs(entry: float, exit_p: float, qty: int, action: str) -> float:
    if not INCLUDE_COSTS or action not in ("BUY", "SELL") or qty <= 0:
        return 0.0
    buy_val = entry * qty; sell_val = exit_p * qty; turnover = buy_val + sell_val
    brokerage = (min(BROKERAGE_MAX, buy_val * BROKERAGE_PCT)
                 + min(BROKERAGE_MAX, sell_val * BROKERAGE_PCT))
    stt   = STT_PCT * turnover
    exch  = EXCH_TXN_PCT * turnover
    sebi  = SEBI_PCT * turnover
    gst   = GST_PCT * (brokerage + exch + sebi)
    stamp = STAMP_PCT * buy_val
    return round(brokerage + stt + exch + sebi + gst + stamp, 2)


# ── Core: one sample per date ─────────────────────────────────────────

def generate_sample(ta: TechnicalsAnalyzer, symbol: str, date: str,
                    slot: str | None = None) -> dict | None:
    # Determine slot (random if not specified)
    if slot is None:
        slot = random.choice(["open", "mid", "close"])

    # 1. Dedup check
    if _already_exists(symbol, date, slot):
        print(f"  [skip] {symbol} {date} @{slot} already in DB")
        return None

    # 2. Load candles
    try:
        df = ta.load_candles(symbol, date, LOOKBACK_DAYS)
    except Exception as e:
        print(f"  [skip] {symbol} {date} — candles: {e}")
        return None

    # Sanity: last candle should be on or very close to the decision date.
    # A gap > 10 calendar days usually means the CSV ends before this date.
    import pandas as _pd
    last_candle = _pd.Timestamp(df.index[-1]).normalize()
    gap_days = (_pd.Timestamp(date) - last_candle).days
    if gap_days > 10:
        print(f"  [warn] {symbol} {date} — last candle is {last_candle.date()} "
              f"({gap_days}d gap); CSV may be stale")

    # 3. Compute technicals for the chosen slot, VIX, zones
    try:
        tech = ta.compute(df, decision_slot=slot)
    except Exception as e:
        print(f"  [skip] {symbol} {date} — technicals: {e}")
        return None
    vc = ta.vix_context(date)
    zc = ta.compute_zones(symbol, date)
    if vc:
        tech["vix_ctx"] = vc
    if zc:
        tech["zones"] = zc

    # 4. LLM decision (returns exact prompt stored in DB for training alignment)
    dec, prompt_text = decide(symbol, date, tech, vc, zc, decision_slot=slot)
    action  = dec.get("action", "HOLD")
    if action not in ("BUY", "SELL", "HOLD"):
        action = "HOLD"
    rsn = dec.get("reasoning", "—")
    if rsn == "__llm_failed__":
        print(f"  [skip] {symbol} {date} @{slot}: LLM failed, REQUIRE_LLM=1")
        return None
    sl      = _safe_num(dec.get("stop_loss",  0))
    tp      = _safe_num(dec.get("take_profit", 0))
    conf    = _safe_num(dec.get("confidence", 0.5), 0.5)
    qty     = max(1, int(_safe_num(dec.get("quantity", 1), 1)))
    key_risk = dec.get("key_risk", "") or ""

    # planned_entry: LLM's limit price, falls back to current close (market order)
    pe_raw = _safe_num(dec.get("planned_entry", 0))
    planned_entry = pe_raw if pe_raw > 0 else tech["close"]

    # Validate SL/TP are on the correct side of entry
    if action in ("BUY", "SELL") and not _validate_trade(action, planned_entry, sl, tp):
        print(f"  [skip] {symbol} {date} @{slot}: invalid SL/TP sides "
              f"({action} entry={planned_entry:.2f} SL={sl:.2f} TP={tp:.2f})")
        return None

    # 5. Simulate outcome (two-phase if planned_entry differs from current close)
    entry_p = tech["close"]
    try:
        fwd = ta.load_forward(symbol, date, MAX_HOLD_DAYS)
    except Exception:
        fwd = __import__("pandas").DataFrame()
    exit_reason, exit_p, exit_date, _ = simulate_outcome(
        action, entry_p, sl, tp, fwd, planned_entry=planned_entry
    )
    if exit_reason == "ENTRY_NOT_FILLED":
        gross = 0.0
        pnl   = 0.0
    else:
        gross = compute_pnl(action, planned_entry, exit_p, qty)
        pnl   = round(gross - trade_costs(planned_entry, exit_p, qty, action), 2)

    mark = "PROFIT" if pnl > 0 else "LOSS" if pnl < 0 else "FLAT"
    print(f"  {symbol} {date} @{slot} | {action} entry={planned_entry:.2f} SL={sl:.2f} TP={tp:.2f} "
          f"→ {exit_reason} {exit_p:.2f} [{mark} {pnl:+.0f}]")

    # 6. Optional post-trade reasoning (second LLM call)
    post_rsn = None
    if POST_REASONING and action != "HOLD" and exit_reason != "ENTRY_NOT_FILLED":
        post_rsn = _llm_post_reasoning(
            symbol, date, action, planned_entry, sl, tp, exit_reason, exit_p, pnl
        )

    return {
        "symbol": symbol, "decision_date": date,
        "action": action,
        "entry_price": round(planned_entry, 2),
        "stop_loss": sl, "take_profit": tp,
        "exit_price": exit_p, "exit_date": exit_date,
        "exit_reason": exit_reason, "pnl": pnl,
        "reasoning": rsn, "confidence": conf,
        "features_used": dec.get("features_used", []),
        "technicals_json": {k: v for k, v in tech.items() if k != "_df"},
        "prompt_text": prompt_text,
        "decision_slot": slot,
        "planned_entry": round(planned_entry, 2),
        "post_reasoning": post_rsn,
        "key_risk": key_risk,
    }


# ── Date utilities ────────────────────────────────────────────────────

def trading_days(start: str, end: str) -> list[str]:
    """All weekday (Mon-Fri) dates between start and end inclusive, excluding NSE holidays."""
    days, dt = [], datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    while dt <= end_dt:
        d = dt.strftime("%Y-%m-%d")
        if dt.weekday() < 5 and d not in NSE_HOLIDAYS:
            days.append(d)
        dt += timedelta(days=1)
    return days


def available_slots(start: str, end: str, count: int, symbol: str) -> list[tuple[str, str]]:
    """
    Return up to `count` random (date, slot) pairs from [start, end] that are
    NOT already in the DB for this symbol.  Each date has 3 slots: open/mid/close.
    """
    all_days = trading_days(start, end)

    with _db() as con:
        done = {(r[0], r[1]) for r in con.execute(
            "SELECT decision_date, decision_slot FROM trade_log WHERE symbol=?", (symbol,)
        )}

    all_pairs: list[tuple[str, str]] = [
        (d, s) for d in all_days for s in ("open", "mid", "close")
    ]
    available = [(d, s) for d, s in all_pairs if (d, s) not in done]
    random.shuffle(available)

    total_days = len(all_days)
    skipped    = len(all_pairs) - len(available)
    print(f"Slot pool: {total_days} weekdays × 3 slots = {len(all_pairs)} | "
          f"{skipped} already in DB | "
          f"{len(available)} available | picking {min(count, len(available))}")

    return available[:count]


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    init_db()

    # Build full work list: (symbol, date, slot) triples across all symbols
    work: list[tuple[str, str, str]] = []
    for sym in SYMBOLS:
        slots = available_slots(START_DATE, END_DATE, COUNT, sym)
        if not slots:
            print(f"All slots already generated for {sym}.")
        else:
            print(f"Queuing {len(slots)} (date, slot) pairs for {sym}")
            work.extend((sym, d, s) for d, s in slots)

    if not work:
        print("Nothing to do. Extend START_DATE/END_DATE or add more symbols.")
    elif WORKERS <= 1:
        ta = TechnicalsAnalyzer()
        generated = 0
        for sym, date, slot in work:
            row = generate_sample(ta, sym, date, slot)
            if row:
                _store(row)
                generated += 1
        print(f"\nDone. {generated} new samples stored in {DB_PATH}")
    else:
        # NOTE: With a single Ollama instance, LLM calls are still sequential
        # (Ollama queues them). Speedup comes from parallel candle loading,
        # technicals computation, and outcome simulation.
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if HF_MODEL:
            print(f"\nParallel generation: {WORKERS} workers | HF model — true parallel inference")
        else:
            print(f"\nParallel generation: {WORKERS} workers | "
                  f"note: single Ollama instance queues LLM calls; "
                  f"speedup from parallel data loading + multi-symbol interleaving")

        # Preload all symbol caches in the main thread before spawning workers.
        # CSV minute-bar load takes ~36s per symbol; paying it once prevents every
        # thread re-parsing the file and avoids concurrent dict-write races.
        ta_shared = TechnicalsAnalyzer()
        # VIX must be preloaded in the main thread: _vix_loaded flag is set only after
        # the file is read, so a concurrent thread seeing the flag unset, racing into
        # _load_vix, and losing the dict write would return {} silently.
        try:
            ta_shared._load_vix()
        except Exception:
            pass
        for sym in {s for s, _, _ in work}:
            try:
                ta_shared._load_raw(sym)
                ta_shared._load_minute(sym)
            except Exception:
                pass

        def _job(item: tuple[str, str, str]) -> dict | None:
            sym, date, slot = item
            # ta_shared caches are fully populated — workers only read (CPython GIL safe)
            row = generate_sample(ta_shared, sym, date, slot)
            if row:
                _store(row)
            return row

        generated = 0
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = {pool.submit(_job, item): item for item in work}
            for f in as_completed(futures):
                try:
                    if f.result():
                        generated += 1
                except Exception as e:
                    sym, date, slot = futures[f]
                    print(f"  [error] {sym} {date} @{slot}: {e}")

        print(f"\nDone. {generated} new samples stored in {DB_PATH}")
