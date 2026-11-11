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


# ── Database ──────────────────────────────────────────────────────────

_REQUIRED_COLS = {"symbol", "decision_date", "reasoning", "confidence", "features_used"}

def init_db() -> None:
    with sqlite3.connect(DB_PATH) as con:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA busy_timeout=30000")
        # If existing table is missing required columns, rename and start fresh
        cols = {r[1] for r in con.execute("PRAGMA table_info(trade_log)")}
        if cols and not _REQUIRED_COLS.issubset(cols):
            con.execute("ALTER TABLE trade_log RENAME TO trade_log_old")
            print("  Migrated old trade_log → trade_log_old")
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
                created_at      TEXT,
                UNIQUE(symbol, decision_date)
            );
        """)
        # Add prompt_text to existing DBs that predate this column
        cols = {r[1] for r in con.execute("PRAGMA table_info(trade_log)")}
        if "prompt_text" not in cols:
            con.execute("ALTER TABLE trade_log ADD COLUMN prompt_text TEXT")
    print(f"DB ready: {DB_PATH}")


def _already_exists(symbol: str, date: str) -> bool:
    with sqlite3.connect(DB_PATH) as con:
        row = con.execute(
            "SELECT 1 FROM trade_log WHERE symbol=? AND decision_date=? LIMIT 1",
            (symbol, date)
        ).fetchone()
    return row is not None


def _store(row: dict) -> None:
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
            INSERT OR IGNORE INTO trade_log
              (symbol, decision_date, action, entry_price, stop_loss, take_profit,
               exit_price, exit_date, exit_reason, pnl, reasoning, confidence,
               features_used, technicals_json, prompt_text, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            row["symbol"], row["decision_date"], row["action"],
            row["entry_price"], row["stop_loss"], row["take_profit"],
            row["exit_price"], row["exit_date"], row["exit_reason"], row["pnl"],
            row["reasoning"], row["confidence"],
            json.dumps(row.get("features_used", [])),
            json.dumps(row.get("technicals_json", {})),
            row.get("prompt_text"),
            datetime.now().isoformat(),
        ))


# ── LLM ──────────────────────────────────────────────────────────────

_SYSTEM = (
    "You are a disciplined NSE equity trader. Protect capital first. "
    "R:R must be >= 1.5. HOLD if setup unclear. "
    "Ground every decision in the features given — never invent data. "
    "Output JSON with keys: action, stop_loss, take_profit, risk_reward, "
    "reasoning, confidence, features_used. "
    "action must be BUY, SELL, or HOLD."
)


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


# ── Prompt builder ────────────────────────────────────────────────────

def _pct(a, b):
    if not b or b == 0: return None
    return round((a - b) / b * 100, 2)


def build_prompt(symbol: str, date: str, tech: dict, vc: dict, zc: dict) -> str:
    close = tech.get("close", 0)
    rc    = tech.get("recent_candles", [])
    prev_close = rc[-2]["close"] if len(rc) >= 2 else close
    day_chg = _pct(close, prev_close)

    d5_high = max(c["high"] for c in rc) if rc else close
    d5_low  = min(c["low"]  for c in rc) if rc else close
    rng     = d5_high - d5_low
    pct_pos = round((close - d5_low) / rng * 100) if rng > 0 else 50

    avg_vol  = sum(c.get("volume", 0) for c in rc) / len(rc) if rc else 1
    last_vol = rc[-1].get("volume", 0) if rc else 0
    vol_ratio = round(last_vol / avg_vol, 2) if avg_vol > 0 else 1

    daily_ranges = [c["high"] - c["low"] for c in rc] if rc else [0]
    atr = round(sum(daily_ranges) / len(daily_ranges), 2)

    d3_chg = _pct(close, rc[-4]["close"]) if len(rc) >= 4 else None

    e50, e100, e200 = tech.get("ema_50"), tech.get("ema_100"), tech.get("ema_200")
    vwap  = tech.get("vwap")
    stack = tech.get("ema_stack", "—")
    c_vs_200  = _pct(close, e200) if e200 else None
    c_vs_vwap = _pct(close, vwap) if vwap else None

    def fmt_zones(zones, n=2):
        parts = []
        for z in (zones[:n] if zones else []):
            if z and z.get("low"):
                d = _pct(z["mid"], close) if close else 0
                parts.append(f"{z['low']:.1f}-{z['high']:.1f}({d:+.1f}%)")
        return "  ".join(parts) or "—"

    sup_str = fmt_zones(tech.get("support_zones"))
    res_str = fmt_zones(tech.get("resistance_zones"))
    ns = tech.get("nearest_support");  dist_s = _pct(ns["mid"], close) if ns and close else None
    nr = tech.get("nearest_resistance"); dist_r = _pct(nr["mid"], close) if nr and close else None

    sh = tech.get("swing_highs", [])[-2:]
    sl = tech.get("swing_lows",  [])[-2:]
    sh_str = "  ".join(f"{s['price']:.1f}({s['date'][5:]})" for s in sh) or "—"
    sl_str = "  ".join(f"{s['price']:.1f}({s['date'][5:]})" for s in sl) or "—"

    bias = ("bullish" if (stack or "").startswith("bullish") and (c_vs_vwap or 0) > 0 else
            "bearish" if (stack or "").startswith("bearish") and (c_vs_vwap or 0) < 0 else "neutral")

    lv   = tech.get("levels", {}) or {}
    headline_bits = [f"bias:{bias}", f"trend:{tech.get('trend','-')}"]
    if lv.get("pdh_state") in ("broke", "rejected"): headline_bits.append(f"PDH:{lv['pdh_state']}")
    if vc: headline_bits.append(f"VIX:{vc.get('regime')}")
    if (zc.get("confluence") or {}).get("demand"): headline_bits.append("confluent-demand")
    if (zc.get("confluence") or {}).get("supply"): headline_bits.append("confluent-supply")

    levels_line = ""
    if lv:
        parts = []
        if "pdh" in lv: parts.append(f"PDH {lv['pdh']}({lv.get('pdh_state','')})")
        if "pdl" in lv: parts.append(f"PDL {lv['pdl']}({lv.get('pdl_state','')})")
        if "two_day_high" in lv:
            b = "broke" if lv.get("two_day_high_broken") else "intact"
            parts.append(f"2DH {lv['two_day_high']}({b})")
        if "two_day_low" in lv:
            b = "broke" if lv.get("two_day_low_broken") else "intact"
            parts.append(f"2DL {lv['two_day_low']}({b})")
        if "three_day_high" in lv:
            b = "broke" if lv.get("three_day_high_broken") else "intact"
            parts.append(f"3DH {lv['three_day_high']}({b})")
        if "three_day_low" in lv:
            b = "broke" if lv.get("three_day_low_broken") else "intact"
            parts.append(f"3DL {lv['three_day_low']}({b})")
        if "week_high" in lv: parts.append(f"WkH {lv['week_high']}({lv.get('week_high_state','')})")
        if "week_low"  in lv: parts.append(f"WkL {lv['week_low']}({lv.get('week_low_state','')})")
        if parts: levels_line = "LEVELS: " + " | ".join(parts) + "\n"

    vix_line = ""
    if vc:
        vix_line = (f"VIX:{vc.get('vix')} | {vc.get('pctile_1y')}pct(1y) | "
                    f"{vc.get('regime')} | 5d:{vc.get('dir_5d')} | 20dRange:{vc.get('pos_20d')}\n")

    zones_line = ""
    if zc:
        def fmt_z(z):
            if not z: return "—"
            tag = "fresh" if z.get("fresh") else f"tested({z.get('touches')})"
            return f"{z['low']}-{z['high']} {tag}({z.get('dist_pct',0):+.1f}%)"
        s5m = zc.get("tf_5min", {}); b1h = zc.get("tf_1h", {})
        zparts = [f"5m  DZ:{fmt_z(s5m.get('demand'))} SZ:{fmt_z(s5m.get('supply'))}",
                  f"1h  DZ:{fmt_z(b1h.get('demand'))} SZ:{fmt_z(b1h.get('supply'))}"]
        cf = [k for k in ("demand", "supply") if (zc.get("confluence") or {}).get(k)]
        if cf: zparts.append("CONFLUENCE:" + "+".join(cf))
        zones_line = "ZONES:\n  " + "\n  ".join(zparts) + "\n"

    return (
        f"[{symbol} {date}]\n"
        f"HEADLINE: {' | '.join(headline_bits)}\n"
        f"C:{close} ({'+' if (day_chg or 0)>=0 else ''}{day_chg}%) | "
        f"5d:{d5_low}-{d5_high} C@{pct_pos}pct\n"
        f"ATR:{atr} | Vol:{last_vol/1e6:.1f}M({vol_ratio}x avg) | "
        f"3d:{'+' if (d3_chg or 0)>=0 else ''}{d3_chg if d3_chg is not None else 0}%\n\n"
        f"EMA: 50@{e50} {'<' if (e50 or 0)<(e100 or 0) else '>'} "
        f"100@{e100} {'<' if (e100 or 0)<(e200 or 0) else '>'} "
        f"200@{e200} [{stack.split()[0].upper()}]\n"
        f"VWAP:{vwap} C{'+' if (c_vs_vwap or 0)>=0 else ''}{c_vs_vwap}% | "
        f"C vs EMA200:{'+' if (c_vs_200 or 0)>=0 else ''}{c_vs_200}%\n"
        f"Trend:{tech.get('trend','-')} | "
        f"Struct:{','.join(tech.get('structure',[]) or ['-'])}\n\n"
        f"S:{sup_str}  (nearest {(dist_s or 0):+.1f}%)\n"
        f"R:{res_str}  (nearest {'+' if (dist_r or 0)>=0 else ''}{dist_r or 0}%)\n"
        f"SH:{sh_str} | SL:{sl_str}\n"
        f"{levels_line}"
        f"{vix_line}"
        f"{zones_line}"
    )


# ── Decision (LLM or rule-based fallback) ────────────────────────────

def _safe_num(x, default=0.0):
    try:
        v = float(x)
        return v if v == v else default
    except (TypeError, ValueError):
        return default


def decide(symbol: str, date: str, tech: dict, vc: dict, zc: dict) -> tuple[dict, str]:
    """Returns (decision_dict, prompt_text) so the caller can store the exact prompt."""
    user = build_prompt(symbol, date, tech, vc, zc)
    dec  = _parse_json(_llm_generate(user))
    if dec.get("action") in ("BUY", "SELL", "HOLD"):
        return dec, user

    # Rule-based fallback
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
                     fwd_df, max_hold: int = MAX_HOLD_DAYS):
    if action == "HOLD" or not sl or not tp or fwd_df.empty:
        return "HOLD", entry, None, 0
    for i, (idx, row) in enumerate(fwd_df.head(max_hold).iterrows()):
        lo, hi = float(row["low"]), float(row["high"])
        ds = str(idx.date())
        if action == "BUY":
            if lo <= sl: return "SL_HIT", round(sl, 2), ds, i + 1
            if hi >= tp: return "TP_HIT", round(tp, 2), ds, i + 1
        elif action == "SELL":
            if hi >= sl: return "SL_HIT", round(sl, 2), ds, i + 1
            if lo <= tp: return "TP_HIT", round(tp, 2), ds, i + 1
    last = fwd_df.head(max_hold).iloc[-1]
    return "TIME_EXIT", round(float(last["close"]), 2), str(fwd_df.head(max_hold).index[-1].date()), max_hold


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

def generate_sample(ta: TechnicalsAnalyzer, symbol: str, date: str) -> dict | None:
    # 1. Dedup check
    if _already_exists(symbol, date):
        print(f"  [skip] {symbol} {date} already in DB")
        return None

    # 2. Load candles
    try:
        df = ta.load_candles(symbol, date, LOOKBACK_DAYS)
    except Exception as e:
        print(f"  [skip] {symbol} {date} — candles: {e}")
        return None

    # 3. Compute technicals, VIX, zones
    try:
        tech = ta.compute(df)
    except Exception as e:
        print(f"  [skip] {symbol} {date} — technicals: {e}")
        return None
    vc = ta.vix_context(date)
    zc = ta.compute_zones(symbol, date)
    if vc:
        tech["vix_ctx"] = vc
    if zc:
        tech["zones"] = zc

    # 4. LLM decision (also returns the exact prompt so we can store it)
    dec, prompt_text = decide(symbol, date, tech, vc, zc)
    action  = dec.get("action", "HOLD")
    if action not in ("BUY", "SELL", "HOLD"):
        action = "HOLD"
    sl      = _safe_num(dec.get("stop_loss",  0))
    tp      = _safe_num(dec.get("take_profit", 0))
    rsn     = dec.get("reasoning", "—")
    conf    = _safe_num(dec.get("confidence", 0.5), 0.5)
    qty     = max(1, int(_safe_num(dec.get("quantity", 1), 1)))

    # 5. Simulate outcome
    entry_p = tech["close"]
    try:
        fwd = ta.load_forward(symbol, date, MAX_HOLD_DAYS)
    except Exception:
        fwd = __import__("pandas").DataFrame()
    exit_reason, exit_p, exit_date, _ = simulate_outcome(action, entry_p, sl, tp, fwd)
    gross = compute_pnl(action, entry_p, exit_p, qty)
    pnl   = round(gross - trade_costs(entry_p, exit_p, qty, action), 2)

    mark = "PROFIT" if pnl > 0 else "LOSS" if pnl < 0 else "FLAT"
    print(f"  {symbol} {date} | {action} SL={sl:.2f} TP={tp:.2f} "
          f"→ {exit_reason} {exit_p:.2f} [{mark} {pnl:+.0f}]")

    return {
        "symbol": symbol, "decision_date": date,
        "action": action, "entry_price": round(entry_p, 2),
        "stop_loss": sl, "take_profit": tp,
        "exit_price": exit_p, "exit_date": exit_date,
        "exit_reason": exit_reason, "pnl": pnl,
        "reasoning": rsn, "confidence": conf,
        "features_used": dec.get("features_used", []),
        "technicals_json": {k: v for k, v in tech.items() if k != "_df"},
        "prompt_text": prompt_text,
    }


# ── Date utilities ────────────────────────────────────────────────────

def trading_days(start: str, end: str) -> list[str]:
    """All weekday (Mon-Fri) dates between start and end inclusive."""
    days, dt = [], datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    while dt <= end_dt:
        if dt.weekday() < 5:
            days.append(dt.strftime("%Y-%m-%d"))
        dt += timedelta(days=1)
    return days


def random_trading_days(start: str, end: str, count: int, symbol: str) -> list[str]:
    """
    Pick up to `count` random weekday dates from [start, end] that are NOT
    already in the DB for this symbol.  Returns a shuffled list — each call
    gives a different ordering so repeated runs fill in different gaps.
    """
    all_days = trading_days(start, end)

    # Pre-fetch already-processed dates in one query (avoids N round-trips)
    with sqlite3.connect(DB_PATH) as con:
        done = {r[0] for r in con.execute(
            "SELECT decision_date FROM trade_log WHERE symbol=?", (symbol,)
        )}

    available = [d for d in all_days if d not in done]
    random.shuffle(available)

    skipped = len(all_days) - len(available)
    print(f"Date pool: {len(all_days)} weekdays | "
          f"{skipped} already in DB | "
          f"{len(available)} available | picking {min(count, len(available))}")

    return available[:count]


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    init_db()

    # Build full work list: (symbol, date) pairs across all symbols
    work: list[tuple[str, str]] = []
    for sym in SYMBOLS:
        dates = random_trading_days(START_DATE, END_DATE, COUNT, sym)
        if not dates:
            print(f"All dates already generated for {sym}.")
        else:
            print(f"Queuing {len(dates)} dates for {sym}")
            work.extend((sym, d) for d in dates)

    if not work:
        print("Nothing to do. Extend START_DATE/END_DATE or add more symbols.")
    elif WORKERS <= 1:
        ta = TechnicalsAnalyzer()
        generated = 0
        for sym, date in work:
            row = generate_sample(ta, sym, date)
            if row:
                _store(row)
                generated += 1
        print(f"\nDone. {generated} new samples stored in {DB_PATH}")
    else:
        # NOTE: With a single Ollama instance, LLM calls are still sequential
        # (Ollama queues them). Speedup comes from parallel candle loading,
        # technicals computation, and outcome simulation — and from running
        # multiple symbols whose LLM calls can interleave in Ollama's queue.
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if HF_MODEL:
            print(f"\nParallel generation: {WORKERS} workers | HF model — true parallel inference")
        else:
            print(f"\nParallel generation: {WORKERS} workers | "
                  f"note: single Ollama instance queues LLM calls; "
                  f"speedup from parallel data loading + multi-symbol interleaving")

        def _job(sym_date: tuple[str, str]) -> dict | None:
            sym, date = sym_date
            ta_local = TechnicalsAnalyzer()   # one per thread — avoids shared state
            row = generate_sample(ta_local, sym, date)
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
                    sym, date = futures[f]
                    print(f"  [error] {sym} {date}: {e}")

        print(f"\nDone. {generated} new samples stored in {DB_PATH}")
