"""
mock_data.py
────────────
Inserts synthetic BUY/SELL trade samples directly into the DB.
Used ONLY for smoke-testing the splitter → train → evaluate pipeline.
Does NOT call Ollama. Does NOT affect real data generation.

Usage:
    python mock_data.py              # inserts 60 samples
    python mock_data.py --count 30   # insert N samples
    python mock_data.py --clear      # wipe mock rows first, then insert
"""
from __future__ import annotations
import argparse, json, random, sqlite3
from datetime import datetime, timedelta
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env", override=False)
except ImportError:
    pass

import os
DB_PATH = os.environ.get("DB_PATH", "trading_agent.db")

# ── Make sure DB has the right schema ────────────────────────────────────────
def _ensure_db():
    with sqlite3.connect(DB_PATH) as con:
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
        cols = {r[1] for r in con.execute("PRAGMA table_info(trade_log)")}
        if "prompt_text" not in cols:
            con.execute("ALTER TABLE trade_log ADD COLUMN prompt_text TEXT")


# ── Synthetic sample builder ──────────────────────────────────────────────────
_SYMBOLS  = ["CANBK", "CANBK", "CANBK"]   # single symbol, many dates

def _weekdays(start="2022-01-03", n=300):
    days, dt = [], datetime.strptime(start, "%Y-%m-%d")
    while len(days) < n:
        if dt.weekday() < 5:
            days.append(dt.strftime("%Y-%m-%d"))
        dt += timedelta(days=1)
    return days

_ALL_DAYS = _weekdays()

def _make_sample(date: str, symbol: str = "CANBK") -> dict:
    rng = random.Random(date + symbol)   # deterministic per date

    base = rng.uniform(50, 150)
    close = round(base, 2)
    ema50 = round(base * rng.uniform(0.95, 1.05), 2)
    ema100 = round(ema50 * rng.uniform(0.97, 1.03), 2)
    ema200 = round(ema100 * rng.uniform(0.97, 1.03), 2)
    vwap = round(base * rng.uniform(0.97, 1.03), 2)
    atr  = round(base * 0.015, 2)

    above_vwap = close > vwap
    stack = ("bullish (50>100>200)" if ema50 > ema100 > ema200
             else "bearish (50<100<200)" if ema50 < ema100 < ema200
             else "mixed")
    trend_choices = ["uptrend", "downtrend", "ranging"]
    trend = rng.choice(trend_choices)

    sup_low = round(close * rng.uniform(0.94, 0.98), 2)
    sup_hi  = round(sup_low * 1.005, 2)
    res_low = round(close * rng.uniform(1.02, 1.06), 2)
    res_hi  = round(res_low * 1.005, 2)

    pdh = round(close * rng.uniform(1.005, 1.02), 2)
    pdl = round(close * rng.uniform(0.98, 0.995), 2)
    vix = round(rng.uniform(10, 30), 2)
    vix_regime = "LOW" if vix < 13 else "NORMAL" if vix < 18 else "ELEVATED" if vix < 25 else "HIGH"

    # Decide action based on mock signals
    bullish = stack.startswith("bullish") and above_vwap and trend == "uptrend"
    bearish = stack.startswith("bearish") and not above_vwap and trend == "downtrend"
    if bullish and rng.random() > 0.3:
        action = "BUY"
    elif bearish and rng.random() > 0.3:
        action = "SELL"
    else:
        action = rng.choice(["BUY", "SELL", "HOLD"])

    # Prices — set SL/TP so actual R:R (recalculated from prices) is >= 1.5
    if action == "BUY":
        risk_pct = rng.uniform(0.015, 0.025)      # 1.5–2.5% stop
        sl = round(close * (1 - risk_pct), 2)
        reward_pct = risk_pct * rng.uniform(1.6, 2.5)  # 1.6–2.5× risk
        tp = round(close * (1 + reward_pct), 2)
        rr = round((tp - close) / max(close - sl, 0.01), 2)
    elif action == "SELL":
        risk_pct = rng.uniform(0.015, 0.025)
        sl = round(close * (1 + risk_pct), 2)
        reward_pct = risk_pct * rng.uniform(1.6, 2.5)
        tp = round(close * (1 - reward_pct), 2)
        rr = round((close - tp) / max(sl - close, 0.01), 2)
    else:
        sl = tp = rr = 0.0

    conf = round(rng.uniform(0.55, 0.90), 2)

    # Simulate outcome
    exit_choices = ["TP_HIT", "SL_HIT", "TIME_EXIT"]
    exit_reason = rng.choices(exit_choices, weights=[0.55, 0.25, 0.20])[0]
    exit_date_dt = datetime.strptime(date, "%Y-%m-%d") + timedelta(days=rng.randint(1, 5))
    exit_date = exit_date_dt.strftime("%Y-%m-%d")

    if action == "HOLD":
        exit_price = close; exit_reason = "HOLD"; pnl = 0.0
    elif exit_reason == "TP_HIT":
        exit_price = tp
        pnl = round((tp - close if action == "BUY" else close - tp) * 10, 2)
    elif exit_reason == "SL_HIT":
        exit_price = sl
        pnl = round(-abs(close - sl) * 10, 2)
    else:
        drift = rng.uniform(-0.01, 0.015)
        exit_price = round(close * (1 + drift), 2)
        pnl = round((exit_price - close if action == "BUY" else close - exit_price) * 10, 2)

    # Build technicals_json
    tech = {
        "close": close, "vwap": vwap, "above_vwap": above_vwap,
        "ema_50": ema50, "ema_100": ema100, "ema_200": ema200,
        "ema_stack": stack, "trend": trend, "structure": [],
        "nearest_support":    {"low": sup_low, "high": sup_hi, "mid": round((sup_low+sup_hi)/2,2)},
        "nearest_resistance": {"low": res_low, "high": res_hi, "mid": round((res_low+res_hi)/2,2)},
        "levels": {"pdh": pdh, "pdl": pdl, "pdh_state": "intact", "pdl_state": "intact"},
        "vix_ctx": {"vix": vix, "regime": vix_regime, "dir_5d": "flat", "pctile_1y": 50, "pos_20d": "mid"},
        "support_zones":    [{"low": sup_low, "high": sup_hi, "mid": round((sup_low+sup_hi)/2,2)}],
        "resistance_zones": [{"low": res_low, "high": res_hi, "mid": round((res_low+res_hi)/2,2)}],
        "recent_candles": [],
    }

    features = ["ema_stack", "vwap", "nearest_support", "nearest_resistance", "levels", "vix_ctx"]

    # Build prompt_text matching pipeline.py build_prompt() format
    bias = "bullish" if bullish else "bearish" if bearish else "neutral"
    prompt = (
        f"[{symbol} {date}]\n"
        f"HEADLINE: bias:{bias} | trend:{trend} | VIX:{vix_regime}\n"
        f"C:{close} (+0.5%) | 5d:{round(close*0.96,2)}-{round(close*1.04,2)} C@50pct\n"
        f"ATR:{atr} | Vol:10.0M(1.0x avg) | 3d:+0.3%\n\n"
        f"EMA: 50@{ema50} > 100@{ema100} > 200@{ema200} [{stack.split()[0].upper()}]\n"
        f"VWAP:{vwap} C{'+' if above_vwap else ''}{round((close-vwap)/vwap*100,2)}% | C vs EMA200:+{round((close-ema200)/ema200*100,2)}%\n"
        f"Trend:{trend} | Struct:-\n\n"
        f"S:{sup_low}-{sup_hi}(-2.0%)  (nearest -2.0%)\n"
        f"R:{res_low}-{res_hi}(+3.0%)  (nearest +3.0%)\n"
        f"SH:{round(close*1.05,2)} | SL:{round(close*0.95,2)}\n"
        f"LEVELS: PDH {pdh}(intact) | PDL {pdl}(intact)\n"
        f"VIX:{vix} | 50pct(1y) | {vix_regime} | 5d:flat | 20dRange:mid\n"
        f"ZONES:\n  5m  DZ:— SZ:—\n  1h  DZ:— SZ:—\n"
    )

    reasoning_map = {
        "BUY":  f"Bullish EMA stack ({stack}) with price above VWAP at {vwap}. "
                f"Strong support at {sup_low}-{sup_hi}. PDH {pdh} acts as next resistance. "
                f"VIX {vix_regime} regime allows participation.",
        "SELL": f"Bearish EMA stack ({stack}) with price below VWAP at {vwap}. "
                f"Resistance at {res_low}-{res_hi} capping upside. "
                f"PDL {pdl} is the first target.",
        "HOLD": f"Conflicting signals: EMA stack {stack} but trend is {trend}. "
                f"No clear directional edge. Waiting for confirmation.",
    }

    return {
        "symbol": symbol, "decision_date": date,
        "action": action, "entry_price": close,
        "stop_loss": sl, "take_profit": tp,
        "exit_price": exit_price, "exit_date": exit_date,
        "exit_reason": exit_reason, "pnl": pnl,
        "reasoning": reasoning_map[action],
        "confidence": conf,
        "features_used": json.dumps(features),
        "technicals_json": json.dumps(tech),
        "prompt_text": prompt,
        "created_at": datetime.now().isoformat(),
    }


def insert_mock(count: int = 60, clear_mock: bool = False):
    _ensure_db()

    with sqlite3.connect(DB_PATH) as con:
        if clear_mock:
            # Only remove rows whose reasoning contains "Mock:" marker
            con.execute("DELETE FROM trade_log WHERE reasoning LIKE 'Bullish EMA stack%' "
                        "OR reasoning LIKE 'Bearish EMA stack%' "
                        "OR reasoning LIKE 'Conflicting signals%'")
            print("Cleared previous mock rows.")

        existing = {r[0] for r in con.execute(
            "SELECT decision_date FROM trade_log WHERE symbol='CANBK'"
        )}

    available = [d for d in _ALL_DAYS if d not in existing]
    random.shuffle(available)
    chosen = available[:count]

    inserted = skipped = 0
    with sqlite3.connect(DB_PATH) as con:
        for date in chosen:
            row = _make_sample(date)
            try:
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
                    row["features_used"], row["technicals_json"],
                    row["prompt_text"], row["created_at"],
                ))
                inserted += 1
            except Exception as e:
                skipped += 1

    # Summary
    with sqlite3.connect(DB_PATH) as con:
        actions = con.execute(
            "SELECT action, COUNT(*) FROM trade_log GROUP BY action"
        ).fetchall()

    print(f"\nInserted {inserted} mock samples (skipped {skipped} duplicates)")
    print("DB totals:")
    for a, n in actions:
        print(f"  {a}: {n}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=60)
    p.add_argument("--clear", action="store_true", help="remove previous mock rows first")
    args = p.parse_args()
    insert_mock(args.count, args.clear)
