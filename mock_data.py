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
                decision_slot   TEXT DEFAULT 'close',
                planned_entry   REAL,
                post_reasoning  TEXT,
                created_at      TEXT,
                UNIQUE(symbol, decision_date, decision_slot)
            );
        """)
        cols = {r[1] for r in con.execute("PRAGMA table_info(trade_log)")}
        for col, defn in [("prompt_text", "TEXT"), ("decision_slot", "TEXT"),
                          ("planned_entry", "REAL"), ("post_reasoning", "TEXT")]:
            if col not in cols:
                con.execute(f"ALTER TABLE trade_log ADD COLUMN {col} {defn}")


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

def _make_sample(date: str, symbol: str = "CANBK", slot: str | None = None) -> dict:
    rng = random.Random(date + symbol + (slot or ""))   # deterministic per (date, symbol, slot)

    if slot is None:
        slot = rng.choice(["open", "mid", "close"])

    base = rng.uniform(50, 150)
    close = round(base, 2)
    # Vary price slightly by slot
    if slot == "open":
        cur = round(base * rng.uniform(0.99, 1.01), 2)
    elif slot == "mid":
        cur = round(base * rng.uniform(0.995, 1.005), 2)
    else:
        cur = close
    ema50 = round(base * rng.uniform(0.95, 1.05), 2)
    ema100 = round(ema50 * rng.uniform(0.97, 1.03), 2)
    ema200 = round(ema100 * rng.uniform(0.97, 1.03), 2)
    vwap = round(base * rng.uniform(0.97, 1.03), 2)
    atr  = round(base * 0.015, 2)

    above_vwap = cur > vwap
    stack = ("bullish (50>100>200)" if ema50 > ema100 > ema200
             else "bearish (50<100<200)" if ema50 < ema100 < ema200
             else "mixed")
    trend_choices = ["uptrend", "downtrend", "ranging"]
    trend = rng.choice(trend_choices)

    sup_low = round(cur * rng.uniform(0.94, 0.98), 2)
    sup_hi  = round(sup_low * 1.005, 2)
    res_low = round(cur * rng.uniform(1.02, 1.06), 2)
    res_hi  = round(res_low * 1.005, 2)

    pdh = round(cur * rng.uniform(1.005, 1.02), 2)
    pdl = round(cur * rng.uniform(0.98, 0.995), 2)
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

    # planned_entry: sometimes use a limit near S/R rather than cur price
    if action == "BUY":
        risk_pct = rng.uniform(0.015, 0.025)
        # ~40% chance of planning a limit entry near support rather than current price
        if rng.random() < 0.4 and sup_hi < cur:
            planned_entry = round(cur * rng.uniform(0.985, 0.998), 2)
        else:
            planned_entry = cur
        sl = round(planned_entry * (1 - risk_pct), 2)
        reward_pct = risk_pct * rng.uniform(1.6, 2.5)
        tp = round(planned_entry * (1 + reward_pct), 2)
        rr = round((tp - planned_entry) / max(planned_entry - sl, 0.01), 2)
    elif action == "SELL":
        risk_pct = rng.uniform(0.015, 0.025)
        if rng.random() < 0.4 and res_low > cur:
            planned_entry = round(cur * rng.uniform(1.002, 1.015), 2)
        else:
            planned_entry = cur
        sl = round(planned_entry * (1 + risk_pct), 2)
        reward_pct = risk_pct * rng.uniform(1.6, 2.5)
        tp = round(planned_entry * (1 - reward_pct), 2)
        rr = round((planned_entry - tp) / max(sl - planned_entry, 0.01), 2)
    else:
        planned_entry = cur
        sl = tp = rr = 0.0

    conf = round(rng.uniform(0.55, 0.90), 2)

    # Simulate outcome
    exit_choices = ["TP_HIT", "SL_HIT", "TIME_EXIT", "ENTRY_NOT_FILLED"]
    weights = [0.50, 0.25, 0.18, 0.07] if planned_entry != cur else [0.55, 0.25, 0.20, 0.0]
    exit_reason = rng.choices(exit_choices, weights=weights)[0]
    exit_date_dt = datetime.strptime(date, "%Y-%m-%d") + timedelta(days=rng.randint(1, 5))
    exit_date = exit_date_dt.strftime("%Y-%m-%d")

    if action == "HOLD":
        exit_price = cur; exit_reason = "HOLD"; pnl = 0.0
    elif exit_reason == "TP_HIT":
        exit_price = tp
        pnl = round((tp - planned_entry if action == "BUY" else planned_entry - tp) * 10, 2)
    elif exit_reason == "SL_HIT":
        exit_price = sl
        pnl = round(-abs(planned_entry - sl) * 10, 2)
    elif exit_reason == "ENTRY_NOT_FILLED":
        exit_price = cur; pnl = 0.0
    else:
        drift = rng.uniform(-0.01, 0.015)
        exit_price = round(cur * (1 + drift), 2)
        pnl = round((exit_price - planned_entry if action == "BUY" else planned_entry - exit_price) * 10, 2)

    # Optional mock post_reasoning (50% of non-HOLD, non-ENTRY_NOT_FILLED rows)
    post_rsn = None
    if action != "HOLD" and exit_reason not in ("ENTRY_NOT_FILLED",) and rng.random() < 0.5:
        outcome_word = "hit target" if exit_reason == "TP_HIT" else ("stopped out" if exit_reason == "SL_HIT" else "expired")
        post_rsn = (
            f"Price {outcome_word} after {action} from {planned_entry:.2f}. "
            f"EMA stack was {stack}; VWAP at {vwap:.2f} provided {'support' if action=='BUY' else 'resistance'}. "
            f"The setup {'worked as planned' if pnl > 0 else 'failed because momentum reversed before target'}."
        )

    # Build technicals_json
    vol_trend_5d = rng.choice(["increasing", "decreasing", "flat"])
    price_dir_5d = rng.choice(["up", "down", "flat"])
    vol_sentiment = {
        ("up",   "increasing"): "bullish_momentum",
        ("up",   "decreasing"): "exhaustion",
        ("down", "increasing"): "bearish_pressure",
        ("down", "decreasing"): "potential_base",
    }.get((price_dir_5d, vol_trend_5d), "neutral")
    avg_5d_vol  = round(rng.uniform(8e6, 25e6))
    avg_20d_vol = round(rng.uniform(8e6, 25e6))
    vol_ratio_5_20 = round(avg_5d_vol / avg_20d_vol, 2) if avg_20d_vol > 0 else 1.0

    regime_trading = "consolidation" if vix < 12 else "volatile" if vix >= 15 else "sideways"
    trend_10d = rng.choice(["rising", "falling", "flat"])
    quantity_bias = ("fewer shares, bigger point targets" if regime_trading == "volatile"
                     else "more shares, tighter targets" if regime_trading == "consolidation"
                     else "standard sizing")

    step = 25 if cur < 500 else 100
    import math
    nearest_round = round(round(cur / step) * step, 2)
    major_step = step * 4
    round_levels = [r for r in [
        {"level": float(nearest_round - step),
         "type": "major" if (nearest_round - step) % major_step == 0 else "minor",
         "dist_pct": round((nearest_round - step - cur) / cur * 100, 2),
         "side": "support"},
        {"level": float(nearest_round),
         "type": "major" if nearest_round % major_step == 0 else "minor",
         "dist_pct": round((nearest_round - cur) / cur * 100, 2),
         "side": "support" if nearest_round < cur else "resistance"},
        {"level": float(nearest_round + step),
         "type": "major" if (nearest_round + step) % major_step == 0 else "minor",
         "dist_pct": round((nearest_round + step - cur) / cur * 100, 2),
         "side": "resistance"},
    ] if abs(r["dist_pct"]) <= 8.0 and abs(r["dist_pct"]) > 0.1]

    sh_30d = [
        {"date": str((datetime.strptime(date, "%Y-%m-%d") - timedelta(days=rng.randint(2, 25))).date()),
         "price": round(cur * rng.uniform(1.01, 1.05), 2)}
        for _ in range(rng.randint(1, 3))
    ]
    sl_30d = [
        {"date": str((datetime.strptime(date, "%Y-%m-%d") - timedelta(days=rng.randint(2, 25))).date()),
         "price": round(cur * rng.uniform(0.95, 0.99), 2)}
        for _ in range(rng.randint(1, 3))
    ]

    tech = {
        "close": cur, "vwap": vwap, "above_vwap": above_vwap,
        "ema_50": ema50, "ema_100": ema100, "ema_200": ema200,
        "ema_stack": stack, "trend": trend, "structure": [],
        "nearest_support":    {"low": sup_low, "high": sup_hi, "mid": round((sup_low+sup_hi)/2,2)},
        "nearest_resistance": {"low": res_low, "high": res_hi, "mid": round((res_low+res_hi)/2,2)},
        "levels": {"pdh": pdh, "pdl": pdl, "pdh_state": "intact", "pdl_state": "intact"},
        "vix_ctx": {"vix": vix, "regime": vix_regime, "dir_5d": "flat", "pctile_1y": 50,
                    "pos_20d": "mid", "regime_trading": regime_trading,
                    "trend_10d": trend_10d, "quantity_bias": quantity_bias},
        "support_zones":    [{"low": sup_low, "high": sup_hi, "mid": round((sup_low+sup_hi)/2,2)}],
        "resistance_zones": [{"low": res_low, "high": res_hi, "mid": round((res_low+res_hi)/2,2)}],
        "recent_candles": [],
        "volume_analysis": {
            "avg_5d": avg_5d_vol, "avg_20d": avg_20d_vol,
            "ratio": vol_ratio_5_20, "vol_trend": vol_trend_5d,
            "price_dir_5d": price_dir_5d, "sentiment": vol_sentiment,
        },
        "swing_highs_30d": sh_30d,
        "swing_lows_30d":  sl_30d,
        "round_levels": round_levels,
    }

    features = ["ema_stack", "vwap", "nearest_support", "nearest_resistance", "levels", "vix_ctx"]

    # Build prompt_text matching pipeline.py build_prompt() format (with @slot)
    bias = "bullish" if bullish else "bearish" if bearish else "neutral"
    sh30_str = "  ".join(f"{s['price']:.1f}({s['date'][5:]})" for s in sh_30d) or "—"
    sl30_str = "  ".join(f"{s['price']:.1f}({s['date'][5:]})" for s in sl_30d) or "—"
    rl_parts = [f"{r['level']}({'M' if r['type']=='major' else 'm'},{r['dist_pct']:+.1f}%)"
                for r in round_levels]
    round_str = "ROUND: " + "  ".join(rl_parts) + "\n" if rl_parts else ""
    prompt = (
        f"[{symbol} {date} @{slot}]\n"
        f"HEADLINE: bias:{bias} | trend:{trend} | VIX:{vix_regime}\n"
        f"C:{cur} (+0.5%) | 5d:{round(cur*0.96,2)}-{round(cur*1.04,2)} C@50pct\n"
        f"ATR:{atr} | Vol:10.0M(1.0x avg) | 3d:+0.3%\n"
        f"VOL_TREND: 5d:{avg_5d_vol/1e6:.1f}M 20d:{avg_20d_vol/1e6:.1f}M ({vol_ratio_5_20:.2f}x) | {vol_sentiment}\n"
        f"\nEMA: 50@{ema50} > 100@{ema100} > 200@{ema200} [{stack.split()[0].upper()}]\n"
        f"VWAP:{vwap} C{'+' if above_vwap else ''}{round((cur-vwap)/vwap*100,2)}% | C vs EMA200:+{round((cur-ema200)/ema200*100,2)}%\n"
        f"Trend:{trend} | Struct:-\n\n"
        f"S:{sup_low}-{sup_hi}(-2.0%)  (nearest -2.0%)\n"
        f"R:{res_low}-{res_hi}(+3.0%)  (nearest +3.0%)\n"
        f"{round_str}"
        f"SWH/L-90d: H:{round(cur*1.05,2)}  L:{round(cur*0.95,2)}\n"
        f"SWH/L-30d: H:{sh30_str}  L:{sl30_str}\n"
        f"LEVELS: PDH {pdh}(intact) | PDL {pdl}(intact)\n"
        f"VIX:{vix} | 50pct(1y) | {vix_regime} | {regime_trading} | "
        f"5d:flat 10d:{trend_10d} | 20dRange:mid | size:{quantity_bias}\n"
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
        "action": action, "entry_price": planned_entry,
        "stop_loss": sl, "take_profit": tp,
        "exit_price": exit_price, "exit_date": exit_date,
        "exit_reason": exit_reason, "pnl": pnl,
        "reasoning": reasoning_map[action],
        "confidence": conf,
        "features_used": json.dumps(features),
        "technicals_json": json.dumps(tech),
        "prompt_text": prompt,
        "decision_slot": slot,
        "planned_entry": planned_entry,
        "post_reasoning": post_rsn,
        "created_at": datetime.now().isoformat(),
    }


def insert_mock(count: int = 60, clear_mock: bool = False):
    _ensure_db()

    with sqlite3.connect(DB_PATH) as con:
        if clear_mock:
            con.execute("DELETE FROM trade_log WHERE reasoning LIKE 'Bullish EMA stack%' "
                        "OR reasoning LIKE 'Bearish EMA stack%' "
                        "OR reasoning LIKE 'Conflicting signals%'")
            print("Cleared previous mock rows.")

        existing = {(r[0], r[1]) for r in con.execute(
            "SELECT decision_date, decision_slot FROM trade_log WHERE symbol='CANBK'"
        )}

    # Build (date, slot) pairs not yet in DB
    all_pairs = [(d, s) for d in _ALL_DAYS for s in ("open", "mid", "close")]
    available = [(d, s) for d, s in all_pairs if (d, s) not in existing]
    random.shuffle(available)
    chosen = available[:count]

    inserted = skipped = 0
    with sqlite3.connect(DB_PATH) as con:
        for date, slot in chosen:
            row = _make_sample(date, slot=slot)
            try:
                con.execute("""
                    INSERT OR IGNORE INTO trade_log
                      (symbol, decision_date, action, entry_price, stop_loss, take_profit,
                       exit_price, exit_date, exit_reason, pnl, reasoning, confidence,
                       features_used, technicals_json, prompt_text,
                       decision_slot, planned_entry, post_reasoning,
                       created_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    row["symbol"], row["decision_date"], row["action"],
                    row["entry_price"], row["stop_loss"], row["take_profit"],
                    row["exit_price"], row["exit_date"], row["exit_reason"], row["pnl"],
                    row["reasoning"], row["confidence"],
                    row["features_used"], row["technicals_json"],
                    row["prompt_text"],
                    row["decision_slot"], row["planned_entry"], row["post_reasoning"],
                    row["created_at"],
                ))
                inserted += 1
            except Exception as e:
                print(f"  [error] {date} @{slot}: {e}")
                skipped += 1

    # Summary
    with sqlite3.connect(DB_PATH) as con:
        actions = con.execute(
            "SELECT action, COUNT(*) FROM trade_log GROUP BY action"
        ).fetchall()
        slots = con.execute(
            "SELECT decision_slot, COUNT(*) FROM trade_log GROUP BY decision_slot"
        ).fetchall()

    print(f"\nInserted {inserted} mock samples (skipped {skipped} duplicates)")
    print("DB totals by action:", {a: n for a, n in actions})
    print("DB totals by slot:  ", {s: n for s, n in slots})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=60)
    p.add_argument("--clear", action="store_true", help="remove previous mock rows first")
    args = p.parse_args()
    insert_mock(args.count, args.clear)
