"""
splitter.py
───────────
Reads trading samples from SQLite → quality-filtered JSONL for fine-tuning.

Each row in the DB becomes ONE training example (pre-trade decision only —
stateless pipeline has no post-trade analysis).

Usage:
    python splitter.py
    python splitter.py --db trading_agent.db --out data/trading.jsonl
    python splitter.py --min_confidence 0.5 --min_rr 1.5 --val_ratio 0.1 --test_ratio 0.1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sqlite3
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env", override=False)
except ImportError:
    pass

from prompts import SYSTEM_PROMPT


def _db(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path, timeout=30)
    con.execute("PRAGMA busy_timeout=30000")
    return con


# ── Load ──────────────────────────────────────────────────────────────

def load_trades(db_path: str) -> list[dict]:
    with _db(db_path) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("""
            SELECT * FROM trade_log
            WHERE action != 'HOLD'
              AND reasoning IS NOT NULL
              AND (is_mock = 0 OR is_mock IS NULL)
            ORDER BY decision_date, id
        """).fetchall()
    return [dict(r) for r in rows]


# ── Build training example ────────────────────────────────────────────

def _parse_technicals(tech_json: str) -> str:
    try:
        t = json.loads(tech_json or "{}")
    except Exception:
        return "(technical data unavailable)"
    lines = [
        f"Close: {t.get('close','—')}  VWAP: {t.get('vwap','—')} "
        f"({'above' if t.get('above_vwap') else 'below'} VWAP)",
        f"EMA50: {t.get('ema_50','—')}  EMA100: {t.get('ema_100','—')}  EMA200: {t.get('ema_200','—')}",
        f"Stack: {t.get('ema_stack','—')}   Trend: {t.get('trend','—')}  "
        f"Struct: {', '.join(t.get('structure',[]) or ['—'])}",
        f"Nearest Support:    {t.get('nearest_support','—')}",
        f"Nearest Resistance: {t.get('nearest_resistance','—')}",
    ]
    lv = t.get("levels", {}) or {}
    if lv.get("pdh"):
        lines.append(f"PDH: {lv['pdh']} ({lv.get('pdh_state','')})  "
                     f"PDL: {lv.get('pdl','—')} ({lv.get('pdl_state','')})")
    vc = t.get("vix_ctx", {}) or {}
    if vc.get("vix"):
        lines.append(f"VIX: {vc['vix']} | {vc.get('regime','')} | {vc.get('dir_5d','')}")
    return "\n".join(lines)


def _tech_field(trade: dict, key: str, default=None):
    try:
        return json.loads(trade.get("technicals_json") or "{}").get(key, default)
    except Exception:
        return default


def build_example(trade: dict) -> dict:
    # Use the exact prompt sent to the LLM (stored in DB) — ensures training prompt
    # matches data-gen prompt format exactly. Falls back for rows without prompt_text.
    stored_prompt = trade.get("prompt_text")
    if stored_prompt:
        user_prompt = stored_prompt
    else:
        tech_block = _parse_technicals(trade.get("technicals_json", "{}"))
        user_prompt = (
            f"=== MARKET DATA: {trade['symbol']} as of {trade['decision_date']} ===\n"
            f"{tech_block}\n\n"
            f"Decide. Output JSON only."
        )

    entry = trade["entry_price"]
    sl    = trade["stop_loss"]
    tp    = trade["take_profit"]
    if trade["action"] == "BUY":
        rr = round((tp - entry) / max(entry - sl, 0.01), 2)
    elif trade["action"] == "SELL":
        rr = round((entry - tp) / max(sl - entry, 0.01), 2)
    else:
        rr = 0.0

    confidence = _tech_field(trade, "confidence", None)
    try:
        confidence = round(float(confidence), 3)
    except Exception:
        try:
            confidence = round(float(trade.get("confidence", 0.7)), 3)
        except Exception:
            confidence = 0.7

    # Build the assistant response from what the LLM actually produced.
    # features_used and key_risk come from the LLM's stored reasoning — do not fabricate.
    features_used = _tech_field(trade, "features_used", None)
    if not isinstance(features_used, list) or not features_used:
        raw = trade.get("features_used")
        try:
            features_used = json.loads(raw) if raw else None
        except Exception:
            features_used = None

    key_risk = trade.get("key_risk") or None

    assistant_payload: dict = {
        "action":        trade["action"],
        "planned_entry": round(float(trade.get("planned_entry") or trade["entry_price"]), 2),
        "stop_loss":     sl,
        "take_profit":   tp,
        "risk_reward":   rr,
        "reasoning":     trade.get("reasoning", "—"),
        "confidence":    confidence,
    }
    if features_used:
        assistant_payload["features_used"] = features_used
    if key_risk:
        assistant_payload["key_risk"] = key_risk

    assistant_json = json.dumps(assistant_payload, indent=2)

    post_rsn = trade.get("post_reasoning")
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": user_prompt},
        {"role": "assistant", "content": assistant_json},
    ]
    if post_rsn:
        exit_reason = trade.get("exit_reason", "")
        exit_price  = trade.get("exit_price", 0) or 0
        pnl         = trade.get("pnl", 0) or 0
        messages.append({
            "role": "user",
            "content": f"Outcome: {exit_reason} @ {exit_price:.2f} ({pnl:+.0f}). Why?",
        })
        messages.append({"role": "assistant", "content": post_rsn})

    return {
        "messages": messages,
        "_meta": {
            "symbol":       trade["symbol"],
            "date":         trade["decision_date"],
            "action":       trade["action"],
            "sl":           sl,
            "tp":           tp,
            "pnl":          trade["pnl"],
            "exit_reason":  trade["exit_reason"],
            "confidence":   confidence,
            "rr":           rr,
        },
    }


# ── Quality filter ────────────────────────────────────────────────────

def quality_filter(example: dict, min_confidence: float, min_rr: float) -> bool:
    meta = example.get("_meta", {})
    # Exclude ENTRY_NOT_FILLED — no trade happened, nothing to learn from outcome
    if meta.get("exit_reason") == "ENTRY_NOT_FILLED":
        return False
    try:
        # First assistant turn is always messages[2]; last may be post_reasoning text
        d = json.loads(example["messages"][2]["content"])
    except Exception:
        return False
    if d.get("action") == "HOLD":
        return False
    if d.get("confidence", 1.0) < min_confidence:
        return False
    if d.get("risk_reward", 99) < min_rr:
        return False
    # Reject rows where the LLM returned raw indicator text in features_used
    # (items like "EMA: 50@109.13 > 100@108.14 > 200@107.41" instead of "ema_stack")
    fu = d.get("features_used")
    if isinstance(fu, list) and any(len(str(f)) > 25 for f in fu):
        return False
    return True


# ── Dedup + balance ───────────────────────────────────────────────────

def _dedup_key(example: dict) -> str:
    m = example.get("_meta", {})
    key = (f"{m.get('symbol')}|{m.get('date')}|{m.get('action')}"
           f"|{round(float(m.get('sl') or 0), 1)}|{round(float(m.get('tp') or 0), 1)}")
    return hashlib.sha1(key.encode()).hexdigest()[:20]


def _action(example: dict) -> str:
    try:
        return json.loads(example["messages"][2]["content"]).get("action", "")
    except Exception:
        return ""


def balance_and_dedup(examples: list[dict], dedup: bool = True,
                      balance: bool = True, max_ratio: float = 1.5,
                      seed: int = 42) -> list[dict]:
    random.seed(seed)

    if dedup:
        seen, unique = set(), []
        for e in examples:
            k = _dedup_key(e)
            if k not in seen:
                seen.add(k)
                unique.append(e)
        print(f"  After dedup:   {len(unique)} (removed {len(examples)-len(unique)} duplicates)")
        examples = unique

    if balance:
        buys  = [e for e in examples if _action(e) == "BUY"]
        sells = [e for e in examples if _action(e) == "SELL"]
        other = [e for e in examples if _action(e) not in ("BUY", "SELL")]
        if buys and sells:
            cap = int(min(len(buys), len(sells)) * max_ratio)
            random.shuffle(buys); random.shuffle(sells)
            examples = buys[:cap] + sells[:cap] + other
            random.shuffle(examples)
            print(f"  After balance: {len(examples)} "
                  f"(BUY {min(len(buys),cap)} / SELL {min(len(sells),cap)} / other {len(other)})")

    return examples


# ── Split + write ─────────────────────────────────────────────────────

def build_dataset(
    db_path: str = "trading_agent.db",
    out_path: str = "data/trading.jsonl",
    min_confidence: float = 0.5,
    min_rr: float = 1.5,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, int]:
    import os
    do_balance = os.environ.get("BALANCE_DATA", "1").strip().lower() in ("1","true","yes","on")
    do_dedup   = os.environ.get("DEDUP_DATA",   "1").strip().lower() in ("1","true","yes","on")
    split_mode = os.environ.get("SPLIT_MODE",   "time").strip().lower()

    trades = load_trades(db_path)
    print(f"Loaded {len(trades)} trades from {db_path}")

    examples = [build_example(t) for t in trades]
    print(f"Built {len(examples)} raw examples")

    filtered = [e for e in examples if quality_filter(e, min_confidence, min_rr)]
    print(f"After quality filter: {len(filtered)} examples")

    filtered = balance_and_dedup(filtered, dedup=do_dedup, balance=do_balance, seed=seed)

    if len(filtered) < 50:
        print("WARNING: < 50 examples — run more pipeline dates before training.")

    # Split
    n = len(filtered)
    n_test = max(1, int(n * test_ratio)) if n >= 3 else 0
    n_val  = max(1, int(n * val_ratio))  if n >= 3 else 0
    while n_test + n_val >= n and n_test + n_val > 0:
        if n_val > 0: n_val -= 1
        elif n_test > 0: n_test -= 1

    if split_mode == "time":
        filtered.sort(key=lambda e: e.get("_meta", {}).get("date") or "")
        splits = {
            "train": filtered[: n - n_val - n_test],
            "val":   filtered[n - n_val - n_test: n - n_test],
            "test":  filtered[n - n_test:],
        }
        # 5-day embargo: drop val/test examples within 5 calendar days of any train date
        from datetime import datetime as _dt
        train_dates = {e["_meta"]["date"] for e in splits["train"]}

        def _embargoed(date: str, gap: int = 5) -> bool:
            try:
                dt = _dt.strptime(date, "%Y-%m-%d")
                return any(abs((_dt.strptime(d, "%Y-%m-%d") - dt).days) <= gap
                           for d in train_dates)
            except Exception:
                return False

        before_embargo = len(splits["val"]) + len(splits["test"])
        splits["val"]  = [e for e in splits["val"]  if not _embargoed(e["_meta"]["date"])]
        splits["test"] = [e for e in splits["test"] if not _embargoed(e["_meta"]["date"])]
        removed = before_embargo - len(splits["val"]) - len(splits["test"])
        print(f"Split: time (walk-forward) | embargo removed {removed} val/test examples")
        if not splits["val"]:
            print("WARNING: val set empty after 5-day embargo — consider SPLIT_MODE=random")
        if not splits["test"]:
            print("WARNING: test set empty after 5-day embargo — consider SPLIT_MODE=random")
    else:
        random.seed(seed); random.shuffle(filtered)
        splits = {
            "train": filtered[n_test + n_val:],
            "val":   filtered[n_test: n_test + n_val],
            "test":  filtered[:n_test],
        }
        print("Split: random")

    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(out_path).stem

    counts = {}
    for split_name, split_data in splits.items():
        path      = out_dir / f"{stem}_{split_name}.jsonl"
        meta_path = out_dir / f"{stem}_{split_name}_meta.jsonl"
        with open(path, "w") as f, open(meta_path, "w") as mf:
            for ex in split_data:
                clean = {k: v for k, v in ex.items() if not k.startswith("_")}
                f.write(json.dumps(clean) + "\n")
                mf.write(json.dumps(ex.get("_meta", {})) + "\n")
        counts[split_name] = len(split_data)
        print(f"  {split_name:5s}: {len(split_data):4d} examples → {path}")

    return counts


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--db",             default="trading_agent.db")
    p.add_argument("--out",            default="data/trading.jsonl")
    p.add_argument("--min_confidence", type=float, default=0.5)
    p.add_argument("--min_rr",         type=float, default=1.5)
    p.add_argument("--val_ratio",      type=float, default=0.1)
    p.add_argument("--test_ratio",     type=float, default=0.1)
    args = p.parse_args()
    counts = build_dataset(
        db_path=args.db, out_path=args.out,
        min_confidence=args.min_confidence, min_rr=args.min_rr,
        val_ratio=args.val_ratio, test_ratio=args.test_ratio,
    )
    print(f"\nDone. {sum(counts.values())} total examples")
    print("Next: python train.py --train data/trading_train.jsonl --val data/trading_val.jsonl")
