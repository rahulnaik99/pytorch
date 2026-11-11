"""
evaluate.py
───────────
Evaluation harness for the fine-tuned trading model. Two layers:

  A) MODEL-QUALITY (no market replay) — on the held-out TEST jsonl:
       - json_validity   : fraction that parse as JSON
       - schema_ok       : fraction with all required keys
       - reasoning_ok    : fraction with non-empty reasoning
       - features_used_ok: fraction that include a non-empty features_used list
       - action_match    : agreement with the reference (teacher) action

  B) TRADING METRICS (pure math) — from realized P&L of a set of trades:
       - win_rate, avg_win, avg_loss, expectancy, profit_factor,
         sharpe (per-trade), max_drawdown_pct

The TEST set is produced by the walk-forward split (SPLIT_MODE=time), so these
numbers reflect performance on dates the model never trained on.

Usage:
  # quick self-check of the harness (uses the stored answers as the "model")
  python evaluate.py --test data/rounds/round_01/trading_test.jsonl

  # evaluate a fine-tuned MLX adapter/fused model
  python evaluate.py --test data/rounds/round_01/trading_test.jsonl \
      --model output/adapters/round_01/fused_model

  # trading metrics from the test set's recorded outcomes
  python evaluate.py --test data/rounds/round_01/trading_test.jsonl --pnl-from-meta
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env", override=False)
except ImportError:
    pass

REQUIRED_KEYS = ["action", "stop_loss", "take_profit", "risk_reward", "reasoning", "confidence"]


# ── parsing helpers ──────────────────────────────────────────────────────────
def _parse_json_block(text: str) -> dict:
    if not text:
        return {}
    import re
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


def _messages(ex):
    sys_c = next((m["content"] for m in ex["messages"] if m["role"] == "system"), "")
    usr_c = next((m["content"] for m in ex["messages"] if m["role"] == "user"), "")
    asst  = next((m["content"] for m in ex["messages"] if m["role"] == "assistant"), "")
    return sys_c, usr_c, asst


# ── A) model-quality ─────────────────────────────────────────────────────────
def evaluate_model_quality(test_path: str, generate_fn) -> dict:
    """
    generate_fn(system, user) -> model's raw text answer.
    For a self-check, pass a function that returns the stored assistant text.
    """
    rows = [json.loads(l) for l in open(test_path) if l.strip()]
    n = len(rows)
    jv = sch = rz = fu = am = 0
    n_decision = 0            # only PRE-TRADE (decision) rows have action/SL/TP
    for ex in rows:
        sys_c, usr_c, ref_txt = _messages(ex)
        pred = generate_fn(sys_c, usr_c)
        d = _parse_json_block(pred)
        ref = _parse_json_block(ref_txt)
        if d: jv += 1
        if str(d.get("reasoning") or "").strip() not in ("", "—"): rz += 1
        if isinstance(d.get("features_used"), list) and d["features_used"]: fu += 1
        # Schema + action-match only make sense for decision rows (post-trade
        # "lesson" rows have no action/SL/TP and must not pollute these scores).
        if ref.get("action"):
            n_decision += 1
            if d and all(k in d for k in REQUIRED_KEYS): sch += 1
            if d.get("action") and d.get("action") == ref.get("action"): am += 1
    pct = lambda x, base=n: round(100 * x / base, 1) if base else 0.0
    return {"n": n, "n_decision": n_decision,
            "json_validity": pct(jv), "reasoning_ok": pct(rz),
            "features_used_ok": pct(fu),
            "schema_ok": pct(sch, n_decision),
            "action_match": pct(am, n_decision)}


# ── B) trading metrics ───────────────────────────────────────────────────────
def trading_metrics(pnls: list[float]) -> dict:
    pnls = [float(p) for p in pnls if p is not None]
    n = len(pnls)
    if n == 0:
        return {"n": 0}
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_win = sum(wins)
    gross_loss = -sum(losses)
    mean = sum(pnls) / n
    var = sum((p - mean) ** 2 for p in pnls) / n
    std = math.sqrt(var)
    sharpe = round(mean / std, 2) if std > 0 else 0.0   # per-trade Sharpe

    # Sortino: like Sharpe but only penalises DOWNSIDE volatility (industry
    # standard for asymmetric return streams — losses hurt, upside doesn't).
    downside = [p for p in pnls if p < 0]
    dvar = sum(p * p for p in downside) / n if downside else 0.0
    dstd = math.sqrt(dvar)
    sortino = round(mean / dstd, 2) if dstd > 0 else 0.0

    # Statistical significance of the edge: t-stat of mean P&L > 0.
    # |t| >= ~2 ≈ 95% confidence the positive expectancy is not luck.
    t_stat = round(mean / (std / math.sqrt(n)), 2) if std > 0 else 0.0

    # max drawdown on the cumulative P&L curve
    cum, peak, mdd = 0.0, 0.0, 0.0
    for p in pnls:
        cum += p
        peak = max(peak, cum)
        mdd = min(mdd, cum - peak)

    return {
        "n": n,
        "win_rate": round(100 * len(wins) / n, 1),
        "avg_win": round(gross_win / len(wins), 2) if wins else 0.0,
        "avg_loss": round(-gross_loss / len(losses), 2) if losses else 0.0,
        "expectancy": round(mean, 2),                      # avg P&L per trade
        # JSON-safe: float('inf') makes FastAPI's JSON encoder 500 on
        # /evaluation/metrics. Use None ("undefined — no losing trades") instead.
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else None,
        "sharpe_per_trade": sharpe,
        "sortino_per_trade": sortino,
        "t_stat": t_stat,                                  # edge significance
        "significant_95": abs(t_stat) >= 2.0,
        "total_pnl": round(sum(pnls), 2),
        "max_drawdown": round(mdd, 2),
    }


def multi_fold_metrics(pnls: list[float], k: int = 3) -> dict:
    """Walk-forward robustness: split the P&L stream into K chronological folds
    and report per-fold metrics + the stability (mean/std) of the edge across
    folds. A strategy that only works in one period shows high variance here —
    the standard guard against overfitting to a single stretch of history.
    """
    pnls = [float(p) for p in pnls if p is not None]
    n = len(pnls)
    if n < k or k < 2:
        return {"folds": k, "note": "not enough trades for multi-fold", "per_fold": []}
    size = n // k
    folds = []
    for i in range(k):
        lo = i * size
        hi = n if i == k - 1 else (i + 1) * size
        folds.append(trading_metrics(pnls[lo:hi]))
    exps = [f.get("expectancy", 0.0) for f in folds]
    mean_exp = sum(exps) / k
    var_exp = sum((e - mean_exp) ** 2 for e in exps) / k
    return {
        "folds": k,
        "per_fold": folds,
        "expectancy_mean": round(mean_exp, 2),
        "expectancy_std": round(math.sqrt(var_exp), 2),
        "profitable_folds": sum(1 for e in exps if e > 0),
        "consistent": all(e > 0 for e in exps),   # edge holds in every period?
    }


def pnls_from_test_meta(test_path: str) -> list[float]:
    """Realized P&L recorded in each test example's _meta (pre_trade rows only).

    The split step (data_pipeline.build_dataset) strips ``_meta`` from the
    training JSONL and writes it to a ``*_meta.jsonl`` sidecar instead. We read
    inline ``_meta`` if present (backwards compatible) and otherwise fall back to
    the sidecar, so walk-forward trading metrics always have data to work with.
    """
    out = []
    saw_inline_meta = False
    for l in open(test_path):
        if not l.strip():
            continue
        ex = json.loads(l)
        meta = ex.get("_meta")
        if meta is not None:
            saw_inline_meta = True
            # Accept rows regardless of whether "type" field exists —
            # our splitter.py does not set "type", just check pnl directly.
            if meta.get("pnl") is not None:
                out.append(meta["pnl"])

    if out or saw_inline_meta:
        return out

    # Fall back to the sidecar written alongside the split.
    sidecar = Path(test_path).with_name(Path(test_path).stem + "_meta.jsonl")
    if sidecar.exists():
        for l in open(sidecar):
            if not l.strip():
                continue
            meta = json.loads(l)
            if meta.get("pnl") is not None:
                out.append(meta["pnl"])
    return out


# ── generate functions ───────────────────────────────────────────────────────
def stored_answer_generate(system, user):
    """Self-check: the 'model' just echoes the reference — harness sanity test."""
    return None   # replaced per-row below; see _selfcheck


def hf_generate_factory(model_path: str, adapter_path: str | None = None):
    """Return a generate_fn backed by a HuggingFace fused model or adapter."""
    from inference import hf_generate_factory as _factory
    return _factory(model_path, adapter_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test", required=True, help="held-out test JSONL")
    p.add_argument("--model", default=None, help="fused model or adapter path")
    p.add_argument("--pnl-from-meta", action="store_true",
                   help="also report trading metrics from the test set's recorded P&L")
    args = p.parse_args()

    if args.model:
        gen = hf_generate_factory(args.model)
        print(f"Evaluating model: {args.model}")
    else:
        # self-check: echo the stored assistant so the harness reports ~100%
        rows = [json.loads(l) for l in open(args.test) if l.strip()]
        answers = {}
        for ex in rows:
            _, usr, asst = _messages(ex)
            answers[usr] = asst
        gen = lambda s, u: answers.get(u, "")
        print("No --model given → harness self-check (echoes stored answers)")

    q = evaluate_model_quality(args.test, gen)
    print("\nMODEL QUALITY (held-out test):")
    for k, v in q.items():
        print(f"  {k:18s}: {v}")

    if args.pnl_from_meta:
        tm = trading_metrics(pnls_from_test_meta(args.test))
        print("\nTRADING METRICS (recorded outcomes on test dates):")
        for k, v in tm.items():
            print(f"  {k:18s}: {v}")

    out = {"model_quality": q}
    if args.pnl_from_meta:
        out["trading_metrics"] = trading_metrics(pnls_from_test_meta(args.test))
    Path("metrics").mkdir(exist_ok=True)
    Path("metrics/eval_results.json").write_text(json.dumps(out, indent=2))
    print("\nWrote metrics/eval_results.json")


if __name__ == "__main__":
    main()
