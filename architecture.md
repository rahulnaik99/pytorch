# Architecture

## Overview

Seven flat Python files. No packages, no services, no API layers.

```
pipeline.py    technicals.py    splitter.py
    ↓         ↑    prompts.py ←──┘
SQLite DB ←───┘                  ↓
                            JSONL files
                                 ↓
                           train.py  ←→  MLflow
                                 ↓
                           inference.py
                                ↑    ↓
                          serve.py  evaluate.py
```

---

## Components

### `prompts.py` — Canonical Prompts

Single source of truth for the system prompt and prompt builder.

```python
SYSTEM_PROMPT          # shared by pipeline, splitter, train, serve
build_prompt(symbol, date, tech, vc, zc, decision_slot)  → (system, user) tuple
```

All modules import from here — no local copies.

---

### `technicals.py` — TechnicalsAnalyzer

Single class. Loads price data, computes all indicators. No side effects.

```
TechnicalsAnalyzer
  ├── load_candles(symbol, date, lookback=90)  → daily DataFrame
  ├── load_forward(symbol, date, n=5)          → next N days of minute bars
  ├── load_intraday(symbol, date, rule, n)     → 5min or 1h DataFrame
  ├── _load_vix()                              → preload VIX CSV (call before forking threads)
  ├── compute(df, decision_slot)               → indicator dict
  ├── vix_context(date)                        → VIX regime dict
  └── compute_zones(symbol, date)             → 5min + 1h demand/supply zones
```

**Indicators computed:**
- EMA 50 / 100 / 200 + stack (bullish/bearish/mixed)
- VWAP (90-bar cumulative, reset each day), average range (5-day)
- Trend: uptrend / downtrend / ranging
- Structure: HH, HL, LH, LL (swing pivot classification)
- Levels: PDH/PDL, 2DH/2DL (broken/intact), 3DH/3DL (broken/intact), week H/L
- Support / resistance zones (clustered pivot levels)
- Demand / supply zones on 5-min and 1-hour timeframes
- VIX regime: LOW / NORMAL / ELEVATED / EXTREME

**Look-ahead guard:** For `open` and `mid` slots, `compute()` uses `ind_df = df.iloc[:-1]`
so today's full OHLCV bar is never used for indicator computation.

**CSV formats supported (both auto-detected):**

| File | Format |
|------|--------|
| `CANBK.csv` | `date, open, high, low, close, volume` — ISO datetime |
| `INDIAVIX.csv` | `Ticker, Date/Time, ..., Open, High, Low, Close` — DD-MM-YYYY |

---

### `pipeline.py` — Data Generation

Stateless loop. Each date is independent — no capital tracking, no history.

```
For each trading date:
  1. _already_exists(symbol, date, slot)  → skip if in DB  (no LLM call)
  2. load_candles(90 days)
  3. compute(slot) + vix_context() + compute_zones()
  4. build_prompt()                 → compact feature string
  5. _llm_generate()                → Ollama (data-gen) or HF model (fine-tuned)
  6. simulate_outcome()             → SL_HIT / TP_HIT / TIME_EXIT (minute-bar resolution)
  7. _store()  →  INSERT OR IGNORE  (UNIQUE symbol+date+slot enforced at DB level)
```

**Deduplication is at two levels:**
1. `_already_exists()` — checks DB before calling LLM (saves cost + time)
2. `INSERT OR IGNORE` + `UNIQUE(symbol, decision_date, decision_slot)` — DB-level safety net

**Parallel mode:** With `WORKERS > 1`, a single `TechnicalsAnalyzer` is created and all
symbol CSVs + VIX are preloaded in the main thread. Workers only read the shared cache
(CPython GIL makes concurrent dict reads safe). This avoids the ~36 s CSV parse per thread.

**LLM backend priority:**
```
HF_MODEL set?  →  generate_hf()  (fine-tuned model)
otherwise      →  Ollama HTTP    (Nemotron Bolt for data generation)
```

---

### `splitter.py` — JSONL Builder

Reads `trade_log` from SQLite → applies quality filters → walk-forward split.

```
trade_log rows
  ↓ quality_filter(min_confidence=0.5, min_rr=1.5)
  ↓ balance_and_dedup (SHA-1 dedup, BUY/SELL balance ≤ 1.5× ratio)
  ↓ walk-forward split (chronological) + 5-day embargo
  → data/trading_train.jsonl  +  _meta.jsonl
  → data/trading_val.jsonl    +  _meta.jsonl
  → data/trading_test.jsonl   +  _meta.jsonl
```

**Output format** (HuggingFace-compatible):
```json
{
  "messages": [
    {"role": "system",    "content": "You are a disciplined NSE equity trader..."},
    {"role": "user",      "content": "=== MARKET DATA: CANBK as of 2024-03-15 ===\n..."},
    {"role": "assistant", "content": "{\"action\": \"BUY\", \"stop_loss\": 118.5, ...}"}
  ]
}
```

---

### `train.py` — LoRA Fine-Tuning

```
AutoModelForCausalLM (float16, device_map=mps/cuda/cpu)
  + LoraConfig(r=16, alpha=32, target=q/k/v/o_proj)
  + SFTTrainer(messages column, completion_only_loss=True, max_length=1024)
  + MLflowCallback → mlruns/ (loss per step, params, adapter artifact)
```

**Two modes:**
- `python train.py --train ... --val ...` → trains, saves adapter
- `python train.py --merge --adapter output/nemotron-lora` → fuses into base model

**Token length check:** Before training, logs the p50/p95/max token counts for a sample
of the training set and warns if p95 exceeds 90% of `MAX_SEQ_LEN`. Raise `MAX_SEQ_LEN`
if examples are being truncated.

**Why no bitsandbytes (QLoRA)?**  
`bitsandbytes` is CUDA-only. On MPS we use full float16 LoRA instead.  
Effective batch = `BATCH_SIZE(1) × GRAD_ACCUM(8)` = 8.

---

### `inference.py` — Model Inference

Lazy-loads and caches the model on first call. Singleton per `(model_path, adapter_path)`.
Two locks: `_cache_lock` (model load) and `_gen_lock` (GPU forward pass — not thread-safe on MPS/CUDA).

```python
generate_hf(model_path, system, user, adapter_path=None)
  → apply_chat_template → model.generate → decode → return str
```

**Also exports:**
```python
hf_generate_factory(model_path)  →  generate_fn(system, user)
# Used by evaluate.py for batch evaluation
```

---

### `evaluate.py` — Evaluation

Two independent metric groups:

**A) Model quality** (requires test JSONL + model):
| Metric | What it measures |
|--------|-----------------|
| `json_validity` | % responses that parse as valid JSON |
| `schema_ok` | % with all required keys (action/SL/TP/RR/reasoning/confidence) |
| `action_match` | % where model action agrees with reference |
| `reasoning_ok` | % with non-empty reasoning |

**B) Trading metrics** (from `_meta.jsonl` sidecar — no model needed):
| Metric | What it measures |
|--------|-----------------|
| `win_rate` | % trades with positive P&L |
| `expectancy` | Average P&L per trade (₹) |
| `profit_factor` | Gross win ÷ gross loss |
| `sharpe_per_trade` | mean(pnl) / std(pnl) |
| `max_drawdown` | Largest peak-to-trough on cumulative P&L |
| `t_stat` | Statistical significance of edge (\|t\| ≥ 2 → 95% confidence) |

---

### `serve.py` — Local Deployment

Ollama-compatible REST endpoint. Drop-in replacement for Ollama in `pipeline.py`.
Binds to `127.0.0.1` by default (`SERVE_HOST` env var to override).

```
POST http://localhost:8000/api/chat
  ← {"messages": [...], "stream": false}
  → {"message": {"role": "assistant", "content": "..."}, "done": true}
```

---

## Database Schema

```sql
CREATE TABLE trade_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT,
    decision_date   TEXT,
    action          TEXT,            -- BUY / SELL / HOLD
    entry_price     REAL,
    stop_loss       REAL,
    take_profit     REAL,
    exit_price      REAL,
    exit_date       TEXT,
    exit_reason     TEXT,            -- SL_HIT / TP_HIT / TIME_EXIT / HOLD
    pnl             REAL,            -- net P&L after Indian transaction costs
    reasoning       TEXT,
    confidence      REAL,
    features_used   TEXT,            -- JSON array
    technicals_json TEXT,            -- full indicator snapshot as JSON
    prompt_text     TEXT,            -- exact prompt sent to LLM (used by splitter)
    decision_slot   TEXT DEFAULT 'close',   -- open / mid / close
    planned_entry   REAL,
    post_reasoning  TEXT,
    is_mock         INTEGER DEFAULT 0,
    key_risk        TEXT,
    created_at      TEXT,
    UNIQUE(symbol, decision_date, decision_slot)   -- 3-column dedup
);
```

---

## MLflow Tracking

Every training run logs to `mlruns/` (local filesystem, default):

```
mlruns/
  trading-nemotron-lora/
    run_<id>/
      params:    model, lora_r, lora_alpha, epochs, lr, device, train_size
      metrics:   loss, eval_loss, grad_norm (per step)
      artifacts: output/nemotron-lora/ (adapter weights)
```

View: `mlflow ui` → http://localhost:5000  
Override: `MLFLOW_URI=sqlite:///mlflow.db` to use a SQLite backend.

---

## Design Decisions

| Decision | Reason |
|----------|--------|
| Stateless data generation | Each date is independent → cleaner training signal, resumable, parallelizable |
| No fundamentals | Quarterly data repeated 60+ times per year — no signal for 1-5 day holds |
| 90-day lookback | More context for EMA-200 and multi-month structure detection |
| 5-min zones (not 15-min) | Tighter intraday precision for short-term swing trades |
| PyTorch + HF (not MLX) | Universal stack — works on any hardware, appears in job descriptions |
| Walk-forward split + 5-day embargo | Preserves time order and prevents leakage from adjacent dates |
| Float16 LoRA on MPS | bitsandbytes is CUDA-only; float16 LoRA works fine on M-series |
| messages column for SFTTrainer | Preserves structure so trl can mask non-assistant tokens for completion-only loss |
