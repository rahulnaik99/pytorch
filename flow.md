# Operational Flow

Step-by-step guide from raw price data to a deployed fine-tuned model.

---

## Step 1 — Generate Training Data

```bash
SYMBOL=CANBK START_DATE=2024-01-01 END_DATE=2024-06-30 python pipeline.py
```

**What happens for each date:**
1. Checks if `(CANBK, 2024-01-10)` already exists in `trading_agent.db` → skips if yes
2. Loads last 90 days of daily candles
3. Computes EMA 50/100/200, VWAP, ATR, HH/HL/LH/LL structure, PDH/PDL, 2DH/3DH breaks, support/resistance zones, 5min + 1h demand/supply zones, VIX regime
4. Builds a compact prompt and sends to Ollama (Nemotron Bolt)
5. LLM returns: `action, stop_loss, take_profit, reasoning, confidence, features_used`
6. Loads next 5 forward candles and simulates: `SL_HIT / TP_HIT / TIME_EXIT`
7. Calculates net P&L including Indian transaction costs (brokerage, STT, GST, stamp)
8. Stores row in `trading_agent.db`

**Output printed per date:**
```
CANBK 2024-01-10 | BUY SL=97.50 TP=104.20 → TP_HIT 104.20 [PROFIT +312]
```

---

## Where is generated data stored?

All generated data lives in **`trading_agent.db`** (SQLite, single file).

```bash
# Count rows
python -c "import sqlite3; c=sqlite3.connect('trading_agent.db'); print(c.execute('SELECT count(*) FROM trade_log').fetchone())"

# See last 10 rows
python -c "
import sqlite3, pandas as pd
df = pd.read_sql('SELECT symbol, decision_date, action, exit_reason, pnl, confidence FROM trade_log ORDER BY decision_date DESC LIMIT 10', sqlite3.connect('trading_agent.db'))
print(df.to_string())
"
```

---

## How to run on multiple days without creating duplicates

**No manual merging is needed.** The DB is cumulative.

Run `pipeline.py` on Monday → 100 rows stored.  
Run again on Tuesday for more dates → new rows added, existing dates skipped.

Two-level protection against duplicates:

```
Level 1 (before LLM):
  _already_exists(symbol, date)
  → if True: print "[skip] already in DB", return None
  → LLM is never called for already-processed dates

Level 2 (DB constraint):
  INSERT OR IGNORE INTO trade_log ...  (UNIQUE symbol+decision_date)
  → even if Level 1 somehow missed it, the DB rejects the duplicate silently
```

**Example — running across multiple sessions:**
```bash
# Session 1 (Monday)
START_DATE=2024-01-01 END_DATE=2024-03-31 python pipeline.py
# → 63 rows stored

# Session 2 (Tuesday — extend the range)
START_DATE=2024-01-01 END_DATE=2024-06-30 python pipeline.py
# → first 63 dates: "[skip] already in DB"
# → new dates (Apr-Jun): processed and stored
# → 126 rows total, zero duplicates
```

---

## Step 2 — Split to JSONL

```bash
python splitter.py
# or with custom thresholds:
python splitter.py --min_confidence 0.5 --min_rr 1.5 --val_ratio 0.1 --test_ratio 0.1
```

**What it does:**
1. Reads all non-HOLD rows from `trade_log`
2. Quality filter: drops rows where `confidence < 0.5` or `risk_reward < 1.5`
3. SHA-1 dedup: removes near-identical responses
4. BUY/SELL balance: caps majority class at `1.5× minority` to avoid bias
5. Walk-forward split (chronological — no data leakage):
   - `data/trading_train.jsonl` — oldest 80%
   - `data/trading_val.jsonl`   — middle 10%
   - `data/trading_test.jsonl`  — newest 10% (never seen during training)

Each split also writes a `*_meta.jsonl` sidecar with P&L and outcome data.

---

## Step 3 — Train the LoRA Adapter

```bash
# Smoke-test (1 epoch, ~10 min on M5)
EPOCHS=1 python train.py \
    --train data/trading_train.jsonl \
    --val   data/trading_val.jsonl

# Full training (3 epochs, overnight on M5)
EPOCHS=3 python train.py \
    --train data/trading_train.jsonl \
    --val   data/trading_val.jsonl
```

**What gets saved:**
```
output/nemotron-lora/
  adapter_model.safetensors   ← LoRA weights only (~20 MB)
  adapter_config.json
  tokenizer files
  checkpoint-*/               ← intermediate checkpoints
```

**Track loss in real time:**
```bash
mlflow ui   # → http://localhost:5000
# shows: train/loss, eval/loss, grad_norm per step
# compare multiple runs side by side
```

---

## Step 4 — Fuse Adapter into Base Model

The adapter is a small delta on top of the base model. For deployment, merge them:

```bash
python train.py --merge \
    --base_model nvidia/Nemotron-Mini-4B-Instruct \
    --adapter    output/nemotron-lora \
    --fused_dir  output/nemotron-lora/fused_model
```

**What happens:**
```
Base model weights (float16)
  + LoRA adapter weights (small delta matrices)
  → merge_and_unload()
  → single fused model saved to output/nemotron-lora/fused_model/
     model.safetensors  (~8 GB)
     tokenizer.json
     config.json
```

The fused model behaves identically to the base model + adapter but loads faster (no adapter overhead at inference time).

---

## Step 5 — Deploy for Local Inference

### Option A: serve.py (REST endpoint)

```bash
HF_MODEL=output/nemotron-lora/fused_model python serve.py
# → http://localhost:8000
```

Test it:
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a disciplined NSE equity trader. Output JSON only."},
      {"role": "user",   "content": "[CANBK 2024-10-15]\nC:103.5 | EMA50:101.2 EMA200:96.8\nDecide."}
    ],
    "stream": false
  }'
```

Use fine-tuned model for new data generation (swap it in for Ollama):
```bash
HF_MODEL=output/nemotron-lora/fused_model \
OLLAMA_URL=http://localhost:8000/api/chat \
python pipeline.py
```

### Option B: Direct Python

```python
from inference import generate_hf

response = generate_hf(
    model_path="output/nemotron-lora/fused_model",
    system="You are a disciplined NSE equity trader. Output JSON only.",
    user="[CANBK 2024-10-15]\nC:103.5 | EMA50:101.2 ...\nDecide.",
)
print(response)
# → {"action": "BUY", "stop_loss": 100.5, "take_profit": 109.0, ...}
```

---

## Step 6 — Evaluate Quality & Performance

### Model quality (does the model output correct JSON with good reasoning?)

```bash
python evaluate.py \
    --test  data/trading_test.jsonl \
    --model output/nemotron-lora/fused_model
```

Output:
```
MODEL QUALITY (held-out test):
  n                   : 42
  json_validity       : 97.6    # % valid JSON responses
  schema_ok           : 95.2    # % with all required keys
  action_match        : 71.4    # % model agrees with reference action
  reasoning_ok        : 97.6    # % with non-empty reasoning
  features_used_ok    : 92.9    # % that cite specific indicator values
```

### Trading metrics (does the strategy actually make money?)

```bash
python evaluate.py \
    --test data/trading_test.jsonl \
    --model output/nemotron-lora/fused_model \
    --pnl-from-meta
```

Additional output:
```
TRADING METRICS (recorded P&L on test dates):
  win_rate              : 58.3    # % profitable trades
  expectancy            : 148.5   # average P&L per trade (₹)
  profit_factor         : 1.72    # gross wins / gross losses
  sharpe_per_trade      : 0.41
  t_stat                : 2.31    # ≥ 2.0 = statistically significant ✓
  significant_95        : True
  max_drawdown          : -842.0  # worst peak-to-trough (₹)
  total_pnl             : 6234.5
```

Results saved to `metrics/eval_results.json`.

### Self-check (no model needed — tests the harness itself)

```bash
python evaluate.py --test data/trading_test.jsonl
# echoes stored answers → expect ~100% on all metrics
```

---

## Multi-round Training (flywheel)

Once the fine-tuned model is better than the base model, use it to generate the next round of data:

```bash
# Archive current round
cp -r data/trading_*.jsonl data/rounds/round_01/

# Use fine-tuned model to generate better training data
HF_MODEL=output/nemotron-lora/fused_model \
OLLAMA_URL=http://localhost:8000/api/chat \
START_DATE=2023-01-01 END_DATE=2023-12-31 \
python pipeline.py

# Split + train again
python splitter.py
EPOCHS=3 OUTPUT_DIR=output/nemotron-lora-round2 python train.py \
    --train data/trading_train.jsonl --val data/trading_val.jsonl

# Evaluate improvement
python evaluate.py --test data/trading_test.jsonl \
    --model output/nemotron-lora-round2/fused_model --pnl-from-meta
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `CANBK.csv` not found | Set `DATA_SOURCE=yfinance` or place CSV in `data/prices/` |
| VIX data missing | Zones still computed without VIX context; run with `DATA_SOURCE=yfinance` |
| Ollama connection refused | Start Ollama: `ollama serve`, pull model: `ollama pull hf.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB:Q4_K_M` |
| MPS out of memory | Reduce `MAX_SEQ_LEN=512` or `GRAD_ACCUM=4` |
| `< 50 examples` warning | Generate more dates or lower `--min_confidence 0.4 --min_rr 1.2` |
| Model outputs plain text not JSON | Confidence may be low — re-run data gen with a better model |
