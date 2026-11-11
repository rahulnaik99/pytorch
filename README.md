# Trading LLM Fine-Tune Pipeline

Fine-tune a 4B LLM (Nemotron Bolt) to make NSE equity trading decisions using technical analysis.

**Stack**: PyTorch + HuggingFace PEFT LoRA + trl SFTTrainer + MLflow  
**Hardware**: MacBook M5 (Apple Silicon MPS) — also works on NVIDIA CUDA

---

## How it works

```
1. pipeline.py   — fetch 90-day candles + VIX → compute technicals → LLM trade → SQLite
2. splitter.py   — SQLite → quality-filtered JSONL (train / val / test)
3. train.py      — LoRA fine-tune the model on that JSONL
4. evaluate.py   — measure model quality + trading metrics on held-out test set
5. serve.py      — local REST endpoint to test the fine-tuned model
```

---

## Quick start

```bash
pip install -r requirements.txt

# Data generation
SYMBOL=CANBK START_DATE=2024-01-01 END_DATE=2024-06-30 python pipeline.py

# Parallel date generation (4 workers — speeds up data loading; LLM calls still queue in Ollama)
SYMBOL=CANBK COUNT=500 WORKERS=4 python pipeline.py

# 2. Split to JSONL
python splitter.py

# 3. Train (smoke-test: EPOCHS=1, full run: EPOCHS=3)
EPOCHS=1 python train.py --train data/trading_train.jsonl --val data/trading_val.jsonl

# 4. View loss in browser
mlflow ui   # → http://localhost:5000

# 5. Merge adapter → fused model
python train.py --merge --adapter output/nemotron-lora

# 6. Evaluate
python evaluate.py --test data/trading_test.jsonl \
    --model output/nemotron-lora/fused_model --pnl-from-meta

# 7. Serve + test
HF_MODEL=output/nemotron-lora/fused_model python serve.py
```

---

## Files

| File | Purpose |
|------|---------|
| `pipeline.py` | Data generation loop — one SQLite row per trading date |
| `technicals.py` | `TechnicalsAnalyzer` — EMA, VWAP, S/R zones, VIX, structure |
| `splitter.py` | SQLite → JSONL with quality filter, dedup, walk-forward split |
| `train.py` | LoRA training + adapter merge |
| `inference.py` | HuggingFace model inference with lazy model cache |
| `evaluate.py` | Model quality metrics + trading P&L metrics |
| `serve.py` | Ollama-compatible local REST server for the fine-tuned model |
| `explore.ipynb` | Notebook: inspect candles, prompts, DB, train, evaluate |

---

## Data

| Path | Contents |
|------|----------|
| `data/prices/CANBK.csv` | Minute-level OHLCV — `date, open, high, low, close, volume` |
| `data/prices/INDIAVIX.csv` | VIX intraday — `Ticker, Date/Time, ..., Open, High, Low, Close` |
| `trading_agent.db` | Generated trade decisions (SQLite, cumulative, dedup enforced) |
| `data/trading_*.jsonl` | Train / val / test splits |
| `data/rounds/round_01/` | Archived first-round splits |

---

## Configuration (env vars)

```bash
# Data generation
SYMBOL=CANBK                          # NSE symbol
START_DATE=2024-01-01
END_DATE=2024-12-31
LOOKBACK_DAYS=90                      # candle lookback
OLLAMA_MODEL=hf.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB:Q4_K_M
DATA_SOURCE=csv                       # "csv" or "yfinance"

# Training
BASE_MODEL=nvidia/Nemotron-Mini-4B-Instruct
OUTPUT_DIR=output/nemotron-lora
EPOCHS=3
LORA_R=16

# Inference (after training)
HF_MODEL=output/nemotron-lora/fused_model
```

---

## See also

- [architecture.md](architecture.md) — system components and design decisions
- [flow.md](flow.md) — step-by-step operational guide (data → train → deploy → evaluate)
