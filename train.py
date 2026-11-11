"""
train.py
────────
LoRA fine-tuning for the trading LLM.

Stack: HuggingFace Transformers + PEFT + trl SFTTrainer + MLflow
Hardware: Apple Silicon (MPS), NVIDIA (CUDA), or CPU — auto-detected

Usage:
    # Train
    python train.py --train data/trading_train.jsonl --val data/trading_val.jsonl

    # Train with custom model / output
    BASE_MODEL=nvidia/Nemotron-Mini-4B-Instruct OUTPUT_DIR=output/run1 python train.py ...

    # Merge adapter into base model (after training)
    python train.py --merge \
        --base_model nvidia/Nemotron-Mini-4B-Instruct \
        --adapter output/nemotron-lora \
        --fused_dir output/nemotron-lora/fused_model

    # Inference test
    python train.py --infer --model output/nemotron-lora/fused_model
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env", override=False)
except ImportError:
    pass

import torch

# ── Config (env vars) ─────────────────────────────────────────────────────────
BASE_MODEL  = os.environ.get("BASE_MODEL",   "nvidia/Nemotron-Mini-4B-Instruct")
OUTPUT_DIR  = os.environ.get("OUTPUT_DIR",   "output/nemotron-lora")
TRAIN_PATH  = os.environ.get("TRAIN_JSONL",  "data/trading_train.jsonl")
VAL_PATH    = os.environ.get("VAL_JSONL",    "data/trading_val.jsonl")
LORA_R      = int(os.environ.get("LORA_R",       16))
LORA_ALPHA  = int(os.environ.get("LORA_ALPHA",   32))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", 0.05))
EPOCHS      = int(os.environ.get("EPOCHS",        3))
LR          = float(os.environ.get("LR",           2e-4))
BATCH_SIZE  = int(os.environ.get("BATCH_SIZE",     1))
GRAD_ACCUM  = int(os.environ.get("GRAD_ACCUM",     8))
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN",    1024))
MAX_STEPS   = int(os.environ.get("MAX_STEPS",      -1))   # -1 = full epoch
LOGGING_STEPS = int(os.environ.get("LOGGING_STEPS",  10))
EVAL_STEPS    = int(os.environ.get("EVAL_STEPS",     50))
SAVE_STEPS    = int(os.environ.get("SAVE_STEPS",    100))
MLFLOW_URI  = os.environ.get("MLFLOW_URI",   "mlruns")
EXPERIMENT  = os.environ.get("MLFLOW_EXPERIMENT", "trading-nemotron-lora")
# When 1, only compute loss on assistant tokens (ignores system+user tokens).
# This focuses the gradient signal on the model's own output — strongly
# recommended for instruction-following fine-tunes.
RESPONSE_ONLY_LOSS = os.environ.get("RESPONSE_ONLY_LOSS", "1").strip().lower() in ("1","true","yes","on")


# ── Device ────────────────────────────────────────────────────────────────────

def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ── Model loading ─────────────────────────────────────────────────────────────

def load_base_model(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = get_device()
    print(f"Loading {model_name} → {device}")
    dtype = torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map={"": device},
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def apply_lora(model):
    from peft import LoraConfig, TaskType, get_peft_model

    # Default targets for Llama/Nemotron; override via TARGET_MODULES env var.
    # "all-linear" works across any architecture (PEFT >= 0.7).
    default_targets = os.environ.get(
        "TARGET_MODULES", "q_proj,k_proj,v_proj,o_proj"
    ).strip()
    if default_targets == "all-linear":
        target_modules = "all-linear"
    else:
        specified = [t.strip() for t in default_targets.split(",") if t.strip()]
        # Check if any specified module exists; fall back to all-linear if none match
        model_modules = {name.split(".")[-1] for name, _ in model.named_modules()}
        target_modules = specified if any(t in model_modules for t in specified) else "all-linear"
        if target_modules == "all-linear":
            print(f"[lora] target modules {specified} not found → using all-linear")

    cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=target_modules,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model


# ── Dataset ───────────────────────────────────────────────────────────────────

def load_chat_dataset(path: str, tokenizer):
    from datasets import load_dataset

    # Minimal fallback template for models without a built-in chat template (e.g. gpt2)
    _FALLBACK_TMPL = (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}### System:\n{{ message['content'] }}\n{% endif %}"
        "{% if message['role'] == 'user' %}### User:\n{{ message['content'] }}\n{% endif %}"
        "{% if message['role'] == 'assistant' %}### Assistant:\n{{ message['content'] }}\n{% endif %}"
        "{% endfor %}"
    )

    def _format(ex):
        try:
            text = tokenizer.apply_chat_template(
                ex["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        except ValueError:
            # Tokenizer has no chat template — use simple delimited format
            tokenizer.chat_template = _FALLBACK_TMPL
            text = tokenizer.apply_chat_template(
                ex["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        return {"text": text}

    ds = load_dataset("json", data_files=path, split="train")
    ds = ds.map(_format, remove_columns=ds.column_names)
    return ds


def _detect_response_template(tokenizer) -> str | None:
    """Find the exact string that immediately precedes the assistant's reply
    in the model's chat format. Used by DataCollatorForCompletionOnlyLM so
    that loss is computed only on assistant tokens."""
    try:
        # Build a minimal two-turn example and locate where the assistant text starts.
        dummy_text = tokenizer.apply_chat_template(
            [{"role": "user",      "content": "__USR__"},
             {"role": "assistant", "content": "__AST__"}],
            tokenize=False,
            add_generation_prompt=False,
        )
        idx = dummy_text.find("__AST__")
        if idx < 0:
            return None
        # Walk back to find a meaningful boundary (newline, >, or special token)
        before = dummy_text[:idx]
        for sep in ("\n", ">", "]"):
            pos = before.rfind(sep)
            if pos >= 0:
                candidate = before[pos + 1:]
                if candidate:
                    return candidate
        return before[-10:] if len(before) >= 10 else before
    except Exception:
        return None


# ── Completion-only collator (manual, for trl >= 1.0 where DataCollatorForCompletionOnlyLM was removed)
def _make_completion_only_collator(tokenizer, response_template: str):
    """Mask system+user tokens so loss is computed on assistant tokens only."""
    import torch
    from transformers import DataCollatorForLanguageModeling

    template_ids = tokenizer.encode(response_template, add_special_tokens=False)

    class CompletionOnlyCollator(DataCollatorForLanguageModeling):
        def __init__(self):
            super().__init__(tokenizer=tokenizer, mlm=False)

        def torch_call(self, examples):
            batch = super().torch_call(examples)
            for i, labels in enumerate(batch["labels"]):
                input_ids = batch["input_ids"][i].tolist()
                n = len(template_ids)
                # Find last occurrence of response_template token sequence
                mask_until = 0
                for j in range(len(input_ids) - n, -1, -1):
                    if input_ids[j:j+n] == template_ids:
                        mask_until = j + n
                        break
                batch["labels"][i][:mask_until] = -100
            return batch

    return CompletionOnlyCollator()


# ── MLflow callback ───────────────────────────────────────────────────────────

def _make_mlflow_callback():
    from transformers import TrainerCallback
    import mlflow

    class MLflowStepCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                for k, v in logs.items():
                    try:
                        mlflow.log_metric(k, float(v), step=state.global_step)
                    except (TypeError, ValueError):
                        pass

    return MLflowStepCallback()


# ── Training ──────────────────────────────────────────────────────────────────

def train_lora(
    train_path: str = TRAIN_PATH,
    val_path: str = VAL_PATH,
    model_name: str = BASE_MODEL,
    output_dir: str = OUTPUT_DIR,
):
    import mlflow
    from trl import SFTTrainer, SFTConfig

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    device = get_device()
    use_fp16 = device == "cuda"   # MPS uses float16 dtype natively, no fp16 flag

    model, tokenizer = load_base_model(model_name)
    model = apply_lora(model)

    train_ds = load_chat_dataset(train_path, tokenizer)
    val_ds   = load_chat_dataset(val_path,   tokenizer)
    print(f"Train: {len(train_ds)} examples | Val: {len(val_ds)} examples")

    if RESPONSE_ONLY_LOSS:
        print("[train] completion-only loss: ON (via SFTConfig)")
    else:
        print("[train] completion-only loss: OFF (RESPONSE_ONLY_LOSS=0)")

    # SFTConfig = TrainingArguments + SFT-specific fields (trl >= 1.0)
    sft_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=EPOCHS,
        max_steps=MAX_STEPS,          # -1 = full epochs; set > 0 to cap steps (smoke test)
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        fp16=use_fp16,
        bf16=False,
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps" if MAX_STEPS < 0 else "no",
        eval_steps=EVAL_STEPS,
        save_strategy="steps" if MAX_STEPS < 0 else "no",
        save_steps=SAVE_STEPS,
        load_best_model_at_end=(MAX_STEPS < 0),
        report_to="none",
        warmup_steps=max(1, int(EPOCHS * 0.03)),  # ~3% warmup (SFTConfig has no warmup_ratio)
        lr_scheduler_type="cosine",
        dataloader_pin_memory=(device == "cuda"),
        # SFT-specific
        dataset_text_field="text",
        max_length=MAX_SEQ_LEN,
        completion_only_loss=bool(RESPONSE_ONLY_LOSS),
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run():
        mlflow.log_params({
            "base_model":    model_name,
            "lora_r":        LORA_R,
            "lora_alpha":    LORA_ALPHA,
            "lora_dropout":  LORA_DROPOUT,
            "epochs":        EPOCHS,
            "lr":            LR,
            "batch_size":    BATCH_SIZE,
            "grad_accum":    GRAD_ACCUM,
            "max_seq_len":   MAX_SEQ_LEN,
            "device":        device,
            "train_size":    len(train_ds),
            "val_size":      len(val_ds),
        })
        trainer.add_callback(_make_mlflow_callback())
        trainer.train()
        trainer.save_model(output_dir)
        mlflow.log_artifact(output_dir, artifact_path="adapter")
        print(f"\nAdapter saved → {output_dir}")
        print("View run: mlflow ui")


# ── Merge adapter → fused model ───────────────────────────────────────────────

def merge_adapter(base_model: str, adapter_path: str, fused_dir: str):
    from peft import PeftModel
    print(f"Loading base: {base_model}")
    model, tokenizer = load_base_model(base_model)
    print(f"Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    print("Merging…")
    model = model.merge_and_unload()
    Path(fused_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(fused_dir)
    tokenizer.save_pretrained(fused_dir)
    print(f"Fused model saved → {fused_dir}")


# ── Quick inference test ──────────────────────────────────────────────────────

def infer_test(model_path: str):
    from inference import generate_hf
    system = (
        "You are a disciplined NSE equity trader. Protect capital first. "
        "R:R must be >= 1.5. Output JSON only."
    )
    user = (
        "[CANBK 2024-10-10]\n"
        "C:103.5 | EMA50:101.2 EMA200:96.8 [BULLISH]\n"
        "VWAP:102.8 C+0.7% | Trend:uptrend\n"
        "S:99.5-100.2 R:106.0-107.5\n"
        "Decide. Output JSON only."
    )
    print("Prompt:\n", user)
    out = generate_hf(model_path, system, user)
    print("\nModel output:\n", out)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train",      default=TRAIN_PATH)
    p.add_argument("--val",        default=VAL_PATH)
    p.add_argument("--model",      default=BASE_MODEL,  help="base model name or path")
    p.add_argument("--output_dir", default=OUTPUT_DIR)
    # merge mode
    p.add_argument("--merge",      action="store_true", help="merge adapter into base model")
    p.add_argument("--base_model", default=BASE_MODEL)
    p.add_argument("--adapter",    default=OUTPUT_DIR,  help="adapter path (from training)")
    p.add_argument("--fused_dir",  default=None,        help="output path for fused model")
    # infer mode
    p.add_argument("--infer",      action="store_true", help="quick inference test")
    args = p.parse_args()

    if args.merge:
        fused = args.fused_dir or str(Path(args.adapter) / "fused_model")
        merge_adapter(args.base_model, args.adapter, fused)
    elif args.infer:
        infer_test(args.model)
    else:
        train_lora(
            train_path=args.train,
            val_path=args.val,
            model_name=args.model,
            output_dir=args.output_dir,
        )
