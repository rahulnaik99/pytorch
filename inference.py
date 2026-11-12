"""
inference.py
────────────
HuggingFace inference for the fine-tuned trading model.

Works with:
  - Base model only (no adapter)
  - Base model + LoRA adapter (loaded via PeftModel)
  - Fused/merged model (adapter baked in)

Model is lazy-loaded and cached per (model_path, adapter_path) pair — safe to
call repeatedly from pipeline.py without re-loading on every date.

Usage:
    from inference import generate_hf
    out = generate_hf("output/nemotron-lora/fused_model", system, user)

    # With separate adapter (not fused)
    out = generate_hf("nvidia/Nemotron-Mini-4B-Instruct", system, user,
                      adapter_path="output/nemotron-lora")
"""
from __future__ import annotations

import threading
import torch
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env", override=False)
except ImportError:
    pass

_cache: dict = {}   # {(model_path, adapter_path): (model, tokenizer)}
_cache_lock = threading.Lock()


def _load(model_path: str, adapter_path: str | None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = _get_device()
    print(f"[inference] Loading {model_path} → {device}")
    dtype = torch.float32 if device == "cpu" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        device_map={"": device},
        trust_remote_code=True,
    )
    if adapter_path:
        from peft import PeftModel
        print(f"[inference] Attaching adapter {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def generate_hf(
    model_path: str,
    system: str,
    user: str,
    adapter_path: str | None = None,
    max_tokens: int = 512,
    temperature: float = 0.2,
) -> str:
    """
    Generate a response from the fine-tuned model.

    Args:
        model_path:   Path to the fused model directory or HuggingFace model ID.
        system:       System prompt string.
        user:         User prompt string.
        adapter_path: Optional path to a LoRA adapter directory (not needed for fused models).
        max_tokens:   Maximum new tokens to generate.
        temperature:  Sampling temperature (0.1–0.5 typical).

    Returns:
        The model's raw text response (may include JSON).
    """
    key = (model_path, adapter_path)
    with _cache_lock:
        if key not in _cache:
            _cache[key] = _load(model_path, adapter_path)
        model, tokenizer = _cache[key]

    _FALLBACK_TMPL = (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}### System:\n{{ message['content'] }}\n{% endif %}"
        "{% if message['role'] == 'user' %}### User:\n{{ message['content'] }}\n{% endif %}"
        "{% if message['role'] == 'assistant' %}### Assistant:\n{{ message['content'] }}\n{% endif %}"
        "{% endfor %}{% if add_generation_prompt %}### Assistant:\n{% endif %}"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except ValueError:
        tokenizer.chat_template = _FALLBACK_TMPL
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else 1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def hf_generate_factory(model_path: str, adapter_path: str | None = None):
    """
    Returns a generate_fn(system, user) -> str backed by a HuggingFace model.
    Compatible with evaluate.py's evaluate_model_quality().
    Set MAX_EVAL_TOKENS env var to limit generation length (default: 512).
    """
    import os
    max_tok = int(os.environ.get("MAX_EVAL_TOKENS", 512))
    def _gen(system: str, user: str) -> str:
        return generate_hf(model_path, system, user, adapter_path=adapter_path,
                           max_tokens=max_tok)
    return _gen
