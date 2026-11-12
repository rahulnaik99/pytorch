"""
serve.py
────────
Lightweight local REST server for the fine-tuned trading model.

Exposes an Ollama-compatible /api/chat endpoint so pipeline.py can test
the fine-tuned model without any code changes — just swap OLLAMA_URL.

Usage:
    HF_MODEL=output/nemotron-lora/fused_model python serve.py
    HF_MODEL=nvidia/Nemotron-Mini-4B-Instruct HF_ADAPTER=output/nemotron-lora python serve.py

Then test with pipeline.py:
    OLLAMA_URL=http://localhost:8000/api/chat python pipeline.py

Or test directly:
    curl -X POST http://localhost:8000/api/chat \
      -H "Content-Type: application/json" \
      -d '{"messages": [{"role":"user","content":"Hello"}], "stream": false}'
"""
from __future__ import annotations

import os
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env", override=False)
except ImportError:
    pass

HF_MODEL   = os.environ.get("HF_MODEL",   "")
HF_ADAPTER = os.environ.get("HF_ADAPTER", "")
HOST       = os.environ.get("SERVE_HOST", os.environ.get("HOST", "127.0.0.1"))
PORT       = int(os.environ.get("PORT", 8000))

if not HF_MODEL:
    raise SystemExit("Set HF_MODEL env var to a model path or HuggingFace model ID")


def _build_app():
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel

    app = FastAPI(title="Trading LLM serve")

    class Message(BaseModel):
        role: str
        content: str

    class ChatRequest(BaseModel):
        messages: list[Message]
        stream: bool = False
        model: str = ""

    @app.on_event("startup")
    def _preload():
        from inference import generate_hf
        generate_hf.__module__  # touch import; actual load is lazy on first call

    @app.post("/api/chat")
    def chat(req: ChatRequest):
        from inference import generate_hf
        sys_msg  = next((m.content for m in req.messages if m.role == "system"), "")
        user_msg = next((m.content for m in req.messages if m.role == "user"),   "")
        t0 = time.time()
        response = generate_hf(HF_MODEL, sys_msg, user_msg,
                                adapter_path=HF_ADAPTER or None)
        elapsed_ms = int((time.time() - t0) * 1000)
        return JSONResponse({
            "model": HF_MODEL,
            "message": {"role": "assistant", "content": response},
            "done": True,
            "total_duration": elapsed_ms * 1_000_000,
        })

    @app.get("/health")
    def health():
        return {"status": "ok", "model": HF_MODEL}

    return app


if __name__ == "__main__":
    import uvicorn
    app = _build_app()
    print(f"Serving {HF_MODEL} on http://{HOST}:{PORT}")
    print(f"Ollama-compatible endpoint: http://localhost:{PORT}/api/chat")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
