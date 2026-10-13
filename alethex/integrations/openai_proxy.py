"""
Universal OpenAI-Compatible Reverse Proxy with ALETHEX Temporal Memory Reconciliation.
Attaches to ANY LLM backend (ChatGPT, Claude, Grok, Qwen, Ollama) by acting as an
intelligent gateway that reconciles memory context before calling the LLM.

Usage:
    python -m alethex.integrations.openai_proxy --port 8000 --upstream http://localhost:11434/v1
    python -m alethex.integrations.openai_proxy --port 8000 --upstream https://api.openai.com/v1
"""

import os
import re
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [ALETHEX-PROXY] %(message)s")
logger = logging.getLogger("alethex-proxy")

app = FastAPI(
    title="ALETHEX Universal LLM Memory Gateway",
    description="Drop-in OpenAI-compatible proxy that reconciles temporal contradictions and removes stale facts for any LLM.",
    version="0.1.0"
)

_ENGINE = None

def get_engine():
    global _ENGINE
    if _ENGINE is None:
        from alethex.api import ConsistencyEngine
        _ENGINE = ConsistencyEngine(device="cpu")
    return _ENGINE


UPSTREAM_BASE_URL = os.environ.get("ALETHEX_UPSTREAM_URL", "http://localhost:11434/v1")
UPSTREAM_API_KEY = os.environ.get("UPSTREAM_API_KEY", os.environ.get("OPENAI_API_KEY", ""))


def reconcile_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Scans system and user messages for multi-statement memory contexts,
    runs ALETHEX reconciliation, and replaces contradictory/superseded statements.
    """
    engine = get_engine()
    clean_messages = []

    for msg in messages:
        content = msg.get("content", "")
        # Check if message contains temporal memory markers or multiple dated sentences
        if isinstance(content, str) and ("[" in content or "moved to" in content or "works at" in content):
            # Split into candidate sentences
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", content) if len(s.strip()) > 5]
            if len(sentences) >= 2:
                chunks = [{"id": f"s_{i}", "text": s, "timestamp": datetime.now().isoformat()} for i, s in enumerate(sentences)]
                try:
                    reconciled = engine.filter_context(chunks)
                    valid_texts = [r["text"] for r in reconciled if r.get("is_valid", True)]
                    new_content = " ".join(valid_texts)
                    clean_messages.append({"role": msg["role"], "content": new_content})
                    continue
                except Exception as ex:
                    logger.warning(f"Reconciliation fallback: {ex}")
        clean_messages.append(msg)

    return clean_messages


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    messages = body.get("messages", [])
    model = body.get("model", "default")
    stream = body.get("stream", False)

    logger.info(f"Incoming chat request for model '{model}' with {len(messages)} messages.")
    reconciled_msgs = reconcile_messages(messages)
    body["messages"] = reconciled_msgs

    # Forward to upstream LLM
    headers = {"Content-Type": "application/json"}
    auth_header = request.headers.get("Authorization")
    if auth_header:
        headers["Authorization"] = auth_header
    elif UPSTREAM_API_KEY:
        headers["Authorization"] = f"Bearer {UPSTREAM_API_KEY}"

    target_url = f"{UPSTREAM_BASE_URL.rstrip('/')}/chat/completions"

    try:
        if stream:
            upstream_resp = requests.post(target_url, json=body, headers=headers, stream=True)
            return StreamingResponse(upstream_resp.iter_content(chunk_size=1024), media_type="text/event-stream")
        else:
            upstream_resp = requests.post(target_url, json=body, headers=headers)
            return JSONResponse(status_code=upstream_resp.status_code, content=upstream_resp.json())
    except Exception as e:
        logger.error(f"Error forwarding to upstream LLM ({target_url}): {e}")
        raise HTTPException(status_code=502, detail=f"Failed connecting to upstream LLM: {str(e)}")


@app.get("/v1/models")
async def list_models(request: Request):
    headers = {}
    auth_header = request.headers.get("Authorization")
    if auth_header:
        headers["Authorization"] = auth_header
    try:
        resp = requests.get(f"{UPSTREAM_BASE_URL.rstrip('/')}/models", headers=headers)
        return JSONResponse(status_code=resp.status_code, content=resp.json())
    except Exception:
        return {"data": [{"id": "alethex-reconciled", "object": "model"}]}


@app.get("/health")
def health():
    return {"status": "ok", "service": "ALETHEX Universal Gateway", "backend": "ONNX INT8 / Mini"}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ALETHEX Universal OpenAI-Compatible Memory Gateway")
    parser.add_argument("--port", type=int, default=8000, help="Proxy port (default: 8000)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    parser.add_argument("--upstream", type=str, default="http://localhost:11434/v1", help="Upstream LLM base URL (Ollama, OpenAI, Grok, etc.)")
    args = parser.parse_args()

    global UPSTREAM_BASE_URL
    UPSTREAM_BASE_URL = args.upstream
    logger.info(f"Starting ALETHEX Gateway on http://{args.host}:{args.port} pointing to {UPSTREAM_BASE_URL}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
