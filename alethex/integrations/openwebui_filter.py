"""
Open-WebUI Pipeline Filter for ALETHEX.
Drop-in Function for Open-WebUI (the standard interface for Ollama, Qwen, and local LLMs).
Reconciles user memory and conversation history in real-time before sending to the model.

How to use in Open-WebUI:
1. Open Admin Settings -> Functions / Pipelines.
2. Click "+" to create a new Function.
3. Paste the contents of this file.
4. Enable the filter on your model (Qwen, Llama 3, Mistral, etc.).
"""

from typing import Optional, Callable, Awaitable
from datetime import datetime
from pydantic import BaseModel, Field

class Filter:
    class Valves(BaseModel):
        priority: int = Field(default=0, description="Execution priority in pipeline")
        enabled: bool = Field(default=True, description="Enable ALETHEX temporal reconciliation")
        device: str = Field(default="cpu", description="Inference device (cpu or cuda)")

    def __init__(self):
        self.valves = self.Valves()
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            try:
                from alethex.api import ConsistencyEngine
                self._engine = ConsistencyEngine(device=self.valves.device)
            except Exception as e:
                print(f"[ALETHEX Filter] Failed loading engine: {e}")
        return self._engine

    async def inlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        """
        Runs before the prompt is sent to the LLM (Ollama / Qwen / etc.).
        Reconciles contradictory facts in the dialogue turns.
        """
        if not self.valves.enabled:
            return body

        messages = body.get("messages", [])
        if len(messages) <= 1:
            return body

        engine = self._get_engine()
        if not engine:
            return body

        # Extract timestamped chunks from previous user turns
        chunks = []
        for i, m in enumerate(messages[:-1]):
            chunks.append({
                "id": f"turn_{i}",
                "text": m.get("content", ""),
                "timestamp": datetime.now().isoformat()
            })

        try:
            results = engine.filter_context(chunks)
            valid_ids = {r["id"] for r in results if r.get("is_valid", True)}
            
            # Keep only valid non-contradicted turns, always preserving the latest prompt
            filtered_messages = []
            for i, m in enumerate(messages[:-1]):
                if f"turn_{i}" in valid_ids:
                    filtered_messages.append(m)
            filtered_messages.append(messages[-1])

            body["messages"] = filtered_messages
        except Exception as e:
            print(f"[ALETHEX Filter Error] {e}")

        return body

    async def outlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        """
        Runs after the LLM completes generation.
        """
        return body
