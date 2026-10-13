import json
import logging
from openai import OpenAI
from alethex.ingestion.schema import ClaimPair
import time
from datetime import datetime

class LLMJudge:
    def __init__(self, api_key: str = "dummy", model: str = "gpt-4o", log_file: str = "logs/llm_calls.jsonl"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.log_file = log_file
        
    def _log_call(self, prompt: str, response: str, latency: float):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model,
            "input": prompt,
            "output": response,
            "latency": latency
        }
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

    def adjudicate(self, pair: ClaimPair) -> ClaimPair:
        a = pair.claim_a
        b = pair.claim_b
        
        prompt = (
            f"Given two claims about the same entity, classify their relationship.\n"
            f"Claim A (timestamp: {a.timestamp}): \"{a.subject} {a.predicate} {a.object_}\"\n"
            f"Claim B (timestamp: {b.timestamp}): \"{b.subject} {b.predicate} {b.object_}\"\n"
            f"Output JSON only: {{\"relation\": \"consistent\"|\"superseded\"|\"conflicting\", \"confidence\": float, \"justification\": str}}"
        )
        
        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a logical consistency judge."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            latency = time.time() - start_time
            content = response.choices[0].message.content
            self._log_call(prompt, content, latency)
            
            parsed = json.loads(content)
            pair.relation = parsed.get("relation", "ambiguous")
            pair.confidence = float(parsed.get("confidence", 0.0))
            pair.justification = parsed.get("justification", "LLM Adjudication")
        except Exception as e:
            logging.error(f"LLM Judge failed: {e}")
            
        return pair