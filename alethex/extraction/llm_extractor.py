import json
import logging
import time
from typing import List
from uuid import uuid4
from openai import OpenAI
from pathlib import Path
from alethex.ingestion.schema import Claim, RawDocument
from datetime import datetime

class LLMExtractor:
    def __init__(self, api_key: str = "dummy", model: str = "gpt-4o", log_file: str = "logs/llm_calls.jsonl"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
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

    def extract(self, doc: RawDocument, sentences: List[str]) -> List[Claim]:
        claims = []
        system_prompt = (
            "Extract all factual claims from the text as a JSON object with a key 'claims' containing a JSON array. "
            "Each claim: {\"subject\": str, \"predicate\": str, \"object\": str, \"confidence\": float 0-1}. "
            "Output ONLY valid JSON, no prose."
        )
        for sent in sentences:
            start_time = time.time()
            prompt = f"Text: {sent}"
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"}
                )
                latency = time.time() - start_time
                content = response.choices[0].message.content
                self._log_call(prompt, content, latency)
                
                parsed = json.loads(content)
                extracted_claims = parsed.get("claims", [])
                for item in extracted_claims:
                    claims.append(
                        Claim(
                            claim_id=str(uuid4()),
                            source_id=doc.source_id,
                            timestamp=doc.timestamp,
                            subject=item.get("subject", "").lower(),
                            predicate=item.get("predicate", "").lower(),
                            object_=item.get("object", "").lower(),
                            confidence=float(item.get("confidence", 0.0)),
                            raw_sentence=sent.strip(),
                            validity_start=doc.timestamp,
                            validity_end=None
                        )
                    )
            except Exception as e:
                logging.error(f"LLM extraction failed: {e}")
                
        return claims