from typing import List, Dict
from alethex.ingestion.schema import Claim

class ClaimMerger:
    def __init__(self, min_confidence: float = 0.4):
        self.min_confidence = min_confidence

    def merge(self, claims: List[Claim]) -> List[Claim]:
        valid_claims = [c for c in claims if c.confidence >= self.min_confidence]
        
        merged_dict: Dict[str, Claim] = {}
        for claim in valid_claims:
            key = f"{claim.source_id}::{claim.subject}::{claim.predicate}::{claim.object_}"
            if key not in merged_dict:
                merged_dict[key] = claim
            else:
                existing = merged_dict[key]
                if claim.confidence > existing.confidence:
                    existing.confidence = claim.confidence
                
        return list(merged_dict.values())