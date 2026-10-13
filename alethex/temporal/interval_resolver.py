from typing import List, Dict
from alethex.ingestion.schema import Claim
from alethex.temporal.timex_tagger import TimexTagger

class IntervalResolver:
    def __init__(self, staleness_threshold_days: int = 30):
        self.staleness_threshold_days = staleness_threshold_days
        self.tagger = TimexTagger()
        self.end_markers = ["until", "before", "prior to", "ending", "expired"]

    def resolve_intervals(self, claims: List[Claim]) -> List[Claim]:
        for claim in claims:
            resolved = self.tagger.extract_time(claim.raw_sentence, claim.timestamp)
            
            claim.validity_start = claim.timestamp
            claim.validity_end = None
            
            if resolved:
                expr_str, dt = resolved
                expr_lower = expr_str.lower()
                
                is_end = False
                for marker in self.end_markers:
                    if f"{marker} {expr_lower}" in claim.raw_sentence.lower():
                        is_end = True
                        break
                        
                if is_end:
                    claim.validity_end = dt
                else:
                    claim.validity_start = dt

        groups: Dict[str, List[Claim]] = {}
        for claim in claims:
            key = f"{claim.subject}::{claim.predicate}"
            if key not in groups:
                groups[key] = []
            groups[key].append(claim)
            
        for key, group in groups.items():
            group.sort(key=lambda x: x.validity_start or x.timestamp)
            
            for i in range(len(group) - 1):
                current_claim = group[i]
                next_claim = group[i + 1]
                
                if current_claim.validity_end is None and next_claim.validity_start > current_claim.validity_start:
                    current_claim.validity_end = next_claim.validity_start

        return claims
