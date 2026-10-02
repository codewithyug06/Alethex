from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from alethex.graph.belief_graph import BeliefGraph


def generate_report(graph: BeliefGraph, config: Optional[Any] = None) -> List[Dict[str, Any]]:
    stale_claims = []
    now = datetime.now()
    days = getattr(config, "staleness_threshold_days", 30) if config is not None else 30
    threshold = now - timedelta(days=days)

    for entity_id in graph.entities:
        for _, _, data in graph.graph.edges(entity_id, data=True):
            claim = data.get("claim_obj")
            if claim and claim.validity_end is not None:
                if claim.validity_end < threshold:
                    stale_claims.append({
                        "entity": graph.entities[entity_id].canonical_name,
                        "claim_id": claim.claim_id,
                        "text": claim.raw_sentence,
                        "validity_end": claim.validity_end.isoformat(),
                        "days_stale": (now - claim.validity_end).days
                    })

    return stale_claims