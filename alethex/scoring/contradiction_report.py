from typing import List, Dict, Any, Union
from alethex.graph.belief_graph import BeliefGraph
from alethex.graph import query


def generate_report(graph: BeliefGraph, top_n: Union[int, Any] = 10) -> List[Dict[str, Any]]:
    if not isinstance(top_n, int):
        top_n = 10

    all_conflicts = []

    for entity_id in graph.entities:
        conflicts = query.get_conflicts(graph, entity_id)
        for c1, c2 in conflicts:
            v1 = f"val::{c1.object_}"
            v2 = f"val::{c2.object_}"
            edge_data = graph.graph.get_edge_data(v1, v2)
            confidence = 1.0
            if edge_data:
                for k, attrs in edge_data.items():
                    if attrs.get("type") == "conflict":
                        confidence = attrs.get("confidence", 1.0)
                        break

            recent_time = max(c1.timestamp, c2.timestamp)

            all_conflicts.append({
                "entity": graph.entities[entity_id].canonical_name,
                "predicate": c1.predicate,
                "claim_A_text": c1.raw_sentence,
                "claim_A_timestamp": c1.timestamp.isoformat(),
                "claim_B_text": c2.raw_sentence,
                "claim_B_timestamp": c2.timestamp.isoformat(),
                "confidence": confidence,
                "justification": "Conflict detected by NLI/Judge",
                "_recent_time": recent_time
            })

    all_conflicts.sort(key=lambda x: (x["_recent_time"], x["confidence"]), reverse=True)

    for c in all_conflicts:
        del c["_recent_time"]

    return all_conflicts[:top_n]