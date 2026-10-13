from dataclasses import dataclass, field
from typing import Dict, Tuple, Iterator, Any
from alethex.graph.belief_graph import BeliefGraph
from alethex.graph import query


@dataclass
class ConsistencyReport:
    corpus_ci: float
    total_entities: int = 0
    total_claims: int = 0
    total_conflict_pairs: int = 0
    entity_cis: Dict[str, float] = field(default_factory=dict)

    def __iter__(self) -> Iterator[Any]:
        yield self.corpus_ci
        yield self.entity_cis


def calculate_ci(graph: BeliefGraph) -> ConsistencyReport:
    entity_cis = {}
    weighted_sum = 0.0
    total_claims = 0
    total_conflicts = 0

    for entity_id in graph.entities:
        claims = []
        for _, _, data in graph.graph.edges(entity_id, data=True):
            if "claim_obj" in data:
                claims.append(data["claim_obj"])

        num_claims = len(claims)
        total_pairs = num_claims * (num_claims - 1) / 2

        conflicts = query.get_conflicts(graph, entity_id)
        num_conflicts = len(conflicts)
        total_conflicts += num_conflicts

        if total_pairs > 0:
            ci = 1.0 - (num_conflicts / total_pairs)
            ci = max(0.0, ci)
        else:
            ci = 1.0

        entity_cis[entity_id] = ci
        weighted_sum += ci * num_claims
        total_claims += num_claims

    corpus_ci = 1.0
    if total_claims > 0:
        corpus_ci = weighted_sum / total_claims

    return ConsistencyReport(
        corpus_ci=corpus_ci,
        total_entities=len(graph.entities),
        total_claims=total_claims,
        total_conflict_pairs=total_conflicts,
        entity_cis=entity_cis,
    )