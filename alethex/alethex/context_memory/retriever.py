"""Consistency-filtered retrieval: given a query and an entity, returns only the
currently-valid, non-superseded claims, with conflicting pairs resolved in favor of the
more recent claim (both sides flagged so callers know what was excluded and why)."""
import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np

from alethex.graph.belief_graph import BeliefGraph
from alethex.graph.query import get_conflicts
from alethex.ingestion.schema import Claim

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Output of `retrieve`.

    Attributes:
        claims: Top-k claims surfaced to the caller, ranked by relevance then recency.
        filtered_contradictions: Claim pairs that conflicted; only the newer claim of each
            pair was kept in `claims`, but both are listed here so callers can surface a
            "conflicting claims detected" warning.
        filtered_superseded: Claims excluded because a newer claim has already updated them.
    """
    claims: List[Claim] = field(default_factory=list)
    filtered_contradictions: List[Tuple[Claim, Claim]] = field(default_factory=list)
    filtered_superseded: List[Claim] = field(default_factory=list)


def _embedding_similarity(model: Any, query: str, texts: List[str]) -> List[float]:
    query_vec = model.encode([query])[0]
    query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    doc_vecs = model.encode(texts)
    doc_vecs = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-12)
    return list(np.dot(doc_vecs, query_vec))


def retrieve(
    query: str,
    entity_id: str,
    belief_graph: BeliefGraph,
    top_k: int = 8,
    embedding_model: Optional[Any] = None,
) -> RetrievalResult:
    """Retrieves the top-k relevant, non-superseded, non-contradicted claims for an entity."""
    if not belief_graph.graph.has_node(entity_id):
        return RetrievalResult()

    # Steps 1-2: entity-scoped, non-superseded claims only.
    valid_claims = belief_graph.get_current_valid_claims(entity_id)
    all_claim_ids = {c.claim_id for c in valid_claims}

    superseded = [
        c for c in _all_entity_claims(belief_graph, entity_id)
        if c.claim_id not in all_claim_ids
    ]

    # Step 3: resolve conflicts among the remaining valid claims, keep the newer side.
    conflicts = get_conflicts(belief_graph, entity_id)
    excluded_ids: set = set()
    filtered_contradictions: List[Tuple[Claim, Claim]] = []
    for c1, c2 in conflicts:
        if c1.claim_id not in all_claim_ids or c2.claim_id not in all_claim_ids:
            continue
        older, newer = (c1, c2) if c1.timestamp <= c2.timestamp else (c2, c1)
        excluded_ids.add(older.claim_id)
        filtered_contradictions.append((c1, c2))

    remaining = [c for c in valid_claims if c.claim_id not in excluded_ids]

    # Step 4: rank by relevance (if possible) + recency, then truncate.
    if embedding_model is not None and remaining:
        try:
            texts = [f"{c.subject} {c.predicate} {c.object_}" for c in remaining]
            sims = _embedding_similarity(embedding_model, query, texts)
            ranked = [c for _, c in sorted(zip(sims, remaining), key=lambda x: x[0], reverse=True)]
        except Exception as e:
            logger.warning(f"Embedding similarity failed: {e}; falling back to recency")
            ranked = sorted(remaining, key=lambda c: c.timestamp, reverse=True)
    else:
        ranked = sorted(remaining, key=lambda c: c.timestamp, reverse=True)

    return RetrievalResult(
        claims=ranked[:top_k],
        filtered_contradictions=filtered_contradictions,
        filtered_superseded=superseded,
    )


def _all_entity_claims(belief_graph: BeliefGraph, entity_id: str) -> List[Claim]:
    claims = []
    if belief_graph.graph.has_node(entity_id):
        for _, _, data in belief_graph.graph.out_edges(entity_id, data=True):
            claim = data.get("claim_obj") or data.get("claim")
            if claim is not None:
                claims.append(claim)
    return claims
