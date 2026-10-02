"""Compresses a conversation session's text into a salience-ranked, deduplicated claim
list plus a short narrative summary, for storage in a long-lived claim store."""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from uuid import uuid4

from alethex.config import Config, default_config_dir
from alethex.extraction.claim_merger import ClaimMerger
from alethex.extraction.openie_extractor import OpenIEExtractor
from alethex.ingestion.schema import Claim, RawDocument

logger = logging.getLogger(__name__)


@dataclass
class CompressionResult:
    """Output of `compress_session`.

    Attributes:
        claims: Salience-ranked, deduplicated claims kept within the compression budget.
        narrative_summary: One-paragraph free-text summary capturing context that doesn't
            fit the structured claim schema (tone, open questions, unresolved threads).
    """
    claims: List[Claim] = field(default_factory=list)
    narrative_summary: str = ""


def _salience_score(claim: Claim, high_salience_predicates: List[str]) -> float:
    """Ranks claims so high-value facts survive aggressive compression budgets."""
    score = claim.confidence
    if claim.predicate.lower() in high_salience_predicates:
        score += 0.5
    if claim.subject.lower() in ("user", "i", "the user", "me"):
        score += 0.3
    return score


def _dedupe_keep_latest(claims: List[Claim]) -> List[Claim]:
    """Collapses claims identical in (subject, predicate, object) to the most recent one."""
    latest_by_key: dict = {}
    for claim in claims:
        key = (claim.subject.lower(), claim.predicate.lower(), claim.object_.lower())
        existing = latest_by_key.get(key)
        if existing is None or claim.timestamp >= existing.timestamp:
            latest_by_key[key] = claim
    return list(latest_by_key.values())


def _narrative_summary(claims: List[Claim], session_text: str, doc_timestamp: datetime) -> str:
    """Builds a one-paragraph narrative summary of the session."""
    if not claims:
        snippet = session_text.strip().splitlines()[0][:200] if session_text.strip() else "no content"
        return f"Session on {doc_timestamp.date().isoformat()} recorded no structured claims ({snippet})."

    top = sorted(claims, key=lambda c: c.confidence, reverse=True)[:5]
    facts = "; ".join(f"{c.subject} {c.predicate} {c.object_}" for c in top)
    return f"Session on {doc_timestamp.date().isoformat()}: {facts}."


def compress_session(
    session_text: str,
    doc_timestamp: datetime,
    source_id: Optional[str] = None,
    config: Optional[Config] = None,
    max_claims: Optional[int] = None,
) -> CompressionResult:
    """Compresses one session's raw text into a bounded, salience-ranked claim list."""
    config = config or Config(default_config_dir())
    budget = max_claims if max_claims is not None else config.context_memory_max_claims_per_session
    high_salience = [p.lower() for p in config.context_memory_high_salience_predicates]

    doc = RawDocument(source_id=source_id or f"session_{uuid4()}", timestamp=doc_timestamp, text=session_text)

    openie = OpenIEExtractor()
    raw_claims = openie.extract(doc)
    merged = ClaimMerger(min_confidence=config.min_confidence).merge(raw_claims)
    deduped = _dedupe_keep_latest(merged)
    ranked = sorted(deduped, key=lambda c: _salience_score(c, high_salience), reverse=True)
    kept = ranked[:budget]
    summary = _narrative_summary(kept, session_text, doc_timestamp)
    return CompressionResult(claims=kept, narrative_summary=summary)
