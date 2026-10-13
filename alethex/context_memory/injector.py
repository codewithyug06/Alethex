"""Formats retrieved claims and narrative summary into a prompt block within a token budget."""
from typing import List, Optional
from alethex.context_memory.retriever import RetrievalResult
from alethex.ingestion.schema import Claim


def inject(
    result: RetrievalResult,
    narrative_summary: Optional[str] = None,
    max_tokens: int = 512,
) -> str:
    """Formats retrieved claims into a system prompt injection block."""
    lines = []
    if narrative_summary:
        lines.append(f"Summary: {narrative_summary}")

    if result.claims:
        lines.append("Active Beliefs:")
        for c in result.claims:
            ts = c.timestamp.date().isoformat() if c.timestamp else "unknown"
            lines.append(f"- [{ts}] {c.subject} {c.predicate} {c.object_}")
    else:
        lines.append("No relevant claims found.")

    if result.filtered_contradictions:
        lines.append("\nWarning: Conflicting claims were detected and newer values were prioritized.")

    text = "\n".join(lines)
    # Estimate tokens: 4 chars per token
    char_limit = max_tokens * 4
    if len(text) > char_limit:
        text = text[:char_limit].rsplit("\n", 1)[0]
    return text
