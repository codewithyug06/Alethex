"""
Turn-by-turn dynamic belief tracking for real-time conversational agents.
Maintains incremental knowledge state, detecting contradictions and supersessions immediately on each turn.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from alethex.config import Config, default_config_dir, PROJECT_ROOT
from alethex.coreference.entity_linker import EntityLinker
from alethex.coreference.within_doc_coref import WithinDocCoref
from alethex.extraction.claim_merger import ClaimMerger
from alethex.extraction.openie_extractor import OpenIEExtractor
from alethex.graph.belief_graph import BeliefGraph
from alethex.ingestion.schema import Claim, ClaimPair, EntityNode, RawDocument
from alethex.relation.nli_classifier import NLIClassifier
from alethex.temporal.interval_resolver import IntervalResolver


@dataclass
class TurnAuditReport:
    """Report generated immediately after auditing a conversational turn."""
    turn_id: str
    speaker: str
    timestamp: datetime
    raw_text: str
    extracted_claims: List[Claim] = field(default_factory=list)
    contradictions: List[ClaimPair] = field(default_factory=list)
    supersessions: List[ClaimPair] = field(default_factory=list)
    is_consistent: bool = True

    def summary(self) -> str:
        """Human-readable status summary for logging or agent self-monitoring."""
        if not self.extracted_claims:
            return f"[{self.turn_id}] {self.speaker}: No structured claims extracted."
        status = "CONSISTENT" if self.is_consistent else "CONTRADICTION DETECTED"
        parts = [f"[{self.turn_id}] {self.speaker} -> {status} ({len(self.extracted_claims)} claims)"]
        if self.contradictions:
            parts.append(f"  Alert: {len(self.contradictions)} unresolved contradiction(s) found!")
            for c in self.contradictions:
                parts.append(f"    - '{c.claim_a.raw_sentence}' vs '{c.claim_b.raw_sentence}' ({c.confidence:.2f})")
        if self.supersessions:
            parts.append(f"  Updated: {len(self.supersessions)} past belief(s) superseded.")
            for s in self.supersessions:
                parts.append(f"    - Replaced '{s.claim_a.raw_sentence}' with '{s.claim_b.raw_sentence}'")
        return "\n".join(parts)


class OnlineBeliefTracker:
    """
    Incremental in-memory state tracker for multi-turn conversations.
    Unlike full-corpus batch processing, OnlineBeliefTracker executes in O(K) time
    per turn by comparing new claims strictly against active slots in the entity graph.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        device: str = "cpu",
        use_onnx: bool = True,
        dummy: bool = False,
    ):
        self.config = config or Config(default_config_dir())
        self.device = device
        self.dummy = dummy

        # Initialize core components
        self.openie = OpenIEExtractor()
        self.merger = ClaimMerger()
        self.coref = WithinDocCoref(mode="heuristic")
        self.linker = EntityLinker(
            eps=self.config.entity_linking_eps,
            min_samples=self.config.entity_linking_min_samples,
            device=self.device,
        )
        self.resolver = IntervalResolver(staleness_threshold_days=self.config.staleness_threshold_days)

        # Prefer ONNX INT8 model if present and requested
        onnx_model_path = PROJECT_ROOT / "models" / "alethex-nli-int8"
        if not onnx_model_path.exists():
            onnx_model_path = Path("models/alethex-nli-int8")
        if use_onnx and (onnx_model_path / "model_int8.onnx").exists():
            model_to_use = str(onnx_model_path)
            actual_use_onnx = True
        else:
            model_to_use = self.config.nli_model
            actual_use_onnx = False

        self.nli = NLIClassifier(
            model_name=model_to_use,
            dummy=dummy,
            device=self.device,
            use_onnx=actual_use_onnx,
        )

        # Dynamic state
        self.graph = BeliefGraph()
        self.entities: Dict[str, EntityNode] = {}
        # Mapping: entity_name -> predicate -> List[Claim]
        self.active_slots: Dict[str, Dict[str, List[Claim]]] = {}
        self.turn_history: List[TurnAuditReport] = []
        self.classified_pairs: List[ClaimPair] = []

    def process_turn(
        self,
        speaker: str,
        text: str,
        timestamp: Optional[datetime] = None,
    ) -> TurnAuditReport:
        """
        Processes a single conversational turn from user or assistant.

        Args:
            speaker: Speaker identifier (e.g. "User", "Assistant").
            text: Utterance text.
            timestamp: Turn timestamp (defaults to datetime.now()).

        Returns:
            TurnAuditReport with extracted claims, conflict flags, and supersession updates.
        """
        now = timestamp or datetime.now()
        turn_num = len(self.turn_history) + 1
        turn_id = f"turn_{turn_num:03d}"

        # 1. Ingest turn as raw document
        doc = RawDocument(source_id=turn_id, timestamp=now, text=text)

        # 2. Extract & resolve claims
        claims = self.openie.extract(doc)
        claims = self.merger.merge(claims)
        claims = self.coref.resolve(doc, claims)
        claims, new_nodes = self.linker.link(claims)
        claims = self.resolver.resolve_intervals(claims)

        # Register any new entity nodes
        for node in new_nodes:
            self.entities[node.entity_id] = node

        turn_contradictions: List[ClaimPair] = []
        turn_supersessions: List[ClaimPair] = []

        # 3. Compare new claims against currently active claims
        for new_c in claims:
            subj = new_c.subject
            pred = new_c.predicate

            if subj not in self.active_slots:
                self.active_slots[subj] = {}

            existing_claims = self.active_slots[subj].get(pred, [])
            superseded_indices = []

            for idx, old_c in enumerate(existing_claims):
                # Run NLI classification between old belief and new statement
                pair = self.nli.classify_pair(old_c, new_c)
                self.classified_pairs.append(pair)

                if pair.relation == "conflicting":
                    turn_contradictions.append(pair)
                elif pair.relation == "superseded":
                    turn_supersessions.append(pair)
                    superseded_indices.append(idx)

            # Prune superseded claims from active slots
            for idx in reversed(superseded_indices):
                existing_claims.pop(idx)

            # Add new claim to active tracking
            existing_claims.append(new_c)
            self.active_slots[subj][pred] = existing_claims

        # 4. Update underlying belief graph
        all_active_claims = self.get_active_claims()
        self.graph.build(
            nodes=list(self.entities.values()),
            claims=all_active_claims,
            pairs=self.classified_pairs,
        )

        report = TurnAuditReport(
            turn_id=turn_id,
            speaker=speaker,
            timestamp=now,
            raw_text=text,
            extracted_claims=claims,
            contradictions=turn_contradictions,
            supersessions=turn_supersessions,
            is_consistent=len(turn_contradictions) == 0,
        )
        self.turn_history.append(report)
        return report

    def get_active_claims(self, entity: Optional[str] = None) -> List[Claim]:
        """Returns all currently active, non-superseded claims."""
        claims = []
        if entity is not None:
            slots = self.active_slots.get(entity, {})
            for pred_claims in slots.values():
                claims.extend(pred_claims)
        else:
            for entity_slots in self.active_slots.values():
                for pred_claims in entity_slots.values():
                    claims.extend(pred_claims)
        return claims

    def get_context_prompt(self, max_tokens: int = 500) -> str:
        """
        Formats currently verified, active beliefs into a structured context block
        suitable for LLM prompt injection or system message memory.
        """
        active = self.get_active_claims()
        if not active:
            return ""

        lines = ["### Verified Active Beliefs (ALETHEX Temporal Memory)"]
        for c in active:
            ts_str = c.timestamp.strftime("%Y-%m-%d")
            lines.append(f"- [{ts_str}] {c.subject} {c.predicate} {c.object_}")

        # Check if unresolved contradictions exist in history
        active_conflicts = []
        for r in self.turn_history:
            active_conflicts.extend(r.contradictions)

        if active_conflicts:
            lines.append("\n> [!WARNING] Unresolved Contradictions Detected:")
            for p in active_conflicts[-3:]:
                lines.append(f"> - '{p.claim_a.raw_sentence}' CONFLICTS WITH '{p.claim_b.raw_sentence}'")

        return "\n".join(lines)

    def get_current_ci(self) -> float:
        """Calculates the current consistency index over the underlying belief graph."""
        from alethex.scoring.consistency_index import calculate_ci
        report = calculate_ci(self.graph)
        return float(report.corpus_ci)

