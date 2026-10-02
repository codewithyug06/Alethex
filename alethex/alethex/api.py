"""
High-level public Python API for ALETHEX.
Enables plug-and-play temporal belief reconciliation for RAG pipelines, LLM agents, and memory stores.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from alethex.config import Config, default_config_dir, load_all_configs
from alethex.context_memory.compressor import CompressionResult, compress_session
from alethex.context_memory.injector import inject
from alethex.context_memory.retriever import RetrievalResult, retrieve
from alethex.coreference.entity_linker import EntityLinker
from alethex.coreference.within_doc_coref import WithinDocCoref
from alethex.extraction.claim_merger import ClaimMerger
from alethex.extraction.openie_extractor import OpenIEExtractor
from alethex.graph.belief_graph import BeliefGraph
from alethex.ingestion.schema import Claim, ClaimPair, RawDocument
from alethex.relation.nli_classifier import NLIClassifier
from alethex.relation.pair_generator import PairGenerator
from alethex.scoring.consistency_index import calculate_ci
from alethex.scoring.contradiction_report import generate_report as generate_contradictions
from alethex.scoring.staleness_report import generate_report as generate_staleness
from alethex.temporal.interval_resolver import IntervalResolver


class ConsistencyEngine:
    """
    In-memory Temporal Belief Consistency Engine.
    Maintains a dynamic belief graph across multi-turn interactions or retrieved memory stores.
    """

    def __init__(
        self,
        config_dir: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        config: Optional[Config] = None,
    ):
        """
        Args:
            config_dir: Directory containing ALETHEX yaml configs. Ignored if `config` is given.
            device: Torch device override for the NLI model (e.g. "cpu", "cuda").
            config: A pre-loaded `Config` instance to reuse instead of loading from `config_dir`.
                Lets callers (e.g. the `alethex-pipeline` CLI) share one loaded config across
                the engine and any additional standalone components (LLMExtractor, LLMJudge)
                instead of loading configs twice.
        """
        if config is not None:
            self.config = config
            self.config_dir = getattr(config, "config_dir", None) or default_config_dir()
            self.configs = getattr(config, "configs", {})
        else:
            config_dir = Path(config_dir) if config_dir is not None else default_config_dir()
            self.config_dir = config_dir
            self.configs = load_all_configs(config_dir) if config_dir.exists() else {}
            self.config = Config(config_dir)
            self.config.configs = self.configs
        self.device = device or self.config.nli_device

        # Modules
        self.openie = OpenIEExtractor()
        self.merger = ClaimMerger()
        self.coref = WithinDocCoref(mode="heuristic")
        self.linker = EntityLinker(
            eps=self.config.entity_linking_eps,
            min_samples=self.config.entity_linking_min_samples,
            device=self.device
        )
        self.resolver = IntervalResolver(staleness_threshold_days=self.config.staleness_threshold_days)
        self.pair_gen = PairGenerator(self.config)
        self.nli = NLIClassifier(dummy=False, device=self.device)
        self.graph = BeliefGraph()

    def run_core_pipeline(
        self,
        documents: List[RawDocument],
        llm_extractor: Optional[Any] = None,
        llm_judge: Optional[Any] = None,
        llm_judge_threshold: Optional[float] = None,
    ) -> Tuple[List[Claim], Dict[str, Any], List[Tuple[Claim, Claim]], List[ClaimPair], BeliefGraph]:
        """
        Executes the canonical claim -> coref -> linking -> temporal -> NLI -> belief graph sequence.
        """
        # 1. Claim extraction (OpenIE + optional LLM-assist)
        all_claims = []
        for d in documents:
            all_claims.extend(self.openie.extract(d))
        if llm_extractor is not None:
            import re
            for d in documents:
                sentences = [s for s in re.split(r"(?<=[.!?])\s+", d.text.strip()) if s]
                all_claims.extend(llm_extractor.extract(d, sentences))
        claims = self.merger.merge(all_claims)

        # 2. Coref & Entity Linking
        for d in documents:
            claims = self.coref.resolve(d, claims)
        claims, entities = self.linker.link(claims)

        # 3. Temporal resolution
        claims = self.resolver.resolve_intervals(claims)

        # 4. Pair generation & NLI classification
        pairs = list(self.pair_gen.generate(entities, claims))
        classified_pairs = [self.nli.classify_pair(c1, c2) for (c1, c2) in pairs]

        # 4b. Optional LLM-judge adjudication for low-confidence/conflicting pairs
        if llm_judge is not None:
            threshold = llm_judge_threshold if llm_judge_threshold is not None else self.config.nli_threshold
            for pair in classified_pairs:
                if pair.confidence < threshold or pair.relation == "conflicting":
                    judged = llm_judge.adjudicate(pair)
                    pair.relation = judged.relation
                    pair.confidence = judged.confidence
                    pair.justification = judged.justification
                    pair.classified_by = "llm_judge"

        # 5. Build Belief Graph
        graph = BeliefGraph()
        ent_list = list(entities.values()) if isinstance(entities, dict) else entities
        graph.build(ent_list, claims, classified_pairs)

        return claims, entities, pairs, classified_pairs, graph

    def check_documents(self, documents: List[RawDocument]) -> Dict[str, Any]:
        """Runs consistency checking over a list of RawDocument instances."""
        if not documents:
            return {
                "corpus_ci": 1.0,
                "total_claims": 0,
                "contradictions": [],
                "stale_claims": [],
                "graph": self.graph,
            }

        claims, entities, pairs, classified_pairs, self.graph = self.run_core_pipeline(documents)

        # Scoring
        ci_report = calculate_ci(self.graph)
        contradictions = generate_contradictions(self.graph, self.config)
        stale_claims = generate_staleness(self.graph, self.config)

        return {
            "corpus_ci": ci_report.corpus_ci,
            "entity_ci": ci_report.entity_cis,
            "total_claims": len(claims),
            "canonical_entities": len(entities),
            "contradictions": contradictions,
            "stale_claims": stale_claims,
            "graph": self.graph,
        }

    def filter_context(
        self,
        retrieved_entries: List[Union[str, Dict[str, Any]]],
        as_of_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Filters out superseded and conflicting memory entries from retrieved RAG context.
        Returns only currently-valid, consistent factual statements.
        """
        now = as_of_date or datetime.now()
        raw_docs = []

        for idx, entry in enumerate(retrieved_entries):
            if isinstance(entry, str):
                raw_docs.append(RawDocument(
                    source_id=f"entry_{idx}",
                    timestamp=now,
                    text=entry
                ))
            elif isinstance(entry, dict):
                ts = entry.get("timestamp")
                if isinstance(ts, str):
                    ts = datetime.fromisoformat(ts)
                elif not isinstance(ts, datetime):
                    ts = now
                raw_docs.append(RawDocument(
                    source_id=str(entry.get("id", f"entry_{idx}")),
                    timestamp=ts,
                    text=entry.get("text", "")
                ))

        audit = self.check_documents(raw_docs)

        # Identify conflicting claim sources
        conflicting_sources = set()
        for c in audit.get("contradictions", []):
            claim_a_text = c.get("claim_A_text")
            claim_b_text = c.get("claim_B_text")
            for d in raw_docs:
                if (claim_a_text and claim_a_text in d.text) or (claim_b_text and claim_b_text in d.text):
                    conflicting_sources.add(d.source_id)

        # Identify stale claim sources
        stale_sources = set()
        for s in audit.get("stale_claims", []):
            stale_text = s.get("text")
            for d in raw_docs:
                if stale_text and stale_text in d.text:
                    stale_sources.add(d.source_id)

        filtered = []
        for doc in raw_docs:
            is_stale = doc.source_id in stale_sources
            is_conflicted = doc.source_id in conflicting_sources
            filtered.append({
                "source_id": doc.source_id,
                "timestamp": doc.timestamp.isoformat(),
                "text": doc.text,
                "is_valid": not (is_stale or is_conflicted),
                "status": "conflicting" if is_conflicted else ("stale" if is_stale else "valid"),
            })
        return filtered


def check(
    documents: List[Union[str, Dict[str, Any], RawDocument]],
    config_dir: Optional[Union[str, Path]] = None,
    device: Optional[str] = None
) -> Dict[str, Any]:
    """Top-level convenience function for auditing belief consistency across documents."""
    engine = ConsistencyEngine(config_dir=config_dir, device=device)
    raw_docs = []
    for idx, doc in enumerate(documents):
        if isinstance(doc, RawDocument):
            raw_docs.append(doc)
        elif isinstance(doc, str):
            raw_docs.append(RawDocument(source_id=f"doc_{idx}", timestamp=datetime.now(), text=doc))
        elif isinstance(doc, dict):
            ts = doc.get("timestamp")
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts)
            elif not isinstance(ts, datetime):
                ts = datetime.now()
            raw_docs.append(RawDocument(
                source_id=str(doc.get("source_id", f"doc_{idx}")),
                timestamp=ts,
                text=str(doc.get("text", ""))
            ))
    return engine.check_documents(raw_docs)


def filter_context(
    retrieved_entries: List[Union[str, Dict[str, Any]]],
    config_dir: Optional[Union[str, Path]] = None,
    device: Optional[str] = None,
    as_of_date: Optional[datetime] = None
) -> List[Dict[str, Any]]:
    """Top-level convenience function to filter out superseded/conflicting entries."""
    engine = ConsistencyEngine(config_dir=config_dir, device=device)
    return engine.filter_context(retrieved_entries, as_of_date=as_of_date)
