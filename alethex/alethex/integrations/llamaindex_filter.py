"""
LlamaIndex Node Postprocessor Integration for ALETHEX.
Filters out superseded and conflicting memory nodes from retrieved RAG context prior to generation.
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from alethex.api import ConsistencyEngine


class AlethexConsistencyFilter:
    """
    LlamaIndex-compatible Node Postprocessor for temporal belief reconciliation.
    Intercepts retrieved chunks and filters out superseded or contradictory context.
    """

    def __init__(
        self,
        filter_mode: str = "drop",  # "drop" or "annotate"
        device: str = "cpu",
        dummy: bool = False,
    ):
        self.filter_mode = filter_mode
        self.engine = ConsistencyEngine(device=device)
        if dummy:
            self.engine.nli.dummy = True
            self.engine.nli.classifier = None

    def postprocess_nodes(
        self,
        nodes: List[Any],
        query_bundle: Optional[Any] = None,
    ) -> List[Any]:
        """
        LlamaIndex node postprocessor entry point.

        Args:
            nodes: List of NodeWithScore objects (or objects with .text / .node attribute).
            query_bundle: Optional query bundle from LlamaIndex retriever.

        Returns:
            Filtered or annotated list of nodes.
        """
        if not nodes:
            return []

        # Extract text and timestamp from each node defensively
        entries = []
        for idx, n in enumerate(nodes):
            text = ""
            if hasattr(n, "node") and hasattr(n.node, "get_content"):
                text = n.node.get_content()
            elif hasattr(n, "text"):
                text = n.text
            elif hasattr(n, "get_content"):
                text = n.get_content()
            elif isinstance(n, dict):
                text = n.get("text", "")

            # Look for metadata timestamp if present
            ts = datetime.now()
            node_obj = getattr(n, "node", n)
            metadata = getattr(node_obj, "metadata", {}) if not isinstance(node_obj, dict) else node_obj.get("metadata", {})
            if "timestamp" in metadata:
                raw_ts = metadata["timestamp"]
                if isinstance(raw_ts, str):
                    try:
                        ts = datetime.fromisoformat(raw_ts)
                    except Exception:
                        pass
                elif isinstance(raw_ts, datetime):
                    ts = raw_ts

            entries.append({
                "id": str(getattr(node_obj, "id_", f"node_{idx}")),
                "text": text,
                "timestamp": ts.isoformat(),
            })

        # Run ALETHEX context filter
        filtered_results = self.engine.filter_context(entries)

        surviving_nodes = []
        for n, f_res in zip(nodes, filtered_results):
            node_obj = getattr(n, "node", n)
            is_valid = f_res.get("is_valid", True)
            status = f_res.get("status", "valid")

            if self.filter_mode == "drop":
                if is_valid:
                    surviving_nodes.append(n)
            else:  # annotate mode
                if hasattr(node_obj, "metadata"):
                    node_obj.metadata["alethex_status"] = status
                    node_obj.metadata["alethex_valid"] = is_valid
                elif isinstance(node_obj, dict):
                    node_obj["metadata"] = node_obj.get("metadata", {})
                    node_obj["metadata"]["alethex_status"] = status
                    node_obj["metadata"]["alethex_valid"] = is_valid
                surviving_nodes.append(n)

        return surviving_nodes

    def filter_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generic convenience method for vector DB retrieval pipelines (Chroma, Qdrant, Pinecone).
        """
        return self.engine.filter_context(chunks)
