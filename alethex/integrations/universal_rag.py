"""
Universal RAG Temporal Consistency & Reconciliation Layer for ALETHEX.
Universal middleware for ANY AI framework, ANY Vector Database, and ANY LLM.

Works across:
- All Frameworks: LangChain, LlamaIndex, Haystack, Semantic Kernel, Dify, Flowise, custom RAG.
- All Vector Stores: ChromaDB, Pinecone, Qdrant, Weaviate, Milvus, FAISS, pgvector, Elastic.
- All LLMs: Claude, ChatGPT, Grok, Qwen, Ollama, DeepSeek, Mistral, Gemini, Llama 3.

Usage:
    from alethex.integrations.universal_rag import UniversalRAG

    rag_layer = UniversalRAG(mode="drop") # or mode="annotate" / "prompt"

    # 1. Direct Python filtering for any retrieved chunks:
    clean_chunks = rag_layer.filter(retrieved_chunks)

    # 2. Reconciled prompt string ready for any LLM:
    prompt_context = rag_layer.get_reconciled_context(retrieved_chunks)

    # 3. Wrap any retriever function:
    safe_retriever = rag_layer.wrap(my_vector_db_retriever)
"""

import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

from alethex.api import ConsistencyEngine

logger = logging.getLogger("alethex-universal-rag")


class UniversalRAG:
    """
    Universal RAG reconciliation middleware.
    Inspects retrieved chunks from any vector database or search engine,
    detects temporal drift and logical contradictions, and suppresses or
    annotates invalid assertions before prompt synthesis.
    """

    def __init__(
        self,
        mode: str = "drop",  # "drop", "annotate", or "context_string"
        device: str = "cpu",
        use_onnx: bool = True,
        timestamp_key: str = "timestamp",
        text_key: str = "text",
        id_key: str = "id",
    ):
        """
        Args:
            mode: "drop" (remove stale/contradictory chunks),
                  "annotate" (keep chunks but prepend validity warnings),
                  "context_string" (return a clean unified prompt string).
            device: Inference device ("cpu" or "cuda").
            use_onnx: Enable fast INT8 ONNX inference.
            timestamp_key: Metadata key used for chunk timestamp.
            text_key: Dict attribute for chunk text.
            id_key: Dict attribute for chunk ID.
        """
        self.mode = mode.lower()
        self.device = device
        self.use_onnx = use_onnx
        self.timestamp_key = timestamp_key
        self.text_key = text_key
        self.id_key = id_key
        self._engine: Optional[ConsistencyEngine] = None

    @property
    def engine(self) -> ConsistencyEngine:
        if self._engine is None:
            self._engine = ConsistencyEngine(device=self.device)
        return self._engine

    def _extract_chunk_info(self, chunk: Any, index: int) -> Dict[str, Any]:
        """
        Defensively normalizes chunks from LangChain, LlamaIndex, Haystack,
        ChromaDB, Pinecone, or raw dictionaries/strings.
        """
        text = ""
        cid = f"chunk_{index}"
        ts = datetime.now()
        meta = {}

        # 1. Plain String
        if isinstance(chunk, str):
            text = chunk

        # 2. Standard Dictionary
        elif isinstance(chunk, dict):
            text = chunk.get(self.text_key, chunk.get("page_content", chunk.get("content", "")))
            cid = str(chunk.get(self.id_key, chunk.get("source_id", f"chunk_{index}")))
            meta = chunk.get("metadata", chunk.get("meta", {}))
            raw_ts = chunk.get(self.timestamp_key, meta.get(self.timestamp_key))
            if raw_ts:
                ts = self._parse_timestamp(raw_ts)

        # 3. LangChain Document (has page_content and metadata)
        elif hasattr(chunk, "page_content"):
            text = chunk.page_content
            meta = getattr(chunk, "metadata", {}) or {}
            cid = str(meta.get(self.id_key, f"chunk_{index}"))
            raw_ts = meta.get(self.timestamp_key)
            if raw_ts:
                ts = self._parse_timestamp(raw_ts)

        # 4. LlamaIndex NodeWithScore / TextNode
        elif hasattr(chunk, "node"):
            node = chunk.node
            text = node.get_content() if hasattr(node, "get_content") else str(node)
            meta = getattr(node, "metadata", {}) or {}
            cid = str(getattr(node, "node_id", f"chunk_{index}"))
            raw_ts = meta.get(self.timestamp_key)
            if raw_ts:
                ts = self._parse_timestamp(raw_ts)

        # 5. Generic object with .text or .content
        elif hasattr(chunk, "text"):
            text = chunk.text
            meta = getattr(chunk, "metadata", {}) or {}
            raw_ts = meta.get(self.timestamp_key)
            if raw_ts:
                ts = self._parse_timestamp(raw_ts)
        elif hasattr(chunk, "content"):
            text = chunk.content
            meta = getattr(chunk, "metadata", {}) or {}
            raw_ts = meta.get(self.timestamp_key)
            if raw_ts:
                ts = self._parse_timestamp(raw_ts)
        else:
            text = str(chunk)

        return {
            "id": cid,
            "text": text,
            "timestamp": ts.isoformat(),
            "original_object": chunk,
            "metadata": meta,
        }

    def _parse_timestamp(self, val: Any) -> datetime:
        if isinstance(val, datetime):
            return val
        if isinstance(val, (int, float)):
            try:
                return datetime.fromtimestamp(val)
            except Exception:
                pass
        if isinstance(val, str):
            try:
                return datetime.fromisoformat(val)
            except Exception:
                try:
                    import dateparser
                    parsed = dateparser.parse(val)
                    if parsed:
                        return parsed
                except Exception:
                    pass
        return datetime.now()

    def filter(self, chunks: List[Any], mode: Optional[str] = None) -> List[Any]:
        """
        Reconciles a list of retrieved chunks and returns surviving or annotated chunks.

        Args:
            chunks: List of retrieved chunks (strings, dicts, LangChain Docs, LlamaIndex Nodes).
            mode: Optional override for "drop" or "annotate".
        """
        if not chunks:
            return []

        active_mode = (mode or self.mode).lower()
        extracted = [self._extract_chunk_info(c, i) for i, c in enumerate(chunks)]

        # Prepare payload for ALETHEX context filtering
        filter_payload = [{"id": item["id"], "text": item["text"], "timestamp": item["timestamp"]} for item in extracted]
        reconciled_results = self.engine.filter_context(filter_payload)

        surviving = []
        for orig, item, res in zip(chunks, extracted, reconciled_results):
            is_valid = res.get("is_valid", True)
            status = res.get("status", "valid")

            if active_mode == "drop":
                if is_valid:
                    surviving.append(orig)
            else:  # annotate mode
                annotated_prefix = f"[ALETHEX: {status.upper()}] "
                if isinstance(orig, str):
                    surviving.append(f"{annotated_prefix}{orig}")
                elif isinstance(orig, dict):
                    annotated_dict = dict(orig)
                    text_k = self.text_key if self.text_key in annotated_dict else "text"
                    annotated_dict[text_k] = f"{annotated_prefix}{annotated_dict.get(text_k, '')}"
                    annotated_dict["alethex_status"] = status
                    annotated_dict["alethex_valid"] = is_valid
                    surviving.append(annotated_dict)
                elif hasattr(orig, "page_content"):  # LangChain
                    orig.page_content = f"{annotated_prefix}{orig.page_content}"
                    if hasattr(orig, "metadata") and isinstance(orig.metadata, dict):
                        orig.metadata["alethex_status"] = status
                        orig.metadata["alethex_valid"] = is_valid
                    surviving.append(orig)
                elif hasattr(orig, "node") and hasattr(orig.node, "text"):  # LlamaIndex
                    orig.node.text = f"{annotated_prefix}{orig.node.text}"
                    surviving.append(orig)
                else:
                    surviving.append(orig)

        return surviving

    def get_reconciled_context(self, chunks: List[Any], separator: str = "\n\n") -> str:
        """
        Returns a single, clean, formatted context string containing ONLY
        valid, non-contradicted, non-superseded facts. Ready to drop into ANY prompt:
        e.g., f"Answer using the reconciled context:\n{context}\nQuestion: {query}"
        """
        valid_chunks = self.filter(chunks, mode="drop")
        texts = []
        for c in valid_chunks:
            info = self._extract_chunk_info(c, 0)
            if info["text"].strip():
                texts.append(info["text"].strip())
        return separator.join(texts)

    def audit(self, chunks: List[Any]) -> Dict[str, Any]:
        """
        Audits retrieved chunks and returns the Consistency Index (CI),
        detected contradiction pairs, and stale/superseded statements.
        """
        extracted = [self._extract_chunk_info(c, i) for i, c in enumerate(chunks)]
        docs = [{"source_id": e["id"], "text": e["text"], "timestamp": e["timestamp"]} for e in extracted]
        return self.engine.check_documents(docs)

    def wrap(self, retriever_fn: Callable[..., List[Any]]) -> Callable[..., List[Any]]:
        """
        Decorator / wrapper that transforms ANY retriever function into a
        temporally reconciled retriever:
        e.g.:
            @rag.wrap
            def search_vector_db(query):
                return chroma.query(...)
        """
        def wrapped_retriever(*args, **kwargs) -> List[Any]:
            raw_results = retriever_fn(*args, **kwargs)
            return self.filter(raw_results)
        return wrapped_retriever
