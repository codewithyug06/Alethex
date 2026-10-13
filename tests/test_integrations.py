"""
Tests for LangChain and LlamaIndex framework integrations.
"""

from dataclasses import dataclass, field
from typing import Dict, Any

from alethex.integrations.langchain_memory import AlethexReconciledMemory
from alethex.integrations.llamaindex_filter import AlethexConsistencyFilter


@dataclass
class MockLlamaIndexNode:
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    id_: str = "node_1"

    def get_content(self) -> str:
        return self.text


def test_langchain_reconciled_memory():
    memory = AlethexReconciledMemory(dummy=True)
    assert "history" in memory.memory_variables
    assert "reconciled_context" in memory.memory_variables

    # Save a dialogue turn
    reports = memory.save_context(
        inputs={"input": "Alice moved to London."},
        outputs={"output": "Noted! Alice lives in London."},
    )
    assert "user" in reports
    assert "ai" in reports

    # Load memory variables
    loaded = memory.load_memory_variables()
    assert "Alice moved to London." in loaded["history"]
    assert "reconciled_context" in loaded

    # Clear memory
    memory.clear()
    cleared = memory.load_memory_variables()
    assert cleared["history"] == ""


def test_llamaindex_consistency_filter_drop_mode():
    filt = AlethexConsistencyFilter(filter_mode="drop", dummy=True)
    nodes = [
        MockLlamaIndexNode(text="Alice lives in Paris.", id_="node_1"),
        MockLlamaIndexNode(text="The sky is blue.", id_="node_2"),
    ]

    processed = filt.postprocess_nodes(nodes)
    assert isinstance(processed, list)
    assert len(processed) >= 1


def test_llamaindex_consistency_filter_annotate_mode():
    filt = AlethexConsistencyFilter(filter_mode="annotate", dummy=True)
    nodes = [
        MockLlamaIndexNode(text="Alice works at Meta.", id_="node_1"),
    ]

    processed = filt.postprocess_nodes(nodes)
    assert len(processed) == 1
    assert "alethex_valid" in processed[0].metadata
    assert "alethex_status" in processed[0].metadata
