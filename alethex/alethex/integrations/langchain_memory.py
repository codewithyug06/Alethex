"""
LangChain Conversation Memory Integration for ALETHEX.
Suppresses superseded facts and warns agents of temporal contradictions in memory.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime

from alethex.streaming.online_session import OnlineBeliefTracker, TurnAuditReport


class AlethexReconciledMemory:
    """
    Drop-in LangChain-compatible conversation memory component.
    Tracks dialogue turns and continuously maintains a consistent temporal belief state.

    Attributes:
        memory_key: Key under which raw conversation turns are exposed.
        reconciled_key: Key under which clean reconciled beliefs are exposed.
        input_key: Key for user prompt in inputs dict.
        output_key: Key for assistant response in outputs dict.
    """

    def __init__(
        self,
        memory_key: str = "history",
        reconciled_key: str = "reconciled_context",
        input_key: str = "input",
        output_key: str = "output",
        device: str = "cpu",
        dummy: bool = False,
    ):
        self.memory_key = memory_key
        self.reconciled_key = reconciled_key
        self.input_key = input_key
        self.output_key = output_key

        self.tracker = OnlineBeliefTracker(device=device, dummy=dummy)
        self.chat_history: List[Dict[str, str]] = []

    @property
    def memory_variables(self) -> List[str]:
        """Exposed keys in loaded memory variables."""
        return [self.memory_key, self.reconciled_key]

    def load_memory_variables(self, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Loads the conversation history along with the reconciled temporal beliefs.
        """
        formatted_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.chat_history])
        reconciled_beliefs = self.tracker.get_context_prompt()

        return {
            self.memory_key: formatted_history,
            self.reconciled_key: reconciled_beliefs,
        }

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> Dict[str, TurnAuditReport]:
        """
        Saves user input and AI output from a dialogue turn, updating the belief graph.

        Returns:
            Dict containing audit reports for both user and AI turns.
        """
        user_text = inputs.get(self.input_key, "")
        ai_text = outputs.get(self.output_key, "")

        reports = {}
        now = datetime.now()

        if user_text:
            self.chat_history.append({"role": "Human", "content": user_text})
            reports["user"] = self.tracker.process_turn("Human", user_text, timestamp=now)

        if ai_text:
            self.chat_history.append({"role": "AI", "content": ai_text})
            reports["ai"] = self.tracker.process_turn("AI", ai_text, timestamp=now)

        return reports

    def clear(self) -> None:
        """Clears both raw conversation history and active belief state."""
        self.chat_history.clear()
        self.tracker = OnlineBeliefTracker(device=self.tracker.device, dummy=self.tracker.dummy)
