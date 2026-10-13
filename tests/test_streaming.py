"""
Tests for OnlineBeliefTracker and turn-by-turn conversational streaming.
"""

from datetime import datetime

from alethex.streaming.online_session import OnlineBeliefTracker, TurnAuditReport


def test_turn_audit_report_summary():
    report = TurnAuditReport(
        turn_id="turn_001",
        speaker="User",
        timestamp=datetime.now(),
        raw_text="Alice works at Google.",
        extracted_claims=[],
        contradictions=[],
        supersessions=[],
        is_consistent=True,
    )
    summary = report.summary()
    assert "No structured claims extracted" in summary


def test_online_belief_tracker_multi_turn_flow():
    # Use dummy=True for rapid, deterministic test execution
    tracker = OnlineBeliefTracker(dummy=True)

    t0 = datetime(2024, 1, 1, 10, 0, 0)
    t1 = datetime(2024, 1, 1, 10, 5, 0)

    # Turn 1: Add employment claim
    rep1 = tracker.process_turn("User", "Alice works at Google.", timestamp=t0)
    assert rep1.turn_id == "turn_001"
    assert rep1.is_consistent is True

    # Turn 2: Add hobby claim
    rep2 = tracker.process_turn("User", "Alice loves Python.", timestamp=t1)
    assert rep2.turn_id == "turn_002"
    assert rep2.is_consistent is True

    active = tracker.get_active_claims()
    assert len(active) >= 1

    prompt = tracker.get_context_prompt()
    assert "Verified Active Beliefs" in prompt
