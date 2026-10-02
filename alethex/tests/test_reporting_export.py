from alethex.reporting.export import export_html, export_markdown
from alethex.scoring.consistency_index import ConsistencyReport


def test_export_markdown_writes_report_with_expected_sections(tmp_path):
    report = ConsistencyReport(
        corpus_ci=0.85,
        total_entities=2,
        total_claims=10,
        total_conflict_pairs=1,
        entity_cis={"user_1": 1.0, "user_2": 0.7},
    )
    contradictions = [
        {
            "claim_1": {"subject": "user", "predicate": "lives in", "object_": "NY", "source_id": "s1"},
            "claim_2": {"subject": "user", "predicate": "lives in", "object_": "CA", "source_id": "s2"},
            "confidence": 0.95,
            "justification": "Mutually exclusive locations",
        }
    ]
    stale_claims = [
        {"subject": "user", "predicate": "works at", "object_": "Startup", "validity_end": "2024-01-01"}
    ]
    out_file = tmp_path / "report.md"
    md = export_markdown(report, contradictions, stale_claims, output_path=out_file)
    assert out_file.exists()
    assert "Corpus Consistency Index" in md
    assert "0.8500" in md
    assert "Contradiction 1" in md
    assert "Startup" in md


def test_export_markdown_empty_contradictions_and_stale(tmp_path):
    report = ConsistencyReport(
        corpus_ci=1.0,
        total_entities=1,
        total_claims=5,
        total_conflict_pairs=0,
        entity_cis={"user_1": 1.0},
    )
    out_file = tmp_path / "clean_report.md"
    md = export_markdown(report, [], [], output_path=out_file)
    assert "No unresolved contradictions found." in md
    assert "No stale claims found." in md


def test_export_html_writes_self_contained_page(tmp_path):
    report = ConsistencyReport(
        corpus_ci=0.9,
        total_entities=1,
        total_claims=10,
        total_conflict_pairs=1,
        entity_cis={"user_1": 0.9},
    )
    contradictions = [
        {
            "claim_1": {"subject": "user", "predicate": "role", "object_": "Dev", "source_id": "s1"},
            "claim_2": {"subject": "user", "predicate": "role", "object_": "PM", "source_id": "s2"},
            "confidence": 0.9,
            "justification": "Role conflict",
        }
    ]
    out_file = tmp_path / "report.html"
    html = export_html(report, contradictions, [], output_path=out_file)
    assert out_file.exists()
    assert "<!DOCTYPE html>" in html
    assert "ALETHEX Belief Consistency Report" in html
    assert "badge-conflict" in html


def test_export_html_no_contradictions_shows_clean_state(tmp_path):
    report = ConsistencyReport(
        corpus_ci=1.0,
        total_entities=1,
        total_claims=5,
        total_conflict_pairs=0,
        entity_cis={"user_1": 1.0},
    )
    out_file = tmp_path / "clean_report.html"
    html = export_html(report, [], [], output_path=out_file)
    assert "badge-clean" in html
    assert "Clean State" in html
