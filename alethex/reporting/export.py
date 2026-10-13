from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from alethex.graph.belief_graph import BeliefGraph
from alethex.scoring.consistency_index import ConsistencyReport


def export_markdown(
    ci_report: ConsistencyReport,
    contradictions: List[Dict[str, Any]],
    stale_claims: List[Dict[str, Any]],
    output_path: Optional[Union[str, Path]] = None,
) -> str:
    lines = [
        "# ALETHEX Belief Consistency Report",
        "",
        f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Executive Summary",
        "",
        f"- **Corpus Consistency Index (CI):** `{ci_report.corpus_ci:.4f}`",
        f"- **Total Entities:** `{ci_report.total_entities}`",
        f"- **Total Claims:** `{ci_report.total_claims}`",
        f"- **Conflict Pairs:** `{ci_report.total_conflict_pairs}`",
        "",
        "## Entity Consistency Scores",
        "",
        "| Entity | Consistency Index |",
        "| :--- | :---: |",
    ]
    for ent, ci in ci_report.entity_cis.items():
        lines.append(f"| {ent} | {ci:.4f} |")

    lines.extend([
        "",
        f"## Contradictions ({len(contradictions)})",
        "",
    ])
    if contradictions:
        for i, c in enumerate(contradictions, 1):
            c1 = c.get("claim_1", {})
            c2 = c.get("claim_2", {})
            conf = c.get("confidence", 0.0)
            just = c.get("justification", "")
            lines.extend([
                f"### Contradiction {i}",
                f"- **Claim A:** {c1.get('subject')} {c1.get('predicate')} {c1.get('object_')} (from {c1.get('source_id')})",
                f"- **Claim B:** {c2.get('subject')} {c2.get('predicate')} {c2.get('object_')} (from {c2.get('source_id')})",
                f"- **Confidence:** {conf:.2f}",
                f"- **Justification:** {just}",
                "",
            ])
    else:
        lines.append("No unresolved contradictions found.")

    lines.extend([
        "",
        f"## Stale Claims ({len(stale_claims)})",
        "",
    ])
    if stale_claims:
        for s in stale_claims:
            lines.append(f"- **{s.get('subject')} {s.get('predicate')} {s.get('object_')}**: expired on {s.get('validity_end')}")
    else:
        lines.append("No stale claims found.")

    md_content = "\n".join(lines) + "\n"
    if output_path:
        Path(output_path).write_text(md_content, encoding="utf-8")
    return md_content


def export_html(
    ci_report: ConsistencyReport,
    contradictions: List[Dict[str, Any]],
    stale_claims: List[Dict[str, Any]],
    output_path: Optional[Union[str, Path]] = None,
) -> str:
    md = export_markdown(ci_report, contradictions, stale_claims)
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>ALETHEX Belief Consistency Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; max-width: 900px; margin: 40px auto; padding: 0 20px; color: #1e293b; background: #f8fafc; }}
h1, h2, h3 {{ color: #0f172a; }}
table {{ border-collapse: collapse; width: 100%; margin: 20px 0; background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
th, td {{ border: 1px solid #e2e8f0; padding: 12px 16px; text-align: left; }}
th {{ background: #f1f5f9; font-weight: 600; }}
pre {{ background: #1e293b; color: #f8fafc; padding: 16px; border-radius: 8px; overflow-x: auto; }}
code {{ font-family: monospace; background: #f1f5f9; padding: 2px 6px; border-radius: 4px; }}
.badge {{ display: inline-block; padding: 4px 8px; border-radius: 4px; font-weight: 600; font-size: 12px; }}
.badge-clean {{ background: #dcfce7; color: #15803d; }}
.badge-conflict {{ background: #fee2e2; color: #b91c1c; }}
</style>
</head>
<body>
<h1>ALETHEX Belief Consistency Report</h1>
<p><strong>Generated on:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
<h2>Executive Summary</h2>
<ul>
<li><strong>Corpus Consistency Index:</strong> <code>{ci_report.corpus_ci:.4f}</code></li>
<li><strong>Total Entities:</strong> <code>{ci_report.total_entities}</code></li>
<li><strong>Total Claims:</strong> <code>{ci_report.total_claims}</code></li>
<li><strong>Conflict Pairs:</strong> <code>{ci_report.total_conflict_pairs}</code></li>
</ul>
<h2>Status</h2>
<p><span class="badge {'badge-clean' if len(contradictions) == 0 else 'badge-conflict'}">{'Clean State: No Contradictions' if len(contradictions) == 0 else f'{len(contradictions)} Contradictions Detected'}</span></p>
<hr>
<pre>{md}</pre>
</body>
</html>"""
    if output_path:
        Path(output_path).write_text(html, encoding="utf-8")
    return html
