"""
ALETHEX Training Result Interpretation.

Replaces the previous version of this module, which was found corrupted on
disk (2KB of truncated JSON-escaped text, not valid importable Python) and,
per its still-readable fragment, stamped a hardcoded "Optimal" verdict on
every metric regardless of value -- that is how a 5.4% accuracy / -0.42 MCC
run got reported as "Optimal" / "Superlative Discrimination" in
results/training_3_epochs/training_interpretation_report.md.

This version reads the real Hugging Face Trainer log history
(trainer_state.json, written automatically under the training output_dir)
and reports each metric's status by comparing it against an explicit
threshold -- no metric is ever labelled good without clearing its threshold.
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# metric_key: (comparison, threshold, label)
THRESHOLDS = {
    "eval_accuracy": (">=", 0.60, "3-way NLI accuracy"),
    "eval_macro_f1": (">=", 0.55, "Macro F1"),
    "eval_mcc": (">=", 0.30, "Matthews correlation"),
}


def _status(value: float, comparison: str, threshold: float) -> str:
    passed = value >= threshold if comparison == ">=" else value <= threshold
    return "PASS" if passed else "FAIL"


def find_latest_trainer_state(output_dir: Path) -> Optional[Path]:
    """Returns the newest checkpoint's trainer_state.json that actually parses as JSON.

    Older/interrupted runs can leave a truncated trainer_state.json behind (seen with
    checkpoint-500 from a crashed pre-fix run); skip those rather than crash the report.
    """
    checkpoints = sorted(output_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    for ckpt in reversed(checkpoints):
        state_path = ckpt / "trainer_state.json"
        if not state_path.exists():
            continue
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                json.load(f)
        except json.JSONDecodeError:
            print(f"Skipping unparseable/truncated {state_path} (likely from an interrupted run).")
            continue
        return state_path
    return None


def load_eval_history(trainer_state_path: Path) -> List[Dict[str, Any]]:
    with open(trainer_state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    return [entry for entry in state.get("log_history", []) if "eval_accuracy" in entry]


def build_report(eval_history: List[Dict[str, Any]]) -> str:
    if not eval_history:
        return "# ALETHEX Training Report\n\nNo eval entries found in trainer_state.json.\n"

    best = max(eval_history, key=lambda e: e.get("eval_accuracy", 0.0))

    lines = ["# ALETHEX Training & Evaluation Report", ""]
    lines.append("| Metric | Best Value | Epoch | Threshold | Status |")
    lines.append("|---|---|---|---|---|")
    for key, (comparison, threshold, label) in THRESHOLDS.items():
        value = best.get(key)
        if value is None:
            lines.append(f"| {label} | n/a | - | {comparison} {threshold} | MISSING |")
            continue
        status = _status(value, comparison, threshold)
        lines.append(f"| {label} | {value:.4f} | {best.get('epoch', '?'):.1f} | {comparison} {threshold} | {status} |")

    overall_pass = all(
        _status(best[key], comp, thresh) == "PASS"
        for key, (comp, thresh, _) in THRESHOLDS.items()
        if key in best
    )
    lines.append("")
    lines.append(f"**Overall: {'READY (all thresholds cleared)' if overall_pass else 'NOT READY (one or more thresholds not cleared)'}**")
    lines.append("")
    lines.append(f"Full eval history has {len(eval_history)} epoch(s); best checkpoint shown above.")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/training_latest")
    parser.add_argument("--model-dir", default="models/alethex-nli")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    trainer_state_path = find_latest_trainer_state(model_dir)
    if trainer_state_path is None:
        print(f"No trainer_state.json found under {model_dir}/checkpoint-*/. Run training first.")
        return

    eval_history = load_eval_history(trainer_state_path)
    report = build_report(eval_history)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "training_interpretation_report.md"
    out_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
