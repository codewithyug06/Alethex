"""
ALETHEX Command-Line Interface (CLI).
Exposes commands for:
  - run-pipeline: Ingests timestamped corpus, extracts claims, builds belief graph, calculates CI.
  - benchmark: Runs multi-model and end-to-end evaluation suite.
  - stream: Runs interactive or turn-by-turn online belief tracking.
  - version: Displays ALETHEX version and backend capabilities.
  - dashboard: Launches interactive Streamlit analytical web dashboard.
"""

import os
import sys
import warnings

# Suppress noisy external warnings across this process and child subprocesses
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

from datetime import datetime
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from alethex.config import PROJECT_ROOT, default_config_dir, default_output_dir

console = Console()


def resolve_corpus_path(corpus: Optional[Path]) -> Path:
    """Resolves corpus path across CWD and PROJECT_ROOT."""
    if corpus is None:
        target = PROJECT_ROOT / "dataset" / "corpus.jsonl"
        if target.exists():
            return target
        target = Path.cwd() / "dataset" / "corpus.jsonl"
        return target

    if corpus.is_absolute() and corpus.exists():
        return corpus

    # Check CWD
    cwd_path = (Path.cwd() / corpus).resolve()
    if cwd_path.exists():
        return cwd_path

    # Check PROJECT_ROOT
    project_path = (PROJECT_ROOT / corpus).resolve()
    if project_path.exists():
        return project_path

    # Check dataset directory in PROJECT_ROOT
    candidate = (PROJECT_ROOT / "dataset" / corpus.name).resolve()
    if candidate.exists():
        return candidate

    return cwd_path


def resolve_output_dir(output_dir: Optional[Path]) -> Path:
    """Resolves output directory, anchoring relative paths to PROJECT_ROOT when outside."""
    if output_dir is None:
        return PROJECT_ROOT / "output"
    if output_dir.is_absolute():
        return output_dir
    if Path.cwd() == PROJECT_ROOT:
        return (Path.cwd() / output_dir).resolve()
    if (Path.cwd() / output_dir).exists():
        return (Path.cwd() / output_dir).resolve()
    return (PROJECT_ROOT / output_dir).resolve()


def resolve_config_dir(config_dir: Optional[Path]) -> Path:
    """Resolves config directory across CWD and PROJECT_ROOT."""
    if config_dir is None:
        return default_config_dir()
    if config_dir.is_absolute() and config_dir.exists():
        return config_dir
    if (Path.cwd() / config_dir).exists():
        return (Path.cwd() / config_dir).resolve()
    if (PROJECT_ROOT / config_dir).exists():
        return (PROJECT_ROOT / config_dir).resolve()
    return default_config_dir()


@click.group()
def cli():
    """ALETHEX: Temporal Belief Consistency Engine for LLM Memory Systems."""
    pass


@cli.command("version")
def version():
    """Displays version and system architecture information."""
    console.print(
        Panel(
            "[bold cyan]ALETHEX[/bold cyan] v0.1.0\n"
            "[green]Corpus-Scale Temporal Belief Consistency Engine[/green]\n"
            "Backends: DeBERTa-v3 FP32, ONNX Dynamic INT8, ALETHEX-Mini (22.7M)",
            title="System Info",
            border_style="cyan",
        )
    )


@cli.command("run-pipeline")
@click.option(
    "--corpus",
    "-c",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to timestamped corpus JSONL (defaults to dataset/corpus.jsonl).",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory to save belief graph and consistency reports.",
)
@click.option(
    "--config-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to configuration directory.",
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda", "mps"]),
    default="cpu",
    help="Inference compute device.",
)
@click.option("--skip-llm-extraction/--no-skip-llm-extraction", default=True, help="Skip OpenAI LLM extraction fallback.")
@click.option("--skip-llm-judge/--no-skip-llm-judge", default=True, help="Skip OpenAI LLM judge escalation.")
def run_pipeline_cmd(
    corpus: Optional[Path],
    output_dir: Optional[Path],
    config_dir: Optional[Path],
    device: str,
    skip_llm_extraction: bool,
    skip_llm_judge: bool,
):
    """Runs the full ALETHEX belief graph auditing pipeline over a corpus."""
    from alethex.config import Config, load_all_configs
    from alethex.pipeline import run_pipeline

    resolved_corpus = resolve_corpus_path(corpus)
    resolved_output = resolve_output_dir(output_dir)
    resolved_config = resolve_config_dir(config_dir)

    if not resolved_corpus.exists():
        console.print(f"[bold red]Error: Corpus file '{resolved_corpus}' not found.[/bold red]")
        console.print(f"[yellow]Default project corpus expected at: {PROJECT_ROOT / 'dataset' / 'corpus.jsonl'}[/yellow]")
        sys.exit(1)

    console.print(f"[bold cyan]Launching ALETHEX pipeline on:[/bold cyan] {resolved_corpus}")
    configs = load_all_configs(resolved_config)
    cfg = Config(resolved_config)
    cfg.configs = configs

    graph, ci = run_pipeline(
        corpus_path=resolved_corpus,
        config=cfg,
        output_dir=resolved_output,
        skip_llm_extraction=skip_llm_extraction,
        skip_llm_judge=skip_llm_judge,
        device=device,
    )
    console.print(f"[bold green]ALETHEX execution complete! Corpus CI: {ci:.4f}[/bold green]")
    console.print(f"[bold]Artifacts written to:[/bold] {resolved_output}")


@cli.command("benchmark")
@click.option("--samples", type=int, default=150, help="Number of evaluation pairs to evaluate.")
@click.option("--test", is_flag=True, help="Quick smoke test on 30 pairs.")
@click.option("--skip-fp32", is_flag=True, help="Skip heavy FP32 baseline.")
@click.option("--skip-e2e", is_flag=True, help="Skip end-to-end pipeline benchmark.")
@click.option("--output-dir", type=str, default=None, help="Directory for benchmark results.")
def benchmark_cmd(samples: int, test: bool, skip_fp32: bool, skip_e2e: bool, output_dir: Optional[str]):
    """Executes multi-model latency, accuracy, and compression benchmarks."""
    from alethex.evaluation.eval_pipeline import main as run_benchmark

    out_dir = output_dir if output_dir else str(PROJECT_ROOT / "results")
    args = ["eval_pipeline.py", "--output-dir", out_dir]
    if test:
        args.append("--test")
    else:
        args.extend(["--samples", str(samples)])
    if skip_fp32:
        args.append("--skip-fp32")
    if skip_e2e:
        args.append("--skip-e2e")

    sys.argv = args
    run_benchmark()


@cli.command("stream")
@click.option("--speaker", default="user", help="Speaker identifier.")
@click.option("--text", required=True, help="Utterance text to ingest.")
def stream_cmd(speaker: str, text: str):
    """Audits a single conversational turn through the online belief tracker."""
    from alethex.streaming.online_session import OnlineBeliefTracker

    tracker = OnlineBeliefTracker()
    now = datetime.now()
    report = tracker.process_turn(speaker=speaker, text=text, timestamp=now)

    table = Table(title=f"Turn Audit Summary: {speaker}", header_style="bold green")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Turn ID", str(report.turn_id))
    table.add_row("Timestamp", str(report.timestamp))
    table.add_row("Extracted Claims", str(len(report.extracted_claims)))
    table.add_row("Contradictions Detected", str(len(report.contradictions)))
    table.add_row("Superseded Beliefs", str(len(report.supersessions)))
    table.add_row("Current State Consistency Index", f"{tracker.get_current_ci():.4f}")
    table.add_row("Status", "Consistent" if report.is_consistent else "Contradiction Detected")

    console.print(table)


@cli.command("dashboard")
@click.option("--port", default=8501, help="Port to run Streamlit on.")
def dashboard_cmd(port: int):
    """Launches the ALETHEX Streamlit interactive web dashboard."""
    import subprocess
    dashboard_path = PROJECT_ROOT / "alethex" / "dashboard" / "app.py"
    if not dashboard_path.exists():
        dashboard_path = Path(__file__).parent / "dashboard" / "app.py"
    console.print(f"[bold cyan]Launching ALETHEX Web Dashboard on port {port}...[/bold cyan]")
    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "ignore"
    cmd = [sys.executable, "-m", "streamlit", "run", str(dashboard_path), "--server.port", str(port)]
    subprocess.run(cmd, env=env)


@cli.command("mcp")
def mcp_cmd():
    """Runs the ALETHEX Model Context Protocol (MCP) server for Claude Desktop, Claude Code, and Cursor."""
    from alethex.integrations.mcp_server import main as run_mcp
    run_mcp()


@cli.command("proxy")
@click.option("--port", default=8000, help="Proxy port (default: 8000).")
@click.option("--host", default="0.0.0.0", help="Host interface (default: 0.0.0.0).")
@click.option("--upstream", default="http://localhost:11434/v1", help="Upstream LLM base URL (Ollama, OpenAI, Grok, etc.).")
def proxy_cmd(port: int, host: str, upstream: str):
    """Runs universal OpenAI-compatible reverse proxy with ALETHEX reconciliation."""
    from alethex.integrations.openai_proxy import main as run_proxy
    import sys
    sys.argv = ["alethex-proxy", "--port", str(port), "--host", host, "--upstream", upstream]
    run_proxy()


@cli.command("rag-service")
@click.option("--port", default=8080, help="RAG microservice port (default: 8080).")
@click.option("--host", default="0.0.0.0", help="Host interface (default: 0.0.0.0).")
def rag_service_cmd(port: int, host: str):
    """Runs universal language-agnostic RAG reconciliation microservice."""
    from alethex.integrations.rag_service import main as run_rag_svc
    import sys
    sys.argv = ["alethex-rag-service", "--port", str(port), "--host", host]
    run_rag_svc()


@cli.command("attach")
def attach_cmd():
    """Scans system for AI applications (Claude, Cursor, Ollama) and auto-attaches ALETHEX."""
    from alethex.integrations.auto_attach import main as run_auto_attach
    run_auto_attach()


def main():
    cli()



if __name__ == "__main__":
    main()
