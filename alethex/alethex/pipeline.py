import json
import sys
from pathlib import Path
from typing import Optional, Union, Tuple, List, Any

import click
from loguru import logger
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn
from rich.table import Table

from alethex.api import ConsistencyEngine
from alethex.config import Config, default_config_dir, default_output_dir, load_all_configs
from alethex.data.synthetic_benchmark.generate import generate_benchmark
from alethex.extraction.llm_extractor import LLMExtractor
from alethex.graph.belief_graph import BeliefGraph
from alethex.ingestion.loaders import load_corpus
from alethex.relation.llm_judge import LLMJudge
from alethex.scoring import consistency_index, contradiction_report, staleness_report

console = Console(force_terminal=True, legacy_windows=True)

# Configure loguru - avoid unicode chars for Windows compatibility
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}")


@click.command()
@click.option("--corpus", type=click.Path(exists=True, path_type=Path), help="Path to corpus.jsonl")
@click.option("--scale", type=click.Choice(["mini", "small", "medium", "large"]), default="mini", help="Benchmark scale for synthetic generation")
@click.option("--generate/--no-generate", default=False, help="Generate synthetic benchmark first")
@click.option("--skip-llm-extraction/--no-skip-llm-extraction", default=True, help="Skip LLM extraction (use OpenIE only)")
@click.option("--skip-llm-judge/--no-skip-llm-judge", default=True, help="Skip LLM judge (use NLI only)")
@click.option("--device", type=click.Choice(["cpu", "cuda", "mps"]), default="cpu", help="Device for NLI model")
@click.option("--output-dir", type=click.Path(path_type=Path), default=None, help="Output directory for artifacts (default: $ALETHEX_OUTPUT_DIR or <repo>/output)")
@click.option("--config-dir", type=click.Path(exists=True, path_type=Path), default=None, help="Config directory (default: $ALETHEX_CONFIG_DIR or <repo>/configs)")
def main(
    corpus: Optional[Path],
    scale: str,
    generate: bool,
    skip_llm_extraction: bool,
    skip_llm_judge: bool,
    device: str,
    output_dir: Optional[Path],
    config_dir: Optional[Path],
):
    """
    ALETHEX Temporal Belief Consistency Engine

    Extracts factual claims from text, links entities, resolves temporal validity intervals,
    detects contradictions/supersessions, and scores belief consistency over time.
    """
    output_dir = output_dir or default_output_dir()
    config_dir = config_dir or default_config_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load configs
    logger.info(f"Loading configs from {config_dir}")
    configs = load_all_configs(config_dir)
    config = Config(config_dir)
    config.configs = configs

    corpus_path = corpus

    if generate or corpus_path is None:
        logger.info(f"Generating synthetic benchmark ({scale})...")
        benchmark_dir = output_dir / "synthetic_benchmark"
        generate_benchmark(scale=scale, output_dir=str(benchmark_dir))
        if corpus_path is None:
            corpus_path = benchmark_dir / "corpus.jsonl"

    if corpus_path is None:
        logger.error("Error: You must specify a --corpus or use --generate")
        sys.exit(1)

    run_pipeline(
        corpus_path=corpus_path,
        config=config,
        output_dir=output_dir,
        skip_llm_extraction=skip_llm_extraction,
        skip_llm_judge=skip_llm_judge,
        device=device,
    )


def run_pipeline(
    corpus_path: Union[str, Path, List[Any]],
    config: Optional[Config] = None,
    output_dir: Optional[Union[str, Path]] = None,
    skip_llm_extraction: bool = True,
    skip_llm_judge: bool = True,
    device: str = "cpu",
    dummy_ml: bool = False,
) -> Tuple[BeliefGraph, float]:
    """Run the full ALETHEX pipeline."""
    if output_dir is None:
        output_dir = default_output_dir()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config is None:
        cfg_dir = default_config_dir()
        config = Config(cfg_dir)
        config.configs = load_all_configs(cfg_dir) if cfg_dir.exists() else {}

    engine = ConsistencyEngine(config=config, device=device)
    if dummy_ml:
        engine.nli.dummy = True
        engine.nli.classifier = None

    llm_extractor = LLMExtractor(config) if not skip_llm_extraction else None
    llm_judge = LLMJudge(config) if not skip_llm_judge else None

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        # Task 1: Load corpus
        task_load = progress.add_task("Loading corpus...", total=None)
        if isinstance(corpus_path, list):
            docs = corpus_path
        else:
            docs = list(load_corpus(str(corpus_path)))
        progress.update(task_load, completed=True, description=f"Loaded {len(docs)} documents")

        # Tasks 2-7: Extraction, coref, entity linking, temporal resolution, pairing,
        # NLI classification, optional LLM judge, and belief graph construction.
        task_run = progress.add_task("Running core pipeline (extract -> merge -> coref -> link -> temporal -> NLI -> judge -> graph)...", total=None)
        if not skip_llm_extraction:
            logger.info("Running LLM extraction...")
        claims, entities, pairs, classified_pairs, graph = engine.run_core_pipeline(
            docs,
            llm_extractor=llm_extractor,
            llm_judge=llm_judge,
        )
        progress.update(
            task_run,
            completed=True,
            description=(
                f"Extracted {len(claims)} claims, {len(entities)} entities, "
                f"{len(pairs)} pairs, belief graph built"
            ),
        )
        logger.info(f"Extracted {len(claims)} unique claims")
        logger.info("Within-doc coreference resolved")
        logger.info(f"Discovered {len(entities)} canonical entities")
        logger.info("Temporal intervals resolved")
        logger.info(f"Generated {len(pairs)} claim pairs")
        logger.info("NLI classification complete")
        if not skip_llm_judge:
            logger.info("LLM judge complete")
        logger.info("Belief graph built")

        # Save claims artifact
        claims_file = output_dir / "claims.jsonl"
        with open(claims_file, "w", encoding="utf-8") as f:
            for c in claims:
                f.write(c.model_dump_json() + "\n")
        logger.info(f"Saved claims to {claims_file}")

        # Save entities artifact
        entities_file = output_dir / "entities.json"
        with open(entities_file, "w", encoding="utf-8") as f:
            json.dump({eid: e.model_dump() for eid, e in (entities.items() if isinstance(entities, dict) else {getattr(e, 'entity_id', str(i)): e for i, e in enumerate(entities)}).items()}, f, indent=2, default=str)
        logger.info(f"Saved entities to {entities_file}")

        # Save claims with intervals
        claims_intervals_file = output_dir / "claims_with_intervals.jsonl"
        with open(claims_intervals_file, "w", encoding="utf-8") as f:
            for c in claims:
                f.write(c.model_dump_json() + "\n")
        logger.info(f"Saved claims with intervals to {claims_intervals_file}")

        # Save pairs artifact
        pairs_file = output_dir / "claim_pairs.jsonl"
        with open(pairs_file, "w", encoding="utf-8") as f:
            for p in classified_pairs:
                f.write(p.model_dump_json() + "\n")
        logger.info(f"Saved claim pairs to {pairs_file}")

        # Save graph artifact
        graph_file = output_dir / "belief_graph.graphml"
        graph.save(str(graph_file))
        logger.info(f"Saved belief graph to {graph_file}")

        # Task 8: Calculating Consistency Index
        task_ci = progress.add_task("Calculating consistency index...", total=None)
        ci_report = consistency_index.calculate_ci(graph)
        progress.update(task_ci, completed=True, description=f"Corpus CI: {ci_report.corpus_ci:.4f}")
        logger.info(f"Corpus CI: {ci_report.corpus_ci:.4f}")

        # Task 9: Contradiction Report
        task_contradiction = progress.add_task("Generating contradiction report...", total=None)
        contradictions = contradiction_report.generate_report(graph, config)
        progress.update(task_contradiction, completed=True, description=f"Found {len(contradictions)} contradictions")
        logger.info(f"Found {len(contradictions)} contradictions")

        # Save contradiction report
        contrad_file = output_dir / "contradictions.json"
        with open(contrad_file, "w", encoding="utf-8") as f:
            json.dump(contradictions, f, indent=2, default=str)
        logger.info(f"Saved contradictions to {contrad_file}")

        # Task 10: Staleness Report
        task_staleness = progress.add_task("Generating staleness report...", total=None)
        stale_claims = staleness_report.generate_report(graph, config)
        progress.update(task_staleness, completed=True, description=f"Found {len(stale_claims)} stale claims")
        logger.info(f"Found {len(stale_claims)} stale claims")

        # Save staleness report
        staleness_file = output_dir / "staleness.json"
        with open(staleness_file, "w", encoding="utf-8") as f:
            json.dump(stale_claims, f, indent=2, default=str)
        logger.info(f"Saved staleness report to {staleness_file}")

        # Summary Table
        summary = Table(title="ALETHEX Pipeline Summary", show_header=True, header_style="bold magenta")
        summary.add_column("Metric", style="cyan")
        summary.add_column("Value", style="green")
        summary.add_row("Documents Processed", str(len(docs)))
        summary.add_row("Total Claims", str(len(claims)))
        summary.add_row("Canonical Entities", str(len(entities)))
        summary.add_row("Claim Pairs", str(len(pairs)))
        summary.add_row("Conflicts", str(ci_report.total_conflict_pairs))
        summary.add_row("Corpus CI", f"{ci_report.corpus_ci:.4f}")
        summary.add_row("Contradictions (top-n)", str(len(contradictions)))
        summary.add_row("Stale Claims", str(len(stale_claims)))
        summary.add_row("Output Directory", str(output_dir))
        console.print(summary)

    return graph, ci_report.corpus_ci


if __name__ == "__main__":
    main()