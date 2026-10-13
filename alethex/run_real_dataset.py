"""
Runs the ALETHEX pipeline on real-world preprocessed dialogue & memory datasets.
"""

from pathlib import Path

import click
from rich.console import Console

from alethex.config import Config, load_all_configs
from alethex.pipeline import run_pipeline

console = Console()


@click.command()
@click.option(
    "--corpus",
    type=click.Path(exists=True, path_type=Path),
    default=Path("dataset/preprocessed/corpora/benchmark/alethex_corpus_sample.jsonl"),
    help="Path to preprocessed corpus JSONL",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("output_real_dataset"),
    help="Output directory for belief graph and consistency reports",
)
@click.option(
    "--config-dir",
    type=click.Path(exists=True, path_type=Path),
    default=Path("configs"),
    help="Config directory",
)
@click.option("--device", type=click.Choice(["cpu", "cuda", "mps"]), default="cpu")
@click.option("--skip-llm-extraction/--no-skip-llm-extraction", default=True)
@click.option("--skip-llm-judge/--no-skip-llm-judge", default=True)
def main(
    corpus: Path,
    output_dir: Path,
    config_dir: Path,
    device: str,
    skip_llm_extraction: bool,
    skip_llm_judge: bool,
):
    console.print(f"[bold cyan]Launching ALETHEX on real-world corpus:[/bold cyan] {corpus}")
    configs = load_all_configs(config_dir)
    config = Config(config_dir)
    config.configs = configs

    graph, ci = run_pipeline(
        corpus_path=corpus,
        config=config,
        output_dir=output_dir,
        skip_llm_extraction=skip_llm_extraction,
        skip_llm_judge=skip_llm_judge,
        device=device,
    )
    console.print(f"[bold green]ALETHEX run complete! Consistency Index (CI): {ci:.4f}[/bold green]")
    console.print(f"[bold]Results written to:[/bold] {output_dir.resolve()}")


if __name__ == "__main__":
    main()