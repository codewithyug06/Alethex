"""Smoke test for the run_real_dataset CLI entry point."""

from click.testing import CliRunner

from alethex.data.synthetic_benchmark.generate import generate_benchmark
from alethex.run_real_dataset import main


def test_run_real_dataset_cli_smoke(tmp_path):
    corpus_dir = tmp_path / "corpus_src"
    generate_benchmark(num_entities=3, output_dir=str(corpus_dir))
    corpus_file = corpus_dir / "corpus.jsonl"
    assert corpus_file.exists()

    output_dir = tmp_path / "output"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--corpus",
            str(corpus_file),
            "--output-dir",
            str(output_dir),
            "--config-dir",
            "configs",
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Consistency Index" in result.output
    assert (output_dir / "belief_graph.graphml").exists()

