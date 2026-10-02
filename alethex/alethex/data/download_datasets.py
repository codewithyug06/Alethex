from datasets import load_dataset
from rich.console import Console

console = Console()

def main():
    datasets = [
        ("nyu-mll/multi_nli", None),
        ("stanfordnlp/snli", None),
        ("tasksource/temporal-nli", None),
        ("snap-research/LoCoMo", None),
        ("sentence-transformers/all-nli", None),
        ("jihyoung/ConversationChronicles", None),
        ("Mohammadta/BEAM", None),
        ("xiaowu0162/longmemeval-cleaned", "longmemeval_s")
    ]
    
    for ds_name, config in datasets:
        console.print(f"[bold blue]Downloading {ds_name}...[/bold blue]")
        try:
            if config:
                load_dataset(ds_name, config)
            else:
                load_dataset(ds_name)
            console.print(f"[bold green]Successfully downloaded {ds_name}[/bold green]")
        except Exception as e:
            console.print(f"[bold red]Failed to download {ds_name}: {e}[/bold red]")

if __name__ == "__main__":
    main()