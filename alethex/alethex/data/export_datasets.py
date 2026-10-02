from datasets import load_dataset
import os
from rich.console import Console

console = Console()

def main():
    datasets = [
        ("nyu-mll/multi_nli", None),
        ("stanfordnlp/snli", None),
        ("tasksource/temporal-nli", None),
        ("sentence-transformers/all-nli", None),
        ("jihyoung/ConversationChronicles", None),
        ("Mohammadta/BEAM", None),
        ("xiaowu0162/longmemeval-cleaned", "longmemeval_s")
    ]
    
    export_dir = "dataset"
    os.makedirs(export_dir, exist_ok=True)
    
    for ds_name, config in datasets:
        console.print(f"[bold blue]Exporting {ds_name}...[/bold blue]")
        try:
            if config:
                ds = load_dataset(ds_name, config)
            else:
                ds = load_dataset(ds_name)
                
            safe_name = ds_name.replace("/", "_")
            ds_dir = os.path.join(export_dir, safe_name)
            os.makedirs(ds_dir, exist_ok=True)
            
            for split, split_ds in ds.items():
                out_path = os.path.join(ds_dir, f"{split}.jsonl")
                console.print(f"  Saving {split} to {out_path}...")
                split_ds.to_json(out_path)
            
            console.print(f"[bold green]Successfully exported {ds_name}[/bold green]")
        except Exception as e:
            console.print(f"[bold red]Failed to export {ds_name}: {e}[/bold red]")

if __name__ == "__main__":
    main()