import json
import re
from pathlib import Path
from datetime import datetime
from typing import Iterator, Union
from dateutil import parser
from alethex.ingestion.schema import RawDocument

def load_json(filepath: Path) -> Iterator[RawDocument]:
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, list):
            for i, item in enumerate(data):
                yield RawDocument(
                    source_id=item.get("source_id", f"{filepath.name}_{i}"),
                    timestamp=parser.parse(item["timestamp"]),
                    text=item["text"]
                )
        else:
            yield RawDocument(
                source_id=data.get("source_id", filepath.name),
                timestamp=parser.parse(data["timestamp"]),
                text=data["text"]
            )

def load_jsonl(filepath: Path) -> Iterator[RawDocument]:
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            yield RawDocument(
                source_id=item.get("source_id", f"{filepath.name}_{i}"),
                timestamp=parser.parse(item["timestamp"]),
                text=item["text"]
            )

def load_txt(filepath: Path) -> Iterator[RawDocument]:
    # Matches formats like "[2026-06-01T12:00:00] text" or "2026-06-01 text"
    timestamp_pattern = re.compile(r"^\[?(?P<timestamp>\d{4}[-/]\d{2}[-/]\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)\]?\s+(?P<text>.*)")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            match = timestamp_pattern.match(line)
            if match:
                timestamp_str = match.group("timestamp")
                text = match.group("text")
                try:
                    dt = parser.parse(timestamp_str)
                except ValueError:
                    dt = datetime.now()
            else:
                dt = datetime.now()
                text = line.strip()
            
            yield RawDocument(
                source_id=f"{filepath.name}_{i}",
                timestamp=dt,
                text=text
            )

def _load_file(filepath: Path) -> Iterator[RawDocument]:
    if filepath.suffix == ".jsonl":
        yield from load_jsonl(filepath)
    elif filepath.suffix == ".json":
        yield from load_json(filepath)
    elif filepath.suffix == ".txt":
        yield from load_txt(filepath)

def load_corpus(path: Union[str, Path]) -> Iterator[RawDocument]:
    """Loads RawDocuments from a file or directory."""
    path = Path(path)
    if path.is_dir():
        for filepath in path.rglob("*"):
            if filepath.is_file():
                yield from _load_file(filepath)
    else:
        yield from _load_file(path)
