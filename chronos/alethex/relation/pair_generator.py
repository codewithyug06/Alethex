from typing import List, Iterator, Tuple, Any, Optional
from alethex.ingestion.schema import Claim


class PairGenerator:
    def __init__(self, time_window_seconds: Any = 60):
        if hasattr(time_window_seconds, "time_window_seconds"):
            self.time_window = getattr(time_window_seconds, "time_window_seconds", 60)
        elif isinstance(time_window_seconds, (int, float)):
            self.time_window = time_window_seconds
        else:
            self.time_window = 60

    def generate(self, *args, **kwargs) -> Iterator[Tuple[Claim, Claim]]:
        """
        Supports either generate(claims) or generate(entities, claims).
        """
        claims: List[Claim] = []
        if len(args) == 1:
            claims = args[0]
        elif len(args) >= 2:
            claims = args[1]
        elif "claims" in kwargs:
            claims = kwargs["claims"]

        groups = {}
        for c in claims:
            key = f"{c.subject}::{c.predicate}"
            if key not in groups:
                groups[key] = []
            groups[key].append(c)

        for key, group in groups.items():
            group.sort(key=lambda x: x.timestamp)
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    c1 = group[i]
                    c2 = group[j]

                    if c1.source_id == c2.source_id:
                        delta = (c2.timestamp - c1.timestamp).total_seconds()
                        if abs(delta) <= self.time_window:
                            continue

                    yield (c1, c2)
