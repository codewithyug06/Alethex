from datetime import datetime
from typing import Optional, Tuple
import dateparser
from dateparser.search import search_dates

class TimexTagger:
    def __init__(self):
        pass

    def extract_time(self, text: str, reference_date: datetime) -> Optional[Tuple[str, datetime]]:
        """
        Extracts the first found temporal expression from text and resolves it.
        """
        results = search_dates(text, settings={'RELATIVE_BASE': reference_date})
        if results:
            return results[0]
        return None
