import spacy
from typing import List
from uuid import uuid4
from alethex.ingestion.schema import Claim, RawDocument

class OpenIEExtractor:
    def __init__(self, model: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(model)
        except OSError:
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "spacy", "download", model])
            self.nlp = spacy.load(model)
            
    def extract(self, doc: RawDocument) -> List[Claim]:
        spacy_doc = self.nlp(doc.text)
        claims = []
        for sent in spacy_doc.sents:
            root = sent.root
            if root.pos_ not in ("VERB", "AUX"):
                continue
                
            subjects = [w for w in root.lefts if w.dep_ in ("nsubj", "nsubjpass")]
            if not subjects:
                continue
            subj = subjects[0]
            
            obj = None
            for w in root.rights:
                if w.dep_ in ("dobj", "attr", "acomp"):
                    obj = w
                    break
                elif w.dep_ == "prep":
                    pobjs = [p for p in w.rights if p.dep_ == "pobj"]
                    if pobjs:
                        obj = pobjs[0]
                        break
                        
            if subj and obj:
                claim = Claim(
                    claim_id=str(uuid4()),
                    source_id=doc.source_id,
                    timestamp=doc.timestamp,
                    subject=subj.lemma_.lower(),
                    predicate=root.lemma_.lower(),
                    object_=obj.lemma_.lower(),
                    confidence=0.8,
                    raw_sentence=sent.text.strip(),
                    validity_start=doc.timestamp,
                    validity_end=None
                )
                claims.append(claim)
        return claims