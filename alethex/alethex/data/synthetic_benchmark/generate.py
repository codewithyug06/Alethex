import json
import random
import uuid
from datetime import datetime, timedelta
from typing import List
import os

def get_sentence_variations(subject: str, predicate: str, object_: str) -> List[str]:
    """
    Simulates LLM-generated paraphrases of a fact.
    In a full production run, this would call OpenAI/Anthropic APIs.
    """
    return [
        f"{subject} {predicate} {object_}.",
        f"I think {subject} {predicate} {object_}.",
        f"As far as I know, {subject} {predicate} {object_}.",
        f"It is a known fact that {subject} {predicate} {object_}."
    ]

SCALE_MAP = {
    "mini": 3,
    "small": 10,
    "medium": 50,
    "large": 200,
}

def generate_benchmark(num_entities: int = 50, output_dir: str = "data/synthetic_benchmark", scale: str = None):
    if scale is not None:
        num_entities = SCALE_MAP.get(scale, num_entities)
    os.makedirs(output_dir, exist_ok=True)
    
    corpus_file = os.path.join(output_dir, "corpus.jsonl")
    gold_labels_file = os.path.join(output_dir, "gold_labels.jsonl")
    
    attributes = ["lives in", "works at", "likes", "uses"]
    values = {
        "lives in": ["NY", "CA", "TX", "London", "Tokyo"],
        "works at": ["Google", "Apple", "Microsoft", "Amazon", "Meta"],
        "likes": ["pizza", "sushi", "burgers", "tacos", "pasta"],
        "uses": ["Mac", "Windows", "Linux", "iOS", "Android"]
    }
    
    documents = []
    gold_labels = []
    
    random.seed(42) # Deterministic for tests
    
    for i in range(num_entities):
        entity_name = f"User_{i}"
        
        num_attrs = random.randint(1, 3)
        chosen_attrs = random.sample(attributes, num_attrs)
        
        for attr in chosen_attrs:
            num_facts = random.randint(1, 3)
            current_date = datetime(2023, 1, 1)
            attr_claims = []
            
            for j in range(num_facts):
                val = random.choice(values[attr])
                duration_days = random.randint(30, 300)
                end_date = current_date + timedelta(days=duration_days)
                
                # 15% adversarial conflict
                if random.random() < 0.15:
                    conflict_val = random.choice([v for v in values[attr] if v != val])
                    conflict_date = current_date + timedelta(days=random.randint(1, max(1, duration_days-1)))
                    
                    sentences = get_sentence_variations(entity_name, attr, conflict_val)
                    sent = random.choice(sentences)
                    
                    doc_id = str(uuid.uuid4())
                    documents.append({
                        "source_id": doc_id,
                        "timestamp": conflict_date.isoformat(),
                        "text": sent
                    })
                    claim_id = str(uuid.uuid4())
                    attr_claims.append({
                        "claim_id": claim_id,
                        "entity": entity_name,
                        "predicate": attr,
                        "object": conflict_val,
                        "timestamp": conflict_date,
                        "validity_start": conflict_date,
                        "validity_end": end_date
                    })
                
                sentences = get_sentence_variations(entity_name, attr, val)
                sent = random.choice(sentences)
                doc_id = str(uuid.uuid4())
                documents.append({
                    "source_id": doc_id,
                    "timestamp": current_date.isoformat(),
                    "text": sent
                })
                
                claim_id = str(uuid.uuid4())
                attr_claims.append({
                    "claim_id": claim_id,
                    "entity": entity_name,
                    "predicate": attr,
                    "object": val,
                    "timestamp": current_date,
                    "validity_start": current_date,
                    "validity_end": end_date if j < num_facts - 1 else None
                })
                
                current_date = end_date
                
            for idx in range(len(attr_claims)):
                for jdx in range(idx + 1, len(attr_claims)):
                    c1 = attr_claims[idx]
                    c2 = attr_claims[jdx]
                    
                    if c1["object"] == c2["object"]:
                        relation = "consistent"
                    else:
                        c1_end = c1["validity_end"] or datetime.max
                        c2_start = c2["validity_start"]
                        if c2_start < c1_end:
                            relation = "conflicting"
                        else:
                            relation = "superseded"
                            
                    gold_labels.append({
                        "claim_a_id": c1["claim_id"],
                        "claim_b_id": c2["claim_id"],
                        "entity": entity_name,
                        "predicate": attr,
                        "true_relation": relation
                    })

    with open(corpus_file, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc) + "\n")
            
    with open(gold_labels_file, "w", encoding="utf-8") as f:
        for label in gold_labels:
            f.write(json.dumps(label) + "\n")

if __name__ == "__main__":
    generate_benchmark()