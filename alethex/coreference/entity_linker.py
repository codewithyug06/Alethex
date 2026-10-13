import string
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
import numpy as np
from uuid import uuid4

from alethex.ingestion.schema import Claim, EntityNode

class EntityLinker:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        eps: float = 0.25,
        min_samples: int = 2,
        device: Optional[str] = None,
    ):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        self.eps = eps
        self.min_samples = min_samples

    def _normalize(self, text: str) -> str:
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation)).strip()
        for prefix in ("the ", "a ", "an "):
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        return text

    def link(self, claims: List[Claim]) -> Tuple[List[Claim], List[EntityNode]]:
        mentions = set()
        for c in claims:
            if c.subject: mentions.add(c.subject)
            if c.object_: mentions.add(c.object_)
            
        mentions_list = list(mentions)
        if not mentions_list:
            return claims, []
            
        # 1. Hard-merge exact strings after normalization
        norm_to_mentions = {}
        for m in mentions_list:
            norm = self._normalize(m)
            if norm not in norm_to_mentions:
                norm_to_mentions[norm] = []
            norm_to_mentions[norm].append(m)
            
        canonical_candidates = list(norm_to_mentions.keys())
        
        # 2. Cluster using DBSCAN
        embeddings = self.model.encode(canonical_candidates)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric="euclidean").fit(embeddings)
        
        # 3. Create EntityNodes and mapping
        cluster_to_entity = {} # cluster_id -> EntityNode
        mention_to_entity_id = {}
        
        for idx, cluster_id in enumerate(clustering.labels_):
            norm_mention = canonical_candidates[idx]
            original_mentions = norm_to_mentions[norm_mention]
            
            if cluster_id == -1: # Noise / unclustered
                entity_id = str(uuid4())
                node = EntityNode(
                    entity_id=entity_id,
                    canonical_name=original_mentions[0],
                    aliases=original_mentions
                )
                cluster_to_entity[entity_id] = node
                for om in original_mentions:
                    mention_to_entity_id[om] = entity_id
            else:
                if cluster_id not in cluster_to_entity:
                    entity_id = str(uuid4())
                    cluster_to_entity[cluster_id] = EntityNode(
                        entity_id=entity_id,
                        canonical_name=original_mentions[0],
                        aliases=[]
                    )
                node = cluster_to_entity[cluster_id]
                node.aliases.extend(original_mentions)
                for om in original_mentions:
                    mention_to_entity_id[om] = node.entity_id

        final_nodes = []
        for key, node in cluster_to_entity.items():
            node.aliases = list(set(node.aliases))
            final_nodes.append(node)

        # 4. Update claims
        for c in claims:
            if c.subject in mention_to_entity_id:
                c.subject = self._get_canonical_name(c.subject, mention_to_entity_id, final_nodes)
            if c.object_ in mention_to_entity_id:
                c.object_ = self._get_canonical_name(c.object_, mention_to_entity_id, final_nodes)
                
        return claims, final_nodes
        
    def _get_canonical_name(self, mention: str, mapping: Dict[str, str], nodes: List[EntityNode]) -> str:
        ent_id = mapping.get(mention)
        if not ent_id: return mention
        for n in nodes:
            if n.entity_id == ent_id:
                return n.canonical_name
        return mention
