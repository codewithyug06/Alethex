from datetime import datetime
import networkx as nx
from typing import List, Dict, Optional
from alethex.ingestion.schema import Claim, EntityNode, ClaimPair


class BeliefGraph:
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.entities: Dict[str, EntityNode] = {}

    def build(self, nodes: List[EntityNode], claims: List[Claim], pairs: List[ClaimPair]):
        for node in nodes:
            self.entities[node.entity_id] = node
            self.graph.add_node(node.entity_id, type="entity", canonical_name=node.canonical_name)

        for claim in claims:
            ent_id = self._get_entity_id_by_name(claim.subject)
            if not ent_id:
                ent_id = claim.subject
                self.graph.add_node(ent_id, type="entity", canonical_name=claim.subject)

            value_node = f"val::{claim.object_}"
            self.graph.add_node(value_node, type="value", label=claim.object_)

            self.graph.add_edge(
                ent_id,
                value_node,
                key=claim.claim_id,
                predicate=claim.predicate,
                validity_start=claim.validity_start,
                validity_end=claim.validity_end,
                confidence=claim.confidence,
                source_id=claim.source_id,
                claim_obj=claim
            )

        for pair in pairs:
            if pair.relation == "conflicting":
                val1 = f"val::{pair.claim_a.object_}"
                val2 = f"val::{pair.claim_b.object_}"

                if self.graph.has_node(val1) and self.graph.has_node(val2):
                    self.graph.add_edge(val1, val2, type="conflict", style="dashed")
                    self.graph.add_edge(val2, val1, type="conflict", style="dashed")

    def _get_entity_id_by_name(self, name: str) -> Optional[str]:
        for ent_id, node in self.entities.items():
            if node.canonical_name == name:
                return ent_id
        return None

    def get_current_valid_claims(self, entity_id: str) -> List[Claim]:
        valid_claims = []
        now = datetime.now()
        if not self.graph.has_node(entity_id):
            return valid_claims

        for _, _, data in self.graph.edges(entity_id, data=True):
            claim = data.get("claim_obj")
            if claim:
                if claim.validity_end is None or claim.validity_end > now:
                    valid_claims.append(claim)
        return valid_claims

    def save(self, filepath: str):
        export_g = nx.MultiDiGraph()
        for node, data in self.graph.nodes(data=True):
            clean_data = {k: str(v) if not isinstance(v, (int, float, bool, str)) else v for k, v in data.items()}
            export_g.add_node(node, **clean_data)
        for u, v, k, data in self.graph.edges(keys=True, data=True):
            clean_data = {
                k_attr: str(v_attr) if not isinstance(v_attr, (int, float, bool, str)) else v_attr
                for k_attr, v_attr in data.items()
                if k_attr != "claim_obj"
            }
            export_g.add_edge(u, v, key=str(k), **clean_data)
        nx.write_graphml(export_g, filepath)

    @classmethod
    def load(cls, filepath: str) -> "BeliefGraph":
        """Loads belief graph from GraphML file and reconstructs entity nodes."""
        instance = cls()
        loaded_g = nx.read_graphml(filepath)
        instance.graph = nx.MultiDiGraph(loaded_g)
        for node_id, data in instance.graph.nodes(data=True):
            if data.get("type") == "entity":
                canonical_name = data.get("canonical_name", node_id)
                instance.entities[node_id] = EntityNode(
                    entity_id=node_id,
                    canonical_name=canonical_name,
                    aliases=[canonical_name]
                )
        return instance