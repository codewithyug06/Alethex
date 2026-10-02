from typing import List, Tuple
from alethex.graph.belief_graph import BeliefGraph
from alethex.ingestion.schema import Claim

def get_conflicts(graph: BeliefGraph, entity_id: str) -> List[Tuple[Claim, Claim]]:
    conflicts = []
    if not graph.graph.has_node(entity_id):
        return conflicts
        
    claims = []
    for _, _, data in graph.graph.edges(entity_id, data=True):
        if "claim_obj" in data:
            claims.append(data["claim_obj"])
            
    # Check all pairs
    for i in range(len(claims)):
        for j in range(i + 1, len(claims)):
            c1 = claims[i]
            c2 = claims[j]
            v1 = f"val::{c1.object_}"
            v2 = f"val::{c2.object_}"
            
            # MultiDiGraph has_edge checks if any edge exists between u and v
            if graph.graph.has_edge(v1, v2):
                # Check if there is a conflict edge specifically
                edge_data = graph.graph.get_edge_data(v1, v2)
                for k, attrs in edge_data.items():
                    if attrs.get("type") == "conflict":
                        conflicts.append((c1, c2))
                        break
    return conflicts

def get_superseded(graph: BeliefGraph, entity_id: str) -> List[Claim]:
    superseded = []
    if not graph.graph.has_node(entity_id):
        return superseded
        
    for _, _, data in graph.graph.edges(entity_id, data=True):
        claim = data.get("claim_obj")
        if claim and claim.validity_end is not None:
            superseded.append(claim)
    return superseded

def get_timeline(graph: BeliefGraph, entity_id: str, predicate: str) -> List[Claim]:
    timeline = []
    if not graph.graph.has_node(entity_id):
        return timeline
        
    for _, _, data in graph.graph.edges(entity_id, data=True):
        claim = data.get("claim_obj")
        if claim and claim.predicate == predicate:
            timeline.append(claim)
            
    timeline.sort(key=lambda x: x.validity_start or x.timestamp)
    return timeline