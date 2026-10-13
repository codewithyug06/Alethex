from datetime import datetime
from typing import Optional, Literal
from pydantic import BaseModel, Field, ConfigDict

class RawDocument(BaseModel):
    source_id: str
    timestamp: datetime
    text: str

class Claim(BaseModel):
    claim_id: str
    source_id: str
    timestamp: datetime
    subject: str
    predicate: str
    object_: str = Field(alias="object")
    confidence: float = Field(ge=0.0, le=1.0)
    raw_sentence: str
    validity_start: Optional[datetime] = None
    validity_end: Optional[datetime] = None
    
    model_config = ConfigDict(populate_by_name=True)

class EntityNode(BaseModel):
    entity_id: str
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)

class ClaimPair(BaseModel):
    claim_a: Claim
    claim_b: Claim
    relation: Literal["consistent", "superseded", "conflicting", "ambiguous"]
    confidence: float = Field(ge=0.0, le=1.0)
    justification: str
