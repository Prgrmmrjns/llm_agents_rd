from typing import List, Dict
from pydantic import BaseModel, Field

class ReformulatedOptions(BaseModel):
    a: str = Field(..., description="The reformulated option A")
    b: str = Field(..., description="The reformulated option B")
    c: str = Field(..., description="The reformulated option C")
    d: str = Field(..., description="The reformulated option D")

class Chunk(BaseModel):
    content: str = Field(..., description="The content of the text chunk")
    source_url: str = Field(..., description="The URL source of the chunk")
    query: str = Field(..., description="The query that generated this chunk")
    source_type: str = Field("unknown", description="The type of source (findzebra, pubmed, orphanet, gard)")

class Answer(BaseModel):
    chosen_answer: str = Field(..., description="The letter of the chosen answer (A/B/C/D)")
    explanation: str = Field(..., description="Detailed explanation of the decision with evidence")

class SearchKeywords(BaseModel):
    keywords: List[str] = Field(..., description="Keywords for searching evidence about the disease and statements")

class ResearchSummary(BaseModel):
    evidence_by_option: Dict[str, str] = Field(..., description="Evidence supporting each answer option")
    counter_evidence_by_option: Dict[str, str] = Field(..., description="Evidence contradicting each answer option")