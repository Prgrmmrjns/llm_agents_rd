from typing import List
from pydantic import BaseModel, Field

class ReformulatedOptions(BaseModel):
    a: str = Field(..., description="The reformulated option A")
    b: str = Field(..., description="The reformulated option B")
    c: str = Field(..., description="The reformulated option C")
    d: str = Field(..., description="The reformulated option D")

class Validation(BaseModel):
    valid: str = Field(..., description="True or False, indicating if the evidence/counter-evidence is valid")
    explanation: str = Field(..., description="Detailed explanation of the decision with evidence")

class SearchKeywords(BaseModel):
    keywords: List[str] = Field(..., description="Keywords for searching evidence about the disease and statements")
