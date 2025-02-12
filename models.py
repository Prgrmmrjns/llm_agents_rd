from typing import List
from pydantic import BaseModel, Field

class ReformulatedOptions(BaseModel):
    a: str = Field(..., description="The reformulated option A")
    b: str = Field(..., description="The reformulated option B")
    c: str = Field(..., description="The reformulated option C")
    d: str = Field(..., description="The reformulated option D")

class ValidationList(BaseModel):
    a: str = Field(..., description="Validation for option A: True / False / Unclear")
    b: str = Field(..., description="Validation for option B: True / False / Unclear")
    c: str = Field(..., description="Validation for option C: True / False / Unclear")
    d: str = Field(..., description="Validation for option D: True / False / Unclear")
    explanation: str = Field(..., description="Explanation for the validation of the options")

class SearchKeywords(BaseModel):
    keywords: List[str] = Field(..., description="Keywords for searching evidence about the disease and statements")
