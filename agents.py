from __future__ import annotations
from typing import Dict, List
from llm import llm_chat
from models import (
    Validation,
    SearchKeywords,
    ReformulatedOptions
)

def reformulate_options_agent(query) -> dict:
    """
    Reformat the options to make them more specific and easier to search for.
    
    Args:
        query: The reformulated question and original question with options
    """
    prompt = f'''You are tasked with reformulating multiple choice options to make them more specific and easier to search for.
The statement should be reformulated so that they are logically the same as before when the question is: Which statement is TRUE about the disease?     
Note that each statement on its won should be independent of the other statements. 
Sometimes the question is a negated question (e.g. containing the words "EXCEPT" or "NOT"), then the statements should be reformulated so that they are the opposite of the original statements.
Input:
{query}

Output in JSON format with these fields:
{ReformulatedOptions(
    a="Reformulated option A",
    b="Reformulated option B",
    c="Reformulated option C",
    d="Reformulated option D"
).model_dump_json()}

Example: Normal question
Question: Which of the following are characteristic features of Blue Rubber Bleb Nevus Syndrome?
Choices:
A. Presence of multiple, bluish, compressible skin lesions.
B. Occurrence of hemangioma-like lesions in the gastrointestinal tract.
C. Recurrent episodes of bleeding from the lesions.
D. Severe inflammatory joint pain.
Output:
{ReformulatedOptions(
    a="Presence of multiple, bluish, compressible skin lesions is seen in Blue Rubber Bleb Nevus Syndrome.",
    b="Occurrence of hemangioma-like lesions in the gastrointestinal tract is seen in Blue Rubber Bleb Nevus Syndrome.",
    c="Recurrent episodes of bleeding from the lesions is seen in Blue Rubber Bleb Nevus Syndrome.",
    d="Severe inflammatory joint pain is seen in Blue Rubber Bleb Nevus Syndrome."
).model_dump_json()}

Example: Negated question
Question: All of the following are associated with Castleman disease, EXCEPT:
Choices:\n
A. Unicentric lymphadenopathy with hyaline vascular changes.
B. Systemic inflammatory symptoms like fever and weight loss.
C. Rapid progression to lymphoma.
D. Favorable outcomes with surgical resection.
Output:
{ReformulatedOptions(
    a="Unicentric lymphadenopathy with hyaline vascular changes is NOT seen in Castleman disease.",
    b="Systemic inflammatory symptoms like fever and weight loss are NOT seen in Castleman disease.",
    c="Rapid progression to lymphoma is NOT seen in Castleman disease.",
    d="Surgical resection does NOT lead to favorable outcomes in Castleman disease."
).model_dump_json()}

Example: Multiple choice combinations
Question: Genetic testing is diagnostic in:
a) Marfan syndrome
b) Ehlers-Danlos syndrome
c) Osteogenesis imperfecta 
d) Alport syndrome
Choices:
A. acd
B. bcd
C. abd
D. ab
Output:
{ReformulatedOptions(
    a="Marfan syndrome, Osteogenesis imperfecta, and Alport syndrome can be diagnosed using genetic testing, but Ehlers-Danlos syndrome cannot.",
    b="Ehlers-Danlos syndrome, Osteogenesis imperfecta, and Alport syndrome can be diagnosed using genetic testing, but Marfan syndrome cannot.", 
    c="Marfan syndrome, Ehlers-Danlos syndrome, and Alport syndrome can be diagnosed using genetic testing, but Osteogenesis imperfecta cannot.",
    d="Only Marfan syndrome and Ehlers-Danlos syndrome can be diagnosed using genetic testing, while Osteogenesis imperfecta and Alport syndrome cannot."
).model_dump_json()}'''

    return llm_chat(prompt, "reformulated_options")

def keyword_agent(remaining_options: Dict[str, str], rare_disease: str) -> List[str]:
    """
    Generate targeted search keywords based on remaining options.
    
    Args:
        remaining_options: Dictionary of options still under consideration
        rare_disease: Name of the rare disease
    
    Returns:
        List of search keywords
    """
    prompt = f'''Generate search keywords to find evidence about {rare_disease}.
REMAINING STATEMENTS TO PROVE/DISPROVE:
{"\n".join(f"- {text}" for text in remaining_options.values())}

TASK:
Generate keywords that would help find medical texts that could conclusively prove or disprove these statements.
Focus on:
1. Key medical terms and symptoms mentioned
2. Temporal or anatomical aspects (onset, progression, body parts, organ systems)
3. Specific manifestations
4. Technical medical terminology

Output in JSON format matching this Pydantic model:
{SearchKeywords(
    keywords=["keyword1", "keyword2"]
).model_dump_json()}

Example:
Question: Which statement is TRUE about Wilson's Disease?
Choices:
a: "Wilson's Disease is characterized by abnormal copper accumulation in the liver leading to hepatic dysfunction."
b: "Wilson's Disease manifests predominantly as a neurodegenerative disorder with no hepatic involvement."
c: "Wilson's Disease typically causes neuropsychiatric symptoms due to copper deposition in the basal ganglia."
d: "Wilson's Disease is associated with high iron levels causing cardiomyopathy."
Output:
{SearchKeywords(
    keywords=["Wilson's Disease", "copper accumulation", "hepatic dysfunction", "neurodegenerative disorder", "neuropsychiatric symptoms", "copper deposition", "basal ganglia", "high iron levels", "cardiomyopathy"]
).model_dump_json()}'''
    
    response = llm_chat(prompt, "search")
    return response['keywords']

def validation_agent(
    statement: str, 
    chunk: str,
    rare_disease: str
) -> tuple[str, str]:
    """
    Analyze if a chunk provides conclusive evidence about a statement.
    
    Args:
        statement: The statement to validate
        chunk: The source text that contains potential evidence
        rare_disease: Name of the rare disease
    
    Returns:
        Tuple of (validation_result, explanation) where validation_result is:
        - 'True' if chunk conclusively proves the statement
        - 'False' if chunk conclusively disproves the statement
        - 'Unclear' if evidence is inconclusive
    """
    
    prompt = f'''You are tasked with analyzing whether a source text provides CONCLUSIVE evidence about a statement regarding {rare_disease}.
STATEMENT
{statement}

SOURCE TEXT
{chunk}

IMPORTANT RULES:
- Mark as 'False' if the source EXPLICITLY CONTRADICTS the statement
- Mark as 'True' if the source EXPLICITLY CONFIRMS the statement
- Mark as 'Unclear' if the source does not provide any information about the statement

Output in JSON format matching this Pydantic model:
{Validation(
    valid='True / False / Unclear',  
    explanation="""1. Relevant information found: quote specific parts from source
2. Logical connection: explain how the information relates to the statement
3. Alternative interpretations: consider other possible interpretations
4. Conclusion: explain why this proves/disproves the statement or why it's unclear"""
).model_dump_json()}

Example - Evidence for True:

Question: Which statement is TRUE about Gaucher Disease?
STATEMENT
Gaucher Disease is characterized by a deficiency of glucocerebrosidase leading to accumulation of glucocerebroside in macrophages.

SOURCE TEXT
Gaucher Disease, a lysosomal storage disorder, results from mutations that reduce the enzyme activity of glucocerebrosidase.
The deficiency causes glucocerebroside to accumulate within macrophages, which is the pathological hallmark of the disease.
This accumulation is responsible for clinical features such as hepatosplenomegaly and bone pain.
Output:
{Validation(
    valid='True',
    explanation="""1. Relevant information found: 'The deficiency causes glucocerebroside to accumulate within macrophages, which is the pathological hallmark of the disease.'
2. Logical connection: The enzyme deficiency in Gaucher Disease leads to the accumulation of glucocerebroside in macrophages, supporting the statement.
3. Alternative interpretations: There is no alternative interpretation of the source text.
4. Conclusion: The source text provides conclusive evidence that Gaucher Disease is characterized by a deficiency of glucocerebrosidase leading to accumulation of glucocerebroside in macrophages."""
).model_dump_json()}

Example - Evidence for False:
Question: Which statement is TRUE about Fabry Disease?
STATEMENT
Fabry Disease is characterized by a deficiency of beta-galactosidase leading to accumulation of ceramide.

SOURCE TEXT
Fabry Disease is an X-linked lysosomal storage disorder caused by a deficiency of alpha-galactosidase A. This enzymatic defect results in the accumulation of globotriaosylceramide in various tissues, leading to a range of clinical manifestations including neuropathic pain, renal dysfunction, and cardiac involvement.
Output:
{Validation(
    valid='False',
    explanation="""1. Relevant information found: The source text clearly states that Fabry Disease is due to a deficiency of alpha-galactosidase A, and the accumulated substrate is globotriaosylceramide.
2. Logical connection: The statement incorrectly cites beta-galactosidase and ceramide, directly contradicting the reliable source.
3. Alternative interpretations: There is no ambiguity; the enzyme and substrate are unequivocally different.
4. Conclusion: The evidence disproves the statement, confirming it as 'False'."""
).model_dump_json()}

Example - Evidence for Unclear:
Question: Which statement is TRUE about Langerhans Cell Histiocytosis?
STATEMENT
Langerhans Cell Histiocytosis is characterized by the presence of Birbeck granules in all cases.

SOURCE TEXT
Langerhans Cell Histiocytosis is diagnosed through a combination of histopathological, immunohistochemical, and clinical findings. While the identification of Birbeck granules via electron microscopy is a recognized diagnostic feature when present, it is also documented that not every case exhibits these granules despite other supportive diagnostic criteria.
Output:
{Validation(
    valid='Unclear',
    explanation="""1. Relevant information found: The source text indicates that Birbeck granules can be observed in Langerhans Cell Histiocytosis but also emphasizes that their absence does not exclude the disease.
2. Logical connection: The absolute claim in the statement ('in all cases') is at odds with the conditional nature described in the source text.
3. Alternative interpretations: While Birbeck granules are a significant diagnostic marker when present, they are not universally observed, leaving room for ambiguity.
4. Conclusion: The evidence does not conclusively support the statement, resulting in an 'Unclear' validity."""
).model_dump_json()}'''
    
    response = llm_chat(prompt, "validation")
    return response['valid'], response['explanation']