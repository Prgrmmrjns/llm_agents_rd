from __future__ import annotations
from typing import Dict, List
from llm import llm_chat
from models import (
    ValidationList,
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
Note that each statement on its own should be independent of the other statements. 
Sometimes the question is a negated question (e.g. containing the words "EXCEPT" or "NOT"), then the statements should be reformulated so that they are the opposite of the original statements.
Also list the area(s) of the disease that the question falls into. Choose from the following areas:
- Functional consequences
- Genes
- Natural History
- Phenotype
- Prevalence

Input:
{query}

Output in JSON format with these fields:
{ReformulatedOptions(
    a="Reformulated option A",
    b="Reformulated option B",
    c="Reformulated option C",
    d="Reformulated option D",
    keywords=[]
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
    d="Severe inflammatory joint pain is seen in Blue Rubber Bleb Nevus Syndrome.",
    keywords=["Blue Rubber Bleb Nevus Syndrome", "Inflammatory joint pain", "Skin lesions", "Gastrointestinal tract", "Bleeding"]
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
    d="Surgical resection does NOT lead to favorable outcomes in Castleman disease.",
    keywords=["Unicentric lymphadenopathy with hyaline vascular changes", "Systemic inflammatory symptoms", "Rapid progression to lymphoma", "Surgical resection"]
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
    d="Only Marfan syndrome and Ehlers-Danlos syndrome can be diagnosed using genetic testing, while Osteogenesis imperfecta and Alport syndrome cannot.",
    keywords=["Marfan syndrome", "Ehlers-Danlos syndrome", "Osteogenesis imperfecta", "Alport syndrome", "Genetic testing"]
).model_dump_json()}'''

    return llm_chat(prompt, "reformulated_options")

def validation_agent(statements: Dict[str, str], chunk: str, rare_disease: str) -> dict:
    """
    Analyze whether a source text provides conclusive evidence for all statements simultaneously.
    
    Args:
        statements: Dictionary of statements for options (keys should be uppercase letters, e.g., 'A')
        chunk: The source text chunk that may contain evidence.
        rare_disease: Name of the rare disease.
    
    Returns:
        A dictionary matching the ValidationList schema with keys 'a', 'b', 'c', 'd', and 'explanation'.
    """
    # Ensure all four options appear in the prompt. For any missing one, supply a filler.
    complete_options = {}
    for letter in "abcd":
        key_upper = letter.upper()
        if key_upper in statements:
            complete_options[letter] = statements[key_upper]
            
    statements_text = "\n".join(f"{letter.upper()}. {text}" for letter, text in complete_options.items())
    
    prompt = f'''You are tasked with analyzing whether a source text provides CONCLUSIVE evidence about the following statements regarding the rare disease {rare_disease}:
{statements_text}

SOURCE:
{chunk}

IMPORTANT RULES:
For each statement:
- Mark as 'True' if the source explicitly confirms the statement.
- Mark as 'False' if the source explicitly contradicts the statement.
- Mark as 'Unclear' if the source does not provide sufficient information about the statement.
- Note that only one statement is true while three options are false.
- Do not mark all options as 'False' or two or more options as 'True'.
Output in JSON format matching this Pydantic model:
{ValidationList(
    a="True / False / Unclear for option A",
    b="True / False / Unclear for option B",
    c="True / False / Unclear for option C",
    d="True / False / Unclear for option D",
    explanation="Explanation for the validation of the options"
).model_dump_json()}'''
    response = llm_chat(prompt, "validation_list")
    explanation, a, b, c, d = response['explanation'], response['a'], response['b'], response['c'], response['d']
    return explanation, a, b, c, d