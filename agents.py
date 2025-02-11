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
    ).model_dump_json()}'''

    return llm_chat(prompt, "reformulated_options")

def search_agent(question: str, options: Dict[str, str]) -> dict:
    """
    Generate search keywords based on the question and answer options.
    
    Args:
        question: The question being asked
        options: Dictionary mapping option letters to their text
    
    Returns:
        Dict containing search keywords
    """
    prompt = f'''
    You are a medical research expert tasked with generating search keywords.
    Analyze the question and each answer option to extract key medical terms, symptoms, conditions, temporal aspects, and demographics.
    Generate a focused list of keywords (5-10) that would help find evidence about these medical statements.
    Include important synonyms and related terms.

    Question: {question}
    Answer Options:
    {options}

    Output in JSON format:
    {SearchKeywords(
        keywords=["keyword1", "keyword2"]
    ).model_dump_json()}'''
    
    return llm_chat(prompt, "search")

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
    
    prompt = f'''You are a medical expert analyzing whether a source text provides CONCLUSIVE evidence about a statement regarding {rare_disease}.

STATEMENT TO ANALYZE:
"{statement}"

SOURCE TEXT:
{chunk}

IMPORTANT RULES:
1. ABSENCE OF EVIDENCE IS NOT EVIDENCE OF ABSENCE
2. Only mark as 'False' if the source EXPLICITLY CONTRADICTS the statement
3. Only mark as 'True' if the source EXPLICITLY CONFIRMS the statement
4. If the source doesn't mention something, that's 'Unclear', not 'False'
5. Require direct quotes that conclusively prove/disprove the statement

ANALYSIS INSTRUCTIONS:
1. First, identify any relevant information in the source text that directly relates to the statement
2. Check if this information EXPLICITLY proves or disproves the statement
3. Look for specific quotes or medical facts that support your conclusion
4. Consider if there could be alternative interpretations
5. Determine if the evidence is truly conclusive

Your response must be one of:
- 'True' if the source CONCLUSIVELY PROVES the statement with direct evidence
- 'False' if the source EXPLICITLY CONTRADICTS the statement
- 'Unclear' if the evidence is incomplete, inconclusive, or the topic isn't mentioned

Output in JSON format matching this Pydantic model:
{Validation(
    valid='True / False / Unclear',  
    explanation="""1. Relevant information found: <quote specific parts from source>
2. Logical connection: <explain how the information relates to the statement>
3. Alternative interpretations: <consider other possible interpretations>
4. Conclusion: <explain why this proves/disproves the statement or why it's unclear>"""
).model_dump_json()}'''
    
    response = llm_chat(prompt, "validation")
    return response['valid'], response['explanation']

def keyword_agent(remaining_options: Dict[str, str], rare_disease: str) -> List[str]:
    """
    Generate targeted search keywords based on remaining options.
    
    Args:
        remaining_options: Dictionary of options still under consideration
        rare_disease: Name of the rare disease
    
    Returns:
        List of search keywords
    """
    prompt = f'''You are a medical expert generating search keywords to find evidence about {rare_disease}.

REMAINING STATEMENTS TO PROVE/DISPROVE:
{"\n".join(f"- {text}" for text in remaining_options.values())}

TASK:
Generate specific keywords that would help find medical texts that could conclusively prove or disprove these statements.
Focus on:
1. Key medical terms and symptoms mentioned
2. Temporal aspects (onset, progression)
3. Specific manifestations
4. Technical medical terminology

Output in JSON format matching this Pydantic model:
{SearchKeywords(
    keywords=["Each keyword should be specific and medical in nature"]
).model_dump_json()}'''
    
    response = llm_chat(prompt, "search")
    return response['keywords']
