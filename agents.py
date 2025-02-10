from __future__ import annotations
from typing import Tuple, Dict
from llm import llm_chat
from models import (
    Answer,
    ResearchSummary,
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
    If the question is a negated question (e.g. containing the words "EXCEPT" or "NOT"), then the statements should be reformulated so that they are the opposite of the original statements.
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
    Generate a focused list of keywords that would help find evidence about these medical statements.
    Include important synonyms and related terms.

    Question: {question}
    Answer Options:
    {options}

    Output in JSON format:
    {SearchKeywords(
        keywords=["keyword1", "keyword2"]
    ).model_dump_json()}'''
    
    return llm_chat(prompt, "search")

def research_agent(chunk: str, question: str, evidence: Dict[str, str], counter_evidence: Dict[str, str]) -> dict:
    """
    Analyze a single chunk of text and extract relevant information for the question.
    
    Args:
        chunk: Text chunk with source information and content
        question: The question being asked
        evidence: Evidence for the statements
        counter_evidence: Counter evidence for the statements
    
    Returns:
        Dict containing evidence analysis and metadata
    """
    prompt = f'''You are tasked with analyzing a piece of information related to a rare disease and extracting evidence that conclusively proves which one of the following statements is TRUE, while providing counter evidence for the other three options. The aim is to ensure that by the end of the analysis, exactly one option has supporting evidence and the remaining three have counter evidence clearly demonstrating why they are false.

Question: {question}

Instructions:
- Combine this new information with any previous findings.
- Be specific and cite direct quotes where possible.
- Provide clear supporting evidence for the correct statement and clear counter evidence for the incorrect statements.
- If the provided information is not sufficient to decisively determine this pattern, summarize any partial findings; further evidence may be required.

New information to analyze:
{chunk}

Previous Evidence:
{evidence}

Previous Counter Evidence:
{counter_evidence}

Output in JSON format matching this Pydantic model:
{ResearchSummary(
    evidence_by_option={
        "A": "Evidence supporting option A if this statement is true, or an empty string otherwise.",
        "B": "Evidence supporting option B if this statement is true, or an empty string otherwise.",
        "C": "Evidence supporting option C if this statement is true, or an empty string otherwise.",
        "D": "Evidence supporting option D if this statement is true, or an empty string otherwise."
    },
    counter_evidence_by_option={
        "A": "Counter evidence demonstrating why option A is false, or an empty string if not applicable.",
        "B": "Counter evidence demonstrating why option B is false, or an empty string if not applicable.",
        "C": "Counter evidence demonstrating why option C is false, or an empty string if not applicable.",
        "D": "Counter evidence demonstrating why option D is false, or an empty string if not applicable."
    }
).model_dump_json()}'''
    
    return llm_chat(prompt, "research")

def validation_agent(options: Dict[str, str], chunk_info: dict, rare_disease: str) -> Tuple[str, Dict[str, str], Dict[str, str]]:
    """
    Validate evidence and counter-evidence for each option.
    If evidence is valid, that option is chosen as the answer.
    If counter-evidence is valid, that option is removed from consideration.
    
    Args:
        options: Dictionary of options to validate
        chunk_info: Dictionary containing evidence and counter-evidence
        rare_disease: Name of the rare disease
    
    Returns:
        Tuple of (chosen answer, remaining evidence dict, remaining counter evidence dict)
    """
    options_text = "\nOptions to validate:\n" + "\n".join(
        f"{k}. {v}" for k, v in options.items()
    )

    # Format evidence for validation
    evidence_text = "\nEvidence to validate:\n" + "\n".join(
        f"{k}. Statement: {options[k]}\nEvidence: {v}"
        for k, v in chunk_info['evidence_by_option'].items() if v
    )

    counter_evidence_text = "\nCounter-evidence to validate:\n" + "\n".join(
        f"{k}. Statement: {options[k]}\nCounter-evidence: {v}"
        for k, v in chunk_info['counter_evidence_by_option'].items() if v
    )

    prompt = f'''You are a medical expert validating evidence for statements about {rare_disease}.
For each piece of evidence and counter-evidence:
1. Check if it DIRECTLY proves or disproves the associated statement
2. Check if it is logically sound and based on the actual text
3. Check if it contains relevant quotes or specific information

For evidence: Only keep it if it CONCLUSIVELY proves the statement is TRUE
For counter-evidence: Only keep it if it CONCLUSIVELY proves the statement is FALSE

{options_text}
{evidence_text}
{counter_evidence_text}

Output in JSON format matching this Pydantic model:
{Answer(
    chosen_answer="Letter of option with valid evidence (A/B/C/D), or Unclear if no conclusive evidence",
    explanation="For each piece of evidence and counter-evidence, explain why it was kept or dismissed. If any evidence was conclusively valid, explain why that option was chosen."
).model_dump_json()}'''
    
    response = llm_chat(prompt, "answer")
    
    # Process validation results
    answer = response['chosen_answer']
    
    # If we found a valid evidence that proves an option is true, return it
    if answer != "Unclear":
        return answer, {}, {}
        
    # Remove any options that were conclusively disproven
    # Parse the explanation to determine which counter-evidence was valid
    explanation = response['explanation'].lower()
    remaining_evidence = {}
    remaining_counter_evidence = {}
    
    for option in options.keys():
        if f"option {option.lower()} is conclusively false" in explanation:
            continue
        if chunk_info['evidence_by_option'].get(option):
            remaining_evidence[option] = chunk_info['evidence_by_option'][option]
        if chunk_info['counter_evidence_by_option'].get(option):
            remaining_counter_evidence[option] = chunk_info['counter_evidence_by_option'][option]
            
    return "Unclear", remaining_evidence, remaining_counter_evidence
