from __future__ import annotations
from llm import llm_chat
from models import (
    Answer,
    ResearchSummary,
    SearchKeywords,
    ReformulatedOptions
)
from typing import List, Tuple, Dict

def reformulate_options_agent(query) -> dict:
    """
    Reformat the options to make them more specific and easier to search for.
    
    Args:
        query: The reformulated question and original question with options
    """
    prompt = f'''You are a medical research expert tasked with reformulating multiple choice options to make them more specific and easier to search for.
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
        Dict containing search keywords for each option and general keywords
    """
    prompt = f'''
You are a medical research expert tasked with generating search keywords to help find evidence for or against multiple choice options.
Follow these steps carefully:

1. Analyze the question and each answer option
2. For each option:
   - Extract key medical terms, symptoms, conditions
   - Consider temporal aspects (onset, progression, stages)
   - Consider demographic factors (age groups, populations)
   - Include synonyms and related terms
   - Include terms that could disprove the option
3. Generate general keywords relevant to the overall question
4. Identify any time-related terms (early onset, late stage, etc.)
5. Identify demographic-related terms (pediatric, adult, elderly, etc.)

Question: {question}

Answer Options:
{options}

Output in JSON format:
{SearchKeywords(
    keywords_by_option={
        "A": ["keyword1", "keyword2"],
        "B": ["keyword1", "keyword2"],
        "C": ["keyword1", "keyword2"],
        "D": ["keyword1", "keyword2"]
    },
    general_keywords=["keyword1", "keyword2"],
    temporal_keywords=["early onset", "late stage"],
    demographic_keywords=["pediatric", "adult"]
).model_dump_json()}'''
    
    return llm_chat(prompt, "search")

def research_agent(chunk: str, question: str, previous_findings: str) -> dict:
    """
    Analyze a single chunk of text and extract relevant information for the question.
    
    Args:
        chunk: Text chunk with source information and content
        question: The question being asked
        previous_findings: Previously accumulated evidence
    
    Returns:
        Dict containing evidence analysis and metadata
    """
    prompt = f'''You are a medical research assistant analyzing evidence for a multiple choice question.
Your task is to find evidence that proves or disproves the following statements about a rare disease.
For each answer option (A, B, C, D):
   - Find evidence that PROVES OR DISPROVES the positive statement is true

Instructions:
- Consider how this new evidence combines with previous findings.
- Be specific and cite direct quotes where possible.
- Maintain objectivity - do not make final decisions about the answer
- When you do not find any evidence for or against a statement leave the evidence field blank.
- Keep evidence summaries concise and focused on key points.

Question: {question}

Information to analyze:
{chunk}

Previous Evidence:
{previous_findings}

Output in JSON format:
{ResearchSummary(
    evidence_by_option={
        "A": "Evidence proving this positive statement is TRUE",
        "B": "Evidence proving this positive statement is TRUE",
        "C": "Evidence proving this positive statement is TRUE",
        "D": "Evidence proving this positive statement is TRUE"
    },
    counter_evidence_by_option={
        "A": "Evidence proving this positive statement is FALSE",
        "B": "Evidence proving this positive statement is FALSE",
        "C": "Evidence proving this positive statement is FALSE",
        "D": "Evidence proving this positive statement is FALSE"
    },
    accumulated_evidence="Combined relevant evidence from this and previous chunks"
).model_dump_json()}'''
    
    return llm_chat(prompt, "research")

def answer_agent(question_info: dict, chunk_info: dict) -> Tuple[str, str]:
    """
    Make a decision based on the current chunk of information.
    If unclear, indicates more information is needed.
    
    Args:
        question_info: Dict containing question input, reformulated options, and logical flags
        chunk_info: Dictionary containing current chunk's summary and source info
    """
    # Extract source URL and similarity from the source_info string
    source_lines = chunk_info['source_info'].split('\n')
    source_parts = source_lines[0].split(' - ')
    source_type = source_parts[0].replace('Source: ', '')
    source_url = source_parts[1].split(' (')[0] if ' (' in source_parts[1] else source_parts[1]
    similarity = source_lines[1]  # Similarity score line
    
    # Format the evidence text with logical context
    evidence_text = (
        f"You are a medical expert analyzing evidence to answer a multiple choice question.\n"
        f"IMPORTANT: For regular questions, choose the option that is TRUE.\n"
        f"For EXCEPT/NOT questions, choose the option that is FALSE while others are TRUE.\n\n"
        f"Original Question: {question_info['original_question']}\n\n"
        f"Reformulated as: Which statement is TRUE about the disease?\n"
    )
    
    # Add reformulated options
    evidence_text += "\nReformulated Options:\n"
    for option, text in question_info['reformulated_options'].items():
        evidence_text += f"{option}: {text}\n"
    
    evidence_text += "\nEvidence Analysis:\n"
    for option in ['A', 'B', 'C', 'D']:
        evidence_text += f"\nOption {option}:\n"
        evidence_text += f"Evidence proving statement TRUE: {chunk_info['evidence_by_option'][option]}\n"
        evidence_text += f"Evidence proving statement FALSE: {chunk_info['counter_evidence_by_option'][option]}\n"
    
    evidence_text += f"\nAccumulated Evidence:\n{chunk_info['accumulated_evidence']}\n"
    
    # Add clear instructions about logical flow
    evidence_text += "\nINSTRUCTIONS:\n"
    if "EXCEPT" in question_info['original_question'] or "NOT" in question_info['original_question']:
        evidence_text += (
            "1. This is an EXCEPT/NOT question.\n"
            "2. The reformulated options are stated as negatives.\n"
            "3. Choose the option where the evidence proves the NEGATIVE statement is TRUE.\n"
            "4. In other words, choose the option that is FALSE in the original question.\n"
        )
    else:
        evidence_text += (
            "1. This is a regular question.\n"
            "2. Choose the option where the evidence proves the statement is TRUE.\n"
            "3. All other options should be proven FALSE by the evidence.\n"
        )
    
    prompt = f'''{evidence_text}

Output in JSON format matching this Pydantic model:
{Answer(
    chosen_answer="The letter of the chosen answer (A/B/C/D)",
    explanation="Detailed explanation of why this answer was chosen, explicitly stating how the evidence proves this statement is TRUE while others are FALSE."
).model_dump_json()}'''
    
    response = llm_chat(prompt, "answer")
    return response['chosen_answer'], response['explanation']
