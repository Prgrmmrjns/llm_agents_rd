import pandas as pd
import asyncio
from dataclasses import dataclass
from typing import Dict

from agents import (
    reformulate_options_agent,
    validation_agent,
    keyword_agent,
)
from rag import (
    build_rag,
)

@dataclass
class Option:
    letter: str
    statement: str
    is_valid: bool = True

class OptionsManager:
    def __init__(self, options_dict: Dict[str, str]):
        self.options = [Option(letter, statement) for letter, statement in options_dict.items()]
    
    def get_valid_options(self) -> Dict[str, str]:
        return {opt.letter: opt.statement for opt in self.options if opt.is_valid}
    
    def eliminate_option(self, letter: str):
        for opt in self.options:
            if opt.letter == letter:
                opt.is_valid = False
                break
    
    def get_chosen_option(self) -> Option:
        valid_options = [opt for opt in self.options if opt.is_valid]
        if len(valid_options) == 1:
            return valid_options[0]
        return None

def print_separator():
    print("\n" + "="*80 + "\n")

def print_section(title):
    print(f"\n{'-'*20} {title} {'-'*20}\n")

async def main():
    """Main function to process questions and save results."""
    df = pd.read_parquet("hf://datasets/guan-wang/ReDis-QA/data/test-00000-of-00001.parquet")
    results = []
    answer_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    
    for idx, row in df.iterrows():
        disease = str(df['rare disease']).strip("[]'") if isinstance(row['rare disease'], str) else str(row['rare disease'][0])
        print_section("Question")
        print(row['input'])
        
        # Initialize options
        raw_options = reformulate_options_agent(row['input'])
        options_mgr = OptionsManager({k: raw_options[k.lower()] for k in 'ABCD'})
        
        query = f"Which statement is TRUE about the rare disease {disease}?"
        print_section("Reformulated Options")
        print(query)
        for letter, statement in options_mgr.get_valid_options().items():
            print(f"{letter}. {statement}")

        # Get keywords for valid options
        keywords = keyword_agent(options_mgr.get_valid_options(), disease)
        print_section("Search Keywords")
        print(", ".join(keywords))

        # Get and process chunks - use only essential keywords
        essential_keywords = [disease] + keywords[:3]  # Use disease name and top 3 keywords
        chunks = await build_rag(essential_keywords, disease)
        
        if not chunks:  # If first search fails, try with just the disease name
            chunks = await build_rag([disease], disease)
        
        print_section("Search Results")
        # Get unique sources with similarity scores
        sources_with_scores = {}
        for i, chunk in enumerate(chunks):
            source = chunk.split('\n')[0].replace('Source: ', '')
            if source not in sources_with_scores:
                sources_with_scores[source] = i  # Lower index means higher similarity
                
        print(f"Found {len(sources_with_scores)} sources: {sources_with_scores}")
        
        chosen_option = None
        last_explanation = ""
        last_source = ""
        
        for i, chunk in enumerate(chunks):
            source = chunk.split('\n')[0].replace('Source: ', '')
            print_section(f"Source: {source}")
            print(f'Chunk length: {len(chunk)}')
            print(f'Chunk preview: {chunk[:400]}')
            # Only validate remaining options
            valid_options = options_mgr.get_valid_options()
            if not valid_options:
                break
                
            explanation, a, b, c, d = validation_agent(valid_options, chunk, disease)
            print_section(f"Validation (Source: {source})")
            print(f'A: {a}\nB: {b}\nC: {c}\nD: {d}\nExplanation: {explanation}')
            
            # Process validation results
            for letter, result in zip('ABCD', [a, b, c, d]):
                if result == 'False':
                    options_mgr.eliminate_option(letter)
                elif result == 'True':
                    chosen_option = Option(letter, valid_options[letter])
                    last_explanation = explanation
                    last_source = source
            
            # Check if we have a definitive answer
            if chosen_option or len(valid_options) == 1:
                break
        
        if not chosen_option:
            chosen_option = options_mgr.get_chosen_option()
        
        if chosen_option:
            print_section("Final Answer")
            correct_answer = answer_dict.get(row['cop'])
            print(f'{chosen_option.letter}. {chosen_option.statement}. Correct Answer: {correct_answer}')
            results.append({
                'question': query,
                'answer': chosen_option.statement,
                'correct_answer': correct_answer,
                'explanation': last_explanation,
                'source': last_source
            })

if __name__ == "__main__":
    asyncio.run(main())