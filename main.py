import pandas as pd
import asyncio
from dataclasses import dataclass
from typing import Dict
from rag import build_rag
from agents import (
    reformulate_options_agent,
    validation_agent,
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
        return valid_options[0] if len(valid_options) == 1 else None

def get_disease_data(df, disease_name):
    return df[df['Name'].str.contains(disease_name, case=False, na=False)]

async def main():
    """Main function to process questions and save results."""
    # Load datasets
    df = pd.read_parquet("hf://datasets/guan-wang/ReDis-QA/data/test-00000-of-00001.parquet")
    datasets = {
        "Functional Consequences": pd.read_csv("csv/functional_consequences_data.csv"),
        "Genes": pd.read_csv("csv/genes_data.csv"),
        "Natural History": pd.read_csv("csv/natural_history_data.csv"),
        "Phenotype": pd.read_csv("csv/phenotype_data.csv"),
    }
    
    results = []
    answer_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    correct_count = total_count = 0
    
    df = df.head(100)  # Process first few rows for testing
    for idx, row in df.iterrows():
        print("\n" + "="*80)
        rare_disease = str(df['rare disease']).strip("[]'") if isinstance(row['rare disease'], str) else str(row['rare disease'][0])
        print(f"\nProcessing Question {idx}/100 about: {rare_disease}")
        print(f"Question: {row['input']}")
        
        # Initialize options
        response = reformulate_options_agent(row['input'])
        options_mgr = OptionsManager({k: response[k.lower()] for k in 'ABCD'})
        keywords = response['keywords']
        
        print("\nReformulated Options:")
        for letter, statement in options_mgr.get_valid_options().items():
            print(f"{letter}. {statement}")
        
        # Collect relevant data from datasets
        relevant_data = []
        area_to_columns = {
            "Functional Consequences": ["FunctionalConsequence", "DisabilityType"],
            "Genes": ["Genes"],
            "Natural History": ["AgeOfOnset", "TypeOfInheritance"],
            "Phenotype": ["HPO_Term"],
            #"Prevalence": ["Prevalences"]
        }
        
        for area in ["Functional Consequences", "Genes", "Natural History", "Phenotype"]:
            if area in datasets:
                disease_data = get_disease_data(datasets[area], rare_disease)
                if not disease_data.empty:
                    data_str = disease_data[area_to_columns[area]].to_string()
                    relevant_data.append(f"=== {area} Data ===\n{data_str}")
        
        # Process data and validate options
        chosen_option = None
        explanation = ""
        source = ""
        answer_source = ""  # Track if answer came from local or web
        
        # Try local data first
        if relevant_data:
            print("\nValidating with local data...")
            context = "\n\n".join(relevant_data)
            
            # Check context length before validation
            if len(context.split()) > 1000:  # Rough estimate of 1000 words ~ 1500 tokens
                print("Local data context too long, skipping validation")
                chosen_option = None
            else:
                try:
                    explanation, *validations = validation_agent(
                        options_mgr.get_valid_options(),
                        context,
                        rare_disease
                    )
                    
                    print("Validation results:")
                    for letter, result in zip('ABCD', validations):
                        print(f"Option {letter}: {result}")
                        if result == 'False':
                            options_mgr.eliminate_option(letter)
                        elif result == 'True':
                            # Found our answer - eliminate all other options
                            for other_letter in 'ABCD':
                                if other_letter != letter:
                                    options_mgr.eliminate_option(other_letter)
                            answer_source = "local"
                    chosen_option = options_mgr.get_chosen_option()
                except Exception as e:
                    print(f"Error during local validation: {str(e)}")
                    chosen_option = None
        
        # If no definitive answer, try RAG
        if not chosen_option:
            print("\nNo definitive answer from local data, searching external sources...")
            chunks = await build_rag(keywords, rare_disease)
            
            for chunk in chunks:
                source = chunk.split('\n')[0].replace('Source: ', '')
                print(f"\nAnalyzing source: {source}")
                
                # Check chunk length before validation
                if len(chunk.split()) > 1000:  # Rough estimate of 1000 words ~ 1500 tokens
                    print("Chunk too long, skipping...")
                    continue
                
                valid_options = options_mgr.get_valid_options()
                if not valid_options:
                    break
                
                try:
                    explanation, *validations = validation_agent(valid_options, chunk, rare_disease)
                    print("Validation results:")
                    for letter, result in zip('ABCD', validations):
                        print(f"Option {letter}: {result}")
                        if result == 'False':
                            options_mgr.eliminate_option(letter)
                    
                    chosen_option = options_mgr.get_chosen_option()
                    if chosen_option:
                        answer_source = "web"
                        break
                except Exception as e:
                    print(f"Error during chunk validation: {str(e)}")
                    continue
        
        # Record results
        if chosen_option:
            correct_answer = answer_dict.get(row['cop'])
            total_count += 1
            if chosen_option.letter == correct_answer:
                correct_count += 1
            
            print("\nFinal Answer:")
            print(f"Selected: {chosen_option.letter}. {chosen_option.statement}")
            print(f"Correct Answer: {correct_answer}")
            print(f"Source: {answer_source}")  # Print source of answer
            print(f"Explanation: {explanation}")
            if source:  # Only print URL if from web
                print(f"URL: {source}")
            
            results.append({
                'index': idx,
                'question': f"Which statement is TRUE about the rare disease {rare_disease}?",
                'answer': chosen_option.letter,
                'answer_statement': chosen_option.statement,
                'correct_answer': correct_answer,
                'is_correct': chosen_option.letter == correct_answer,
                'explanation': explanation,
                'source': answer_source,  # Add source to results
                'url': source if answer_source == "web" else ""  # Add URL if from web
            })

    # Save results and print summary
    pd.DataFrame(results).to_csv("llm_responses_main.csv", index=False)
    print("\n" + "="*80)
    print("\nFinal Results:")
    print(f"Total questions processed: {total_count}")
    print(f"Correct answers: {correct_count}")
    print(f"Accuracy: {(correct_count / total_count if total_count > 0 else 0):.2%}")

if __name__ == "__main__":
    asyncio.run(main())