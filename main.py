import pandas as pd
import numpy as np
import asyncio

from agents import (
    research_agent,
    reformulate_options_agent,
    validation_agent
)
from rag import (
    get_context_from_rag
)

async def main():
    """Main function to process questions and save results."""
    # Load dataset
    df = pd.read_parquet("hf://datasets/guan-wang/ReDis-QA/data/test-00000-of-00001.parquet")
    
    # Initialize results storage
    results = []
    
    # Process questions
    for idx in range(20):  # Process first 20 questions
        print(f"\nProcessing question {idx + 1}")
        row = df.iloc[idx]
    
        # Extract disease and ensure it's a string
        disease_value = row['rare disease']
        if isinstance(disease_value, (list, np.ndarray)):
            disease_value = str(disease_value[0]) if len(disease_value) > 0 else str(disease_value)
        disease = str(disease_value).strip("[]'")
        
        # Reformulate the question for clarity
        query = f"Which statement is TRUE about the rare disease {disease}?"
        
        # Get reformulated options
        options = reformulate_options_agent(row['input'])
        # Use reformulated options for further processing
        options = {
            'A': options['a'],
            'B': options['b'],
            'C': options['c'],
            'D': options['d']
        }
        print(f'Options: {options}')
        
        # Get context chunks from RAG database
        all_chunks = await get_context_from_rag(query, disease, options)
        
        # Process chunks with research agent
        chunks_analyzed = 0
        evidence = {}
        counter_evidence = {}
        final_answer = "Unclear"
        final_evidence = {}
        source_url = ""
        
        # Process each chunk until conclusive evidence is found
        for i, chunk in enumerate(all_chunks, 1):
            print(f'Analyzing chunk {i} of {len(all_chunks)}')
            
            # Extract source URL from chunk
            chunk_lines = chunk.split('\n')
            current_source = chunk_lines[0].replace('Source: ', '')
            
            analysis = research_agent(chunk, query, evidence, counter_evidence)
            
            # Validate evidence and counter-evidence
            answer, valid_evidence, valid_counter_evidence = validation_agent(
                options=options,
                chunk_info=analysis,
                rare_disease=disease
            )
            
            chunks_analyzed = i
            
            # If we found valid evidence for an answer, stop processing
            if valid_evidence:
                for opt, evid in valid_evidence.items():
                    if evid:  # If there's actual evidence content
                        final_answer = opt
                        final_evidence = valid_evidence
                        source_url = current_source
                        print(f"Found valid evidence for option {opt} from source: {source_url}")
                        print("Stopping analysis.")
                        break
                if final_answer != "Unclear":
                    break
            
            # Update tracking dictionaries for next iteration
            evidence = valid_evidence
            counter_evidence = valid_counter_evidence
        
        correct_answer = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}.get(row['cop'], 'Unknown')
        print(f'Chosen answer: {final_answer}')
        print(f'Correct answer: {correct_answer}')
        print(f'Evidence: {final_evidence}')
        if source_url:
            print(f'Source: {source_url}\n')
        else:
            print('No conclusive source found\n')
        
        results.append({
            'question': query,
            'reformulated_options': options,
            'llm_answer': final_answer,
            'correct_answer': correct_answer,
            'evidence': final_evidence,
            'source_url': source_url,
            'num_chunks_analyzed': chunks_analyzed,
            'total_chunks': len(all_chunks)
        })
        
        # Save results after each question
        pd.DataFrame(results).to_csv("results_rag.csv", index=False)

if __name__ == "__main__":
    asyncio.run(main())