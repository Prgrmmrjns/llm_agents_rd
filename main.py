import pandas as pd
import asyncio
from typing import Dict
import re

from agents import (
    reformulate_options_agent,
    validation_agent,
    keyword_agent
)
from rag import (
    get_context_from_rag
)

def print_separator():
    print("\n" + "="*80 + "\n")

def print_section(title):
    print(f"\n{'-'*20} {title} {'-'*20}\n")

def clean_chunk(chunk: str) -> str:
    """Clean a text chunk by removing URLs, double brackets, and navigation text."""
    # Remove URLs and link references
    chunk = re.sub(r'\[.*?\]\(.*?\)', '', chunk)  # Remove markdown links
    chunk = re.sub(r'<https?://[^>]*>', '', chunk)  # Remove HTML URLs
    chunk = re.sub(r'\[\[.*?\]\]', '', chunk)  # Remove double brackets
    chunk = re.sub(r'\[Go to:.*?\]', '', chunk)  # Remove "Go to" navigation
    chunk = re.sub(r'\[.*?\]', '', chunk)  # Remove remaining brackets
    
    # Clean up whitespace
    chunk = re.sub(r'\n\s*\n', '\n\n', chunk)  # Normalize multiple newlines
    chunk = re.sub(r'^\s+|\s+$', '', chunk, flags=re.MULTILINE)  # Trim lines
    
    return chunk

async def main():
    """Main function to process questions and save results."""
    df = pd.read_parquet("hf://datasets/guan-wang/ReDis-QA/data/test-00000-of-00001.parquet")
    results = []
    
    for idx in range(100):
        print_separator()
        print(f"Processing Question {idx + 1}/100\n {df.iloc[idx]['input']}")
        print_separator()
        
        row = df.iloc[idx]
        disease = str(row['rare disease']).strip("[]'") if isinstance(row['rare disease'], str) else str(row['rare disease'][0])
        query = f"Which statement is TRUE about the rare disease {disease}?"
        
        # Get reformulated options
        options = reformulate_options_agent(row['input'])
        options = {k: options[k.lower()] for k in 'ABCD'}
        
        print_section(f"Reformulated Options: {query}")
        for k, v in options.items():
            print(f"{k}. {v}")
        
        chunks_analyzed = 0
        final_answer = "Unclear"
        final_evidence = {}
        source_url = ""
        remaining_options: Dict[str, str] = {k: v for k, v in options.items()}
        analyzed_urls = set()  # Track analyzed URLs
        
        while remaining_options and chunks_analyzed < 10:
            try:
                # Get keywords for remaining options
                if chunks_analyzed > 0:
                    keywords = keyword_agent(remaining_options, disease)
                    print_section("New Search Keywords")
                    print(", ".join(keywords))
                    
                    search_query = f"{disease} {' '.join(keywords)}"
                    new_chunks = await get_context_from_rag(search_query, disease, remaining_options)
                    
                    # Filter out already analyzed URLs
                    new_chunks = [c for c in new_chunks if c.split('\n')[0].replace('Source: ', '') not in analyzed_urls]
                    
                    if not new_chunks:
                        # Only stop if we've analyzed at least 5 chunks
                        if chunks_analyzed >= 5:
                            print("\nNo new relevant chunks found after analyzing 5+ chunks")
                            break
                        # Try broader search
                        print("\nNo new chunks found, trying broader search...")
                        new_chunks = await get_context_from_rag(disease, disease, remaining_options)
                        new_chunks = [c for c in new_chunks if c.split('\n')[0].replace('Source: ', '') not in analyzed_urls]
                        if not new_chunks:
                            break
                    
                    chunk = new_chunks[0]
                else:
                    all_chunks = await get_context_from_rag(query, disease, options)
                    if not all_chunks:
                        break
                    chunk = all_chunks[0]
                
                # Clean and process chunk
                chunk = clean_chunk(chunk)
                current_source = chunk.split('\n')[0].replace('Source: ', '')
                
                # Skip if we've already analyzed this URL
                if current_source in analyzed_urls:
                    continue
                
                chunks_analyzed += 1
                print_section(f"Analyzing Chunk ({chunks_analyzed}/10)")
                analyzed_urls.add(current_source)
                print(f"Chunk length: {len(chunk)}. Chunk preview: {chunk[:500]}")
                
                # Process options
                all_disproven = True
                disproven_this_round = set()
                
                # Check for process of elimination
                if len(remaining_options) == 1 and chunks_analyzed >= 2:
                    last_option = list(remaining_options.keys())[0]
                    print(f"\n✓ Selecting Option {last_option} as it's the only remaining option after disproving all others")
                    print("Multiple chunks analyzed without finding contradicting evidence.")
                    final_answer = last_option
                    final_evidence = {last_option: "Selected by process of elimination after analyzing multiple chunks"}
                    source_url = "Multiple sources"
                    break
                
                # Analyze each remaining option against current chunk
                for opt in list(remaining_options.keys()):
                    try:
                        valid, explanation = validation_agent(
                            statement=remaining_options[opt],
                            chunk=chunk,
                            rare_disease=disease
                        )
                        
                        if valid == 'True':
                            print(f"\n✓ Option {opt} is PROVEN TRUE")
                            print(explanation)
                            final_answer = opt
                            final_evidence = {opt: explanation}
                            source_url = current_source
                            remaining_options.clear()
                            all_disproven = False
                            break
                        elif valid == 'False':
                            print(f"\n✗ Option {opt} DISPROVEN")
                            print(explanation)
                            disproven_this_round.add(opt)
                        else:
                            print(f"\n? Option {opt}: Evidence inconclusive")
                            all_disproven = False
                    
                    except Exception as e:
                        print(f"\n! Error analyzing option {opt}: {str(e)}")
                        all_disproven = False
                        continue
                
                # Remove disproven options
                for opt in disproven_this_round:
                    if opt in remaining_options:
                        del remaining_options[opt]
                
                # If all options were disproven this round, restart analysis
                if all_disproven and disproven_this_round:
                    print_section("Validation Reset")
                    print("All options were marked as disproven - restarting analysis with all options")
                    remaining_options = {k: v for k, v in options.items()}
                    continue
                
            except Exception as e:
                print(f"\n! Error processing chunk: {str(e)}")
                continue
        
        # Record results
        correct_answer = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}.get(row['cop'], 'Unknown')
        
        print_section("Results")
        print(f"Final Answer: {final_answer}")
        print(f"Correct Answer: {correct_answer}")
        if final_evidence:
            print(f"\nEvidence:")
            print(final_evidence[final_answer])
        print(f"\nSource: {source_url or 'No conclusive source found'}")
        print(f"Chunks analyzed: {chunks_analyzed}")
        
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
        
        if idx % 5 == 0:
            pd.DataFrame(results).to_csv("results_rag.csv", index=False)
    
    pd.DataFrame(results).to_csv("results_rag.csv", index=False)

if __name__ == "__main__":
    asyncio.run(main())