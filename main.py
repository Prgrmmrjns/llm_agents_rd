import pandas as pd
import json
import numpy as np
from typing import List, Dict
import asyncio
from pathlib import Path
from datetime import datetime

from agents import (
    answer_agent,
    research_agent,
    search_agent,
    reformulate_options_agent
)
from rag import (
    search_duckduckgo,
    process_search_results,
    get_embedding
)

async def build_question_specific_rag(query: str, disease: str, options: Dict[str, str]) -> List[str]:
    """
    Build a RAG database specific to the question using keywords from search_agent.
    Only uses DuckDuckGo as the search source and stores embeddings locally.
    
    Args:
        query: The question being asked
        disease: The rare disease name
        options: The answer options
        
    Returns:
        List of formatted context chunks
    """
    # Create directories if they don't exist
    data_dir = Path("data")
    chunks_dir = data_dir / "chunks"
    embeddings_dir = data_dir / "embeddings"
    disease_dir = chunks_dir / disease.replace(" ", "_").lower()
    disease_embeddings_dir = embeddings_dir / disease.replace(" ", "_").lower()
    
    for directory in [data_dir, chunks_dir, embeddings_dir, disease_dir, disease_embeddings_dir]:
        directory.mkdir(exist_ok=True, parents=True)
    
    # Get search keywords using search_agent
    search_keywords = search_agent(query, options)
    
    # Combine all relevant keywords
    all_keywords = (
        search_keywords['general_keywords'][:2] +  # Top 2 general keywords
        search_keywords['temporal_keywords'][:1] +  # Top temporal keyword
        search_keywords['demographic_keywords'][:1]  # Top demographic keyword
    )
    
    # Remove duplicates while preserving order
    unique_keywords = list(dict.fromkeys(all_keywords))
    print(f"\nUsing keywords: {unique_keywords}")
    
    processed_urls = set()
    all_chunks = []
    
    # Search DuckDuckGo with keywords
    for keyword in unique_keywords[:3]:  # Use top 3 keywords for speed
        search_query = f"{disease} {keyword}"
        duckduckgo_results = await search_duckduckgo(search_query, max_results=5)
        for result in duckduckgo_results:
            if result['url'] not in processed_urls:
                processed_urls.add(result['url'])
                chunks = process_search_results(
                    result['text'],
                    result['url'],
                    disease,
                    max_chunk_size=3000,  # Smaller chunks for better relevance
                    source_type='duckduckgo'
                )
                all_chunks.extend(chunks)
    
    # Process chunks and store locally
    formatted_chunks = []
    chunks_data = []
    
    for i, chunk in enumerate(all_chunks[:20]):  # Limit to 20 chunks for speed
        try:
            # Get embedding for the chunk
            embedding = get_embedding(chunk.content)
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            
            if len(embedding) != 1536:  # Skip if embedding is invalid
                continue
                
            # Create chunk data
            chunk_data = {
                "id": f"{disease}_{i}",
                "content": chunk.content,
                "rare_disease": disease,
                "source_url": chunk.source_url,
                "source_type": chunk.source_type,
                "embedding": embedding,
                "metadata": {
                    "query": query,
                    "keywords": unique_keywords,
                    "chunk_type": "question_specific",
                    "processed_at": datetime.now().isoformat()
                }
            }
            
            # Save chunk data
            chunk_file = disease_dir / f"chunk_{i}.json"
            with open(chunk_file, 'w', encoding='utf-8') as f:
                json.dump(chunk_data, f, ensure_ascii=False, indent=2)
            
            # Save embedding separately for faster similarity search
            embedding_file = disease_embeddings_dir / f"embedding_{i}.npy"
            np.save(embedding_file, np.array(embedding))
            
            chunks_data.append(chunk_data)
            
            # Format chunk for return
            source_info = f"Source: {chunk.source_type.upper()} - {chunk.source_url}"
            formatted_chunk = f"{source_info}\n{chunk.content}\n"
            formatted_chunks.append(formatted_chunk)
            
        except Exception as e:
            print(f"Error processing chunk: {str(e)}")
            continue
    
    # Save metadata index
    index_file = disease_dir / "index.json"
    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump({
            "disease": disease,
            "chunk_count": len(chunks_data),
            "last_updated": datetime.now().isoformat(),
            "chunks": [{"id": c["id"], "source_url": c["source_url"]} for c in chunks_data]
        }, f, indent=2)
    
    return formatted_chunks

async def get_context_from_rag(query: str, disease: str, options: Dict[str, str]) -> List[str]:
    """
    Get relevant context chunks from local storage or build new ones.
    Uses numpy for fast similarity search.
    
    Args:
        query: The query to search for
        disease: The rare disease name
        options: The answer options
        
    Returns:
        List of formatted context chunks
    """
    # Setup paths
    chunks_dir = Path("data/chunks") / disease.replace(" ", "_").lower()
    embeddings_dir = Path("data/embeddings") / disease.replace(" ", "_").lower()
    
    # Get query embedding
    query_embedding = get_embedding(query)
    if isinstance(query_embedding, np.ndarray):
        query_embedding = query_embedding.tolist()
    query_embedding = np.array(query_embedding)
    
    print(f"Searching for disease: {disease}")
    
    # Check if we have existing chunks
    if not chunks_dir.exists() or not embeddings_dir.exists():
        print("No existing chunks found. Building new ones...")
        chunks = await build_question_specific_rag(query, disease, options)
        return chunks
    
    # Load all embeddings and calculate similarities
    similarities = []
    chunk_files = []
    
    for embedding_file in embeddings_dir.glob("embedding_*.npy"):
        chunk_id = embedding_file.stem.replace("embedding_", "")
        chunk_file = chunks_dir / f"chunk_{chunk_id}.json"
        
        if chunk_file.exists():
            chunk_embedding = np.load(embedding_file)
            similarity = np.dot(query_embedding, chunk_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
            )
            # Skip chunks with NaN similarity
            if not np.isnan(similarity):
                similarities.append(similarity)
                chunk_files.append(chunk_file)
    
    if not similarities:
        print("No valid chunks found. Building new ones...")
        chunks = await build_question_specific_rag(query, disease, options)
        return chunks
    
    # Sort by similarity
    sorted_indices = np.argsort(similarities)[::-1]
    top_indices = sorted_indices[:10]  # Get top 10 most similar chunks
    
    # Load and format chunks
    formatted_chunks = []
    for idx in top_indices:
        with open(chunk_files[idx], 'r', encoding='utf-8') as f:
            chunk_data = json.load(f)
        
        source_info = f"Source: {chunk_data['source_type'].upper()} - {chunk_data['source_url']}"
        formatted_chunk = (
            f"{source_info}\n"
            f"{chunk_data['content']}\n"
            f"Similarity Score: {similarities[idx]:.3f}"
        )
        formatted_chunks.append(formatted_chunk)
        
        print(f"Chunk from {chunk_data['source_type'].upper()} - Similarity: {similarities[idx]:.3f}")
    
    return formatted_chunks

async def process_question(row: pd.Series) -> Dict:
    """Process a single question and return the result."""
    original_query = row['input']
    print(f"Original question: {original_query}")
    
    # Extract disease and ensure it's a string
    disease_value = row['rare disease']
    if isinstance(disease_value, (list, np.ndarray)):
        disease_value = str(disease_value[0]) if len(disease_value) > 0 else str(disease_value)
    disease = str(disease_value).strip("[]'")
    
    # Reformulate the question for clarity
    query = f"Which statement is TRUE about the rare disease {disease}?"
    print(f"Reformulated question: {query}")
    
    # Get reformulated options
    reformulated_options = reformulate_options_agent(f"{query}\n\nOriginal question and options:\n{original_query}")
    print("\nReformatted Options for Analysis:")
    for option, text in reformulated_options.items():
        print(f"Option {option.upper()}: {text}")
    
    # Use reformulated options for further processing
    options = {
        'A': reformulated_options['a'],
        'B': reformulated_options['b'],
        'C': reformulated_options['c'],
        'D': reformulated_options['d']
    }
    
    # Get context chunks from RAG database
    all_chunks = await get_context_from_rag(query, disease, options)
    
    # Process chunks with research agent
    final_answer = "Unclear"
    final_explanation = "No chunks analyzed yet."
    chunks_analyzed = 0
    accumulated_evidence = ""
    
    # Process each chunk with research agent until we get a clear answer
    for i, chunk in enumerate(all_chunks, 1):
        print(f"\nAnalyzing chunk {i}/{len(all_chunks)}...")
        
        # Analyze the chunk with explicit logical context
        analysis = research_agent(
            chunk, 
            query,
            f"{accumulated_evidence}\n\nAnalyze these statements about {disease}:\n" + 
            "\n".join(f"{k.upper()}: {v}" for k, v in options.items())
        )
        
        # Add source information to the analysis
        analysis['source_info'] = chunk
        
        # Update accumulated evidence with logical context
        accumulated_evidence = analysis['accumulated_evidence']
        
        # Print analysis details with logical context
        chunk_lines = chunk.split('\n')
        print(f"Source: {chunk_lines[0]}")
        print(f"{chunk_lines[2] if len(chunk_lines) > 2 else ''}")
        print("Evidence analysis:")
        for option in ['A', 'B', 'C', 'D']:
            print(f"Option {option}:")
            print(f"  Evidence proving TRUE: {analysis['evidence_by_option'][option]}")
            print(f"  Evidence proving FALSE: {analysis['counter_evidence_by_option'][option]}")
        print(f"Accumulated evidence: {accumulated_evidence}")
        
        # Try to get an answer based on this chunk with logical context
        answer, explanation = answer_agent(
            {
                'input': query,
                'original_question': original_query,
                'reformulated_options': options,
            },
            {
                **analysis
            }
        )
        
        print(f"Current decision: {answer}")
        print(f"Reasoning: {explanation}")
        
        chunks_analyzed += 1
        
        if answer != "Unclear":
            final_answer = answer
            final_explanation = explanation
            break
    
    # If we've gone through all chunks and still unclear, make best guess from last explanation
    if final_answer == "Unclear" and chunks_analyzed > 0:
        print("\nProcessed all chunks but no clear answer found. Using best available evidence...")
    else:
        print(f"\nFound clear answer after analyzing {chunks_analyzed} chunks.")
    
    # Get correct answer
    answer_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    correct_answer = answer_dict.get(row['cop'], 'Unknown')
    
    # Print results with logical context
    print(f'Chosen answer: {final_answer}. Correct answer: {correct_answer}')
    print(f'Explanation: {final_explanation}\n')
    
    return {
        'question': query,
        'original_question': original_query,
        'reformulated_options': options,
        'llm_answer': final_answer,
        'correct_answer': correct_answer,
        'rationale': final_explanation,
        'num_chunks_analyzed': chunks_analyzed,
        'total_chunks': len(all_chunks)
    }

async def main():
    """Main function to process questions and save results."""
    # Load dataset
    df = pd.read_parquet("hf://datasets/guan-wang/ReDis-QA/data/test-00000-of-00001.parquet")
    
    # Initialize results storage
    results = []
    
    # Process questions
    for idx in range(20):  # Process first 20 questions
        print(f"\nProcessing question {idx + 1}")
        
        result = await process_question(df.iloc[idx])
        result['index'] = idx
        results.append(result)
        
        # Save results after each question
        pd.DataFrame(results).to_csv("results_rag.csv", index=False)

if __name__ == "__main__":
    asyncio.run(main())