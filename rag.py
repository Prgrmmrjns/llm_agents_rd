import os
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from duckduckgo_search import DDGS
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

# Initialize OpenAI client and paths
load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
DATA_DIR = Path("data")
CHUNKS_DIR = DATA_DIR / "chunks"
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class TextChunk:
    content: str
    source_url: str
    query: str = ''

def get_embedding(text: str) -> List[float]:
    """Get embedding vector from OpenAI."""
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return list(response.data[0].embedding)

async def crawl_parallel(urls: List[str]) -> List[Dict[str, Any]]:
    """Fast parallel crawling of URLs."""
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-javascript", "--disable-images"]
    )
    
    results = []
    async with AsyncWebCrawler(config=browser_config) as crawler:
        tasks = [crawler.arun(
            url=url, 
            config=CrawlerRunConfig(cache_mode=CacheMode.BYPASS, page_timeout=10000),
            session_id=f"s_{i}"
        ) for i, url in enumerate(urls)]
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        results = [
            {'text': r.markdown, 'url': url} 
            for url, r in zip(urls, responses)
            if not isinstance(r, Exception) and r.success
        ]
    
    return results

async def search_duckduckgo(query: str) -> List[Dict[str, str]]:
    """Search DuckDuckGo and crawl results."""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=3))
        urls = [r.get('link') or r.get('url') or r.get('href', '') 
                for r in results 
                if (r.get('link') or r.get('url') or r.get('href', '')).startswith(('http://', 'https://'))][:3]
        return await crawl_parallel(urls) if urls else []

def process_chunks(text: str, url: str, query: str = '') -> List[TextChunk]:
    """Split text into chunks."""
    if len(text) < 200:
        return []

    paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 100][:2]
    return [TextChunk(content="\n\n".join(paragraphs), source_url=url, query=query)]

async def get_context_from_rag(query: str, disease: str, options: Dict[str, str]) -> List[str]:
    """Get relevant context chunks."""
    chunks_dir = DATA_DIR / "chunks" / disease.replace(" ", "_").lower()
    embeddings_dir = DATA_DIR / "embeddings" / disease.replace(" ", "_").lower()
    
    query_embedding = np.array(get_embedding(query))
    
    # Load and rank existing chunks
    chunk_scores = []
    for emb_file in embeddings_dir.glob("embedding_*.npy"):
        chunk_file = chunks_dir / f"chunk_{emb_file.stem.replace('embedding_', '')}.json"
        if chunk_file.exists():
            chunk_embedding = np.load(emb_file)
            similarity = np.dot(query_embedding, chunk_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
            )
            if not np.isnan(similarity):
                with open(chunk_file, 'r') as f:
                    chunk = json.load(f)
                chunk_scores.append((similarity, chunk))
    
    if not chunk_scores:
        # If no relevant chunks found in cache, build new ones
        chunks = await build_rag(query, disease)
        if not chunks:  # If build_rag failed to get chunks
            # Try a broader search
            print("\nTrying broader search...")
            chunks = await build_rag(disease, disease)
        return chunks
    
    # Return top chunks
    chunk_scores.sort(key=lambda x: x[0], reverse=True)
    return [
        f"Source: {chunk['source_url']}\n{chunk['content']}\nSimilarity: {score:.3f}"
        for score, chunk in chunk_scores[:10]
    ]

async def build_rag(query: str, disease: str) -> List[str]:
    """Build new RAG database."""
    disease_dir = CHUNKS_DIR / disease.replace(" ", "_").lower()
    embeddings_dir = DATA_DIR / "embeddings" / disease.replace(" ", "_").lower()
    disease_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    # Get chunks from search
    results = await search_duckduckgo(f"{disease} {query}")
    if not results:
        # Try searching just the disease name
        results = await search_duckduckgo(disease)
    
    chunks = []
    for result in results:
        chunks.extend(process_chunks(result['text'], result['url'], query))
    
    if not chunks:
        return []
        
    # Process and store chunks
    formatted_chunks = []
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk.content)
        if len(embedding) != 1536:
            continue
            
        chunk_data = {
            "content": chunk.content,
            "source_url": chunk.source_url,
        }
        
        with open(disease_dir / f"chunk_{i}.json", 'w') as f:
            json.dump(chunk_data, f)
        np.save(embeddings_dir / f"embedding_{i}.npy", embedding)
        
        formatted_chunks.append(f"Source: {chunk.source_url}\n{chunk.content}")
    
    return formatted_chunks