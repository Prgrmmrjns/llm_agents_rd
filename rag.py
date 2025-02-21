import asyncio
from pathlib import Path
from typing import List, Dict, Any, Iterator, Optional
from dataclasses import dataclass
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from bs4 import BeautifulSoup
import requests
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize

# Initialize OpenAI client and paths
DATA_DIR = Path("data")
CHUNKS_DIR = DATA_DIR / "chunks"
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

# Add after the existing imports
model = SentenceTransformer('all-MiniLM-L6-v2')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Add this configuration section after the initial constants
FINDZEBRA_API_URL = "https://www.findzebra.com/api/v1/query"
FINDZEBRA_API_KEY = "c6213a1c-1092-4eb8-92a6-564b7068925e"  # Replace with your actual API key

@dataclass
class TextChunk:
    content: str
    source_url: str
    query: str = ''

async def crawl_parallel(urls: List[str], max_concurrent: int = 5) -> List[Dict[str, str]]:
    """Crawl multiple URLs in parallel with a concurrency limit."""
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
    results = []

    # Create the crawler instance
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_url(url: str):
            async with semaphore:
                result = await crawler.arun(
                    url=url,
                    config=crawl_config,
                    session_id="session1"
                )
                if result.success:
                    print(f"Successfully crawled: {url}")
                    results.append({
                        'text': result.markdown_v2.raw_markdown,
                        'url': url
                    })
                else:
                    print(f"Failed: {url} - Error: {result.error_message}")
        
        # Process all URLs in parallel with limited concurrency
        await asyncio.gather(*[process_url(url) for url in urls])
        return results
    finally:
        await crawler.close()

async def search_findzebra(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """Search FindZebra API and return results."""
    params = {
        "api_key": FINDZEBRA_API_KEY,
        "q": query,
        "response_format": "json",
        "rows": max_results,
        "fl": "title,display_content,source_url,genes,source"
    }
    
    try:
        response = requests.get(FINDZEBRA_API_URL, params=params)
        response.raise_for_status()
        results = response.json()
        
        # Transform the results into the expected format
        processed_results = []
        for result in results.get("response", {}).get("docs", []):
            processed_results.append({
                'text': f"{result.get('title', '')}\n\n{result.get('display_content', '')}",
                'url': result.get('source_url', ''),
                'genes': result.get('genes', []),
                'source': result.get('source', '')
            })
        return processed_results
    except Exception as e:
        print(f"FindZebra API error: {str(e)}")
        return []

def process_chunks(text: str, url: str, query: str = '') -> List[TextChunk]:
    """Split text into chunks."""

    paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 100][:2]
    return [TextChunk(content="\n\n".join(paragraphs), source_url=url, query=query)]

def clean_text(text: str) -> str:
    """Clean extracted text by removing extra whitespace and normalizing newlines."""
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\n\n', '\n').strip()
    return text

def get_sentences(text: str) -> List[str]:
    """Extract sentences from text using NLTK."""
    return sent_tokenize(text)

def group_sentences_by_similarity(sentences: List[str], 
                                target_chunk_size: int = 5) -> Iterator[List[str]]:
    """Group sentences into semantically coherent chunks."""
    if not sentences:
        return []
        
    # Get embeddings for all sentences
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
    current_chunk = []
    current_size = 0
    
    for i, sentence in enumerate(sentences):
        current_chunk.append(sentence)
        current_size += 1
        
        if current_size >= target_chunk_size:
            # Check if next sentence is semantically similar
            if i + 1 < len(sentences):
                current_embedding = sentence_embeddings[i]
                next_embedding = sentence_embeddings[i + 1]
                
                similarity = cosine_similarity(
                    current_embedding.cpu().numpy().reshape(1, -1),
                    next_embedding.cpu().numpy().reshape(1, -1)
                )[0][0]
                
                # If similarity is low or we're at max size, yield chunk
                if similarity < 0.7 or current_size >= target_chunk_size * 1.5:
                    yield current_chunk
                    current_chunk = []
                    current_size = 0
                    
    # Yield any remaining sentences
    if current_chunk:
        yield current_chunk

def chunk_text(text: str, max_size: int = 2000) -> List[str]:
    """Split text into coherent chunks up to max_size characters."""
    # Clean text first
    text = clean_text(text)
    
    # Split into paragraphs first
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for paragraph in paragraphs:
        # If adding this paragraph would exceed max_size,
        # start a new chunk
        if current_size + len(paragraph) > max_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_size = 0
            
        current_chunk.append(paragraph)
        current_size += len(paragraph)
        
        # If we're above max_size, force start a new chunk
        if current_size > max_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_size = 0
    
    # Add any remaining content
    if current_chunk:
        # If the last chunk is too small, merge it with the previous chunk
        if len(chunks) > 0:
            last_chunk = chunks.pop()
            merged = last_chunk + ' ' + ' '.join(current_chunk)
            chunks.append(merged)
        else:
            chunks.append(' '.join(current_chunk))
    
    return chunks

async def fetch_and_parse_url(url: str, max_chunks_per_source: int = 5) -> List[str]:
    """Fetch URL content and split into chunks using Crawl4AI."""
    try:
        # Create an instance of AsyncWebCrawler
        async with AsyncWebCrawler() as crawler:
            # Run the crawler on the URL
            result = await crawler.arun(url=url)
            
            if not result.success:
                print(f"Failed to crawl {url}")
                return []
            
            # Get the markdown content
            text = result.markdown
            if not text.strip():
                return []
            
            # Create larger chunks
            chunks = chunk_text(text, max_size=2000)[:max_chunks_per_source]
            return [f"Source: {url}\n{chunk}" for chunk in chunks]
            
    except Exception as e:
        print(f"Error processing {url}: {str(e)}")
        return []

def rank_chunks_by_similarity(chunks: List[str], keywords: List[str]) -> List[str]:
    """Rank chunks by similarity to keywords."""
    if not chunks:
        return []
        
    # Create query embedding from keywords
    query = " ".join(keywords)
    query_embedding = model.encode([query], convert_to_tensor=True)
    
    # Create chunk embeddings
    chunk_embeddings = model.encode(chunks, convert_to_tensor=True)
    
    # Calculate similarities
    similarities = cosine_similarity(
        query_embedding.cpu().numpy(),
        chunk_embeddings.cpu().numpy()
    )[0]
    
    # Sort chunks by similarity
    ranked_pairs = sorted(zip(similarities, chunks), reverse=True)
    return [chunk for _, chunk in ranked_pairs]

async def build_rag(keywords: List[str], disease: str, max_results: int = 3) -> List[str]:
    """Build RAG context from FindZebra search results."""
    query = " ".join([disease] + keywords[:3])
    
    try:
        # Search FindZebra
        search_results = await search_findzebra(query, max_results=max_results)
        
        if not search_results:
            return []
        
        # Get URLs to crawl
        urls = [result['url'] for result in search_results]
        
        # Crawl URLs in parallel
        crawled_results = await crawl_parallel(urls)
        
        # Process each result into chunks
        all_chunks = []
        for result in crawled_results:
            text = clean_text(result['text'])
            chunks = chunk_text(text, max_size=2000)
            
            # Add source information to each chunk
            source_chunks = [
                f"Source: {result['url']}\n{chunk}" 
                for chunk in chunks
            ]
            all_chunks.extend(source_chunks)
        
        # Rank chunks by similarity
        ranked_chunks = rank_chunks_by_similarity(all_chunks, [disease] + keywords)
        return ranked_chunks
            
    except Exception as e:
        print(f"Search error: {str(e)}")
        return []