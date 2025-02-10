import os
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
from duckduckgo_search import DDGS
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
import os
from typing import List, Dict, Any, Set
from dataclasses import dataclass
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from dataclasses import dataclass

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Create data directory if it doesn't exist
DATA_DIR = Path("data")
CHUNKS_DIR = DATA_DIR / "chunks"
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

# Global set for tracking used URLs
used_urls: Set[str] = set()

def add_used_url(url: str) -> None:
    """Add a URL to the global set of used URLs."""
    used_urls.add(url)

def is_url_used(url: str) -> bool:
    """Check if a URL has already been used."""
    return url in used_urls

@dataclass
class TextChunk:
    content: str
    embedding: List[float]
    metadata: Dict[str, Any]

def get_embedding(text: str) -> List[float]:
    """Get embedding vector from OpenAI."""
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    embedding = response.data[0].embedding
    # Ensure we return a list, not ndarray
    return list(embedding) if isinstance(embedding, np.ndarray) else embedding

async def crawl_parallel(urls: List[str], max_concurrent: int = 5) -> List[Dict[str, Any]]:
    """Fast parallel crawling with minimal overhead."""
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=[
            "--disable-gpu",
            "--disable-dev-shm-usage",
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-extensions",
            "--disable-javascript",  # Faster loading
            "--disable-images"  # Skip image loading
        ],
    )
    crawl_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        page_timeout=10000  # 10 second timeout
    )

    processed_results = []
    crawler = None
    
    try:
        # Process all URLs in a single batch
        crawler = AsyncWebCrawler(config=browser_config)
        await crawler.start()
        
        tasks = [crawler.arun(url=url, config=crawl_config, session_id=f"s_{i}") 
                for i, url in enumerate(urls)]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for url, result in zip(urls, results):
            if not isinstance(result, Exception) and result.success:
                processed_results.append({
                    'text': result.markdown,
                    'url': url,
                    'source_type': 'duckduckgo'
                })

    finally:
        if crawler:
            try:
                await crawler.close()
            except:
                pass

    return processed_results

async def search_duckduckgo(disease: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Fast DuckDuckGo search with minimal results."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(
                f"{disease} symptoms clinical features",
                max_results=max_results,
                region='wt-wt'
            ))
            
            # Get valid URLs
            urls = []
            for result in results:
                link = (result.get('link') or result.get('url') or result.get('href', ''))
                if link and link.startswith(('http://', 'https://')):
                    urls.append(link)
                    if len(urls) >= 5:  # Hard limit at 5 URLs for speed
                        break
            
            if not urls:
                return []
            
            return await crawl_parallel(urls, max_concurrent=5)
            
    except Exception as e:
        print(f"Search error: {str(e)}")
        return []

def process_search_results(text: str, url: str, disease: str, max_chunk_size: int = 3000, source_type: str = 'unknown') -> List[Dict[str, Any]]:
    """Process search results with optimized chunking."""
    
    
    @dataclass
    class Chunk:
        content: str
        source_url: str
        source_type: str
        query: str

    # Skip if text is too short
    if len(text) < 200:
        return []

    # Split into paragraphs and filter out short ones
    paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 100]
    
    # Combine short paragraphs
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        if len(current_chunk) + len(para) < max_chunk_size:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(Chunk(
                    content=current_chunk.strip(),
                    source_url=url,
                    source_type=source_type,
                    query=disease
                ))
            current_chunk = para + "\n\n"
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(Chunk(
            content=current_chunk.strip(),
            source_url=url,
            source_type=source_type,
            query=disease
        ))
    
    return chunks[:5]  # Limit to 5 chunks per URL for speed

def save_chunks_locally(chunks: List[Dict], disease: str):
    """Save chunks to local JSON files."""
    # Create disease-specific directory
    disease_dir = CHUNKS_DIR / disease.replace(" ", "_").lower()
    disease_dir.mkdir(exist_ok=True)
    
    # Save each chunk in a separate file
    for i, chunk in enumerate(chunks):
        chunk_file = disease_dir / f"chunk_{i}.json"
        with open(chunk_file, 'w', encoding='utf-8') as f:
            json.dump(chunk, f, ensure_ascii=False, indent=2)

def load_chunks_for_disease(disease: str) -> List[Dict]:
    """Load chunks for a specific disease from local storage."""
    disease_dir = CHUNKS_DIR / disease.replace(" ", "_").lower()
    if not disease_dir.exists():
        return []
    
    chunks = []
    for chunk_file in disease_dir.glob("chunk_*.json"):
        with open(chunk_file, 'r', encoding='utf-8') as f:
            chunks.append(json.load(f))
    return chunks
