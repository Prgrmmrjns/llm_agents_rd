import asyncio
from pathlib import Path
from typing import List, Dict, Any, Iterator
from dataclasses import dataclass
from duckduckgo_search import DDGS
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

@dataclass
class TextChunk:
    content: str
    source_url: str
    query: str = ''

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

def chunk_text(text: str, min_size: int = 2000, max_size: int = 5000) -> List[str]:
    """Split text into coherent chunks between min_size and max_size characters."""
    # Clean text first
    text = clean_text(text)
    
    # Split into paragraphs first
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for paragraph in paragraphs:
        # If adding this paragraph would exceed max_size and we're above min_size,
        # start a new chunk
        if current_size + len(paragraph) > max_size and current_size >= min_size:
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
        if len(chunks) > 0 and current_size < min_size:
            last_chunk = chunks.pop()
            merged = last_chunk + ' ' + ' '.join(current_chunk)
            chunks.append(merged)
        else:
            chunks.append(' '.join(current_chunk))
    
    return chunks

async def fetch_and_parse_url(url: str, max_chunks_per_source: int = 5) -> List[str]:
    """Fetch URL content and split into chunks."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124'}
        response = requests.get(url, timeout=10, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove non-content elements
        for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
            element.decompose()
        
        # Get main content
        content_tags = soup.find_all(['article', 'main', 'div'], 
                                   class_=lambda x: x and any(word in str(x).lower() 
                                   for word in ['content', 'article', 'main', 'text']))
        
        text = ' '.join(tag.get_text() for tag in content_tags) if content_tags else soup.get_text()
        text = clean_text(text)
        
        if not text.strip():
            return []
        
        # Create larger chunks
        chunks = chunk_text(text, min_size=2000, max_size=8000)[:max_chunks_per_source]
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
    """Build RAG context from search results."""
    query = " ".join([disease] + keywords[:3])
    
    try:
        with DDGS() as ddgs:
            search_results = list(ddgs.text(query, max_results=10))
            
            # Get unique domains
            unique_urls = set()
            for result in search_results:
                url = (result.get('link') or result.get('url') or 
                      result.get('href') or result.get('source'))
                
                if url and url.startswith(('http://', 'https://')):
                    domain = '/'.join(url.split('/')[:3])
                    if domain not in ['/'.join(u.split('/')[:3]) for u in unique_urls]:
                        unique_urls.add(url)
                        if len(unique_urls) >= max_results:
                            break
            
            if not unique_urls:
                return []
            
            # Fetch and parse URLs concurrently
            tasks = [fetch_and_parse_url(url) for url in unique_urls]
            chunks_list = await asyncio.gather(*tasks)
            
            # After getting chunks, rank them by similarity
            all_chunks = [chunk for chunks in chunks_list for chunk in chunks]
            ranked_chunks = rank_chunks_by_similarity(all_chunks, [disease] + keywords)
            
            return ranked_chunks
            
    except Exception as e:
        print(f"Search error: {str(e)}")
        return []