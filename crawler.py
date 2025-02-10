import os
from typing import List, Dict
from dotenv import load_dotenv
import requests
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from openai import AsyncOpenAI
from supabase import create_client, Client
import tiktoken
import json
from pydantic_basemodels import ChunkScore
from ollama import chat

load_dotenv()

# Initialize clients
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

def chunk_text(text: str, chunk_size: int = 3000) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Try to find a paragraph break
        chunk = text[start:end]
        if '\n\n' in chunk:
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:
                end = start + last_break

        # If no paragraph break, try to break at a sentence
        elif '. ' in chunk:
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:
                end = start + last_period + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end
    return chunks

async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from OpenAI."""
    response = await openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


async def check_chunk_exists(url: str, chunk_number: int) -> bool:
    """Check if a chunk already exists in Supabase."""
    result = supabase.table("content_chunks").select("url").eq("url", url).eq("chunk_number", chunk_number).execute()
    return len(result.data) > 0

async def insert_chunk(chunk):
    """Insert a processed chunk into Supabase."""

    # Check if chunk already exists
    if await check_chunk_exists(chunk.url, chunk.chunk_number):
        return None
        
    data = {
        "url": chunk.url,
        "chunk_number": chunk.chunk_number,
        "title": chunk.title,
        "content": chunk.content,
        "embedding": chunk.embedding
    }
    
    result = supabase.table("content_chunks").upsert(
        data,
        on_conflict="url,chunk_number"
    ).execute()
    return result


def search_findzebra(query: str) -> Dict:
    """
    Search FindZebra API for rare disease information
    
    Args:
        query (str): The search query for the disease
        api_key (str): Your FindZebra API key
        
    Returns:
        Dict: JSON response from the API
    """
    base_url = "https://www.findzebra.com/api/v1/query"
    params = {
        "api_key": "c6213a1c-1092-4eb8-92a6-564b7068925e",
        "q": query,
        "response_format": "json",
        "rows": 3,
        "fl": "title,display_content,source,source_url,genes,cui,score"
    }
    response = requests.get(base_url, params=params)
    response.raise_for_status()  # Raise an exception for bad status codes
    return response.json()


async def crawl_url_with_browser(url: str):
    """Crawl a URL using browser-based crawler and process the content."""
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        timeout=30000,  # Reduce timeout to 30 seconds
        extra_args=[
            "--disable-gpu",
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-dev-shm-usage",
            "--disable-accelerated-2d-canvas",
            "--disable-gpu-sandbox",
            "--single-process"
        ]
    )
    crawl_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        timeout=30000  # Also set crawl timeout to 30 seconds
    )

    try:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(url, config=crawl_config)
            return result.html if result else ""
    except Exception as e:
        print(f"Crawler error: {str(e)}")
        return ""  # Return empty string on error


def truncate_to_token_limit(text: str, max_tokens: int = 8000) -> str:
    """Truncate text to stay within token limit."""
    enc = tiktoken.encoding_for_model("text-embedding-ada-002")
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])

async def extract_keywords(question: str, options: Dict) -> KeywordAnalysis:
    """Extract structured keywords from the question and options using LLM."""
    completion = chat(
        messages=[
            {
                "role": "system",
                "content": """Analyze the medical question and extract keywords in a structured way.
                Focus on:
                1. Required keywords that must be present to answer the question
                2. Supporting keywords that help validate answers
                3. Keywords that could help exclude certain options
                4. Option-specific keywords for each choice
                5. Important medical terms
                6. Time-related terms (onset, progression, etc.)
                
                Be specific and thorough in identifying keywords that will help find relevant content."""
            },
            {
                "role": "user",
                "content": f"Question: {question}\nOptions: {json.dumps(options, indent=2)}"
            }
        ],
        model="phi4:latest",
        format=KeywordAnalysis.model_json_schema()
    )
    return KeywordAnalysis(**json.loads(completion['message']['content']))

def calculate_keyword_score(content: str, keywords: KeywordAnalysis) -> tuple[float, List[str]]:
    """Calculate keyword-based score for a chunk of content."""
    content_lower = content.lower()
    matched_keywords = []
    score = 0.0
    
    # Check required keywords (highest weight)
    for kw in keywords.required_keywords:
        if kw.lower() in content_lower:
            score += 3.0
            matched_keywords.append(kw)
    
    # Check supporting keywords
    for kw in keywords.supporting_keywords:
        if kw.lower() in content_lower:
            score += 1.5
            matched_keywords.append(kw)
    
    # Check medical terms
    for kw in keywords.medical_terms:
        if kw.lower() in content_lower:
            score += 2.0
            matched_keywords.append(kw)
    
    # Check temporal terms
    for kw in keywords.temporal_terms:
        if kw.lower() in content_lower:
            score += 1.0
            matched_keywords.append(kw)
    
    # Check option-specific keywords
    for opt_kws in keywords.option_specific_keywords.values():
        for kw in opt_kws:
            if kw.lower() in content_lower:
                score += 1.0
                matched_keywords.append(kw)
    
    # Normalize score to 0-1 range
    max_possible_score = (
        len(keywords.required_keywords) * 3.0 +
        len(keywords.supporting_keywords) * 1.5 +
        len(keywords.medical_terms) * 2.0 +
        len(keywords.temporal_terms) * 1.0 +
        sum(len(kws) for kws in keywords.option_specific_keywords.values()) * 1.0
    )
    
    normalized_score = score / max_possible_score if max_possible_score > 0 else 0
    return normalized_score, matched_keywords

async def gather_context_for_question(search_query: str, options: Dict, excluded_terms: List[str] = None) -> List[str]:
    """Gather context using hybrid embedding-based and keyword-based approach."""
    try:
        # Extract structured keywords
        keywords = await extract_keywords(search_query, options)
        
        # Get results from FindZebra
        findzebra_results = search_findzebra(search_query)
        scored_chunks = []

        for doc in findzebra_results.get("response", {}).get("docs", []):
            content = doc["display_content"]
            
            # Skip content containing excluded terms
            if excluded_terms and any(term.lower() in content.lower() for term in excluded_terms):
                continue
            
            # Skip if this chunk is too similar to existing ones
            if any(chunk.content == content for chunk in scored_chunks):
                continue
            
            # Truncate content
            truncated_content = truncate_to_token_limit(content)
            
            try:
                # Get embedding score
                query_embedding = await get_embedding(truncated_content)
                embedding_results = supabase.rpc(
                    'match_chunks',
                    {
                        'query_embedding': query_embedding,
                        'match_count': 1
                    }
                ).execute()
                
                if embedding_results.data:
                    for chunk in embedding_results.data:
                        # Skip excluded content
                        if excluded_terms and any(term.lower() in chunk["content"].lower() for term in excluded_terms):
                            continue
                            
                        # Calculate keyword score
                        keyword_score, matched_kws = calculate_keyword_score(chunk["content"], keywords)
                        
                        # Calculate combined score (equal weight to both approaches)
                        embedding_score = float(chunk["similarity"])
                        combined_score = (embedding_score + keyword_score) / 2
                        
                        scored_chunk = ChunkScore(
                            content=truncate_to_token_limit(chunk["content"]),
                            embedding_score=embedding_score,
                            keyword_score=keyword_score,
                            combined_score=combined_score,
                            matched_keywords=matched_kws
                        )
                        scored_chunks.append(scored_chunk)
                        
            except Exception as e:
                print(f"Error processing chunk: {str(e)}")
                continue
        
        # Sort chunks by combined score and take top results
        scored_chunks.sort(key=lambda x: x.combined_score, reverse=True)
        top_chunks = scored_chunks[:3]  # Take top 3 chunks
        
        # Print scoring details for debugging
        for i, chunk in enumerate(top_chunks):
            print(f"\nChunk {i+1} scores:")
            print(f"Embedding score: {chunk.embedding_score:.3f}")
            print(f"Keyword score: {chunk.keyword_score:.3f}")
            print(f"Combined score: {chunk.combined_score:.3f}")
            print(f"Matched keywords: {', '.join(chunk.matched_keywords)}")
        
        return [chunk.content for chunk in top_chunks]
        
    except Exception as e:
        print(f"Error gathering context: {str(e)}")
        return []

def similar_text(text1: str, text2: str) -> float:
    """Calculate similarity ratio between two texts using difflib."""
    from difflib import SequenceMatcher
    return SequenceMatcher(None, text1, text2).ratio() 