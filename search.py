import requests
from typing import Dict, List, Tuple
import time

def search_findzebra(query: str) -> List[Dict]:
    """
    Search FindZebra API for rare disease information with expanded results
    
    Args:
        query (str): The search query for the disease
        
    Returns:
        List[Dict]: List of JSON responses from the API for different search variations
    """
    base_url = "https://www.findzebra.com/api/v1/query"
    
    # Create variations of the search query
    query_variations = [
        query,  # exact disease name
        f"{query} symptoms treatment diagnosis",  # clinical focus
        f"{query} pathophysiology genetics",  # scientific focus
        f"{query} epidemiology prognosis",  # outcomes focus
    ]
    
    all_results = []
    seen_urls = set()  # Track unique URLs to avoid duplicates
    
    for variation in query_variations:
        try:
            params = {
                "api_key": "c6213a1c-1092-4eb8-92a6-564b7068925e",
                "q": variation,
                "response_format": "json",
                "rows": 50,  # Increased from 30 to 50
                "fl": "title,display_content,source,source_url,genes,cui,score"
            }
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            results = response.json()
            
            # Add only unique results
            for doc in results['response']['docs']:
                if doc['source_url'] not in seen_urls:
                    seen_urls.add(doc['source_url'])
                    all_results.append(doc)
            
            # Respect API rate limits
            time.sleep(1)
            
        except Exception as e:
            print(f"Error searching FindZebra with query '{variation}': {str(e)}")
            continue
    
    return all_results

def format_results(results: List[Dict]) -> List[Tuple[str, str]]:
    """
    Format and clean the search results
    
    Args:
        results (List[Dict]): List of documents from FindZebra API
    
    Returns:
        List[Tuple[str, str]]: List of (cleaned_text, source_url) pairs
    """
    formatted_results = []
    
    for doc in results:
        try:
            text = doc['display_content']
            source_url = doc['source_url']
            
            # Remove HTML tags for cleaner output
            text = text.replace("<br>", "\n").replace("<p>", "\n").replace("</p>", "")
            text = text.replace("<h2>", "\n").replace("</h2>", "\n")
            text = text.replace("<h3>", "\n").replace("</h3>", "\n")
            text = text.replace("<i>", "\n").replace("</i>", "\n")
            text = text.replace("<div>", "").replace("</div>", "")
            text = text.replace("&amp;", "&")
            
            # Add title if available
            if 'title' in doc:
                text = f"{doc['title']}\n\n{text}"
            
            formatted_results.append((text, source_url))
            
        except Exception as e:
            print(f"Error formatting result: {str(e)}")
            continue
    
    return formatted_results