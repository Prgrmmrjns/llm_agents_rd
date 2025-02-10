import sys
from typing import List, Dict, Tuple
from duckduckgo_search import DDGS
import asyncio
from crawl4ai import AsyncWebCrawler
from pathlib import Path
import html2text

async def fetch_webpage_content(url: str) -> Tuple[str, str, str]:
    """
    Fetch and parse webpage content using Crawl4AI.
    
    Args:
        url: The URL to fetch
        
    Returns:
        Tuple of (cleaned markdown content, raw HTML, error message if any)
    """
    try:
        # Initialize crawler with configuration
        crawler = AsyncWebCrawler(
            lazy_loading=True,  # Handle dynamic content
            cache_mode='memory',  # Faster processing
            extract_metadata=True  # Get additional page info
        )
        
        async with crawler as c:
            # Crawl the webpage
            result = await c.arun(url=url)
            
            # Get the markdown content
            markdown_content = result.markdown
            
            # Get the raw HTML
            raw_html = result.html
            
            return markdown_content.strip(), raw_html, ""
            
    except Exception as e:
        error_msg = f"Error fetching webpage: {str(e)}"
        print(f"Error details: {e.__class__.__name__}: {str(e)}")  # Add more error details
        return "", "", error_msg

def save_html_content(html: str, markdown: str, url: str, idx: int) -> str:
    """
    Save HTML and markdown content to files.
    
    Args:
        html: Raw HTML content
        markdown: Markdown content
        url: Source URL
        idx: Result index
        
    Returns:
        Path to saved HTML file
    """
    # Create results directory if it doesn't exist
    results_dir = Path("search_results")
    results_dir.mkdir(exist_ok=True)
    
    # Create a safe filename from URL
    safe_name = "".join(c if c.isalnum() else "_" for c in url.split("//")[-1])
    filename = f"result_{idx}_{safe_name[:50]}.html"
    filepath = results_dir / filename
    
    # Save HTML content with metadata
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"<!-- Source URL: {url} -->\n")
        f.write(html)
    
    # Save markdown version
    markdown_path = filepath.with_suffix('.md')
    with open(markdown_path, 'w', encoding='utf-8') as f:
        f.write(f"# Content from {url}\n\n")
        f.write(markdown)  # Use the markdown from Crawl4AI
    
    return str(filepath)

async def search_duckduckgo(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Search DuckDuckGo for a query and return results.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        
    Returns:
        List of dictionaries containing search results with 'title', 'link', and 'snippet' keys
    """
    try:
        # Initialize the DDGS class
        with DDGS() as ddgs:
            # Perform the search with region set to 'wt-wt' for better results
            results = list(ddgs.text(
                query,
                max_results=max_results,
                region='wt-wt'  # Worldwide results
            ))
            
            # Format results and fetch webpage content
            formatted_results = []
            for idx, result in enumerate(results, 1):
                # Extract URL - try different possible keys
                link = (result.get('link') or 
                       result.get('url') or 
                       result.get('href', ''))
                
                # Skip if no valid URL
                if not link or not link.startswith(('http://', 'https://')):
                    print(f"Skipping result {idx} due to invalid URL: {link}")
                    continue
                
                # Fetch webpage content using Crawl4AI
                content, html, error = await fetch_webpage_content(link)
                
                # Only save content if we successfully fetched it
                html_file = ""
                if html and not error:
                    html_file = save_html_content(html, content, link, idx)
                
                formatted_results.append({
                    'title': result.get('title', '').strip(),
                    'link': link,
                    'snippet': result.get('body', '').strip(),
                    'content': content,
                    'html_file': html_file,
                    'error': error
                })
                
                # Print debug info
                print(f"Processed result {idx}:")
                print(f"Title: {result.get('title', '')}")
                print(f"URL: {link}")
                print(f"Error: {error or 'None'}\n")
                
            return formatted_results
            
    except Exception as e:
        print(f"Error searching DuckDuckGo: {str(e)}")
        return []

def display_results(results: list):
    """
    Display search results in a human-friendly format.
    
    Args:
        results (list): The list of search results.
    """
    for idx, result in enumerate(results, 1):
        print(f"\n{'='*80}\n")
        print(f"Result {idx}:")
        
        # Only display results with valid URLs
        if result.get('link'):
            print(f"Title: {result.get('title', 'No title')}")
            print(f"URL: {result['link']}")
            print(f"\nSnippet: {result.get('snippet', 'No snippet provided')}")
            
            if result.get('error'):
                print(f"\nError: {result['error']}")
            else:
                print(f"\nFiles saved:")
                print(f"- HTML: {result.get('html_file', 'Not saved')}")
                print(f"- Markdown: {result.get('html_file', '').replace('.html', '.md') if result.get('html_file') else 'Not saved'}")
                print(f"\nExtracted Content:\n{'-'*40}\n")
                print(result.get('content', 'No content available'))
        else:
            print("Invalid result - missing URL")
        
        print(f"\n{'='*80}\n")

async def main():
    """
    Main function to handle command-line arguments, perform the DuckDuckGo search,
    and display the results.
    """
    query = 'Abetalipoproteinemia symptoms treatment'
    print(f"Searching DuckDuckGo for: {query}\n")
    
    # Try to get more results since some might be invalid
    results = await search_duckduckgo(query, max_results=5)  # Increased max_results
    
    # Filter out results with no valid URL
    valid_results = [r for r in results if r.get('link')]
    
    if valid_results:
        display_results(valid_results)
        print(f"\nFiles have been saved in the 'search_results' directory")
        print(f"Found {len(valid_results)} valid results out of {len(results)} total results")
    else:
        print("No valid results found.")

if __name__ == "__main__":
    asyncio.run(main()) 