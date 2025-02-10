import requests
import xml.etree.ElementTree as ET
from typing import List, Dict
import time

def search_pubmed_and_get_urls(disease: str, max_results: int = 100) -> List[Dict[str, str]]:
    """
    Search PubMed for articles about a rare disease and return article details
    
    Args:
        disease (str): The rare disease to search for
        max_results (int): Maximum number of results to return
        
    Returns:
        List[Dict[str, str]]: List of article details including title, abstract, and URL
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    
    # Create search variations for more comprehensive results
    search_variations = [
        f'"{disease}"[Title/Abstract] AND (Review[ptyp] OR Clinical Trial[ptyp])',
        f'"{disease}"[Title/Abstract] AND (pathophysiology OR genetics OR diagnosis)',
        f'"{disease}"[Title/Abstract] AND (treatment OR therapy OR management)',
        f'"{disease}"[Title/Abstract] AND (epidemiology OR prevalence OR incidence)',
        f'"{disease} rare disease"[Title/Abstract]'
    ]
    
    all_pmids = set()
    articles = []
    
    for search_query in search_variations:
        try:
            # First, search for article IDs
            search_params = {
                'db': 'pubmed',
                'term': search_query,
                'retmax': max_results // len(search_variations),
                'retmode': 'xml',
                'usehistory': 'y'
            }
            
            search_response = requests.get(f"{base_url}esearch.fcgi", params=search_params)
            search_response.raise_for_status()
            
            search_root = ET.fromstring(search_response.content)
            pmids = [id_elem.text for id_elem in search_root.findall('.//Id')]
            
            # Only process new PMIDs
            new_pmids = [pmid for pmid in pmids if pmid not in all_pmids]
            all_pmids.update(new_pmids)
            
            if new_pmids:
                # Fetch details for new articles in batches
                batch_size = 20
                for i in range(0, len(new_pmids), batch_size):
                    batch_pmids = new_pmids[i:i + batch_size]
                    
                    fetch_params = {
                        'db': 'pubmed',
                        'id': ','.join(batch_pmids),
                        'retmode': 'xml'
                    }
                    
                    fetch_response = requests.get(f"{base_url}efetch.fcgi", params=fetch_params)
                    fetch_response.raise_for_status()
                    
                    root = ET.fromstring(fetch_response.content)
                    
                    for article in root.findall('.//PubmedArticle'):
                        try:
                            # Get PMID
                            pmid = article.find('.//PMID').text
                            
                            # Get title
                            title_elem = article.find('.//ArticleTitle')
                            title = title_elem.text if title_elem is not None else "No title available"
                            
                            # Get abstract
                            abstract_parts = []
                            abstract_elems = article.findall('.//Abstract/AbstractText')
                            for abstract_elem in abstract_elems:
                                # Check for labeled sections
                                label = abstract_elem.get('Label')
                                text = abstract_elem.text or ""
                                if label:
                                    abstract_parts.append(f"{label}: {text}")
                                else:
                                    abstract_parts.append(text)
                            
                            abstract = "\n".join(abstract_parts) if abstract_parts else "No abstract available"
                            
                            # Get publication year
                            year_elem = article.find('.//PubDate/Year')
                            year = year_elem.text if year_elem is not None else "Year not available"
                            
                            # Get journal name
                            journal_elem = article.find('.//Journal/Title')
                            journal = journal_elem.text if journal_elem is not None else "Journal not available"
                            
                            articles.append({
                                'title': title,
                                'abstract': abstract,
                                'pubmed_url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                                'year': year,
                                'journal': journal,
                                'pmid': pmid
                            })
                            
                        except Exception as e:
                            print(f"Error processing article {pmid}: {str(e)}")
                            continue
                    
                    # Respect API rate limits
                    time.sleep(0.34)  # PubMed allows 3 requests per second
            
        except Exception as e:
            print(f"Error searching PubMed with query '{search_query}': {str(e)}")
            continue
    
    # Sort articles by year (newest first) and remove duplicates
    articles.sort(key=lambda x: x.get('year', '0'), reverse=True)
    seen_pmids = set()
    unique_articles = []
    for article in articles:
        if article['pmid'] not in seen_pmids:
            seen_pmids.add(article['pmid'])
            unique_articles.append(article)
    
    return unique_articles[:max_results]

if __name__ == "__main__":
    disease = "rare disease"
    max_results = 100
    articles = search_pubmed_and_get_urls(disease, max_results)
    for article in articles:
        print(f"\nTitle: {article['title']}")
        print(f"URL: {article['pubmed_url']}")
        print(f"Abstract: {article['abstract'][:200]}...")