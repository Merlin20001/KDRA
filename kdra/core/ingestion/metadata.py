import os
import re
import requests
import arxiv
from datetime import datetime
from pypdf import PdfReader
from typing import Optional, Dict, Any, List
from kdra.core.schemas import PaperMetadata

class MetadataExtractor:
    """
    Extracts real metadata (title, authors, year, venue) from a PDF file using 
    Academic APIs (arXiv, CrossRef) based on identifiers found in the file or filename.
    """
    
    @staticmethod
    def extract(file_path: str) -> PaperMetadata:
        filename = os.path.basename(file_path)
        paper_id = filename
        title = filename
        year = datetime.now().year
        authors = []
        venue = "Unknown"
        url = f"file://{os.path.abspath(file_path)}"
        
        # 1. Try to find ArXiv ID or DOI in filename or text
        arxiv_id, doi = MetadataExtractor._find_identifiers(file_path)
        
        # 2. Fetch from APIs
        api_data = None
        if arxiv_id:
            print(f"[{filename}] Found arXiv ID: {arxiv_id}. Fetching metadata...")
            api_data = MetadataExtractor._fetch_arxiv(arxiv_id)
        
        if not api_data and doi:
            print(f"[{filename}] Found DOI: {doi}. Fetching metadata from CrossRef...")
            api_data = MetadataExtractor._fetch_crossref(doi)
            
        # 3. Apply fetched data
        if api_data:
            title = api_data.get("title", title)
            year = api_data.get("year", year)
            authors = api_data.get("authors", authors)
            venue = api_data.get("venue", venue)
            url = api_data.get("url", url)
            paper_id = api_data.get("paper_id", paper_id)
            print(f"[{filename}] Successfully resolved metadata: '{title[:50]}...' by {', '.join(authors[:2])}")
        else:
            print(f"[{filename}] Could not resolve external metadata. Falling back to basics.")

        return PaperMetadata(
            paper_id=paper_id,
            title=title,
            year=year,
            venue=venue,
            authors=authors,
            url=url
        )

    @staticmethod
    def _find_identifiers(file_path: str) -> tuple[Optional[str], Optional[str]]:
        filename = os.path.basename(file_path)
        arxiv_id = None
        doi = None
        
        # Check filename for arXiv pattern (e.g. 2306.05212 or 2306.05212v1)
        arxiv_match_fn = re.search(r'(\d{4}\.\d{4,5}(?:v\d+)?)', filename)
        if arxiv_match_fn:
            arxiv_id = arxiv_match_fn.group(1)
            return arxiv_id, doi
            
        # If not, read first few pages of PDF
        if file_path.lower().endswith('.pdf'):
            try:
                reader = PdfReader(file_path)
                text = ""
                for i in range(min(2, len(reader.pages))):
                    text += reader.pages[i].extract_text() or ""
                    
                arxiv_match = re.search(r'arXiv:(\d{4}\.\d{4,5}(?:v\d+)?)', text, re.IGNORECASE)
                doi_match = re.search(r'\b(10.\d{4,9}/[-._;()/:A-Z0-9]+)\b', text, re.IGNORECASE)
                
                if arxiv_match:
                    arxiv_id = arxiv_match.group(1)
                if doi_match:
                    doi = doi_match.group(1)
            except Exception as e:
                print(f"Warning: Failed to read PDF {filename} for metadata IDs: {e}")
                pass
                
        return arxiv_id, doi

    @staticmethod
    def _fetch_arxiv(arxiv_id: str) -> Optional[Dict[str, Any]]:
        try:
            # Clean ID (strip v1 etc if needed, but arxiv library handles versions usually)
            search = arxiv.Search(id_list=[arxiv_id])
            # We must use a generator as search results are paginated
            client = arxiv.Client()
            results = list(client.results(search))
            if results:
                paper = results[0]
                return {
                    "title": paper.title,
                    "authors": [a.name for a in paper.authors],
                    "year": paper.published.year,
                    "venue": "arXiv",
                    "url": paper.pdf_url,
                    "paper_id": arxiv_id
                }
        except Exception as e:
            print(f"arXiv API error: {e}")
        return None

    @staticmethod
    def _fetch_crossref(doi: str) -> Optional[Dict[str, Any]]:
        try:
            headers = {"User-Agent": "KDRA/1.0 (mailto:admin@example.com)"}
            resp = requests.get(f"https://api.crossref.org/works/{doi}", headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()["message"]
                
                # Title
                title = data.get("title", [""])[0]
                
                # Authors
                authors = []
                for author in data.get("author", []):
                    name = f"{author.get('given', '')} {author.get('family', '')}".strip()
                    if name:
                        authors.append(name)
                        
                # Year (complex parsing from crossref)
                year = datetime.now().year
                if "published-print" in data:
                    year = data["published-print"]["date-parts"][0][0]
                elif "published-online" in data:
                    year = data["published-online"]["date-parts"][0][0]
                
                # Venue
                venue = data.get("container-title", [""])[0]
                if not venue:
                    venue = data.get("publisher", "Unknown")

                return {
                    "title": title,
                    "authors": authors,
                    "year": int(year),
                    "venue": venue,
                    "url": data.get("URL", f"https://doi.org/{doi}"),
                    "paper_id": doi
                }
        except Exception as e:
            print(f"CrossRef API error: {e}")
        return None
