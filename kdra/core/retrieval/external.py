import arxiv
import os
import requests
from typing import List, Dict, Optional

class ExternalRetriever:
    """
    Handles searching and retrieving papers from external sources like arXiv.
    """
    
    def __init__(self):
        self.client = arxiv.Client()

    def search_arxiv(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Search arXiv for papers matching the query.
        
        Args:
            query: The search query string.
            max_results: Maximum number of results to return.
            
        Returns:
            List of dictionaries containing paper details.
        """
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        results = []
        try:
            for result in self.client.results(search):
                results.append({
                    "title": result.title,
                    "authors": [a.name for a in result.authors],
                    "summary": result.summary,
                    "published": result.published.strftime("%Y-%m-%d"),
                    "pdf_url": result.pdf_url,
                    "arxiv_id": result.entry_id.split('/')[-1],
                    "source": "arXiv"
                })
        except Exception as e:
            print(f"Error searching arXiv: {e}")
            
        return results

    def download_pdf(self, url: str, save_dir: str, filename: Optional[str] = None) -> str:
        """
        Download a PDF from a URL to the specified directory.
        
        Args:
            url: The URL of the PDF.
            save_dir: The directory to save the file in.
            filename: Optional filename. If None, derived from URL or ID.
            
        Returns:
            The absolute path to the downloaded file.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        if not filename:
            filename = url.split('/')[-1]
            if not filename.endswith('.pdf'):
                filename += '.pdf'
        
        # Sanitize filename
        filename = "".join([c for c in filename if c.isalpha() or c.isdigit() or c in (' ', '-', '_', '.')]).rstrip()
        file_path = os.path.join(save_dir, filename)
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return file_path
        except Exception as e:
            raise Exception(f"Failed to download PDF: {e}")
