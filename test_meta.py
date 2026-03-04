import re
import PyPDF2
from pypdf import PdfReader

def extract_ids_from_pdf(filepath):
    try:
        reader = PdfReader(filepath)
        text = ""
        for i in range(min(2, len(reader.pages))): # Scan first 2 pages
            text += reader.pages[i].extract_text()
            
        arxiv_match = re.search(r'arXiv:(\d{4}\.\d{4,5}(v\d+)?)', text, re.IGNORECASE)
        doi_match = re.search(r'\b(10.\d{4,9}/[-._;()/:A-Z0-9]+)\b', text, re.IGNORECASE)
        
        return {
            "arxiv_id": arxiv_match.group(1) if arxiv_match else None,
            "doi": doi_match.group(1) if doi_match else None
        }
    except Exception as e:
        return {"error": str(e)}

print(extract_ids_from_pdf('data/2306.05212v1.pdf'))
