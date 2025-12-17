import os
from pypdf import PdfReader

file_path = "data/2410.00193v3.pdf"
if os.path.exists(file_path):
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        print(f"Successfully read PDF. Length: {len(text)} chars")
        print("First 200 chars:")
        print(text[:200])
    except Exception as e:
        print(f"Error reading PDF: {e}")
else:
    print(f"File not found: {file_path}")
