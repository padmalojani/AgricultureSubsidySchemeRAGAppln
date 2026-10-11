import os
import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config
from utils import compute_file_hash

def process_uploaded_pdf(uploaded_file, chunk_size=config.DEFAULT_CHUNK_SIZE, chunk_overlap=config.DEFAULT_CHUNK_OVERLAP):
    """
    Extracts text from uploaded PDF, checks for scans, and chunks the content.
    Returns list of LangChain Document objects with robust metadata.
    """
    file_bytes = uploaded_file.read()
    file_hash = compute_file_hash(file_bytes)
    filename = uploaded_file.name
    
    # Save file locally
    file_path = os.path.join(config.PDF_DIR, filename)
    with open(file_path, "wb") as f:
        f.write(file_bytes)
        
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    extracted_docs = []
    scanned_pages = 0
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        
        # Detect potential scanned pages (low text yield)
        if len(text.strip()) < 50:
            scanned_pages += 1
            
        metadata = {
            "source": filename,
            "page": page_num + 1,
            "file_hash": file_hash,
            "scheme_name": filename.replace(".pdf", "").replace("_", " ")
        }
        
        extracted_docs.append(Document(page_content=text, metadata=metadata))
        
    if scanned_pages > 0 and scanned_pages == len(doc):
        raise ValueError(f"'{filename}' appears to be a scanned PDF. OCR processing is required.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    split_docs = text_splitter.split_documents(extracted_docs)
    return split_docs