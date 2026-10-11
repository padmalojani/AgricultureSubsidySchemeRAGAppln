import hashlib
import os

def compute_file_hash(file_bytes: bytes) -> str:
    """Computes SHA-256 hash of a file to detect duplicates."""
    hasher = hashlib.sha256()
    hasher.update(file_bytes)
    return hasher.hexdigest()

def validate_pdf_file(uploaded_file) -> tuple[bool, str]:
    """Validates file type and size (max 25MB)."""
    if not uploaded_file.name.lower().endswith(".pdf"):
        return False, "Only PDF files are supported."
    
    # Check file size (e.g., max 25MB)
    uploaded_file.seek(0, os.SEEK_END)
    size = uploaded_file.tell()
    uploaded_file.seek(0)
    
    if size > 25 * 1024 * 1024:
        return False, "File size exceeds the 25MB limit."
    if size == 0:
        return False, "Uploaded file is empty."
        
    return True, "Valid"

def format_sources(docs) -> str:
    """Formats source citations from retrieved document metadata."""
    sources = set()
    for doc in docs:
        source = doc.metadata.get("source", "Unknown Document")
        page = doc.metadata.get("page", "Unknown")
        sources.add(f"- **{os.path.basename(source)}** (Page {page})")
    return "\n".join(sorted(sources))