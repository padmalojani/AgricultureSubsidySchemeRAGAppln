import pytest
from utils import compute_file_hash, validate_pdf_file
from langchain_core.documents import Document
from utils import format_sources

def test_compute_file_hash():
    dummy_bytes = b"test pdf content"
    hash1 = compute_file_hash(dummy_bytes)
    hash2 = compute_file_hash(dummy_bytes)
    assert hash1 == hash2
    assert len(hash1) == 64

def test_validate_pdf_file_extension():
    class DummyFile:
        name = "scheme.txt"
        def seek(self, *args): pass
        def tell(self): return 100
    
    valid, msg = validate_pdf_file(DummyFile())
    assert not valid
    assert "Only PDF files" in msg

def test_format_sources():
    docs = [
        Document(page_content="text", metadata={"source": "drip_irrigation.pdf", "page": 3}),
        Document(page_content="text", metadata={"source": "drip_irrigation.pdf", "page": 3}),
        Document(page_content="text", metadata={"source": "machinery.pdf", "page": 5})
    ]
    formatted = format_sources(docs)
    assert "drip_irrigation.pdf" in formatted
    assert "machinery.pdf" in formatted
    assert "Page 3" in formatted