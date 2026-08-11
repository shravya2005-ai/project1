from pathlib import Path
from src.document_processor import extract_texts_from_file


def test_extract_texts_from_txt(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("This is a test document. It contains a few sentences.")

    chunks = extract_texts_from_file(file_path)
    assert chunks
    assert chunks[0]["metadata"]["source"] == "sample.txt"
    assert "This is a test document" in chunks[0]["document"]
