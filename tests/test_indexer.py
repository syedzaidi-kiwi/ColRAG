import pytest
import os
import tempfile
import json
import pandas as pd
from colrag import index_documents, read_file, process_file
from colrag.config import config

@pytest.fixture
def temp_directory():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname

def test_read_txt_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt") as tmp:
        tmp.write("This is a test document.")
        tmp.flush()
        content = read_file(tmp.name)
        assert content == "This is a test document."

def test_read_json_file():
    test_data = {"key": "value", "number": 42}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as tmp:
        json.dump(test_data, tmp)
        tmp.flush()
        content = read_file(tmp.name)
        assert content == test_data

def test_read_csv_file():
    test_data = pd.DataFrame({"A": [1, 2, 3], "B": ["a", "b", "c"]})
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv") as tmp:
        test_data.to_csv(tmp.name, index=False)
        content = read_file(tmp.name)
        assert len(content) == 3
        assert content[0] == {"A": 1, "B": "a"}

def test_process_file_txt():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt") as tmp:
        tmp.write("This is a test document.")
        tmp.flush()
        result = process_file(tmp.name)
        assert len(result["documents"]) == 1
        assert result["documents"][0] == "This is a test document."
        assert result["document_ids"] == [tmp.name]
        assert result["document_metadatas"] == [{"source": tmp.name}]

def test_process_file_json():
    test_data = [{"key1": "value1"}, {"key2": "value2"}]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as tmp:
        json.dump(test_data, tmp)
        tmp.flush()
        result = process_file(tmp.name)
        assert len(result["documents"]) == 2
        assert result["documents"] == test_data
        assert result["document_ids"] == [tmp.name, tmp.name]
        assert result["document_metadatas"] == [{"source": tmp.name}, {"source": tmp.name}]

@pytest.mark.skip(reason="This test requires a working RAGPretrainedModel, which might not be available in the test environment")
def test_index_documents(temp_directory):
    # Create test files
    with open(os.path.join(temp_directory, "test1.txt"), "w") as f:
        f.write("This is the first test document.")
    
    with open(os.path.join(temp_directory, "test2.json"), "w") as f:
        json.dump({"key": "value"}, f)

    index_name = "test_index"
    index_path = index_documents(temp_directory, index_name)

    expected_index_path = os.path.join(".ragatouille", "colbert", "indexes", index_name)
    assert index_path == expected_index_path
    assert os.path.exists(index_path)
    assert os.path.isdir(index_path)

def test_read_file_unsupported_format():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".unsupported") as tmp:
        tmp.write("This is an unsupported format.")
        tmp.flush()
        with pytest.raises(ValueError, match="Unsupported file format: .unsupported"):
            read_file(tmp.name)

if __name__ == "__main__":
    pytest.main()