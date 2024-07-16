import json
import tempfile
import os
import pytest
from colrag.indexer import process_file  # Adjust the import based on your actual module

def test_process_file_json():
    test_data = [{"key1": "value1"}, {"key2": "value2"}]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(test_data, tmp)
        tmp.flush()
        tmp_name = tmp.name  # Save the name to re-open it later
    try:
        result = process_file(tmp_name)
        assert len(result["documents"]) == 2
        result_docs = [json.loads(doc) for doc in result["documents"]]
        assert result_docs == test_data
    finally:
        os.remove(tmp_name)  # Clean up the temporary file