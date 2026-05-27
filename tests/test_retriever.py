from src.schema import Chunk
from src.retriever import rrf_merge

def test_rrf_merge():
    chunk_a = Chunk(id="a", filepath="file.py", function_name="func_a", start_line=1, end_line=10, docstring="", source="", language="python")
    chunk_b = Chunk(id="b", filepath="file.py", function_name="func_b", start_line=11, end_line=20, docstring="", source="", language="python")
    chunk_c = Chunk(id="c", filepath="file.py", function_name="func_c", start_line=21, end_line=30, docstring="", source="", language="python")

    # Dense ranked: A (rank 0), B (rank 1)
    dense_results = [
        {"chunk": chunk_a.to_dict(), "score": 0.9, "rank": 1, "retriever": "dense"},
        {"chunk": chunk_b.to_dict(), "score": 0.8, "rank": 2, "retriever": "dense"}
    ]

    # BM25 ranked: B (rank 0), C (rank 1)
    bm25_results = [chunk_b, chunk_c]

    # K=60
    # A score = 1/(60 + 0 + 1) = 1/61 = 0.01639
    # B score = 1/(60 + 1 + 1) [dense] + 1/(60 + 0 + 1) [bm25] = 1/62 + 1/61 = 0.016129 + 0.01639 = 0.0325
    # C score = 1/(60 + 1 + 1) = 1/62 = 0.016129
    # Expected order: B, A, C

    fused = rrf_merge(dense_results, bm25_results, k=60)
    
    assert len(fused) == 3
    assert fused[0]["chunk"]["id"] == "b"
    assert fused[1]["chunk"]["id"] == "a"
    assert fused[2]["chunk"]["id"] == "c"

    # Assert scores are calculated correctly
    assert abs(fused[0]["score"] - (1/62 + 1/61)) < 1e-5
    assert abs(fused[1]["score"] - (1/61)) < 1e-5
    assert abs(fused[2]["score"] - (1/62)) < 1e-5

if __name__ == "__main__":
    test_rrf_merge()
    print("test_rrf_merge passed!")

