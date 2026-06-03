from src.schema import Chunk
import src.retriever as retriever
from src.retriever import hybrid_search, rrf_merge

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
    assert fused[0]["retrieval_debug"]["dense_rank"] == 2
    assert fused[0]["retrieval_debug"]["bm25_rank"] == 1
    assert fused[0]["retrieval_debug"]["rrf_score"] == fused[0]["score"]


def test_rrf_merge_preserves_linked_dense_chunks():
    chunk_a = Chunk(id="a", filepath="file.py", function_name="func_a", start_line=1, end_line=10, docstring="", source="", language="python")
    linked = Chunk(id="a-neighbor", filepath="file.py", function_name="func_a", start_line=11, end_line=20, docstring="", source="", language="python")

    fused = rrf_merge(
        [{"chunk": chunk_a.to_dict(), "score": 0.9, "linked_chunks": [linked.to_dict()]}],
        [],
    )

    assert fused[0]["chunk"]["id"] == "a"
    assert fused[0]["linked_chunks"] == [linked.to_dict()]


def test_hybrid_search_fetches_dense_and_bm25_then_fuses(monkeypatch):
    chunk_a = Chunk(id="a", filepath="file.py", function_name="func_a", start_line=1, end_line=10, docstring="", source="", language="python")
    chunk_b = Chunk(id="b", filepath="file.py", function_name="func_b", start_line=11, end_line=20, docstring="", source="", language="python")
    chunk_c = Chunk(id="c", filepath="file.py", function_name="func_c", start_line=21, end_line=30, docstring="", source="", language="python")

    dense_calls = []

    def fake_query_dense(query, k, linked_window=0):
        dense_calls.append((query, k, linked_window))
        return [
            {"chunk": chunk_a.to_dict(), "score": 0.9, "rank": 1, "retriever": "dense"},
            {"chunk": chunk_b.to_dict(), "score": 0.8, "rank": 2, "retriever": "dense"},
        ]

    class FakeBM25Index:
        def __init__(self):
            self.calls = []

        def query_bm25(self, query, k):
            self.calls.append((query, k))
            return [chunk_b, chunk_c]

    bm25_index = FakeBM25Index()
    monkeypatch.setattr(retriever, "query_dense", fake_query_dense)

    results = hybrid_search("request validation", top_k=2, bm25_index=bm25_index, dense_k_multiplier=3, linked_window=2)

    assert dense_calls == [("request validation", 6, 2)]
    assert bm25_index.calls == [("request validation", 6)]
    assert [result["chunk"]["id"] for result in results] == ["b", "a"]

if __name__ == "__main__":
    test_rrf_merge()
    print("test_rrf_merge passed!")
