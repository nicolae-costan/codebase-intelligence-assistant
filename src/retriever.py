"""Hybrid retrieval using Reciprocal Rank Fusion."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from src.bm25_index import BM25Index
from src.index_dense import query_dense
from src.schema import Chunk

def rrf_merge(
    dense_results: Sequence[dict[str, object]],
    bm25_results: Sequence[Chunk],
    k: int = 60,
) -> list[dict[str, object]]:
    """Merge dense and sparse retrieval results using Reciprocal Rank Fusion.

    Args:
        dense_results: Output from index_dense.query_dense.
        bm25_results: Output from bm25_index.query_bm25.
        k: Smoothing constant for RRF.

    Returns:
        A list of dictionaries with 'chunk' and 'score' sorted by combined RRF score.
    """
    scores: dict[str, float] = {}
    chunk_map: dict[str, dict[str, object]] = {}
    linked_map: dict[str, list[dict[str, object]]] = {}
    debug_map: dict[str, dict[str, object]] = {}

    # Process dense results
    for rank, dense_result in enumerate(dense_results):
        chunk_dict = dense_result["chunk"]
        if not isinstance(chunk_dict, dict):
            continue
            
        chunk_id = str(chunk_dict.get("id", ""))
        if not chunk_id:
            continue
            
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
        chunk_map[chunk_id] = chunk_dict
        debug_map.setdefault(chunk_id, {})["dense_rank"] = rank + 1
        debug_map[chunk_id]["dense_score"] = dense_result.get("score")
        linked_chunks = dense_result.get("linked_chunks", [])
        if isinstance(linked_chunks, list):
            linked_map[chunk_id] = [chunk for chunk in linked_chunks if isinstance(chunk, dict)]

    # Process BM25 results
    for rank, chunk in enumerate(bm25_results):
        chunk_id = chunk.id
        if not chunk_id:
            continue
            
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
        debug_map.setdefault(chunk_id, {})["bm25_rank"] = rank + 1
        if chunk_id not in chunk_map:
            chunk_map[chunk_id] = chunk.to_dict()

    # Sort by RRF score descending
    ranked_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    fused: list[dict[str, object]] = []
    for chunk_id, score in ranked_items:
        result: dict[str, object] = {
            "chunk": chunk_map[chunk_id],
            "score": score,
            "retrieval_debug": {
                **debug_map.get(chunk_id, {}),
                "rrf_score": score,
            },
        }
        if linked_map.get(chunk_id):
            result["linked_chunks"] = linked_map[chunk_id]
        fused.append(result)
    return fused

def hybrid_search(
    query: str,
    top_k: int = 5,
    *,
    bm25_index: BM25Index | None = None,
    dense_k_multiplier: int = 2,
    linked_window: int = 1,
) -> list[dict[str, object]]:
    """Perform hybrid search over both dense and sparse indexes.

    Fetches candidates from both indexes, merges them with RRF, and returns the top k.
    
    Args:
        query: The search string.
        top_k: The final number of results to return.
        bm25_index: Loaded BM25 index. If None, it will be loaded from disk.
        dense_k_multiplier: How many more candidates to fetch initially before fusing.
                            A higher number gives RRF more candidates to work with.
        linked_window: Neighboring dense subchunks from the same symbol to attach.

    Returns:
        List of fused results.
    """
    if bm25_index is None:
        bm25_index = BM25Index.load()

    fetch_k = top_k * dense_k_multiplier

    # 1. Fetch from dense index
    dense_results = query_dense(query, k=fetch_k, linked_window=linked_window)

    # 2. Fetch from sparse index
    bm25_results = bm25_index.query_bm25(query, k=fetch_k)

    # 3. Merge with RRF
    fused = rrf_merge(dense_results, bm25_results)

    # 4. Truncate to top_k
    return fused[:top_k]

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Query both dense and sparse indexes using RRF.")
    parser.add_argument("--query", required=True, help="Search query.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of final results to return.")
    parser.add_argument("--json", action="store_true", help="Emit query results as JSON.")
    args = parser.parse_args(argv)

    results = hybrid_search(args.query, top_k=args.top_k)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print()
        print(f"Top {len(results)} Hybrid (RRF) results for '{args.query}'")
        for rank, result in enumerate(results, start=1):
            chunk = result["chunk"]
            if isinstance(chunk, dict):
                print(
                    f"{rank}. {chunk.get('filepath')}::{chunk.get('function_name')} "
                    f"lines {chunk.get('start_line')}-{chunk.get('end_line')} "
                    f"rrf_score={result['score']:.4f}"
                )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
