"""Dense GraphCodeBERT embeddings and ChromaDB retrieval."""

from __future__ import annotations

import argparse
import hashlib
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from src.chunker import DEFAULT_CHUNK_OVERLAP_LINES, chunk_repository
from src.schema import Chunk

MODEL_NAME = "microsoft/graphcodebert-base"
CACHE_VERSION = 1
DEFAULT_CACHE_PATH = Path("data/chunks/embeddings.pkl")
DEFAULT_CHROMA_PATH = Path("index/chroma_db")
DEFAULT_COLLECTION_NAME = "code_chunks_dense"
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_CHUNK_LINES = 220
DEFAULT_LINKED_WINDOW = 1
COMMON_TEST_PATH_PREFIXES = ("test/", "tests/")
COMMON_TEST_PATH_PARTS = {"test", "tests"}

EmbeddingFunction = Callable[[Sequence[str], int], list[list[float]]]


class Embedder(Protocol):
    """Minimal embedding backend used by dense retrieval."""

    def embed_texts(self, texts: Sequence[str], batch_size: int = DEFAULT_BATCH_SIZE) -> list[list[float]]:
        """Embed a batch of texts."""


@dataclass
class DenseIndex:
    """Queryable dense index backed by a ChromaDB collection."""

    collection: Any
    embedder: Embedder | EmbeddingFunction | None = None
    model_name: str = MODEL_NAME
    batch_size: int = DEFAULT_BATCH_SIZE

    def query_dense(self, query: str, k: int = 5, *, linked_window: int = 0) -> list[dict[str, object]]:
        """Return ranked dense retrieval results for a query."""

        if k <= 0:
            return []
        if linked_window < 0:
            raise ValueError("linked_window cannot be negative.")
        query_embedding = _embed_texts([query], self.embedder, self.model_name, self.batch_size)[0]
        raw_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        results = _retrieval_results_from_chroma(raw_results)
        if linked_window:
            _attach_linked_chunks(self.collection, results, linked_window=linked_window)
        return results


class GraphCodeBertEmbedder:
    """HuggingFace GraphCodeBERT embedding backend with mean pooling."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: str | None = None,
        max_length: int = 512,
        show_progress: bool = False,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.show_progress = show_progress

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Dense retrieval requires the `torch` and `transformers` packages. "
                "Install the project dependencies before building the dense index."
            ) from exc

        self._torch = torch
        requested_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested device {requested_device!r}, but PyTorch cannot access CUDA. "
                "Check `nvidia-smi` and whether the current shell/container exposes the GPU."
            )

        self.device = requested_device
        if self.show_progress:
            _progress(f"Loading {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        if self.show_progress:
            _progress("Model loaded.")

    def embed_texts(self, texts: Sequence[str], batch_size: int = DEFAULT_BATCH_SIZE) -> list[list[float]]:
        """Embed texts in batches using mean-pooled token states."""

        if not texts:
            return []

        embeddings: list[list[float]] = []
        torch = self._torch
        total = len(texts)
        total_batches = (total + batch_size - 1) // batch_size
        started_at = time.perf_counter()
        with torch.no_grad():
            for batch_index, start in enumerate(range(0, total, batch_size), start=1):
                batch = list(texts[start : start + batch_size])
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                output = self.model(**encoded)
                token_embeddings = output.last_hidden_state
                attention_mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
                summed = (token_embeddings * attention_mask).sum(dim=1)
                counts = attention_mask.sum(dim=1).clamp(min=1e-9)
                pooled = summed / counts
                normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
                embeddings.extend(normalized.cpu().tolist())

                if self.show_progress:
                    _batch_progress(
                        "Embedding chunks",
                        batch_index=batch_index,
                        total_batches=total_batches,
                        completed_items=len(embeddings),
                        total_items=total,
                        started_at=started_at,
                    )
        if self.show_progress:
            _progress("")
        return embeddings


def chunk_to_embedding_text(chunk: Chunk) -> str:
    """Build the text representation embedded for a code chunk."""

    parts = [
        f"path: {chunk.filepath}",
        f"symbol: {chunk.function_name}",
        f"language: {chunk.language}",
    ]
    if chunk.docstring:
        parts.append(f"docstring: {chunk.docstring}")
    parts.append("source:")
    parts.append(chunk.source)
    return "\n".join(parts)


def index_chunks(
    chunks: Sequence[Chunk],
    *,
    cache_path: str | Path = DEFAULT_CACHE_PATH,
    persist_dir: str | Path = DEFAULT_CHROMA_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    model_name: str = MODEL_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
    force_rebuild: bool = False,
    embedder: Embedder | EmbeddingFunction | None = None,
    collection: Any | None = None,
    device: str | None = None,
    show_progress: bool = False,
) -> DenseIndex:
    """Embed chunks, cache vectors, and store them in a Chroma collection."""

    chunk_list = list(chunks)
    embeddings = load_or_compute_embeddings(
        chunk_list,
        cache_path=cache_path,
        model_name=model_name,
        batch_size=batch_size,
        force_rebuild=force_rebuild,
        embedder=embedder,
        device=device,
        show_progress=show_progress,
    )
    if show_progress:
        _progress(f"Writing {len(chunk_list)} chunks to Chroma collection {collection_name!r}...")
    target_collection = collection or _reset_collection(persist_dir, collection_name)
    if chunk_list:
        target_collection.upsert(
            ids=[chunk.id for chunk in chunk_list],
            embeddings=embeddings,
            documents=[chunk.source for chunk in chunk_list],
            metadatas=[_metadata_for_chunk(chunk, model_name) for chunk in chunk_list],
        )
    if show_progress:
        _progress("Chroma index updated.")
    return DenseIndex(
        collection=target_collection,
        embedder=embedder,
        model_name=model_name,
        batch_size=batch_size,
    )


def filter_chunks_by_path(
    chunks: Sequence[Chunk],
    *,
    include_path_prefixes: Sequence[str] = (),
    exclude_path_prefixes: Sequence[str] = (),
    exclude_tests: bool = False,
) -> list[Chunk]:
    """Filter chunks by repository-relative path prefixes."""

    include_prefixes = tuple(_normalize_path_prefix(prefix) for prefix in include_path_prefixes)
    exclude_prefixes = tuple(_normalize_path_prefix(prefix) for prefix in exclude_path_prefixes)

    filtered: list[Chunk] = []
    for chunk in chunks:
        filepath = chunk.filepath.replace("\\", "/")
        if include_prefixes and not any(_path_matches_prefix(filepath, prefix) for prefix in include_prefixes):
            continue
        if exclude_prefixes and any(_path_matches_prefix(filepath, prefix) for prefix in exclude_prefixes):
            continue
        if exclude_tests and _is_common_test_path(filepath):
            continue
        filtered.append(chunk)
    return filtered


def load_or_compute_embeddings(
    chunks: Sequence[Chunk],
    *,
    cache_path: str | Path = DEFAULT_CACHE_PATH,
    model_name: str = MODEL_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
    force_rebuild: bool = False,
    embedder: Embedder | EmbeddingFunction | None = None,
    device: str | None = None,
    show_progress: bool = False,
) -> list[list[float]]:
    """Load cached embeddings or compute and persist them for the given chunks."""

    chunk_list = list(chunks)
    if not chunk_list:
        return []

    path = Path(cache_path)
    cache_key = _cache_key(chunk_list, model_name)
    if not force_rebuild:
        cached_embeddings = _load_cached_embeddings(path, cache_key)
        if cached_embeddings is not None:
            if show_progress:
                _progress(f"Loaded {len(cached_embeddings)} embeddings from cache: {path}")
            return cached_embeddings

    texts = [chunk_to_embedding_text(chunk) for chunk in chunk_list]
    if show_progress:
        _progress(f"Computing embeddings for {len(texts)} chunks...")
    embeddings = _embed_texts(texts, embedder, model_name, batch_size, device=device, show_progress=show_progress)
    if len(embeddings) != len(chunk_list):
        raise RuntimeError(f"Expected {len(chunk_list)} embeddings, got {len(embeddings)}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(
            {
                "version": CACHE_VERSION,
                "cache_key": cache_key,
                "dimension": len(embeddings[0]) if embeddings else 0,
                "embeddings": embeddings,
            },
            file,
        )
    if show_progress:
        _progress(f"Saved embeddings cache: {path}")
    return embeddings


def query_dense(
    q: str,
    k: int = 5,
    *,
    persist_dir: str | Path = DEFAULT_CHROMA_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    model_name: str = MODEL_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
    embedder: Embedder | EmbeddingFunction | None = None,
    linked_window: int = 0,
) -> list[dict[str, object]]:
    """Load the persisted dense index and return top-k retrieval results."""

    collection = _load_collection(persist_dir, collection_name)
    return DenseIndex(
        collection=collection,
        embedder=embedder,
        model_name=model_name,
        batch_size=batch_size,
    ).query_dense(q, k=k, linked_window=linked_window)


def main(argv: Sequence[str] | None = None) -> int:
    """Build a dense index for a repository and optionally query it."""

    parser = argparse.ArgumentParser(description="Build and query a GraphCodeBERT dense Chroma index.")
    parser.add_argument("--repo", required=True, help="Repository or directory path to chunk and index.")
    parser.add_argument("--query", help="Optional query to run after indexing.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of dense results to return for --query.")
    parser.add_argument("--cache-path", default=str(DEFAULT_CACHE_PATH), help="Embedding cache path.")
    parser.add_argument("--index-path", default=str(DEFAULT_CHROMA_PATH), help="Chroma persistence directory.")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME, help="Chroma collection name.")
    parser.add_argument("--model", default=MODEL_NAME, help="HuggingFace model name.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Embedding batch size.")
    parser.add_argument(
        "--max-chunk-lines",
        type=int,
        default=DEFAULT_MAX_CHUNK_LINES,
        help="Split symbols longer than this many lines into overlapping linked subchunks. Use 0 to disable.",
    )
    parser.add_argument(
        "--chunk-overlap-lines",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP_LINES,
        help="Line overlap between subchunks created by --max-chunk-lines.",
    )
    parser.add_argument(
        "--linked-window",
        type=int,
        default=DEFAULT_LINKED_WINDOW,
        help="For each query hit, also return this many neighboring chunks from the same symbol.",
    )
    parser.add_argument(
        "--include-path-prefix",
        action="append",
        default=[],
        help="Only index chunks whose repository-relative path starts with this prefix. Repeatable.",
    )
    parser.add_argument(
        "--exclude-path-prefix",
        action="append",
        default=[],
        help="Skip chunks whose repository-relative path starts with this prefix. Repeatable.",
    )
    parser.add_argument("--exclude-tests", action="store_true", help="Skip common test directories and test_*.py files.")
    parser.add_argument(
        "--device",
        help="Torch device to use, such as 'cuda', 'cuda:0', or 'cpu'. Defaults to CUDA when PyTorch can access it.",
    )
    parser.add_argument("--force-rebuild", action="store_true", help="Ignore any existing embedding cache.")
    parser.add_argument("--quiet", action="store_true", help="Disable progress logging.")
    args = parser.parse_args(argv)
    show_progress = not args.quiet

    if show_progress:
        _progress(f"Chunking repository: {args.repo}")
    max_chunk_lines = args.max_chunk_lines if args.max_chunk_lines > 0 else None
    chunks = chunk_repository(
        args.repo,
        max_chunk_lines=max_chunk_lines,
        chunk_overlap_lines=args.chunk_overlap_lines,
    )
    chunks = filter_chunks_by_path(
        chunks,
        include_path_prefixes=args.include_path_prefix,
        exclude_path_prefixes=args.exclude_path_prefix,
        exclude_tests=args.exclude_tests,
    )
    if show_progress:
        _progress(f"Found {len(chunks)} chunks.")
    embedder = (
        GraphCodeBertEmbedder(model_name=args.model, device=args.device, show_progress=show_progress) if args.query else None
    )
    dense_index = index_chunks(
        chunks,
        cache_path=args.cache_path,
        persist_dir=args.index_path,
        collection_name=args.collection,
        model_name=args.model,
        batch_size=args.batch_size,
        force_rebuild=args.force_rebuild,
        embedder=embedder,
        device=args.device,
        show_progress=show_progress,
    )

    print(f"Indexed {len(chunks)} chunks")
    print(f"Embedding cache: {args.cache_path}")
    print(f"Chroma index: {args.index_path}")

    if args.query:
        results = dense_index.query_dense(args.query, k=args.top_k, linked_window=args.linked_window)
        print()
        print(f"Top {len(results)} dense results")
        for result in results:
            chunk = result["chunk"]
            print(
                f"{result['rank']}. {chunk['filepath']}::{chunk['function_name']} "
                f"lines {chunk['start_line']}-{chunk['end_line']} "
                f"score={result['score']:.4f}"
            )
            for linked_chunk in result.get("linked_chunks", []):
                print(
                    f"   linked: {linked_chunk['filepath']}::{linked_chunk['function_name']} "
                    f"lines {linked_chunk['start_line']}-{linked_chunk['end_line']}"
                )
    return 0


def _embed_texts(
    texts: Sequence[str],
    embedder: Embedder | EmbeddingFunction | None,
    model_name: str,
    batch_size: int,
    *,
    device: str | None = None,
    show_progress: bool = False,
) -> list[list[float]]:
    if embedder is None:
        embedder = GraphCodeBertEmbedder(model_name=model_name, device=device, show_progress=show_progress)

    if hasattr(embedder, "embed_texts"):
        return embedder.embed_texts(texts, batch_size=batch_size)
    return embedder(texts, batch_size)


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _batch_progress(
    label: str,
    *,
    batch_index: int,
    total_batches: int,
    completed_items: int,
    total_items: int,
    started_at: float,
) -> None:
    elapsed = max(time.perf_counter() - started_at, 1e-9)
    rate = completed_items / elapsed
    remaining_items = max(total_items - completed_items, 0)
    eta_seconds = int(remaining_items / rate) if rate > 0 else 0
    print(
        "\r"
        f"{label}: batch {batch_index}/{total_batches} "
        f"({completed_items}/{total_items}, {rate:.1f} chunks/s, eta {_format_duration(eta_seconds)})",
        end="",
        file=sys.stderr,
        flush=True,
    )


def _format_duration(seconds: int) -> str:
    minutes, second = divmod(seconds, 60)
    hours, minute = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minute:02d}m"
    if minute:
        return f"{minute}m {second:02d}s"
    return f"{second}s"


def _attach_linked_chunks(collection: Any, results: list[dict[str, object]], *, linked_window: int) -> None:
    for result in results:
        chunk = result.get("chunk")
        if not isinstance(chunk, dict):
            continue

        siblings = _fetch_symbol_siblings(collection, chunk)
        if len(siblings) <= 1:
            continue

        center_index = _find_matching_chunk_index(siblings, chunk)
        if center_index is None:
            continue

        start = max(center_index - linked_window, 0)
        end = min(center_index + linked_window + 1, len(siblings))
        result["linked_chunks"] = [
            sibling
            for index, sibling in enumerate(siblings[start:end], start=start)
            if index != center_index
        ]


def _fetch_symbol_siblings(collection: Any, chunk: dict[str, object]) -> list[dict[str, object]]:
    filepath = str(chunk.get("filepath", ""))
    function_name = str(chunk.get("function_name", ""))
    if not filepath or not function_name or not hasattr(collection, "get"):
        return []

    try:
        raw_siblings = collection.get(where={"function_name": function_name}, include=["documents", "metadatas"])
    except Exception:
        return []

    ids = _flat_result_list(raw_siblings, "ids")
    documents = _flat_result_list(raw_siblings, "documents")
    metadatas = _flat_result_list(raw_siblings, "metadatas")

    siblings: list[dict[str, object]] = []
    for index, chunk_id in enumerate(ids):
        metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
        if str(metadata.get("filepath", "")) != filepath:
            continue
        document = documents[index] if index < len(documents) and documents[index] is not None else ""
        siblings.append(
            Chunk(
                id=str(chunk_id),
                filepath=str(metadata.get("filepath", "")),
                function_name=str(metadata.get("function_name", "")),
                start_line=int(metadata.get("start_line", 0)),
                end_line=int(metadata.get("end_line", 0)),
                docstring=str(metadata.get("docstring", "")),
                source=str(document),
                language=str(metadata.get("language", "")),
            ).to_dict()
        )
    return sorted(siblings, key=lambda sibling: (int(sibling["start_line"]), int(sibling["end_line"])))


def _find_matching_chunk_index(siblings: Sequence[dict[str, object]], chunk: dict[str, object]) -> int | None:
    chunk_id = chunk.get("id")
    for index, sibling in enumerate(siblings):
        if sibling.get("id") == chunk_id:
            return index
    return None


def _load_cached_embeddings(path: Path, cache_key: dict[str, object]) -> list[list[float]] | None:
    if not path.exists():
        return None
    try:
        with path.open("rb") as file:
            payload = pickle.load(file)
    except (OSError, pickle.PickleError, EOFError):
        return None

    if not isinstance(payload, dict):
        return None
    if payload.get("version") != CACHE_VERSION:
        return None
    if payload.get("cache_key") != cache_key:
        return None

    embeddings = payload.get("embeddings")
    if not isinstance(embeddings, list):
        return None
    expected_count = len(cache_key["chunks"])
    if len(embeddings) != expected_count:
        return None
    return embeddings


def _cache_key(chunks: Sequence[Chunk], model_name: str) -> dict[str, object]:
    return {
        "model_name": model_name,
        "chunks": [(chunk.id, _hash_text(chunk_to_embedding_text(chunk))) for chunk in chunks],
    }


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_path_prefix(prefix: str) -> str:
    normalized = prefix.replace("\\", "/").strip("/")
    if normalized in {"", "."}:
        return ""
    return normalized


def _path_matches_prefix(filepath: str, prefix: str) -> bool:
    if not prefix:
        return True
    return filepath == prefix or filepath.startswith(f"{prefix}/")


def _is_common_test_path(filepath: str) -> bool:
    path = filepath.replace("\\", "/")
    if path.startswith(COMMON_TEST_PATH_PREFIXES):
        return True
    if Path(path).name.startswith("test_"):
        return True
    return any(part in COMMON_TEST_PATH_PARTS for part in Path(path).parts)


def _metadata_for_chunk(chunk: Chunk, model_name: str) -> dict[str, str | int]:
    return {
        "filepath": chunk.filepath,
        "function_name": chunk.function_name,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "docstring": chunk.docstring,
        "language": chunk.language,
        "model_name": model_name,
    }


def _reset_collection(persist_dir: str | Path, collection_name: str) -> Any:
    client = _chroma_client(persist_dir)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    return client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})


def _load_collection(persist_dir: str | Path, collection_name: str) -> Any:
    client = _chroma_client(persist_dir)
    try:
        return client.get_collection(name=collection_name)
    except Exception as exc:
        raise FileNotFoundError(
            f"Dense index collection {collection_name!r} was not found under {Path(persist_dir)}. "
            "Build it first with `python -m src.index_dense --repo <path>`."
        ) from exc


def _chroma_client(persist_dir: str | Path) -> Any:
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError(
            "Dense retrieval requires the `chromadb` package. "
            "Install the project dependencies before building or querying the dense index."
        ) from exc

    path = Path(persist_dir)
    path.mkdir(parents=True, exist_ok=True)
    try:
        from chromadb.config import Settings
    except ImportError:
        return chromadb.PersistentClient(path=str(path))

    return chromadb.PersistentClient(
        path=str(path),
        settings=Settings(
            anonymized_telemetry=False,
            chroma_product_telemetry_impl="src.chroma_noop_telemetry.NoopTelemetry",
        ),
    )


def _retrieval_results_from_chroma(raw_results: dict[str, Any]) -> list[dict[str, object]]:
    ids = _first_result_list(raw_results, "ids")
    documents = _first_result_list(raw_results, "documents")
    metadatas = _first_result_list(raw_results, "metadatas")
    distances = _first_result_list(raw_results, "distances")

    results: list[dict[str, object]] = []
    for index, chunk_id in enumerate(ids):
        metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
        document = documents[index] if index < len(documents) and documents[index] is not None else ""
        distance = distances[index] if index < len(distances) else None
        chunk = Chunk(
            id=str(chunk_id),
            filepath=str(metadata.get("filepath", "")),
            function_name=str(metadata.get("function_name", "")),
            start_line=int(metadata.get("start_line", 0)),
            end_line=int(metadata.get("end_line", 0)),
            docstring=str(metadata.get("docstring", "")),
            source=str(document),
            language=str(metadata.get("language", "")),
        )
        results.append(
            {
                "chunk": chunk.to_dict(),
                "score": _score_from_distance(distance),
                "rank": index + 1,
                "retriever": "dense",
            }
        )
    return results


def _first_result_list(raw_results: dict[str, Any], key: str) -> list[Any]:
    value = raw_results.get(key, [[]])
    if not value:
        return []
    first = value[0]
    return first if first is not None else []


def _flat_result_list(raw_results: dict[str, Any], key: str) -> list[Any]:
    value = raw_results.get(key, [])
    if not value:
        return []
    if isinstance(value[0], list):
        return value[0]
    return value


def _score_from_distance(distance: object) -> float:
    if distance is None:
        return 0.0
    try:
        return 1.0 - float(distance)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    raise SystemExit(main())
