"""Lesson 6: save chunks and their vectors to disk, and reload them instead of re-embedding."""

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from lesson01_load import load_documents
from lesson02_chunk import Chunk
from lesson03_tokens import MAX_TOKENS, MODEL_NAME, OVERLAP_TOKENS, chunk_documents_by_tokens
from lesson05_embed import Embedder, embedding_text

INDEX_DIR = Path(__file__).parent.parent / "index"
POOLING = "cls"


def fingerprint(documents: list[dict]) -> str:
    """A short hash of everything the index depends on: the source texts and the settings.

    If any source file or setting changes, the fingerprint changes and the index is rebuilt.
    """
    h = hashlib.sha256()
    h.update(f"{MODEL_NAME}|{POOLING}|{MAX_TOKENS}|{OVERLAP_TOKENS}".encode())
    for doc in documents:
        h.update(doc["source"].encode())
        h.update(doc["text"].encode())
    return h.hexdigest()[:16]


def build_index(documents: list[dict], embedder: Embedder) -> tuple[list[Chunk], torch.Tensor]:
    chunks = chunk_documents_by_tokens(documents, embedder.tokenizer)
    vectors = embedder.embed([embedding_text(chunk) for chunk in chunks])
    return chunks, vectors


def save_index(chunks: list[Chunk], vectors: torch.Tensor, fp: str, index_dir: Path = INDEX_DIR):
    index_dir.mkdir(exist_ok=True)
    torch.save(vectors, index_dir / "vectors.pt")
    meta = {
        "fingerprint": fp,
        "model": MODEL_NAME,
        "pooling": POOLING,
        "chunks": [asdict(chunk) for chunk in chunks],  # row i belongs to vectors[i]
    }
    with open(index_dir / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def load_index(index_dir: Path = INDEX_DIR) -> tuple[str, list[Chunk], torch.Tensor] | None:
    """Return (fingerprint, chunks, vectors), or None if there's no saved index yet."""
    chunks_file, vectors_file = index_dir / "chunks.json", index_dir / "vectors.pt"
    if not (chunks_file.exists() and vectors_file.exists()):
        return None
    with open(chunks_file, encoding="utf-8") as f:
        meta = json.load(f)
    chunks = [Chunk(**fields) for fields in meta["chunks"]]
    # weights_only=True: only load plain tensors, never arbitrary Python objects (safer).
    vectors = torch.load(vectors_file, weights_only=True)
    return meta["fingerprint"], chunks, vectors


def load_or_build_index(embedder: Embedder | None = None,
                        verbose: bool = True) -> tuple[list[Chunk], torch.Tensor]:
    """Load the saved index if it's up to date; otherwise (re)build and save it."""
    documents = load_documents()
    fp = fingerprint(documents)

    saved = load_index()
    if saved is not None and saved[0] == fp:
        if verbose:
            print(f"index is up to date (fingerprint {fp}), loaded from {INDEX_DIR.name}/")
        return saved[1], saved[2]

    if verbose:
        reason = "no saved index" if saved is None else f"sources or settings changed ({saved[0]} -> {fp})"
        print(f"building index: {reason}")
    embedder = embedder or Embedder(pooling=POOLING)
    chunks, vectors = build_index(documents, embedder)
    save_index(chunks, vectors, fp)
    return chunks, vectors


def main():
    start = time.perf_counter()
    chunks, vectors = load_or_build_index()
    print(f"took {time.perf_counter() - start:.2f} s\n")

    print("Files:")
    for path in sorted(INDEX_DIR.iterdir()):
        print(f"  {path.relative_to(INDEX_DIR.parent)}  {path.stat().st_size / 1024:.0f} KB")

    print(f"\n{len(chunks)} chunks, vectors of shape {tuple(vectors.shape)}")
    assert len(chunks) == vectors.shape[0], "every chunk needs exactly one vector"

    # Sanity check: re-embed one chunk and compare it with the saved vector.
    i = 42
    fresh = Embedder(pooling=POOLING).embed([embedding_text(chunks[i])])[0]
    print(f"chunk {i} ({chunks[i].heading_path[-45:]!r})")
    print(f"  saved vector == freshly computed vector: {torch.allclose(fresh, vectors[i], atol=1e-4)}")


if __name__ == "__main__":
    main()
