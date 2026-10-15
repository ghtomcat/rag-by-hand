"""Lesson 7: search: find the chunks whose vectors are closest to the question's vector.

Usage:
    uv run lessons/lesson07_search.py                     # run the example questions
    uv run lessons/lesson07_search.py "your question"     # search for your own question
"""

import sys

import torch

from lesson02_chunk import Chunk
from lesson05_embed import Embedder
from lesson06_index import POOLING, load_or_build_index

# BGE models were trained with this prefix on short search queries (but not on the passages).
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Example questions, each with a snippet that the chunk holding the answer must contain
# (in its heading path or text). None means the documents don't answer the question.
EXAMPLES = [
    ("How do I run OpenSim locally?", "python3 -m http.server"),
    ("What happens during step 5 of mission loading?", "2.2 Load lifecycle"),
    ("Which file decides which aircraft loads, the catalog or the mission file?", "2.1 Two files per mission"),
    ("What are the crash conditions?", "7.5 Crash conditions"),
    ("Does OpenSim support multiplayer?", "Multi-crew isn't wired up"),
    ("Who is the author of OpenSim?", None),
    ("When does the first ferry leave Port Alda on a Sunday?", "there is no morning sailing"),
    ("Can I bring my car to Kestrel Island?", "Private cars are not allowed"),
    ("When can visitors climb the lighthouse?", "127 steps"),
]


def search(query: str, chunks: list[Chunk], vectors: torch.Tensor, embedder: Embedder,
           k: int = 5, instruction: bool = True) -> list[tuple[float, Chunk]]:
    """Return the k best (score, chunk) pairs for the query, best first."""
    text = QUERY_INSTRUCTION + query if instruction else query
    q = embedder.embed([text])[0]              # (384,)
    scores = vectors @ q                       # (chunks, 384) @ (384,) -> (chunks,)
    top = torch.topk(scores, k=min(k, len(chunks)))
    return [(score, chunks[i]) for score, i in zip(top.values.tolist(), top.indices.tolist())]


def rank_of(expected: str, results: list[tuple[float, Chunk]]) -> int | None:
    """Position (1 = best) of the first result containing the expected snippet, or None."""
    for rank, (_, chunk) in enumerate(results, start=1):
        if expected in chunk.heading_path or expected in chunk.text:
            return rank
    return None


def print_results(query: str, results: list[tuple[float, Chunk]]):
    print(f"\nQ: {query}")
    for rank, (score, chunk) in enumerate(results, start=1):
        first_line = chunk.text.strip().splitlines()[0][:70]
        print(f"  {rank}. {score:.3f}  {chunk.heading_path[-55:]:55}  {first_line}")


def main():
    embedder = Embedder(pooling=POOLING)
    chunks, vectors = load_or_build_index(embedder)

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print_results(query, search(query, chunks, vectors, embedder))
        return

    for query, _ in EXAMPLES:
        print_results(query, search(query, chunks, vectors, embedder))

    print("\nWhere does the expected answer rank? (top 10, '-' = not found)")
    print(f"  {'question':58} {'with':>6} {'without':>8}  instruction")
    for query, expected in EXAMPLES:
        if expected is None:
            print(f"  {query[:58]:58} {'(no answer in the documents)':>16}")
            continue
        ranks = [rank_of(expected, search(query, chunks, vectors, embedder, k=10, instruction=flag))
                 for flag in (True, False)]
        print(f"  {query[:58]:58} {ranks[0] or '-':>6} {ranks[1] or '-':>8}")


if __name__ == "__main__":
    main()
