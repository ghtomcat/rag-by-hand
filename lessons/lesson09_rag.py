"""Lesson 9: RAG: retrieve chunks (lesson 7) and let the LLM (lesson 8) answer from them.

Usage:
    uv run lessons/lesson09_rag.py                     # example questions, with and without RAG
    uv run lessons/lesson09_rag.py "your question"     # one question, with RAG
"""

import sys
import time

from lesson02_chunk import Chunk
from lesson05_embed import Embedder
from lesson06_index import POOLING, load_or_build_index
from lesson07_search import EXAMPLES, search
from lesson08_llm import Generator, ask

TOP_K = 4

RAG_SYSTEM_PROMPT = """You answer questions about the user's documents.
Use only the numbered context passages in the user's message, never outside knowledge.
Answer the question directly, in one to three sentences of your own words. Don't copy passages.
Cite the passages you used, like [1] or [2].
If the passages don't contain the answer, reply exactly: "The documentation doesn't say."
Be concise."""


def build_prompt(question: str, results: list[tuple[float, Chunk]]) -> str:
    """Put the retrieved chunks, numbered and labelled with their source, before the question."""
    passages = []
    for n, (_, chunk) in enumerate(results, start=1):
        passages.append(f"[{n}] ({chunk.source} › {chunk.heading_path})\n{chunk.text}")
    context = "\n\n".join(passages)
    return f"Context:\n\n{context}\n\nQuestion: {question}"


class RAG:
    """The whole pipeline: index + embedder for retrieval, generator for the answer."""

    def __init__(self, k: int = TOP_K):
        self.k = k
        self.embedder = Embedder(pooling=POOLING)
        self.chunks, self.vectors = load_or_build_index(self.embedder)
        self.generator = Generator()

    def answer(self, question: str) -> tuple[str, list[tuple[float, Chunk]], str]:
        """Return (answer, retrieved results, the full user prompt)."""
        results = search(question, self.chunks, self.vectors, self.embedder, k=self.k)
        prompt = build_prompt(question, results)
        answer, _ = ask(self.generator, prompt, system=RAG_SYSTEM_PROMPT)
        return answer, results, prompt


def print_rag_answer(rag: RAG, question: str):
    start = time.perf_counter()
    answer, results, prompt = rag.answer(question)
    seconds = time.perf_counter() - start
    n_prompt_tokens = len(rag.generator.tokenizer(prompt)["input_ids"])
    print("  retrieved:")
    for n, (score, chunk) in enumerate(results, start=1):
        print(f"    [{n}] {score:.3f}  {chunk.heading_path[-60:]}")
    print(f"  with RAG:    {answer}")
    print(f"               ({n_prompt_tokens} prompt tokens, {seconds:.1f} s)")


def main():
    rag = RAG()

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print(f"\nQ: {question}")
        print_rag_answer(rag, question)
        return

    question = EXAMPLES[2][0]
    print(f"\n=== The prompt the LLM sees, for: {question!r} ===\n")
    print(f"--- system ---\n{RAG_SYSTEM_PROMPT}\n--- user ---")
    _, _, prompt = rag.answer(question)
    print(prompt[:1200] + "\n[…]\n" + prompt[-300:])

    print("\n=== Without vs. with RAG ===")
    for question, _ in EXAMPLES:
        print(f"\nQ: {question}")
        without, _ = ask(rag.generator, question)
        print(f"  without RAG: {without}")
        print_rag_answer(rag, question)


if __name__ == "__main__":
    main()
