"""Lesson 10: evaluation: measure retrieval and answers on a fixed set of questions.

Usage:
    uv run lessons/lesson10_eval.py             # retrieval: baseline and experiments (about a minute)
    uv run lessons/lesson10_eval.py --answers   # also grade the generated answers (a few minutes more)
"""

import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch

from lesson01_load import load_documents
from lesson02_chunk import Chunk, chunk_documents, naive_chunks
from lesson03_tokens import MAX_TOKENS, OVERLAP_TOKENS, make_token_counter
from lesson05_embed import Embedder, embedding_text
from lesson06_index import POOLING
from lesson07_search import search
from lesson08_llm import ask
from lesson09_rag import RAG, TOP_K

QUESTIONS_FILE = Path(__file__).parent.parent / "eval" / "questions.jsonl"
KS = (1, 3, 5)

# Phrases that count as "the model said it doesn't know".
REFUSALS = ["doesn't say", "does not say", "not specify", "doesn't specify", "not specified",
            "doesn't mention", "does not mention", "not mentioned", "no information",
            "not provided", "doesn't state", "does not state", "not stated"]


def load_questions(path: Path = QUESTIONS_FILE) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def rank_of(question: dict, results: list[tuple[float, Chunk]]) -> int | None:
    """Position (1 = best) of the first result that contains one of the expected snippets."""
    for rank, (_, chunk) in enumerate(results, start=1):
        if chunk.source == question["source"] and any(s in chunk.text for s in question["snippets"]):
            return rank
    return None


@dataclass
class Setup:
    """One retrieval configuration to evaluate."""
    name: str
    chunker: str = "structure"   # "structure" (lessons 2-3) or "naive" (fixed-size characters)
    max_tokens: int = MAX_TOKENS
    heading: bool = True         # prepend the heading path before embedding (lesson 5)
    instruction: bool = True     # use the query instruction (lesson 7)
    min_tokens: int = 0          # drop chunks smaller than this


def build_chunks(setup: Setup, documents: list[dict], count) -> list[Chunk]:
    if setup.chunker == "naive":
        chunks = [Chunk(source=doc["source"], heading_path=doc["source"], text=text)
                  for doc in documents for text in naive_chunks(doc["text"], size=1200, overlap=200)]
    else:
        overlap = setup.max_tokens // 4 if setup.max_tokens != MAX_TOKENS else OVERLAP_TOKENS
        chunks = chunk_documents(documents, setup.max_tokens, overlap, count)
    return [chunk for chunk in chunks if count(chunk.text) >= setup.min_tokens]


def evaluate_retrieval(questions: list[dict], chunks: list[Chunk], vectors: torch.Tensor,
                       embedder: Embedder, instruction: bool = True) -> dict:
    """Search every question and collect ranks, recall@k and MRR."""
    ranks, top5 = [], []
    for q in questions:
        results = search(q["question"], chunks, vectors, embedder, k=10, instruction=instruction)
        top5.append([chunk for _, chunk in results[:5]])
        ranks.append(rank_of(q, results) if q["source"] else None)

    answerable = [(q, r) for q, r in zip(questions, ranks) if q["source"]]
    def recall(k, source=None):
        rows = [(q, r) for q, r in answerable if source is None or q["source"] == source]
        return sum(1 for _, r in rows if r is not None and r <= k) / len(rows)

    return {
        "ranks": ranks,
        "top5": top5,
        **{f"R@{k}": recall(k) for k in KS},
        "MRR": sum(1 / r for _, r in answerable if r is not None) / len(answerable),
        "md R@3": recall(3, "architecture.md"),
        "txt R@3": recall(3, "kestrel-island.txt"),
    }


def run_setup(setup: Setup, questions, documents, embedder, count) -> tuple[int, dict]:
    chunks = build_chunks(setup, documents, count)
    texts = [embedding_text(chunk) if setup.heading else chunk.text for chunk in chunks]
    vectors = embedder.embed(texts)
    return len(chunks), evaluate_retrieval(questions, chunks, vectors, embedder, setup.instruction)


def grade_answer(question: dict, answer: str) -> bool:
    text = answer.lower()
    if question["source"] is None:
        return any(phrase in text for phrase in REFUSALS)
    return any(keyword.lower() in text for keyword in question["answer_keywords"])


def main():
    questions = load_questions()
    documents = load_documents()
    embedder = Embedder(pooling=POOLING)
    count = make_token_counter(embedder.tokenizer)

    n_none = sum(1 for q in questions if q["source"] is None)
    per_source = Counter(q["source"] for q in questions if q["source"])
    print(f"=== The test set: {len(questions)} questions ===\n")
    for source, n in per_source.items():
        print(f"  {n:3} about {source}")
    print(f"  {n_none:3} without an answer in the documents\n")

    print("=== 1. Baseline retrieval (the setup from lessons 3-7) ===\n")
    n_chunks, base = run_setup(Setup("baseline"), questions, documents, embedder, count)
    for q, rank in zip(questions, base["ranks"]):
        if q["source"]:
            mark = f"{rank:>2}" if rank else " -"
            print(f"  rank {mark}  {q['question'][:70]}")
    print(f"\n  R@1 {base['R@1']:.2f}   R@3 {base['R@3']:.2f}   R@5 {base['R@5']:.2f}   MRR {base['MRR']:.2f}")

    print("\n=== 2. Hubness: chunks that appear most often in the top 5 ===")
    for source in per_source:
        group = [top for q, top in zip(questions, base["top5"]) if q["source"] == source]
        hubs = Counter((c.heading_path, c.text[:40]) for top in group for c in top if c.source == source)
        print(f"\n  questions about {source} ({len(group)}):")
        for (path, start), n in hubs.most_common(3):
            print(f"    in {n:2} of {len(group)} top-5 lists   {path[-50:]}   ({start.splitlines()[0][:28]!r})")

    print("\n=== 3. Experiments: change one thing at a time ===\n")
    setups = [
        Setup("baseline"),
        Setup("no query instruction", instruction=False),
        Setup("no heading path", heading=False),
        Setup("smaller chunks (200 tokens)", max_tokens=200),
        Setup("tiny chunks removed (< 30 tokens)", min_tokens=30),
        Setup("naive chunks (1200 chars)", chunker="naive", heading=False),
    ]
    print(f"  {'setup':36}{'chunks':>7}{'R@1':>7}{'R@3':>7}{'R@5':>7}{'MRR':>7}{'md R@3':>9}{'txt R@3':>9}")
    for setup in setups:
        n_chunks, r = run_setup(setup, questions, documents, embedder, count)
        print(f"  {setup.name:36}{n_chunks:>7}{r['R@1']:>7.2f}{r['R@3']:>7.2f}{r['R@5']:>7.2f}"
              f"{r['MRR']:>7.2f}{r['md R@3']:>9.2f}{r['txt R@3']:>9.2f}")

    if "--answers" not in sys.argv:
        print("\n(Run with --answers to also grade the generated answers.)")
        return

    print(f"\n=== 4. Answers (RAG with the baseline index, top {TOP_K}) ===\n")
    rag = RAG()
    tally = Counter()
    guessable = []
    for q, rank in zip(questions, base["ranks"]):
        answer, _, _ = rag.answer(q["question"])
        correct = grade_answer(q, answer)
        if q["source"] is None:
            kind = "refused ✓" if correct else "answered ✗"
        elif correct:
            kind = "correct ✓"
        else:
            # Was the right chunk in the prompt? Then it's a generation problem.
            kind = "wrong ✗ (generation)" if rank and rank <= TOP_K else "wrong ✗ (retrieval)"
        tally[kind] += 1
        print(f"  {kind:22} {q['question'][:55]}")
        print(f"  {'':22} → {answer.replace(chr(10), ' ')[:110]}")
        if q["source"]:
            # The same question without any context: can the model answer it on its own?
            without, _ = ask(rag.generator, q["question"])
            if grade_answer(q, without):
                guessable.append((q, without))
                print(f"  {'':22} ⚠ also passes without RAG: {without.replace(chr(10), ' ')[:85]}")
    print()
    for kind, n in sorted(tally.items()):
        print(f"  {kind:22} {n}")

    n_answerable = sum(1 for q in questions if q["source"])
    print(f"\n  {len(guessable)} of {n_answerable} answerable questions also pass without RAG:")
    for q, _ in guessable:
        print(f"    {q['question']}")


if __name__ == "__main__":
    main()
