"""Lesson 3: see text the way the embedding model sees it, as tokens."""

from transformers import AutoTokenizer

from lesson01_load import load_documents
from lesson02_chunk import Chunk, chunk_documents

# The embedding model we'll use from lesson 5 on. Each model has its own tokenizer,
# so we load the tokenizer that belongs to it.
MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Chunk limits in tokens. The model reads at most 512; we stay well below that to leave
# room for the heading path we'll prepend in lesson 5, plus [CLS] and [SEP].
MAX_TOKENS = 400
OVERLAP_TOKENS = 100


def make_token_counter(tokenizer):
    """Return a function that counts the tokens of a text (without [CLS]/[SEP])."""
    def count_tokens(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])
    return count_tokens


def chunk_documents_by_tokens(documents: list[dict], tokenizer) -> list[Chunk]:
    """Lesson 2's chunker, but measuring chunk size in tokens instead of words."""
    return chunk_documents(documents, MAX_TOKENS, OVERLAP_TOKENS, make_token_counter(tokenizer))


def show_tokens(tokenizer, text: str):
    ids = tokenizer(text)["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(ids)
    print(f"text:   {text!r}")
    print(f"ids:    {ids}")
    print(f"tokens: {tokens}")
    print(f"{len(text.split())} words -> {len(ids)} tokens\n")


def main():
    # Downloads a few small files on the first run, then uses the local cache.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"Tokenizer for {MODEL_NAME}")
    print(f"vocabulary size: {tokenizer.vocab_size:,}, max length: {tokenizer.model_max_length} tokens\n")

    print("=== 1. From text to tokens ===\n")
    show_tokens(tokenizer, "The mission file decides which aircraft loads.")
    show_tokens(tokenizer, "Modules read `S.mission.*` directly via core/mission.js loadMission().")
    show_tokens(tokenizer, "Turbulence modelling uses a Dryden spectrum.")

    print("=== 2. A batch: padding and the attention mask ===\n")
    batch = tokenizer(["Gear down.", "The autopilot holds the selected altitude."],
                      padding=True, return_tensors="pt")
    print(f"input_ids shape: {tuple(batch['input_ids'].shape)}  (texts, tokens)")
    print(f"input_ids:\n{batch['input_ids']}")
    print(f"attention_mask:\n{batch['attention_mask']}\n")

    print("=== 3. Our chunks in tokens ===\n")
    chunks = chunk_documents(load_documents())
    counts = [len(tokenizer(chunk.text)["input_ids"]) for chunk in chunks]
    limit = tokenizer.model_max_length

    print(f"{len(chunks)} chunks: min {min(counts)}, median {sorted(counts)[len(counts) // 2]}, "
          f"max {max(counts)} tokens")
    print(f"total: {sum(c.words for c in chunks):,} words -> {sum(counts):,} tokens\n")

    # Tokens per word shows how "expensive" a chunk's text is for the tokenizer.
    rows = sorted(zip(counts, chunks), key=lambda row: row[0] / row[1].words)
    print("Lowest tokens per word (prose):")
    for count, chunk in rows[:3]:
        print(f"  {count / chunk.words:.2f}  {chunk.heading_path[-60:]}")
    print("Highest tokens per word (code, identifiers, tables):")
    for count, chunk in rows[-3:]:
        print(f"  {count / chunk.words:.2f}  {chunk.heading_path[-60:]}")
    print()

    too_long = [(count, chunk) for count, chunk in zip(counts, chunks) if count > limit]
    print(f"Chunks over the {limit}-token limit: {len(too_long)}")
    for count, chunk in too_long:
        cut = tokenizer.decode(tokenizer(chunk.text)["input_ids"][limit - 1:limit + 30])
        print(f"  {count} tokens  {chunk.heading_path[-60:]}")
        print(f"  the model would stop reading here: {cut[:100]!r} …")
    print()

    print("=== 4. The fix: chunk by tokens instead of words ===\n")
    count_tokens = make_token_counter(tokenizer)
    documents = load_documents()
    before = chunk_documents(documents)  # lesson 2: max 250 words
    after = chunk_documents_by_tokens(documents, tokenizer)  # max 400 tokens

    print(f"{'':24}{'before (words)':>16}{'after (tokens)':>16}")
    for label, func in [("chunks", len),
                        ("min tokens", lambda cs: min(count_tokens(c.text) for c in cs)),
                        ("median tokens", lambda cs: sorted(count_tokens(c.text) for c in cs)[len(cs) // 2]),
                        ("max tokens", lambda cs: max(count_tokens(c.text) for c in cs)),
                        (f"over {limit} (with CLS/SEP)", lambda cs: sum(1 for c in cs if count_tokens(c.text) + 2 > limit))]:
        print(f"{label:24}{func(before):>16}{func(after):>16}")
    print()

    print("Table split with a repeated header (2.3 Field groups): first 3 lines of each part")
    for chunk in after:
        if chunk.heading_path.endswith("2.3 Field groups and their consumers"):
            print(f"  --- {count_tokens(chunk.text)} tokens")
            for line in chunk.text.splitlines()[:3]:
                print(f"  {line[:90]}")


if __name__ == "__main__":
    main()
