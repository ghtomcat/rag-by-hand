"""Lesson 2: split documents into chunks that keep the Markdown structure intact."""

import re
from dataclasses import dataclass

from lesson01_load import load_documents

# Chunk size limits, measured in words for now. Lesson 3 switches to tokens.
MAX_WORDS = 250
OVERLAP_WORDS = 50

# A Markdown heading: 1-6 '#' characters, a space, then the title.
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)")


@dataclass
class Chunk:
    source: str        # file the text came from
    heading_path: str  # e.g. "OpenSim Architecture > 2. The Mission > 2.2 Load lifecycle"
    text: str

    @property
    def words(self) -> int:
        return len(self.text.split())


def count_words(text: str) -> int:
    return len(text.split())


def split_sections(text: str) -> list[tuple[list[str], list[str]]]:
    """Split Markdown at headings into (heading path, body lines) pairs.

    Lines inside ``` code blocks are never treated as headings.
    """
    sections = []
    path: list[str] = []
    lines: list[str] = []
    in_code = False

    for line in text.splitlines():
        if line.startswith("```"):
            in_code = not in_code
        match = None if in_code else HEADING_RE.match(line)
        if match:
            sections.append((path, lines))  # close the previous section
            level = len(match.group(1))     # "##" -> 2
            path = path[:level - 1] + [match.group(2).strip()]
            lines = []
        else:
            lines.append(line)

    sections.append((path, lines))
    return sections


def split_blocks(lines: list[str]) -> list[str]:
    """Group lines into blocks separated by blank lines (paragraphs, tables, lists).

    A ``` code block always stays one block, even if it contains blank lines.
    """
    blocks = []
    current: list[str] = []
    in_code = False

    for line in lines:
        if line.startswith("```"):
            in_code = not in_code
        if line.strip() == "" and not in_code:
            if current:
                blocks.append("\n".join(current))
                current = []
        else:
            current.append(line)

    if current:
        blocks.append("\n".join(current))
    # Drop horizontal rules ("---"): they separate sections visually but carry no content.
    return [block for block in blocks if block.strip() != "---"]


def split_large_block(block: str, max_size: int, count=count_words) -> list[str]:
    """Split a block that is too big on its own into parts, at line boundaries.

    A code block's fence lines are repeated in every part, and so is a table's header
    (its first two lines), so each part still reads correctly on its own.
    """
    lines = block.split("\n")
    prefix: list[str] = []
    suffix: list[str] = []
    if lines[0].startswith("```") and lines[-1].startswith("```") and len(lines) > 2:
        prefix, suffix, lines = [lines[0]], [lines[-1]], lines[1:-1]
    elif lines[0].startswith("|") and len(lines) > 2:
        prefix, lines = lines[:2], lines[2:]

    parts = []
    current: list[str] = []
    for line in lines:
        if current and count("\n".join(prefix + current + [line] + suffix)) > max_size:
            parts.append("\n".join(prefix + current + suffix))
            current = []
        current.append(line)
    if current:
        parts.append("\n".join(prefix + current + suffix))
    return parts


def pack_blocks(blocks: list[str], max_size: int = MAX_WORDS,
                overlap: int = OVERLAP_WORDS, count=count_words) -> list[str]:
    """Combine consecutive blocks into chunks of at most max_size.

    `count` measures the size of a text (words by default, tokens from lesson 3 on).
    A block larger than max_size is first split by lines (split_large_block); otherwise
    blocks are never cut. If the last block of a full chunk is small, it's repeated at
    the start of the next chunk as overlap, so context carries across the boundary.
    """
    pieces = []
    for block in blocks:
        if count(block) > max_size:
            pieces.extend(split_large_block(block, max_size, count))
        else:
            pieces.append(block)

    chunks = []
    current: list[str] = []
    for piece in pieces:
        if current and count("\n\n".join(current + [piece])) > max_size:
            chunks.append("\n\n".join(current))
            last = current[-1]
            fits = count(last) <= overlap and count(last + "\n\n" + piece) <= max_size
            current = [last] if fits else []
        current.append(piece)

    if current:
        chunks.append("\n\n".join(current))
    return chunks


def chunk_documents(documents: list[dict], max_size: int = MAX_WORDS,
                    overlap: int = OVERLAP_WORDS, count=count_words) -> list[Chunk]:
    chunks = []
    for doc in documents:
        for path, lines in split_sections(doc["text"]):
            heading_path = " > ".join(path) or doc["source"]  # plain .txt files have no headings
            for text in pack_blocks(split_blocks(lines), max_size, overlap, count):
                chunks.append(Chunk(source=doc["source"], heading_path=heading_path, text=text))
    return chunks


def naive_chunks(text: str, size: int = 1500, overlap: int = 250) -> list[str]:
    """The simplest possible chunker, for comparison: fixed windows of `size` characters."""
    step = size - overlap
    return [text[start:start + size] for start in range(0, len(text), step)]


def damage_report(texts: list[str]) -> dict[str, int]:
    """Count the chunks that break the document's structure in some way."""
    report = {"cut mid-word": 0, "broken code block": 0, "table rows without header": 0,
              "heading in the middle": 0}
    for text in texts:
        if text[:1].isalnum() and not text[:1].isupper() or text[-1:].isalnum():
            report["cut mid-word"] += 1          # starts with a lower-case letter, or ends in a word
        if text.count("```") % 2 == 1:
            report["broken code block"] += 1     # a fence is opened but not closed, or vice versa
        lines = text.splitlines()
        if any(line.startswith("|") for line in lines) and not any(line.startswith("|---") for line in lines):
            report["table rows without header"] += 1
        if any(HEADING_RE.match(line) for line in lines[1:]):
            report["heading in the middle"] += 1  # two sections mixed in one chunk
    return report


def print_chunk(chunk: Chunk, max_chars: int = 300):
    print(f"[{chunk.source} | {chunk.heading_path} | {chunk.words} words]")
    preview = chunk.text if len(chunk.text) <= max_chars else chunk.text[:max_chars] + " …"
    print(preview)
    print()


def main():
    documents = load_documents()
    chunks = chunk_documents(documents)

    print("=== 1. Naive vs. structure-aware chunking ===\n")
    for doc in documents:
        naive = naive_chunks(doc["text"])
        aware = [chunk.text for chunk in chunks if chunk.source == doc["source"]]
        print(f"{doc['source']}")
        print(f"  {'':28}{'naive':>8}{'structure-aware':>18}")
        print(f"  {'chunks':28}{len(naive):>8}{len(aware):>18}")
        naive_report, aware_report = damage_report(naive), damage_report(aware)
        for problem in naive_report:
            print(f"  {problem:28}{naive_report[problem]:>8}{aware_report[problem]:>18}")
        print()

    print("The start of a naive chunk from architecture.md (cut in the middle of a table):\n")
    md = next(doc for doc in documents if doc["source"].endswith(".md"))
    for text in naive_chunks(md["text"]):
        first_line = text.splitlines()[0]
        if "|" in first_line and not first_line.startswith("|"):
            print("▌" + text[:400] + " …\n")
            break

    print("=== 2. Structure-aware chunks ===\n")
    sizes = sorted(chunk.words for chunk in chunks)
    print(f"{len(chunks)} chunks  (max {MAX_WORDS} words, overlap up to {OVERLAP_WORDS} words)")
    print(f"words per chunk: min {sizes[0]}, median {sizes[len(sizes) // 2]}, max {sizes[-1]}\n")

    print("Size histogram:")
    bucket = 50
    for start in range(0, sizes[-1] + 1, bucket):
        n = sum(1 for size in sizes if start <= size < start + bucket)
        print(f"  {start:4}-{start + bucket - 1:<4} {'█' * n} {n}")
    print()

    print("=== Example: a chunk of the plain-text sample (no headings) ===\n")
    print_chunk(next(chunk for chunk in chunks if chunk.source.endswith(".txt")))

    print("=== Example: the chunk(s) for '2.2 Load lifecycle' ===\n")
    for chunk in chunks:
        if chunk.heading_path.endswith("2.2 Load lifecycle"):
            print_chunk(chunk, max_chars=2000)

    print("=== The 5 smallest chunks ===\n")
    for chunk in sorted(chunks, key=lambda c: c.words)[:5]:
        print_chunk(chunk)


if __name__ == "__main__":
    main()
