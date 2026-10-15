"""Lesson 1: load the source documents (.txt and .md files) from the data/ folder."""

from pathlib import Path

# The data/ folder, found relative to this file (not to the current working directory),
# so the script works no matter where you run it from.
DATA_DIR = Path(__file__).parent.parent / "data"

SUPPORTED_SUFFIXES = {".txt", ".md"}


def load_documents(data_dir: Path = DATA_DIR) -> list[dict]:
    """Read every .txt/.md file under data_dir and return one dict per file."""
    documents = []
    for path in sorted(data_dir.rglob("*")):
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        with open(path, encoding="utf-8") as f:
            text = f.read()
        documents.append({
            "source": str(path.relative_to(data_dir)),
            "text": text,
        })
    return documents


def main():
    documents = load_documents()
    print(f"Loaded {len(documents)} document(s) from {DATA_DIR.name}/\n")

    for doc in documents:
        text = doc["text"]
        lines = text.splitlines()
        headings = [line for line in lines if line.startswith("#")]

        print(f"── {doc['source']}")
        print(f"   characters: {len(text):,}")
        print(f"   words:      {len(text.split()):,}")
        print(f"   lines:      {len(lines):,}")
        print(f"   headings:   {len(headings)}")
        print("   first 3 headings:")
        for heading in headings[:3]:
            print(f"     {heading}")
        print(f"   preview: {text[:200]!r}")
        print()


if __name__ == "__main__":
    main()
