# Lesson 1: Loading the documents

[← Lesson 0](lesson00_setup.md) · [Overview](../README.md) · [Lesson 2 →](lesson02_chunk.md)

> **You'll learn:** how to read the source documents into a list of texts with metadata,
> and what the two course documents look like.
>
> **Run:** `uv run lessons/lesson01_load.py`

```mermaid
flowchart LR
    L[Load] --> C[Chunk] --> E[Embed] --> I[(Index)] --> S[Search] --> P[Prompt + LLM] --> A[Answer]
    classDef here fill:#ffd166,stroke:#333,color:#000
    class L here
```

## Concept

Every RAG system starts by turning "files somewhere" into a **list of texts with
metadata**. Here, the only metadata is the file name (`source`). It stays attached to every
piece of text from now on, because an answer should be able to say where its information
came from.

## The code

```python
DATA_DIR = Path(__file__).parent.parent / "data"

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
```

- **`Path(__file__).parent.parent / "data"`**: `__file__` is the path of the script
  itself. The data folder is found relative to the script, not to the folder you run it
  from, so the script works from anywhere.
- **`rglob("*")`** finds all files, including those in subfolders. `sorted()` makes the
  order the same on every run.
- **`encoding="utf-8"`** matters: the documents contain characters like `│`, `—` and `…`.
- Each document becomes a dict with two keys, `source` and `text`.

`main()` then prints a few statistics per document.

## The output

```
Loaded 2 document(s) from data/

── architecture.md
   characters: 79,305
   words:      10,892
   lines:      1,294
   headings:   76
   first 3 headings:
     # OpenSim Architecture
     ## 1. Directory Structure
     ### 1.1 Data directories
   preview: '# OpenSim Architecture\n\nThis document describes OpenSim **bottom-up**: ...'

── kestrel-island.txt
   characters: 12,058
   words:      2,174
   lines:      168
   headings:   0
   first 3 headings:
   preview: "A Visitor's Guide to Kestrel Island\n\nKestrel Island is a small island in ..."
```

The two documents are deliberately very different:

| | `architecture.md` | `kestrel-island.txt` |
|---|---|---|
| Content | architecture of [OpenSim](https://github.com/ghtomcat/opensim), an open-source browser flight simulator | visitor's guide to a fictional island |
| Size | ~11,000 words | ~2,000 words |
| Structure | 76 headings (14 sections, ~60 subsections), tables, code blocks, identifiers | none: only paragraphs |

Neither can be known to a language model from its training: OpenSim's internals are too
specific, and the island doesn't exist. When the system answers correctly in lesson 9, the
answer must have come from retrieval.

Two details in the output:

1. **Characters vs. bytes.** `architecture.md` has 79,305 characters, but the file is 80,579
   bytes. Characters like `│` or `—` take 2–3 bytes each in UTF-8. Lesson 3 adds a third
   unit, **tokens**, which is the one models actually count.
2. **The heading count is naive.** It counts every line that starts with `#`. That's correct
   for these files, but a code block containing a shell comment (`# ...`) would be counted
   as a heading. The chunker in lesson 2 has to know whether it's inside a code block.

## Python notes

- **`Path`** (from `pathlib`) represents file paths as objects; `/` joins path parts.
- **`list[dict]`** in the type hint: a list of dictionaries. A **list** is an ordered
  collection; a **dict** maps keys to values.
- **`{".txt", ".md"}`** is a **set**; `x not in s` checks membership.
- **`with open(...) as f:`** opens a file and closes it automatically when the block ends,
  even if an error happens.
- **`text[:200]`** is a **slice**: the first 200 characters.
- **`[line for line in lines if line.startswith("#")]`** is a **list comprehension**: it
  builds a new list from the lines that match the condition.
- **`f"{len(text):,}"`** formats a number with thousands separators; **`{text[:200]!r}`**
  shows the string's raw form, so line breaks appear as `\n`.

## Key takeaways

- Loading turns files into a list of `{"source", "text"}` dicts. The source stays with the
  text through the whole pipeline.
- Plain text and Markdown look the same at this point. The difference starts to matter in
  lesson 2.

## Exercises

1. Put a small `.txt` file of your own into `data/` and run the script again. Then remove
   it, so the outputs in the next lessons match.
2. Import the loader in the Python REPL. Start it **inside `lessons/`**:
   ```bash
   cd lessons
   uv run python
   ```
   ```python
   from lesson01_load import load_documents
   docs = load_documents()
   docs[0].keys()
   docs[1]["text"][:300]
   ```
   Does importing the file run its `main()`?
3. Change the heading list so it only counts `##` headings (sections, not subsections).

<details>
<summary>Solution 1</summary>

The output shows 3 documents. Files are sorted by name, so yours appears in alphabetical
order with the others.
</details>

<details>
<summary>Solution 2</summary>

`docs[0].keys()` shows `dict_keys(['source', 'text'])`. Importing doesn't run `main()`,
thanks to the `if __name__ == "__main__":` guard.

If you get `ModuleNotFoundError: No module named 'lesson01_load'`, the REPL was started
from a different folder. The REPL looks for modules in the folder it was started from,
while a script looks in its own folder. See [troubleshooting](../docs/troubleshooting.md).
</details>

<details>
<summary>Solution 3</summary>

```python
headings = [line for line in lines if line.startswith("## ")]
```

The space matters: `startswith("##")` also matches `###` subsections and finds 75 lines;
`startswith("## ")` finds the 14 sections.
</details>

---

[← Lesson 0: Setup](lesson00_setup.md) · [Overview](../README.md) · [Lesson 2: Chunking →](lesson02_chunk.md)
