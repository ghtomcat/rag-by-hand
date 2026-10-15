# Python primer

[Overview](../README.md)

The Python features used in the course, in the order they appear. Each lesson also explains
the features it introduces. This page is for looking things up.

## Running Python

```bash
uv run lessons/lesson01_load.py   # run a script with the project's Python and packages
uv run python                     # start an interactive Python session (the REPL); exit() leaves it
```

`uv` keeps the project's packages in a local `.venv/` folder and the exact versions in
`uv.lock`, so they don't conflict with anything else on your computer.

## Basics

```python
# Indentation is part of the syntax: a block starts after ":" and lasts while lines stay indented.
if count > 0:
    print("positive")
else:
    print("zero or negative")

name = "Kestrel"                # variables need no declaration
MAX_WORDS = 250                 # UPPER_CASE means "constant", by convention only
f"{name} has {count:,} words"   # f-string: values inside {}; ":," adds thousands separators
f"{score:.2f}"                  # 2 decimal places
f"{text!r}"                     # show the raw form of a string, e.g. with \n visible
"""A string
over several lines"""           # triple quotes
```

`None` means "no value". `and`, `or`, `not` are the logical operators. `a or b` returns `a`
if it's "truthy" (not empty, not zero, not None), otherwise `b`.

## Collections

```python
words = ["dog", "cat"]                     # list: ordered, changeable
words.append("jet")
words[0], words[-1]                        # first, last
words[1:3]                                 # slice: elements 1 and 2
doc = {"source": "a.md", "text": "..."}    # dict: key → value
doc["source"]
suffixes = {".txt", ".md"}                 # set: unique values, fast "in" checks
".md" in suffixes
pair = (0.79, "chunk")                     # tuple: fixed-size, unchangeable
score, chunk = pair                        # unpacking
```

## Loops and comprehensions

```python
for word in words: ...
for i, word in enumerate(words, start=1): ...      # with a counter
for score, chunk in zip(scores, chunks): ...       # two lists side by side
for start in range(0, 100, 16): ...                # 0, 16, 32, ... 96

[w.upper() for w in words]                         # list comprehension: transform
[w for w in words if len(w) > 3]                   # ... and filter
sum(1 for w in words if w.startswith("d"))         # count matches
```

## Functions

```python
def count_words(text: str) -> int:        # type hints: documentation, not enforced
    """Docstring: what the function does."""
    return len(text.split())

def pack(blocks, max_size=250, count=count_words): ...   # default values; functions are values
pack(blocks, max_size=400)                # keyword argument
pack(**options)                           # pass a dict's entries as keyword arguments
sorted(chunks, key=lambda c: c.words)     # lambda: a small anonymous function
```

A function defined inside another function can use the outer function's variables (a
closure), as `make_token_counter` does in lesson 3.

## Classes and dataclasses

```python
from dataclasses import dataclass

@dataclass                     # generates the constructor from the fields
class Chunk:
    source: str
    text: str

    @property                  # computed attribute: chunk.words, without ()
    def words(self) -> int:
        return len(self.text.split())

class Embedder:
    def __init__(self, model_name: str):   # constructor; self is the object itself
        self.model_name = model_name
```

`@something` above a function or class is a **decorator**: it wraps or modifies what
follows. `@torch.inference_mode()` (lesson 5) is another example.

## Modules and imports

```python
import sys                                   # a module from the standard library
import torch.nn.functional as F              # with a short alias
from lesson01_load import load_documents     # a function from another lesson's file
```

A file name is a module name, so it can't start with a digit. That's why the lessons are
named `lesson01_load.py` and not `01_load.py`. The block

```python
if __name__ == "__main__":
    main()
```

runs `main()` only when the file is executed directly, not when it's imported by another
lesson.

## Files, errors and checks

```python
from pathlib import Path
path = Path(__file__).parent / "data"        # paths as objects; "/" joins them
with open(path, encoding="utf-8") as f:      # "with" closes the file automatically
    text = f.read()

try:
    risky()
except RuntimeError as error:                # catch only this kind of error
    print(error)

assert len(chunks) == len(vectors), "message"   # stop immediately if this is false
```
