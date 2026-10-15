# Lesson 5: Embedding by hand

[← Lesson 4](lesson04_tensors.md) · [Overview](../README.md) · [Lesson 6 →](lesson06_index.md)

> **You'll learn:** how a transformer model turns a text into one vector, why the pooling
> method must match the model, and how to embed all chunks in batches.
>
> **Run:** `uv run lessons/lesson05_embed.py`

```mermaid
flowchart LR
    L[Load] --> C[Chunk] --> E[Embed] --> I[(Index)] --> S[Search] --> P[Prompt + LLM] --> A[Answer]
    classDef here fill:#ffd166,stroke:#333,color:#000
    class E here
```

## Concept: from tokens to one vector

```
"The mission file decides…"
     │ tokenizer (lesson 3)
     ▼
[CLS] the mission file decides … [SEP]         10 token IDs
     │ model: 12 transformer layers
     ▼
10 vectors × 384 numbers                        one vector per token, shape (1, 10, 384)
     │ pooling
     ▼
1 vector × 384 numbers                          shape (384,)
     │ normalize to length 1
     ▼
the embedding
```

**What the layers do:** in each of the 12 layers, every token "looks at" all the other
tokens (this is called *attention*) and updates its vector accordingly. After 12 layers, the
vector for "loads" no longer just means "the word loads", but "loads, in a sentence about
a mission file and an aircraft".

**Pooling** turns the token vectors into a single vector for the whole text. There are
two common methods:

- **Mean pooling:** average all token vectors, ignoring padding (lesson 4, section 5).
- **CLS pooling:** take the vector at position 0, the `[CLS]` token. It doesn't stand for a
  word; during training, the model learned to collect the meaning of the whole text there.

**The pooling method must match how the model was trained.** You can check it in the
model's files on the Hugging Face Hub. For `bge-small-en-v1.5`, `1_Pooling/config.json`
says:

```
pooling_mode_cls_token: True, pooling_mode_mean_tokens: False
```

So we use CLS pooling. The script implements both and compares them.

## The code

```python
class Embedder:
    def __init__(self, model_name=MODEL_NAME, pooling="cls", device=None):
        self.device = device or pick_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.pool = POOLING[pooling]

    @torch.inference_mode()
    def embed(self, texts, batch_size=16):
        vectors = []
        for start in range(0, len(texts), batch_size):
            batch = self.tokenizer(texts[start:start + batch_size], padding=True, truncation=True,
                                   max_length=512, return_tensors="pt").to(self.device)
            token_vectors = self.model(**batch).last_hidden_state       # (texts, positions, 384)
            pooled = self.pool(token_vectors, batch["attention_mask"])  # (texts, 384)
            vectors.append(F.normalize(pooled, dim=1).cpu())
        return torch.cat(vectors)                                       # (all texts, 384)
```

- **`.eval()`** switches off dropout, a source of randomness used only during training.
  Without it, the same text would get slightly different vectors each time.
- **`@torch.inference_mode()`** tells PyTorch we only run the model and never train it, so
  it skips the bookkeeping training needs. Faster, and uses less memory.
- **Batches of 16:** texts are processed 16 at a time, padded to the same length, with the
  attention mask from lesson 3.
- **`truncation=True`** is a safety net: thanks to lesson 3, no chunk is too long, but if
  one were, it would be cut instead of causing an error.
- **`F.normalize`** gives every vector length 1, so that search can use the dot product as
  cosine similarity (lesson 4).

The text that is actually embedded includes the chunk's heading path:

```python
def embedding_text(chunk: Chunk) -> str:
    return f"{chunk.heading_path}\n\n{chunk.text}"
```

This gives a chunk like a table part its context ("2. The Mission > 2.3 Field groups"),
even if its own text never says what it's about.

## The output

```
BAAI/bge-small-en-v1.5 on mps
33.4M parameters, 12 layers, vectors of 384 numbers
```

**One sentence, step by step:** 10 tokens go in, the model returns shape `(1, 10, 384)`,
CLS pooling keeps `(384,)`, and normalizing scales the vector from length 9.10 to 1.00.

**Similar meaning, similar vectors:**

```
  A: How is the mission file loaded?
  B: loadMission fetches the mission JSON, then the aircraft JSON.
  C: The mission picker shows one card per mission.
  D: Rockets switch between pad hold, ascent and orbit.
  E: My grandmother bakes bread on Sundays.

cls pooling:
           A      B      C      D      E
    A   1.00   0.79   0.68   0.58   0.39
    ...
```

- **The question (A) is closest to its answer (B), at 0.79**, although they share almost no
  words ("loaded" vs. "loadMission fetches"). This is why embeddings are used instead of
  keyword search.
- The related sentence C scores 0.68, the same-project sentence D 0.58, and the bread
  sentence E 0.39.
- **Even the unrelated sentence scores 0.39, not 0.** This model places all texts within a
  narrow range of directions. **The order of the scores matters, not their absolute
  values.**
- **Mean pooling** gives the same order with lower values. Both work here, but CLS is what
  this model was trained for.

**All chunks:**

```
119 chunks with heading path: max 434 tokens, 0 over 512 (would be truncated)
embedded in 1.9 s -> tensor of shape (119, 384), 178 KB
```

Lesson 3's limit of 400 tokens paid off: with the heading path added, the longest text is
434 tokens, and nothing is truncated. Both documents together fit into 178 KB: 119 × 384
numbers × 4 bytes.

## Python notes

- **`class Embedder:`** defines a class. **`__init__`** is the constructor; **`self`** is
  the object itself and is the first parameter of every method.
- **`str | None`** in a type hint means "a string or None".
- **`device or pick_device()`** uses the given device, or picks one if none was given.
- **`@torch.inference_mode()`** is a **decorator**: it wraps the method that follows.
- **`self.model(**batch)`** passes the dict's entries (`input_ids=…`, `attention_mask=…`)
  as keyword arguments.
- **`range(0, len(texts), batch_size)`** counts 0, 16, 32, …
- **`self.pool = POOLING[pooling]`** looks up a function in a dict and stores it.
- **`torch.cat(vectors)`** joins a list of tensors into one.

## Key takeaways

- A transformer produces one vector per token; pooling combines them into one vector per
  text.
- Use the pooling the model was trained with; its configuration tells you which.
- Similarity scores are only meaningful relative to each other.

## Exercises

1. Add a sentence F to section 3, e.g. "What happens when the aircraft crashes?". Which of
   A–E is it closest to?
2. What would go wrong in search if `F.normalize(...)` were missing from `embed()`?
3. Change `batch_size=16` to `1` and to `64`, and compare the time for embedding all
   chunks. Why does batching speed things up on a GPU?

<details>
<summary>Solution 1</summary>

F's similarities: A 0.48, **B 0.58**, C 0.48, **D 0.53**, E 0.31. It's closest to B and D,
the two sentences about what the simulation *does* (loading, rocket phases), and furthest
from the bread sentence. None of A–E is about crashes, so all scores are moderate.
</details>

<details>
<summary>Solution 2</summary>

Without normalizing, the dot product depends on the vectors' **lengths** as well as their
directions (the sentence in section 2 had length 9.10). Chunks with long vectors would
score high for every question, regardless of meaning. Normalizing makes the dot product
equal to cosine similarity, which compares directions only.
</details>

<details>
<summary>Solution 3</summary>

With `batch_size=1`, embedding takes noticeably longer; with 64, about as long as with 16
or slightly less. A GPU computes many operations in parallel, and one short text doesn't
give it enough work to keep it busy. Larger batches use the hardware better, up to the
point where memory or padding (long and short texts in the same batch) limits the gain.
</details>

---

[← Lesson 4: Tensors 101](lesson04_tensors.md) · [Overview](../README.md) · [Lesson 6: The index →](lesson06_index.md)
