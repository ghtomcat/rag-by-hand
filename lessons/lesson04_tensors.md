# Lesson 4: Tensors 101

[← Lesson 3](lesson03_tokens.md) · [Overview](../README.md) · [Lesson 5 →](lesson05_embed.md)

> **You'll learn:** the PyTorch operations needed for embedding and search (shapes,
> broadcasting, reductions, masking, cosine similarity, top-k and devices), on numbers
> small enough to check by hand.
>
> **Run:** `uv run lessons/lesson04_tensors.py`

This lesson is a detour: it doesn't add a step to the pipeline. Every tensor operation used
in lessons 5–7 appears here once.

## Concept

A **tensor** is a multi-dimensional array of numbers: rectangular, all of one type
(`dtype`, e.g. `float32`), stored on one **device**, and with fast math built in. When in
doubt, look at **`.shape`**: most PyTorch bugs are shape mismatches.

## 1–2. Creating and indexing

```
x = [[1, 2, 3],
     [4, 5, 6]]        shape (2, 3), dtype float32, device cpu
x[0]    -> [1, 2, 3]   first row, shape (3,)
x[:, 1] -> [2, 5]      column 1 of every row (":" means "all")
x[1, 2] -> 6.0         a single number; .item() turns it into a plain Python number
```

`x[1, 2]` indexes two dimensions at once.

## 3. Broadcasting

```
x + [10, 20, 30]  ->  [[11, 22, 33],
                       [14, 25, 36]]   the row is added to every row of x
```

PyTorch automatically "stretches" the smaller tensor to fit, without copying it. To multiply
each **row** by its own number instead, the shape has to say so: **`unsqueeze(1)`** turns
shape `(2,)` into `(2, 1)`, a column:

```
x * [1, 0].unsqueeze(1)  ->  [[1, 2, 3],
                              [0, 0, 0]]   row 0 kept, row 1 zeroed
```

That's exactly how an attention mask is applied.

## 4. Reductions along a dimension

`dim` says **which dimension disappears**:

```
x.sum(dim=0)  -> [5, 7, 9]   down the columns, shape (3,)
x.mean(dim=1) -> [2, 5]      across each row, shape (2,)
```

## 5. Masked mean pooling

One common way to get a single vector for a whole text is to average its token vectors.
With padding, that needs care. Here are two texts with 4 token positions each, every token
a vector of 3 numbers. Text 0 has only 2 real tokens, `[1,0,0]` and `[3,0,0]`; positions 2
and 3 are padding, filled with 9s:

```
naive mean:   [5.50, 4.50, 4.50]   ← wrong: the padding was averaged in
masked mean:  [2.00, 0.00, 0.00]   ← correct: (1 + 3) / 2 = 2
```

```python
m = mask.unsqueeze(-1).float()                 # (2, 4) -> (2, 4, 1)
masked = (tokens * m).sum(dim=1) / m.sum(dim=1)
```

Expand the mask so it broadcasts over the vectors, zero out the padding, and divide by the
number of **real** tokens. Without the mask, a short text would get a different vector
depending on which longer texts happened to be in the same batch. Lesson 5 shows that our
embedding model uses a different pooling method, but many other models use exactly this
one.

## 6. Vectors as meaning: cosine similarity and top-k

Six words, each with a hand-made vector of 3 numbers, think of them as
`[animal, flying, size]`:

```python
E = F.normalize(E, dim=1)   # every row now has length 1
sims = E @ E.T              # (6, 3) @ (3, 6) -> (6, 6): every word against every word
```

```
                dog    puppy      cat airplane      jet    eagle
       dog     1.00     0.96     0.99     0.34     0.30     0.69
  airplane     0.34     0.15     0.25     1.00     0.99     0.70
     eagle     0.69     0.68     0.69     0.70     0.74     1.00
```

- **`F.normalize`** scales each vector to length 1. After that, the **dot product is the
  cosine similarity**: 1 means same direction, 0 means unrelated.
- **`E @ E.T`** compares all words with each other in one matrix multiplication. Animals
  are close to animals, aircraft to aircraft, and the **eagle** sits in between: it's an
  animal that flies.

**Search** is the same operation with one query vector:

```python
scores = E @ query                 # (6, 3) @ (3,) -> (6,): one score per word
top = torch.topk(scores, k=3)      # top 3: jet, airplane, eagle
```

Lesson 7 does exactly this, with 119 chunks instead of 6 words and 384 numbers instead of
3.

The example also shows a limit: dog–cat (0.99) comes out *more* similar than dog–puppy
(0.96). Three hand-picked numbers can only capture three ideas. Real embedding models learn
hundreds of dimensions from large amounts of text.

## 7. Devices

```
computed on mps:0: tensor([0.37, 0.22, 0.30, 0.99, 1.00, 0.79], device='mps:0')
back to plain Python: [0.37, 0.22, 0.3, 0.99, 1.0, 0.79]
mixing devices fails: Expected vec.is_mps() to be true, but got false.
```

- **`.to(device)`** copies a tensor to a device; **`.cpu().tolist()`** brings it back as a
  plain Python list.
- All tensors in one operation must be on the **same device**. The error message isn't
  helpful, so it's worth recognizing.

## Python notes

- **`import torch.nn.functional as F`** imports a module under a short alias.
- **`@`** is matrix multiplication, **`.T`** transposes (swaps rows and columns).
- **`try: ... except RuntimeError as error:`** catches only errors of that type.
- **`f"{w:>9}"`** right-aligns a value in 9 characters.

## Key takeaways

- Check `.shape` first when something goes wrong.
- `dim` says which dimension a reduction removes; `unsqueeze` adds one so broadcasting
  works the way you mean it.
- Normalized vectors + matrix multiplication + `topk` = similarity search.

## Exercises

Start a REPL in `lessons/` with `uv run python`, then:

```python
import torch
x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
x.mean(dim=0)                        # 1. predict the result first, then check
x.sum(dim=1, keepdim=True).shape     # 2. what does keepdim do?
x @ torch.tensor([1.0, 0.0, 1.0])    # 3. what does this compute for each row?
```

4. In section 6 of the script, change the query to `[0.9, 0.0, 0.3]`. Which words come out
   on top?

<details>
<summary>Solution 1</summary>

`tensor([2.5, 3.5, 4.5])`: the mean of each column, `(1+4)/2`, `(2+5)/2`, `(3+6)/2`.
</details>

<details>
<summary>Solution 2</summary>

`torch.Size([2, 1])`. `keepdim=True` keeps the reduced dimension with size 1 instead of
removing it, so the result is a column and can be broadcast against `x` again, e.g.
`x / x.sum(dim=1, keepdim=True)` divides each row by its own sum.
</details>

<details>
<summary>Solution 3</summary>

`tensor([4., 10.])`: for each row, the sum of its first and third elements (`1+3`, `4+6`).
Multiplying a matrix by a vector computes one dot product per row, which is exactly what
search does in lesson 7.
</details>

<details>
<summary>Solution 4</summary>

`cat` (1.00), `puppy` (0.99), `dog` (0.98): a query that is "animal, not flying, small"
matches the three ground animals.
</details>

---

[← Lesson 3: Tokens](lesson03_tokens.md) · [Overview](../README.md) · [Lesson 5: Embedding by hand →](lesson05_embed.md)
