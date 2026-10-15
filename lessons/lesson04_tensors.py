"""Lesson 4: the PyTorch tensor operations we need for embedding and search."""

import torch
import torch.nn.functional as F

from lesson00_setup import pick_device


def section(title: str):
    print(f"\n=== {title} ===\n")


def main():
    torch.set_printoptions(precision=2, sci_mode=False)

    section("1. Creating tensors")
    x = torch.tensor([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]])
    print(x)
    print(f"shape: {tuple(x.shape)}  (2 rows, 3 columns)")
    print(f"dtype: {x.dtype}, device: {x.device}")
    print(f"zeros(2, 4):\n{torch.zeros(2, 4)}")

    section("2. Indexing and slicing")
    print(f"x[0]    -> {x[0]}     first row, shape {tuple(x[0].shape)}")
    print(f"x[:, 1] -> {x[:, 1]}          column 1 of every row, shape {tuple(x[:, 1].shape)}")
    print(f"x[1, 2] -> {x[1, 2]}           a single number (0-d tensor); .item() -> {x[1, 2].item()}")

    section("3. Element-wise operations and broadcasting")
    print(f"x * 2:\n{x * 2}")
    row = torch.tensor([10.0, 20.0, 30.0])         # shape (3,)
    print(f"x + row  (row added to every row of x):\n{x + row}")
    col = torch.tensor([1.0, 0.0])                 # shape (2,)
    print(f"col.unsqueeze(1) has shape {tuple(col.unsqueeze(1).shape)}")
    print(f"x * col.unsqueeze(1)  (row 0 kept, row 1 zeroed):\n{x * col.unsqueeze(1)}")

    section("4. Reductions along a dimension")
    print(f"x.sum(dim=0)  -> {x.sum(dim=0)}   (down the columns, shape {tuple(x.sum(dim=0).shape)})")
    print(f"x.mean(dim=1) -> {x.mean(dim=1)}          (across each row, shape {tuple(x.mean(dim=1).shape)})")

    section("5. Masked mean pooling (one way to pool token vectors, see lesson 5)")
    # A batch of 2 texts, 4 token positions each, every token a 3-number vector.
    # Text 0 has only 2 real tokens; positions 2 and 3 are padding.
    tokens = torch.tensor([[[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [9.0, 9.0, 9.0], [9.0, 9.0, 9.0]],
                           [[0.0, 1.0, 0.0], [0.0, 3.0, 0.0], [0.0, 5.0, 0.0], [0.0, 7.0, 0.0]]])
    mask = torch.tensor([[1, 1, 0, 0],
                         [1, 1, 1, 1]])
    print(f"tokens shape: {tuple(tokens.shape)}  (texts, positions, vector size)")
    print(f"mask shape:   {tuple(mask.shape)}")

    naive = tokens.mean(dim=1)
    print(f"naive mean over positions (padding included!):\n{naive}")

    m = mask.unsqueeze(-1).float()                 # (2, 4) -> (2, 4, 1), broadcasts over the vector
    masked = (tokens * m).sum(dim=1) / m.sum(dim=1)
    print(f"masked mean (padding ignored):\n{masked}")

    section("6. Vectors as meaning: cosine similarity and top-k (preview of lesson 7)")
    # Hand-made 3-number "embeddings". Think of the numbers as [animal, flying, size].
    words = ["dog", "puppy", "cat", "airplane", "jet", "eagle"]
    E = torch.tensor([[0.9, 0.0, 0.5],
                      [0.9, 0.0, 0.2],
                      [0.8, 0.0, 0.3],
                      [0.0, 0.9, 0.9],
                      [0.0, 1.0, 0.8],
                      [0.7, 0.8, 0.3]])
    E = F.normalize(E, dim=1)                      # every row now has length 1
    print(f"row lengths after normalize: {E.norm(dim=1)}")

    sims = E @ E.T                                 # (6, 3) @ (3, 6) -> (6, 6)
    print(f"similarity of every word with every word, shape {tuple(sims.shape)}:")
    print("          " + "".join(f"{w:>9}" for w in words))
    for word, row_sims in zip(words, sims):
        print(f"{word:>10}" + "".join(f"{s:9.2f}" for s in row_sims.tolist()))

    query = F.normalize(torch.tensor([0.1, 0.9, 0.7]), dim=0)   # something "flying and big"
    scores = E @ query                             # (6, 3) @ (3,) -> (6,): one score per word
    top = torch.topk(scores, k=3)
    print(f"\nquery scores: {scores}")
    print("top 3:", [(words[i], round(s, 2)) for s, i in zip(top.values.tolist(), top.indices.tolist())])

    section("7. Devices")
    device = pick_device()
    E_dev, q_dev = E.to(device), query.to(device)
    scores_dev = E_dev @ q_dev
    print(f"computed on {scores_dev.device}: {scores_dev}")
    print(f"back to plain Python: {[round(s, 2) for s in scores_dev.cpu().tolist()]}")
    try:
        E_dev @ query                              # query is still on the CPU
    except RuntimeError as error:
        print(f"mixing devices fails: {str(error).splitlines()[0][:90]}")


if __name__ == "__main__":
    main()
