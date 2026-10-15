"""Lesson 5: turn text into vectors with the embedding model, step by step."""

import time

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from lesson00_setup import pick_device
from lesson01_load import load_documents
from lesson02_chunk import Chunk
from lesson03_tokens import MODEL_NAME, chunk_documents_by_tokens


def mean_pool(token_vectors: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Average the token vectors of each text, ignoring padding (see lesson 4, section 5)."""
    m = attention_mask.unsqueeze(-1).float()
    return (token_vectors * m).sum(dim=1) / m.sum(dim=1)


def cls_pool(token_vectors: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Take the vector at position 0, the [CLS] token, as the vector for the whole text."""
    return token_vectors[:, 0]


POOLING = {"cls": cls_pool, "mean": mean_pool}


def embedding_text(chunk: Chunk) -> str:
    """The text we actually embed: the heading path gives the chunk its context."""
    return f"{chunk.heading_path}\n\n{chunk.text}"


class Embedder:
    """Wraps tokenizer + model: a list of texts in, one normalized vector per text out."""

    def __init__(self, model_name: str = MODEL_NAME, pooling: str = "cls", device: str | None = None):
        self.device = device or pick_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # eval() switches off training-only behaviour (dropout), so results are deterministic.
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.pool = POOLING[pooling]
        self.dim = self.model.config.hidden_size

    @torch.inference_mode()  # no gradient bookkeeping: we only run the model, never train it
    def embed(self, texts: list[str], batch_size: int = 16) -> torch.Tensor:
        vectors = []
        for start in range(0, len(texts), batch_size):
            batch = self.tokenizer(texts[start:start + batch_size], padding=True, truncation=True,
                                   max_length=512, return_tensors="pt").to(self.device)
            token_vectors = self.model(**batch).last_hidden_state   # (texts, positions, 384)
            pooled = self.pool(token_vectors, batch["attention_mask"])  # (texts, 384)
            vectors.append(F.normalize(pooled, dim=1).cpu())
        return torch.cat(vectors)                                   # (all texts, 384)


def print_similarities(labels: list[str], vectors: torch.Tensor):
    sims = vectors @ vectors.T
    print("     " + "".join(f"{label:>7}" for label in labels))
    for label, row in zip(labels, sims.tolist()):
        print(f"{label:>5}" + "".join(f"{s:7.2f}" for s in row))


def main():
    torch.set_printoptions(precision=3, sci_mode=False)

    print("=== 1. The model ===\n")
    embedder = Embedder()
    config = embedder.model.config
    n_params = sum(p.numel() for p in embedder.model.parameters())
    print(f"{MODEL_NAME} on {embedder.device}")
    print(f"{n_params / 1e6:.1f}M parameters, {config.num_hidden_layers} layers, "
          f"vectors of {config.hidden_size} numbers")

    print("\n=== 2. One sentence, step by step ===\n")
    sentence = "The mission file decides which aircraft loads."
    batch = embedder.tokenizer([sentence], return_tensors="pt").to(embedder.device)
    print(f"tokens:              {embedder.tokenizer.convert_ids_to_tokens(batch['input_ids'][0])}")
    with torch.inference_mode():
        token_vectors = embedder.model(**batch).last_hidden_state
    print(f"model output shape:  {tuple(token_vectors.shape)}  (texts, tokens, numbers per token)")
    cls = cls_pool(token_vectors, batch["attention_mask"])[0]
    print(f"[CLS] vector:        shape {tuple(cls.shape)}, first 6 numbers {cls[:6].cpu()}")
    print(f"length before/after normalize: {cls.norm():.2f} -> {F.normalize(cls, dim=0).norm():.2f}")

    print("\n=== 3. Similar meaning -> similar vectors ===\n")
    sentences = {
        "A": "How is the mission file loaded?",
        "B": "loadMission fetches the mission JSON, then the aircraft JSON.",
        "C": "The mission picker shows one card per mission.",
        "D": "Rockets switch between pad hold, ascent and orbit.",
        "E": "My grandmother bakes bread on Sundays.",
    }
    for label, text in sentences.items():
        print(f"  {label}: {text}")
    for pooling in ["cls", "mean"]:
        embedder.pool = POOLING[pooling]
        print(f"\n{pooling} pooling:")
        print_similarities(list(sentences), embedder.embed(list(sentences.values())))
    embedder.pool = cls_pool

    print("\n=== 4. All chunks ===\n")
    chunks = chunk_documents_by_tokens(load_documents(), embedder.tokenizer)
    texts = [embedding_text(chunk) for chunk in chunks]
    lengths = [len(embedder.tokenizer(text)["input_ids"]) for text in texts]
    print(f"{len(texts)} chunks with heading path: max {max(lengths)} tokens, "
          f"{sum(1 for n in lengths if n > 512)} over 512 (would be truncated)")

    start = time.perf_counter()
    E = embedder.embed(texts)
    seconds = time.perf_counter() - start
    print(f"embedded in {seconds:.1f} s -> tensor of shape {tuple(E.shape)}, "
          f"{E.numel() * E.element_size() / 1024:.0f} KB")
    print(f"first chunk: {chunks[0].heading_path!r}\n  vector starts {E[0, :6]}")


if __name__ == "__main__":
    main()
