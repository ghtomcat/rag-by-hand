# Going further

[Overview](../README.md)

This course writes every step by hand, so you can see how each part works. Real projects
usually use libraries for most of these steps. This page shows which libraries do what you
built, and which techniques go beyond the course.

## Libraries for what you built by hand

| Step | In this course | Libraries |
|---|---|---|
| Loading | `pathlib` + `open()` (lesson 1) | [Unstructured](https://github.com/Unstructured-IO/unstructured), [Docling](https://github.com/docling-project/docling): extract text **and structure** from PDF, HTML, Word, slides |
| Chunking | structure-aware chunker (lessons 2–3) | LangChain text splitters (`MarkdownHeaderTextSplitter`, `RecursiveCharacterTextSplitter`), LlamaIndex node parsers, [Chonkie](https://github.com/chonkie-inc/chonkie), Docling's `HybridChunker` |
| Embedding | `AutoModel` + pooling (lesson 5) | [sentence-transformers](https://www.sbert.net/): `SentenceTransformer(name).encode(texts)` does tokenizing, pooling and normalizing in one call, using the pooling the model was trained with |
| Index and search | one tensor + `topk` (lessons 6–7) | [FAISS](https://github.com/facebookresearch/faiss) for fast (approximate) search over millions of vectors; vector databases such as Chroma, Qdrant, Weaviate or pgvector for storage, filtering and updates |
| Generation | `AutoModelForCausalLM` (lesson 8) | [Ollama](https://ollama.com/) or [llama.cpp](https://github.com/ggml-org/llama.cpp) for fast local models (quantized, so they need less memory); hosted APIs for large models |
| The whole pipeline | `RAG` class (lesson 9) | [LangChain](https://www.langchain.com/), [LlamaIndex](https://www.llamaindex.ai/), [Haystack](https://haystack.deepset.ai/) |
| Evaluation | lesson 10 *(coming soon)* | [Ragas](https://docs.ragas.io/), [DeepEval](https://github.com/confident-ai/deepeval) |

When something goes wrong inside a library, you now know what it's doing underneath: what a
chunker decides, why a chunk gets truncated, what pooling means, and why a retrieved chunk
can be "similar" and still wrong.

## Techniques beyond this course

**Hybrid search.** Combine embedding search with keyword search (BM25, e.g. with
`rank_bm25`). Embeddings find meaning; keywords find exact terms such as identifiers, names
or error codes, which embeddings handle poorly (lesson 3 shows why).

**Reranking.** Retrieve more candidates (e.g. 20), then re-score each one with a
cross-encoder that reads the question and the chunk *together*. More accurate than
comparing two separate vectors, but too slow to run on every chunk. *(Lesson 11, coming
soon.)*

**Contextual retrieval.** Before embedding, let an LLM write one or two sentences per chunk
that explain where it fits in the whole document, and embed those together with the chunk.
It goes further than the heading path from lesson 5, and it helps most for documents
without structure, like the plain-text sample.

**Parent–child chunks ("small-to-big").** Search over small, precise chunks, but give the
LLM the larger section each one belongs to.

**Query rewriting.** Let an LLM rephrase the question before searching, or generate a
hypothetical answer and search with that (HyDE).

**Larger or better models.** Bigger embedding models (e.g. `bge-base`, `bge-large`,
multilingual models for non-English text) and bigger LLMs improve both retrieval and
answers. The [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) compares
embedding models.

**Fine-tuning.** Train the embedding model on pairs of questions and passages from your own
domain. Worth it when general models don't understand your terminology.
