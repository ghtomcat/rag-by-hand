# Glossary

[Overview](../README.md)

The terms used in the course, in alphabetical order. The lesson in brackets is where the
term is introduced.

**Attention mask** *(lesson 3)*: a list of 1s and 0s that tells the model which positions in
a batch are real tokens (1) and which are padding (0).

**Batch** *(lesson 3)*: several inputs processed together in one step. Faster on a GPU than
processing them one by one.

**Broadcasting** *(lesson 4)*: PyTorch's rule for combining tensors of different shapes, by
virtually repeating the smaller one, e.g. adding a row to every row of a matrix.

**Causal language model** *(lesson 8)*: a model that predicts the next token from the tokens
before it. Language models that generate text, such as Qwen or GPT, work this way.

**Chat template** *(lesson 8)*: the exact text format, with special tokens, that an
instruction-tuned model was trained on. Each model family has its own; the tokenizer
applies it with `apply_chat_template`.

**Chunk** *(lesson 2)*: a piece of a document, small enough to embed and to put into a
prompt. Retrieval returns chunks, not whole documents.

**Chunking** *(lesson 2)*: splitting documents into chunks. Structure-aware chunking cuts
along the document's own boundaries (headings, paragraphs, tables, code blocks).

**CLS pooling** *(lesson 5)*: using the output vector of the `[CLS]` token as the vector for
the whole text. Used by BERT-style models trained that way, such as BGE.

**Context window** *(lesson 9)*: the maximum number of tokens a model can read at once
(512 for our embedding model, 32k for Qwen2.5).

**Cosine similarity** *(lesson 4)*: a measure of how closely two vectors point in the same
direction, from −1 to 1. For vectors of length 1, it equals the dot product.

**Device** *(lesson 0)*: where a tensor lives and is computed: `cpu`, `cuda` (NVIDIA GPU) or
`mps` (Apple Silicon GPU).

**dtype** *(lesson 4)*: the number type of a tensor, e.g. `float32` (4 bytes) or `float16`
(2 bytes).

**Embedding** *(lesson 5)*: a vector of numbers that represents the meaning of a text. Texts
with similar meanings get vectors that point in similar directions.

**Fingerprint** *(lesson 6)*: a hash of everything an index depends on (sources and
settings), used to detect when the index is out of date.

**Greedy decoding** *(lesson 8)*: always choosing the most likely next token. Reproducible,
and a good default for factual answers.

**Hallucination** *(lesson 8)*: a fluent, confident answer that isn't true. Language models
produce them when they don't know the answer.

**Heading path** *(lesson 2)*: the chain of headings above a chunk, e.g.
`OpenSim Architecture > 2. The Mission > 2.2 Load lifecycle`. Gives a chunk its context.

**Hubness** *(lesson 7)*: some chunks, often overviews, are moderately similar to almost
every question and show up in many search results.

**Index** *(lesson 6)*: the stored chunks and their vectors, ready to search.

**Logits** *(lesson 8)*: a model's raw scores for every token in its vocabulary. Softmax turns
them into probabilities.

**Mean pooling** *(lesson 4)*: averaging all token vectors of a text, ignoring padding, to get
one vector for the whole text.

**Normalization** *(lesson 4)*: scaling a vector to length 1, so that the dot product equals
cosine similarity.

**Overlap** *(lesson 2)*: text repeated at the end of one chunk and the start of the next, so
information at the boundary isn't lost.

**Padding** *(lesson 3)*: filler tokens that make all texts in a batch the same length.

**Pooling** *(lesson 5)*: combining a text's token vectors into a single vector, e.g. by CLS
pooling or mean pooling.

**Prompt** *(lesson 8)*: the full text given to a language model, including the system
message, the context and the question.

**Query instruction** *(lesson 7)*: a fixed prefix some embedding models expect before search
queries (but not before documents).

**RAG** (retrieval-augmented generation) *(lesson 9)*: answering a question by first
retrieving relevant text, then letting a language model answer from that text.

**Retrieval** *(lesson 7)*: finding the chunks most relevant to a question.

**Shape** *(lesson 4)*: the size of each dimension of a tensor, e.g. `(119, 384)` for 119
vectors of 384 numbers.

**Softmax** *(lesson 8)*: a function that turns a list of scores into probabilities that add
up to 1.

**Special tokens** *(lesson 3)*: tokens with a fixed role rather than a meaning, e.g. `[CLS]`,
`[SEP]`, `[PAD]`, or `<|im_start|>` in chat templates.

**System prompt** *(lesson 8)*: the first message of a chat, with instructions the model
should follow throughout.

**Temperature** *(lesson 8)*: how random the choice of the next token is when sampling. 0
means greedy; higher values give more varied answers.

**Tensor** *(lesson 4)*: a multi-dimensional array of numbers of one type, stored on one
device.

**Token** *(lesson 3)*: the unit a model reads: a word, part of a word, or a symbol. Models
count length in tokens, not words or characters.

**Tokenizer** *(lesson 3)*: converts text into token IDs and back, using a fixed vocabulary
that belongs to one model.

**Top-k** *(lesson 4)*: the k highest values in a tensor and their positions; used to pick the
best search results.

**Truncation** *(lesson 3)*: cutting off input that is longer than a model's limit. Happens
silently, so the end of a long text is simply ignored.

**Vocabulary** *(lesson 3)*: the fixed set of tokens a tokenizer knows (30,522 for BGE,
151,665 for Qwen2.5).

**WordPiece** *(lesson 3)*: the tokenization method of BERT-style models. Unknown words are
split into known pieces, marked with `##`.
