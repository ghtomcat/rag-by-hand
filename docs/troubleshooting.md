# Troubleshooting

[Overview](../README.md)

## Setup

**`uv: command not found`**
uv isn't installed, or your shell hasn't picked it up yet. Install it following
[the uv documentation](https://docs.astral.sh/uv/getting-started/installation/), then open a
new terminal.

**PyTorch fails to install**
The project pins Python 3.12 in `.python-version`, the version the lessons were tested with.
Make sure you run commands with `uv run` (which uses that version), not with a system
`python`. A very new Python version may not have PyTorch packages yet.

**Lesson 0 shows `Device: cpu` on a Mac with Apple Silicon**
Check that you aren't running an x86 (Intel) build of Python through Rosetta:
`uv run python -c "import platform; print(platform.machine())"` should print `arm64`.

## Running the lessons

**`ModuleNotFoundError: No module named 'lesson01_load'` in the Python REPL**
The REPL looks for modules in the folder it was started from. Start it inside `lessons/`:
```bash
cd lessons
uv run python
```
Running the lesson scripts themselves (`uv run lessons/lesson02_chunk.py`) works from any
folder, because Python then searches the script's own folder.

**`Warning: You are sending unauthenticated requests to the HF Hub`**
Harmless. A free Hugging Face token (set as the environment variable `HF_TOKEN`) only
raises the download rate limit.

**`Token indices sequence length is longer than the specified maximum sequence length`**
The tokenizer warns that a text is longer than 512 tokens. In lesson 3 this is expected:
it's the problem the lesson investigates. Later lessons count the tokens of long texts
too, which triggers the same warning; nothing is truncated there.

**Model downloads are slow or interrupted**
The two models are about 3.3 GB together. If a download is interrupted, run the lesson
again: downloads resume. Models are cached in `~/.cache/huggingface/hub/`.

**Not enough disk space**
List the downloaded models and their sizes, then delete the one you don't need:
```bash
uv run hf cache ls
uv run hf cache rm model/Qwen/Qwen2.5-1.5B-Instruct
```
Deleting the model's folder by hand isn't enough: the large files are stored in a shared
`~/.cache/huggingface/hub/blobs/` folder, and the model folder only links to them. A deleted
model is downloaded again the next time it's needed.

## Hardware

**An operation fails on Apple Silicon (MPS) with "not implemented"**
Some PyTorch operations aren't available on MPS yet. Let PyTorch fall back to the CPU for
those:
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run lessons/lesson08_llm.py
```

**Out of memory in lessons 8–9**
The language model needs about 3 GB (on a GPU, in float16) or 6 GB (on the CPU, in
float32), plus room for the rest of your system. Close other applications, or switch to a
smaller model, e.g. `LLM_NAME = "Qwen/Qwen2.5-0.5B-Instruct"` in `lesson08_llm.py`. Answers
will get worse.

**Lessons 8–9 are very slow on the CPU**
Expect a few tokens per second instead of 10–20 on a GPU. Reduce `max_new_tokens` in
`Generator.generate`, or run fewer example questions.

## Results

**My numbers differ from the ones in the lessons**
Small differences in scores, timings and even generated text are normal across hardware,
operating systems and library versions. `uv.lock` pins the library versions used for the
lessons; `uv sync` installs exactly those.

**The index isn't rebuilt after I changed the chunking code**
The index fingerprint (lesson 6) covers the documents and the settings, but not the code.
Delete the `index/` folder to force a rebuild.
