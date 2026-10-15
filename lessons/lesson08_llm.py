"""Lesson 8: a local LLM on its own, without retrieval."""

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lesson00_setup import pick_device
from lesson07_search import EXAMPLES

# A small instruction-tuned model: ~1.5B parameters, about 3 GB download.
LLM_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

SYSTEM_PROMPT = "You are a helpful assistant. Answer concisely."


class Generator:
    """Wraps tokenizer + causal language model: chat messages in, generated answer out."""

    def __init__(self, model_name: str = LLM_NAME, device: str | None = None):
        self.device = device or pick_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # float16 halves memory compared to float32 (2 bytes per parameter instead of 4).
        # On the CPU, float16 is slow or unsupported for some operations, so use float32 there.
        dtype = torch.float32 if self.device == "cpu" else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        self.model.to(self.device).eval()

    def chat_inputs(self, messages: list[dict]):
        """Turn chat messages into model input, using the model's own chat template."""
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(self.device)

    @torch.inference_mode()
    def generate(self, messages: list[dict], max_new_tokens: int = 200) -> tuple[str, int]:
        """Return (answer, number of generated tokens). Greedy decoding: always the most likely token."""
        inputs = self.chat_inputs(messages)
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                     temperature=None, top_p=None, top_k=None)
        new_tokens = output[0, inputs["input_ids"].shape[1]:]   # drop the prompt, keep the answer
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip(), len(new_tokens)


def ask(generator: Generator, question: str, system: str = SYSTEM_PROMPT) -> tuple[str, int]:
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": question}]
    return generator.generate(messages)


@torch.inference_mode()
def show_next_token_steps(generator: Generator, text: str, steps: int = 6):
    """Generate a few tokens by hand: predict, pick the most likely, append, repeat."""
    ids = generator.tokenizer(text, return_tensors="pt")["input_ids"].to(generator.device)
    print(f"start: {text!r}")
    for _ in range(steps):
        logits = generator.model(input_ids=ids).logits[0, -1]       # scores for every vocab entry
        probs = torch.softmax(logits.float(), dim=0)                # scores -> probabilities
        top = torch.topk(probs, k=3)
        options = [(generator.tokenizer.decode([i]), p) for p, i in zip(top.values.tolist(), top.indices.tolist())]
        print("  " + "   ".join(f"{tok!r} {p:.0%}" for tok, p in options))
        ids = torch.cat([ids, top.indices[:1].unsqueeze(0)], dim=1)  # append the winner
    print(f"result: {generator.tokenizer.decode(ids[0])!r}")


def main():
    print("=== 1. The model ===\n")
    start = time.perf_counter()
    generator = Generator()
    n_params = sum(p.numel() for p in generator.model.parameters())
    print(f"{LLM_NAME} on {generator.device}, loaded in {time.perf_counter() - start:.0f} s")
    print(f"{n_params / 1e9:.2f}B parameters, {generator.model.dtype}, "
          f"~{n_params * generator.model.dtype.itemsize / 1e9:.1f} GB in memory, vocabulary of {len(generator.tokenizer):,} tokens")

    print("\n=== 2. Next-token prediction, by hand ===\n")
    show_next_token_steps(generator, "The capital of Switzerland is")

    print("\n=== 3. The chat template ===\n")
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "What does an altimeter measure?"}]
    print(generator.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False))

    print("=== 4. A question it can answer from training ===\n")
    answer, _ = ask(generator, "What does an altimeter measure?")
    print(f"A: {answer}\n")

    print("=== 5. Questions about OpenSim, without any context ===")
    for question, _ in EXAMPLES:
        start = time.perf_counter()
        answer, n_tokens = ask(generator, question)
        seconds = time.perf_counter() - start
        print(f"\nQ: {question}\nA: {answer}")
        print(f"   ({n_tokens} tokens in {seconds:.1f} s, {n_tokens / seconds:.0f} tokens/s)")


if __name__ == "__main__":
    main()
