"""Lesson 0: check that Python, PyTorch and Hugging Face are installed and find the best device."""

import sys

import torch
import transformers


def pick_device() -> str:
    # Prefer the Apple GPU ("mps"), then an NVIDIA GPU ("cuda"), else the CPU.
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    print(f"Python:       {sys.version.split()[0]}")
    print(f"PyTorch:      {torch.__version__}")
    print(f"Transformers: {transformers.__version__}")

    device = pick_device()
    print(f"Device:       {device}")

    # A tiny computation on that device to prove it really works.
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
    b = torch.tensor([[1.0], [1.0]], device=device)
    result = a @ b  # matrix multiplication, shape [2, 2] @ [2, 1] -> [2, 1]
    print(f"Test matmul:  {result.tolist()} (computed on {result.device})")


# Only run main() when this file is executed directly, not when it's imported.
if __name__ == "__main__":
    main()
