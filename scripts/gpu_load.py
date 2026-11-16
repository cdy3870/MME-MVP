"""Keep the GPU busy with real model inference, so `gpu capture` has something to measure.

Loads a Hugging Face causal language model onto the GPU and generates text in a loop for
--duration seconds, printing throughput and memory as it goes. Run it in one terminal and
`migration-eval gpu capture` in another.

Not a project dependency. Install into the venv first (cu126 wheels still support Pascal
cards such as the GTX 1080 Ti; newer CUDA builds may not):

    pip install torch --index-url https://download.pytorch.org/whl/cu126
    pip install transformers

Examples:

    python scripts/gpu_load.py                                   # gpt2-medium, 2 minutes
    python scripts/gpu_load.py --model gpt2-large --batch-size 8 --duration 600
"""

from __future__ import annotations

import argparse
import sys
import time

PROMPT = "The history of computing hardware began with"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="gpt2-medium", help="Hugging Face model id (default: gpt2-medium).")
    parser.add_argument("--batch-size", type=int, default=16, help="Prompts generated at once; raise for more load.")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Tokens generated per prompt per round.")
    parser.add_argument("--duration", type=float, default=120, help="Seconds to keep generating.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        sys.exit(f"Missing dependency ({exc.name}). See the install lines at the top of this script.")

    if not torch.cuda.is_available():
        sys.exit(
            "PyTorch can't see a CUDA GPU. Usually this means the CPU-only build is installed: "
            "pip uninstall torch, then reinstall with the --index-url line at the top of this script."
        )

    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    arch = f"sm_{major}{minor}"
    if arch not in torch.cuda.get_arch_list():
        print(
            f"Warning: this PyTorch build lists {torch.cuda.get_arch_list()} but the GPU is {arch}. "
            "If generation fails with 'no kernel image is available', install the cu126 build.",
            file=sys.stderr,
        )

    # Pascal (compute 6.x) runs fp16 far slower than fp32, so only use fp16 on Volta and newer.
    dtype = torch.float16 if major >= 7 else torch.float32
    print(f"GPU: {name} ({arch})")

    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # decoder-only models generate from the right edge
    model = AutoModelForCausalLM.from_pretrained(args.model).to("cuda", dtype=dtype).eval()
    print(f"Model loaded: {torch.cuda.memory_allocated() / 2**20:.0f} MiB on the GPU")

    inputs = tokenizer([PROMPT] * args.batch_size, return_tensors="pt", padding=True).to("cuda")
    prompt_len = inputs["input_ids"].shape[1]

    print(f"Generating for {args.duration:g}s (batch {args.batch_size}, {args.max_new_tokens} tokens). Ctrl-C stops.")
    end = time.monotonic() + args.duration
    rounds = 0
    total_tokens = 0
    started = time.monotonic()
    output = None
    try:
        with torch.inference_mode():
            while time.monotonic() < end:
                t0 = time.monotonic()
                output = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    min_new_tokens=args.max_new_tokens,  # fixed work per round
                    do_sample=True,
                    top_k=50,
                    pad_token_id=tokenizer.pad_token_id,
                )
                torch.cuda.synchronize()
                elapsed = time.monotonic() - t0
                tokens = (output.shape[1] - prompt_len) * args.batch_size
                rounds += 1
                total_tokens += tokens
                peak = torch.cuda.max_memory_allocated() / 2**20
                print(f"  round {rounds}: {tokens / elapsed:,.0f} tokens/s, peak memory {peak:,.0f} MiB")
    except KeyboardInterrupt:
        print("Stopped.")
    except torch.cuda.OutOfMemoryError:
        sys.exit("Out of GPU memory: lower --batch-size or --max-new-tokens, or use a smaller --model.")

    wall = time.monotonic() - started
    if rounds:
        print(f"Done: {rounds} rounds, {total_tokens:,} tokens in {wall:.0f}s ({total_tokens / wall:,.0f} tokens/s).")
        sample = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
        print(f"Sample: {PROMPT}{sample[:200]}...")


if __name__ == "__main__":
    main()
