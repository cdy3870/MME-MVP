"""Replay DeepSeek-style serving traffic on the real GPU, so `gpu capture` measures something
that looks like production inference rather than a fixed generate() loop.

What it does, scaled down to GPT-2 on one GPU:

  Workload    Poisson arrivals with a chat / agent / doc mix. Each request has a context
              length, a cached prefix and an output length, drawn like make_workload in the
              "Serve Like DeepSeek" simulator, scaled so context + output fits in 1024 tokens.
  Prefix      Each class shares a prefix (think system prompt or repeated document). Its KV is
  cache       computed once; a request copies the first `cached` tokens of it and only
              prefills the rest, like DeepSeek's context caching.
  Prefill /   Prefill and decode are separate phases with their own queue, as in a
  decode      disaggregated deployment, interleaved on one GPU: each loop admits up to
              --prefill-per-step new requests, then runs one decode step for every active one.
  Continuous  The KV cache is a fixed pool of slots (--slots). A request takes a slot at
  batching    prefill and frees it when done, so new requests join the running decode batch
              immediately instead of waiting for a whole batch to finish.

The forward pass is written out by hand on top of the Hugging Face GPT-2 weights, so the
slot cache doesn't depend on transformers' cache API. GPT-2 family models only.

Not a project dependency. Needs torch (1.12+ works, with a CUDA build), transformers and numpy;
it runs on Python 3.7+, so an existing ML env works.

    python scripts/gpu_serve_sim.py                         # 400 requests at 4 req/s
    python scripts/gpu_serve_sim.py --rate 10 --n 1000      # push it harder
    python scripts/gpu_serve_sim.py --mix 0.2 0.7 0.1       # mostly agent traffic
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

# class -> (median context, lognormal sigma, max context, mean output, cache ratio)
# Same shape as the simulator's CLASS_SPEC, scaled to GPT-2's 1024-token window.
SCALED_SPEC = {
    "chat": (160, 0.6, 512, 48, 0.3),
    "agent": (480, 0.4, 768, 24, 0.8),
    "doc": (720, 0.3, 896, 96, 0.1),
}
BLOCK = 16  # cached prefixes are whole blocks, like paged KV (the simulator uses 128)


@dataclass
class Request:
    id: int
    cls: str
    t_arrive: float
    ctx: int  # prompt tokens
    cached: int  # of which served from the prefix cache
    out: int  # tokens to generate
    # Filled in while serving (seconds since the run started).
    t_prefill_start: float | None = None
    t_first: float | None = None
    t_done: float | None = None
    slot: int = -1
    pos: int = 0  # next KV position to write
    generated: int = 0
    last_token: int = 0


# --- workload --------------------------------------------------------------------------------


def make_workload(n: int, seed: int, rate: float, mix: tuple[float, ...], n_positions: int) -> list[Request]:
    """Draw n requests: gaps, classes, then per request ctx / cached / out, in that order."""
    rng = np.random.default_rng(seed)
    t_arrive = np.cumsum(rng.exponential(1 / rate, size=n))
    classes = rng.choice(list(SCALED_SPEC), size=n, p=list(mix))

    requests = []
    for i in range(n):
        cls = str(classes[i])
        median, sigma, cap, mean_out, ratio = SCALED_SPEC[cls]
        ctx = int(min(cap, max(BLOCK, rng.lognormal(math.log(median), sigma))))
        cached = int(ctx * ratio * rng.uniform(0.5, 1.0)) // BLOCK * BLOCK
        out = int(max(1, rng.poisson(mean_out)))
        ctx = min(ctx, n_positions - out)  # prompt + output must fit the position table
        cached = min(cached, (ctx - 1) // BLOCK * BLOCK)  # always prefill at least one token
        requests.append(Request(i, cls, float(t_arrive[i]), ctx, cached, out))
    return requests


# --- engine ----------------------------------------------------------------------------------


class SlotEngine:
    """GPT-2 forward pass over a fixed pool of KV slots.

    k/v have shape (layers, slots, heads, max_len, head_dim). A row of a batch names its slot
    and the absolute positions of its tokens; its keys/values are written at those positions,
    and it attends to keys at positions <= its own. That one mask is both the causal mask and
    the "ignore whatever is left in this slot past my length" mask.
    """

    def __init__(self, model, slots: int, device, dtype):
        import torch

        self.torch = torch
        self.tr = model.transformer
        self.lm_head = model.lm_head
        cfg = model.config
        self.n_layer, self.n_head, self.n_embd = cfg.n_layer, cfg.n_head, cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self.max_len = cfg.n_positions
        self.device = device
        shape = (self.n_layer, slots, self.n_head, self.max_len, self.head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)

    @property
    def cache_bytes(self) -> int:
        return 2 * self.k.numel() * self.k.element_size()

    def forward(self, tokens, positions, slots):
        """tokens, positions: (B, T) long. slots: (B,) long. Returns last-token logits (B, vocab)."""
        torch = self.torch
        F = torch.nn.functional
        B, T = tokens.shape
        H, D, C = self.n_head, self.head_dim, self.n_embd

        x = self.tr.wte(tokens) + self.tr.wpe(positions)
        span = int(positions.max()) + 1
        key_pos = torch.arange(span, device=self.device)
        mask = key_pos.view(1, 1, 1, span) <= positions.view(B, 1, T, 1)  # (B, 1, T, span)
        write_slots = slots.view(B, 1)

        for layer, block in enumerate(self.tr.h):
            q, k, v = block.attn.c_attn(block.ln_1(x)).split(C, dim=2)
            q = q.view(B, T, H, D).transpose(1, 2)
            # Index (slot, :, position): result dims are (B, T, H, D).
            self.k[layer][write_slots, :, positions] = k.view(B, T, H, D)
            self.v[layer][write_slots, :, positions] = v.view(B, T, H, D)
            keys = self.k[layer][slots, :, :span]  # (B, H, span, D)
            values = self.v[layer][slots, :, :span]
            if hasattr(F, "scaled_dot_product_attention"):  # torch >= 2.0
                attn = F.scaled_dot_product_attention(q, keys, values, attn_mask=mask)
            else:  # torch 1.x
                scores = (q @ keys.transpose(-1, -2)) / math.sqrt(D)
                attn = scores.masked_fill(~mask, float("-inf")).softmax(-1) @ values
            x = x + block.attn.c_proj(attn.transpose(1, 2).reshape(B, T, C))
            x = x + block.mlp(block.ln_2(x))

        return self.lm_head(self.tr.ln_f(x[:, -1]))

    def prefill(self, slot: int, tokens: list[int], start: int):
        """Run `tokens` at positions start.. in `slot`; return the logits after the last one."""
        torch = self.torch
        t = torch.tensor([tokens], device=self.device)
        pos = torch.arange(start, start + len(tokens), device=self.device).view(1, -1)
        return self.forward(t, pos, torch.tensor([slot], device=self.device))

    def decode(self, slots: list[int], tokens: list[int], positions: list[int]):
        """One token for each of several slots, each at its own position."""
        torch = self.torch
        dev = self.device
        return self.forward(
            torch.tensor(tokens, device=dev).view(-1, 1),
            torch.tensor(positions, device=dev).view(-1, 1),
            torch.tensor(slots, device=dev),
        )

    def save_prefix(self, slot: int, length: int):
        """Copy the first `length` positions of a slot out, as a reusable prefix."""
        return (self.k[:, slot, :, :length].clone(), self.v[:, slot, :, :length].clone())

    def load_prefix(self, slot: int, prefix, length: int) -> None:
        k, v = prefix
        self.k[:, slot, :, :length] = k[:, :, :length]
        self.v[:, slot, :, :length] = v[:, :, :length]


# --- serving loop ----------------------------------------------------------------------------


def serve(engine: SlotEngine, requests: list[Request], slots: int, prefill_per_step: int, vocab: int, seed: int):
    """Replay `requests` in real time. Returns (finished requests, stats dict)."""
    torch = engine.torch
    gen = torch.Generator().manual_seed(seed)

    # One shared prefix per class, long enough for the largest cached span of that class.
    prefix_len = {cls: max([r.cached for r in requests if r.cls == cls], default=0) for cls in SCALED_SPEC}
    prefix_tokens: dict[str, list[int]] = {}
    prefix_kv = {}
    for cls, length in prefix_len.items():
        if length == 0:
            continue
        prefix_tokens[cls] = torch.randint(vocab, (length,), generator=gen).tolist()
        engine.prefill(0, prefix_tokens[cls], 0)
        prefix_kv[cls] = engine.save_prefix(0, length)

    pending = deque(sorted(requests, key=lambda r: r.t_arrive))
    prefill_queue: deque[Request] = deque()
    free_slots = list(range(slots))[::-1]
    active: list[Request] = []
    done: list[Request] = []
    stats = {"decode_steps": 0, "decode_rows": 0, "prefill_tokens": 0, "cached_tokens": 0, "decode_time": 0.0, "prefill_time": 0.0}

    start = time.perf_counter()
    last_report = 0.0
    tokens_at_report = 0

    def now() -> float:
        return time.perf_counter() - start

    def finish(req: Request) -> None:
        req.t_done = now()
        free_slots.append(req.slot)
        done.append(req)

    while pending or prefill_queue or active:
        t = now()
        while pending and pending[0].t_arrive <= t:
            prefill_queue.append(pending.popleft())
        if not prefill_queue and not active:
            time.sleep(max(0.0, pending[0].t_arrive - t))
            continue

        # Prefill phase: admit new requests while there are free slots.
        admitted = 0
        while prefill_queue and free_slots and admitted < prefill_per_step:
            req = prefill_queue.popleft()
            req.slot = free_slots.pop()
            req.t_prefill_start = now()
            if req.cached:
                engine.load_prefix(req.slot, prefix_kv[req.cls], req.cached)
            fresh = torch.randint(vocab, (req.ctx - req.cached,), generator=gen).tolist()
            t0 = time.perf_counter()
            logits = engine.prefill(req.slot, fresh, req.cached)
            req.last_token = int(logits.argmax(-1))  # .item-style read: waits for the GPU
            stats["prefill_time"] += time.perf_counter() - t0
            stats["prefill_tokens"] += len(fresh)
            stats["cached_tokens"] += req.cached
            req.t_first = now()
            req.pos = req.ctx
            req.generated = 1
            admitted += 1
            if req.generated >= req.out:
                finish(req)
            else:
                active.append(req)

        # Decode phase: one token for every active request.
        if active:
            t0 = time.perf_counter()
            logits = engine.decode([r.slot for r in active], [r.last_token for r in active], [r.pos for r in active])
            next_tokens = logits.argmax(-1).tolist()
            stats["decode_time"] += time.perf_counter() - t0
            stats["decode_steps"] += 1
            stats["decode_rows"] += len(active)
            still_active = []
            for req, token in zip(active, next_tokens):
                req.last_token = token
                req.pos += 1
                req.generated += 1
                if req.generated >= req.out:
                    finish(req)
                else:
                    still_active.append(req)
            active = still_active

        t = now()
        if t - last_report >= 5:
            out_tokens = sum(r.generated for r in done) + sum(r.generated for r in active)
            rate = (out_tokens - tokens_at_report) / (t - last_report)
            print(
                f"  t={t:6.1f}s  arrived {len(requests) - len(pending):4d}  queued {len(prefill_queue):3d}  "
                f"active {len(active):3d}/{slots}  done {len(done):4d}  {rate:7.0f} out tok/s"
            )
            last_report, tokens_at_report = t, out_tokens

    stats["wall"] = now()
    return done, stats


# --- report ----------------------------------------------------------------------------------


def _pct(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


def report(done: list[Request], stats: dict, slots: int) -> None:
    print()
    print(f"{'class':6} {'n':>5} {'TTFT p50':>9} {'TTFT p95':>9} {'TPOT p50':>9} {'TPOT p95':>9} {'queue p95':>10}")
    for cls in [*SCALED_SPEC, "all"]:
        rows = [r for r in done if cls in ("all", r.cls)]
        if not rows:
            continue
        ttft = [(r.t_first - r.t_arrive) * 1000 for r in rows]
        tpot = [(r.t_done - r.t_first) / (r.out - 1) * 1000 for r in rows if r.out > 1]
        queue = [(r.t_prefill_start - r.t_arrive) * 1000 for r in rows]
        print(
            f"{cls:6} {len(rows):5d} {_pct(ttft, 50):7.0f}ms {_pct(ttft, 95):7.0f}ms "
            f"{_pct(tpot, 50):7.1f}ms {_pct(tpot, 95):7.1f}ms {_pct(queue, 95):8.0f}ms"
        )

    wall = stats["wall"]
    out_tokens = sum(r.out for r in done)
    prompt_tokens = stats["prefill_tokens"] + stats["cached_tokens"]
    steps = max(stats["decode_steps"], 1)
    print()
    print(f"Wall time:          {wall:.1f}s for {len(done)} requests ({len(done) / wall:.2f} req/s)")
    print(f"Output tokens:      {out_tokens:,} ({out_tokens / wall:,.0f} tok/s)")
    print(
        f"Prompt tokens:      {prompt_tokens:,}, of which {stats['cached_tokens']:,} from the prefix cache "
        f"({stats['cached_tokens'] / max(prompt_tokens, 1):.0%} prefill saved)"
    )
    print(f"Decode batch:       mean {stats['decode_rows'] / steps:.1f} of {slots} slots over {stats['decode_steps']:,} steps")
    print(f"Mean decode step:   {stats['decode_time'] / steps * 1000:.1f}ms")
    print(
        f"GPU time split:     prefill {stats['prefill_time']:.1f}s, decode {stats['decode_time']:.1f}s, "
        f"idle/overhead {max(wall - stats['prefill_time'] - stats['decode_time'], 0):.1f}s"
    )


# --- main ------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="gpt2-medium", help="GPT-2 family model id (default: gpt2-medium).")
    parser.add_argument("--n", type=int, default=400, help="Requests to replay.")
    parser.add_argument("--rate", type=float, default=4.0, help="Mean arrivals per second.")
    parser.add_argument("--mix", type=float, nargs=3, default=(0.7, 0.2, 0.1), metavar=("CHAT", "AGENT", "DOC"))
    parser.add_argument("--slots", type=int, default=16, help="KV slots = max requests decoding at once.")
    parser.add_argument("--prefill-per-step", type=int, default=1, help="New requests admitted per loop.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if abs(sum(args.mix) - 1) > 1e-6:
        sys.exit("--mix must sum to 1.")

    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        sys.exit(f"Missing dependency ({exc.name}). See the install lines at the top of this script.")
    if not torch.cuda.is_available():
        sys.exit("PyTorch can't see a CUDA GPU (CPU-only build?). Reinstall torch with the cu126 index URL.")

    major, _ = torch.cuda.get_device_capability(0)
    dtype = torch.float16 if major >= 7 else torch.float32  # Pascal is slow at fp16
    model = AutoModelForCausalLM.from_pretrained(args.model).to("cuda", dtype=dtype).eval()
    if not hasattr(model, "transformer") or not hasattr(model.transformer, "wpe"):
        sys.exit(f"{args.model} is not a GPT-2 family model; this script's forward pass is GPT-2 specific.")

    requests = make_workload(args.n, args.seed, args.rate, tuple(args.mix), model.config.n_positions)
    counts = {cls: sum(r.cls == cls for r in requests) for cls in SCALED_SPEC}
    print(f"GPU: {torch.cuda.get_device_name(0)}, model {args.model} ({str(dtype).replace('torch.', '')})")
    print(
        f"Workload: {args.n} requests over ~{requests[-1].t_arrive:.0f}s at {args.rate:g} req/s, "
        + ", ".join(f"{c} {n}" for c, n in counts.items())
    )

    with torch.inference_mode():
        engine = SlotEngine(model, args.slots, "cuda", dtype)
        print(f"KV slots: {args.slots} x {engine.max_len} tokens = {engine.cache_bytes / 2**30:.2f} GiB")
        print("Serving. Run `migration-eval gpu capture` in another terminal now. Ctrl-C stops.")
        try:
            done, stats = serve(engine, requests, args.slots, args.prefill_per_step, model.config.vocab_size, args.seed)
        except KeyboardInterrupt:
            sys.exit("Stopped.")
    report(done, stats, args.slots)


if __name__ == "__main__":
    main()
