"""GPU capture (GPU-1): read a partner's GPUs with `nvidia-smi` and write `gpu_profile.json`.

Modules:

    nvidia_smi  find and run the binary (the only module that touches a subprocess)
    snapshot    the one-time snapshot: `nvidia-smi -q -x` (XML) and `nvidia-smi topo -m`
    samples     the timed samples: `nvidia-smi --query-gpu=... --format=csv`
    profile     the GpuProfile model and reading/writing `gpu_profile.json`

The parsers take text, not a process, so `gpu import` (GPU-3) can reuse them on files the
partner ran by hand.
"""
