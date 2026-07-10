#!/usr/bin/env python3
"""Generate the numeric-semantics fixtures gguf-recast is verified against.

Two operations in the remix pipeline depend on the exact float semantics of the
Python stack that produced the accepted reference GGUF
(/opt/models/Qwen3.6-27B-Q4_0AR16-b9222.gguf, PyTorch 2.13.0+cpu with Intel MKL,
NumPy from the same venv):

1. `bf16_to_f16.bin` (65536 x u16, little-endian): numpy's
   float32 -> float16 cast (`astype(np.float16)`) applied to every BF16 bit
   pattern upcast to f32. Every F16 tensor the remix emits is cast from
   BF16-derived f32 values, so this table is the full input domain.
   The Rust implementation computes the cast itself (IEEE 754 round-to-nearest-
   even with overflow to inf); the table binds the exhaustive equivalence test.

2. `torch_exp_bf16.bin` (65536 x u32, little-endian): torch.exp(float32) over
   the same BF16 domain. PyTorch CPU routes exp through Intel MKL vmsExp
   (VML_HA), which is closed source and deviates from correctly-rounded exp on
   1794 of the 65536 inputs (and quiets NaN payloads differently). The remix's
   only transcendental is `-exp(A_log)` where A_log is stored BF16, so the
   domain table IS the semantic, embedded in the binary. No published algorithm
   reproduces MKL bit-for-bit; the table keeps the tool deterministic and
   self-contained.

Run with the venv that produced the reference:
    <venv>/bin/python scripts/gen_numeric_fixtures.py fixtures/
"""
import sys

import numpy as np
import torch


def main(outdir: str) -> None:
    bits = np.arange(65536, dtype=np.uint32)
    f32 = (bits << 16).view(np.float32).copy()

    f16 = f32.astype(np.float16)
    with open(f"{outdir}/bf16_to_f16.bin", "wb") as f:
        f.write(f16.view(np.uint16).astype("<u2").tobytes())

    ex = torch.exp(torch.from_numpy(f32)).numpy()
    with open(f"{outdir}/torch_exp_bf16.bin", "wb") as f:
        f.write(ex.view(np.uint32).astype("<u4").tobytes())

    print(f"numpy {np.__version__}, torch {torch.__version__}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "fixtures")
