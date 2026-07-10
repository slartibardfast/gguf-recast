#!/usr/bin/env python3
"""Build a synthetic BF16 GGUF fixture for the recast-bf16-fp16 differential.

Covers the tool's routing space: BF16 tensors in absmax bands A/B/C, a zero
tensor, a policy-preserved tensor, an F32 tensor, an F16 tensor, a Q8_0
tensor (raw passthrough), and a KV section with strings, scalars, and
arrays (string/int/float/bool).

Run with the reference venv and its gguf-py on PYTHONPATH:
    PYTHONPATH=<llama.cpp>/gguf-py <venv>/bin/python scripts/gen_recast_fixture.py <out.gguf>
"""
import sys

import numpy as np
from gguf import GGMLQuantizationType, GGUFWriter
import gguf.quants


def bf16(a: np.ndarray) -> np.ndarray:
    """f32 -> BF16 raw uint8 bytes (truncation is fine for fixture data)."""
    u = a.astype(np.float32).view(np.uint32)
    return (u >> 16).astype(np.uint16).view(np.uint8).reshape(*a.shape[:-1], a.shape[-1] * 2)


def main(out: str) -> None:
    rng = np.random.default_rng(42)
    w = GGUFWriter(out, arch="llama", use_temp_file=False)
    w.add_string("fixture.note", "recast differential fixture")
    w.add_uint32("fixture.u32", 12345)
    w.add_float32("fixture.f32", 0.25)
    w.add_bool("fixture.bool", True)
    w.add_array("fixture.strings", ["alpha", "beta"])
    w.add_array("fixture.ints", [1, 2, 3])
    w.add_array("fixture.floats", [0.5, 1.5])
    w.add_uint64("fixture.u64", 1 << 40)

    band_a = (rng.standard_normal((8, 16)) * 100.0).astype(np.float32)
    band_b = band_a.copy()
    band_b[0, 0] = 50000.0
    band_c = band_a.copy()
    band_c[0, 0] = 100000.0
    zero = np.zeros((4, 8), dtype=np.float32)
    preserve = (rng.standard_normal((4, 16)) * 10.0).astype(np.float32)
    vec = (rng.standard_normal(32) * 1000.0).astype(np.float32)  # 1D band A

    w.add_tensor("blk.0.band_a.weight", bf16(band_a), raw_dtype=GGMLQuantizationType.BF16)
    w.add_tensor("blk.0.band_b.weight", bf16(band_b), raw_dtype=GGMLQuantizationType.BF16)
    w.add_tensor("blk.0.band_c.weight", bf16(band_c), raw_dtype=GGMLQuantizationType.BF16)
    w.add_tensor("blk.0.zero.weight", bf16(zero), raw_dtype=GGMLQuantizationType.BF16)
    w.add_tensor("blk.0.keepme.weight", bf16(preserve), raw_dtype=GGMLQuantizationType.BF16)
    w.add_tensor("blk.0.vec.bias", bf16(vec.reshape(1, -1)).reshape(-1), raw_dtype=GGMLQuantizationType.BF16,
                 raw_shape=[32])
    w.add_tensor("blk.0.norm.weight", np.linspace(-1, 1, 24, dtype=np.float32))
    w.add_tensor("blk.0.half.weight", (rng.standard_normal((8, 8)) * 3.0).astype(np.float16))
    q8 = gguf.quants.quantize((rng.standard_normal((4, 64)) * 5.0).astype(np.float32),
                              GGMLQuantizationType.Q8_0)
    w.add_tensor("blk.0.q8.weight", q8, raw_dtype=GGMLQuantizationType.Q8_0)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1])
