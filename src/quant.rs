//! AutoRound INT4 (AutoGPTQ packing) -> Q4_0 / Q4_0_AR16 lossless repack
//! primitives. Codes and scales copy verbatim; no rounding anywhere.
//!
//! Format traps (bound by tests):
//! - Q4_0 block: fp16 d + 16 bytes; qs[j] low nibble = code[j], high = code[j+16]
//!   (split-halves layout), 18 bytes per 32 input elements.
//! - Q4_0_AR16 block: fp16 d + 8 bytes; INTERLEAVED layout
//!   qs[j] = code[2j] | code[2j+1] << 4, 10 bytes per 16 elements.
//! - AutoGPTQ int32 packing: word (r, c) holds codes for input rows 8r..8r+8
//!   of output channel c; code k in bits [4k, 4k+4).

/// Unpack AutoGPTQ INT4: `qweight` is `[in/8, out]` u32 words (LE already
/// decoded). Returns codes `[in, out]` row-major, values in 0..=15.
pub fn unpack_autogptq_int4(qweight: &[u32], in_packed_rows: usize, out_f: usize) -> Vec<u8> {
    assert_eq!(qweight.len(), in_packed_rows * out_f);
    let in_f = in_packed_rows * 8;
    let mut codes = vec![0u8; in_f * out_f];
    for r in 0..in_packed_rows {
        let row = &qweight[r * out_f..(r + 1) * out_f];
        for k in 0..8 {
            let dst = &mut codes[(8 * r + k) * out_f..(8 * r + k + 1) * out_f];
            let shift = 4 * k as u32;
            for (d, w) in dst.iter_mut().zip(row.iter()) {
                *d = ((w >> shift) & 0xf) as u8;
            }
        }
    }
    codes
}

/// Blocked transpose of a `[rows, cols]` u8 matrix into `[cols, rows]`.
pub fn transpose_u8(src: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    assert_eq!(src.len(), rows * cols);
    let mut dst = vec![0u8; rows * cols];
    const B: usize = 64;
    for r0 in (0..rows).step_by(B) {
        for c0 in (0..cols).step_by(B) {
            for r in r0..(r0 + B).min(rows) {
                let s = &src[r * cols..];
                for c in c0..(c0 + B).min(cols) {
                    dst[c * rows + r] = s[c];
                }
            }
        }
    }
    dst
}

/// Pack Q4_0 blocks from codes in `[out, in]` order plus per-AutoRound-group
/// fp16 scale bits in `[n_groups, out]` order (as stored in the checkpoint).
/// `group_size` inputs share one scale, replicated across group_size/32 blocks.
pub fn pack_q4_0(
    codes_out_in: &[u8],
    scales_groups_out: &[u16],
    out_f: usize,
    in_f: usize,
    group_size: usize,
) -> Vec<u8> {
    assert_eq!(in_f % 32, 0, "in_features must be divisible by 32");
    assert_eq!(group_size % 32, 0);
    let n_blocks = in_f / 32;
    let n_groups = in_f / group_size;
    let blocks_per_grp = group_size / 32;
    assert_eq!(codes_out_in.len(), out_f * in_f);
    assert_eq!(scales_groups_out.len(), n_groups * out_f);

    let mut out = vec![0u8; out_f * n_blocks * 18];
    for o in 0..out_f {
        let row = &codes_out_in[o * in_f..(o + 1) * in_f];
        for b in 0..n_blocks {
            let d_bits = scales_groups_out[(b / blocks_per_grp) * out_f + o];
            let blk = &mut out[(o * n_blocks + b) * 18..(o * n_blocks + b + 1) * 18];
            blk[0..2].copy_from_slice(&d_bits.to_le_bytes());
            let c = &row[b * 32..(b + 1) * 32];
            for j in 0..16 {
                blk[2 + j] = c[j] | (c[j + 16] << 4);
            }
        }
    }
    out
}

/// Pack Q4_0_AR16 blocks (interleaved nibbles) from codes in `[out, in]`
/// order plus a per-16-element-block scale table in `[out, n_blocks]` order.
pub fn pack_q4_0_ar16(
    codes_out_in: &[u8],
    scales_out_blocks: &[u16],
    out_f: usize,
    in_f: usize,
) -> Vec<u8> {
    assert_eq!(in_f % 16, 0, "in_features must be divisible by 16");
    let n_blocks = in_f / 16;
    assert_eq!(codes_out_in.len(), out_f * in_f);
    assert_eq!(scales_out_blocks.len(), out_f * n_blocks);

    let mut out = vec![0u8; out_f * n_blocks * 10];
    for o in 0..out_f {
        let row = &codes_out_in[o * in_f..(o + 1) * in_f];
        for b in 0..n_blocks {
            let d_bits = scales_out_blocks[o * n_blocks + b];
            let blk = &mut out[(o * n_blocks + b) * 10..(o * n_blocks + b + 1) * 10];
            blk[0..2].copy_from_slice(&d_bits.to_le_bytes());
            let c = &row[b * 16..(b + 1) * 16];
            for j in 0..8 {
                blk[2 + j] = c[2 * j] | (c[2 * j + 1] << 4);
            }
        }
    }
    out
}

/// V-head reorder permutation: grouped-by-K-head -> tiled order.
/// `perm[t]` gives the ORIGINAL index feeding target position `t`, over a
/// range of `num_v_heads * head_dim` positions.
pub fn v_reorder_perm(num_k_heads: usize, num_v_heads: usize, head_dim: usize) -> Vec<usize> {
    assert_eq!(num_v_heads % num_k_heads, 0);
    let num_v_per_k = num_v_heads / num_k_heads;
    let mut perm = Vec::with_capacity(num_v_heads * head_dim);
    for vpk in 0..num_v_per_k {
        for kh in 0..num_k_heads {
            for d in 0..head_dim {
                perm.push((kh * num_v_per_k + vpk) * head_dim + d);
            }
        }
    }
    perm
}

/// The V-col permutation must move whole 16-aligned chunks for Q4_0_AR16 to
/// be lossless; assert that invariant (mirrors the Python tool's check).
pub fn assert_16_aligned_chunks(perm: &[usize]) {
    assert_eq!(perm.len() % 16, 0, "col_perm length not a multiple of 16");
    for chunk in perm.chunks_exact(16) {
        assert_eq!(chunk[0] % 16, 0, "col_perm chunk start not 16-aligned");
        for (i, &p) in chunk.iter().enumerate() {
            assert_eq!(
                p,
                chunk[0] + i,
                "col_perm within-chunk order not contiguous"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unpack_matches_autogptq_layout() {
        // One packed word per column: codes 0..8 across input rows.
        // word = sum(k << 4k) so code for row k is k.
        let mut w = 0u32;
        for k in 0..8u32 {
            w |= k << (4 * k);
        }
        let codes = unpack_autogptq_int4(&[w, w], 1, 2);
        assert_eq!(codes.len(), 16);
        for r in 0..8 {
            assert_eq!(codes[r * 2], r as u8);
            assert_eq!(codes[r * 2 + 1], r as u8);
        }
    }

    #[test]
    fn q4_0_split_halves_nibble_layout() {
        // 32 codes 0,1,2,...,31 (mod 16) for one out row; scale bits 0x1234.
        let codes: Vec<u8> = (0..32u8).map(|x| x & 0xf).collect();
        let blob = pack_q4_0(&codes, &[0x1234], 1, 32, 32);
        assert_eq!(blob.len(), 18);
        assert_eq!(&blob[0..2], &0x1234u16.to_le_bytes());
        // qs[j] = code[j] | code[j+16] << 4
        for j in 0..16 {
            assert_eq!(blob[2 + j], codes[j] | (codes[j + 16] << 4));
        }
    }

    #[test]
    fn ar16_interleaved_nibble_layout() {
        let codes: Vec<u8> = (0..16u8).collect();
        let blob = pack_q4_0_ar16(&codes, &[0xbeef], 1, 16);
        assert_eq!(blob.len(), 10);
        assert_eq!(&blob[0..2], &0xbeefu16.to_le_bytes());
        // byte j = code[2j] | code[2j+1] << 4  (NOT the Q4_0 split layout)
        for j in 0..8 {
            assert_eq!(blob[2 + j], codes[2 * j] | (codes[2 * j + 1] << 4));
        }
    }

    #[test]
    fn ar16_scale_replication_and_block_perm_identity() {
        // 2 out rows, 32 inputs = 2 AR16 blocks, group_size 32 -> 2 blocks per
        // group would be wrong; use group replication done by the caller.
        // Here: distinct scale per block, verify placement.
        let codes = vec![0u8; 2 * 32];
        let scales = vec![1u16, 2, 3, 4]; // [out=2][blocks=2]
        let blob = pack_q4_0_ar16(&codes, &scales, 2, 32);
        assert_eq!(blob.len(), 40);
        assert_eq!(&blob[0..2], &1u16.to_le_bytes());
        assert_eq!(&blob[10..12], &2u16.to_le_bytes());
        assert_eq!(&blob[20..22], &3u16.to_le_bytes());
        assert_eq!(&blob[30..32], &4u16.to_le_bytes());
    }

    #[test]
    fn v_perm_matches_numpy_transpose_semantics() {
        // numpy: arange(n).reshape(k, vpk, d).transpose(1, 0, 2).reshape(n)
        // k=2, vpk=3, d=2 -> n=12
        let perm = v_reorder_perm(2, 6, 2);
        let expect = vec![0, 1, 6, 7, 2, 3, 8, 9, 4, 5, 10, 11];
        assert_eq!(perm, expect);
    }

    #[test]
    fn v_col_perm_16_aligned_for_qwen36_topology() {
        // num_k=16, num_v=48, head_v_dim=128 (Qwen3.6-27B GDN topology)
        let perm = v_reorder_perm(16, 48, 128);
        assert_eq!(perm.len(), 6144);
        assert_16_aligned_chunks(&perm);
    }

    #[test]
    fn transpose_u8_blocked() {
        let rows = 70;
        let cols = 130;
        let src: Vec<u8> = (0..rows * cols).map(|i| (i % 251) as u8).collect();
        let dst = transpose_u8(&src, rows, cols);
        for r in 0..rows {
            for c in 0..cols {
                assert_eq!(dst[c * rows + r], src[r * cols + c]);
            }
        }
    }
}
