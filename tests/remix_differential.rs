//! Acceptance gate for the remix (call/0018): byte-identical GGUF to the
//! Python recipe on the same sources. The sources and the Python-emitted
//! reference are multi-gigabyte host artifacts, so this test is opt-in:
//!
//! ```sh
//! GGUF_RECAST_REMIX_SRC=/opt/models/hf-cache/models--Intel--Qwen3.6-27B-int4-AutoRound/snapshots/abc86de19eb1ebbf6a7df4582341325c22ddcb7d \
//! GGUF_RECAST_REMIX_REF=/opt/models/Qwen3.6-27B-Q4_0AR16-b9222.gguf \
//! cargo test --release -- --ignored remix_byte_identical
//! ```
//!
//! Passed on 2026-07-10 against the reference
//! sha256 656c898770e835d85ef0f4eee57b057f03d6d7e1d3e0a1be64aca1e0d7480422
//! (19,246,993,696 bytes, 866 tensors, 42 KVs).

use std::io::Read;

#[test]
#[ignore = "needs the AutoRound snapshot and the 19G Python reference; see module docs"]
fn remix_byte_identical() {
    let src = std::env::var("GGUF_RECAST_REMIX_SRC").expect("set GGUF_RECAST_REMIX_SRC");
    let reference = std::env::var("GGUF_RECAST_REMIX_REF").expect("set GGUF_RECAST_REMIX_REF");
    let out = std::env::temp_dir().join("gguf-recast-remix-differential.gguf");

    gguf_recast::remix::run(std::path::Path::new(&src), &out).expect("remix failed");

    let mut a = std::fs::File::open(&out).unwrap();
    let mut b = std::fs::File::open(&reference).unwrap();
    assert_eq!(
        a.metadata().unwrap().len(),
        b.metadata().unwrap().len(),
        "size mismatch"
    );
    let mut ba = vec![0u8; 8 << 20];
    let mut bb = vec![0u8; 8 << 20];
    let mut off: u64 = 0;
    loop {
        let na = a.read(&mut ba).unwrap();
        let nb = b.read(&mut bb).unwrap();
        assert_eq!(na, nb);
        if na == 0 {
            break;
        }
        assert_eq!(
            &ba[..na],
            &bb[..nb],
            "first differing block at offset {off}"
        );
        off += na as u64;
    }
    std::fs::remove_file(&out).ok();
}
