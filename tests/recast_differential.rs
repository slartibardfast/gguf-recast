//! Differential gate for recast-bf16-fp16: byte-identical GGUF (and TSV)
//! output versus the Python tool `scripts/recast_bf16_to_fp16.py` on the
//! committed synthetic fixture.
//!
//! The fixture and the expected outputs were produced with the reference
//! venv + gguf-py (see scripts/gen_recast_fixture.py); the expected files
//! are the Python tool's actual outputs, committed verbatim.

use std::path::PathBuf;

use gguf_recast::recast;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/recast")
        .join(name)
}

fn run_tier(tier: &str, force_rescale: bool, tag: &str) {
    let out_dir = std::env::temp_dir().join(format!("gguf-recast-diff-{tag}"));
    std::fs::create_dir_all(&out_dir).unwrap();
    let out_gguf = out_dir.join("out.gguf");
    let out_tsv = out_dir.join("out.tsv");
    let opts = recast::Options {
        input: fixture("recast-fixture.gguf"),
        output: (tier != "dry-run").then(|| out_gguf.clone()),
        policy: fixture("recast-policy.json"),
        tier: tier.to_string(),
        absmax_tsv: Some(out_tsv.clone()),
        force_rescale,
    };
    recast::run(&opts).unwrap();

    let expect_tsv = std::fs::read(fixture(&format!("recast-py-{tag}.tsv"))).unwrap();
    let got_tsv = std::fs::read(&out_tsv).unwrap();
    assert_eq!(got_tsv, expect_tsv, "{tag}: TSV differs from Python tool");

    if tier != "dry-run" {
        let expect = std::fs::read(fixture(&format!("recast-py-{tag}.gguf"))).unwrap();
        let got = std::fs::read(&out_gguf).unwrap();
        assert_eq!(got, expect, "{tag}: GGUF differs from Python tool");
    }
    std::fs::remove_dir_all(&out_dir).ok();
}

#[test]
fn t1_byte_identical() {
    run_tier("T1", false, "T1");
}

#[test]
fn t2_byte_identical() {
    run_tier("T2", false, "T2");
}

#[test]
fn t5_force_rescale_byte_identical() {
    run_tier("T5", true, "T5f");
}

#[test]
fn dry_run_tsv_identical() {
    run_tier("dry-run", false, "dry");
}
