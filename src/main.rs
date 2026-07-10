//! gguf-recast: maintained Rust home of the GGUF recast tooling (call/0018).
//!
//! Subcommands:
//! - `remix`: AutoRound INT4 HF checkpoint -> GGUF (Q4_0 trunk, V-row-perm
//!   Q4_0, V-col-perm Q4_0_AR16 at id 42), byte-identical to the Python
//!   recipe `autoround_to_q4_0_gguf.py` on the same sources.
//! - `recast-bf16-fp16`: per-tensor absmax-aware BF16 -> FP16 selective
//!   recast, byte-identical to `recast_bf16_to_fp16.py`.

mod fp;
mod gguf;
mod quant;
mod recast;
mod remix;
mod safetensors;

use std::path::PathBuf;
use std::process::ExitCode;

fn usage() -> ExitCode {
    eprintln!(
        "usage:\n  gguf-recast remix --model-dir <hf-snapshot> --outfile <out.gguf>\n  \
         gguf-recast recast-bf16-fp16 --input <src.gguf> --policy <policy.yaml|json> \
         --tier <dry-run|T1|T2|T3|T4|T5> [--output <dst.gguf>] [--absmax-tsv <out.tsv>] \
         [--force-rescale]"
    );
    ExitCode::from(2)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = args.first() else {
        return usage();
    };

    let get_opt = |flag: &str| -> Option<String> {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let has_flag = |flag: &str| args.iter().any(|a| a == flag);

    match cmd.as_str() {
        "remix" => {
            let (Some(model_dir), Some(outfile)) = (get_opt("--model-dir"), get_opt("--outfile"))
            else {
                return usage();
            };
            match remix::run(&PathBuf::from(model_dir), &PathBuf::from(outfile)) {
                Ok(()) => ExitCode::SUCCESS,
                Err(e) => {
                    eprintln!("remix failed: {e}");
                    ExitCode::FAILURE
                }
            }
        }
        "recast-bf16-fp16" => {
            let (Some(input), Some(policy), Some(tier)) =
                (get_opt("--input"), get_opt("--policy"), get_opt("--tier"))
            else {
                return usage();
            };
            let output = get_opt("--output");
            if tier != "dry-run" && output.is_none() {
                eprintln!("--output is required unless --tier dry-run");
                return ExitCode::from(2);
            }
            let opts = recast::Options {
                input: PathBuf::from(input),
                output: output.map(PathBuf::from),
                policy: PathBuf::from(policy),
                tier,
                absmax_tsv: get_opt("--absmax-tsv").map(PathBuf::from),
                force_rescale: has_flag("--force-rescale"),
            };
            match recast::run(&opts) {
                Ok(()) => ExitCode::SUCCESS,
                Err(e) => {
                    eprintln!("recast failed: {e}");
                    ExitCode::FAILURE
                }
            }
        }
        _ => usage(),
    }
}
