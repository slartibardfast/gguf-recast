//! Port of `scripts/recast_bf16_to_fp16.py` (Tool 3): per-tensor
//! absmax-aware BF16 -> FP16 selective recast, byte-identical output.
//!
//! Faithfully replicated quirks of the Python tool:
//! - copied KV arrays are re-encoded with inferred element types (any int
//!   array becomes I32, any float array F32), as gguf-py's `add_array` does;
//! - T4 computes and *records* per-channel scales in the KV section but the
//!   second (data) pass never applies them: the emitted data is a plain RNE
//!   cast. The Python plan tuple only carries (per_tensor_scale, hadamard_d).

use std::collections::HashMap;
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;

use crate::fp::{bf16_to_f16_table, bf16_to_f32, f32_to_f16_bits};
use crate::gguf::{self, Reader, ReaderTensor, Value, Writer};

const FP16_MAX: f64 = 65504.0;
const FP16_HALF_RANGE: f64 = 32768.0;

pub struct Options {
    pub input: PathBuf,
    pub output: Option<PathBuf>,
    pub policy: PathBuf,
    pub tier: String,
    pub absmax_tsv: Option<PathBuf>,
    pub force_rescale: bool,
}

struct Policy {
    preserve_bf16: Vec<regex::Regex>,
}

impl Policy {
    fn load(path: &PathBuf) -> io::Result<Policy> {
        let raw = std::fs::read_to_string(path)?;
        let v: serde_yaml::Value = serde_yaml::from_str(&raw)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("policy: {e}")))?;
        let list = |key: &str, default: &[&str]| -> io::Result<Vec<regex::Regex>> {
            let pats: Vec<String> = match v.get(key) {
                Some(serde_yaml::Value::Sequence(s)) => s
                    .iter()
                    .filter_map(|x| x.as_str().map(String::from))
                    .collect(),
                None => default.iter().map(|s| s.to_string()).collect(),
                Some(_) => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("policy key {key} is not a list"),
                    ))
                }
            };
            pats.iter()
                .map(|p| {
                    regex::Regex::new(p).map_err(|e| {
                        io::Error::new(io::ErrorKind::InvalidData, format!("policy regex: {e}"))
                    })
                })
                .collect()
        };
        // cast_fp16 is parsed for validation parity with the Python tool but,
        // like there, never consulted (the tier dispatch decides casts).
        let _ = list("cast_fp16", &[".*"])?;
        Ok(Policy {
            preserve_bf16: list("preserve_bf16", &[])?,
        })
    }

    fn matches_preserve(&self, name: &str) -> bool {
        self.preserve_bf16.iter().any(|p| p.is_match(name))
    }
}

fn type_name(t: u32) -> &'static str {
    match t {
        gguf::T_F32 => "F32",
        gguf::T_F16 => "F16",
        gguf::T_Q4_0 => "Q4_0",
        8 => "Q8_0",
        gguf::T_BF16 => "BF16",
        gguf::T_Q4_0_AR16 => "Q4_0_AR16",
        _ => "?",
    }
}

fn classify_band(absmax: f64) -> &'static str {
    // NaN falls through both comparisons into band C, as in Python.
    if absmax <= FP16_HALF_RANGE {
        "A"
    } else if absmax <= FP16_MAX {
        "B"
    } else {
        "C"
    }
}

/// BF16 absmax via the sign-cleared bit-pattern maximum (Python bf16_absmax).
fn bf16_absmax(elems: &[u16]) -> f64 {
    if elems.is_empty() {
        return 0.0;
    }
    let max_bits = elems.iter().map(|&b| b & 0x7fff).max().unwrap();
    bf16_to_f32(max_bits) as f64
}

#[derive(Clone, PartialEq)]
enum Kind {
    Bf16Passthrough,
    F32Passthrough,
    RawPassthrough,
    CastF16 { scale: f64, hadamard_d: usize },
}

struct PlanEntry {
    kind: Kind,
    out_dtype: u32,
    band: &'static str,
    absmax: f64,
    note: String,
}

pub fn run(opts: &Options) -> io::Result<()> {
    let tier = opts.tier.as_str();
    if !["dry-run", "T1", "T2", "T3", "T4", "T5"].contains(&tier) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("unknown tier {tier:?}"),
        ));
    }
    let policy = Policy::load(&opts.policy)?;
    let reader = Reader::open(&opts.input)?;

    let arch = match reader.get("general.architecture") {
        Some(Value::Str(s)) => s.clone(),
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "source GGUF lacks general.architecture",
            ))
        }
    };

    // ---- Pass 1: classify. ----
    let mut plan: Vec<PlanEntry> = Vec::new();
    let mut tsv: Vec<String> = Vec::new();
    let mut band_count: HashMap<&str, usize> = HashMap::new();
    let mut per_tensor_scales: Vec<(String, f64)> = Vec::new();
    let mut hadamard_sizes: Vec<(String, usize)> = Vec::new();
    let mut pc_names: Vec<String> = Vec::new();
    let mut pc_lengths: Vec<i32> = Vec::new();
    let mut pc_values: Vec<f32> = Vec::new();

    for t in &reader.tensors {
        let entry = recast_tensor(&reader, &opts.input, t, &policy, tier, opts.force_rescale)?;
        *band_count.entry(entry.band).or_default() += 1;
        let n_elem: u64 = t.dims.iter().product();
        tsv.push(format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}",
            t.name,
            type_name(t.dtype),
            type_name(entry.out_dtype),
            entry.band,
            fmt_g6(entry.absmax),
            n_elem,
            entry.note,
        ));
        if let Kind::CastF16 { scale, hadamard_d } = &entry.kind {
            if *scale != 1.0 {
                per_tensor_scales.push((t.name.clone(), *scale));
            }
            if *hadamard_d != 0 {
                hadamard_sizes.push((t.name.clone(), *hadamard_d));
            }
        }
        // T4 per-channel scale recording (data-pass never applies them).
        if tier == "T4" && matches!(entry.kind, Kind::CastF16 { .. }) && t.dims.len() == 2 {
            let (rows, row_scales) = t4_row_scales(&reader, &opts.input, t, opts.force_rescale)?;
            if !row_scales.is_empty() {
                pc_names.push(t.name.clone());
                pc_lengths.push(rows as i32);
                pc_values.extend(row_scales);
            }
        }
        plan.push(entry);
    }

    if let Some(tsv_path) = &opts.absmax_tsv {
        let mut f = std::fs::File::create(tsv_path)?;
        writeln!(f, "name\tsrc_type\tout_type\tband\tabsmax\tn_elem\tnote")?;
        for row in &tsv {
            writeln!(f, "{row}")?;
        }
        eprintln!("absmax tsv -> {}", tsv_path.display());
    }
    let n_total: usize = band_count.values().sum();
    eprintln!(
        "summary: A={}  B={}  C={}  passthrough/preserved={}  total={}",
        band_count.get("A").unwrap_or(&0),
        band_count.get("B").unwrap_or(&0),
        band_count.get("C").unwrap_or(&0),
        band_count.get("-").unwrap_or(&0),
        n_total
    );

    if tier == "dry-run" {
        return Ok(());
    }
    let output = opts.output.as_ref().expect("checked by caller");

    // ---- KV section: arch, copied KVs, recast.* metadata. ----
    let mut w = Writer::new(BufWriter::with_capacity(
        4 << 20,
        std::fs::File::create(output)?,
    ));
    w.add_string("general.architecture", &arch);
    copy_kvs(&reader, &mut w)?;

    w.add_string("recast.tier", tier);
    if opts.force_rescale {
        w.add_kv("recast.force_rescale", Value::Bool(true));
    }
    if !per_tensor_scales.is_empty() {
        w.add_kv(
            "recast.scales.names",
            Value::ArrStr(per_tensor_scales.iter().map(|(n, _)| n.clone()).collect()),
        );
        w.add_kv(
            "recast.scales.values",
            Value::ArrF32(per_tensor_scales.iter().map(|(_, v)| *v as f32).collect()),
        );
    }
    if !hadamard_sizes.is_empty() {
        w.add_kv(
            "recast.hadamard.names",
            Value::ArrStr(hadamard_sizes.iter().map(|(n, _)| n.clone()).collect()),
        );
        w.add_kv(
            "recast.hadamard.values",
            Value::ArrI32(hadamard_sizes.iter().map(|(_, v)| *v as i32).collect()),
        );
    }
    if !pc_names.is_empty() {
        w.add_kv("recast.per_channel.names", Value::ArrStr(pc_names.clone()));
        w.add_kv(
            "recast.per_channel.lengths",
            Value::ArrI32(pc_lengths.clone()),
        );
        w.add_kv(
            "recast.per_channel.values",
            Value::ArrF32(pc_values.clone()),
        );
    }

    // ---- Tensor infos (dims preserved, dtype per plan). ----
    for (t, entry) in reader.tensors.iter().zip(&plan) {
        let numpy_shape: Vec<u64> = t.dims.iter().rev().copied().collect();
        let n_elem: u64 = t.dims.iter().product();
        match &entry.kind {
            Kind::CastF16 { .. } => {
                w.add_tensor_info(&t.name, numpy_shape, gguf::T_F16, n_elem * 2)
            }
            Kind::Bf16Passthrough => {
                w.add_tensor_info(&t.name, numpy_shape, gguf::T_BF16, t.nbytes)
            }
            Kind::F32Passthrough => {
                w.add_tensor_info(&t.name, numpy_shape, gguf::T_F32, n_elem * 4)
            }
            Kind::RawPassthrough => w.add_tensor_info(&t.name, numpy_shape, t.dtype, t.nbytes),
        }
    }
    w.write_header()?;

    // ---- Pass 2: stream data. ----
    for (t, entry) in reader.tensors.iter().zip(&plan) {
        match &entry.kind {
            Kind::Bf16Passthrough | Kind::F32Passthrough | Kind::RawPassthrough => {
                let data = reader.tensor_data(&opts.input, t)?;
                w.write_tensor_data(&data)?;
            }
            Kind::CastF16 { scale, hadamard_d } => {
                let data = reader.tensor_data(&opts.input, t)?;
                let elems: Vec<u16> = data
                    .chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                    .collect();
                let out = cast_pass2(&t.name, &elems, t, *scale, *hadamard_d)?;
                w.write_tensor_data(&out)?;
            }
        }
    }
    let inner = w.finish()?;
    inner
        .into_inner()
        .map_err(|e| io::Error::other(e.to_string()))?
        .sync_all()?;
    eprintln!("wrote {}  ({} tensors)", output.display(), plan.len());
    Ok(())
}

fn cast_pass2(
    name: &str,
    elems: &[u16],
    t: &ReaderTensor,
    scale: f64,
    hadamard_d: usize,
) -> io::Result<Vec<u8>> {
    let mut out = Vec::with_capacity(elems.len() * 2);
    if hadamard_d == 0 && scale == 1.0 {
        let table = bf16_to_f16_table();
        for &bf in elems {
            let h = table[bf as usize];
            if h & 0x7fff == 0x7c00 {
                return Err(inf_error(name, scale, hadamard_d));
            }
            out.extend_from_slice(&h.to_le_bytes());
        }
        return Ok(out);
    }
    // Materialize f32, optionally rotate rows, rescale, cast.
    let mut f: Vec<f32> = elems.iter().map(|&b| bf16_to_f32(b)).collect();
    if hadamard_d != 0 && t.dims.len() >= 2 {
        // numpy row-major shape: in_dim is the last stored dim reversed = dims[0].
        let in_dim = t.dims[0] as usize;
        if hadamard_d <= in_dim {
            let rows = f.len() / in_dim;
            for r in 0..rows {
                walsh_hadamard_row(&mut f[r * in_dim..r * in_dim + hadamard_d]);
            }
        }
    }
    let scale32 = scale as f32;
    for x in &mut f {
        if scale != 1.0 {
            *x /= scale32;
        }
    }
    for x in &f {
        let h = f32_to_f16_bits(*x);
        if h & 0x7fff == 0x7c00 {
            return Err(inf_error(name, scale, hadamard_d));
        }
        out.extend_from_slice(&h.to_le_bytes());
    }
    Ok(out)
}

fn inf_error(name: &str, scale: f64, hadamard_d: usize) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("{name}: produced Inf in pass-2 cast (scale={scale}, hadamard_d={hadamard_d})"),
    )
}

/// In-place normalized fast Walsh-Hadamard transform of one row (butterfly
/// order identical to the Python `_walsh_hadamard_rows`).
fn walsh_hadamard_row(row: &mut [f32]) {
    let d = row.len();
    assert!(d.is_power_of_two(), "WHT requires power-of-2 length");
    let mut h = 1;
    while h < d {
        let mut i = 0;
        while i < d {
            for j in i..i + h {
                let a = row[j];
                let b = row[j + h];
                row[j] = a + b;
                row[j + h] = a - b;
            }
            i += 2 * h;
        }
        h *= 2;
    }
    let norm = (d as f64).sqrt() as f32;
    for x in row.iter_mut() {
        *x /= norm;
    }
}

fn largest_pow2_le(n: usize) -> usize {
    let mut d = 1;
    while d * 2 <= n {
        d *= 2;
    }
    d
}

fn recast_tensor(
    reader: &Reader,
    input: &std::path::Path,
    t: &ReaderTensor,
    policy: &Policy,
    tier: &str,
    force_rescale: bool,
) -> io::Result<PlanEntry> {
    if t.dtype != gguf::T_BF16 {
        let kind = if t.dtype == gguf::T_F32 {
            Kind::F32Passthrough
        } else {
            Kind::RawPassthrough
        };
        return Ok(PlanEntry {
            kind,
            out_dtype: t.dtype,
            band: "-",
            absmax: 0.0,
            note: "passthrough".into(),
        });
    }
    if policy.matches_preserve(&t.name) {
        return Ok(PlanEntry {
            kind: Kind::Bf16Passthrough,
            out_dtype: gguf::T_BF16,
            band: "-",
            absmax: 0.0,
            note: "policy-preserve".into(),
        });
    }
    let data = reader.tensor_data(input, t)?;
    let elems: Vec<u16> = data
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    let absmax = bf16_absmax(&elems);

    if absmax == 0.0 {
        return Ok(PlanEntry {
            kind: Kind::CastF16 {
                scale: 1.0,
                hadamard_d: 0,
            },
            out_dtype: gguf::T_F16,
            band: "A",
            absmax: 0.0,
            note: "zero-tensor".into(),
        });
    }
    let band = classify_band(absmax);

    if tier == "dry-run" {
        return Ok(PlanEntry {
            kind: Kind::Bf16Passthrough,
            out_dtype: t.dtype,
            band,
            absmax,
            note: format!("dry-run band {band}"),
        });
    }

    match tier {
        "T1" => {
            if band == "A" || band == "B" {
                Ok(PlanEntry {
                    kind: Kind::CastF16 {
                        scale: 1.0,
                        hadamard_d: 0,
                    },
                    out_dtype: gguf::T_F16,
                    band,
                    absmax,
                    note: format!("T1 RNE-cast (band {band})"),
                })
            } else {
                Ok(PlanEntry {
                    kind: Kind::Bf16Passthrough,
                    out_dtype: gguf::T_BF16,
                    band: "C",
                    absmax,
                    note: "T1 BF16 fallback".into(),
                })
            }
        }
        "T2" | "T3" => {
            let scale = if band == "C" || force_rescale {
                absmax / 30000.0
            } else {
                1.0
            };
            let note = if scale != 1.0 {
                format!("{tier} per-tensor /{}", fmt_g6(scale))
            } else {
                format!("{tier} no-rescale")
            };
            Ok(PlanEntry {
                kind: Kind::CastF16 {
                    scale,
                    hadamard_d: 0,
                },
                out_dtype: gguf::T_F16,
                band,
                absmax,
                note,
            })
        }
        "T4" => {
            if t.dims.len() != 2 {
                let scale = if band == "C" || force_rescale {
                    absmax / 30000.0
                } else {
                    1.0
                };
                return Ok(PlanEntry {
                    kind: Kind::CastF16 {
                        scale,
                        hadamard_d: 0,
                    },
                    out_dtype: gguf::T_F16,
                    band,
                    absmax,
                    note: format!("T4 falls-back-T2 /{}", fmt_g6(scale)),
                });
            }
            let (rows, row_scales) = t4_row_scales(reader, input, t, force_rescale)?;
            let n_scaled = row_scales.iter().filter(|&&s| s != 1.0).count();
            Ok(PlanEntry {
                kind: Kind::CastF16 {
                    scale: 1.0,
                    hadamard_d: 0,
                },
                out_dtype: gguf::T_F16,
                band,
                absmax,
                note: format!("T4 per-channel ({n_scaled}/{rows} rows scaled)"),
            })
        }
        "T5" => {
            if t.dims.len() != 2 {
                let scale = if band == "C" || force_rescale {
                    absmax / 30000.0
                } else {
                    1.0
                };
                return Ok(PlanEntry {
                    kind: Kind::CastF16 {
                        scale,
                        hadamard_d: 0,
                    },
                    out_dtype: gguf::T_F16,
                    band,
                    absmax,
                    note: format!("T5 falls-back-T2 /{}", fmt_g6(scale)),
                });
            }
            let in_dim = t.dims[0] as usize; // numpy shape[1]
            let d = largest_pow2_le(in_dim);
            let mut f: Vec<f32> = elems.iter().map(|&b| bf16_to_f32(b)).collect();
            let rows = f.len() / in_dim;
            for r in 0..rows {
                walsh_hadamard_row(&mut f[r * in_dim..r * in_dim + d]);
            }
            let post_absmax = f.iter().fold(0.0f32, |m, &x| m.max(x.abs())) as f64;
            let scale = if post_absmax > FP16_HALF_RANGE || force_rescale {
                (post_absmax / 30000.0).max(1e-30)
            } else {
                1.0
            };
            Ok(PlanEntry {
                kind: Kind::CastF16 {
                    scale,
                    hadamard_d: d,
                },
                out_dtype: gguf::T_F16,
                band,
                absmax,
                note: format!(
                    "T5 hadamard d={d}, post_absmax={}, /{}",
                    fmt_g6(post_absmax),
                    fmt_g6(scale)
                ),
            })
        }
        other => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("unknown tier {other:?}"),
        )),
    }
}

/// T4 pass-1 per-row scale computation (recorded in KVs only).
fn t4_row_scales(
    reader: &Reader,
    input: &std::path::Path,
    t: &ReaderTensor,
    force_rescale: bool,
) -> io::Result<(usize, Vec<f32>)> {
    let data = reader.tensor_data(input, t)?;
    let in_dim = t.dims[0] as usize;
    let elems: Vec<u16> = data
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    let rows = elems.len() / in_dim;
    let mut scales = Vec::with_capacity(rows);
    for r in 0..rows {
        let row = &elems[r * in_dim..(r + 1) * in_dim];
        let row_absmax: f32 = row
            .iter()
            .map(|&b| bf16_to_f32(b).abs())
            .fold(0.0f32, f32::max);
        let s = if force_rescale {
            (row_absmax / 30000.0f32).max(1e-30f32)
        } else if (row_absmax as f64) > FP16_MAX {
            row_absmax / 30000.0f32
        } else {
            1.0f32
        };
        scales.push(s);
    }
    Ok((rows, scales))
}

/// Copy KV pairs with gguf-py `copy_kvs` semantics (see module docs).
fn copy_kvs(reader: &Reader, w: &mut Writer<impl Write>) -> io::Result<()> {
    for (key, value) in &reader.kvs {
        if key == "general.architecture" {
            continue;
        }
        match value {
            Value::Str(s) => w.add_string(key, s),
            Value::ArrStr(v) => {
                if !v.is_empty() {
                    w.add_kv(key, Value::ArrStr(v.clone()));
                }
            }
            Value::ArrI32(v) => {
                if !v.is_empty() {
                    w.add_kv(key, Value::ArrI32(v.clone()));
                }
            }
            Value::ArrF32(v) => {
                if !v.is_empty() {
                    w.add_kv(key, Value::ArrF32(v.clone()));
                }
            }
            Value::ArrBool(v) => {
                if !v.is_empty() {
                    w.add_kv(key, Value::ArrBool(v.clone()));
                }
            }
            scalar => w.add_kv(key, scalar.clone()),
        }
    }
    Ok(())
}

/// Python `f"{x:.6g}"` (C printf %.6g) for non-negative finite values.
pub fn fmt_g6(x: f64) -> String {
    if x == 0.0 {
        return "0".to_string();
    }
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf".into() } else { "-inf".into() };
    }
    // 6 significant digits via exponential formatting (correctly rounded).
    let s = format!("{:.5e}", x);
    let (mant, exp) = s.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let digits: String = mant.chars().filter(|c| c.is_ascii_digit()).collect();
    let neg = mant.starts_with('-');
    debug_assert_eq!(digits.len(), 6);
    let mut out = String::new();
    if neg {
        out.push('-');
    }
    if !(-4..6).contains(&exp) {
        // exponent form: d.ddddd with trailing zeros stripped
        let frac = digits[1..].trim_end_matches('0');
        out.push(digits.as_bytes()[0] as char);
        if !frac.is_empty() {
            out.push('.');
            out.push_str(frac);
        }
        out.push('e');
        if exp < 0 {
            out.push('-');
        } else {
            out.push('+');
        }
        out.push_str(&format!("{:02}", exp.abs()));
    } else if exp >= 0 {
        let e = exp as usize;
        out.push_str(&digits[..=e]);
        let frac = digits[e + 1..].trim_end_matches('0');
        if !frac.is_empty() {
            out.push('.');
            out.push_str(frac);
        }
    } else {
        out.push_str("0.");
        for _ in 0..(-exp - 1) {
            out.push('0');
        }
        let frac = digits.trim_end_matches('0');
        out.push_str(frac);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn g6_formatting_matches_python() {
        // pairs generated with python: f"{x:.6g}"
        let cases: &[(f64, &str)] = &[
            (0.0, "0"),
            (1.0, "1"),
            (0.95, "0.95"),
            (65504.0, "65504"),
            (65536.0, "65536"),
            (1234567.0, "1.23457e+06"),
            (0.000123456789, "0.000123457"),
            (0.0000123456789, "1.23457e-05"),
            (32768.0, "32768"),
            (3.14159265, "3.14159"),
            (123456.7, "123457"),
            (2.5e-07, "2.5e-07"),
            (1e30, "1e+30"),
        ];
        for (x, expect) in cases {
            assert_eq!(&fmt_g6(*x), expect, "for {x}");
        }
    }

    #[test]
    fn band_classification() {
        assert_eq!(classify_band(0.0), "A");
        assert_eq!(classify_band(32768.0), "A");
        assert_eq!(classify_band(32769.0), "B");
        assert_eq!(classify_band(65504.0), "B");
        assert_eq!(classify_band(65505.0), "C");
        assert_eq!(classify_band(f64::NAN), "C");
    }

    #[test]
    fn wht_butterfly_matches_reference() {
        // H_2 on [1, 0] -> [1/sqrt(2), 1/sqrt(2)]
        let mut row = vec![1.0f32, 0.0];
        walsh_hadamard_row(&mut row);
        let s = 1.0 / 2.0f32.sqrt();
        assert_eq!(row, vec![s, s]);
        // Orthonormality: WHT twice = identity (up to f32 rounding on d=4).
        let mut r2 = vec![1.0f32, 2.0, 3.0, 4.0];
        let orig = r2.clone();
        walsh_hadamard_row(&mut r2);
        walsh_hadamard_row(&mut r2);
        for (a, b) in r2.iter().zip(&orig) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn bf16_absmax_bit_trick() {
        // -3.0 (0xc040), 2.0 (0x4000) -> absmax 3.0
        assert_eq!(bf16_absmax(&[0xc040, 0x4000]), 3.0);
        assert_eq!(bf16_absmax(&[]), 0.0);
        assert_eq!(bf16_absmax(&[0x0000, 0x8000]), 0.0);
    }
}
