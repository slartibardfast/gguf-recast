//! AutoRound INT4 HF checkpoint -> GGUF remix (Q4_0 trunk, V-row-perm Q4_0,
//! V-col-perm Q4_0_AR16 at type id 42), byte-identical to the Python recipe
//! `scripts/autoround_to_q4_0_gguf.py` riding the llama.cpp `autoround`
//! conversion package (`Qwen3_5TextModel` path, `--outtype f16`).

pub mod meta;
pub mod vocab;

use std::collections::HashSet;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use crate::fp::{bf16_to_f16_table, bf16_to_f32, torch_exp_bf16_bits};
use crate::gguf::{self, Value, Writer};
use crate::quant::{
    assert_16_aligned_chunks, pack_q4_0, pack_q4_0_ar16, transpose_u8, unpack_autogptq_int4,
    v_reorder_perm,
};
use crate::safetensors::{Dtype, ShardSet, TensorRef};

const GROUP_SIZE: usize = 128;

pub struct Hparams {
    root: serde_json::Value,
}

impl Hparams {
    fn load(model_dir: &Path) -> io::Result<Hparams> {
        let cfg: serde_json::Value =
            serde_json::from_slice(&std::fs::read(model_dir.join("config.json"))?)?;
        // TextModel moves text_config entries to the root level (overriding).
        let mut root = cfg.clone();
        if let Some(tc) = cfg.get("text_config").and_then(|v| v.as_object()) {
            let obj = root.as_object_mut().unwrap();
            for (k, v) in tc {
                obj.insert(k.clone(), v.clone());
            }
        }
        Ok(Hparams { root })
    }

    fn u64(&self, key: &str) -> u64 {
        self.root[key]
            .as_u64()
            .unwrap_or_else(|| panic!("hparam {key} missing or not an integer"))
    }

    fn f64(&self, key: &str) -> f64 {
        self.root[key]
            .as_f64()
            .unwrap_or_else(|| panic!("hparam {key} missing or not a number"))
    }

    fn get(&self, key: &str) -> Option<&serde_json::Value> {
        match &self.root[key] {
            serde_json::Value::Null => None,
            v => Some(v),
        }
    }
}

/// `TextModel.filter_tensors` multimodal skip list (checked on the original
/// checkpoint name) + `ModelBase.filter_tensors` renames. `_Qwen35MtpMixin`
/// keeps `mtp.*` names verbatim, bypassing the rest of the chain.
fn filter_name(orig: &str) -> Option<String> {
    if orig.starts_with("mtp.") {
        return Some(orig.to_string());
    }
    const PREFIXES: [&str; 10] = [
        "mlp",
        "vit.",
        "vpm.",
        "siglip2.",
        "conformer.",
        "merger.",
        "resampler.",
        "sound_encoder.",
        "sound_projection.",
        "speech_embeddings.",
    ];
    const SUBSTRINGS: [&str; 17] = [
        "visual.",
        "vision.",
        "audio.",
        "talker.",
        "vision_",
        "audio_",
        "sam_model",
        "token2wav.",
        "code2wav.",
        "projector.",
        "pre_mm_projector_norm",
        "image_newline",
        "view_seperator",
        "patch_embed",
        "patch_embedding",
        "patch_merger.",
        "model.connector.",
    ];
    if PREFIXES.iter().any(|p| orig.starts_with(p)) || SUBSTRINGS.iter().any(|s| orig.contains(s)) {
        return None;
    }
    let mut name = orig.to_string();
    if name.ends_with("e_score_correction_bias") {
        name = name.replace("e_score_correction_bias", "e_score_correction.bias");
    }
    name = name.replace("language_model.", "");
    Some(name)
}

/// gguf-py `TensorNameMap` subset for the QWEN35 arch, restricted to the
/// tensors this checkpoint family carries. Fails loudly on anything else.
fn map_tensor_name(base: &str) -> Option<String> {
    match base {
        "model.embed_tokens" => return Some("token_embd".into()),
        "model.norm" => return Some("output_norm".into()),
        "lm_head" => return Some("output".into()),
        _ => {}
    }
    let rest = base.strip_prefix("model.layers.")?;
    let (bid, tail) = rest.split_once('.')?;
    if !bid.bytes().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let mapped = match tail {
        "input_layernorm" => "attn_norm",
        "post_attention_layernorm" => "post_attention_norm",
        "self_attn.q_proj" => "attn_q",
        "self_attn.k_proj" => "attn_k",
        "self_attn.v_proj" => "attn_v",
        "self_attn.o_proj" => "attn_output",
        "self_attn.q_norm" => "attn_q_norm",
        "self_attn.k_norm" => "attn_k_norm",
        "mlp.gate_proj" => "ffn_gate",
        "mlp.up_proj" => "ffn_up",
        "mlp.down_proj" => "ffn_down",
        "linear_attn.in_proj_qkv" => "attn_qkv",
        "linear_attn.in_proj_z" => "attn_gate",
        "linear_attn.out_proj" => "ssm_out",
        "linear_attn.in_proj_a" => "ssm_alpha",
        "linear_attn.in_proj_b" => "ssm_beta",
        "linear_attn.conv1d" => "ssm_conv1d",
        "linear_attn.dt_proj" => "ssm_dt",
        "linear_attn.A_log" => "ssm_a",
        "linear_attn.norm" => "ssm_norm",
        "eh_proj" => "nextn.eh_proj",
        "enorm" => "nextn.enorm",
        "hnorm" => "nextn.hnorm",
        "shared_head.norm" => "nextn.shared_head_norm",
        _ => return None,
    };
    Some(format!("blk.{bid}.{mapped}"))
}

/// `map_tensor_name(name, try_suffixes=(".weight", ".bias"))`
fn map_with_suffix(name: &str) -> Option<String> {
    for suffix in [".weight", ".bias"] {
        if let Some(base) = name.strip_suffix(suffix) {
            return map_tensor_name(base).map(|m| m + suffix);
        }
    }
    map_tensor_name(name)
}

enum Source {
    QuantTrunk,
    QuantVRowQkv,
    QuantVRowZ,
    QuantVCol,
    /// BF16 -> F16, identity layout.
    Bf16ToF16,
    /// BF16 -> F16 with a row gather (rows of `row_len` elements).
    Bf16ToF16RowPerm {
        perm: Vec<usize>,
        row_len: usize,
    },
    /// BF16 -> F32 (+ optional element permutation), plus-one, or -exp.
    F32Plain {
        perm: Option<Vec<usize>>,
    },
    F32PlusOne,
    F32NegExp {
        perm: Vec<usize>,
    },
    /// conv1d: squeeze + V-channel row reorder, F32 out.
    F32Conv1d {
        perm: Vec<usize>,
        qk_rows: usize,
        row_len: usize,
    },
}

struct Planned {
    gguf_name: String,
    dtype: u32,
    /// numpy row-major logical shape (reversed on disk).
    shape: Vec<u64>,
    nbytes: u64,
    /// index of the .qweight tensor (quant) or the source tensor.
    tensor: usize,
    /// index of the .scales tensor for quant sources.
    scales: Option<usize>,
    src: Source,
}

struct Topology {
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    n_layer: usize,
}

pub fn run(model_dir: &Path, outfile: &Path) -> io::Result<()> {
    let hp = Hparams::load(model_dir)?;
    let arch = hp
        .root
        .get("architectures")
        .and_then(|a| a[0].as_str())
        .unwrap_or_default()
        .to_string();
    if arch != "Qwen3_5ForConditionalGeneration" && arch != "Qwen3_5ForCausalLM" {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "unsupported architecture {arch:?}; this port carries the Qwen3_5TextModel path"
            ),
        ));
    }
    let qc = &hp.root["quantization_config"];
    let method = qc["quant_method"].as_str().unwrap_or_default();
    if method != "auto-round" && method != "auto_round" {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "not an auto-round checkpoint",
        ));
    }
    if qc["bits"].as_u64() != Some(4) || qc["sym"].as_bool() != Some(true) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "only auto-round bits=4 sym=true is supported",
        ));
    }
    if qc["group_size"].as_u64() != Some(GROUP_SIZE as u64) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "only group_size=128 is supported",
        ));
    }

    let topo = Topology {
        num_k_heads: hp.u64("linear_num_key_heads") as usize,
        num_v_heads: hp.u64("linear_num_value_heads") as usize,
        head_k_dim: hp.u64("linear_key_head_dim") as usize,
        head_v_dim: hp.u64("linear_value_head_dim") as usize,
        n_layer: hp.u64("num_hidden_layers") as usize,
    };

    // ---- Index the shards in gguf-py model_tensors order. ----
    let index: serde_json::Value = serde_json::from_slice(&std::fs::read(
        model_dir.join("model.safetensors.index.json"),
    )?)?;
    let weight_map = index["weight_map"]
        .as_object()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "no weight_map in index"))?;
    let mut part_names: Vec<String> = weight_map
        .values()
        .filter_map(|v| v.as_str().map(String::from))
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    part_names.sort();
    let shards = ShardSet::open(model_dir, &part_names)?;

    let mut model_tensors: Vec<(String, TensorRef)> = Vec::new();
    for (orig, tref) in &shards.tensors {
        if let Some(name) = filter_name(orig) {
            model_tensors.push((name, tref.clone()));
        }
    }
    let idx_of =
        |name: &str| -> Option<usize> { model_tensors.iter().position(|(n, _)| n == name) };

    // ---- Phase A: AutoRound .qweight triples. ----
    let mut plan: Vec<Planned> = Vec::new();
    let mut consumed: HashSet<usize> = HashSet::new();
    let qweight_order: Vec<usize> = model_tensors
        .iter()
        .enumerate()
        .filter(|(_, (n, _))| n.ends_with(".qweight"))
        .map(|(i, _)| i)
        .collect();

    for &qi in &qweight_order {
        let (qname, qref) = &model_tensors[qi];
        let base = qname.strip_suffix(".qweight").unwrap().to_string();
        let si = idx_of(&(base.clone() + ".scales")).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{base}: missing .scales"),
            )
        })?;
        if qref.dtype != Dtype::I32 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{qname}: not I32"),
            ));
        }
        let in_f = qref.shape[0] * 8;
        let out_f = qref.shape[1];

        let (src, gguf_name, dtype, nbytes) = if base.ends_with(".linear_attn.in_proj_qkv") {
            let name = map_with_suffix(&(base.clone() + ".weight")).unwrap();
            (
                Source::QuantVRowQkv,
                name,
                gguf::T_Q4_0,
                (in_f / 32 * 18 * out_f) as u64,
            )
        } else if base.ends_with(".linear_attn.in_proj_z") {
            let name = map_with_suffix(&(base.clone() + ".weight")).unwrap();
            (
                Source::QuantVRowZ,
                name,
                gguf::T_Q4_0,
                (in_f / 32 * 18 * out_f) as u64,
            )
        } else if base.ends_with(".linear_attn.out_proj") {
            let name = map_with_suffix(&(base.clone() + ".weight")).unwrap();
            (
                Source::QuantVCol,
                name,
                gguf::T_Q4_0_AR16,
                (in_f / 16 * 10 * out_f) as u64,
            )
        } else {
            let mapped_base = remap_mtp_layers(&base, topo.n_layer);
            let name = map_with_suffix(&(mapped_base + ".weight")).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, format!("cannot map {base}"))
            })?;
            (
                Source::QuantTrunk,
                name,
                gguf::T_Q4_0,
                (in_f / 32 * 18 * out_f) as u64,
            )
        };

        plan.push(Planned {
            gguf_name,
            dtype,
            shape: vec![out_f as u64, in_f as u64],
            nbytes,
            tensor: qi,
            scales: Some(si),
            src,
        });
        for suffix in [".qweight", ".qzeros", ".scales", ".g_idx"] {
            if let Some(i) = idx_of(&(base.clone() + suffix)) {
                consumed.insert(i);
            }
        }
    }

    // ---- Phase B: the standard converter path for everything left. ----
    let perm48 = v_reorder_perm(topo.num_k_heads, topo.num_v_heads, 1);
    let perm_v = v_reorder_perm(topo.num_k_heads, topo.num_v_heads, topo.head_v_dim);
    for (i, (name, tref)) in model_tensors.iter().enumerate() {
        if consumed.contains(&i) {
            continue;
        }
        if tref.dtype != Dtype::BF16 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{name}: unexpected non-BF16 passthrough tensor"),
            ));
        }
        plan.push(plan_standard(name, i, tref, &topo, &perm48, &perm_v)?);
    }

    // ---- KV section. ----
    let total_params: u64 = plan.iter().map(|p| p.shape.iter().product::<u64>()).sum();
    let mut out = Writer::new(BufWriter::with_capacity(
        8 << 20,
        std::fs::File::create(outfile)?,
    ));
    build_kvs(&mut out, model_dir, &hp, total_params)?;
    for p in &plan {
        out.add_tensor_info(&p.gguf_name, p.shape.clone(), p.dtype, p.nbytes);
    }
    out.write_header()?;

    // ---- Data section. ----
    let full_perm_qkv: Vec<usize> = {
        let qk = topo.head_k_dim * topo.num_k_heads * 2;
        (0..qk).chain(perm_v.iter().map(|&p| qk + p)).collect()
    };
    assert_16_aligned_chunks(&perm_v);

    for p in &plan {
        let tref = &model_tensors[p.tensor].1;
        match &p.src {
            Source::QuantTrunk | Source::QuantVRowQkv | Source::QuantVRowZ | Source::QuantVCol => {
                let sref = &model_tensors[p.scales.unwrap()].1;
                let blob = emit_quant(p, tref, sref, &shards, &full_perm_qkv, &perm_v)?;
                out.write_tensor_data(&blob)?;
            }
            Source::Bf16ToF16 => {
                let table = bf16_to_f16_table();
                let bytes = shards.bytes(tref);
                out.begin_tensor();
                let mut buf = Vec::with_capacity(8 << 20);
                for chunk in bytes.chunks(8 << 20) {
                    buf.clear();
                    for pair in chunk.chunks_exact(2) {
                        let bf = u16::from_le_bytes([pair[0], pair[1]]);
                        buf.extend_from_slice(&table[bf as usize].to_le_bytes());
                    }
                    out.write_chunk(&buf)?;
                }
                out.end_tensor()?;
            }
            Source::Bf16ToF16RowPerm { perm, row_len } => {
                let table = bf16_to_f16_table();
                let elems = shards.u16_elems(tref);
                let mut buf = Vec::with_capacity(elems.len() * 2);
                for &src_row in perm {
                    for &bf in &elems[src_row * row_len..(src_row + 1) * row_len] {
                        buf.extend_from_slice(&table[bf as usize].to_le_bytes());
                    }
                }
                out.write_tensor_data(&buf)?;
            }
            Source::F32Plain { perm } => {
                let elems = shards.u16_elems(tref);
                let mut buf = Vec::with_capacity(elems.len() * 4);
                match perm {
                    Some(perm) => {
                        for &pi in perm {
                            buf.extend_from_slice(&bf16_to_f32(elems[pi]).to_le_bytes());
                        }
                    }
                    None => {
                        for &bf in &elems {
                            buf.extend_from_slice(&bf16_to_f32(bf).to_le_bytes());
                        }
                    }
                }
                out.write_tensor_data(&buf)?;
            }
            Source::F32PlusOne => {
                let elems = shards.u16_elems(tref);
                let mut buf = Vec::with_capacity(elems.len() * 4);
                for &bf in &elems {
                    buf.extend_from_slice(&(bf16_to_f32(bf) + 1.0).to_le_bytes());
                }
                out.write_tensor_data(&buf)?;
            }
            Source::F32NegExp { perm } => {
                let elems = shards.u16_elems(tref);
                let mut buf = Vec::with_capacity(elems.len() * 4);
                for &pi in perm {
                    // -exp(x): torch negation is a sign-bit flip of the exp result.
                    let bits = torch_exp_bf16_bits(elems[pi]) ^ 0x8000_0000;
                    buf.extend_from_slice(&bits.to_le_bytes());
                }
                out.write_tensor_data(&buf)?;
            }
            Source::F32Conv1d {
                perm,
                qk_rows,
                row_len,
            } => {
                let elems = shards.u16_elems(tref);
                let mut buf = Vec::with_capacity(elems.len() * 4);
                let row = |r: usize| &elems[r * row_len..(r + 1) * row_len];
                for r in 0..*qk_rows {
                    for &bf in row(r) {
                        buf.extend_from_slice(&bf16_to_f32(bf).to_le_bytes());
                    }
                }
                for &p in perm {
                    for &bf in row(qk_rows + p) {
                        buf.extend_from_slice(&bf16_to_f32(bf).to_le_bytes());
                    }
                }
                out.write_tensor_data(&buf)?;
            }
        }
    }
    let inner = out.finish()?;
    inner.into_inner()?.sync_all()?;
    Ok(())
}

fn remap_mtp_layers(base: &str, n_layer: usize) -> String {
    if let Some(rest) = base.strip_prefix("mtp.layers.") {
        if let Some((k, tail)) = rest.split_once('.') {
            if let Ok(k) = k.parse::<usize>() {
                return format!("model.layers.{}.{}", k + n_layer, tail);
            }
        }
    }
    base.to_string()
}

fn plan_standard(
    name: &str,
    tensor_idx: usize,
    tref: &TensorRef,
    topo: &Topology,
    perm48: &[usize],
    perm_v: &[usize],
) -> io::Result<Planned> {
    // _Qwen35MtpMixin.modify_tensors: remap mtp.* names.
    let name: String = if let Some(rest) = name.strip_prefix("mtp.") {
        if let Some(rest) = rest.strip_prefix("layers.") {
            let (k, tail) = rest.split_once('.').unwrap();
            let k: usize = k.parse().unwrap();
            format!("model.layers.{}.{}", k + topo.n_layer, tail)
        } else {
            // mtp.fc.weight etc: Path(name).stem drops the final suffix.
            let full = format!("mtp.{rest}");
            let (stem, suffix) = match full.rfind('.') {
                Some(i) => (&full[..i], &full[i..]),
                None => (full.as_str(), ""),
            };
            let mapped = match stem {
                "mtp.fc" => "eh_proj",
                "mtp.pre_fc_norm_embedding" => "enorm",
                "mtp.pre_fc_norm_hidden" => "hnorm",
                "mtp.norm" => "shared_head.norm",
                other => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("unknown mtp tensor {other}"),
                    ))
                }
            };
            format!("model.layers.{}.{}{}", topo.n_layer, mapped, suffix)
        }
    } else {
        name.to_string()
    };

    // _LinearAttentionVReorderBase.modify_tensors (num_k != num_v), then
    // Qwen3NextModel.modify_tensors transforms, then the final name map.
    let is_linear_attn = name.contains("linear_attn.");
    let src: Source;
    let mut out_name = name.clone();
    let mut shape: Vec<u64> = tref.shape.iter().map(|&d| d as u64).collect();

    if is_linear_attn
        && (name.ends_with(".in_proj_a.weight") || name.ends_with(".in_proj_b.weight"))
    {
        src = Source::Bf16ToF16RowPerm {
            perm: perm48.to_vec(),
            row_len: tref.shape[1],
        };
    } else if is_linear_attn && name.ends_with(".A_log") {
        src = Source::F32NegExp {
            perm: perm48.to_vec(),
        };
    } else if is_linear_attn && name.ends_with(".dt_bias") {
        // rename dt_bias -> dt_proj.bias, reordered, plain upcast (1D -> F32)
        out_name = name.replace(".dt_bias", ".dt_proj.bias");
        src = Source::F32Plain {
            perm: Some(perm48.to_vec()),
        };
    } else if is_linear_attn && name.contains("conv1d") {
        let qk_rows = topo.head_k_dim * topo.num_k_heads * 2;
        // squeeze [C, 1, K] -> [C, K]
        shape = vec![tref.shape[0] as u64, tref.shape[2] as u64];
        src = Source::F32Conv1d {
            perm: perm_v.to_vec(),
            qk_rows,
            row_len: tref.shape[2],
        };
    } else if name.ends_with("norm.weight") && !name.ends_with("linear_attn.norm.weight") {
        src = Source::F32PlusOne;
    } else if is_linear_attn && name.ends_with("norm.weight") {
        src = Source::F32Plain { perm: None };
    } else {
        // Plain tensors: 1D -> F32 upcast, 2D -> F16 cast.
        if tref.shape.len() == 1 {
            src = Source::F32Plain { perm: None };
        } else {
            src = Source::Bf16ToF16;
        }
    }

    let gguf_name = map_with_suffix(&out_name).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("cannot map tensor {out_name}"),
        )
    })?;

    // ModelBase.prepare_tensors dtype rules.
    let n_dims = shape.len();
    let is_f32 = n_dims <= 1
        || gguf_name.ends_with("_norm.weight")
        || gguf_name.contains(".ssm_conv1d.")
        || !matches!(
            &gguf_name[gguf_name.len().saturating_sub(7)..],
            ".weight" | ".lora_a" | ".lora_b"
        );
    let (dtype, elem) = if is_f32 {
        (gguf::T_F32, 4)
    } else {
        (gguf::T_F16, 2)
    };
    // The planned Sources must agree with the converter's dtype decision.
    match (&src, dtype) {
        (Source::Bf16ToF16 | Source::Bf16ToF16RowPerm { .. }, gguf::T_F16) => {}
        (
            Source::F32Plain { .. }
            | Source::F32PlusOne
            | Source::F32NegExp { .. }
            | Source::F32Conv1d { .. },
            gguf::T_F32,
        ) => {}
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{gguf_name}: dtype routing mismatch"),
            ))
        }
    }

    let n_elems: u64 = shape.iter().product();
    Ok(Planned {
        gguf_name,
        dtype,
        shape,
        nbytes: n_elems * elem,
        tensor: tensor_idx,
        scales: None,
        src,
    })
}

fn emit_quant(
    p: &Planned,
    qref: &TensorRef,
    sref: &TensorRef,
    shards: &ShardSet,
    full_perm_qkv: &[usize],
    perm_v: &[usize],
) -> io::Result<Vec<u8>> {
    let in_packed = qref.shape[0];
    let out_f = qref.shape[1];
    let in_f = in_packed * 8;
    let n_groups = in_f / GROUP_SIZE;
    if sref.shape != [n_groups, out_f] {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "{}: scales shape {:?} != [{n_groups}, {out_f}]",
                p.gguf_name, sref.shape
            ),
        ));
    }
    let qwords = shards.i32_elems(qref);
    let codes_in_out = unpack_autogptq_int4(&qwords, in_packed, out_f);
    let scales = shards.u16_elems(sref); // [n_groups, out] fp16 bits

    Ok(match p.src {
        Source::QuantTrunk => {
            let codes_t = transpose_u8(&codes_in_out, in_f, out_f);
            pack_q4_0(&codes_t, &scales, out_f, in_f, GROUP_SIZE)
        }
        Source::QuantVRowQkv | Source::QuantVRowZ => {
            let perm: &[usize] = if matches!(p.src, Source::QuantVRowQkv) {
                full_perm_qkv
            } else {
                perm_v
            };
            assert_eq!(perm.len(), out_f, "{}: out perm length", p.gguf_name);
            // Permute the OUT dim: gather columns of [in, out], via transpose.
            let codes_t = transpose_u8(&codes_in_out, in_f, out_f); // [out, in]
            let mut codes_p = vec![0u8; codes_t.len()];
            for (j, &pj) in perm.iter().enumerate() {
                codes_p[j * in_f..(j + 1) * in_f]
                    .copy_from_slice(&codes_t[pj * in_f..(pj + 1) * in_f]);
            }
            let mut scales_p = vec![0u16; scales.len()];
            for g in 0..n_groups {
                for (j, &pj) in perm.iter().enumerate() {
                    scales_p[g * out_f + j] = scales[g * out_f + pj];
                }
            }
            pack_q4_0(&codes_p, &scales_p, out_f, in_f, GROUP_SIZE)
        }
        Source::QuantVCol => {
            assert_eq!(perm_v.len(), in_f, "{}: col perm length", p.gguf_name);
            // Permute the IN dim: gather rows of [in, out].
            let mut codes_p = vec![0u8; codes_in_out.len()];
            for (i, &pi) in perm_v.iter().enumerate() {
                codes_p[i * out_f..(i + 1) * out_f]
                    .copy_from_slice(&codes_in_out[pi * out_f..(pi + 1) * out_f]);
            }
            let codes_t = transpose_u8(&codes_p, in_f, out_f); // [out, in]
            let n_blocks = in_f / 16;
            let blocks_per_grp = GROUP_SIZE / 16;
            let mut sc_out_blocks = vec![0u16; out_f * n_blocks];
            for b in 0..n_blocks {
                let src_block = perm_v[b * 16] / 16;
                let g = src_block / blocks_per_grp;
                for o in 0..out_f {
                    sc_out_blocks[o * n_blocks + b] = scales[g * out_f + o];
                }
            }
            pack_q4_0_ar16(&codes_t, &sc_out_blocks, out_f, in_f)
        }
        _ => unreachable!(),
    })
}

fn build_kvs(
    out: &mut Writer<impl Write>,
    model_dir: &Path,
    hp: &Hparams,
    total_params: u64,
) -> io::Result<()> {
    out.add_string("general.architecture", "qwen35");
    out.add_string("general.type", "model");

    // Metadata heuristics (gguf-py Metadata.load path for this checkpoint
    // family: model-card frontmatter + generation_config + dir-name fallback).
    let card = load_model_card(model_dir)?;
    let gen_config: Option<serde_json::Value> =
        std::fs::read(model_dir.join("generation_config.json"))
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok());

    if let Some(gc) = &gen_config {
        if let Some(s) = gc["sequence"].as_str() {
            out.add_string("general.sampling.sequence", s);
        }
        if let Some(v) = gc["top_k"].as_i64() {
            out.add_kv("general.sampling.top_k", Value::I32(v as i32));
        }
        if let Some(v) = gc["top_p"].as_f64() {
            out.add_kv("general.sampling.top_p", Value::F32(v as f32));
        }
        if let Some(v) = gc["min_p"].as_f64() {
            out.add_kv("general.sampling.min_p", Value::F32(v as f32));
        }
        if let Some(v) = gc["xtc_probability"].as_f64() {
            out.add_kv("general.sampling.xtc_probability", Value::F32(v as f32));
        }
        if let Some(v) = gc["xtc_threshold"].as_f64() {
            out.add_kv("general.sampling.xtc_threshold", Value::F32(v as f32));
        }
        if let Some(v) = gc["temperature"].as_f64() {
            out.add_kv("general.sampling.temp", Value::F32(v as f32));
        }
        if let Some(v) = gc["penalty_last_n"].as_i64() {
            out.add_kv("general.sampling.penalty_last_n", Value::I32(v as i32));
        }
        if let Some(v) = gc["penalty_repeat"].as_f64() {
            out.add_kv("general.sampling.penalty_repeat", Value::F32(v as f32));
        }
        if let Some(v) = gc["mirostat"].as_i64() {
            out.add_kv("general.sampling.mirostat", Value::I32(v as i32));
        }
        if let Some(v) = gc["mirostat_tau"].as_f64() {
            out.add_kv("general.sampling.mirostat_tau", Value::F32(v as f32));
        }
        if let Some(v) = gc["mirostat_eta"].as_f64() {
            out.add_kv("general.sampling.mirostat_eta", Value::F32(v as f32));
        }
    }

    // Metadata fields: model-card keys first, then the hf `_name_or_path`
    // heuristic, then the directory-name fallback (gguf-py order).
    let dir_name = model_dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("ggml-model")
        .to_string();
    let card_str = |key: &str| -> Option<String> {
        card.as_ref()
            .and_then(|c| c[key].as_str().map(String::from))
    };
    let mut name = card_str("name").or_else(|| card_str("model_name"));
    let author = card_str("author")
        .or_else(|| card_str("model_author"))
        .or_else(|| card_str("model_creator"));
    let mut version = card_str("version").or_else(|| card_str("model_version"));
    let mut organization = card_str("organization").or_else(|| card_str("model_organization"));
    let mut finetune = card_str("finetune").or_else(|| card_str("model_finetune"));
    let mut basename = card_str("basename")
        .or_else(|| card_str("model_basename"))
        .or_else(|| card_str("model_type"));
    let description = card_str("description").or_else(|| card_str("model_description"));
    let mut size_label = card_str("size_label").or_else(|| card_str("model_size_label"));

    let hf_name_or_path = hp.root["_name_or_path"].as_str().unwrap_or_default();
    if !hf_name_or_path.is_empty() && hf_name_or_path.matches('/').count() <= 1 {
        let c = meta::get_model_id_components(hf_name_or_path, total_params as i64);
        if name.is_none() {
            name = c.model_full_name.as_deref().map(meta::id_to_title);
        }
        if organization.is_none() {
            organization = c.org.as_deref().map(meta::id_to_title);
        }
        if basename.is_none() {
            basename = c.basename.clone();
        }
        if finetune.is_none() {
            finetune = c.finetune.clone();
        }
        if version.is_none() {
            version = c.version.clone();
        }
        if size_label.is_none() {
            size_label = c.size_label.clone();
        }
    }
    let dir_components = meta::get_model_id_components(&dir_name, total_params as i64);
    if name.is_none() {
        name = dir_components
            .model_full_name
            .as_deref()
            .map(meta::id_to_title);
    }
    if organization.is_none() {
        organization = dir_components.org.as_deref().map(meta::id_to_title);
    }
    if basename.is_none() {
        basename = dir_components.basename.clone();
    }
    if finetune.is_none() {
        finetune = dir_components.finetune.clone();
    }
    if version.is_none() {
        version = dir_components.version.clone();
    }
    if size_label.is_none() {
        size_label = dir_components.size_label.clone();
    }

    out.add_string("general.name", &name.unwrap_or(dir_name));
    if let Some(v) = &author {
        out.add_string("general.author", v);
    }
    if let Some(v) = &version {
        out.add_string("general.version", v);
    }
    if let Some(v) = &organization {
        out.add_string("general.organization", v);
    }
    if let Some(v) = &finetune {
        out.add_string("general.finetune", v);
    }
    if let Some(v) = &basename {
        out.add_string("general.basename", v);
    }
    if let Some(v) = &description {
        out.add_string("general.description", v);
    }
    let size_label = size_label.unwrap_or_else(|| meta::size_label_from_params(total_params));
    out.add_string("general.size_label", &size_label);

    if let Some(card) = &card {
        match &card["license"] {
            serde_json::Value::String(s) => out.add_string("general.license", s),
            serde_json::Value::Array(a) => {
                let joined = a
                    .iter()
                    .filter_map(|v| v.as_str())
                    .collect::<Vec<_>>()
                    .join(",");
                out.add_string("general.license", &joined);
            }
            _ => {}
        }
        if let Some(s) = card["license_name"].as_str() {
            out.add_string("general.license_name", s);
        }
        if let Some(s) = card["license_link"].as_str() {
            out.add_string("general.license_link", s);
        }

        let base_model_value = card
            .get("base_model")
            .or_else(|| card.get("base_models"))
            .or_else(|| card.get("base_model_sources"));
        if let Some(bm) = base_model_value {
            let ids: Vec<&str> = match bm {
                serde_json::Value::String(s) => vec![s.as_str()],
                serde_json::Value::Array(a) => a.iter().filter_map(|v| v.as_str()).collect(),
                _ => Vec::new(),
            };
            out.add_kv("general.base_model.count", Value::U32(ids.len() as u32));
            for (k, id) in ids.iter().enumerate() {
                if id.starts_with("http://")
                    || id.starts_with("https://")
                    || id.starts_with("ssh://")
                {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "URL-form base_model entries are not supported by this port",
                    ));
                }
                let c = meta::get_model_id_components(id, total_params as i64);
                if let Some(full) = &c.model_full_name {
                    out.add_string(
                        &format!("general.base_model.{k}.name"),
                        &meta::id_to_title(full),
                    );
                }
                if let Some(v) = &c.version {
                    out.add_string(&format!("general.base_model.{k}.version"), v);
                }
                if let Some(org) = &c.org {
                    out.add_string(
                        &format!("general.base_model.{k}.organization"),
                        &meta::id_to_title(org),
                    );
                }
                if let (Some(org), Some(full)) = (&c.org, &c.model_full_name) {
                    out.add_string(
                        &format!("general.base_model.{k}.repo_url"),
                        &format!("https://huggingface.co/{org}/{full}"),
                    );
                }
            }
        }
        if card.get("datasets").is_some() || card.get("dataset").is_some() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "model-card dataset metadata is not supported by this port",
            ));
        }
        let str_list = |key: &str| -> Vec<String> {
            match &card[key] {
                serde_json::Value::String(s) => vec![s.clone()],
                serde_json::Value::Array(a) => a
                    .iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect(),
                _ => Vec::new(),
            }
        };
        let mut tags = str_list("tags");
        tags.extend(str_list("pipeline_tag"));
        if !tags.is_empty() {
            out.add_kv("general.tags", Value::ArrStr(tags));
        }
        let mut languages = str_list("languages");
        languages.extend(str_list("language"));
        if !languages.is_empty() {
            out.add_kv("general.languages", Value::ArrStr(languages));
        }
    }

    // TextModel.set_gguf_parameters + Qwen3Next + MtpMixin.
    let block_count =
        hp.u64("num_hidden_layers") + hp.root["mtp_num_hidden_layers"].as_u64().unwrap_or(0);
    out.add_kv("qwen35.block_count", Value::U32(block_count as u32));
    out.add_kv(
        "qwen35.context_length",
        Value::U32(hp.u64("max_position_embeddings") as u32),
    );
    out.add_kv(
        "qwen35.embedding_length",
        Value::U32(hp.u64("hidden_size") as u32),
    );
    out.add_kv(
        "qwen35.feed_forward_length",
        Value::U32(hp.u64("intermediate_size") as u32),
    );
    out.add_kv(
        "qwen35.attention.head_count",
        Value::U32(hp.u64("num_attention_heads") as u32),
    );
    out.add_kv(
        "qwen35.attention.head_count_kv",
        Value::U32(hp.u64("num_key_value_heads") as u32),
    );

    let rope = hp
        .get("rope_parameters")
        .or_else(|| hp.get("rope_scaling"))
        .cloned()
        .unwrap_or(serde_json::json!({}));
    if let Some(rt) = rope["rope_type"].as_str() {
        // "default" and other unscaled types add no scaling KVs.
        if matches!(rt, "linear" | "yarn" | "su" | "longrope") {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("rope scaling type {rt:?} is not supported by this port"),
            ));
        }
    }
    if let Some(sections) = rope["mrope_section"].as_array() {
        let mut v: Vec<i32> = sections
            .iter()
            .filter_map(|x| x.as_i64())
            .map(|x| x as i32)
            .collect();
        while v.len() < 4 {
            v.push(0);
        }
        v.truncate(4);
        out.add_kv("qwen35.rope.dimension_sections", Value::ArrI32(v));
    }
    if let Some(theta) = rope["rope_theta"].as_f64() {
        out.add_kv("qwen35.rope.freq_base", Value::F32(theta as f32));
    }
    out.add_kv(
        "qwen35.attention.layer_norm_rms_epsilon",
        Value::F32(hp.f64("rms_norm_eps") as f32),
    );
    let head_dim = hp.u64("head_dim");
    out.add_kv("qwen35.attention.key_length", Value::U32(head_dim as u32));
    out.add_kv("qwen35.attention.value_length", Value::U32(head_dim as u32));
    out.add_kv("general.file_type", Value::U32(1)); // MOSTLY_F16

    out.add_kv(
        "qwen35.ssm.conv_kernel",
        Value::U32(hp.u64("linear_conv_kernel_dim") as u32),
    );
    out.add_kv(
        "qwen35.ssm.state_size",
        Value::U32(hp.u64("linear_key_head_dim") as u32),
    );
    out.add_kv(
        "qwen35.ssm.group_count",
        Value::U32(hp.u64("linear_num_key_heads") as u32),
    );
    out.add_kv(
        "qwen35.ssm.time_step_rank",
        Value::U32(hp.u64("linear_num_value_heads") as u32),
    );
    out.add_kv(
        "qwen35.ssm.inner_size",
        Value::U32((hp.u64("linear_value_head_dim") * hp.u64("linear_num_value_heads")) as u32),
    );
    out.add_kv(
        "qwen35.full_attention_interval",
        Value::U32(hp.root["full_attention_interval"].as_u64().unwrap_or(4) as u32),
    );
    let partial = hp.root["partial_rotary_factor"].as_f64().unwrap_or(0.25);
    out.add_kv(
        "qwen35.rope.dimension_count",
        Value::U32((head_dim as f64 * partial) as u32),
    );
    let nextn = hp.root["mtp_num_hidden_layers"].as_u64().unwrap_or(0);
    if nextn > 0 {
        out.add_kv("qwen35.nextn_predict_layers", Value::U32(nextn as u32));
    }
    out.add_kv("general.quantization_version", Value::U32(2));

    // Vocab (TextModel.prepare_metadata -> set_vocab -> _set_vocab_gpt2).
    let model_type = hp.root["model_type"].as_str().unwrap_or_default();
    let tokpre = match model_type {
        "qwen3_5" | "qwen3_5_text" => "qwen35",
        other => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("no tokenizer.ggml.pre mapping for model_type {other:?}"),
            ))
        }
    };
    let vocab_size = hp.u64("vocab_size") as usize;
    let v = vocab::load_vocab(model_dir, vocab_size)?;
    out.add_string("tokenizer.ggml.model", "gpt2");
    out.add_string("tokenizer.ggml.pre", tokpre);
    out.add_kv("tokenizer.ggml.tokens", Value::ArrStr(v.tokens));
    out.add_kv("tokenizer.ggml.token_type", Value::ArrI32(v.token_types));
    out.add_kv("tokenizer.ggml.merges", Value::ArrStr(v.merges));
    for (typ, id) in vocab::special_token_ids(model_dir)? {
        let key = match typ {
            "bos" => "tokenizer.ggml.bos_token_id",
            "eos" => "tokenizer.ggml.eos_token_id",
            "unk" => "tokenizer.ggml.unknown_token_id",
            "sep" => "tokenizer.ggml.seperator_token_id",
            "pad" => "tokenizer.ggml.padding_token_id",
            "cls" => "tokenizer.ggml.cls_token_id",
            "mask" => "tokenizer.ggml.mask_token_id",
            _ => unreachable!(),
        };
        out.add_kv(key, Value::U32(id));
    }
    if let Some(ct) = vocab::chat_template(model_dir)? {
        out.add_string("tokenizer.chat_template", &ct);
    }
    Ok(())
}

/// README.md YAML front-matter (gguf-py `Metadata.load_model_card`).
fn load_model_card(model_dir: &Path) -> io::Result<Option<serde_json::Value>> {
    let path = model_dir.join("README.md");
    if !path.is_file() {
        return Ok(None);
    }
    let content = std::fs::read_to_string(path)?;
    let mut lines = content.lines();
    if lines.next() != Some("---") {
        return Ok(None);
    }
    let mut yaml_lines: Vec<&str> = Vec::new();
    for line in lines {
        if line == "---" {
            break;
        }
        yaml_lines.push(line);
    }
    let mut yaml = yaml_lines.join("\n") + "\n";
    yaml = yaml.replace("- no\n", "- \"no\"\n").replace('\t', "  ");
    let value: serde_yaml::Value = serde_yaml::from_str(&yaml)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("model card YAML: {e}")))?;
    let json = serde_json::to_value(value)?;
    Ok(json.as_object().is_some().then_some(json))
}
