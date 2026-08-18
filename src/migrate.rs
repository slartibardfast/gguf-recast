//! Type-code migration for GGUF files written by earlier fork enums.
//!
//! Rewrites a GGUF so every tensor carries the type code of the current
//! enum, remapping ids that moved between enums (see
//! `gguf::TYPE_CODE_MIGRATIONS` / `gguf::FTYPE_MIGRATIONS`). Tensor data is
//! copied byte-identical; only the ids in the header change. The flow is
//! generic: adding a row to the migration tables extends it to the next enum
//! move.

use std::io::{self, BufWriter};
use std::path::PathBuf;

use crate::gguf::{self, Reader, Value, Writer};

pub fn run(input: &PathBuf, output: &PathBuf) -> io::Result<()> {
    let reader = Reader::open(input)?;

    let mut w = Writer::new(BufWriter::with_capacity(
        4 << 20,
        std::fs::File::create(output)?,
    ));

    // ---- KV section: architecture, every other KV verbatim, ftype remapped. ----
    let mut ftype: Option<u32> = None;
    for (key, value) in &reader.kvs {
        if key == "general.architecture" {
            if let Value::Str(s) = value {
                w.add_string(key, s);
            }
            continue;
        }
        if key == "general.ftype" {
            let f = match value {
                Value::U32(v) => *v,
                Value::I32(v) => *v as u32,
                _ => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "general.ftype is not an integer",
                    ))
                }
            };
            let mut remapped = f;
            for (legacy, current) in gguf::FTYPE_MIGRATIONS {
                if *legacy == f {
                    remapped = *current;
                }
            }
            ftype = Some(remapped);
            w.add_kv(key, Value::U32(remapped));
            continue;
        }
        match value {
            Value::Str(s) => w.add_string(key, s),
            Value::ArrStr(v) if !v.is_empty() => w.add_kv(key, Value::ArrStr(v.clone())),
            Value::ArrI32(v) if !v.is_empty() => w.add_kv(key, Value::ArrI32(v.clone())),
            Value::ArrF32(v) if !v.is_empty() => w.add_kv(key, Value::ArrF32(v.clone())),
            Value::ArrBool(v) if !v.is_empty() => w.add_kv(key, Value::ArrBool(v.clone())),
            scalar => w.add_kv(key, scalar.clone()),
        }
    }

    // ---- Tensor infos with codes resolved to the current enum. ----
    let mut migrated: Vec<(String, u32, u32)> = Vec::new();
    for t in &reader.tensors {
        if t.dtype_orig != t.dtype {
            migrated.push((t.name.clone(), t.dtype_orig, t.dtype));
        }
        let numpy_shape: Vec<u64> = t.dims.iter().rev().copied().collect();
        w.add_tensor_info(&t.name, numpy_shape, t.dtype, t.nbytes);
    }
    if !migrated.is_empty() {
        w.add_kv(
            "recast.migrated",
            Value::ArrStr(
                migrated
                    .iter()
                    .map(|(n, o, c)| format!("{n}: {o}->{c}"))
                    .collect(),
            ),
        );
    }
    w.write_header()?;

    // ---- Pass 2: stream data (byte-identical). ----
    for t in &reader.tensors {
        let data = reader.tensor_data(input, t)?;
        w.write_tensor_data(&data)?;
    }
    let inner = w.finish()?;
    inner
        .into_inner()
        .map_err(|e| io::Error::other(e.to_string()))?
        .sync_all()?;

    eprintln!(
        "wrote {}  ({} tensors, {} type codes migrated{})",
        output.display(),
        reader.tensors.len(),
        migrated.len(),
        match ftype {
            Some(f) => format!(", general.ftype = {f}"),
            None => String::new(),
        },
    );
    for (name, orig, cur) in &migrated {
        eprintln!("  migrate {name}: code {orig} -> {cur}");
    }
    Ok(())
}