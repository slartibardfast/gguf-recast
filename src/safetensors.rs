//! Memory-mapped multi-shard safetensors reader mirroring gguf-py's
//! `SafetensorsLocal`: within each shard, tensors are ordered by name
//! (lexicographic), and shards are visited in sorted filename order.

use std::collections::BTreeMap;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use memmap2::Mmap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    BF16,
    F16,
    F32,
    I32,
}

impl Dtype {
    fn parse(s: &str) -> io::Result<Dtype> {
        Ok(match s {
            "BF16" => Dtype::BF16,
            "F16" => Dtype::F16,
            "F32" => Dtype::F32,
            "I32" => Dtype::I32,
            other => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unsupported safetensors dtype {other}"),
                ))
            }
        })
    }
}

#[derive(Debug, Clone)]
pub struct TensorRef {
    pub shard: usize,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub start: usize,
    pub len: usize,
}

pub struct ShardSet {
    mmaps: Vec<Mmap>,
    /// (name, tensor) in gguf-py model_tensors insertion order.
    pub tensors: Vec<(String, TensorRef)>,
}

impl ShardSet {
    /// `part_names` must already be in the order gguf-py visits them
    /// (sorted unique weight_map values).
    pub fn open(dir: &Path, part_names: &[String]) -> io::Result<ShardSet> {
        let mut mmaps = Vec::new();
        let mut tensors = Vec::new();
        for (shard, part) in part_names.iter().enumerate() {
            let path: PathBuf = dir.join(part);
            let mut f = std::fs::File::open(&path)
                .map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", path.display())))?;
            let mut lenb = [0u8; 8];
            f.read_exact(&mut lenb)?;
            let hdr_len = u64::from_le_bytes(lenb) as usize;
            let mut hdr = vec![0u8; hdr_len];
            f.read_exact(&mut hdr)?;
            let hdr: serde_json::Value = serde_json::from_slice(&hdr)?;
            let obj = hdr.as_object().ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "bad safetensors header")
            })?;
            let data_start = 8 + hdr_len;

            // BTreeMap gives the by-name ordering SafetensorsLocal applies.
            let mut by_name: BTreeMap<&String, &serde_json::Value> = BTreeMap::new();
            for (k, v) in obj {
                if k != "__metadata__" {
                    by_name.insert(k, v);
                }
            }
            for (name, meta) in by_name {
                let dtype = Dtype::parse(meta["dtype"].as_str().unwrap_or(""))?;
                let shape: Vec<usize> = meta["shape"]
                    .as_array()
                    .map(|a| a.iter().map(|d| d.as_u64().unwrap_or(0) as usize).collect())
                    .unwrap_or_default();
                let offs = meta["data_offsets"].as_array().ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "missing data_offsets")
                })?;
                let start = offs[0].as_u64().unwrap() as usize;
                let end = offs[1].as_u64().unwrap() as usize;
                tensors.push((
                    name.clone(),
                    TensorRef {
                        shard,
                        dtype,
                        shape,
                        start: data_start + start,
                        len: end - start,
                    },
                ));
            }
            mmaps.push(unsafe { Mmap::map(&f)? });
        }
        Ok(ShardSet { mmaps, tensors })
    }

    pub fn bytes(&self, t: &TensorRef) -> &[u8] {
        &self.mmaps[t.shard][t.start..t.start + t.len]
    }

    /// View a BF16/F16 tensor as its raw u16 little-endian element stream.
    pub fn u16_elems(&self, t: &TensorRef) -> Vec<u16> {
        let b = self.bytes(t);
        b.chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect()
    }

    /// View an I32 tensor as its little-endian element stream.
    pub fn i32_elems(&self, t: &TensorRef) -> Vec<u32> {
        let b = self.bytes(t);
        b.chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
}
