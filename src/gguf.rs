//! Minimal GGUF v3 writer/reader mirroring gguf-py's byte-level behavior
//! (little-endian, alignment 32, KV and tensor-info sections in insertion
//! order, tensor data padded to the alignment after each tensor).

use std::io::{self, Read, Seek, SeekFrom, Write};

pub const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF"
pub const GGUF_VERSION: u32 = 3;
pub const ALIGNMENT: u64 = 32;

// ggml type ids used by this tool (current fork numbering).
pub const T_F32: u32 = 0;
pub const T_F16: u32 = 1;
pub const T_Q4_0: u32 = 2;
pub const T_BF16: u32 = 30;
pub const T_Q4_0_AR16: u32 = 43;

/// (block_size, type_size) for the types this tool reads or writes.
pub fn type_sizes(t: u32) -> Option<(u64, u64)> {
    match t {
        T_F32 => Some((1, 4)),
        T_F16 => Some((1, 2)),
        T_Q4_0 => Some((32, 18)),
        T_BF16 => Some((1, 2)),
        T_Q4_0_AR16 => Some((16, 10)),
        8 => Some((32, 34)), // Q8_0
        _ => None,
    }
}

/// Type-code migrations for files written by earlier fork enums. The GGUF
/// format carries no enum version, so when a type's numeric id moves between
/// enums the recast tool must rewrite old files. Entry: (legacy, current).
/// The data layout is unchanged (same quantization), only the id moved.
pub const TYPE_CODE_MIGRATIONS: &[(u32, u32)] = &[
    // 2026-08-17 (fork 5990f5946): upstream ggml took GGML_TYPE_Q2_0 = 42; the
    // fork-local Q4_0_AR16 re-anchored to the tail, 42 -> 43 (count 43 -> 44).
    // Files written before this date carry Q4_0_AR16 as 42.
    (42, 43),
];

/// Same for GGML_FTYPE metadata values (moved with their type code).
pub const FTYPE_MIGRATIONS: &[(u32, u32)] = &[
    (28, 29), // MOSTLY_Q4_0_AR16 28 -> 29
];

/// Resolve a type code read from a (possibly legacy) GGUF to the current enum.
/// Codes already current are returned unchanged.
pub fn resolve_type_code(code: u32) -> u32 {
    for (legacy, current) in TYPE_CODE_MIGRATIONS {
        if *legacy == code {
            return *current;
        }
    }
    code
}

pub fn ggml_pad(x: u64, align: u64) -> u64 {
    x.div_ceil(align) * align
}

// GGUFValueType ids.
pub const V_U8: u32 = 0;
pub const V_I8: u32 = 1;
pub const V_U16: u32 = 2;
pub const V_I16: u32 = 3;
pub const V_U32: u32 = 4;
pub const V_I32: u32 = 5;
pub const V_F32: u32 = 6;
pub const V_BOOL: u32 = 7;
pub const V_STRING: u32 = 8;
pub const V_ARRAY: u32 = 9;
pub const V_U64: u32 = 10;
pub const V_I64: u32 = 11;
pub const V_F64: u32 = 12;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    U64(u64),
    I64(i64),
    F64(f64),
    /// (element type id, raw elements). Strings carry their own encoding.
    ArrStr(Vec<String>),
    ArrI32(Vec<i32>),
    ArrF32(Vec<f32>),
    ArrBool(Vec<bool>),
}

impl Value {
    pub fn type_id(&self) -> u32 {
        match self {
            Value::U8(_) => V_U8,
            Value::I8(_) => V_I8,
            Value::U16(_) => V_U16,
            Value::I16(_) => V_I16,
            Value::U32(_) => V_U32,
            Value::I32(_) => V_I32,
            Value::F32(_) => V_F32,
            Value::Bool(_) => V_BOOL,
            Value::Str(_) => V_STRING,
            Value::U64(_) => V_U64,
            Value::I64(_) => V_I64,
            Value::F64(_) => V_F64,
            Value::ArrStr(_) | Value::ArrI32(_) | Value::ArrF32(_) | Value::ArrBool(_) => V_ARRAY,
        }
    }
}

fn write_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn write_value_payload(out: &mut Vec<u8>, v: &Value) {
    match v {
        Value::U8(x) => out.push(*x),
        Value::I8(x) => out.push(*x as u8),
        Value::U16(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::I16(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::U32(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::I32(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::F32(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::Bool(x) => out.push(*x as u8),
        Value::Str(s) => write_str(out, s),
        Value::U64(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::I64(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::F64(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::ArrStr(items) => {
            out.extend_from_slice(&V_STRING.to_le_bytes());
            out.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for s in items {
                write_str(out, s);
            }
        }
        Value::ArrI32(items) => {
            out.extend_from_slice(&V_I32.to_le_bytes());
            out.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for x in items {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        Value::ArrF32(items) => {
            out.extend_from_slice(&V_F32.to_le_bytes());
            out.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for x in items {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        Value::ArrBool(items) => {
            out.extend_from_slice(&V_BOOL.to_le_bytes());
            out.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for x in items {
                out.push(*x as u8);
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// Logical shape in numpy (row-major) order; written reversed on disk.
    pub shape: Vec<u64>,
    pub dtype: u32,
    pub nbytes: u64,
}

/// Streaming GGUF writer: all KV pairs and tensor infos are declared first,
/// then tensor data is appended in declaration order.
pub struct Writer<W: Write> {
    out: W,
    kvs: Vec<(String, Value)>,
    tensors: Vec<TensorInfo>,
    header_written: bool,
    data_pos: u64,
    next_tensor: usize,
}

impl<W: Write> Writer<W> {
    pub fn new(out: W) -> Self {
        Writer {
            out,
            kvs: Vec::new(),
            tensors: Vec::new(),
            header_written: false,
            data_pos: 0,
            next_tensor: 0,
        }
    }

    pub fn add_kv(&mut self, key: &str, value: Value) {
        assert!(!self.header_written, "KV added after header");
        assert!(
            !self.kvs.iter().any(|(k, _)| k == key),
            "duplicated KV key {key:?}"
        );
        self.kvs.push((key.to_string(), value));
    }

    /// gguf-py's add_string drops empty strings; mirror that.
    pub fn add_string(&mut self, key: &str, value: &str) {
        if value.is_empty() {
            return;
        }
        self.add_kv(key, Value::Str(value.to_string()));
    }

    pub fn add_tensor_info(&mut self, name: &str, shape: Vec<u64>, dtype: u32, nbytes: u64) {
        assert!(!self.header_written, "tensor info added after header");
        assert!(
            !self.tensors.iter().any(|t| t.name == name),
            "duplicated tensor name {name:?}"
        );
        self.tensors.push(TensorInfo {
            name: name.to_string(),
            shape,
            dtype,
            nbytes,
        });
    }

    /// Write header, KV section, tensor-info section, and padding up to the
    /// data section start.
    pub fn write_header(&mut self) -> io::Result<()> {
        assert!(!self.header_written);
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&GGUF_VERSION.to_le_bytes());
        buf.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        buf.extend_from_slice(&(self.kvs.len() as u64).to_le_bytes());
        for (k, v) in &self.kvs {
            write_str(&mut buf, k);
            buf.extend_from_slice(&v.type_id().to_le_bytes());
            write_value_payload(&mut buf, v);
        }
        let mut offset = 0u64;
        for t in &self.tensors {
            write_str(&mut buf, &t.name);
            buf.extend_from_slice(&(t.shape.len() as u32).to_le_bytes());
            for d in t.shape.iter().rev() {
                buf.extend_from_slice(&d.to_le_bytes());
            }
            buf.extend_from_slice(&t.dtype.to_le_bytes());
            buf.extend_from_slice(&offset.to_le_bytes());
            offset += ggml_pad(t.nbytes, ALIGNMENT);
        }
        let pad = ggml_pad(buf.len() as u64, ALIGNMENT) - buf.len() as u64;
        buf.extend(std::iter::repeat_n(0u8, pad as usize));
        self.out.write_all(&buf)?;
        self.header_written = true;
        Ok(())
    }

    /// Append one tensor's data (must match the declared nbytes and order).
    pub fn write_tensor_data(&mut self, data: &[u8]) -> io::Result<()> {
        assert!(self.header_written, "header not yet written");
        let ti = &self.tensors[self.next_tensor];
        assert_eq!(
            data.len() as u64,
            ti.nbytes,
            "tensor {} nbytes mismatch",
            ti.name
        );
        self.next_tensor += 1;
        self.out.write_all(data)?;
        self.data_pos += data.len() as u64;
        self.pad_to_alignment()
    }

    /// For chunked emission: declare the start of the next tensor, stream
    /// bytes with `write_chunk`, then `end_tensor`.
    pub fn begin_tensor(&mut self) -> u64 {
        assert!(self.header_written);
        let n = self.tensors[self.next_tensor].nbytes;
        self.next_tensor += 1;
        n
    }

    pub fn write_chunk(&mut self, data: &[u8]) -> io::Result<()> {
        self.out.write_all(data)?;
        self.data_pos += data.len() as u64;
        Ok(())
    }

    pub fn end_tensor(&mut self) -> io::Result<()> {
        self.pad_to_alignment()
    }

    fn pad_to_alignment(&mut self) -> io::Result<()> {
        let pad = ggml_pad(self.data_pos, ALIGNMENT) - self.data_pos;
        if pad > 0 {
            self.out.write_all(&vec![0u8; pad as usize])?;
            self.data_pos += pad;
        }
        Ok(())
    }

    pub fn finish(mut self) -> io::Result<W> {
        assert_eq!(
            self.next_tensor,
            self.tensors.len(),
            "not all tensors written"
        );
        self.out.flush()?;
        Ok(self.out)
    }
}

/// Parsed GGUF file (metadata in file order; data accessed by range).
pub struct Reader {
    pub kvs: Vec<(String, Value)>,
    pub tensors: Vec<ReaderTensor>,
    pub data_start: u64,
}

#[derive(Debug, Clone)]
pub struct ReaderTensor {
    pub name: String,
    /// Shape exactly as stored on disk (GGUF/ggml order, i.e. reversed numpy).
    pub dims: Vec<u64>,
    /// Type code as stored in the file (may be a legacy enum id).
    pub dtype_orig: u32,
    /// Type code resolved to the current enum (see TYPE_CODE_MIGRATIONS).
    pub dtype: u32,
    pub offset: u64,
    pub nbytes: u64,
}

impl Reader {
    pub fn open(path: &std::path::Path) -> io::Result<Reader> {
        let mut f = io::BufReader::new(std::fs::File::open(path)?);
        let mut u32b = [0u8; 4];
        let mut u64b = [0u8; 8];
        let mut rd_u32 = |f: &mut dyn Read| -> io::Result<u32> {
            f.read_exact(&mut u32b)?;
            Ok(u32::from_le_bytes(u32b))
        };
        let mut rd_u64 = |f: &mut dyn Read| -> io::Result<u64> {
            f.read_exact(&mut u64b)?;
            Ok(u64::from_le_bytes(u64b))
        };
        fn rd_str(f: &mut dyn Read) -> io::Result<String> {
            let mut u64b = [0u8; 8];
            f.read_exact(&mut u64b)?;
            let n = u64::from_le_bytes(u64b) as usize;
            let mut buf = vec![0u8; n];
            f.read_exact(&mut buf)?;
            String::from_utf8(buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
        }
        fn rd_scalar(f: &mut dyn Read, t: u32) -> io::Result<Value> {
            let mut b = [0u8; 8];
            Ok(match t {
                V_U8 => {
                    f.read_exact(&mut b[..1])?;
                    Value::U8(b[0])
                }
                V_I8 => {
                    f.read_exact(&mut b[..1])?;
                    Value::I8(b[0] as i8)
                }
                V_U16 => {
                    f.read_exact(&mut b[..2])?;
                    Value::U16(u16::from_le_bytes([b[0], b[1]]))
                }
                V_I16 => {
                    f.read_exact(&mut b[..2])?;
                    Value::I16(i16::from_le_bytes([b[0], b[1]]))
                }
                V_U32 => {
                    f.read_exact(&mut b[..4])?;
                    Value::U32(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                }
                V_I32 => {
                    f.read_exact(&mut b[..4])?;
                    Value::I32(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                }
                V_F32 => {
                    f.read_exact(&mut b[..4])?;
                    Value::F32(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                }
                V_BOOL => {
                    f.read_exact(&mut b[..1])?;
                    Value::Bool(b[0] != 0)
                }
                V_U64 => {
                    f.read_exact(&mut b)?;
                    Value::U64(u64::from_le_bytes(b))
                }
                V_I64 => {
                    f.read_exact(&mut b)?;
                    Value::I64(i64::from_le_bytes(b))
                }
                V_F64 => {
                    f.read_exact(&mut b)?;
                    Value::F64(f64::from_le_bytes(b))
                }
                _ => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "bad scalar type",
                    ))
                }
            })
        }

        let magic = rd_u32(&mut f)?;
        if magic != GGUF_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "not a GGUF file",
            ));
        }
        let version = rd_u32(&mut f)?;
        if version != GGUF_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported GGUF version {version}"),
            ));
        }
        let n_tensors = rd_u64(&mut f)?;
        let n_kv = rd_u64(&mut f)?;

        let mut kvs = Vec::with_capacity(n_kv as usize);
        for _ in 0..n_kv {
            let key = rd_str(&mut f)?;
            let t = rd_u32(&mut f)?;
            let value = if t == V_STRING {
                Value::Str(rd_str(&mut f)?)
            } else if t == V_ARRAY {
                let et = rd_u32(&mut f)?;
                let n = rd_u64(&mut f)? as usize;
                match et {
                    V_STRING => {
                        let mut v = Vec::with_capacity(n);
                        for _ in 0..n {
                            v.push(rd_str(&mut f)?);
                        }
                        Value::ArrStr(v)
                    }
                    V_I32 => {
                        let mut v = Vec::with_capacity(n);
                        for _ in 0..n {
                            if let Value::I32(x) = rd_scalar(&mut f, V_I32)? {
                                v.push(x);
                            }
                        }
                        Value::ArrI32(v)
                    }
                    V_F32 => {
                        let mut v = Vec::with_capacity(n);
                        for _ in 0..n {
                            if let Value::F32(x) = rd_scalar(&mut f, V_F32)? {
                                v.push(x);
                            }
                        }
                        Value::ArrF32(v)
                    }
                    V_BOOL => {
                        let mut v = Vec::with_capacity(n);
                        for _ in 0..n {
                            if let Value::Bool(x) = rd_scalar(&mut f, V_BOOL)? {
                                v.push(x);
                            }
                        }
                        Value::ArrBool(v)
                    }
                    _ => {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("unsupported array element type {et} for {key}"),
                        ))
                    }
                }
            } else {
                rd_scalar(&mut f, t)?
            };
            kvs.push((key, value));
        }

        let mut tensors = Vec::with_capacity(n_tensors as usize);
        for _ in 0..n_tensors {
            let name = rd_str(&mut f)?;
            let n_dims = rd_u32(&mut f)? as usize;
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(rd_u64(&mut f)?);
            }
            let dtype = rd_u32(&mut f)?;
            let offset = rd_u64(&mut f)?;
            let resolved = resolve_type_code(dtype);
            let (bs, ts) = type_sizes(resolved).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unsupported tensor type {resolved} (stored {dtype}) for {name}"),
                )
            })?;
            let n_elems: u64 = dims.iter().product();
            let nbytes = n_elems / bs * ts;
            tensors.push(ReaderTensor {
                name,
                dims,
                dtype_orig: dtype,
                dtype: resolved,
                offset,
                nbytes,
            });
        }
        let pos = f.stream_position()?;
        let data_start = ggml_pad(pos, ALIGNMENT);
        Ok(Reader {
            kvs,
            tensors,
            data_start,
        })
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.kvs.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// Read one tensor's raw bytes.
    pub fn tensor_data(&self, path: &std::path::Path, t: &ReaderTensor) -> io::Result<Vec<u8>> {
        let mut f = std::fs::File::open(path)?;
        f.seek(SeekFrom::Start(self.data_start + t.offset))?;
        let mut buf = vec![0u8; t.nbytes as usize];
        f.read_exact(&mut buf)?;
        Ok(buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writer_reader_roundtrip() {
        let mut w = Writer::new(Vec::new());
        w.add_string("general.architecture", "llama");
        w.add_kv("a.u32", Value::U32(7));
        w.add_kv("a.f32", Value::F32(0.25));
        w.add_kv("a.arr", Value::ArrI32(vec![1, 2, 3]));
        w.add_string("a.empty", ""); // dropped
        w.add_tensor_info("t0", vec![2, 4], T_F32, 32);
        w.add_tensor_info("t1", vec![3], T_F16, 6);
        w.write_header().unwrap();
        w.write_tensor_data(&[1u8; 32]).unwrap();
        w.write_tensor_data(&[2u8; 6]).unwrap();
        let bytes = w.finish().unwrap();

        let dir = std::env::temp_dir().join("gguf-recast-test-rt.gguf");
        std::fs::write(&dir, &bytes).unwrap();
        let r = Reader::open(&dir).unwrap();
        assert_eq!(r.kvs.len(), 4);
        assert_eq!(r.get("a.u32"), Some(&Value::U32(7)));
        assert_eq!(r.tensors.len(), 2);
        assert_eq!(r.tensors[0].dims, vec![4, 2]); // reversed on disk
        assert_eq!(r.tensors[1].offset, 32); // 32-byte tensor needs no padding
        assert_eq!(r.data_start % ALIGNMENT, 0);
        let d = r.tensor_data(&dir, &r.tensors[1]).unwrap();
        assert_eq!(d, vec![2u8; 6]);
        std::fs::remove_file(&dir).unwrap();
    }

    #[test]
    fn padding_math() {
        assert_eq!(ggml_pad(0, 32), 0);
        assert_eq!(ggml_pad(1, 32), 32);
        assert_eq!(ggml_pad(32, 32), 32);
        assert_eq!(ggml_pad(33, 32), 64);
    }
}
