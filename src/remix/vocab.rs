//! GPT-2 style vocab extraction from tokenizer.json, mirroring the byte
//! behavior of the converter's `get_vocab_base` + `SpecialVocab` for
//! fast-tokenizer checkpoints (added tokens are matched whole by the
//! tokenizer, so the "normalize non-normalized added tokens" encode/decode
//! round trip is the identity and is not re-run here).

use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;

pub const TOKTYPE_NORMAL: i32 = 1;
pub const TOKTYPE_CONTROL: i32 = 3;
pub const TOKTYPE_USER_DEFINED: i32 = 4;
pub const TOKTYPE_UNUSED: i32 = 5;

#[derive(Deserialize)]
struct AddedToken {
    id: u32,
    content: String,
    #[serde(default)]
    special: bool,
}

#[derive(Deserialize)]
struct TokenizerModel {
    vocab: HashMap<String, u32>,
    merges: Vec<serde_json::Value>,
}

#[derive(Deserialize)]
struct TokenizerJson {
    #[serde(default)]
    added_tokens: Vec<AddedToken>,
    model: TokenizerModel,
}

pub struct Vocab {
    pub tokens: Vec<String>,
    pub token_types: Vec<i32>,
    pub merges: Vec<String>,
}

/// `TextModel.does_token_look_special`
fn looks_special(token: &str) -> bool {
    matches!(token, "<pad>" | "<mask>" | "<2mass>" | "[@BOS@]")
        || (token.starts_with("<|") && token.ends_with("|>"))
        || (token.starts_with("<\u{ff5c}") && token.ends_with("\u{ff5c}>"))
        || (token.starts_with("<unused") && token.ends_with('>'))
}

pub fn load_vocab(model_dir: &Path, vocab_size: usize) -> std::io::Result<Vocab> {
    let raw = std::fs::read(model_dir.join("tokenizer.json"))?;
    let tj: TokenizerJson = serde_json::from_slice(&raw)?;

    let mut id_to_token: HashMap<u32, &String> = HashMap::with_capacity(tj.model.vocab.len());
    for (tok, id) in &tj.model.vocab {
        id_to_token.insert(*id, tok);
    }
    let added: HashMap<u32, &AddedToken> = tj.added_tokens.iter().map(|a| (a.id, a)).collect();

    let mut tokens = Vec::with_capacity(vocab_size);
    let mut token_types = Vec::with_capacity(vocab_size);
    for i in 0..vocab_size as u32 {
        if let Some(a) = added.get(&i) {
            if a.special || looks_special(&a.content) {
                tokens.push(a.content.clone());
                token_types.push(TOKTYPE_CONTROL);
            } else {
                // pre-normalize user-defined spaces (Gemma-style ▁)
                tokens.push(a.content.replace('\u{2581}', " "));
                token_types.push(TOKTYPE_USER_DEFINED);
            }
        } else if let Some(tok) = id_to_token.get(&i) {
            tokens.push((*tok).clone());
            token_types.push(TOKTYPE_NORMAL);
        } else {
            tokens.push(format!("[PAD{i}]"));
            token_types.push(TOKTYPE_UNUSED);
        }
    }

    // Merges: either legacy "a b" strings or pair arrays (transformers >=4.45).
    // Pair form joins with ' '; literal spaces inside parts are encoded as
    // chr(ord(' ') + 256), mirroring SpecialVocab.
    let mut merges = Vec::with_capacity(tj.model.merges.len());
    for m in &tj.model.merges {
        match m {
            serde_json::Value::String(s) => merges.push(s.clone()),
            serde_json::Value::Array(pair) if pair.len() == 2 => {
                let enc = |v: &serde_json::Value| -> String {
                    v.as_str()
                        .unwrap_or_default()
                        .chars()
                        .map(|c| if c == ' ' { '\u{120}' } else { c })
                        .collect()
                };
                merges.push(format!("{} {}", enc(&pair[0]), enc(&pair[1])));
            }
            _ => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "unknown tokenizer merges format",
                ))
            }
        }
    }

    Ok(Vocab {
        tokens,
        token_types,
        merges,
    })
}

/// Special token ids in gguf-py `SpecialVocab` discovery order:
/// tokenizer_config.json `{typ}_token` entries first (matched against
/// added_tokens contents), then config.json `{typ}_token_id` (root, falling
/// back to text_config). First assignment per type wins.
pub fn special_token_ids(model_dir: &Path) -> std::io::Result<Vec<(&'static str, u32)>> {
    const TYPES: [&str; 7] = ["bos", "eos", "unk", "sep", "pad", "cls", "mask"];

    let tokenizer: serde_json::Value =
        serde_json::from_slice(&std::fs::read(model_dir.join("tokenizer.json"))?)?;
    let added = tokenizer["added_tokens"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let tok_config: Option<serde_json::Value> =
        std::fs::read(model_dir.join("tokenizer_config.json"))
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok());

    let mut ids: Vec<(&'static str, u32)> = Vec::new();
    let set = |typ: &'static str, id: Option<u32>, ids: &mut Vec<(&'static str, u32)>| {
        if let Some(id) = id {
            if !ids.iter().any(|(t, _)| *t == typ) {
                ids.push((typ, id));
            }
        }
    };

    if let Some(tc) = &tok_config {
        for typ in TYPES {
            let entry = &tc[format!("{typ}_token")];
            let content = if let Some(s) = entry.as_str() {
                Some(s.to_string())
            } else {
                entry["content"].as_str().map(String::from)
            };
            if let Some(content) = content {
                let id = added
                    .iter()
                    .find(|a| a["content"].as_str() == Some(content.as_str()))
                    .and_then(|a| a["id"].as_u64())
                    .map(|x| x as u32);
                set(typ, id, &mut ids);
            }
        }
    }

    if let Ok(cfgb) = std::fs::read(model_dir.join("config.json")) {
        let cfg: serde_json::Value = serde_json::from_slice(&cfgb)?;
        for typ in TYPES {
            let key = format!("{typ}_token_id");
            let mut id = cfg[&key].as_u64();
            if id.is_none() {
                id = cfg["text_config"][&key].as_u64();
            }
            set(typ, id.map(|x| x as u32), &mut ids);
        }
    }
    Ok(ids)
}

/// Chat template resolution order: tokenizer_config `chat_template`, else
/// chat_template.jinja, else chat_template.json.
pub fn chat_template(model_dir: &Path) -> std::io::Result<Option<String>> {
    let tok_config: Option<serde_json::Value> =
        std::fs::read(model_dir.join("tokenizer_config.json"))
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok());
    // SpecialVocab only reads templates when tokenizer_config.json exists.
    let Some(tc) = tok_config else {
        return Ok(None);
    };
    if let Some(s) = tc["chat_template"].as_str() {
        return Ok(Some(s.to_string()));
    }
    if tc.get("chat_template").is_some_and(|v| !v.is_null()) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "non-string chat_template in tokenizer_config.json is not supported",
        ));
    }
    let jinja = model_dir.join("chat_template.jinja");
    if jinja.is_file() {
        if model_dir.join("additional_chat_templates").is_dir() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "additional_chat_templates/ is not supported",
            ));
        }
        return Ok(Some(std::fs::read_to_string(jinja)?));
    }
    let ctj = model_dir.join("chat_template.json");
    if ctj.is_file() {
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(ctj)?)?;
        return Ok(v["chat_template"].as_str().map(String::from));
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn special_looking_tokens() {
        assert!(looks_special("<|fim_prefix|>"));
        assert!(looks_special("<pad>"));
        assert!(looks_special("<unused42>"));
        assert!(!looks_special("<tool_call>"));
        assert!(!looks_special("<think>"));
    }
}
