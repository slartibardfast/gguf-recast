//! Port of gguf-py `Metadata` heuristics (metadata.py) for the pieces the
//! remix exercises: model-card base_model/license, directory-name fallback,
//! generation_config sampling values, and the parameter-count size label.
//! ASCII-only string semantics (the inputs here are ASCII model ids).

/// Python `str.title()` + `islower()` gate from `Metadata.id_to_title`:
/// title-case a word unless it is not all-lowercase or looks like a version.
pub fn id_to_title(s: &str) -> String {
    s.trim()
        .replace('-', " ")
        .split_whitespace()
        .map(|w| {
            if is_lower(w) && !is_version_like(w) {
                title_word(w)
            } else {
                w.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Python str.islower(): at least one cased char, and no uppercase chars.
fn is_lower(w: &str) -> bool {
    let mut has_cased = false;
    for c in w.chars() {
        if c.is_uppercase() {
            return false;
        }
        if c.is_lowercase() {
            has_cased = true;
        }
    }
    has_cased
}

/// re.match(r'^(v\d+(?:\.\d+)*|\d.*)$', w)
fn is_version_like(w: &str) -> bool {
    let b = w.as_bytes();
    if b.first().is_some_and(|c| c.is_ascii_digit()) {
        return true;
    }
    if b.first() == Some(&b'v') {
        let rest = &w[1..];
        if !rest.is_empty() {
            let mut it = rest.split('.');
            if it.all(|p| !p.is_empty() && p.bytes().all(|c| c.is_ascii_digit())) {
                return true;
            }
        }
    }
    false
}

/// Python str.title(): uppercase the first letter of each alphabetic run.
fn title_word(w: &str) -> String {
    let mut out = String::with_capacity(w.len());
    let mut prev_alpha = false;
    for c in w.chars() {
        if c.is_alphabetic() {
            if prev_alpha {
                out.extend(c.to_lowercase());
            } else {
                out.extend(c.to_uppercase());
            }
            prev_alpha = true;
        } else {
            out.push(c);
            prev_alpha = false;
        }
    }
    out
}

#[derive(Debug, Default, PartialEq)]
pub struct IdComponents {
    pub model_full_name: Option<String>,
    pub org: Option<String>,
    pub basename: Option<String>,
    pub finetune: Option<String>,
    pub version: Option<String>,
    pub size_label: Option<String>,
}

/// Port of `Metadata.get_model_id_components` (total_params semantics kept).
pub fn get_model_id_components(model_id: &str, total_params: i64) -> IdComponents {
    if model_id.contains(' ') {
        return IdComponents {
            model_full_name: Some(model_id.to_string()),
            ..Default::default()
        };
    }
    let (org, full) = match model_id.split_once('/') {
        Some((o, rest)) => (Some(o.to_string()), rest.to_string()),
        None => (None, model_id.to_string()),
    };
    let org = org.filter(|o| !o.starts_with('.') && !o.is_empty());

    let mut parts: Vec<String> = full
        .split('-')
        .filter(|p| !p.is_empty())
        .map(String::from)
        .collect();

    #[derive(Default, Clone, PartialEq)]
    struct Ann {
        version: bool,
        typ: bool,
        size_label: bool,
        finetune: bool,
        basename: bool,
    }
    impl Ann {
        fn n(&self) -> usize {
            [
                self.version,
                self.typ,
                self.size_label,
                self.finetune,
                self.basename,
            ]
            .iter()
            .filter(|b| **b)
            .count()
        }
    }
    let mut ann = vec![Ann::default(); parts.len()];

    for i in 0..parts.len() {
        let part = parts[i].clone();
        if fullmatch_version(&part) {
            ann[i].version = true;
        } else if fullmatch_quant_type(&part) {
            ann[i].typ = true;
            parts[i] = part.to_uppercase();
        } else if i > 0 && fullmatch_size(&part) {
            let mut p = part.replace('_', ".");
            let pb: Vec<char> = p.chars().collect();
            if pb.last().is_some_and(|c| c.is_ascii_digit()) {
                // weird bloom-7b1 notation
                let n = pb.len();
                p = format!(
                    "{}.{}{}",
                    pb[..n - 2].iter().collect::<String>(),
                    pb[n - 1],
                    pb[n - 2]
                );
            }
            let pb: Vec<char> = p.chars().collect();
            if pb.len() > 1 && pb[pb.len() - 2].is_ascii_digit() {
                let last = pb[pb.len() - 1];
                if "kmbt".contains(last) {
                    p = format!(
                        "{}{}",
                        pb[..pb.len() - 1].iter().collect::<String>(),
                        last.to_ascii_uppercase()
                    );
                }
            }
            if total_params != 0 {
                let pb: Vec<char> = p.chars().collect();
                let suffix = *pb.last().unwrap();
                let num: Result<f64, _> = pb[..pb.len() - 1].iter().collect::<String>().parse();
                if let Ok(num) = num {
                    let pow = " KMBT".find(suffix).map(|i| 1000f64.powi(i as i32));
                    if let Some(scale) = pow {
                        let label_params = num * scale;
                        let tp = total_params;
                        let too_far = (tp < 0 && (label_params as i64) < tp.abs() / 8)
                            || (tp > 0 && ((label_params as i64) - tp).abs() > 7 * tp / 8);
                        if too_far {
                            ann[i].finetune = true;
                            p = format!(
                                "{}{}",
                                pb[..pb.len() - 1].iter().collect::<String>(),
                                suffix.to_ascii_lowercase()
                            );
                        }
                    }
                }
            }
            if ann[i].n() == 0 {
                ann[i].size_label = true;
            }
            parts[i] = p;
        } else if i > 0 && fullmatch_finetune_word(&part) {
            if total_params < 0 && part.to_lowercase() == "lora" {
                ann[i].typ = true;
            } else {
                ann[i].finetune = true;
            }
        }
    }

    // Ignore word-based size labels when a number-based one exists.
    let any_numeric_size = parts
        .iter()
        .zip(&ann)
        .any(|(p, a)| a.size_label && p.chars().any(|c| c.is_ascii_digit()));
    if any_numeric_size {
        for (p, a) in parts.iter().zip(ann.iter_mut()) {
            if a.size_label && p.chars().all(|c| c.is_alphabetic()) {
                a.size_label = false;
            }
        }
    }

    let mut at_start = true;
    for (p, a) in parts.iter().zip(ann.iter_mut()) {
        let first_alpha = p.chars().next().is_some_and(|c| c.is_alphabetic());
        if at_start && ((a.n() == 0 && first_alpha) || a.version) {
            a.basename = true;
        } else {
            if at_start {
                at_start = false;
            }
            if a.n() == 0 {
                a.finetune = true;
            }
        }
    }
    // Remove basename annotation from trailing versions.
    for a in ann.iter_mut().rev() {
        if a.basename && a.n() > 1 {
            a.basename = false;
        } else {
            break;
        }
    }

    let join = |f: &dyn Fn(&Ann) -> bool| -> Option<String> {
        let v: Vec<&str> = parts
            .iter()
            .zip(&ann)
            .filter(|(_, a)| f(a))
            .map(|(p, _)| p.as_str())
            .collect();
        if v.is_empty() {
            None
        } else {
            Some(v.join("-"))
        }
    };
    let mut basename = join(&|a: &Ann| a.basename);
    // size labels deduplicated preserving order
    let size_label = {
        let mut seen = Vec::new();
        for (p, a) in parts.iter().zip(&ann) {
            if a.size_label && !seen.contains(p) {
                seen.push(p.clone());
            }
        }
        if seen.is_empty() {
            None
        } else {
            Some(seen.join("-"))
        }
    };
    let finetune = join(&|a: &Ann| a.finetune);
    let version = join(&|a: &Ann| a.version && !a.basename);

    if size_label.is_none() && finetune.is_none() && version.is_none() {
        basename = None;
    }

    IdComponents {
        model_full_name: Some(full),
        org,
        basename,
        finetune,
        version,
        size_label,
    }
}

/// re.fullmatch(r'(v|iter)?\d+([.]\d+)*', part, IGNORECASE)
fn fullmatch_version(part: &str) -> bool {
    let lower = part.to_ascii_lowercase();
    let rest = if let Some(r) = lower.strip_prefix("iter") {
        r
    } else if let Some(r) = lower.strip_prefix('v') {
        r
    } else {
        lower.as_str()
    };
    if rest.is_empty() {
        return false;
    }
    rest.split('.')
        .all(|g| !g.is_empty() && g.bytes().all(|c| c.is_ascii_digit()))
}

/// re.fullmatch(r'i?q\d(_\w)*|b?fp?(16|32)', part, IGNORECASE)
fn fullmatch_quant_type(part: &str) -> bool {
    let p = part.to_ascii_lowercase();
    // i?q\d(_\w)*
    let q = p.strip_prefix('i').unwrap_or(&p);
    if let Some(rest) = q.strip_prefix('q') {
        let b = rest.as_bytes();
        if !b.is_empty() && b[0].is_ascii_digit() {
            let mut i = 1;
            let bytes = rest.as_bytes();
            let mut ok = true;
            while i < bytes.len() {
                // each extra chunk is _\w (exactly one word char per underscore)
                if bytes[i] == b'_'
                    && i + 1 < bytes.len()
                    && (bytes[i + 1].is_ascii_alphanumeric() || bytes[i + 1] == b'_')
                {
                    i += 2;
                } else {
                    ok = false;
                    break;
                }
            }
            if ok {
                return true;
            }
        }
    }
    // b?fp?(16|32)
    let f = p.strip_prefix('b').unwrap_or(&p);
    if let Some(rest) = f.strip_prefix('f') {
        let rest = rest.strip_prefix('p').unwrap_or(rest);
        if rest == "16" || rest == "32" {
            return true;
        }
    }
    false
}

/// re.fullmatch(r'(([A]|\d+[x])?\d+([._]\d+)?[KMBT][\d]?|small|mini|medium|large|x?xl)', IGNORECASE)
fn fullmatch_size(part: &str) -> bool {
    let p = part.to_ascii_lowercase();
    if ["small", "mini", "medium", "large", "xl", "xxl"].contains(&p.as_str()) {
        return true;
    }
    let b = p.as_bytes();
    let mut i = 0;
    // optional: 'a' or \d+x
    if b.first() == Some(&b'a') && b.len() > 1 && !b[1].is_ascii_digit() {
        return false; // 'a' must be followed by the numeric body; handled below
    }
    if b.first() == Some(&b'a') {
        i = 1;
    } else {
        let mut j = 0;
        while j < b.len() && b[j].is_ascii_digit() {
            j += 1;
        }
        if j > 0 && j < b.len() && b[j] == b'x' {
            i = j + 1;
        }
    }
    // \d+
    let start = i;
    while i < b.len() && b[i].is_ascii_digit() {
        i += 1;
    }
    if i == start {
        return false;
    }
    // ([._]\d+)?
    if i < b.len() && (b[i] == b'.' || b[i] == b'_') {
        let s2 = i + 1;
        let mut j = s2;
        while j < b.len() && b[j].is_ascii_digit() {
            j += 1;
        }
        if j == s2 {
            return false;
        }
        i = j;
    }
    // [kmbt]
    if i >= b.len() || !b"kmbt".contains(&b[i]) {
        return false;
    }
    i += 1;
    // [\d]?
    if i < b.len() && b[i].is_ascii_digit() {
        i += 1;
    }
    i == b.len()
}

/// re.fullmatch(r'chat|instruct|vision|lora', IGNORECASE)
fn fullmatch_finetune_word(part: &str) -> bool {
    let p = part.to_ascii_lowercase();
    ["chat", "instruct", "vision", "lora"].contains(&p.as_str())
}

/// Port of gguf-py utility.model_weight_count_rounded_notation + size_label
/// (no experts in this model family path).
pub fn size_label_from_params(total_params: u64) -> String {
    let n = total_params as f64;
    let (scaled, suffix) = if n > 1e12 {
        (n * 1e-12, "T")
    } else if n > 1e9 {
        (n * 1e-9, "B")
    } else if n > 1e6 {
        (n * 1e-6, "M")
    } else {
        (n * 1e-3, "K")
    };
    // fix = max(min_digits - len(str(round(scaled)).lstrip('0')), 0)
    let rounded = round_half_even(scaled);
    let digits = format!("{}", rounded as i64).trim_start_matches('0').len();
    let fix = 2usize.saturating_sub(digits);
    format!("{scaled:.fix$}{suffix}")
}

/// Python round() (banker's rounding) for positive values.
fn round_half_even(x: f64) -> f64 {
    let f = x.floor();
    let diff = x - f;
    let round_up = diff > 0.5 || (diff == 0.5 && (f as i64) % 2 != 0);
    if round_up {
        f + 1.0
    } else {
        f
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn title_of_snapshot_sha() {
        assert_eq!(
            id_to_title("abc86de19eb1ebbf6a7df4582341325c22ddcb7d"),
            "Abc86De19Eb1Ebbf6A7Df4582341325C22Ddcb7D"
        );
    }

    #[test]
    fn title_keeps_mixed_case_and_versions() {
        assert_eq!(id_to_title("Qwen3.6-27B"), "Qwen3.6 27B");
        assert_eq!(id_to_title("v1.5-chat"), "v1.5 Chat");
    }

    #[test]
    fn qwen_base_model_components() {
        let c = get_model_id_components("Qwen/Qwen3.6-27B", 28_000_000_000);
        assert_eq!(c.model_full_name.as_deref(), Some("Qwen3.6-27B"));
        assert_eq!(c.org.as_deref(), Some("Qwen"));
        assert_eq!(c.basename.as_deref(), Some("Qwen3.6"));
        assert_eq!(c.size_label.as_deref(), Some("27B"));
        assert_eq!(c.finetune, None);
        assert_eq!(c.version, None);
    }

    #[test]
    fn sha_dirname_components_too_ambiguous() {
        let c = get_model_id_components("abc86de19eb1ebbf6a7df4582341325c22ddcb7d", 28_000_000_000);
        assert_eq!(
            c.model_full_name.as_deref(),
            Some("abc86de19eb1ebbf6a7df4582341325c22ddcb7d")
        );
        assert_eq!(c.org, None);
        assert_eq!(c.basename, None); // no size/finetune/version -> ambiguous
        assert_eq!(c.size_label, None);
    }

    #[test]
    fn size_labels() {
        assert_eq!(size_label_from_params(27_000_000_000), "27B");
        assert_eq!(size_label_from_params(6_500_000_000), "6.5B");
        assert_eq!(size_label_from_params(125_000_000), "125M");
    }
}
