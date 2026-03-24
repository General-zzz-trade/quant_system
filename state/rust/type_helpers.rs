use crate::state::fixed_decimal::Fd8;

/// Compare two Option<String> for equality.
pub fn opt_str_eq(a: &Option<String>, b: &Option<String>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => a == b,
        (None, None) => true,
        _ => false,
    }
}

pub fn opt_str_repr(v: &Option<String>) -> String {
    match v {
        Some(s) => format!("'{}'", s),
        None => "None".to_string(),
    }
}

pub fn opt_i64_repr(v: &Option<i64>) -> String {
    match v {
        Some(raw) => format!("'{}'", Fd8::from_raw(*raw).to_string_stripped()),
        None => "None".to_string(),
    }
}

pub fn i64_repr(raw: i64) -> String {
    Fd8::from_raw(raw).to_string_stripped()
}
