use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::fixed_decimal::Fd8;

#[pyfunction]
#[pyo3(signature = (target_qty, current_qty="0"))]
pub fn rust_build_delta_order_fields(
    target_qty: &str,
    current_qty: &str,
) -> PyResult<Option<(String, String)>> {
    let target = parse_fd8("target_qty", target_qty)?;
    let current = parse_fd8("current_qty", current_qty)?;
    let delta = target - current;
    if delta.is_zero() {
        return Ok(None);
    }
    let side = if delta.is_positive() { "buy" } else { "sell" }.to_string();
    Ok(Some((side, delta.abs().to_string_stripped())))
}

#[pyfunction]
#[pyo3(signature = (side, reference_price, bps, aggressive=true))]
pub fn rust_limit_price(
    side: &str,
    reference_price: &str,
    bps: &str,
    aggressive: bool,
) -> PyResult<String> {
    let px = parse_fd8("reference_price", reference_price)?;
    let bps_v = parse_fd8("bps", bps)?;
    let ratio = bps_v / Fd8::from_f64(10_000.0);
    let one = Fd8::from_f64(1.0);

    let factor = match (side.to_ascii_lowercase().as_str(), aggressive) {
        ("buy", true) => one + ratio,
        ("sell", true) => one - ratio,
        ("buy", false) => one - ratio,
        ("sell", false) => one + ratio,
        _ => {
            return Err(PyValueError::new_err(format!(
                "invalid side '{}', must be buy or sell",
                side
            )))
        }
    };

    Ok((px * factor).to_string_stripped())
}

/// Float-based limit price (avoids Decimal→str→Fd8 round-trip).
#[pyfunction]
#[pyo3(signature = (side, reference_price, bps, aggressive=true))]
pub fn rust_limit_price_f64(
    side: &str,
    reference_price: f64,
    bps: f64,
    aggressive: bool,
) -> PyResult<f64> {
    let ratio = bps / 10_000.0;
    let factor = match (side.to_ascii_lowercase().as_str(), aggressive) {
        ("buy", true) => 1.0 + ratio,
        ("sell", true) => 1.0 - ratio,
        ("buy", false) => 1.0 - ratio,
        ("sell", false) => 1.0 + ratio,
        _ => {
            return Err(PyValueError::new_err(format!(
                "invalid side '{}', must be buy or sell", side
            )))
        }
    };
    Ok(reference_price * factor)
}

#[pyfunction]
#[pyo3(signature = (qty, price=None, price_hint=None, min_qty="0", min_notional="0"))]
pub fn rust_validate_order_constraints(
    qty: &str,
    price: Option<&str>,
    price_hint: Option<&str>,
    min_qty: &str,
    min_notional: &str,
) -> PyResult<Option<String>> {
    let qty_v = parse_fd8("qty", qty)?;
    if qty_v <= Fd8::ZERO {
        return Ok(Some("order.qty must be > 0".to_string()));
    }

    let min_qty_v = parse_fd8("min_qty", min_qty)?;
    if !min_qty_v.is_zero() && qty_v < min_qty_v {
        return Ok(Some(format!("order.qty < min_qty ({} < {})", qty_v, min_qty_v)));
    }

    let p = match (price, price_hint) {
        (Some(px), _) => Some(parse_fd8("price", px)?),
        (None, Some(hint)) => Some(parse_fd8("price_hint", hint)?),
        (None, None) => None,
    };
    if let Some(px) = p {
        if px <= Fd8::ZERO {
            return Ok(Some("order.price must be > 0 when provided".to_string()));
        }
        let min_notional_v = parse_fd8("min_notional", min_notional)?;
        if !min_notional_v.is_zero() && (qty_v * px) < min_notional_v {
            return Ok(Some("order.notional below min_notional".to_string()));
        }
    }

    Ok(None)
}

fn parse_fd8(name: &str, raw: &str) -> PyResult<Fd8> {
    Fd8::from_str_opt(raw).ok_or_else(|| {
        PyValueError::new_err(format!("invalid decimal for {}: {}", name, raw))
    })
}
