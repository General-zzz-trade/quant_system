//! Fixed-point decimal with 8 decimal places (×10^8).
//!
//! Covers crypto price/qty precision (max 8 decimals on exchanges).
//! Range: -92,233,720,368.54775807 to 92,233,720,368.54775807
//! Internal: i64 raw value. Arithmetic uses i128 intermediates to prevent overflow.

use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

pub const SCALE: i64 = 100_000_000; // 10^8

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Fd8(pub i64);

impl Fd8 {
    pub const ZERO: Self = Self(0);

    #[inline]
    pub fn raw(self) -> i64 {
        self.0
    }

    #[inline]
    pub fn from_raw(raw: i64) -> Self {
        Self(raw)
    }

    #[inline]
    pub fn from_f64(v: f64) -> Self {
        Self((v * SCALE as f64).round() as i64)
    }

    #[inline]
    pub fn to_f64(self) -> f64 {
        self.0 as f64 / SCALE as f64
    }

    pub fn from_str_opt(s: &str) -> Option<Self> {
        s.parse::<f64>().ok().map(Self::from_f64)
    }

    #[inline]
    pub fn abs(self) -> Self {
        Self(self.0.abs())
    }

    #[inline]
    pub fn is_zero(self) -> bool {
        self.0 == 0
    }

    #[inline]
    pub fn is_positive(self) -> bool {
        self.0 > 0
    }

    #[inline]
    pub fn is_negative(self) -> bool {
        self.0 < 0
    }

    /// Format as decimal string, stripping trailing zeros.
    pub fn to_string_stripped(self) -> String {
        if self.0 == 0 {
            return "0".to_string();
        }
        let sign = if self.0 < 0 { "-" } else { "" };
        let abs_val = self.0.unsigned_abs() as u64;
        let scale = SCALE as u64;
        let integer = abs_val / scale;
        let frac = abs_val % scale;

        if frac == 0 {
            return format!("{}{}", sign, integer);
        }

        let frac_str = format!("{:08}", frac);
        let trimmed = frac_str.trim_end_matches('0');
        format!("{}{}.{}", sign, integer, trimmed)
    }
}

impl Add for Fd8 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl Sub for Fd8 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl Mul for Fd8 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let result = (self.0 as i128 * rhs.0 as i128) / SCALE as i128;
        Self(result as i64)
    }
}

impl Div for Fd8 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        if rhs.0 == 0 {
            return Self(0);
        }
        let result = (self.0 as i128 * SCALE as i128) / rhs.0 as i128;
        Self(result as i64)
    }
}

impl Neg for Fd8 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

impl fmt::Display for Fd8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_string_stripped())
    }
}

impl fmt::Debug for Fd8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Fd8({})", self.to_string_stripped())
    }
}

// ---------------------------------------------------------------------------
// PyO3 helper: extract from Python value (Decimal, float, int, str)
// ---------------------------------------------------------------------------
use pyo3::prelude::*;

pub fn fd8_from_pyany(val: &Bound<'_, PyAny>) -> PyResult<Fd8> {
    // Fast path: already an i64 (our own output)
    if let Ok(i) = val.extract::<i64>() {
        return Ok(Fd8::from_raw(i));
    }
    // Try f64 (int, float)
    if let Ok(f) = val.extract::<f64>() {
        return Ok(Fd8::from_f64(f));
    }
    // Fall back to str (Decimal, other)
    let s = val.str()?.to_string();
    Fd8::from_str_opt(&s).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!("cannot convert to Fd8: {}", s))
    })
}

pub fn opt_fd8_from_pyany(val: &Bound<'_, PyAny>, attr: &str) -> PyResult<Option<Fd8>> {
    match val.getattr(attr) {
        Ok(v) if !v.is_none() => Ok(Some(fd8_from_pyany(&v)?)),
        _ => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_arithmetic() {
        let a = Fd8::from_f64(1.5);
        let b = Fd8::from_f64(2.25);
        assert_eq!((a + b).to_f64(), 3.75);
        assert_eq!((b - a).to_f64(), 0.75);
    }

    #[test]
    fn test_multiplication() {
        let qty = Fd8::from_f64(0.5);
        let price = Fd8::from_f64(50000.0);
        let notional = qty * price;
        assert_eq!(notional.to_f64(), 25000.0);
    }

    #[test]
    fn test_division() {
        let a = Fd8::from_f64(100.0);
        let b = Fd8::from_f64(3.0);
        let result = a / b;
        assert!((result.to_f64() - 33.33333333).abs() < 0.00000002);
    }

    #[test]
    fn test_string_format() {
        assert_eq!(Fd8::from_f64(1.23).to_string_stripped(), "1.23");
        assert_eq!(Fd8::from_f64(0.0).to_string_stripped(), "0");
        assert_eq!(Fd8::from_f64(42500.0).to_string_stripped(), "42500");
        assert_eq!(Fd8::from_f64(-1.5).to_string_stripped(), "-1.5");
        assert_eq!(Fd8::from_f64(0.00000001).to_string_stripped(), "0.00000001");
    }

    #[test]
    fn test_zero_division() {
        let a = Fd8::from_f64(100.0);
        let b = Fd8::ZERO;
        assert_eq!((a / b).raw(), 0);
    }

    #[test]
    fn test_abs() {
        assert_eq!(Fd8::from_f64(-5.0).abs(), Fd8::from_f64(5.0));
        assert_eq!(Fd8::from_f64(5.0).abs(), Fd8::from_f64(5.0));
    }

    #[test]
    fn test_large_values() {
        // BTC price * large qty shouldn't overflow
        let price = Fd8::from_f64(90000.0);
        let qty = Fd8::from_f64(100.0);
        let notional = price * qty;
        assert_eq!(notional.to_f64(), 9_000_000.0);
    }
}
