// saga_types.inc.rs — SagaState enum, internal types, and transition validator.
// Included by saga.rs via include!().

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum SagaState {
    Pending,
    Submitted,
    Acked,
    PartialFill,
    Filled,
    Rejected,
    Cancelled,
    Expired,
    Compensating,
    Compensated,
    Failed,
}

impl SagaState {
    /// Terminal states match Python's TERMINAL_STATES:
    /// FILLED, COMPENSATED, FAILED only.
    /// REJECTED/CANCELLED/EXPIRED can still transition to COMPENSATING.
    fn is_terminal(&self) -> bool {
        matches!(
            self,
            Self::Filled | Self::Compensated | Self::Failed
        )
    }

    /// Valid transitions matching Python _TRANSITIONS exactly.
    fn valid_transitions(&self) -> &'static [SagaState] {
        match self {
            Self::Pending => &[Self::Submitted, Self::Rejected, Self::Cancelled],
            Self::Submitted => &[
                Self::Acked,
                Self::Filled,
                Self::Rejected,
                Self::Cancelled,
                Self::Compensating,
            ],
            Self::Acked => &[
                Self::PartialFill,
                Self::Filled,
                Self::Cancelled,
                Self::Expired,
                Self::Compensating,
            ],
            Self::PartialFill => &[
                Self::PartialFill,
                Self::Filled,
                Self::Cancelled,
                Self::Expired,
                Self::Compensating,
            ],
            Self::Filled => &[],
            Self::Rejected => &[Self::Compensating],
            Self::Cancelled => &[Self::Compensating],
            Self::Expired => &[Self::Compensating],
            Self::Compensating => &[Self::Compensated, Self::Failed],
            Self::Compensated => &[],
            Self::Failed => &[],
        }
    }

    fn from_str(s: &str) -> PyResult<Self> {
        match s.to_lowercase().as_str() {
            "pending" => Ok(Self::Pending),
            "submitted" => Ok(Self::Submitted),
            "acked" => Ok(Self::Acked),
            "partial_fill" | "partialfill" => Ok(Self::PartialFill),
            "filled" => Ok(Self::Filled),
            "rejected" => Ok(Self::Rejected),
            "cancelled" | "canceled" => Ok(Self::Cancelled),
            "expired" => Ok(Self::Expired),
            "compensating" => Ok(Self::Compensating),
            "compensated" => Ok(Self::Compensated),
            "failed" => Ok(Self::Failed),
            _ => Err(PyValueError::new_err(format!(
                "unknown saga state: {}",
                s
            ))),
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Submitted => "submitted",
            Self::Acked => "acked",
            Self::PartialFill => "partial_fill",
            Self::Filled => "filled",
            Self::Rejected => "rejected",
            Self::Cancelled => "cancelled",
            Self::Expired => "expired",
            Self::Compensating => "compensating",
            Self::Compensated => "compensated",
            Self::Failed => "failed",
        }
    }
}

// ============================================================
// Internal types (no PyO3 dependency)
// ============================================================

#[derive(Clone, Debug)]
struct SagaTransition {
    from: SagaState,
    to: SagaState,
    reason: String,
    timestamp: f64,
}

struct OrderSaga {
    order_id: String,
    intent_id: String,
    state: SagaState,
    symbol: String,
    side: String,
    qty: f64,
    filled_qty: f64,
    avg_fill_price: f64,
    fill_count: u32,
    created_at: f64,
    submitted_at: Option<f64>,
    last_transition_at: f64,
    ttl_sec: f64,
    history: Vec<SagaTransition>,
    meta: HashMap<String, String>,
}

// ============================================================
// Helper: check if transition is valid
// ============================================================

fn is_valid_transition(from: SagaState, to: SagaState) -> bool {
    from.valid_transitions().contains(&to)
}
