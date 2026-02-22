# execution/adapters/common
from execution.adapters.common.decimals import safe_decimal, require_decimal, round_down
from execution.adapters.common.hashing import payload_digest, stable_hash, fill_key, order_key
from execution.adapters.common.idempotency import make_fill_idem_key, make_order_idem_key
from execution.adapters.common.schema_checks import require_keys, require_non_empty, safe_get, SchemaError
from execution.adapters.common.symbols import normalize_symbol, normalize_side, normalize_order_type, normalize_tif
from execution.adapters.common.time import now_ms, now_utc, ms_to_datetime, coerce_ts_ms
