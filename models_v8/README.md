# Model Directory

- `config.json`: version-controlled (model parameters + rebuild instructions)
- `.pkl` files: gitignored (machine-specific, too large)

## Rebuild

```bash
python3 -m alpha.auto_retrain --force              # All 1h models
python3 -m alpha.auto_retrain --only-4h --force     # 4h models only
python3 -m alpha.retrain_15m --force                # 15m models only
```

## Status

| Model | Status | WF Sharpe |
|-------|--------|-----------|
| BTCUSDT_gate_v2 (1h) | ACTIVE | 2.34 |
| ETHUSDT_gate_v2 (1h) | ACTIVE | 3.92 |
| BTCUSDT_4h | ACTIVE (retrain on fresh deploy) | 3.62 |
| ETHUSDT_4h | ACTIVE (retrain on fresh deploy) | 4.57 |
| BTCUSDT_15m | DISABLED (WF FAIL) | 0.27 |
| ETHUSDT_15m | DISABLED (WF FAIL) | -1.36 |
