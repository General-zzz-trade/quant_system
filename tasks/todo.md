# P1: 内核 Python → Rust PyO3 迁移 — COMPLETED

## Sprint A: state/ 层 — 数据结构 + reducers [DONE]
- [x] A1: `state_types.rs` (769 LOC) — RustMarketState, RustPositionState, RustAccountState, RustPortfolioState, RustRiskState, RustRiskLimits, RustReducerResult
- [x] A2: `state_reducers.rs` (953 LOC) — RustMarketReducer, RustPositionReducer, RustAccountReducer, RustPortfolioReducer, RustRiskReducer
- [x] A3: Registered in lib.rs
- [x] A4: Build + 2569 tests pass

## Sprint B: engine/pipeline + guards [DONE]
- [x] B1: `pipeline.rs` (249 LOC) — rust_detect_event_kind, rust_normalize_to_facts
- [x] B2: Dispatcher routing already in `route_match.rs` (existing)
- [x] B3: `engine_guards.rs` (219 LOC) — RustGuardConfig, RustBasicGuard
- [x] B4: Build + tests pass

## Sprint C: risk/ + execution/safety/ + core/ [DONE]
- [x] C1: `risk_engine.rs` (635 LOC) — RustKillSwitch, RustCircuitBreaker, RustOrderLimiter, RustRiskGate
- [x] C2: `core_types.rs` (254 LOC) — RustInterceptorChain, RustSystemClock, RustSimulatedClock, RustTradingGate
- [x] C3: Build + tests pass

## Sprint D: event/ types + validators [DONE]
- [x] D1: `event_types.rs` (340 LOC) — enums + RustSignalResult, RustDecisionOutput, RustTargetPosition, RustOrderSpec
- [x] D2: `event_validators.rs` (212 LOC) — 10 validation functions
- [x] D3: Build + 2615 tests pass

## Summary
| Sprint | New Rust Files | New LOC | Exports |
|--------|---------------|---------|---------|
| A | state_types.rs, state_reducers.rs | 1,722 | 7 cls + 5 cls |
| B | pipeline.rs, engine_guards.rs | 468 | 2 fn + 2 cls |
| C | risk_engine.rs, core_types.rs | 889 | 4 cls + 4 cls |
| D | event_types.rs, event_validators.rs | 552 | 4 cls + 6 fn + 10 fn |
| **Total** | **8 files** | **3,631** | **26 cls + 18 fn** |

Combined with existing crate: **11,480 total Rust LOC** in `_quant_hotpath`.
Tests: **2615 passing**.
