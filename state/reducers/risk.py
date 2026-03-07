# DEPRECATED: Superseded by RustRiskEvaluator. Retained for parity tests.
from __future__ import annotations

from decimal import Decimal
from typing import Any, Callable, Mapping

from state.portfolio import PortfolioState
from state.risk import RiskLimits, RiskState
from state.position import PositionState
from state.reducers.base import ReducerResult
from state._util import get_event_type, get_event_ts, to_decimal


class RiskReducer:
    """Risk evaluation (Route B).

    Inputs:
      - PortfolioState (equity/exposure/leverage)
      - Optional RISK/CONTROL events to set manual blocks/flags.
    """

    def __init__(
        self,
        *,
        limits: RiskLimits,
        get_portfolio: Callable[[], PortfolioState],
        get_positions: Callable[[], Mapping[str, PositionState]],
    ) -> None:
        self._limits = limits
        self._get_portfolio = get_portfolio
        self._get_positions = get_positions

    def reduce(self, state: RiskState, event: Any) -> ReducerResult[RiskState]:
        et = get_event_type(event)
        ts = get_event_ts(event)

        portfolio = self._get_portfolio()
        equity = portfolio.total_equity

        # peak/drawdown
        peak = state.equity_peak
        if equity > peak:
            peak = equity

        dd = Decimal("0")
        if peak > 0:
            dd = (peak - equity) / peak
            if dd < 0:
                dd = Decimal("0")

        halted = state.halted
        level = state.level
        message = state.message

        flags = set(state.flags)

        # Manual risk events
        if et == "risk":
            lvl = getattr(event, "level", None)
            msg = getattr(event, "message", None)
            if lvl is not None:
                level = str(lvl).strip().lower()
            if msg is not None:
                message = str(msg)
            if level == "block":
                flags.add("risk_block_event")
        elif et == "control":
            cmd = getattr(event, "command", None) or getattr(event, "action", None)
            cmd_s = str(cmd).strip().lower() if cmd is not None else ""
            if cmd_s in ("halt", "pause", "stop"):
                halted = True
                flags.add("manual_halt")
            elif cmd_s in ("resume", "unpause", "start"):
                halted = False
                # remove only the manual halt flag; other flags persist
                flags.discard("manual_halt")

        # Limit checks
        if self._limits.block_on_equity_le_zero and equity <= 0:
            flags.add("equity_le_zero")

        lev = portfolio.leverage
        if lev is not None and lev > self._limits.max_leverage:
            flags.add("max_leverage")

        if self._limits.max_position_notional is not None:
            cap = self._limits.max_position_notional
            for sym, pos in self._get_positions().items():
                if pos.qty == 0:
                    continue
                # mark = portfolio exposure derived; recompute fast
                mark = None
                # Use last_price from pos if available; for single symbol the portfolio already used market price.
                if pos.last_price is not None:
                    mark = pos.last_price
                if mark is None:
                    continue
                notional = abs(pos.qty) * mark
                if notional > cap:
                    flags.add(f"max_position_notional:{sym}")
                    break

        if dd > self._limits.max_drawdown_pct:
            flags.add("max_drawdown")

        blocked = halted or (len(flags) > 0 and ("risk_block_event" in flags or "equity_le_zero" in flags or "max_drawdown" in flags or "max_leverage" in flags or any(f.startswith("max_position_notional") for f in flags)))

        new_flags = tuple(sorted(flags))

        new_state = state.with_update(
            blocked=blocked,
            halted=halted,
            level=level,
            message=message,
            flags=new_flags,
            equity_peak=peak,
            drawdown_pct=dd,
            ts=ts,
        )
        changed = new_state != state
        return ReducerResult(state=new_state, changed=changed, note="risk_eval")
