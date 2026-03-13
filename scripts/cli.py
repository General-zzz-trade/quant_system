# tools/cli.py
"""CLI entry point for quant system tools."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

from scripts.catalog import render_catalog


def main() -> None:
    """主命令行入口。"""
    parser = argparse.ArgumentParser(
        prog="quant",
        description="Quant Trading System CLI",
    )
    sub = parser.add_subparsers(dest="command")

    # backtest 子命令
    bt = sub.add_parser("backtest", help="Run backtest")
    bt.add_argument("--config", type=str, help="Config file path")
    bt.add_argument("--data", type=str, help="Data directory")
    bt.add_argument("--symbol", type=str, default="BTCUSDT")

    # sync 子命令
    sync = sub.add_parser("sync", help="Sync market data")
    sync.add_argument("--symbol", type=str, default="BTCUSDT")
    sync.add_argument("--interval", type=str, default="1h")

    catalog = sub.add_parser("catalog", help="Show curated scripts catalog")
    catalog.add_argument(
        "--scripts",
        action="store_true",
        help="List maintained scripts groups and primary entrypoints",
    )

    inspect = sub.add_parser("model-inspect", help="Inspect production model artifacts and autoload state")
    inspect.add_argument("--registry-db", type=str, default="model_registry.db")
    inspect.add_argument("--artifact-root", type=str, default="artifacts")
    inspect.add_argument("--model", action="append", dest="models", required=True)

    promote = sub.add_parser("model-promote", help="Promote a model_id to production")
    promote.add_argument("--registry-db", type=str, default="model_registry.db")
    promote.add_argument("--model-id", type=str, required=True)
    promote.add_argument("--reason", type=str)
    promote.add_argument("--actor", type=str, default="cli")

    rollback = sub.add_parser("model-rollback", help="Rollback a model name to previous production version")
    rollback.add_argument("--registry-db", type=str, default="model_registry.db")
    rollback.add_argument("--model", type=str, required=True)
    rollback.add_argument("--to-model-id", type=str)
    rollback.add_argument("--to-version", type=int)
    rollback.add_argument("--reason", type=str)
    rollback.add_argument("--actor", type=str, default="cli")

    history = sub.add_parser("model-history", help="Show recent model promotion / rollback audit records")
    history.add_argument("--registry-db", type=str, default="model_registry.db")
    history.add_argument("--model", type=str, required=True)
    history.add_argument("--limit", type=int, default=20)

    ops_audit = sub.add_parser("ops-audit", help="Show recent operator-control and model-ops audit records")
    ops_audit.add_argument("--registry-db", type=str, default="model_registry.db")
    ops_audit.add_argument("--event-log", type=str, default="data/live/event_log.db")
    ops_audit.add_argument("--model", type=str)
    ops_audit.add_argument("--limit", type=int, default=20)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "backtest":
        from runner.backtest_runner import run_backtest
        run_backtest()
    elif args.command == "sync":
        print(f"Syncing {args.symbol} {args.interval}...")
    elif args.command == "catalog":
        print(render_catalog())
    elif args.command == "model-inspect":
        from alpha.model_loader import ProductionModelLoader
        from research.model_registry.artifact import ArtifactStore
        from research.model_registry.registry import ModelRegistry

        loader = ProductionModelLoader(
            ModelRegistry(args.registry_db),
            ArtifactStore(args.artifact_root),
        )
        print(json.dumps(loader.inspect_production_models(args.models), indent=2, ensure_ascii=False))
    elif args.command == "model-promote":
        from research.model_registry.registry import ModelRegistry

        registry = ModelRegistry(args.registry_db)
        model = registry.get(args.model_id)
        registry.promote(args.model_id, reason=args.reason, actor=args.actor)
        print(
            json.dumps(
                {
                    "ok": True,
                    "action": "promote",
                    "model_id": args.model_id,
                    "model": None if model is None else model.name,
                    "version": None if model is None else model.version,
                    "reason": args.reason,
                    "actor": args.actor,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    elif args.command == "model-rollback":
        from research.model_registry.registry import ModelRegistry

        registry = ModelRegistry(args.registry_db)
        current = registry.get_production(args.model)
        target = registry.rollback_to_previous(
            args.model,
            to_model_id=args.to_model_id,
            to_version=args.to_version,
            reason=args.reason,
            actor=args.actor,
        )
        print(
            json.dumps(
                {
                    "ok": True,
                    "action": "rollback",
                    "model": args.model,
                    "from_model_id": None if current is None else current.model_id,
                    "from_version": None if current is None else current.version,
                    "model_id": target.model_id,
                    "version": target.version,
                    "reason": args.reason,
                    "actor": args.actor,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    elif args.command == "model-history":
        from research.model_registry.registry import ModelRegistry

        registry = ModelRegistry(args.registry_db)
        rows = [
            {
                "action_id": row.action_id,
                "model": row.name,
                "action": row.action,
                "from_model_id": row.from_model_id,
                "to_model_id": row.to_model_id,
                "reason": row.reason,
                "actor": row.actor,
                "created_at": row.created_at.isoformat(),
                "metadata": row.metadata,
            }
            for row in registry.list_actions(args.model, limit=args.limit)
        ]
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    elif args.command == "ops-audit":
        from pathlib import Path

        from execution.store.event_log import SQLiteEventLog
        from research.model_registry.registry import ModelRegistry

        operator_controls = []
        execution_incidents = []
        model_reloads = []
        timeline = []
        event_log_path = Path(args.event_log)
        if event_log_path.exists():
            event_log = SQLiteEventLog(str(event_log_path))
            operator_controls = event_log.list_recent(event_type="operator_control", limit=args.limit)
            execution_incidents = event_log.list_recent(event_type="execution_incident", limit=args.limit)
            model_reloads = event_log.list_recent(event_type="model_reload", limit=args.limit)
            timeline.extend(
                {
                    "kind": "control",
                    "ts": row["payload"].get("ts") or datetime.fromtimestamp(float(row["ts"]), timezone.utc).isoformat(),
                    "title": f"operator_{row['payload'].get('command', row.get('correlation_id', 'control'))}",
                    "category": "operator_control",
                    "source": row["payload"].get("source", ""),
                    "detail": row["payload"],
                }
                for row in operator_controls
            )
            timeline.extend(
                {
                    "kind": "execution_incident",
                    "ts": row["payload"].get("ts") or datetime.fromtimestamp(float(row["ts"]), timezone.utc).isoformat(),
                    "title": row["payload"].get("title", "execution_incident"),
                    "category": row["payload"].get("category", "execution_incident"),
                    "source": row["payload"].get("source", ""),
                    "detail": row["payload"],
                }
                for row in execution_incidents
            )
            timeline.extend(
                {
                    "kind": "model_reload",
                    "ts": row["payload"].get("ts") or datetime.fromtimestamp(float(row["ts"]), timezone.utc).isoformat(),
                    "title": f"model_reload_{row['payload'].get('outcome', 'unknown')}",
                    "category": "model_reload",
                    "source": "model:reload",
                    "detail": row["payload"],
                }
                for row in model_reloads
            )

        registry = ModelRegistry(args.registry_db)
        model_actions = []
        if args.model:
            model_actions = [
                {
                    "action_id": row.action_id,
                    "model": row.name,
                    "action": row.action,
                    "from_model_id": row.from_model_id,
                    "to_model_id": row.to_model_id,
                    "reason": row.reason,
                    "actor": row.actor,
                    "created_at": row.created_at.isoformat(),
                    "metadata": row.metadata,
                }
                for row in registry.list_actions(args.model, limit=args.limit)
            ]
            timeline.extend(
                {
                    "kind": "model_action",
                    "ts": row["created_at"],
                    "title": f"model_{row['action']}",
                    "category": "model_action",
                    "source": row["actor"] or "model_registry",
                    "detail": row,
                }
                for row in model_actions
            )

        timeline.sort(key=lambda row: row.get("ts", ""), reverse=True)
        timeline = timeline[: args.limit]

        print(
            json.dumps(
                {
                    "operator_controls": operator_controls,
                    "execution_incidents": execution_incidents,
                    "model_reloads": model_reloads,
                    "model_actions": model_actions,
                    "timeline": timeline,
                },
                indent=2,
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
