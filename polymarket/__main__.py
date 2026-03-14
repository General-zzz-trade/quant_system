"""Entry point: python3 -m polymarket --config config/polymarket.yaml"""
from __future__ import annotations
import argparse
import sys
sys.path.insert(0, "/quant_system")


def main():
    parser = argparse.ArgumentParser(description="Polymarket prediction market trader")
    parser.add_argument("--config", default="config/polymarket.yaml")
    parser.add_argument("--once", action="store_true", help="Run single cycle then exit")
    args = parser.parse_args()

    from polymarket.config import PolymarketConfig
    from polymarket.runner import PolymarketRunner

    config = PolymarketConfig.from_yaml(args.config) if args.config.endswith(".yaml") else PolymarketConfig()
    runner = PolymarketRunner(config)

    if args.once:
        runner.run_once()
    else:
        try:
            runner.start()
        except KeyboardInterrupt:
            runner.stop()


if __name__ == "__main__":
    main()
