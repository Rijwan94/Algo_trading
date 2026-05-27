from __future__ import annotations

import argparse
import json
import sys

from .pipeline import run_pipeline
from .utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ML models for forex time series.")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--acknowledge-risk", action="store_true", help="Confirm you read the risk disclosure")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    if config.get("risk", {}).get("require_confirmation", True) and not args.acknowledge_risk:
        print(
            "Risk acknowledgement required. Review docs/RISK_DISCLOSURE.md and re-run with --acknowledge-risk.",
            file=sys.stderr,
        )
        return 2

    results = run_pipeline(config)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
