"""CLI entrypoint: python -m forscheule <command>."""

from __future__ import annotations

import argparse
import sys

from forscheule.config import setup_logging


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(
        prog="forscheule",
        description="ForschEule – Daily paper suggestions for spatial transcriptomics",
    )
    sub = parser.add_subparsers(dest="command")

    # backfill
    bf = sub.add_parser("backfill", help="Backfill recommendations for the last N days")
    bf.add_argument("--days", type=int, required=True, help="Number of days to backfill")
    bf.add_argument("--window", type=int, default=None, help="Fetch window in days (default: 7)")

    # run-daily
    rd = sub.add_parser("run-daily", help="Run pipeline for today")
    rd.add_argument("--window", type=int, default=None, help="Fetch window in days (default: 7)")

    # serve
    sv = sub.add_parser("serve", help="Start the API server")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()

    if args.command == "backfill":
        from forscheule.pipeline import backfill

        backfill(days=args.days, window_days=args.window)
    elif args.command == "run-daily":
        from forscheule.pipeline import run_daily

        run_daily(window_days=args.window)
    elif args.command == "serve":
        import uvicorn

        uvicorn.run("forscheule.api.app:app", host=args.host, port=args.port)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
