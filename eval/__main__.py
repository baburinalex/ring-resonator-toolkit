"""CLI стенда: python -m eval {run,summarize}."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .runner import load_logs, run_benchmark, write_summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="прогнать модели на случаях бенчмарка")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--cases", type=Path, required=True, help="папка ring_toolkit.benchmark")
    run.add_argument("--out", type=Path, default=Path("eval_runs"))

    summ = sub.add_parser("summarize", help="пересобрать сводку из логов")
    summ.add_argument("out", type=Path)

    args = parser.parse_args(argv)
    if args.cmd == "run":
        run_benchmark(load_config(args.config), args.cases, args.out)
    else:
        write_summary(load_logs(args.out), args.out)
    print(f"Сводка: {args.out / 'summary.md'}, {args.out / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
