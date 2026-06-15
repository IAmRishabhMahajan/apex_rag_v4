"""CLI entry point for running APEX-RAG benchmark evaluations.

Run a single benchmark or all five in sequence:

    python benchmarks/run_all.py                          # all benchmarks, 200 examples each
    python benchmarks/run_all.py --benchmark hotpotqa     # HotpotQA only
    python benchmarks/run_all.py --max-examples 50        # fast smoke test
    python benchmarks/run_all.py --benchmark ragbench --subsets covidqa finqa

Available benchmarks:
    hotpotqa        HotpotQA multi-hop QA (distractor config)
    nq              Natural Questions open-domain QA
    triviaqa        TriviaQA reading comprehension
    ragbench        RAGBench multi-domain enterprise evaluation
    multihop        MultiHop-RAG purpose-built RAG benchmark
    all             Run all five (default)

Output: per-benchmark tables + a final summary row.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass


@dataclass
class _Summary:
    name: str
    examples: int
    failures: int
    primary_f1: float
    label: str


def _run_hotpotqa(args: argparse.Namespace) -> _Summary:
    from benchmarks.hotpotqa import print_results, run
    r = run(max_examples=args.max_examples)
    print_results(r)
    return _Summary(
        name="HotpotQA",
        examples=r.num_examples,
        failures=r.failures,
        primary_f1=r.joint_f1,
        label="Joint F1",
    )


def _run_nq(args: argparse.Namespace) -> _Summary:
    from benchmarks.natural_questions import print_results, run
    r = run(max_examples=args.max_examples)
    print_results(r)
    return _Summary(
        name="Natural Questions",
        examples=r.num_examples,
        failures=r.failures,
        primary_f1=r.short_f1,
        label="Short-Answer F1",
    )


def _run_triviaqa(args: argparse.Namespace) -> _Summary:
    from benchmarks.triviaqa import print_results, run
    r = run(max_examples=args.max_examples)
    print_results(r)
    return _Summary(
        name="TriviaQA",
        examples=r.num_examples,
        failures=r.failures,
        primary_f1=r.f1,
        label="F1",
    )


def _run_ragbench(args: argparse.Namespace) -> _Summary:
    from benchmarks.ragbench import print_results, run
    subsets = args.subsets if args.subsets else None
    r = run(max_examples=args.max_examples, subsets=subsets)
    print_results(r)
    return _Summary(
        name="RAGBench",
        examples=r.num_examples,
        failures=r.failures,
        primary_f1=r.overall_adherence,
        label="Adherence",
    )


def _run_multihop(args: argparse.Namespace) -> _Summary:
    from benchmarks.multihop_rag import print_results, run
    r = run(max_examples=args.max_examples)
    print_results(r)
    return _Summary(
        name="MultiHop-RAG",
        examples=r.num_examples,
        failures=r.failures,
        primary_f1=r.answer_f1,
        label="Answer F1",
    )


_RUNNERS = {
    "hotpotqa": _run_hotpotqa,
    "nq": _run_nq,
    "triviaqa": _run_triviaqa,
    "ragbench": _run_ragbench,
    "multihop": _run_multihop,
}


def _print_summary(summaries: list[_Summary], total_elapsed: float) -> None:
    print("\n" + "=" * 62)
    print("  APEX-RAG Benchmark Summary")
    print("=" * 62)
    print(f"  {'Benchmark':<22} {'Examples':>8} {'Failures':>8} {'Primary Metric':>16}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*16}")
    for s in summaries:
        metric = f"{s.label}={s.primary_f1:.3f}"
        print(f"  {s.name:<22} {s.examples:>8} {s.failures:>8} {metric:>16}")
    print(f"\n  Total elapsed: {total_elapsed:.1f}s")
    print("=" * 62 + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run APEX-RAG benchmark evaluations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--benchmark",
        choices=[*_RUNNERS.keys(), "all"],
        default="all",
        help="Which benchmark to run (default: all).",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=200,
        dest="max_examples",
        help="Maximum examples per benchmark (default: 200).",
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=None,
        help="RAGBench domain subsets to include (e.g. covidqa finqa).",
    )
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    summaries: list[_Summary] = []

    targets = list(_RUNNERS.keys()) if args.benchmark == "all" else [args.benchmark]

    for name in targets:
        print(f"\n>>> Running {name} ...", flush=True)
        try:
            summary = _RUNNERS[name](args)
            summaries.append(summary)
        except ImportError as exc:
            print(f"  [SKIP] {name}: {exc}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            print(f"  [ERROR] {name}: {exc}", file=sys.stderr)

    if summaries:
        _print_summary(summaries, time.perf_counter() - t0)

    return 0


if __name__ == "__main__":
    sys.exit(main())
