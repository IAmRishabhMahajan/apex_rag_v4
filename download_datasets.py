"""Phase 1 — Download and cache all benchmark datasets to data/.

Downloads each dataset from HuggingFace and saves examples as JSON so that
evaluations never need internet access after this step runs once.

Datasets that are small enough are downloaded fully (HotpotQA, RAGBench subsets).
Datasets that are very large use streaming to extract the first N examples
(Natural Questions: 287 Parquet shards; TriviaQA: 26 shards; MultiHop-RAG).

Usage:
    uv run python download_datasets.py                # 200 examples each
    uv run python download_datasets.py --n 500        # more examples
    uv run python download_datasets.py --dataset nq   # one dataset only
    uv run python download_datasets.py --refresh      # re-download even if cached

Datasets:
    hotpotqa   hotpotqa/hotpot_qa (distractor, validation)
    nq         google-research-datasets/natural_questions (validation, streaming)
    triviaqa   mandarjoshi/trivia_qa rc (validation, streaming)
    ragbench   galileo-ai/ragbench — 5 subsets: covidqa hotpotqa pubmedqa finqa cuad
    multihop   yixuantt/MultiHopRAG (train, streaming)
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

RAGBENCH_SUBSETS = ["covidqa", "hotpotqa", "pubmedqa", "finqa", "cuad"]


# ── helpers ────────────────────────────────────────────────────────────────────

def _cache_path(name: str, n: int) -> Path:
    return DATA_DIR / f"{name}_{n}.json"


def _already_cached(name: str, n: int, refresh: bool) -> bool:
    p = _cache_path(name, n)
    if p.exists() and not refresh:
        size_kb = round(p.stat().st_size / 1024, 1)
        print(f"     Already cached: {p.name}  ({size_kb} KB) — skipping.", flush=True)
        return True
    return False


def _save(name: str, n: int, examples: list[dict]) -> None:
    p = _cache_path(name, n)
    p.write_text(json.dumps(examples, ensure_ascii=False, default=str), encoding="utf-8")
    size_kb = round(p.stat().st_size / 1024, 1)
    print(f"     Saved -> {p.name}  ({len(examples)} examples, {size_kb} KB)", flush=True)


# ── per-dataset downloaders ────────────────────────────────────────────────────

def download_hotpotqa(n: int, refresh: bool) -> bool:
    """Full download — validation split is only 7 405 examples."""
    if _already_cached("hotpotqa", n, refresh):
        return True
    try:
        print("     Connecting to hotpotqa/hotpot_qa [distractor, validation]...", flush=True)
        from datasets import load_dataset  # type: ignore[import-untyped]
        ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
        print(f"     Dataset loaded ({len(ds)} total). Extracting {n} examples...", flush=True)
        raw = list(ds.select(range(min(n, len(ds)))))
        examples = [
            {
                "question": ex["question"],
                "answer":   ex["answer"],
                "type":     ex["type"],
                "level":    ex["level"],
                "context": {
                    "title":     list(ex["context"]["title"]),
                    "sentences": [list(s) for s in ex["context"]["sentences"]],
                },
                "supporting_facts": {
                    "title":   list(ex["supporting_facts"]["title"]),
                    "sent_id": list(ex["supporting_facts"]["sent_id"]),
                },
            }
            for ex in raw
        ]
        _save("hotpotqa", n, examples)
        return True
    except Exception as exc:
        print(f"     ERROR: {exc}", flush=True)
        return False


def download_nq(n: int, refresh: bool) -> bool:
    """Streaming — NQ validation has 287 large Parquet shards; we extract the first n."""
    if _already_cached("nq", n, refresh):
        return True
    try:
        print("     Streaming google-research-datasets/natural_questions [validation]...", flush=True)
        print("     (streaming mode — only the first shard is downloaded)", flush=True)
        from datasets import load_dataset  # type: ignore[import-untyped]
        ds = load_dataset(
            "google-research-datasets/natural_questions",
            split="validation",
            streaming=True,
        )
        MAX_TOKENS = 300
        examples: list[dict] = []
        for i, ex in enumerate(itertools.islice(ds, n)):
            doc   = ex["document"]
            anns  = ex.get("annotations", {})
            query = ex["question"]["text"] if isinstance(ex["question"], dict) else str(ex["question"])

            tokens_field = doc.get("tokens", {}) if isinstance(doc, dict) else {}
            token_list   = tokens_field.get("token", [])
            is_html_list = tokens_field.get("is_html", [])
            text_tokens  = [t for t, h in zip(token_list, is_html_list) if not h]
            passage      = " ".join(text_tokens[:MAX_TOKENS])

            short_answers: list[str] = []
            yn_answers:    list[str] = []
            if isinstance(anns, dict):
                for sa_group in anns.get("short_answers", []):
                    if isinstance(sa_group, dict):
                        for text in sa_group.get("text", []) or []:
                            if text and str(text).strip():
                                short_answers.append(str(text))
                for yn in anns.get("yes_no_answer", []):
                    if yn in (1, "YES", "yes"):
                        yn_answers.append("yes")
                    elif yn in (0, "NO", "no"):
                        yn_answers.append("no")

            examples.append({
                "query":         query,
                "passage":       passage,
                "title":         doc.get("title", "") if isinstance(doc, dict) else "",
                "url":           doc.get("url", "")   if isinstance(doc, dict) else "",
                "short_answers": short_answers,
                "yn_answers":    yn_answers,
            })
            if (i + 1) % 25 == 0:
                print(f"     ... {i+1}/{n} examples", flush=True)

        _save("nq", n, examples)
        return True
    except Exception as exc:
        print(f"     ERROR: {exc}", flush=True)
        return False


def download_triviaqa(n: int, refresh: bool) -> bool:
    """Streaming — rc validation has 26 Parquet shards; we extract the first n."""
    if _already_cached("triviaqa", n, refresh):
        return True
    try:
        print("     Streaming mandarjoshi/trivia_qa [rc, validation]...", flush=True)
        print("     (streaming mode — only the first shard is downloaded)", flush=True)
        from datasets import load_dataset  # type: ignore[import-untyped]
        ds = load_dataset("mandarjoshi/trivia_qa", "rc", split="validation", streaming=True)

        MAX_CHARS = 1500
        examples: list[dict] = []
        for i, ex in enumerate(itertools.islice(ds, n)):
            answer = ex.get("answer", {})
            aliases: list[str] = []
            for field in ("value", "aliases", "normalized_aliases", "normalized_value"):
                val = answer.get(field)
                if isinstance(val, list):
                    aliases.extend(v for v in val if v)
                elif val:
                    aliases.append(str(val))
            aliases = list(dict.fromkeys(a for a in aliases if a))

            pages    = ex.get("entity_pages") or ex.get("search_results") or {}
            contexts = pages.get("wiki_context") or pages.get("search_context") or []
            titles   = pages.get("title") or [""] * len(contexts)
            urls     = pages.get("url")   or [""] * len(contexts)

            items_data: list[dict] = []
            for ctx, ttl, url in zip(contexts, titles, urls):
                content = (ctx or "")[:MAX_CHARS].strip()
                if content:
                    items_data.append({
                        "content":   content,
                        "source_id": url or ttl or f"src-{len(items_data)}",
                        "title":     str(ttl),
                        "url":       str(url),
                    })

            if not items_data:
                continue  # skip examples with no retrievable passages
            examples.append({
                "question": ex["question"],
                "aliases":  aliases,
                "items":    items_data,
            })
            if (i + 1) % 25 == 0:
                print(f"     ... {i+1}/{n} examples", flush=True)

        _save("triviaqa", n, examples)
        return True
    except Exception as exc:
        print(f"     ERROR: {exc}", flush=True)
        return False


def download_ragbench_subset(subset: str, n: int, refresh: bool) -> bool:
    """Full download — each RAGBench subset is small (< 20k examples)."""
    cache_name = f"ragbench_{subset}"
    if _already_cached(cache_name, n, refresh):
        return True
    try:
        print(f"     Downloading galileo-ai/ragbench [{subset}]...", flush=True)
        from datasets import load_dataset  # type: ignore[import-untyped]
        ds  = load_dataset("galileo-ai/ragbench", subset, split="train")
        print(f"     Dataset loaded ({len(ds)} total). Extracting {min(n, len(ds))} examples...", flush=True)
        raw = list(ds.select(range(min(n, len(ds)))))
        examples = [
            {
                "query": r.get("question", "") or r.get("query", ""),
                "docs": (r.get("documents", []) if isinstance(r.get("documents"), list)
                         else ([r["documents"]] if r.get("documents") else [])),
            }
            for r in raw
        ]
        _save(cache_name, n, examples)
        return True
    except Exception as exc:
        print(f"     ERROR [{subset}]: {exc}", flush=True)
        return False


def download_multihop(n: int, refresh: bool) -> bool:
    """Streaming — MultiHopRAG train split; extract the first n examples."""
    if _already_cached("multihop_rag", n, refresh):
        return True
    try:
        print("     Streaming yixuantt/MultiHopRAG [MultiHopRAG, train]...", flush=True)
        from datasets import load_dataset  # type: ignore[import-untyped]
        ds = load_dataset("yixuantt/MultiHopRAG", "MultiHopRAG", split="train", streaming=True)

        examples: list[dict] = []
        for i, ex in enumerate(itertools.islice(ds, n)):
            ev_list  = ex.get("evidence_list", []) or []
            evidence: list[dict] = []
            for ev in ev_list:
                facts     = ev.get("facts", [])
                content   = " ".join(facts) if facts else ev.get("title", "")
                source_id = ev.get("source", ev.get("title", f"src-{len(evidence)}"))
                if content.strip():
                    evidence.append({
                        "content":   content,
                        "source_id": str(source_id),
                        "title":     ev.get("title", ""),
                        "url":       str(source_id) if str(source_id).startswith("http") else "",
                    })
            examples.append({
                "query":         ex.get("query", ""),
                "answer":        ex.get("answer", ""),
                "question_type": ex.get("question_type", "unknown"),
                "evidence":      evidence,
            })
            if (i + 1) % 25 == 0:
                print(f"     ... {i+1}/{n} examples", flush=True)

        _save("multihop_rag", n, examples)
        return True
    except Exception as exc:
        print(f"     ERROR: {exc}", flush=True)
        return False


# ── orchestrator ───────────────────────────────────────────────────────────────

ALL_DATASETS = ["hotpotqa", "nq", "triviaqa", "ragbench", "multihop"]

DOWNLOADERS = {
    "hotpotqa": ("HotpotQA         [full download, 7 405 validation examples]", download_hotpotqa),
    "nq":       ("Natural Questions [streaming, 287 shards — extracts first N]", download_nq),
    "triviaqa": ("TriviaQA          [streaming, 26 shards — extracts first N]",  download_triviaqa),
    "ragbench": ("RAGBench          [full download, 5 subsets]",                  None),
    "multihop": ("MultiHop-RAG      [streaming, extracts first N]",               download_multihop),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset", choices=[*ALL_DATASETS, "all"], default="all",
        help="Which dataset to download (default: all)",
    )
    parser.add_argument(
        "--n", type=int, default=200,
        help="Number of examples to extract per dataset (default: 200)",
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="Re-download even if local cache already exists",
    )
    args = parser.parse_args()

    targets = ALL_DATASETS if args.dataset == "all" else [args.dataset]
    N = args.n

    print(f"\nAPEX-RAG Benchmark Dataset Downloader")
    print(f"  Examples per dataset : {N}")
    print(f"  Cache directory      : {DATA_DIR}/")
    print(f"  Refresh mode         : {'yes' if args.refresh else 'no'}")
    print(f"  Datasets selected    : {', '.join(targets)}\n")

    t0 = time.perf_counter()
    results: list[tuple[str, bool]] = []

    for idx, name in enumerate(targets, 1):
        label, fn = DOWNLOADERS[name]
        print(f"[{idx}/{len(targets)}] {label}", flush=True)
        bt = time.perf_counter()

        if name == "ragbench":
            ok = True
            for subset in RAGBENCH_SUBSETS:
                ok = download_ragbench_subset(subset, N, args.refresh) and ok
        else:
            ok = fn(N, args.refresh)  # type: ignore[misc]

        elapsed = time.perf_counter() - bt
        status = "OK" if ok else "FAILED"
        print(f"     [{status}]  {elapsed:.1f}s\n", flush=True)
        results.append((name, ok))

    total = time.perf_counter() - t0
    passed = sum(1 for _, ok in results if ok)

    print("=" * 58)
    print(f"  Download summary  ({passed}/{len(results)} succeeded, {total:.1f}s total)")
    print("=" * 58)
    for name, ok in results:
        icon = "OK  " if ok else "FAIL"
        print(f"  {icon}  {name}")
    print()
    print("  Next step:")
    print(f"    uv run python run_evals_to_excel.py --max-examples {N}")
    print("=" * 58)


if __name__ == "__main__":
    main()
