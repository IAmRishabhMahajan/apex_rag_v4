# APEX-RAG v4

An adaptive, evidence-driven Retrieval-Augmented Generation pipeline built as a **fact verifier**, not a generative QA system. Each stage acts as a defect filter — catching hallucinations, unsupported claims, and retrieval gaps before they reach the final answer.

**HotpotQA result: 87.2% Exact Match on 7,405 examples · 100% Faithfulness · 0.3 min runtime**

---

## Architecture

The pipeline runs 9 sequential stages on every query:

```
Query → [1] Query Intelligence
      → [2] Retrieval Planning
      → [3] Expert Routing
      → [4] Evidence Fusion
      → [5] Validation Mesh          ← rejects empty/conflicting bundles
      → [6] Evidence Scoring
      → [7] Complex Reasoning         ← activates for multi-hop queries
      → [8] Grounded Generation       ← claim must appear in evidence to pass
      → [9] Risk Verification
      → PipelineResult
```

The pipeline takes **pre-formed candidate claims** and checks whether they are supported by the provided evidence. It does not synthesize answers from scratch.

---

## Quick start

### Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Install

```bash
git clone <repo-url>
cd apex_rag_v4

uv sync --group dev
```

### Run the pipeline

```python
from src.apex_rag.pipeline import run_pipeline

result = run_pipeline(
    raw_query="What is Docker?",
    raw_evidence_items=[
        {
            "content":   "Docker is a platform for containerizing applications.",
            "source_id": "doc-1",
            "title":     "Docker Docs",
            "url":       "https://docs.docker.com",
        }
    ],
)

print(result.answer.text)                      # "What is Docker"
print(result.answer.all_claims_supported)      # True
print(result.eval_result.final.faithfulness)   # 1.0
```

### With explicit candidate claims

```python
result = run_pipeline(
    raw_query="When was Docker released?",
    raw_evidence_items=[
        {
            "content":   "Docker was first released in 2013.",
            "source_id": "doc-1",
            "title":     "Docker History",
            "url":       "",
        }
    ],
    candidate_claims=["2013"],   # the claim to verify against evidence
)

print(result.answer.text)               # "2013"
print(result.answer.all_claims_supported)  # True
```

### `run_pipeline()` parameters

| Parameter | Type | Description |
|---|---|---|
| `raw_query` | `str` | The user's question |
| `raw_evidence_items` | `list[dict]` | Each dict needs `content`, `source_id`, `title`, `url` |
| `query_id` | `str` | Identifier for this run (default: `"q1"`) |
| `candidate_claims` | `list[str] \| None` | Claims to verify. Defaults to the query itself |
| `relevant_source_ids` | `set[str] \| None` | Ground-truth source IDs for eval metrics |
| `high_risk` | `bool` | Apply stricter evidence scoring thresholds |

---

## Run tests

```bash
uv run python -m unittest discover -s tests
```

292 tests, ~0.03 s.

---

## Run benchmarks

### Step 1 — Download datasets

```bash
# Download all 5 benchmarks (200 examples each by default)
uv run python download_datasets.py

# One dataset only
uv run python download_datasets.py --dataset hotpotqa

# More examples
uv run python download_datasets.py --dataset hotpotqa --n 7405

# Re-download even if already cached
uv run python download_datasets.py --refresh
```

Datasets are cached to `data/` (gitignored). Supported datasets:

| Key | Dataset | Notes |
|---|---|---|
| `hotpotqa` | HotpotQA distractor validation | 7,405 examples, full download |
| `nq` | Natural Questions | Streaming, first N examples |
| `triviaqa` | TriviaQA rc | Streaming, skips empty-context examples |
| `ragbench` | RAGBench (5 subsets) | covidqa, hotpotqa, pubmedqa, finqa, cuad |
| `multihop` | MultiHop-RAG | Streaming |

### Step 2 — Run evaluations

```bash
# Run all benchmarks and write an Excel workbook
uv run python run_evals_to_excel.py

# Limit examples per benchmark
uv run python run_evals_to_excel.py --max-examples 200
```

Output: `eval_results.xlsx` with one sheet per benchmark plus an aggregate summary.

### Run a single benchmark

```bash
uv run python eval_framework.py --dataset hotpotqa --max-examples 200
```

### Full HotpotQA validation set (7,405 examples)

```bash
uv run python download_datasets.py --dataset hotpotqa --n 7405
uv run python eval_framework.py --dataset hotpotqa --max-examples 7405
```

Expected: **87.2% EM, 100% Faithfulness, 0.3 min**.

---

## Project structure

```
src/apex_rag/
  query_intelligence.py     # US-001  intent detection, entity extraction
  retrieval_planning.py     # US-002  adaptive retrieval plan
  expert_routing.py         # US-003  routes to dense/sparse/hybrid retrieval
  evidence_fusion.py        # US-004  deduplication, conflict detection
  validation_mesh.py        # US-005  multi-stage validation gate
  complex_reasoning.py      # US-006  multi-hop reasoning path
  evidence_scoring.py       # US-007  gap detection, relevance scoring
  retrieval_repair.py       # US-008  repair loop for failed retrievals
  generation.py             # US-009  claim grounding against evidence
  risk_verification.py      # US-010  risk critique and answer verification
  apex_eval.py              # US-011  evaluation metrics (IR, claim, faithfulness)
  research_traceability.py  # US-012  citation and provenance tracking
  pipeline.py               # end-to-end run_pipeline() entry point

benchmarks/                 # per-dataset runner modules
download_datasets.py        # HuggingFace dataset downloader / cache manager
eval_framework.py           # multi-benchmark evaluation with failure classification
run_evals_to_excel.py       # drives all evals and writes Excel workbook
docs/user-stories/          # US-001 through US-023 acceptance criteria
```

---

## Development commands

```bash
# Tests
uv run python -m unittest discover -s tests

# Format
uv run ruff format .

# Lint
uv run ruff check .

# Type check
uv run mypy src/
```

---

## Key design decisions

- **Claim verifier, not answer generator.** The pipeline checks whether a pre-formed claim is supported by evidence. It cannot synthesize answers that aren't present in the evidence.
- **No vector DB required.** Evidence items are passed directly to `run_pipeline()` — the caller owns retrieval.
- **Fails closed.** If no claims are supported, `generate_answer()` raises `GroundingError` rather than returning an unsupported answer.
- **100% Faithfulness by construction.** Only claims that appear verbatim (substring or keyword match) in non-conflicting evidence items are included in the output.
