# RAG Evaluation Benchmarks (Multi-Document / Multi-File)

A collection of popular benchmarks for evaluating Retrieval-Augmented Generation (RAG) systems, especially those requiring retrieval across multiple documents and answering multiple questions.

---

## 1. HotpotQA

**Description:** Multi-hop question answering requiring evidence from multiple Wikipedia articles.

### Links
- Website: https://hotpotqa.github.io/
- GitHub: https://github.com/hotpotqa/hotpot
- Hugging Face: https://huggingface.co/datasets/hotpotqa/hotpot_qa

### Stats
- ~113,000 questions
- Supporting fact annotations
- Multi-document reasoning

---

## 2. MultiHop-RAG

**Description:** Purpose-built benchmark for evaluating modern RAG systems on multi-hop retrieval and reasoning.

### Links
- GitHub: https://github.com/yixuantt/MultiHop-RAG
- Hugging Face: https://huggingface.co/datasets/yixuantt/MultiHopRAG

### Stats
- 2,556 questions
- Evidence spread across multiple documents
- Designed specifically for RAG

---

## 3. RAGBench

**Description:** Large-scale benchmark for end-to-end RAG evaluation across multiple domains.

### Links
- GitHub: https://github.com/rungalileo/ragbench
- Hugging Face: https://huggingface.co/datasets/galileo-ai/ragbench
- Paper: https://arxiv.org/abs/2407.11005

### Stats
- ~100,000 examples
- Enterprise-style documents
- Multiple domains

---

## 4. Open RAG Bench

**Description:** Benchmark built from real PDFs containing text, tables, and images.

### Links
- GitHub: https://github.com/vectara/open-rag-bench
- Hugging Face: https://huggingface.co/datasets/vectara/open_ragbench

### Stats
- ~1,000 PDFs
- ~3,000 questions
- PDF-focused evaluation

---

## 5. EnterpriseRAG-Bench

**Description:** Enterprise document retrieval benchmark using realistic business documents.

### Links
- GitHub: https://github.com/onyx-dot-app/EnterpriseRAG-Bench

### Stats
- Enterprise-focused
- Internal knowledge base style evaluation

---

## 6. Natural Questions (NQ)

**Description:** Open-domain QA benchmark built from real Google search queries.

### Links
- Website: https://ai.google.com/research/NaturalQuestions
- Hugging Face: https://huggingface.co/datasets/google-research-datasets/natural_questions

### Stats
- 300k+ questions
- Wikipedia corpus
- Strong retrieval benchmark

---

## 7. TriviaQA

**Description:** Large-scale QA benchmark with evidence documents from the web and Wikipedia.

### Links
- Website: http://nlp.cs.washington.edu/triviaqa/
- Hugging Face: https://huggingface.co/datasets/trivia_qa

### Stats
- 650k+ question-answer pairs
- Open-domain retrieval

---

## 8. FanOutQA

**Description:** Cross-document reasoning benchmark that requires synthesizing information from multiple sources.

### Links
- Paper: https://aclanthology.org/2024.acl-short.2/
- GitHub: https://github.com/google-deepmind/fanoutqa

### Stats
- Multi-document aggregation
- Long-context reasoning

---

## 9. T²-RAGBench

**Description:** Financial and tabular-document RAG benchmark.

### Links
- Website: https://t2ragbench.demo.hcds.uni-hamburg.de/
- Paper: https://arxiv.org/abs/2502.13491

### Stats
- 23k+ QA pairs
- 7k+ financial reports
- Tables + text

---

## 10. MEBench

**Description:** Multi-entity benchmark for evaluating cross-document retrieval and aggregation.

### Links
- Paper: https://arxiv.org/abs/2502.18993

### Stats
- 4,780 questions
- Multi-document reasoning

---

## 11. MuDABench

**Description:** Multi-document analytical benchmark focusing on long-document reasoning.

### Links
- Paper: https://arxiv.org/abs/2604.22239

### Stats
- 80,000+ pages
- Quantitative reasoning
- Multi-document analytics

---

# Recommended Benchmark Suite

If evaluating a production-grade RAG system:

1. MultiHop-RAG
2. HotpotQA
3. RAGBench
4. Open RAG Bench

Together they cover:

- Multi-hop retrieval
- Multi-document reasoning
- Enterprise knowledge bases
- PDF ingestion
- End-to-end RAG evaluation

# Quick Download Example

```python
from datasets import load_dataset

hotpot = load_dataset("hotpotqa/hotpot_qa")
multihop = load_dataset("yixuantt/MultiHopRAG")
ragbench = load_dataset("galileo-ai/ragbench")
open_rag = load_dataset("vectara/open_ragbench")
```