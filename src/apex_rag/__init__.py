"""APEX-RAG v5 — adaptive evidence-driven RAG pipeline."""

from src.apex_rag.pipeline import PipelineResult, run_batch, run_pipeline

__all__ = ["run_pipeline", "run_batch", "PipelineResult"]
