"""`jevals bench`: replay a small fixed RAG dataset and print measured numbers.

The README's numbers table is labeled as estimates until this has been run against a real
backend. With --ragas it also runs the four equivalent Ragas metrics for a side by side.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

_DATA = Path(__file__).parent / "data" / "bench_rag.jsonl"


def _rows(n: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in _DATA.read_text().splitlines() if line.strip()]
    out = []
    while len(out) < n:
        out.extend(rows)
    return out[:n]


def run_bench(
    n: int = 20, backend: str | None = None, ragas: bool = False, model: str = "gpt-4.1-mini"
) -> int:
    from ._runner import evaluate_dataset
    from .quality import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

    rows = _rows(n)
    evals = [Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()]
    t0 = time.perf_counter()
    rep = evaluate_dataset(rows, evals, backend, concurrency=8)
    wall = time.perf_counter() - t0
    u = rep.usage
    lat = sorted(rep.latencies_ms)
    p50 = lat[len(lat) // 2] if lat else 0
    p95 = lat[int(len(lat) * 0.95)] if lat else 0
    print(f"jevals ({u.backend})  n={n}")
    print(f"  requests {u.requests:,}  ({u.requests / n:.1f} per sample)")
    print(
        f"  input tokens {u.input_tokens:,}  ({u.input_tokens / n:,.0f} per sample)   output tokens {u.output_tokens:,}"
    )
    if u.cost_usd is not None:
        print(f"  cost ${u.cost_usd:.4f}  (${u.cost_usd / n * 1000:.2f} per 1k samples)")
    print(f"  latency p50 {p50:.0f}ms  p95 {p95:.0f}ms   wall {wall:.1f}s")
    print()
    print(rep.table())

    if ragas:
        _run_ragas(rows, model)
    return 0


def _run_ragas(rows: list[dict[str, Any]], model: str) -> None:
    try:
        from datasets import Dataset  # type: ignore
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # type: ignore
        from ragas import evaluate as ragas_evaluate  # type: ignore
        from ragas.metrics import (  # type: ignore
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError:
        print("ragas side needs: pip install ragas datasets langchain-openai   (and OPENAI_API_KEY)")
        return
    from langchain_community.callbacks import get_openai_callback  # type: ignore

    # OPENAI_API_KEY + OPENAI_BASE_URL are read by langchain. With only an
    # OpenRouter key, point at OpenRouter and use its model ids.

    base_url = os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY")
    embed_model = os.environ.get("JEVALS_BENCH_EMBEDDINGS", "text-embedding-3-small")
    if not api_key and os.environ.get("OPENROUTER_API_KEY"):
        api_key = os.environ["OPENROUTER_API_KEY"]
        base_url = base_url or "https://openrouter.ai/api/v1"
        if "/" not in model:
            model = f"openai/{model}"
        if "/" not in embed_model:
            embed_model = f"openai/{embed_model}"

    ds = Dataset.from_list(
        [
            {
                "question": r["input"],
                "answer": r["output"],
                "contexts": r["contexts"],
                "ground_truth": r.get("expected", ""),
            }
            for r in rows
        ]
    )
    t0 = time.perf_counter()
    with get_openai_callback() as cb:
        res = ragas_evaluate(
            ds,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=ChatOpenAI(model=model, api_key=api_key, base_url=base_url, temperature=0),
            embeddings=OpenAIEmbeddings(
                model=embed_model, api_key=api_key, base_url=base_url, check_embedding_ctx_length=False
            ),
            show_progress=False,
        )
    wall = time.perf_counter() - t0
    n = len(rows)
    cost = cb.total_cost
    priced = "billed"
    if not cost and "4.1-mini" in model:  # langchain does not price OpenRouter model ids
        cost = cb.prompt_tokens / 1e6 * 0.40 + cb.completion_tokens / 1e6 * 1.60
        priced = "at gpt-4.1-mini list price"
    print()
    print(f"ragas ({model})  n={n}")
    print(
        f"  llm requests {cb.successful_requests:,}  ({cb.successful_requests / n:.1f} per sample)  + embeddings"
    )
    print(
        f"  input tokens {cb.prompt_tokens:,}  ({cb.prompt_tokens / n:,.0f} per sample)   "
        f"output tokens {cb.completion_tokens:,}  ({cb.completion_tokens / n:,.0f} per sample)"
    )
    print(f"  cost ${cost:.4f} {priced}  (${cost / n * 1000:.2f} per 1k samples)")
    print(f"  wall {wall:.1f}s")
    print(f"  {res}")
