#!/usr/bin/env python3
"""Batch-evaluate the NASA mission RAG system against test_questions.json.

For every question in the dataset this retrieves context, generates an answer,
scores it with RAGAS, then prints per-question and aggregate metrics and writes
evaluation_results.json.
"""

import argparse
import json
import math
import os
from pathlib import Path
from statistics import fmean

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import SemanticSimilarity

import llm_client
import rag_client
import ragas_evaluator

PROJECT_DIRECTORY = Path(__file__).resolve().parent
CORE_METRICS = {
    "answer_relevancy",
    "faithfulness",
    "llm_context_precision_without_reference",
}

load_dotenv()


def score_semantic_similarity(rows, openai_key):
    """Score every generated answer against its dataset reference answer."""
    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_key)
    )
    metric = SemanticSimilarity(embeddings=embeddings)
    answered = [row for row in rows if row["generated_answer"]]
    if not answered:
        return
    dataset = EvaluationDataset(
        samples=[
            SingleTurnSample(
                response=row["generated_answer"], reference=row["reference_answer"]
            )
            for row in answered
        ]
    )
    result = evaluate(
        dataset=dataset,
        metrics=[metric],
        embeddings=embeddings,
        show_progress=False,
    )
    for row, score in zip(answered, result[metric.name]):
        row["scores"][metric.name] = float(score)


def json_safe(value):
    """Replace NaN/inf with null so the saved report is valid JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(
        description="Batch-evaluate the NASA mission RAG system"
    )
    parser.add_argument(
        "--dataset", type=Path, default=PROJECT_DIRECTORY / "test_questions.json"
    )
    parser.add_argument(
        "--chroma-dir", type=Path, default=PROJECT_DIRECTORY / "chroma_db_openai"
    )
    parser.add_argument("--collection-name", default="nasa_space_missions_text")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model", default="gpt-3.5-turbo")
    parser.add_argument(
        "--output", type=Path, default=PROJECT_DIRECTORY / "evaluation_results.json"
    )
    args = parser.parse_args()

    openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("CHROMA_OPENAI_API_KEY")
    if not openai_key:
        raise SystemExit("Set OPENAI_API_KEY (for example in .env) before running.")
    # The persisted collection embeds queries with OpenAI, which reads this variable.
    os.environ["CHROMA_OPENAI_API_KEY"] = openai_key

    records = json.loads(args.dataset.read_text(encoding="utf-8"))
    print(f"Loaded {len(records)} questions from {args.dataset}")

    collection, connected, error = rag_client.initialize_rag_system(
        str(args.chroma_dir), args.collection_name
    )
    if not connected:
        raise SystemExit(f"ChromaDB unavailable: {error}")

    rows = []
    for position, record in enumerate(records, start=1):
        print(f"[{position}/{len(records)}] {record['id']}")
        try:
            retrieval = rag_client.retrieve_documents(
                collection, record["question"], args.top_k, record["mission"]
            )
            contexts = retrieval["documents"][0]
            context = rag_client.format_context(contexts, retrieval["metadatas"][0])
            answer = llm_client.generate_response(
                openai_key, record["question"], context, [], args.model
            )
            scores = {}
            for attempt in range(2):
                try:
                    scores = ragas_evaluator.evaluate_response_quality(
                        record["question"], answer, contexts
                    )
                except Exception as error:
                    scores = {"error": str(error)}
                complete = "error" not in scores and all(
                    isinstance(scores.get(name), float)
                    and math.isfinite(scores[name])
                    for name in CORE_METRICS
                )
                if complete:
                    break
                if attempt == 0:
                    print("    incomplete RAGAS scores; retrying once")
        except Exception as error:  # one bad question must not void the whole run
            print(f"    failed: {error}")
            answer, contexts, scores = "", [], {"error": str(error)}
        rows.append(
            {
                **record,
                "generated_answer": answer,
                "retrieved_contexts": contexts,
                "scores": scores,
            }
        )

    try:
        score_semantic_similarity(rows, openai_key)
    except Exception as error:
        print(f"semantic similarity failed: {error}")

    print("\nPer-question metrics")
    for row in rows:
        print(f"\n{row['id']} ({row['category']}): {row['question']}")
        for name, value in sorted(row["scores"].items()):
            rendered = f"{value:.4f}" if isinstance(value, float) else value
            print(f"  {name}: {rendered}")

    print("\nAggregate metrics")
    aggregate = {}
    for name in sorted({name for row in rows for name in row["scores"]}):
        values = [
            row["scores"][name]
            for row in rows
            if isinstance(row["scores"].get(name), float)
            and math.isfinite(row["scores"][name])
        ]
        if not values:
            continue
        aggregate[name] = {
            "count": len(values),
            "mean": fmean(values),
            "minimum": min(values),
            "maximum": max(values),
        }
        print(
            f"  {name}: mean={fmean(values):.4f} min={min(values):.4f} "
            f"max={max(values):.4f} n={len(values)}"
        )

    report = {
        "questions_evaluated": len(rows),
        "results": rows,
        "aggregate_metrics": aggregate,
    }
    args.output.write_text(
        json.dumps(json_safe(report), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
