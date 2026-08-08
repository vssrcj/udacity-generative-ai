import os

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from typing import Dict, List

load_dotenv()

# RAGAS imports
try:
    from ragas import EvaluationDataset, SingleTurnSample
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithoutReference,
        ResponseRelevancy
    )
    from ragas import evaluate
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

def evaluate_response_quality(question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics"""
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}

    openai_key = (
        os.getenv("CHROMA_OPENAI_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )

    # Create evaluator LLM with model gpt-3.5-turbo
    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(model="gpt-3.5-turbo", api_key=openai_key)
    )

    # Create evaluator_embeddings with model test-embedding-3-small
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=openai_key
        )
    )

    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts
    )
    dataset = EvaluationDataset(samples=[sample])

    # Define an instance for each metric to evaluate
    metrics = [
        ResponseRelevancy(
            llm=evaluator_llm,
            embeddings=evaluator_embeddings
        ),
        Faithfulness(llm=evaluator_llm),
        LLMContextPrecisionWithoutReference(llm=evaluator_llm)
    ]

    # Evaluate the response using the metrics
    evaluation = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        show_progress=False
    )

    # Return the evaluation results
    return {
        metric.name: float(evaluation[metric.name][0])
        for metric in metrics
    }
