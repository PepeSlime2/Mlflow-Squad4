#!/usr/bin/env python3
"""CLI entry point for running LLM benchmarks with MLflow tracking."""

import argparse
import os

import mlflow

from src.config import DEFAULT_MODEL, MLFLOW_TRACKING_URI, PROMPTS
from src.data_loader import load_qa_dataset
from src.evaluation import run_evaluation


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark LLM models on agricultural Q&A using MLflow.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama model tag (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--prompt",
        default="v3",
        choices=list(PROMPTS.keys()),
        help="Prompt version to use (default: v3)",
    )
    parser.add_argument(
        "--data",
        default="citrus_sugarcane_qa.json",
        help="JSON filename inside data/ (default: citrus_sugarcane_qa.json)",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Custom MLflow run name (default: <model>_<prompt>)",
    )
    args = parser.parse_args()

    os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    eval_data = load_qa_dataset(args.data)
    prompt_template = PROMPTS[args.prompt]

    print(f"Model:   {args.model}")
    print(f"Prompt:  {args.prompt}")
    print(f"Dataset: {args.data} ({len(eval_data)} questions)")
    print()

    results = run_evaluation(
        eval_data,
        model=args.model,
        prompt_template=prompt_template,
        prompt_version=args.prompt,
        run_name=args.run_name,
    )

    print("Results (averages):")
    for name, value in results["averages"].items():
        print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    main()
