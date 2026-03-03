import mlflow
import pandas as pd

from src.inference import ollama_predict
from src.metrics import compute_all_metrics


def run_evaluation(eval_data, model, prompt_template, prompt_version, run_name=None):
    """Run the full evaluation pipeline: inference → metrics → MLflow logging.

    Parameters
    ----------
    eval_data : pd.DataFrame
        Must contain ``inputs`` and ``ground_truth`` columns.
    model : str
        Ollama model tag (e.g. ``"llama3.1:8b"``).
    prompt_template : str
        Prompt string with a ``{pergunta}`` placeholder.
    prompt_version : str
        Label logged as an MLflow parameter (e.g. ``"v3"``).
    run_name : str, optional
        Name shown in the MLflow UI.  Defaults to ``"<model>_<prompt_version>"``.
    """
    run_name = run_name or f"{model}_{prompt_version}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model_name", model)
        mlflow.log_param("prompt_version", prompt_version)
        mlflow.log_param("dataset_size", len(eval_data))

        predictions = ollama_predict(eval_data, prompt_template, model=model)

        results = compute_all_metrics(predictions, eval_data["ground_truth"])

        for name, value in results["averages"].items():
            mlflow.log_metric(name, value)

        eval_table = pd.DataFrame(
            {
                "inputs": eval_data["inputs"],
                "ground_truth": eval_data["ground_truth"],
                "prediction": predictions,
                **results["per_item"],
            }
        )
        mlflow.log_table(eval_table, "eval/evaluation_table.json")

        for i, out in enumerate(predictions):
            mlflow.log_text(out, f"outputs/output_{i}.txt")

    return results
