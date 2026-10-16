import mlflow
import pandas as pd

mlflow.set_tracking_uri("http://localhost:5000")

exp = mlflow.get_experiment_by_name(
    "SB100-TestesWebLlama-1024"
)

runs = mlflow.search_runs(
    experiment_ids=[exp.experiment_id],
    output_format="pandas"
)

runs.to_csv("NoReranker.csv", index=False)

print(f"{len(runs)} runs exportadas")