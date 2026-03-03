import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_qa_dataset(filename):
    """Load a Q&A JSON file from ``data/`` and return a DataFrame.

    The JSON must be a list of objects with ``input`` and ``ground_truth`` keys.
    The returned DataFrame has columns ``inputs`` and ``ground_truth`` to stay
    compatible with the rest of the pipeline.
    """
    path = DATA_DIR / filename
    with open(path, encoding="utf-8") as f:
        records = json.load(f)

    df = pd.DataFrame(records)
    df = df.rename(columns={"input": "inputs"})
    return df
