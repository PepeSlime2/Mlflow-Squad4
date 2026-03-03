import json

import requests

from src.config import DEFAULT_MODEL, OLLAMA_URL


def ollama_predict(df, prompt_template, model=None, url=None):
    """Send each question in *df* to Ollama and return a list of completions."""
    model = model or DEFAULT_MODEL
    url = url or OLLAMA_URL
    outputs = []

    for _, row in df.iterrows():
        payload = {
            "model": model,
            "prompt": prompt_template.format(pergunta=row["inputs"]),
        }

        r = requests.post(url, json=payload)
        r.raise_for_status()

        completion = ""
        for line in r.text.strip().split("\n"):
            completion += json.loads(line).get("response", "")

        outputs.append(completion)

    return outputs
