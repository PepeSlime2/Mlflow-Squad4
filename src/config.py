import os

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama3.1:8b")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

PROMPTS = {
    "v1": "{pergunta}",
    "v2": "Responda de forma técnica e objetiva:\n\n{pergunta}",
    "v3": "Você é um engenheiro agrônomo especialista. Responda com precisão técnica:\n\n{pergunta}",
}
