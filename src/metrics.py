import numpy as np
import textstat
from sentence_transformers import SentenceTransformer, util

from src.config import EMBEDDING_MODEL_NAME

_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embed_model


def cosine_similarity(prediction, ground_truth):
    """Return cosine similarity between the embeddings of two texts."""
    model = _get_embed_model()
    emb_p = model.encode(prediction, convert_to_tensor=True)
    emb_g = model.encode(ground_truth, convert_to_tensor=True)
    return util.cos_sim(emb_p, emb_g).item()


def readability_ari(text):
    """Return the Automated Readability Index for *text*."""
    return textstat.automated_readability_index(text)


def simple_token_metrics(prediction, ground_truth):
    """Token-level precision, recall and exact-match accuracy."""
    pred_tokens = set(prediction.lower().split())
    gt_tokens = set(ground_truth.lower().split())

    tp = len(pred_tokens & gt_tokens)

    precision = tp / len(pred_tokens) if pred_tokens else 0
    recall = tp / len(gt_tokens) if gt_tokens else 0
    accuracy = 1 if prediction.strip().lower() == ground_truth.strip().lower() else 0

    return precision, recall, accuracy


def compute_all_metrics(predictions, ground_truths):
    """Compute every metric for a list of (prediction, ground_truth) pairs.

    Returns a dict with per-item lists and averaged scalars.
    """
    similarities = []
    ari_scores = []
    precisions = []
    recalls = []
    accuracies = []

    for pred, gt in zip(predictions, ground_truths):
        similarities.append(cosine_similarity(pred, gt))
        ari_scores.append(readability_ari(pred))

        p, r, a = simple_token_metrics(pred, gt)
        precisions.append(p)
        recalls.append(r)
        accuracies.append(a)

    return {
        "per_item": {
            "answer_similarity": similarities,
            "ari_grade": ari_scores,
            "precision": precisions,
            "recall": recalls,
            "accuracy": accuracies,
        },
        "averages": {
            "answer_similarity": float(np.mean(similarities)),
            "ari_grade": float(np.mean(ari_scores)),
            "precision": float(np.mean(precisions)),
            "recall": float(np.mean(recalls)),
            "accuracy": float(np.mean(accuracies)),
        },
    }
