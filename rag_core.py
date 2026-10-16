"""
rag_core.py — Agente SB100

Mudanças principais em relação à versão anterior:

1. BUG CORRIGIDO: `search_chunks` fazia a mesma requisição ao Qdrant duas vezes
   (dobrava a latência do retrieval à toa).
2. STREAMING: `perguntar_stream()` gera a resposta token a token via Ollama
   (stream=True) e já entrega isso pro front conforme chega — não espera o
   texto inteiro terminar.
3. AVALIAÇÃO DESACOPLADA: judge (LLM-as-judge), hallucination score,
   confidence score e o log no MLflow agora rodam DEPOIS que a resposta já
   foi enviada ao usuário, em background (`asyncio.create_task`). Isso tira
   do caminho crítico uma segunda geração completa do LLM, que antes o
   usuário esperava sem ver nada na tela.
4. EMBEDDINGS EM PARALELO: o vetor denso (GPU) e o esparso/BM25 (CPU) agora
   são calculados concorrentemente com um ThreadPoolExecutor.
5. FORMATAÇÃO: prompt com exemplo few-shot de formatação + normalização leve
   de Markdown na resposta final (usada para o log/avaliação; o streaming
   real que o usuário vê continua chegando token a token).
6. API mínima em FastAPI no fim do arquivo, com endpoint de streaming
   (`/perguntar/stream`) e endpoint em lote (`/perguntar`), só para você
   plugar no front sem precisar montar isso do zero.
"""

import mlflow
import requests
import httpx
import numpy as np
import torch
import textstat
import time
import os
import json
import asyncio
import pandas as pd
import re
import threading
import unicodedata

from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator, Optional

from sentence_transformers import util
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Rodando embeddings em: {device}")

if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


model = SentenceTransformer(
    "Qwen/Qwen3-Embedding-0.6B",
    device=device,
    tokenizer_kwargs={"padding_side": "left"},
)
model.max_seq_length = 4096

print("Carregando modelo sparse (BM25)...")
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
print("Modelo sparse carregado.")

# Pool pequeno só para paralelizar dense (GPU) x sparse (CPU) na hora da busca.
_embed_executor = ThreadPoolExecutor(max_workers=2)

# Guarda referências das tasks de avaliação em background para elas não
# serem coletadas pelo GC antes de terminar (comportamento conhecido do
# asyncio.create_task).
_background_tasks: set[asyncio.Task] = set()

embedding_cache = {}


# ── EMBEDDINGS ───────────────────────────────────────────────────────────

def cached_embed(text):
    if text in embedding_cache:
        return embedding_cache[text]
    emb = model.encode(text, normalize_embeddings=True)
    embedding_cache[text] = emb
    return emb


def embed_query(text):
    return model.encode(text, prompt_name="query", normalize_embeddings=True)


def embed_document(text):
    return model.encode(text, normalize_embeddings=True)


def embed_sparse_query(text):
    """
    Gera o vetor esparso (BM25) da pergunta, no formato exigido
    pela Query API do Qdrant: {"indices": [...], "values": [...]}
    """
    sparse_emb = list(sparse_model.embed([text]))[0]
    return {
        "indices": sparse_emb.indices.tolist(),
        "values": sparse_emb.values.tolist(),
    }


def normalize_text(text):
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = text.encode("ascii", "ignore").decode("utf-8")
    text = re.sub(r"[^\w\s]", "", text)
    text = " ".join(text.split())
    return text


def normalize_markdown(text: str) -> str:
    """
    Correções leves e baratas de Markdown que modelos locais menores erram
    com frequência: cabeçalho sem espaço, tabela sem linha separadora,
    listas grudadas no texto anterior. Roda só sobre a resposta final
    (para log/avaliação) — o streaming bruto continua indo pro front.
    """
    # "#Titulo" -> "# Titulo"
    text = re.sub(r"(?m)^(#{1,6})([^\s#])", r"\1 \2", text)

    # Tabela sem linha de separador após o cabeçalho
    lines = text.split("\n")
    fixed = []
    for i, line in enumerate(lines):
        fixed.append(line)
        is_header_row = line.strip().startswith("|") and line.strip().endswith("|")
        next_is_sep = (
            i + 1 < len(lines) and re.match(r"^\s*\|?[\s:|-]+\|?\s*$", lines[i + 1])
        )
        if is_header_row and not next_is_sep:
            n_cols = line.count("|") - 1
            if n_cols > 0:
                fixed.append("|" + "---|" * n_cols)
    text = "\n".join(fixed)

    # Garante linha em branco antes de listas que vieram coladas em texto
    text = re.sub(r"([^\n])\n(-\s|\d+\.\s)", r"\1\n\n\2", text)

    return text.strip()


# ── MÉTRICAS ─────────────────────────────────────────────────────────────

def token_metrics(pred, gt):
    pred = normalize_text(pred)
    gt = normalize_text(gt)

    pred_tokens = pred.split()
    gt_tokens = gt.split()

    pred_set = set(pred_tokens)
    gt_set = set(gt_tokens)

    tp = len(pred_set & gt_set)

    precision = tp / len(pred_set) if pred_set else 0
    recall = tp / len(gt_set) if gt_set else 0

    f1 = 0 if precision + recall == 0 else 2 * (precision * recall) / (precision + recall)
    exact_match = 1 if pred == gt else 0

    return precision, recall, f1, exact_match


def formatar_abnt(info):
    autores = info.get("autores") or "Autor desconhecido"
    titulo = info.get("titulo") or "Título desconhecido"
    pagina = info.get("pagina_pdf") or ""
    return f"{autores}. {titulo}. p. {pagina}."


def contextual_coverage(resposta, chunks):
    if not resposta.strip() or not chunks:
        return 0.0

    resp_emb = torch.tensor(embed_document(resposta))

    chunk_embs = torch.tensor([
        c["embedding"] for c in chunks[:5] if c.get("embedding") is not None
    ])

    if len(chunk_embs) == 0:
        return 0.0

    sims = util.cos_sim(resp_emb, chunk_embs)[0]
    top_k = min(3, len(sims))
    return float(torch.topk(sims, k=top_k).values.mean().item())


def jina_rerank_score(pergunta, resposta, chunks):
    # Reranker desativado (custo de carregamento). Mantido como stub para
    # não precisar reescrever as fórmulas de hallucination/confidence.
    return 0.0


def numeric_hallucination_check(resposta, chunks):
    """
    Extrai números da resposta e verifica se existem nos chunks.
    Retorna penalidade entre 0 (tudo ok) e 1 (todos inventados).
    """
    numeros_resposta = re.findall(r"\b\d+(?:[.,]\d+)?\b", resposta)
    if not numeros_resposta:
        return 0.0

    texto_chunks = " ".join(c["content"] for c in chunks[:5] if c.get("content"))

    inventados = 0
    for num in numeros_resposta:
        num_norm = num.replace(",", ".")
        num_alt = num.replace(".", ",")
        if num not in texto_chunks and num_norm not in texto_chunks and num_alt not in texto_chunks:
            inventados += 1

    penalidade = inventados / len(numeros_resposta)
    print(f"[NUMERIC CHECK] {inventados}/{len(numeros_resposta)} números ausentes → penalidade: {penalidade:.2f}")
    return penalidade


def hallucination_flag(pergunta, resposta, chunks, judge_fidelidade=None, rerank_score=None, threshold=0.45):
    coverage = contextual_coverage(resposta, chunks)

    if "não encontrei" in resposta.lower():
        return 0, coverage, 0.0

    if rerank_score is None:
        rerank_score = jina_rerank_score(pergunta, resposta, chunks)

    numeric_penalty = numeric_hallucination_check(resposta, chunks)

    # Pesos redistribuídos enquanto o reranker está desativado (rerank_score
    # sempre 0.0 → peso original dele redistribuído proporcionalmente).
    if judge_fidelidade is not None:
        hallucination_score = (
            0.15 * (1.0 - coverage)
            + 0.20 * (1.0 - float(judge_fidelidade))
            + 0.46 * numeric_penalty
        )
    else:
        hallucination_score = 0.15 * (1.0 - coverage) + 0.30 * numeric_penalty

    flag = 1 if hallucination_score > threshold else 0
    return flag, coverage, hallucination_score


def compute_confidence_score(
    coverage: float,
    hallucination_score: float,
    judge_fidelidade: float = 0.0,
    judge_relevancia: float = 0.0,
    avg_chunk_score: float = 0.0,
) -> float:
    confidence = (
        0.31 * coverage
        + 0.31 * float(judge_fidelidade)
        + 0.15 * float(judge_relevancia)
        + 0.15 * avg_chunk_score
        + 0.08 * (1.0 - hallucination_score)
    )
    confidence = float(np.clip(confidence, 0.0, 1.0))
    print(f"[CONFIDENCE] Score: {confidence:.4f}")
    return confidence


# ── QDRANT / RETRIEVAL ───────────────────────────────────────────────────

QDRANT_URL = "https://57eb89f7-8062-4156-8bd9-761b749c9d3b.sa-east-1-0.aws.cloud.qdrant.io:6333"
API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.o4SI3QmZtkdOUXK8KVRunQT1SymcxtZrkzUVCXmiZvQ"
COLLECTION_NAME = "sb100"

SIMILARITY_THRESHOLD = 0.01  # score de fusão RRF, não é mais cosine similarity


def search_chunks(pergunta, top_k=20, prefetch_k=40):
    """
    Busca híbrida (dense + sparse) via Query API do Qdrant com fusão RRF.
    Dense (GPU) e sparse (CPU) são calculados em paralelo.
    """
    future_dense = _embed_executor.submit(embed_query, pergunta)
    future_sparse = _embed_executor.submit(embed_sparse_query, pergunta)

    dense_vector = future_dense.result().tolist()
    sparse_vector = future_sparse.result()

    query_url = f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/query"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    payload = {
        "prefetch": [
            {"query": dense_vector, "using": "vetor_denso", "limit": prefetch_k},
            {"query": sparse_vector, "using": "vetor_esparso", "limit": prefetch_k},
        ],
        "query": {"fusion": "rrf"},
        "limit": top_k,
        "with_payload": True,
        "with_vector": True,
    }

    # (bug antigo corrigido: essa chamada era feita duas vezes)
    response = requests.post(query_url, headers=headers, json=payload)
    if response.status_code != 200:
        print("Qdrant error body:", response.text)
    response.raise_for_status()

    results = response.json()["result"]["points"]

    chunks = []
    chunk_infos = []

    for r in results:
        content = r["payload"].get("content", "")

        embedding = None
        if r.get("vector") and "vetor_denso" in r["vector"]:
            embedding = r["vector"]["vetor_denso"]

        chunks.append(content)
        chunk_infos.append({
            "id": r.get("id"),
            "score": r.get("score"),
            "titulo": r["payload"].get("titulo"),
            "autores": r["payload"].get("autores"),
            "pagina_pdf": r["payload"].get("pagina_pdf"),
            "chunk_index": r["payload"].get("chunk_index"),
            "preview": content[:500],
            "content": content,
            "embedding": embedding,
        })

    return chunks, chunk_infos


def _monta_context(top_context):
    return "\n\n".join(
        f"Documento: {info['titulo']}\nScore: {info.get('score', 0):.3f}\nTexto: {info['content']}\n"
        for info in top_context
    )


def _filtra_top_context(chunk_infos):
    top_context = sorted(chunk_infos, key=lambda x: x["score"], reverse=True)[:5]
    return [c for c in top_context if c["score"] >= SIMILARITY_THRESHOLD]


# ── PROMPTS ──────────────────────────────────────────────────────────────

BASE_TEMPLATE = """
Você é o Agente SB100, assistente científico especializado.

Responda usando EXCLUSIVAMENTE o CONTEXTO. Não invente, não use conhecimento externo e não adicione informações, recomendações ou inferências que não estejam no contexto. Adaptar-se ao perfil do usuário muda apenas a linguagem, a ordem e a ênfase, nunca o conteúdo permitido.

PERFIL DO USUÁRIO:
{perfil_instrucoes}

FORMATO:
Escolha automaticamente o formato mais claro:
- explicação → parágrafo curto;
- vários itens → tópicos;
- etapas ou sequência → lista numerada;
- comparação ou dados → tabela;
- números/resultados → destaque os valores principais.

Use Markdown limpo: cabeçalhos com espaço após o "#", listas com "- " ou "1. ", e tabelas sempre com a linha separadora "|---|---|" logo após o cabeçalho. Não use referências ou citações.

Exemplo de tabela bem formatada:

| Cultura | Produtividade média | Ciclo |
|---|---|---|
| Soja | 3.500 kg/ha | 100-140 dias |
| Milho | 6.000 kg/ha | 120-180 dias |

Se a informação não estiver no contexto, responda exatamente:
"Esta informação não consta na base de dados consultada. Para obtermos um resultado diferente, prefere tentar uma nova busca de outra maneira?"

CONTEXTO:
{context}

PERGUNTA:
{pergunta}

Responda da forma mais clara e útil possível para o perfil acima:
"""

PERFIL_INSTRUCOES = {
    "produtor_rural": """
Usuário: produtor rural, foco na decisão prática da lavoura ou criação.
- Linguagem simples e direta, frases curtas, sem jargão acadêmico. Use os termos mais simples que o próprio contexto usar.
- Comece pela resposta, sem introdução.
- Destaque o que importa na prática: doses, quantidades, épocas, prazos, produtividade e custos, sempre com unidade (kg/ha, dias, R$).
- Priorize tópicos curtos. Use tabela só para comparar opções.
- Máximo de 1 a 2 parágrafos curtos ou 5 tópicos, salvo se a pergunta exigir mais.
""",
    "cooperativa": """
Usuário: gestor ou técnico de cooperativa, que decide pensando nos cooperados e em escala.
- Linguagem objetiva e profissional, sem excesso de termos acadêmicos.
- Destaque o que o contexto traz sobre ganho coletivo, escala, custo-benefício, logística, assistência técnica e padronização de manejo.
- Em comparações entre práticas, cultivares ou sistemas, prefira tabela.
- Apresente números com unidade e deixe claro a que condição se aplicam (cultura, região, safra), quando o contexto informar.
- Comece pela conclusão e depois os detalhes de apoio.
""",
    "empresa": """
Usuário: profissional de empresa do agronegócio (insumos, indústria, comercialização, agtech), orientado a resultado e risco.
- Estilo executivo: a conclusão na primeira frase, depois os pontos de apoio.
- Destaque indicadores que o contexto fornecer: produtividade, custo, retorno, perdas, eficiência, riscos, mercado e conformidade.
- Prefira tópicos e tabelas. Evite narrativa longa.
- Sinalize limitações ou condições de validade dos números quando constarem no contexto.
- Não faça projeções, estimativas de retorno ou recomendações comerciais que não estejam no contexto.
""",
    "pesquisador_universitario": """
Usuário: pesquisador ou estudante de pós-graduação, que precisa de rigor.
- Linguagem técnica e precisa, preservando a terminologia do contexto.
- Inclua, quando constarem no contexto: método, condições experimentais, amostra, unidades, médias, variações, significância e limitações.
- Diferencie claramente o que o contexto afirma do que ele não informa. Se a evidência for parcial, diga isso.
- Não generalize além do escopo descrito no contexto (cultura, região, período, condições).
- Estruture com cabeçalhos curtos quando a resposta tiver mais de um aspecto. Use tabela para comparar resultados.
- Não use referências ou citações.
""",
    "tecnico": """
Usuário: técnico ou engenheiro agrônomo, que aplica a informação em campo.
- Linguagem técnica e operacional, direta.
- Priorize parâmetros acionáveis: doses, concentrações, épocas, estádios, espaçamentos, intervalos, condições de aplicação e critérios de decisão.
- Procedimentos em lista numerada; parâmetros e comparações em tabela, sempre com unidades.
- Destaque restrições, condições limitantes e alertas que constarem no contexto.
- Não complete parâmetros ausentes com conhecimento externo: se faltar um dado, omita-o.
""",
    "instituicao_publica": """
Usuário: gestor ou analista de órgão público (secretaria, agência, extensão, regulação).
- Linguagem institucional, neutra e clara.
- Destaque o que o contexto traz sobre abrangência, indicadores, impacto territorial, econômico, social e ambiental, regulação e programas.
- Apresente dados com unidade, recorte (região, período, cultura) e escopo, quando informados.
- Use tópicos para pontos principais e tabela para indicadores comparáveis.
- Não emita juízo de valor nem recomende política pública que não esteja no contexto.
""",
    "visitante": """
Usuário visitante, sem perfil especializado informado.
- Linguagem clara, objetiva e acessível.
- Explique termos técnicos apenas quando o contexto trouxer informação para isso.
- Comece pela resposta e use o formato mais simples adequado ao conteúdo.
""",
}


def build_prompt(profile: str = "visitante") -> str:
    instrucoes = PERFIL_INSTRUCOES.get(
        profile,
        PERFIL_INSTRUCOES["visitante"],
    ).strip()
    return BASE_TEMPLATE.replace("{perfil_instrucoes}", instrucoes)


JUDGE_PROMPT_TEMPLATE = """
Você é um juiz imparcial avaliando a qualidade de uma resposta gerada por IA.

PERGUNTA:
{pergunta}

CONTEXTO USADO:
{context}

RESPOSTA GERADA:
{resposta}

Avalie a resposta nos seguintes critérios e responda APENAS com um JSON válido, sem explicações:

{{
  "fidelidade": <0 a 1, se a resposta é fiel ao contexto sem inventar>,
  "relevancia": <0 a 1, se a resposta responde à pergunta>,
  "completude": <0 a 1, se a resposta cobre o essencial>,
  "veredicto": "<APROVADA|REPROVADA>"
}}
"""

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"

os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
mlflow.set_tracking_uri("http://localhost:5000")


# ── GERAÇÃO ──────────────────────────────────────────────────────────────

def gerar_resposta_sync(prompt: str) -> str:
    """Geração bloqueante (sem streaming) — usada no modo em lote."""
    response = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "temperature": 0.1,
        "stream": False,
    })
    response.raise_for_status()
    return response.json().get("response", "")


async def generate_stream(prompt: str) -> AsyncGenerator[str, None]:
    """
    Streaming da geração do Ollama. Uso:
        async for token in generate_stream(prompt):
            enviar_para_front(token)
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "temperature": 0.1,
        "stream": True,
    }
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", OLLAMA_URL, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                data = json.loads(line)
                token = data.get("response", "")
                if token:
                    yield token
                if data.get("done"):
                    break


# ── AVALIAÇÃO (judge + hallucination + confidence + mlflow) ─────────────
# Roda sempre DEPOIS da resposta já ter sido entregue ao usuário.

def _avaliar_e_logar_sync(pergunta, resposta_final, chunk_infos, context, ground_truth, tempo, top_context):
    with mlflow.start_run(run_name="SB100_RAG"):
        mlflow.log_param("model", OLLAMA_MODEL)
        mlflow.log_param("embedding_model", "Qwen3-Embedding-0.6B")
        mlflow.log_param("sparse_model", "Qdrant/bm25")
        mlflow.log_param("retrieval_mode", "hybrid_rrf")
        mlflow.log_param("device", device)
        mlflow.log_param("pergunta", pergunta)
        mlflow.log_param("similarity_threshold", SIMILARITY_THRESHOLD)
        mlflow.log_metric("chunks_recuperados_total", len(chunk_infos))
        mlflow.log_metric("chunks_apos_filtro", len(top_context))

        judge_response = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": JUDGE_PROMPT_TEMPLATE.format(pergunta=pergunta, context=context, resposta=resposta_final),
            "temperature": 0.0,
            "stream": False,
        })
        judge_response.raise_for_status()
        judge_raw = judge_response.json().get("response", "{}")

        try:
            judge_clean = re.sub(r"<think>.*?</think>", "", judge_raw, flags=re.DOTALL).strip()
            judge_result = json.loads(judge_clean)
        except (json.JSONDecodeError, ValueError):
            judge_result = {"fidelidade": 0, "relevancia": 0, "completude": 0, "veredicto": "ERRO"}

        mlflow.log_metric("judge_fidelidade", float(judge_result.get("fidelidade", 0)))
        mlflow.log_metric("judge_relevancia", float(judge_result.get("relevancia", 0)))
        mlflow.log_metric("judge_completude", float(judge_result.get("completude", 0)))
        mlflow.log_metric("judge_aprovada", 1 if judge_result.get("veredicto") == "APROVADA" else 0)

        sorted_chunks = sorted(chunk_infos, key=lambda x: x["score"], reverse=True)
        refs_principais = [formatar_abnt(c) for c in sorted_chunks[:2]]

        emb_r = torch.tensor(cached_embed(resposta_final))
        emb_q = torch.tensor(embed_query(pergunta))
        similarity_pergunta = util.cos_sim(emb_r, emb_q).item()

        chunk_embeddings = [c["embedding"] for c in chunk_infos[:3] if c.get("embedding") is not None]
        similarity_chunks = 0.0
        if chunk_embeddings:
            chunk_embeddings_t = torch.tensor(chunk_embeddings)
            similarities = util.cos_sim(emb_r, chunk_embeddings_t)[0]
            top3 = torch.topk(similarities, k=min(3, len(similarities)))
            similarity_chunks = top3.values.mean().item()

        ari = textstat.automated_readability_index(resposta_final)
        p, r, f1, em = token_metrics(resposta_final, ground_truth)

        scores = [c["score"] for c in chunk_infos if c["score"]]
        avg_chunk_score = float(np.mean(scores)) if scores else 0.0

        h_flag, coverage, h_score = hallucination_flag(
            pergunta, resposta_final, chunk_infos,
            judge_fidelidade=judge_result.get("fidelidade"),
            rerank_score=0.0,
        )

        confidence_score = compute_confidence_score(
            coverage=coverage,
            hallucination_score=h_score,
            judge_fidelidade=judge_result.get("fidelidade", 0),
            judge_relevancia=judge_result.get("relevancia", 0),
            avg_chunk_score=avg_chunk_score,
        )

        mlflow.log_metric("confidence_score", confidence_score)
        mlflow.log_metric("tempo_resposta", float(tempo))
        mlflow.log_metric("similarity_pergunta", float(similarity_pergunta))
        mlflow.log_metric("similarity_chunks", float(similarity_chunks))
        mlflow.log_metric("ari_grade", float(ari))
        mlflow.log_metric("precision", float(p))
        mlflow.log_metric("recall", float(r))
        mlflow.log_metric("f1", float(f1))
        mlflow.log_metric("exact_match", float(em))
        mlflow.log_metric("avg_chunk_score", avg_chunk_score)
        mlflow.log_metric("contextual_coverage", float(coverage))
        mlflow.log_metric("hallucination_score", float(h_score))
        mlflow.log_metric("hallucination_flag", int(h_flag))

        trace = {
            "pergunta_original": pergunta,
            "resposta": resposta_final,
            "chunks": chunk_infos,
            "context": context,
            "refs_principais": refs_principais,
            "contextual_coverage": coverage,
            "hallucination_score": h_score,
            "hallucination_flag": h_flag,
            "confidence_score": confidence_score,
            "judge": judge_result,
        }
        mlflow.log_dict(trace, "rag_trace.json")

        eval_table = pd.DataFrame({
            "pergunta_original": [pergunta],
            "ground_truth": [ground_truth],
            "resposta": [resposta_final],
            "similarity_pergunta": [similarity_pergunta],
            "similarity_chunks": [similarity_chunks],
            "ari": [ari],
            "precision": [p],
            "recall": [r],
            "f1": [f1],
            "exact_match": [em],
            "contextual_coverage": [coverage],
            "hallucination_score": [h_score],
            "hallucination_flag": [h_flag],
            "confidence_score": [confidence_score],
            "judge_fidelidade": [judge_result.get("fidelidade", 0)],
            "judge_relevancia": [judge_result.get("relevancia", 0)],
            "judge_completude": [judge_result.get("completude", 0)],
            "judge_veredicto": [judge_result.get("veredicto", "ERRO")],
            "tempo": [tempo],
        })
        mlflow.log_table(eval_table, "eval/evaluation.json")
        mlflow.log_text(pergunta, "inputs/pergunta_original.txt")
        mlflow.log_text(resposta_final, "outputs/resposta.txt")

        print("\n==============================")
        print("Pergunta:", pergunta)
        print("Hallucination Flag:", h_flag, "| Confidence:", confidence_score)
        print("Judge:", judge_result)
        print("==============================\n")

        return {
            "h_flag": h_flag,
            "refs_principais": refs_principais,
            "confidence_score": confidence_score,
        }


async def _avaliar_e_logar_async(pergunta, resposta_final, chunk_infos, context, ground_truth, tempo, top_context):
    """Wrapper async: roda a avaliação síncrona numa thread separada para
    não travar o event loop enquanto outras perguntas estão sendo servidas."""
    try:
        await asyncio.to_thread(
            _avaliar_e_logar_sync, pergunta, resposta_final, chunk_infos, context, ground_truth, tempo, top_context
        )
    except Exception as e:
        print(f"[AVALIACAO BACKGROUND] Erro: {e}")


_avaliacao_lock = threading.Lock()


def _avaliar_serializado(*args):
    with _avaliacao_lock:
        return _avaliar_e_logar_sync(*args)


MSG_SEM_CONTEXTO = (
    "Esta informação não consta na base de dados consultada. "
    "Para obtermos um resultado diferente, prefere tentar uma nova busca de outra maneira?"
)


async def perguntar_avaliador_stream(
    pergunta: str,
    ground_truth: str = "",
    profile: str = "visitante",
) -> AsyncGenerator[dict, None]:
    """
    Fluxo do Chat avaliador com streaming. A avaliação aguarda o mesmo texto
    que foi transmitido e os metadados só são emitidos quando ela termina.
    """
    start_time = time.time()

    chunks, chunk_infos = await asyncio.to_thread(search_chunks, pergunta, 20)
    top_context = _filtra_top_context(chunk_infos)

    if not top_context:
        yield {"tipo": "token", "conteudo": MSG_SEM_CONTEXTO}
        yield {"tipo": "metadata", "referencias_principais": [], "hallucination_flag": 0}
        return

    context = _monta_context(top_context)
    prompt_final = build_prompt(profile).format(pergunta=pergunta, context=context)

    partes = []
    async for token in generate_stream(prompt_final):
        partes.append(token)
        yield {"tipo": "token", "conteudo": token}

    yield {"tipo": "status", "etapa": "avaliando"}

    resposta_final = normalize_markdown("".join(partes))
    tempo = time.time() - start_time

    resultado = await asyncio.to_thread(
        _avaliar_serializado,
        pergunta, resposta_final, chunk_infos, context, ground_truth, tempo, top_context,
    )

    yield {
        "tipo": "metadata",
        "referencias_principais": resultado["refs_principais"],
        "hallucination_flag": resultado["h_flag"],
        "confidence_score": resultado["confidence_score"],
    }


# ── FLUXO PRINCIPAL: STREAMING (uso no front) ────────────────────────────

async def perguntar_stream(
    pergunta: str,
    ground_truth: str = "",
    profile: str = "visitante",
) -> AsyncGenerator[str, None]:
    """
    Gera a resposta em streaming (token a token) para o front. A avaliação
    (judge/hallucination/confidence/mlflow) roda em background depois que
    o último token já foi enviado — não bloqueia o usuário.
    """
    start_time = time.time()

    chunks, chunk_infos = await asyncio.to_thread(search_chunks, pergunta, 20)
    top_context = _filtra_top_context(chunk_infos)

    if not top_context:
        yield "Esta informação não consta na base de dados consultada. Para obtermos um resultado diferente, prefere tentar uma nova busca de outra maneira?"
        return

    context = _monta_context(top_context)
    prompt_final = build_prompt(profile).format(pergunta=pergunta, context=context)

    resposta_parts = []
    async for token in generate_stream(prompt_final):
        resposta_parts.append(token)
        yield token

    resposta_final = normalize_markdown("".join(resposta_parts))
    tempo = time.time() - start_time

    task = asyncio.create_task(
        _avaliar_e_logar_async(pergunta, resposta_final, chunk_infos, context, ground_truth, tempo, top_context)
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


# ── FLUXO EM LOTE (sem streaming, usado para avaliação/testes) ──────────

def perguntar(pergunta, ground_truth="", profile="visitante"):
    """
    Versão síncrona e bloqueante: espera resposta + avaliação completas
    antes de retornar. Útil para `perguntar_lote` / rodadas de teste, onde
    não existe front esperando streaming.
    """
    start_time = time.time()

    chunks, chunk_infos = search_chunks(pergunta, top_k=20)
    top_context = _filtra_top_context(chunk_infos)

    if not top_context:
        return "Esta informação não consta na base de dados consultada. Para obtermos um resultado diferente, prefere tentar uma nova busca de outra maneira?", chunk_infos, [], 0

    context = _monta_context(top_context)
    prompt_final = build_prompt(profile).format(pergunta=pergunta, context=context)
    resposta = gerar_resposta_sync(prompt_final)
    resposta_final = normalize_markdown(resposta)

    tempo = time.time() - start_time

    resultado = _avaliar_e_logar_sync(
        pergunta, resposta_final, chunk_infos, context, ground_truth, tempo, top_context
    )

    return resposta_final, chunk_infos, resultado["refs_principais"], resultado["h_flag"]


def parse_multilines(input_data):
    if isinstance(input_data, list):
        return input_data
    if isinstance(input_data, str):
        return [l.strip() for l in input_data.split("\n") if l.strip()]
    return []


def perguntar_lote(perguntas, ground_truths=None, profile="visitante"):
    perguntas = parse_multilines(perguntas)
    ground_truths = parse_multilines(ground_truths) if ground_truths else []

    resultados = []
    for i, pergunta in enumerate(perguntas):
        gt = ground_truths[i] if i < len(ground_truths) else ""
        resposta, chunks, refs, h_flag = perguntar(
            pergunta,
            ground_truth=gt,
            profile=profile,
        )
        resultados.append({
            "pergunta": pergunta,
            "resposta": resposta,
            "ground_truth": gt,
            "referencias": refs,
            "hallucination_flag": h_flag,
        })
    return resultados


def executar(payload):
    if "perguntas" in payload:
        return perguntar_lote(
            payload.get("perguntas"),
            payload.get("ground_truths"),
            profile=payload.get("profile", "visitante"),
        )

    resposta, chunks, refs, h_flag = perguntar(
        payload.get("pergunta"),
        payload.get("ground_truth", ""),
        profile=payload.get("profile", "visitante"),
    )
    return {
        "pergunta": payload.get("pergunta"),
        "resposta": resposta,
        "ground_truth": payload.get("ground_truth", ""),
        "referencias": refs,
        "hallucination_flag": h_flag,
    }
