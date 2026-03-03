# NLP MLflow — Benchmark de LLMs em Agricultura Brasileira

Avaliação de modelos LLM (via Ollama) em perguntas e respostas sobre citricultura e cana-de-açúcar, com tracking de experimentos via MLflow.

## Arquitetura

```
Ollama (local)          MLflow UI (localhost:5000)
     ↑                        ↑
     │  POST /api/generate     │  log_metric / log_table
     │                        │
  run_benchmark.py  ──────────┘
     │
     ├── src/inference.py    → chamadas ao Ollama
     ├── src/metrics.py      → similaridade coseno, ARI, precision/recall
     ├── src/evaluation.py   → pipeline completo + logging MLflow
     ├── src/data_loader.py  → carrega JSON → DataFrame
     └── src/config.py       → URLs, prompts, defaults
```

## Quick Start

```bash
# 1. Criar e ativar ambiente virtual
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Iniciar MLflow UI (em outro terminal)
mlflow ui

# 4. Garantir que o Ollama está rodando e baixar um modelo
ollama pull llama3.1:8b

# 5. Rodar benchmark
python run_benchmark.py --model llama3.1:8b --prompt v3 --data citrus_sugarcane_qa.json
```

## Uso via CLI

```bash
# Dataset reduzido (15 perguntas, apenas citros)
python run_benchmark.py --model gemma3:4b --prompt v2 --data citrus_qa.json

# Dataset completo (62 perguntas, citros + cana)
python run_benchmark.py --model llama3.1:8b --prompt v3

# Nome customizado para o run
python run_benchmark.py --model qwen3:8b --prompt v1 --run-name "qwen3_teste_final"
```

## Uso via Notebook

Abra `notebooks/benchmark.ipynb` e altere as variáveis de configuração na célula indicada.

## Datasets

| Arquivo | Perguntas | Domínio |
|---|---|---|
| `citrus_qa.json` | 15 | Citricultura |
| `citrus_sugarcane_qa.json` | 62 | Citricultura + Cana-de-açúcar |

Para adicionar um novo dataset, crie um JSON em `data/` com a estrutura:

```json
[
  {"input": "Pergunta aqui?", "ground_truth": "Resposta esperada."},
  ...
]
```

## Prompts

| Versão | Template |
|---|---|
| v1 | `{pergunta}` |
| v2 | `Responda de forma técnica e objetiva:\n\n{pergunta}` |
| v3 | `Você é um engenheiro agrônomo especialista. Responda com precisão técnica:\n\n{pergunta}` |

Para adicionar um novo prompt, edite o dicionário `PROMPTS` em `src/config.py`.

## Métricas

- **answer_similarity** — similaridade coseno entre embeddings (all-MiniLM-L6-v2)
- **ari_grade** — Automated Readability Index do texto gerado
- **precision / recall** — token-level, comparando tokens preditos vs. ground truth
- **accuracy** — exact match (1 se idêntico, 0 caso contrário)

## Modelos Testados

gemma3:4b, llama3.1:8b, command-r7b-arabic:7b, qwen3:8b, ministral:8b, phi3:8b

## Estrutura do Projeto

```
├── run_benchmark.py           # Entry point CLI
├── data/
│   ├── citrus_qa.json         # 15 Q&A (citros)
│   └── citrus_sugarcane_qa.json  # 62 Q&A (citros + cana)
├── src/
│   ├── config.py              # Configurações e prompts
│   ├── inference.py           # Chamadas ao Ollama
│   ├── metrics.py             # Cálculo de métricas
│   ├── data_loader.py         # Carga de dados JSON
│   └── evaluation.py          # Pipeline de avaliação + MLflow
├── notebooks/
│   └── benchmark.ipynb        # Notebook interativo
├── requirements.txt           # Dependências diretas
└── .env.example               # Template de variáveis de ambiente
```
