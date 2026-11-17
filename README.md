# Como rodar o RAG

Esta instrução executa somente a API do RAG (`api.py`) e suas dependências
principais: Ollama, Qdrant e MLflow.

## 1. Instalar o Ollama e baixar o modelo

1. Acesse https://ollama.com/ e baixe o instalador para Windows.
2. Instale o Ollama e abra o aplicativo. Ele normalmente inicia o servico em
   segundo plano.
3. Abra um terminal e baixe o modelo usado pelo RAG:

   ```powershell
   ollama pull llama3.1:8b
   ```

A instalacao do Ollama e o download do modelo sao feitos uma unica vez. Nao e
necessario manter um terminal aberto com `ollama serve` quando o aplicativo do
Ollama ja estiver em execucao em segundo plano. Se o RAG nao conseguir se
conectar ao Ollama, abra o aplicativo; como alternativa, inicie manualmente
`ollama serve` em um terminal separado e mantenha esse terminal aberto enquanto
usar o RAG.

O nome do modelo esta definido em `rag_core.py`, na constante `OLLAMA_MODEL`.
Para trocar o modelo, edite essa constante e baixe o novo modelo com
`ollama pull nome-do-modelo`.

## 2. Pre-requisitos

- Python 3.11 ou 3.12
- Conexao com a internet
- Acesso ao Qdrant configurado no projeto

## 3. Entrar na pasta do projeto

Abra um terminal na raiz do repositorio, onde estao `api.py` e
`rag_core.py`.

```powershell
cd C:\Users\pedro\Documents\SB-100_Squad4\Mlflow-Squad4
```

## 4. Criar e ativar o ambiente virtual

```powershell
py -3.12 -m venv venv
venv\Scripts\Activate.ps1
```

Caso o PowerShell bloqueie o script de ativacao, execute uma vez:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
venv\Scripts\Activate.ps1
```

Para verificar se o ambiente foi ativado:

```powershell
python --version
where.exe python
```

O caminho exibido deve apontar para `venv\Scripts\python.exe`.

## 5. Instalar as dependencias

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install fastembed
```

Na primeira inicializacao, o sistema tambem pode baixar da internet os
modelos de embeddings `Qwen/Qwen3-Embedding-0.6B` e
`Qdrant/bm25`.

## 6. Iniciar o MLflow

Em outro terminal, com o ambiente virtual ativado, execute:

```powershell
venv\Scripts\activate
mlflow ui --host 127.0.0.1 --port 5000
```

A interface do MLflow ficara disponivel em:

http://127.0.0.1:5000

A URL de rastreamento usada pelo RAG ja esta configurada em `rag_core.py`
como `http://localhost:5000`.

## 7. Iniciar a API do RAG

Abra um terceiro terminal na raiz do projeto, ative o ambiente virtual e
execute:

```powershell
venv\Scripts\Activate.ps1
python -m uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```

Ao iniciar, o terminal deve mostrar o carregamento dos modelos de embedding.
A API ficara disponivel em:

- API: http://127.0.0.1:8000
- Documentacao Swagger: http://127.0.0.1:8000/docs

## 8. Abrir o acesso externo com ngrok

Para disponibilizar temporariamente a API pela internet:

1. Acesse https://ngrok.com/, crie uma conta e faça login.
2. Baixe e instale o ngrok pelo site oficial.
3. Conclua a configuracao da conta e do token diretamente no painel do ngrok,
   seguindo as instrucoes exibidas no site.

Depois de instalado e configurado, abra um terminal separado e encaminhe a
porta da API:

```powershell
ngrok http 8000
```

Mantenha esse terminal aberto. O ngrok mostrara um endereco publico no formato:

```text
https://xxxx-xx-xx-xx-xx.ngrok-free.app
```

Use esse endereco no frontend ou nos testes externos. Por exemplo:

```text
https://xxxx-xx-xx-xx-xx.ngrok-free.app/perguntar
```

O endereco gerado pode mudar sempre que o ngrok for reiniciado. Na inicializacao
do ngrok, acesse tambem o painel de encaminhamento exibido no terminal,
geralmente em `http://127.0.0.1:4040`.

## 9. Resumo dos processos

O aplicativo Ollama deve estar em execucao para que o RAG consiga gerar
respostas. Quando ele ja estiver rodando em segundo plano, nao e necessario
abrir um terminal separado para o Ollama. Mantenha os terminais do MLflow, da
API e do ngrok abertos enquanto precisar desses servicos.

| Componente | Comando | Endereco |
| --- | --- | --- |
| Ollama | Abra o aplicativo; normalmente nao precisa executar comando | http://localhost:11434 |
| MLflow | `mlflow ui --host 127.0.0.1 --port 5000` | http://127.0.0.1:5000 |
| API RAG | `python -m uvicorn api:app --host 127.0.0.1 --port 8000 --reload` | http://127.0.0.1:8000 |
| Ngrok | `ngrok http 8000` | Endereco HTTPS publico exibido pelo ngrok |

## 10. Configuracao usada pelo RAG

O sistema usa:

- **Qdrant** remoto para a busca hibrida;
- **Qwen3-Embedding-0.6B** para embeddings densos;
- **BM25 do FastEmbed** para embeddings esparsos;
- **RRF** para combinar os resultados;
- **Ollama com llama3.1:8b** para gerar as respostas;
- **MLflow** para registrar avaliacoes e metricas.

## Perfil nas perguntas

O frontend envia o perfil selecionado diretamente no corpo de cada chamada a
`POST /perguntar` ou `POST /perguntar/stream`. Não é necessário registrar o
perfil separadamente nem enviar `user_id`.

```json
{
  "pergunta": "Qual é a produtividade média da soja?",
  "profile": "produtor_rural"
}
```

`profile` aceita `produtor_rural`, `cooperativa`, `empresa`,
`pesquisador_universitario`, `tecnico`, `instituicao_publica` ou `visitante`.
Se omitido, o backend usa `visitante`; valores fora da lista retornam `422`.
O mesmo campo funciona com o payload de streaming, por exemplo:

```json
{
  "pergunta": "Qual é a produtividade média da soja?",
  "profile": "tecnico"
}
```

O endpoint `POST /perguntar/stream` também aceita opcionalmente `response_mode`
para selecionar a temperatura de geração configurada pelo servidor:

| `response_mode` | Temperatura |
| --- | ---: |
| `deterministic` | `0.1` |
| `moderate` | `0.5` |
| `creative` | `0.9` |

Exemplo:

```json
{
  "pergunta": "Qual é a produtividade média da soja?",
  "profile": "tecnico",
  "response_mode": "moderate"
}
```

Se `response_mode` for omitido, o endpoint mantém a geração atual
(temperatura `0.1`), garantindo compatibilidade com clientes existentes.
Valores não reconhecidos retornam HTTP `400`. A API não recebe temperatura nem
outras configurações livres de geração do cliente. Esse campo se aplica apenas
a `POST /perguntar/stream`; os formatos e eventos NDJSON permanecem iguais.

O Chat avaliador pode usar `POST /perguntar/avaliador/stream` para receber
eventos NDJSON enquanto a resposta é gerada e aguardar os metadados da avaliação:

```json
{
  "pergunta": "Qual é a produtividade média da soja?",
  "ground_truth": "Resposta esperada para avaliação",
  "profile": "visitante"
}
```

O endpoint emite eventos `token`, depois `status` enquanto a avaliação é
executada e, ao final, `metadata` com referências, flag de alucinação e
confidence score. `ground_truth` pode ser omitido.
