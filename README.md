Passo a Passo para Executar o Projeto

1. Ativar a extensão do Jupyter (caso utilize VS Code)

    Instale a extensão Jupyter no VS Code.

    Selecione o interpretador do ambiente virtual criado (venv).

2. Ativar o ambiente virtual

    No terminal, dentro da pasta do projeto, execute:

         venv\Scripts\activate

3. Instalar as dependências

         pip install -r requirements.txt

4. Rodar o MLflow

         mlflow ui

    Depois, acesse no navegador:

        http://127.0.0.1:5000

5. Instalar o Ollama

    Baixe e instale pelo site oficial:

        https://ollama.com

6. Baixar os modelos que deseja testar
 
     No terminal, execute:

         ollama pull nome-do-modelo

    (Substitua nome-do-modelo pelo modelo que deseja utilizar.)

    Modelos utilizados até o momento:

     command arabic 7b;

     qwen3 8b;

     llama3.1 8b;

     ministral 8b;

     phi3 8b;

     gemma3 4b.

8. Alterar o modelo no código

    No trecho abaixo, substitua pelo modelo correspondente que foi baixado no Ollama:

        for _, row in df.iterrows():
            payload = {
                "model": "gemma3:4b",
                "prompt": row["inputs"]
            }

9. Alterar o nome do experimento no MLflow

    Modifique o nome do run conforme o modelo testado:

        with mlflow.start_run(run_name="command-r7b-arabic:7b_Teste01"):

    Utilize um nome que identifique corretamente o modelo e o teste realizado.

10. Executar o notebook

    Clique em Run All para executar todas as células
    ou

    Execute célula por célula manualmente.

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
| `moderate` | `0.3` |
| `creative` | `0.6` |

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
