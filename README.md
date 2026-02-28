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

7. Alterar o modelo no código

    No trecho abaixo, substitua pelo modelo correspondente que foi baixado no Ollama:

        for _, row in df.iterrows():
            payload = {
                "model": "gemma3:4b",
                "prompt": row["inputs"]
            }

8. Alterar o nome do experimento no MLflow

    Modifique o nome do run conforme o modelo testado:

        with mlflow.start_run(run_name="command-r7b-arabic:7b_Teste01"):

    Utilize um nome que identifique corretamente o modelo e o teste realizado.
