# %%
import mlflow
import pandas as pd
import requests
import json
import textstat
import numpy as np
from sentence_transformers import SentenceTransformer, util
import textstat
from sklearn.metrics import precision_score, recall_score, accuracy_score


# %%
import os
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"

# %%
mlflow.set_tracking_uri(uri="http://localhost:5000")

# %%
def ollama_predict(df, PROMPT_TEMPLATE):
    outputs = []
    url = "http://localhost:11434/api/generate"

    for _, row in df.iterrows():
        payload = {
            "model": "gemma3:4b",
            "prompt": PROMPT_TEMPLATE.format(pergunta=row["inputs"])
        }

        r = requests.post(url, json=payload)
        r.raise_for_status()

        completion = ""
        for line in r.text.strip().split("\n"):
            completion += json.loads(line).get("response", "")

        outputs.append(completion)

    return outputs

# %%
eval_data = pd.DataFrame(
       {
    "inputs": [
        "Qual é a participação da citricultura no volume das oito principais frutíferas brasileiras?",
        "A que gênero pertencem as principais espécies cítricas cultivadas comercialmente?",
        "Qual a faixa de exportação de nitrogênio por tonelada de fruto colhido?",
        "Qual é a faixa adequada de fósforo foliar em citros?",
        "Qual profundidade deve ser amostrada para recomendações de adubação e calagem?",
        "Descreva o procedimento correto de amostragem foliar para citros.",
        "Por que é necessário aguardar 30 dias após pulverizações antes da coleta foliar?",
        "Quais camadas do solo devem ser amostradas para diagnóstico de fertilidade em pomares?",
        "Qual a meta de saturação por bases recomendada na calagem para citros?",
        "Qual teor mínimo de magnésio deve ser mantido na camada de 0–20 cm?",
        "Para laranjas destinadas à indústria com produtividade de 51–60 t ha⁻¹ e N foliar entre 25–30 g kg⁻¹, qual a dose recomendada de N?",
        "Para laranjas destinadas ao consumo in natura com produtividade de 41–50 t ha⁻¹ e K trocável entre 1,6–3,0 mmolc dm⁻³, qual a dose recomendada de K₂O?",
        "Para lima ácida Tahiti com produtividade acima de 60 t ha⁻¹ e N foliar entre 20–24 g kg⁻¹, qual a dose recomendada de N?",
        "O que deve ser feito quando o teor de fósforo no solo for superior a 80 mg dm⁻³?",
        "Por que a análise foliar é considerada essencial para diagnóstico de nitrogênio em citros?",
        "Como o adensamento de plantio influencia o manejo nutricional na citricultura?",
        "Explique por que o parcelamento das doses de nitrogênio e potássio aumenta a eficiência da adubação.",
        "Em solo arenoso com deficiência foliar de cálcio, qual a dose recomendada de gesso agrícola?",
        "Em solo de textura média com deficiência foliar de cálcio, qual a dose recomendada de gesso?",
        "Em solo argiloso com deficiência foliar de cálcio, qual a dose recomendada de gesso?",
        "Caso a saturação por bases na camada de 20–40 cm seja inferior a 25%, como deve ser ajustada a dose de calcário?",
        "Qual a recomendação de aplicação de P₂O₅ no sulco de plantio?",
        "Como deve ser ajustada a dose de P₂O₅ para copas sobre tangerinas Cleópatra ou Sunki?",
        "Como deve ser ajustada a dose de K₂O para copas sobre citrumelo Swingle?",
        "Como deve ser feito o parcelamento de N e K em variedades precoces como Hamlin e Valência Americana?",
        "Qual a concentração recomendada de boro na pulverização foliar?",
        "Qual a concentração recomendada de manganês na pulverização foliar?",
        "Qual a concentração recomendada de zinco na pulverização foliar?",
        "Qual a recomendação de magnésio para correção via aplicação foliar?",
        "Qual o papel do molibdênio no metabolismo do nitrogênio em citros?",
        "Qual a concentração recomendada de molibdênio na pulverização foliar?",
        "Qual o volume ideal de calda para aplicação de micronutrientes foliares em citros?"
	    "Qual é a meta de produtividade mencionada para a cana-de-açúcar segundo a recomendação técnica?",
        "Quais quantidades de N, P2O5 e K2O são exportadas com 100 t ha⁻¹ de colmos?",
        "Quais são as quantidades médias exportadas de micronutrientes por 100 t de colmos?",
        "Qual é a faixa adequada de nitrogênio foliar na cana-de-açúcar?",
        "Qual folha deve ser coletada para diagnose foliar na cana-de-açúcar?",
        "Em qual período ocorre a fase ideal de amostragem foliar na região Centro-Sul?",
        "Quantas plantas devem ser amostradas para análise foliar?",
        "Qual profundidade deve ser amostrada para cálculo de calagem e adubação?",
        "Qual profundidade deve ser amostrada para avaliar acidez em profundidade e necessidade de gessagem?",
        "Qual é a meta de saturação por bases recomendada na calagem?",
        "Qual é a dose mínima de calcário recomendada no plantio?",
        "Qual teor mínimo de magnésio deve ser mantido na camada superficial?",
        "Quando a gessagem é recomendada?",
        "Qual fórmula é utilizada para calcular a dose de gesso com base na textura do solo?",
        "Qual é a recomendação caso o teor de enxofre na camada 25–50 cm seja inferior a 15 mg dm⁻³?",
        "Qual a dose de nitrogênio recomendada no plantio para produtividade inferior a 100 t ha⁻¹?",
        "Qual a dose máxima de P2O5 recomendada para solos com P resina inferior a 7 mg dm⁻³ e produtividade acima de 170 t ha⁻¹?",
        "Qual é o limite máximo de K2O que deve ser aplicado no sulco de plantio?",
        "Em que condição é dispensável a aplicação de adubo potássico?",
        "Qual é a dose recomendada de molibdênio no sulco de plantio?",
        "Qual a alternativa via aplicação foliar para o molibdênio no plantio?",
        "Qual a recomendação de zinco para solo com teor inferior a 0,6 mg dm⁻³?",
        "Qual a dose recomendada de boro para solo com teor inferior a 0,2 mg dm⁻³?",
        "Qual a recomendação de nitrogênio para soqueiras com produtividade esperada acima de 140 t ha⁻¹?",
        "Como deve ser ajustada a adubação nitrogenada em áreas que recebem vinhaça?",
        "Qual o desconto de nitrogênio da vinhaça in natura na adubação?",
        "Qual o desconto de nitrogênio da vinhaça concentrada?",
        "Qual a recomendação de zinco e manganês via foliar para soqueiras?",
        "Qual a dose recomendada de boro para soqueiras?",
        "Quando deve ser aplicada a adubação foliar com molibdênio nas soqueiras?"
    ],
    "ground_truth": [
        "A citricultura representa cerca de 55% do volume das oito principais frutíferas brasileiras.",
        "Pertencem ao gênero Citrus spp.",
        "De 1,9 a 2,4 kg por tonelada de fruto colhido.",
        "De 1,2 a 1,6 g kg⁻¹.",
        "De 0–20 cm.",
        "Deve-se coletar a 3ª ou 4ª folha do ramo com fruto terminal, gerada na primavera, com aproximadamente seis meses de idade, normalmente entre fevereiro e março, em ramos com frutos de 2 a 4 cm de diâmetro, amostrando pelo menos 20 árvores por talhão e coletando quatro folhas por árvore, uma em cada quadrante da copa.",
        "Porque pulverizações podem deixar nutrientes aderidos à superfície das folhas e causar erro na interpretação da análise.",
        "0–20 cm para recomendações de adubação e calagem e 20–40 cm para diagnóstico de barreiras químicas e movimentação de nutrientes.",
        "Elevar a saturação por bases a 70%.",
        "Pelo menos 8 mmolc dm⁻³.",
        "200 kg ha⁻¹ de N.",
        "180 kg ha⁻¹ de K₂O.",
        "180 kg ha⁻¹ de N.",
        "Não aplicar fósforo para evitar desequilíbrios nutricionais.",
        "Porque a análise de solo para nitrogênio não é consistente, sendo o teor foliar o critério direto de avaliação da disponibilidade do nutriente.",
        "O adensamento altera a dinâmica de distribuição de nutrientes no solo e pode exigir ajustes nas doses e no manejo da adubação conforme o sistema de plantio adotado.",
        "Porque reduz perdas por drenagem e ajusta a oferta de nutrientes à demanda da planta ao longo do desenvolvimento.",
        "1,0 t ha⁻¹ de gesso.",
        "1,5 t ha⁻¹ de gesso.",
        "2,0 t ha⁻¹ de gesso.",
        "Aumentar em 50% a dose de calcário aplicada na faixa de plantio.",
        "90 g de P₂O₅ por metro de sulco (equivalente a 120–160 kg ha⁻¹ conforme espaçamento).",
        "Aumentar a dose de P₂O₅ em 20%.",
        "Aumentar a dose de K₂O em 20%.",
        "Aplicar 40% na primeira parcela, 40% na segunda e 20% na última, realizada até o final do verão.",
        "200 a 300 mg L⁻¹.",
        "300 a 700 mg L⁻¹.",
        "500 a 1.000 mg L⁻¹.",
        "Aplicar 5 a 10 kg de sulfato de magnésio hidratado por 2.000 L de calda.",
        "O molibdênio atua como componente da enzima redutase do nitrato, participando da assimilação do nitrogênio pelas plantas.",
        "20 a 40 mg L⁻¹ de Mo (equivalente a 100 a 200 g por 2.000 L de calda).",
        "Aproximadamente 2.000 L ha⁻¹."
	    "Elevar a produtividade média de menos de 80 t ha⁻¹ para mais de 100 t ha⁻¹.",
        "90 kg ha⁻¹ de N, 35 kg ha⁻¹ de P2O5 e 130 kg ha⁻¹ de K2O.",
        "120 g de B, 260 g de Cu, 1.400 g de Fe, 970 g de Mn, 160 g de Mo e 350 g de Zn.",
        "18–25 g kg⁻¹.",
        "A folha +1 (Kuijper), coletando os 20 cm centrais sem a nervura central.",
        "Entre dezembro e fevereiro.",
        "30 plantas.",
        "0–25 cm.",
        "25–50 cm.",
        "Elevar a saturação por bases a 70%.",
        "Não menos que 1,5 t ha⁻¹ (PRNT 100%).",
        "Manter Mg²⁺ acima de 8 mmolc dm⁻³ na camada superficial.",
        "Quando a saturação por bases na camada 25–50 cm for inferior a 40% ou saturação por alumínio superior a 30% da CTC efetiva.",
        "Argila (g kg⁻¹) × 6 = kg ha⁻¹ de gesso.",
        "Aplicar 1 t ha⁻¹ de gesso como fonte de enxofre.",
        "30 kg ha⁻¹ de N.",
        "200 kg ha⁻¹ de P2O5.",
        "80 kg ha⁻¹ de K2O.",
        "Quando o teor de K for superior a 6,0 mmolc dm⁻³.",
        "0,6 kg ha⁻¹ de Mo no sulco de plantio.",
        "0,3 kg ha⁻¹ de Mo via aplicação foliar aos quatro meses após a brotação.",
        "10 kg ha⁻¹ de Zn.",
        "2,0 kg ha⁻¹ de B.",
        "160 kg ha⁻¹ de N.",
        "Reduzir ou dispensar a adubação nitrogenada em áreas com alta saturação de K (>5% da CTC).",
        "Descontar 70% do N contido na vinhaça in natura.",
        "Descontar 50% do N contido na vinhaça concentrada.",
        "Aplicar 2–3 kg ha⁻¹ de Zn e 1–2 kg ha⁻¹ de Mn via foliar, em pelo menos duas aplicações.",
        "1 a 1,5 kg ha⁻¹ de B.",
        "Aos quatro meses após a rebrota ou na fase de máximo perfilhamento, repetindo nas safras subsequentes caso não tenha sido aplicado no plantio."
    ]
}

)

# %%
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def simple_token_metrics(pred, gt):
    pred_tokens = set(pred.lower().split())
    gt_tokens = set(gt.lower().split())

    tp = len(pred_tokens & gt_tokens)

    precision = tp / len(pred_tokens) if pred_tokens else 0
    recall = tp / len(gt_tokens) if gt_tokens else 0
    accuracy = 1 if pred.strip().lower() == gt.strip().lower() else 0

    return precision, recall, accuracy


# %%
PROMPT_VERSION = "v2"

PROMPTS = {
    "v1": "{pergunta}",
    "v2": "Responda de forma técnica e objetiva:\n\n{pergunta}",
    "v3": "Você é um engenheiro agrônomo especialista. Responda com precisão técnica:\n\n{pergunta}"
}

PROMPT_TEMPLATE = PROMPTS[PROMPT_VERSION]

# %%
with mlflow.start_run(run_name="gemma3:4b_TesteVersionamento"):

    mlflow.log_param("model_name", "gemma3:4b")
    mlflow.log_param("prompt_version", PROMPT_VERSION)

    predictions = ollama_predict(eval_data, PROMPT_TEMPLATE)

    similarities = []
    ari_scores = []
    precisions = []
    recalls = []
    accuracies = []

    for pred, gt in zip(predictions, eval_data["ground_truth"]):

        emb_p = embed_model.encode(pred, convert_to_tensor=True)
        emb_g = embed_model.encode(gt, convert_to_tensor=True)
        similarities.append(util.cos_sim(emb_p, emb_g).item())

        ari_scores.append(textstat.automated_readability_index(pred))

        p, r, a = simple_token_metrics(pred, gt)
        precisions.append(p)
        recalls.append(r)
        accuracies.append(a)

    mlflow.log_metric("answer_similarity", float(np.mean(similarities)))
    mlflow.log_metric("ari_grade", float(np.mean(ari_scores)))
    mlflow.log_metric("precision", float(np.mean(precisions)))
    mlflow.log_metric("recall", float(np.mean(recalls)))
    mlflow.log_metric("accuracy", float(np.mean(accuracies)))

    eval_table = pd.DataFrame({
        "inputs": eval_data["inputs"],
        "ground_truth": eval_data["ground_truth"],
        "prediction": predictions,
        "answer_similarity": similarities,
        "ari_grade": ari_scores,
        "precision": precisions,
        "recall": recalls,
        "accuracy": accuracies
    })

    mlflow.log_table(eval_table, "eval/evaluation_table.json")

    for i, out in enumerate(predictions):
        mlflow.log_text(out, f"outputs/output_{i}.txt")



