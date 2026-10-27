from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Any, Optional, List, Union
from fastapi.middleware.cors import CORSMiddleware
import json
from typing import Literal

from rag_core import perguntar, perguntar_lote, perguntar_stream, perguntar_avaliador_stream


Profile = Literal[
    "produtor_rural",
    "cooperativa",
    "empresa",
    "pesquisador_universitario",
    "tecnico",
    "instituicao_publica",
    "visitante",
]

RESPONSE_MODE_TEMPERATURES = {
    "deterministic": 0.1,
    "moderate": 0.5,
    "creative": 0.9,
}


app = FastAPI(
    title="SB100 RAG API",
    version="1.0",
    description="API do Agente SB100 com RAG + MLflow"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PerguntaRequest(BaseModel):
    pergunta: Optional[str] = None
    perguntas: Optional[Union[str, List[str]]] = None
    ground_truths: Optional[Union[str, List[str]]] = None
    profile: Profile = "visitante"


class AvaliadorStreamRequest(BaseModel):
    pergunta: str
    ground_truth: str = ""
    profile: Profile = "visitante"


class PerguntaStreamRequest(PerguntaRequest):
    response_mode: Optional[Any] = None


# ============================================================
# STREAMING — USADO PELO CHAT
# ============================================================

@app.post("/perguntar/stream")
async def fazer_pergunta_stream(data: PerguntaStreamRequest):

    response_mode = None
    if "response_mode" in data.model_fields_set:
        response_mode = data.response_mode
        if (
            not isinstance(response_mode, str)
            or response_mode not in RESPONSE_MODE_TEMPERATURES
        ):
            modos_validos = ", ".join(RESPONSE_MODE_TEMPERATURES)
            raise HTTPException(
                status_code=400,
                detail=f"response_mode inválido. Valores permitidos: {modos_validos}.",
            )

    if not data.pergunta:
        return {
            "erro": "O campo 'pergunta' é obrigatório."
        }

    profile = data.profile
    print(f"[API] Perfil recebido em /perguntar/stream: {profile}", flush=True)
    modo_log = (
        response_mode
        if response_mode is not None
        else "não informado (padrão atual)"
    )
    print(
        f"[API] Modo de resposta recebido em /perguntar/stream: {modo_log}",
        flush=True,
    )

    async def gerar():

        resposta_completa = ""

        # --------------------------------------------
        # STREAM DA RESPOSTA
        # --------------------------------------------

        if response_mode is None:
            stream = perguntar_stream(data.pergunta, profile=profile)
        else:
            stream = perguntar_stream(
                data.pergunta,
                profile=profile,
                temperature=RESPONSE_MODE_TEMPERATURES[response_mode],
            )

        async for token in stream:

            resposta_completa += token

            yield json.dumps({
                "tipo": "token",
                "conteudo": token
            }, ensure_ascii=False) + "\n"

        # --------------------------------------------
        # BUSCA OS METADADOS DA MESMA PERGUNTA
        # --------------------------------------------
        #
        # IMPORTANTE:
        # aqui usamos perguntar() apenas para recuperar
        # chunks, referências e hallucination_flag.
        #
        # A resposta gerada não será usada novamente.
        #
        # --------------------------------------------

        try:

            _, chunks, refs_principais, h_flag = perguntar(
                data.pergunta,
                profile=profile,
            )

            yield json.dumps({
                "tipo": "metadata",
                "referencias_principais": refs_principais,
                "chunks": chunks,
                "hallucination_flag": h_flag
            }, ensure_ascii=False) + "\n"

        except Exception as e:

            yield json.dumps({
                "tipo": "metadata",
                "referencias_principais": [],
                "chunks": [],
                "hallucination_flag": None,
                "erro_metadata": str(e)
            }, ensure_ascii=False) + "\n"

    return StreamingResponse(
        gerar(),
        media_type="application/x-ndjson; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/perguntar/avaliador/stream")
async def perguntar_avaliador_stream_endpoint(data: AvaliadorStreamRequest):
    if not data.pergunta.strip():
        raise HTTPException(status_code=400, detail="O campo 'pergunta' é obrigatório.")

    print(f"[API] Perfil recebido em /perguntar/avaliador/stream: {data.profile}", flush=True)

    async def gerar():
        try:
            async for evento in perguntar_avaliador_stream(
                data.pergunta,
                ground_truth=data.ground_truth,
                profile=data.profile,
            ):
                yield json.dumps(evento, ensure_ascii=False) + "\n"
        except Exception as e:
            print(f"[API] Erro em /perguntar/avaliador/stream: {e!r}", flush=True)
            yield json.dumps(
                {"tipo": "erro", "mensagem": "Falha ao processar a pergunta."},
                ensure_ascii=False,
            ) + "\n"

    return StreamingResponse(
        gerar(),
        media_type="application/x-ndjson; charset=utf-8",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ============================================================
# FLUXO SÍNCRONO — TESTES / BATCH
# ============================================================

@app.post("/perguntar")
async def fazer_pergunta(data: PerguntaRequest):
    profile = data.profile
    print(f"[API] Perfil recebido em /perguntar: {profile}", flush=True)

    # --------------------------------------------
    # MODO AVALIADOR / LOTE
    # --------------------------------------------

    if data.perguntas:

        resultados = perguntar_lote(
            perguntas=data.perguntas,
            ground_truths=data.ground_truths,
            profile=profile,
        )

        return {
            "modo": "avaliador",
            "resultados": resultados
        }

    # --------------------------------------------
    # MODO SÍNCRONO INDIVIDUAL
    # --------------------------------------------

    if not data.pergunta:

        return {
            "erro": "Informe 'pergunta' ou 'perguntas'."
        }

    resposta, chunks, refs_principais, h_flag = perguntar(
        data.pergunta,
        profile=profile,
    )

    return {
        "modo": "chat",
        "pergunta": data.pergunta,
        "resposta": resposta,
        "referencias_principais": refs_principais,
        "chunks": chunks,
        "hallucination_flag": h_flag
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def health():

    return {
        "status": "ok",
        "service": "SB100 RAG API"
    }