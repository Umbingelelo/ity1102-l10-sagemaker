"""El microservicio de IA de EP3, en mínimo: ya no carga el modelo, invoca el endpoint de SageMaker.

Credenciales: ninguna en el código ni en el .env. En la EC2, boto3 las obtiene solo del perfil de
instancia LabInstanceProfile (rol LabRole). En CloudShell, de la sesión de la consola.
"""
import os
import time

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import FastAPI, HTTPException
from pydantic import ConfigDict, create_model

from casos import CASOS

CASO = os.getenv("CASO", "rfm")
ENDPOINT = os.getenv("ENDPOINT") or f"l10-{CASO}"
REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
UMBRAL = float(os.getenv("UMBRAL", "0.5"))
COLUMNAS = CASOS[CASO]["columnas"]

runtime = boto3.client("sagemaker-runtime", region_name=REGION)
sagemaker = boto3.client("sagemaker", region_name=REGION)

# El esquema de entrada se arma con las columnas del caso: un campo numérico obligatorio por columna.
# Si falta uno o llega texto, FastAPI responde 422 antes de llamar a SageMaker.
Entrada = create_model(
    "Entrada",
    __config__=ConfigDict(json_schema_extra={"example": dict(zip(COLUMNAS, CASOS[CASO]["ejemplo"]))}),
    **{c: (float, ...) for c in COLUMNAS},
)

app = FastAPI(title=f"L10 · microservicio de IA · caso {CASO}")


@app.get("/health")
def health():
    try:
        estado = sagemaker.describe_endpoint(EndpointName=ENDPOINT)["EndpointStatus"]
    except (ClientError, BotoCoreError) as e:
        estado = f"error: {e.__class__.__name__}"
    return {"caso": CASO, "endpoint": ENDPOINT, "estado_endpoint": estado}


@app.post("/predict")
def predict(entrada: Entrada):
    fila = ",".join(str(getattr(entrada, c)) for c in COLUMNAS)   # mismo orden que en el entrenamiento
    t = time.perf_counter()
    try:
        r = runtime.invoke_endpoint(EndpointName=ENDPOINT, ContentType="text/csv", Body=fila)
    except ClientError as e:
        raise HTTPException(status_code=502, detail=f"SageMaker respondió: {e.response['Error']['Message']}")
    except BotoCoreError as e:
        raise HTTPException(status_code=503, detail=f"No se pudo llegar a SageMaker: {e}")
    probabilidad = float(r["Body"].read().decode().strip())
    return {
        "caso": CASO,
        "probabilidad": round(probabilidad, 4),
        "decision": int(probabilidad >= UMBRAL),
        "umbral": UMBRAL,
        "endpoint": ENDPOINT,
        "latencia_sagemaker_ms": round((time.perf_counter() - t) * 1000),
    }
