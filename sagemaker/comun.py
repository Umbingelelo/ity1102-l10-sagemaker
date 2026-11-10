"""Lo que comparten los cinco scripts: región, rol, bucket, imagen de XGBoost y nombres.

En AWS Academy Learner Lab no se pueden crear usuarios ni roles de IAM. Todo usa el rol que ya existe,
LabRole, y las credenciales de la sesión de CloudShell. No hay access keys en ningún archivo.
"""
import argparse
import json
import sys
from pathlib import Path

import boto3

from casos import CASOS

REGION = "us-east-1"
# Imagen oficial del algoritmo XGBoost de SageMaker en us-east-1 (versión 1.7-1).
# Es la misma que devuelve sagemaker.image_uris.retrieve("xgboost", "us-east-1", "1.7-1").
IMAGEN_XGBOOST = "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.7-1"
TIPO_ENTRENAMIENTO = "ml.m5.large"

RAIZ = Path(__file__).resolve().parent.parent
DATOS = RAIZ / "datos"
ESTADO = RAIZ / "estado"

sesion = boto3.Session(region_name=REGION)
CUENTA = sesion.client("sts").get_caller_identity()["Account"]
ROL = f"arn:aws:iam::{CUENTA}:role/LabRole"
BUCKET = f"ity1102-l10-{CUENTA}"

sm = sesion.client("sagemaker")
s3 = sesion.client("s3")


def leer_caso():
    p = argparse.ArgumentParser()
    p.add_argument("caso", choices=sorted(CASOS), help="rfm es el común; los otros tres son los casos del ramo")
    p.add_argument("--tiempo-real", action="store_true",
                   help="plan B: endpoint de tiempo real en ml.t2.medium si el serverless falla")
    p.add_argument("--memoria", type=int, default=2048, help="MB del endpoint serverless (1024 a 6144)")
    p.add_argument("--concurrencia", type=int, default=2, help="invocaciones simultáneas del endpoint serverless")
    p.add_argument("--todo", action="store_true", help="5_borrar: borra también el bucket")
    return p.parse_args()


def nombre_endpoint(caso):
    return f"l10-{caso}"


def estado_de(caso):
    """Lo que los pasos anteriores anotaron para este caso. Vacío si no hay nada."""
    ruta = ESTADO / f"{caso}.json"
    return json.loads(ruta.read_text()) if ruta.exists() else {}


def guardar_estado(caso, **datos):
    """Anota datos del caso. Una clave con None se borra."""
    ESTADO.mkdir(exist_ok=True)
    actual = estado_de(caso)
    actual.update(datos)
    actual = {k: v for k, v in actual.items() if v is not None}
    (ESTADO / f"{caso}.json").write_text(json.dumps(actual, indent=2, ensure_ascii=False))
    return actual


def leer_estado(caso, clave, paso):
    estado = estado_de(caso)
    if clave not in estado:
        sys.exit(f"Falta '{clave}' para el caso {caso}. Corre primero {paso}.")
    return estado[clave]
