"""Paso 2: entrena el modelo con un Training Job de SageMaker y deja el artefacto en S3.

    python3 sagemaker/2_entrenar.py rfm

SageMaker enciende una ml.m5.large, descarga la imagen de XGBoost, entrena con los CSV de S3, guarda el
modelo en S3 y apaga la máquina. Se cobra solo el tiempo que estuvo encendida: unos 3 a 5 minutos.

Si cortas el script mientras espera (Ctrl+C, pestaña cerrada), el entrenamiento sigue en AWS. Vuelve a
correr el mismo comando: retoma ese trabajo en vez de lanzar otro.
"""
import time
from datetime import datetime

from comun import (BUCKET, IMAGEN_XGBOOST, ROL, TIPO_ENTRENAMIENTO, estado_de, guardar_estado, leer_caso,
                   leer_estado, sm)

HIPERPARAMETROS = {
    "objective": "binary:logistic",   # clasificación binaria: devuelve una probabilidad entre 0 y 1
    "eval_metric": "auc",
    "num_round": "100",
    "max_depth": "5",
    "eta": "0.2",
    "subsample": "0.8",
    "min_child_weight": "6",
}


def canal(caso, nombre):
    return {
        "ChannelName": nombre,
        "ContentType": "text/csv",
        "DataSource": {"S3DataSource": {
            "S3DataType": "S3Prefix",
            "S3Uri": f"s3://{BUCKET}/{caso}/{nombre}/",
            "S3DataDistributionType": "FullyReplicated",
        }},
    }


def lanzar(caso, trabajo):
    sm.create_training_job(
        TrainingJobName=trabajo,
        AlgorithmSpecification={"TrainingImage": IMAGEN_XGBOOST, "TrainingInputMode": "File"},
        RoleArn=ROL,
        InputDataConfig=[canal(caso, "train"), canal(caso, "validation")],
        OutputDataConfig={"S3OutputPath": f"s3://{BUCKET}/{caso}/modelos/"},
        ResourceConfig={"InstanceType": TIPO_ENTRENAMIENTO, "InstanceCount": 1, "VolumeSizeInGB": 5},
        StoppingCondition={"MaxRuntimeInSeconds": 900},
        HyperParameters=HIPERPARAMETROS,
    )


def main():
    caso = leer_caso().caso
    leer_estado(caso, "datos", "1_datos.py")

    previo = estado_de(caso).get("trabajo_en_curso")
    if previo and sm.describe_training_job(TrainingJobName=previo)["TrainingJobStatus"] in ("InProgress", "Completed"):
        trabajo = previo
        print(f"Retomo el Training Job {trabajo}, que ya estaba lanzado.")
    else:
        trabajo = f"l10-{caso}-{datetime.now():%Y%m%d-%H%M%S}"
        lanzar(caso, trabajo)
        guardar_estado(caso, trabajo_en_curso=trabajo)
        print(f"Training Job {trabajo} creado en {TIPO_ENTRENAMIENTO}. Se revisa cada 20 segundos.")

    inicio, anterior = time.time(), None
    while True:
        d = sm.describe_training_job(TrainingJobName=trabajo)
        actual = (d["TrainingJobStatus"], d.get("SecondaryStatus"))
        if actual != anterior:
            print(f"  {int(time.time() - inicio):>4} s · {actual[0]} · {actual[1]}")
            anterior = actual
        if d["TrainingJobStatus"] in ("Completed", "Failed", "Stopped"):
            break
        time.sleep(20)

    if d["TrainingJobStatus"] != "Completed":
        guardar_estado(caso, trabajo_en_curso=None)
        raise SystemExit(f"El entrenamiento terminó en {d['TrainingJobStatus']}: {d.get('FailureReason')}")

    for m in d.get("FinalMetricDataList", []):
        print(f"  {m['MetricName']}: {m['Value']:.4f}")
    segundos = d.get("BillableTimeInSeconds", 0)
    artefacto = d["ModelArtifacts"]["S3ModelArtifacts"]
    print(f"  Tiempo facturado: {segundos} s")
    print(f"  Modelo: {artefacto}")
    print(f"Siguiente: python3 sagemaker/3_endpoint.py {caso}")
    guardar_estado(caso, trabajo=trabajo, trabajo_en_curso=None, artefacto=artefacto, segundos_facturados=segundos)


if __name__ == "__main__":
    main()
