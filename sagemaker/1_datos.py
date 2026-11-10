"""Paso 1: genera los datos del caso, los separa en entrenamiento, validación y prueba, y sube los dos primeros a S3.

    python3 sagemaker/1_datos.py rfm
"""
import csv
import random

from botocore.exceptions import ClientError

from casos import CASOS, SEMILLA, generar
from comun import BUCKET, DATOS, REGION, guardar_estado, leer_caso, s3


def escribir(ruta, filas, cabecera=None):
    with open(ruta, "w", newline="") as f:
        w = csv.writer(f)
        if cabecera:
            w.writerow(cabecera)
        w.writerows(filas)


def main():
    caso = leer_caso().caso
    filas = generar(caso)
    random.Random(SEMILLA).shuffle(filas)

    # 70% entrena, 15% valida durante el entrenamiento, 15% queda guardado para medir al final.
    # La prueba NUNCA sube a S3: el modelo no la ve hasta el paso 4.
    n = len(filas)
    a, b = int(0.70 * n), int(0.85 * n)
    entrenamiento, validacion, prueba = filas[:a], filas[a:b], filas[b:]

    carpeta = DATOS / caso
    carpeta.mkdir(parents=True, exist_ok=True)
    escribir(carpeta / "train.csv", entrenamiento)          # sin cabecera: así lo exige XGBoost
    escribir(carpeta / "validation.csv", validacion)
    escribir(carpeta / "test.csv", prueba, [CASOS[caso]["etiqueta"]] + CASOS[caso]["columnas"])

    try:
        s3.head_bucket(Bucket=BUCKET)
    except ClientError:
        s3.create_bucket(Bucket=BUCKET)                     # en us-east-1 no lleva LocationConstraint
        print(f"Bucket creado: {BUCKET}")

    for nombre in ("train", "validation"):
        s3.upload_file(str(carpeta / f"{nombre}.csv"), BUCKET, f"{caso}/{nombre}/{nombre}.csv")

    positivos = sum(f[0] for f in filas) / n
    print(f"Caso {caso}: {n} filas, {positivos:.1%} con etiqueta 1.")
    print(f"  entrenamiento {len(entrenamiento)} · validación {len(validacion)} · prueba {len(prueba)} (queda local)")
    print(f"  s3://{BUCKET}/{caso}/train/train.csv")
    print(f"  s3://{BUCKET}/{caso}/validation/validation.csv")
    print(f"  Región {REGION}. Siguiente: python3 sagemaker/2_entrenar.py {caso}")
    guardar_estado(caso, datos=f"s3://{BUCKET}/{caso}/")


if __name__ == "__main__":
    main()
