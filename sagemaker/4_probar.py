"""Paso 4: invoca el endpoint, mide el cold start y evalúa el modelo con el conjunto de prueba.

    python3 sagemaker/4_probar.py rfm
    UMBRAL=0.3 python3 sagemaker/4_probar.py fraude     # otro umbral de decisión

El conjunto de prueba (15% de los datos) nunca subió a S3: el modelo lo ve por primera vez aquí.
"""
import csv
import os
import statistics
import time

from casos import CASOS
from comun import DATOS, guardar_estado, leer_caso, leer_estado, sesion

UMBRAL = float(os.getenv("UMBRAL", "0.5"))
runtime = sesion.client("sagemaker-runtime")


def invocar(endpoint, lineas):
    """Manda una o varias filas en CSV (sin etiqueta) y devuelve las probabilidades y los milisegundos."""
    t = time.perf_counter()
    r = runtime.invoke_endpoint(EndpointName=endpoint, ContentType="text/csv", Body="\n".join(lineas))
    ms = (time.perf_counter() - t) * 1000
    texto = r["Body"].read().decode().replace("\n", ",")
    return [float(x) for x in texto.split(",") if x.strip()], ms


def main():
    caso = leer_caso().caso
    endpoint = leer_estado(caso, "endpoint", "3_endpoint.py")
    ejemplo = ",".join(str(v) for v in CASOS[caso]["ejemplo"])

    # 1. Una fila, dos veces seguidas. La primera puede traer cold start; la segunda no.
    print(f"Endpoint {endpoint} · fila de ejemplo: {ejemplo}")
    p1, ms1 = invocar(endpoint, [ejemplo])
    p2, ms2 = invocar(endpoint, [ejemplo])
    print(f"  1.ª invocación: {ms1:7.0f} ms · probabilidad {p1[0]:.4f}")
    print(f"  2.ª invocación: {ms2:7.0f} ms · probabilidad {p2[0]:.4f}")

    # 2. Latencia en caliente: 20 invocaciones de una fila
    tiempos = [invocar(endpoint, [ejemplo])[1] for _ in range(20)]
    tiempos.sort()
    p50, p95 = statistics.median(tiempos), tiempos[int(0.95 * len(tiempos)) - 1]
    print(f"  20 invocaciones en caliente: p50 {p50:.0f} ms · p95 {p95:.0f} ms (medido desde CloudShell)")

    # 3. Métricas del modelo sobre el conjunto de prueba, en lotes de 200 filas por invocación
    with open(DATOS / caso / "test.csv") as f:
        filas = list(csv.reader(f))[1:]
    reales = [int(float(f[0])) for f in filas]
    probs = []
    for i in range(0, len(filas), 200):
        probs += invocar(endpoint, [",".join(f[1:]) for f in filas[i:i + 200]])[0]
    pred = [int(p >= UMBRAL) for p in probs]

    vp = sum(1 for r, p in zip(reales, pred) if r == 1 and p == 1)
    fp = sum(1 for r, p in zip(reales, pred) if r == 0 and p == 1)
    fn = sum(1 for r, p in zip(reales, pred) if r == 1 and p == 0)
    vn = sum(1 for r, p in zip(reales, pred) if r == 0 and p == 0)
    accuracy = (vp + vn) / len(reales)
    precision = vp / (vp + fp) if vp + fp else 0.0
    recall = vp / (vp + fn) if vp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    print(f"Conjunto de prueba: {len(reales)} filas, {sum(reales)} con etiqueta 1 · umbral {UMBRAL}")
    print(f"  matriz de confusión: VP {vp} · FP {fp} · FN {fn} · VN {vn}")
    print(f"  accuracy {accuracy:.3f} · precision {precision:.3f} · recall {recall:.3f} · F1 {f1:.3f}")
    print(f"  línea base «todo 0»: accuracy {(vn + fp) / len(reales):.3f}, recall 0")

    guardar_estado(caso, cold_ms=round(ms1), caliente_ms=round(ms2), p50_ms=round(p50), p95_ms=round(p95),
                   umbral=UMBRAL, accuracy=round(accuracy, 3), precision=round(precision, 3),
                   recall=round(recall, 3), f1=round(f1, 3))


if __name__ == "__main__":
    main()
