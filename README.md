# ITY1102 · L10 · Tu modelo en un endpoint serverless de SageMaker

Repositorio guía del laboratorio 10 de Arquitectura de Sistemas de IA (Duoc UC, 2026-02). Entrena un
modelo XGBoost con un Training Job de SageMaker, lo publica en un endpoint serverless y lo consume desde
un microservicio FastAPI. Está hecho para **AWS Academy Learner Lab**: usa el rol `LabRole`, no crea
usuarios IAM y no necesita access keys.

La guía paso a paso está en [`LABORATORIO.md`](LABORATORIO.md).

## Qué hay

| Archivo | Qué hace |
|---|---|
| `sagemaker/casos.py` | Los cuatro conjuntos de datos: `rfm` (el común, del anexo oficial 3.3.5) y `fraude`, `chatbot` y `recomendacion` (los de L9) |
| `sagemaker/1_datos.py` | Genera los datos, separa 70/15/15 y sube entrenamiento y validación a S3 |
| `sagemaker/2_entrenar.py` | Training Job con el XGBoost integrado de SageMaker en `ml.m5.large` |
| `sagemaker/3_endpoint.py` | Modelo, configuración y endpoint serverless (o de tiempo real con `--tiempo-real`) |
| `sagemaker/4_probar.py` | Cold start, latencia p50/p95 y accuracy, precision, recall y F1 sobre la prueba |
| `sagemaker/5_borrar.py` | Borra endpoint, configuraciones y modelos (con `--todo`, también el bucket) |
| `api/` + `compose.yaml` | El microservicio de IA que invoca el endpoint con el rol de la EC2 |

## En corto

Desde AWS CloudShell, en `us-east-1`:

```bash
git clone https://github.com/Umbingelelo/ity1102-l10-sagemaker.git
cd ity1102-l10-sagemaker
python3 sagemaker/1_datos.py rfm
python3 sagemaker/2_entrenar.py rfm
python3 sagemaker/3_endpoint.py rfm
python3 sagemaker/4_probar.py rfm
python3 sagemaker/5_borrar.py rfm
```

Los scripts solo usan `boto3`, que CloudShell ya trae. No hay que instalar nada.
