"""Paso 5: borra el endpoint, sus configuraciones y sus modelos. Con --todo, también el bucket.

    python3 sagemaker/5_borrar.py rfm
    python3 sagemaker/5_borrar.py rfm --todo

Un endpoint serverless sin uso no cobra, pero uno de tiempo real cobra cada hora. Bórralo siempre.
El modelo entrenado queda en S3 y su ruta queda anotada en estado/<caso>.json (salvo con --todo):
con 3_endpoint.py se vuelve a publicar en unos minutos, sin reentrenar.
"""
from botocore.exceptions import ClientError

from comun import BUCKET, ESTADO, guardar_estado, leer_caso, nombre_endpoint, s3, sesion, sm


def main():
    args = leer_caso()
    caso = args.caso
    endpoint = nombre_endpoint(caso)
    prefijo = f"l10-{caso}-"

    try:
        sm.delete_endpoint(EndpointName=endpoint)
        print(f"Endpoint borrado: {endpoint}")
    except ClientError:
        print(f"No había endpoint {endpoint}")

    for c in sm.list_endpoint_configs(NameContains=prefijo, MaxResults=100)["EndpointConfigs"]:
        sm.delete_endpoint_config(EndpointConfigName=c["EndpointConfigName"])
        print(f"Configuración borrada: {c['EndpointConfigName']}")
    for m in sm.list_models(NameContains=prefijo, MaxResults=100)["Models"]:
        sm.delete_model(ModelName=m["ModelName"])
        print(f"Modelo borrado: {m['ModelName']}")

    if args.todo:
        try:
            sesion.resource("s3").Bucket(BUCKET).objects.all().delete()
            s3.delete_bucket(Bucket=BUCKET)
            print(f"Bucket borrado: {BUCKET}")
        except ClientError as e:
            print(f"No se pudo borrar el bucket {BUCKET}: {e}")

        (ESTADO / f"{caso}.json").unlink(missing_ok=True)
    else:
        # Se olvida solo el endpoint: los datos y el modelo entrenado siguen sirviendo para 3_endpoint.py
        guardar_estado(caso, endpoint=None, modelo=None, config=None, tipo=None)


if __name__ == "__main__":
    main()
