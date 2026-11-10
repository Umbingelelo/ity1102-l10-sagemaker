"""Paso 3: registra el modelo y lo publica en un endpoint serverless.

    python3 sagemaker/3_endpoint.py rfm                      # serverless, 2048 MB, concurrencia 2
    python3 sagemaker/3_endpoint.py rfm --memoria 1024
    python3 sagemaker/3_endpoint.py rfm --tiempo-real        # plan B: ml.t2.medium, cobra por hora

Si el endpoint ya existe, lo actualiza a la configuración nueva sin cambiarle el nombre. Así el
microservicio que lo invoca no se entera de que el modelo cambió.
"""
import time
from datetime import datetime

from botocore.exceptions import ClientError

from comun import IMAGEN_XGBOOST, ROL, guardar_estado, leer_caso, leer_estado, nombre_endpoint, sm

EN_TRANSITO = ("Creating", "Updating", "SystemUpdating", "RollingBack", "Deleting")


def estado_endpoint(nombre):
    """El estado del endpoint, o None si no existe."""
    try:
        return sm.describe_endpoint(EndpointName=nombre)["EndpointStatus"]
    except ClientError:
        return None


def main():
    args = leer_caso()
    caso = args.caso
    artefacto = leer_estado(caso, "artefacto", "2_entrenar.py")
    sello = f"{datetime.now():%Y%m%d-%H%M%S}"
    modelo, config, endpoint = f"l10-{caso}-{sello}", f"l10-{caso}-{sello}", nombre_endpoint(caso)

    # 0. Si una corrida anterior se cortó con el endpoint a medias, se espera a que termine y se usa ese,
    #    en vez de crear otro modelo y otra configuración idénticos.
    estado = estado_endpoint(endpoint)
    if estado in EN_TRANSITO:
        print(f"  El endpoint {endpoint} está en {estado}: espero a que termine (se revisa cada 15 s).")
        while estado in EN_TRANSITO:
            time.sleep(15)
            estado = estado_endpoint(endpoint)
        if estado == "InService":
            d = sm.describe_endpoint(EndpointName=endpoint)
            config = d["EndpointConfigName"]
            modelo = sm.describe_endpoint_config(EndpointConfigName=config)["ProductionVariants"][0]["ModelName"]
            print(f"Endpoint {endpoint} en servicio con el modelo {modelo}.")
            print(f"Siguiente: python3 sagemaker/4_probar.py {caso}")
            guardar_estado(caso, endpoint=endpoint, modelo=modelo, config=config, tipo="el que ya estaba")
            return

    # 1. El modelo: la imagen que sabe servir XGBoost + el archivo entrenado del paso 2
    sm.create_model(
        ModelName=modelo,
        ExecutionRoleArn=ROL,
        PrimaryContainer={"Image": IMAGEN_XGBOOST, "ModelDataUrl": artefacto},
    )

    # 2. La configuración: cuánta memoria y cuántas invocaciones a la vez, o qué instancia
    variante = {"VariantName": "principal", "ModelName": modelo}
    if args.tiempo_real:
        variante.update(InstanceType="ml.t2.medium", InitialInstanceCount=1)
        tipo = "tiempo real · ml.t2.medium · COBRA POR HORA: bórralo al terminar"
    else:
        variante["ServerlessConfig"] = {"MemorySizeInMB": args.memoria, "MaxConcurrency": args.concurrencia}
        tipo = f"serverless · {args.memoria} MB · concurrencia {args.concurrencia}"
    sm.create_endpoint_config(EndpointConfigName=config, ProductionVariants=[variante])

    # 3. El endpoint: se crea la primera vez y se actualiza las siguientes.
    #    Si quedó en Failed, se borra y se crea de nuevo.
    if estado == "Failed":
        print(f"  El endpoint {endpoint} quedó en Failed: lo borro para crearlo de nuevo.")
        sm.delete_endpoint(EndpointName=endpoint)
        sm.get_waiter("endpoint_deleted").wait(EndpointName=endpoint)
        estado = None
    if estado is None:
        sm.create_endpoint(EndpointName=endpoint, EndpointConfigName=config)
        print(f"Creando el endpoint {endpoint} ({tipo})")
    else:
        sm.update_endpoint(EndpointName=endpoint, EndpointConfigName=config)
        print(f"Actualizando el endpoint {endpoint} ({tipo})")

    inicio, anterior = time.time(), None
    while True:
        d = sm.describe_endpoint(EndpointName=endpoint)
        if d["EndpointStatus"] != anterior:
            print(f"  {int(time.time() - inicio):>4} s · {d['EndpointStatus']}")
            anterior = d["EndpointStatus"]
        if d["EndpointStatus"] in ("InService", "Failed"):
            break
        time.sleep(15)

    if d["EndpointStatus"] != "InService":
        raise SystemExit(f"El endpoint quedó en Failed: {d.get('FailureReason')}")

    print(f"Endpoint {endpoint} en servicio con el modelo {modelo}.")
    print(f"Siguiente: python3 sagemaker/4_probar.py {caso}")
    guardar_estado(caso, endpoint=endpoint, modelo=modelo, config=config, tipo=tipo)


if __name__ == "__main__":
    main()
