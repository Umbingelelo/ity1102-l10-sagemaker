---
codigo: L10
titulo: Tu modelo en un endpoint serverless de SageMaker
descripcion: Entrenas un XGBoost con un Training Job de SageMaker, lo publicas en un endpoint serverless, mides su cold start y sus métricas, repites todo con los datos de tu caso y lo consumes desde un microservicio FastAPI sin una sola access key. Todo en AWS Academy Learner Lab, desde CloudShell.
minutos: 110
puntos: 100
orden: 100
experiencia: 3
---

## Qué vas a lograr

Hoy haces la pieza central de EP3: tu modelo deja de vivir dentro de la imagen del microservicio de IA
y pasa a un **endpoint de Amazon SageMaker** (una dirección de AWS que recibe filas de datos y devuelve
predicciones). El microservicio ya no predice: le pregunta a SageMaker.

```text
CloudShell ──1──► S3 (datos de entrenamiento y validación)
           ──2──► Training Job en ml.m5.large ──► S3 (model.tar.gz)
           ──3──► Endpoint serverless  l10-<caso>
           ──4──► invoke_endpoint: cold start, latencia, accuracy, precision, recall, F1

Tramo 5:  FastAPI (/predict) ──invoke_endpoint──► Endpoint l10-<caso>
```

| Tramo | Min | Qué haces |
|---|---|---|
| **1** | 10 | Inicias el Learner Lab, abres CloudShell y clonas el repositorio |
| **2** | 25 | Generas los datos comunes, los subes a S3 y entrenas con un Training Job |
| **3** | 20 | Publicas el endpoint serverless, lo invocas y evalúas el modelo |
| **4** | 35 | Repites el camino con los datos de **tu caso** y pruebas otros umbrales |
| **5** | 20 | Mides el cold start, consumes tu endpoint desde FastAPI y limpias |

**Lo que entregas en Pulso:** las respuestas de las cajas y, en el punto de control final, la salida de
`4_probar.py` de tu caso y la respuesta de `/predict` de tu microservicio.

### El repositorio de la clase

Todo el código está en
[github.com/Umbingelelo/ity1102-l10-sagemaker](https://github.com/Umbingelelo/ity1102-l10-sagemaker).
Son cinco scripts numerados que solo usan `boto3` (la biblioteca de Python para hablarle a AWS), que
CloudShell ya trae instalada, y un microservicio FastAPI.

### Palabras que vas a ver hoy

| Palabra | Qué es |
|---|---|
| **S3 · bucket** | El almacenamiento de archivos de AWS. Un *bucket* es tu carpeta raíz ahí |
| **ARN** | El nombre completo y único de algo en AWS, como `arn:aws:iam::123456789012:role/LabRole` |
| **Instancia** · `ml.m5.large` | Una máquina virtual. El nombre dice su tamaño |
| **XGBoost** | Un modelo hecho de muchos árboles de decisión chicos, cada uno corrigiendo al anterior |
| **AUC** | La probabilidad de que el modelo le dé más puntaje a un «1» que a un «0» tomados al azar. 0,5 es adivinar; 1 es perfecto |
| **Matriz de confusión** | Cuenta los aciertos y errores: VP (dijo 1 y era 1), FP (dijo 1 y era 0), FN (dijo 0 y era 1), VN (dijo 0 y era 0) |
| **Precision · recall · F1** | Precision: de los que marcó como 1, cuántos lo eran. Recall: de los que eran 1, cuántos encontró. F1: un número que junta los dos y baja si cualquiera es bajo |
| **p50 · p95 · p99** | El 50% (o el 95%, o el 99%) de las llamadas tardó menos que ese número |
| **Concurrencia aprovisionada** | Contenedores de un endpoint serverless que AWS mantiene encendidos para que no haya cold start. Se pagan aunque nadie llame |
| **Variable de entorno** | Escribir `NOMBRE=valor` justo antes de un comando le pasa ese valor al programa solo esa vez. Sin espacios alrededor del `=` |

:::alerta
Trabaja en **AWS Academy Learner Lab**, nunca con una cuenta personal. Hoy **no se crean usuarios de
IAM ni access keys** (usuario y contraseña permanentes para programas): el Learner Lab no lo permite y
no hace falta. Todo usa el rol `LabRole`, que ya existe en tu cuenta. Si una guía de internet te dice
«crea un usuario IAM con `AmazonSageMakerFullAccess`», esa guía no es para el Learner Lab.
:::

---

## Tramo 1 · CloudShell y el repositorio

### 1.1 Iniciar el Learner Lab

**EN AWS ACADEMY.** Curso → **Learner Lab** → **Start Lab**. Espera el punto verde junto a **AWS** y
haz clic en **AWS** para abrir la consola. Arriba a la derecha, la región tiene que ser
**N. Virginia (`us-east-1`)**. Arriba a la derecha también ves tu usuario: empieza con `voclabs/`.

### 1.2 Abrir CloudShell

En la barra superior de la consola, haz clic en el ícono de terminal (`>_`, a la izquierda de la
campana), o escribe `CloudShell` en el buscador. La primera vez aparece una ventana de bienvenida:
**Cerrar**. Espera a ver el prompt `~ $`.

CloudShell es una terminal Linux que corre dentro de AWS con **tus mismas credenciales de la
consola**. Por eso los scripts no necesitan claves: `boto3` las toma de la sesión.

:::pista
Si pegas varias líneas a la vez, CloudShell puede mostrar el aviso **Safe paste for multiline text**:
haz clic en **Paste**. Y si CloudShell se cierra (se cierra solo después de un rato sin teclear) o lo
vuelves a abrir, tus archivos siguen ahí: lo primero que escribes es `cd ~/ity1102-l10-sagemaker`. No
hay que clonar de nuevo.
:::

### 1.3 Clonar el repositorio

**EN CLOUDSHELL:**

```bash
git clone https://github.com/Umbingelelo/ity1102-l10-sagemaker.git
cd ity1102-l10-sagemaker
python3 --version
aws sts get-caller-identity
```

**Lo visible:** `Python 3.` y un número (el 10 de noviembre CloudShell traía 3.13), y un JSON con tu
`Account` (12 dígitos) y un `Arn` que contiene `assumed-role/voclabs/`. Ese número de cuenta aparece
en el nombre de tu bucket.

Mira cómo se arma el rol: `grep -n "ROL\|BUCKET\|IMAGEN" sagemaker/comun.py`.

:::caja{1.1 corta}
Copia el `Arn` que devolvió `aws sts get-caller-identity`. Luego, mirando `comun.py`: ¿qué rol usa
SageMaker para leer tus datos de S3 y escribir el modelo? ¿Por qué en este laboratorio no aparece
ninguna `AWS_ACCESS_KEY_ID` en el código?
:::

---

## Tramo 2 · Datos y entrenamiento

Empiezas con el **conjunto común**, el mismo del ejemplo de la guía oficial de la asignatura: 1.000
clientes con recencia, frecuencia y monto, y una etiqueta que dice si es un cliente de alta calidad.
Es chico a propósito: así todos llegan al endpoint antes de pasar a su caso.

### 2.1 Generar y subir los datos

```bash
python3 sagemaker/1_datos.py rfm
```

**Lo visible:**

```text
Bucket creado: ity1102-l10-<tu cuenta>
Caso rfm: 1000 filas, 29.9% con etiqueta 1.
  entrenamiento 700 · validación 150 · prueba 150 (queda local)
  s3://ity1102-l10-<tu cuenta>/rfm/train/train.csv
  s3://ity1102-l10-<tu cuenta>/rfm/validation/validation.csv
```

Mira las primeras filas de los archivos que generó el script:

```bash
head -3 datos/rfm/train.csv
head -3 datos/rfm/test.csv
```

**Lo visible:** `train.csv` empieza con `1,213,40,814`; `test.csv` empieza con
`alta_calidad,recency,frequency,monetary`.

:::caja{2.1 corta}
¿Qué diferencia ves entre la primera línea de `train.csv` y la de `test.csv`? ¿Qué columna va primero
y por qué el XGBoost de SageMaker la necesita ahí? ¿Por qué `test.csv` no se sube a S3?
:::

### 2.2 Lanzar el Training Job

```bash
python3 sagemaker/2_entrenar.py rfm
```

El script crea un **Training Job** y lo revisa cada 20 segundos. SageMaker enciende una `ml.m5.large`,
descarga la imagen de XGBoost, lee tus CSV desde S3, entrena, guarda el modelo en S3 y apaga la
máquina. Tarda **3 a 5 minutos**.

:::ojo
**No aprietes Ctrl+C mientras espera.** Si lo cortas, el entrenamiento sigue en AWS; vuelve a correr
el mismo comando y el script retoma ese trabajo en vez de lanzar otro. Para mirar otra cosa mientras
espera, abre una segunda pestaña de CloudShell con el **+** de arriba, o mira la consola.
:::

Mientras espera, ábrelo en la consola: buscador → **SageMaker AI** → menú izquierdo **Training** →
**Training jobs** (en español: **Entrenamiento** → **Trabajos de entrenamiento**). Entra a tu trabajo
`l10-rfm-…` y busca **Hyperparameters**, **Input data configuration** y **Output**.

**Lo visible al terminar** (los segundos pueden variar):

```text
     0 s · InProgress · Starting
    60 s · InProgress · Downloading
   160 s · InProgress · Uploading
   180 s · Completed · Completed
  validation:auc: 0.9479
  train:auc: 0.9887
  Tiempo facturado: 129 s
  Modelo: s3://ity1102-l10-<tu cuenta>/rfm/modelos/l10-rfm-…/output/model.tar.gz
```

:::caja{2.2 corta}
Copia las dos métricas y el tiempo facturado. ¿Por qué el AUC de entrenamiento es mayor que el de
validación, y qué nombre tiene el problema cuando la diferencia es grande? El trabajo duró unos 180
segundos y se facturaron menos: ¿qué parte no se cobra? Entrenar en `ml.m5.large` en us-east-1 cuesta
alrededor de **0,115 USD por hora** (puedes confirmarlo en aws.amazon.com/sagemaker/pricing →
*On-Demand Pricing* → *Training*): ¿cuánto costó este entrenamiento?
:::

---

## Tramo 3 · El endpoint serverless

### 3.1 Publicar el modelo

```bash
python3 sagemaker/3_endpoint.py rfm
```

El script hace tres llamadas, en este orden (puedes verlas con `grep -n "sm\.create" sagemaker/3_endpoint.py`):

1. `create_model`: une la imagen de XGBoost con tu `model.tar.gz`.
2. `create_endpoint_config`: dice **cómo** se sirve. Aquí, `ServerlessConfig` con 2048 MB de memoria
   y concurrencia máxima 2 (dos llamadas atendidas a la vez).
3. `create_endpoint`: crea el endpoint `l10-rfm`. Si ya existía, lo actualiza con la configuración
   nueva sin cambiarle el nombre.

**Lo visible:** `Creating` y, unos **3 minutos** después, `InService` (listo para recibir llamadas).
Tampoco aquí aprietes Ctrl+C: si se corta, vuelve a correr el mismo comando y espera a que termine.

:::alerta
**Si el endpoint queda en `Failed` o la llamada responde `AccessDeniedException`**, avisa al docente
y usa el plan B: `python3 sagemaker/3_endpoint.py rfm --tiempo-real`. El script borra el endpoint
fallido y crea uno de tiempo real en `ml.t2.medium`, que **cobra cada hora que está encendido**. Si
usas el plan B, en el tramo 4 agrega también `--tiempo-real` al `3_endpoint.py` de tu caso, y al final
borras los dos endpoints (tramo 5).
:::

### 3.2 Invocar y evaluar

```bash
python3 sagemaker/4_probar.py rfm
```

**Lo visible** (los milisegundos varían):

```text
Endpoint l10-rfm · fila de ejemplo: 10,30,800
  1.ª invocación:     156 ms · probabilidad 0.8977
  2.ª invocación:      59 ms · probabilidad 0.8977
  20 invocaciones en caliente: p50 39 ms · p95 48 ms (medido desde CloudShell)
Conjunto de prueba: 150 filas, 44 con etiqueta 1 · umbral 0.5
  matriz de confusión: VP 34 · FP 6 · FN 10 · VN 100
  accuracy 0.893 · precision 0.850 · recall 0.773 · F1 0.810
  línea base «todo 0»: accuracy 0.707, recall 0
```

La **línea base** es el modelo más tonto posible: decir siempre «0». Si tu modelo no le gana, no sirve.

Si vas bien de tiempo, busca la invocación en la consola: **SageMaker AI** → **Inference** →
**Endpoints** (en español: **Inferencia** → **Puntos de enlace**) → `l10-rfm` → pestaña **Monitor** o
el enlace a **CloudWatch** (donde AWS guarda las métricas de tus servicios). Las métricas
`Invocations` y `ModelLatency` son las evidencias que pide EP3. Si no aparecen en dos minutos, sigue y
vuelve a mirar al final.

:::caja{3.1 corta}
Copia la salida de `4_probar.py`. Un modelo que dijera siempre «0» tendría accuracy 0,707: ¿cuánto
mejor es el tuyo y por qué el accuracy solo no basta para decidirlo? ¿Qué significan, en este negocio,
los 10 falsos negativos?
:::

:::control{1}
**Punto de control:** `4_probar.py rfm` muestra probabilidades y métricas, y el endpoint `l10-rfm`
aparece `InService` en la consola. Muéstraselo al docente. Mientras esperas que pase por tu puesto,
**sigue con el tramo 4**: el entrenamiento de tu caso tarda.
:::

---

## Tramo 4 · Los datos de tu caso

Ahora el mismo camino con los datos de tu caso. Son las mismas columnas de los starters de L9. Si
faltaste a L9, la tabla dice qué significa cada una, y los comentarios de `sagemaker/casos.py` explican
cómo se generó la etiqueta.

| Tu caso | Argumento (en minúsculas, sin tilde) | Qué predice el modelo | Columnas |
|---|---|---|---|
| 1 · Recomendación | `recomendacion` | Probabilidad de que el usuario compre el producto que se le muestra | `compras_previas`, `ticket_promedio`, `precio_lista`, `categoria_vista`, `ventas_30d` |
| 2 · Chatbot | `chatbot` | Probabilidad de que la conversación termine necesitando un ejecutivo | `turnos`, `intentos_fallidos`, `puntaje_molestia`, `confianza_intencion`, `deuda_vigente`, `antiguedad_meses` |
| 3 · Fraude | `fraude` | Probabilidad de que la transacción sea fraude | `monto_clp`, `hora`, `intentos_previos_1h`, `edad_cuenta_dias`, `nivel_kyc`, `usuarios_distintos`, `emulador`, `pais_distinto` |

### 4.1 Los cuatro pasos, de corrido

**Copia solo el bloque de tu caso.**

Si tu caso es **fraude**:

```bash
python3 sagemaker/1_datos.py fraude
python3 sagemaker/2_entrenar.py fraude
python3 sagemaker/3_endpoint.py fraude
python3 sagemaker/4_probar.py fraude
```

Si tu caso es **chatbot**:

```bash
python3 sagemaker/1_datos.py chatbot
python3 sagemaker/2_entrenar.py chatbot
python3 sagemaker/3_endpoint.py chatbot
python3 sagemaker/4_probar.py chatbot
```

Si tu caso es **recomendación**:

```bash
python3 sagemaker/1_datos.py recomendacion
python3 sagemaker/2_entrenar.py recomendacion
python3 sagemaker/3_endpoint.py recomendacion
python3 sagemaker/4_probar.py recomendacion
```

Son unos 8 a 10 minutos de espera entre el entrenamiento y el endpoint. Úsalos para leer la función de
tu caso en `casos.py`
[en GitHub](https://github.com/Umbingelelo/ity1102-l10-sagemaker/blob/main/sagemaker/casos.py) o en
una segunda pestaña de CloudShell. **No aprietes Ctrl+C** en la pestaña que está corriendo.

:::caja{4.1 codigo}
Pega la salida completa de `4_probar.py` con tu caso. Compara la proporción de etiquetas 1 de tu caso
con la del conjunto común. ¿Cuál de las cuatro métricas mirarías primero para tu negocio, y por qué esa
y no el accuracy?
:::

### 4.2 El umbral es una decisión de negocio

`4_probar.py` usa 0.5 por defecto. El modelo no cambia: cambia desde qué probabilidad decides «1».
Escribe el número **con punto** y **sin espacios** alrededor del `=`. Por ejemplo, con fraude:

```bash
UMBRAL=0.3 python3 sagemaker/4_probar.py fraude
UMBRAL=0.7 python3 sagemaker/4_probar.py fraude
```

(Cambia `fraude` por `chatbot` o `recomendacion` si ese es tu caso.)

Para justificar tu elección, abre el PDF de tu caso, el mismo de EP1, y mira las secciones **2** (el
problema con números y las metas), **4** (requerimientos funcionales, RF) y **5** (no funcionales, RNF).

:::caja{4.2 corta}
Anota precision y recall con los umbrales 0.3, 0.5 y 0.7. ¿Qué pasa con cada una al subir el umbral?
¿Cuál elegirías para tu caso y qué requerimiento o meta del documento del caso usarías para
justificarlo?
:::

---

## Tramo 5 · Cold start, FastAPI y limpieza

### 5.1 El cold start de verdad

En el tramo 3 la primera invocación fue rápida porque el endpoint recién se había creado. Un endpoint
serverless que no recibe llamadas durante unos minutos libera su capacidad, y la siguiente invocación
paga el arranque: eso es el **cold start**. Tu endpoint `rfm` lleva más de 20 minutos sin uso. Mídelo
**ahora**, antes de cualquier otra cosa del tramo:

```bash
python3 sagemaker/4_probar.py rfm
```

En algunos casos la lenta es la **2.ª** invocación y no la 1.ª: con concurrencia 2, SageMaker puede
levantar un segundo contenedor que paga su propio arranque. Vale igual: copia las dos cifras. Si usaste
el plan B de tiempo real no vas a ver cold start: responde la caja con las cifras del pre-lab
(6.352 ms y 69 ms).

:::caja{5.1 corta}
Copia los tiempos de la 1.ª y la 2.ª invocación. ¿Cuántas veces más lenta fue la primera? Si tu caso
exige p95 bajo 200 ms (recomendación, RNF1), p99 bajo 100 ms (fraude, RNF1) o p95 bajo 3 s con todo
incluido (chatbot, RNF1), ¿qué harías con este cold start? Nombra una opción y su costo.
:::

### 5.2 FastAPI contra tu endpoint

El archivo `api/main.py` es el microservicio de IA de EP3 en mínimo: valida la entrada con las columnas
de tu caso, invoca el endpoint y devuelve la probabilidad, la decisión y la latencia de SageMaker. No
tiene ninguna clave: `boto3` toma las credenciales del entorno.

Primero instala FastAPI (una sola vez) y copia la lista de columnas al lado de la API:

```bash
cd ~/ity1102-l10-sagemaker
pip install -q fastapi==0.115.12 uvicorn==0.34.2
cp sagemaker/casos.py api/
```

Después levanta el servicio **con la línea de tu caso**. La línea deja FastAPI corriendo en segundo
plano (eso hace el `&` del final) y guarda sus mensajes en `api.log`; `pkill -f uvicorn` detiene uno
anterior si lo había.

Si tu caso es **fraude**:

```bash
pkill -f uvicorn; cd ~/ity1102-l10-sagemaker/api && CASO=fraude nohup python3 -m uvicorn main:app --port 8000 > ../api.log 2>&1 &
```

Si tu caso es **chatbot**:

```bash
pkill -f uvicorn; cd ~/ity1102-l10-sagemaker/api && CASO=chatbot nohup python3 -m uvicorn main:app --port 8000 > ../api.log 2>&1 &
```

Si tu caso es **recomendación**:

```bash
pkill -f uvicorn; cd ~/ity1102-l10-sagemaker/api && CASO=recomendacion nohup python3 -m uvicorn main:app --port 8000 > ../api.log 2>&1 &
```

Espera unos segundos y pregúntale si está vivo:

```bash
cd ~/ity1102-l10-sagemaker
sleep 5; curl -sS localhost:8000/health || tail -20 api.log
```

**Lo visible:** `{"caso":"fraude","endpoint":"l10-fraude","estado_endpoint":"InService"}`, con tu caso.
Si dice `"rfm"`, levantaste la línea equivocada: vuelve a pegar la de tu caso. Si sale
`Couldn't connect to server` (o `Connection refused`), espera cinco segundos y repite solo el `curl`.
Si en vez de eso aparecen líneas de `api.log`: las que empiezan con `INFO` significan que está
arrancando (repite el `curl`); si ves `Error` o `Traceback`, ese es el problema.

Ahora una predicción. **Copia el comando de tu caso:**

```bash
# fraude
curl -s -X POST localhost:8000/predict -H 'content-type: application/json' -d '{"monto_clp":250000,"hora":3,"intentos_previos_1h":4,"edad_cuenta_dias":12,"nivel_kyc":0,"usuarios_distintos":3,"emulador":1,"pais_distinto":1}'
```

```bash
# chatbot
curl -s -X POST localhost:8000/predict -H 'content-type: application/json' -d '{"turnos":18,"intentos_fallidos":2,"puntaje_molestia":0.7,"confianza_intencion":0.35,"deuda_vigente":150000,"antiguedad_meses":24}'
```

```bash
# recomendacion
curl -s -X POST localhost:8000/predict -H 'content-type: application/json' -d '{"compras_previas":5,"ticket_promedio":120000,"precio_lista":90000,"categoria_vista":1,"ventas_30d":45}'
```

**Lo visible** (con fraude):
`{"caso":"fraude","probabilidad":0.9943,"decision":1,"umbral":0.5,"endpoint":"l10-fraude","latencia_sagemaker_ms":124}`.
La primera predicción puede tardar segundos (`latencia_sagemaker_ms` de unos 6000): es el cold start
del endpoint de tu caso, que llevaba un rato sin uso. Repítela y verás la cifra normal. Si quieres que
la API use el umbral que elegiste en 4.2, agrega `UMBRAL=0.3` junto a `CASO=…` en la línea que
levanta el servicio.

Y una entrada mala: la misma de tu caso, pero con un texto donde va un número. Con fraude:

```bash
curl -s -o /dev/null -w '%{http_code}\n' -X POST localhost:8000/predict -H 'content-type: application/json' -d '{"monto_clp":"mucho","hora":3,"intentos_previos_1h":4,"edad_cuenta_dias":12,"nivel_kyc":0,"usuarios_distintos":3,"emulador":1,"pais_distinto":1}'
```

Con chatbot:

```bash
curl -s -o /dev/null -w '%{http_code}\n' -X POST localhost:8000/predict -H 'content-type: application/json' -d '{"turnos":"muchos","intentos_fallidos":2,"puntaje_molestia":0.7,"confianza_intencion":0.35,"deuda_vigente":150000,"antiguedad_meses":24}'
```

Con recomendación:

```bash
curl -s -o /dev/null -w '%{http_code}\n' -X POST localhost:8000/predict -H 'content-type: application/json' -d '{"compras_previas":"muchas","ticket_promedio":120000,"precio_lista":90000,"categoria_vista":1,"ventas_30d":45}'
```

:::caja{5.2 codigo}
Pega la respuesta de `/predict` y el código HTTP de la entrada mala. Busca con
`grep -n COLUMNAS api/main.py` en qué línea se decide el orden de las columnas que se mandan a
SageMaker. ¿Qué pasaría si ese orden no fuera el mismo del entrenamiento?
:::

### 5.3 Limpiar

Detén el microservicio y borra el endpoint común. **El de tu caso déjalo**: un endpoint serverless sin
llamadas no cobra, y lo vas a usar en EP3.

```bash
cd ~/ity1102-l10-sagemaker
pkill -f uvicorn
python3 sagemaker/5_borrar.py rfm
```

:::alerta
Solo si usaste el plan B (`--tiempo-real`), borra **también** el endpoint de tu caso, por ejemplo
`python3 sagemaker/5_borrar.py fraude`. El modelo entrenado sigue en S3 y su ruta queda anotada: lo
vuelves a publicar con `python3 sagemaker/3_endpoint.py fraude --tiempo-real` en unos minutos, sin
reentrenar.
:::

:::control{2}
**Punto de control final:** la salida de `4_probar.py` con tu caso, la respuesta de `/predict` y el
endpoint `l10-rfm` ya borrado en la consola. Muéstraselo al docente y sube las cajas a Pulso.
:::

---

## Lo que esto tiene que ver con EP3

En EP3 tu microservicio de IA corre en una EC2 y llama a **este mismo endpoint** de tu caso. Lo que
cambia es de dónde salen las credenciales: en la EC2 no hay sesión de consola, así que la instancia
lleva el perfil `LabInstanceProfile` (al crearla: **Advanced details** → **IAM instance profile**; si
ya existe: **Actions** → **Security** → **Modify IAM role**). Con eso, en una EC2 Ubuntu 24.04:

```bash
sudo apt-get update && sudo apt-get install -y docker.io docker-compose-v2 git
git clone https://github.com/Umbingelelo/ity1102-l10-sagemaker.git && cd ity1102-l10-sagemaker
sudo CASO=fraude docker compose up -d --build
curl -s localhost:8000/health
```

El `compose.yaml` usa `network_mode: host` para que el contenedor llegue al servicio de metadatos de
la instancia, que es de donde `boto3` saca las credenciales del rol. No hay ninguna clave en el
servidor. Y los datos de entrenamiento dejan de ser los sintéticos de hoy: los reemplazas por la
salida de tu `preparar.py` de EP2, con la etiqueta en la primera columna y sin header. Para eso hay
que adaptar `1_datos.py` y la lista de columnas de `casos.py`, y volver a correr los pasos 1 a 3.

### Si algo no responde

| Señal | Causa probable | Qué hacer |
|---|---|---|
| `can't open file … No such file or directory` | Estás en otra carpeta (pasa al reabrir CloudShell) | `cd ~/ity1102-l10-sagemaker` |
| `error: argument caso: invalid choice: 'CASO'` | Copiaste la palabra `CASO` o escribiste el caso con mayúscula o tilde | Usa `fraude`, `chatbot`, `recomendacion` o `rfm`, en minúsculas y sin tilde |
| `could not convert string to float: '0,3'` | Escribiste el umbral con coma | `UMBRAL=0.3`, con punto y sin espacios |
| `Unable to locate credentials` o `ExpiredToken` | La sesión del Learner Lab terminó | Vuelve a AWS Academy, **Start Lab**, abre CloudShell y escribe `cd ~/ity1102-l10-sagemaker` |
| `Falta 'artefacto' para el caso…` | Saltaste un paso o lo corriste con otro argumento | Corre los pasos en orden, con el mismo argumento |
| El Training Job queda en `Failed` | Mira `FailureReason` en la consola del Training Job | Avisa al docente con el mensaje |
| `ValidationError … Could not find endpoint` | El endpoint no existe o tiene otro nombre | `python3 sagemaker/3_endpoint.py` con tu caso |
| `ModelError` al invocar | El número de columnas no coincide con el del entrenamiento | Revisa que el argumento sea el mismo en los cuatro pasos |
| `curl` no muestra nada o dice `Couldn't connect to server` | FastAPI no arrancó | `tail -20 api.log`. Casi siempre falta `cp sagemaker/casos.py api/` |
| `/health` dice `"caso":"rfm"` | Levantaste FastAPI sin tu caso | Pega de nuevo la línea de tu caso del 5.2 |
