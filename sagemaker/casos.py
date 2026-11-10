"""Los cuatro conjuntos de datos del laboratorio. Solo usa la biblioteca estándar: corre en CloudShell sin instalar nada.

- rfm: el del anexo oficial 3.3.5 (calidad de cliente por recencia, frecuencia y monto). Es el común: todos parten con él.
- fraude, chatbot, recomendacion: las mismas columnas y la misma lógica de los starters de L9, uno por caso.

En los cuatro, la etiqueta es 0 o 1 y va en la PRIMERA columna. Es lo que exige el XGBoost integrado de SageMaker.
"""
import math
import random

SEMILLA = 1102


def _sigmoide(z):
    return 1 / (1 + math.exp(-z))


def rfm(rng):
    n = 1000
    filas = []
    for _ in range(n):
        recency = rng.randint(1, 364)        # días desde la última compra (menos es mejor)
        frequency = rng.randint(1, 49)       # visitas
        monetary = rng.randint(10, 999)      # USD gastados
        filas.append([recency, frequency, monetary])
    # El anexo oficial marca como "alta calidad" el 30% de mayor puntaje. Aquí se le agrega ruido,
    # porque sin ruido la etiqueta es una fórmula exacta y el modelo saca 100%, que no enseña nada.
    puntaje = [0.4 * (1 / r) + 0.3 * f + 0.3 * m + rng.gauss(0, 40) for r, f, m in filas]
    corte = sorted(puntaje)[int(0.7 * n)]
    return [[int(p > corte)] + f for p, f in zip(puntaje, filas)]


def fraude(rng):
    n = 5000
    filas = []
    for _ in range(n):
        monto = round(math.exp(rng.gauss(9.9, 1.0)))   # CLP
        hora = rng.randint(0, 23)
        intentos = _poisson(rng, 0.6)                  # transacciones del usuario en la hora previa
        edad = rng.randint(1, 1999)                    # días desde el registro
        kyc = rng.randint(0, 3)                        # nivel de verificación de identidad
        usuarios = 1 + _poisson(rng, 0.3)              # usuarios distintos en el mismo dispositivo
        emulador = int(rng.random() < 0.05)
        pais = int(rng.random() < 0.08)                # 1 = país de la transacción distinto al del usuario
        riesgo = (1.0 * (hora < 6) + 1.0 * (intentos >= 3) + 1.0 * (edad < 30) + 0.7 * (kyc <= 1)
                  + 1.3 * (usuarios >= 3) + 1.6 * emulador + 1.2 * pais + 0.8 * (monto > 150000))
        y = int(rng.random() < _sigmoide(-6.5 + 2.4 * riesgo))
        filas.append([y, monto, hora, intentos, edad, kyc, usuarios, emulador, pais])
    return filas


def chatbot(rng):
    n = 4000
    filas = []
    for _ in range(n):
        turnos = rng.randint(1, 30)
        intentos = rng.randint(0, 2)                   # veces que el bot no entendió
        molestia = round(rng.betavariate(1.5, 4), 2)   # 0 = tranquilo, 1 = muy molesto
        confianza = round(rng.uniform(0.2, 1.0), 2)    # seguridad del bot sobre la intención detectada
        deuda = round(rng.expovariate(1 / 40000))      # CLP
        antiguedad = rng.randint(1, 179)               # meses como cliente
        z = -2.5 + 0.12 * turnos + 0.9 * intentos + 3.0 * molestia - 3.0 * confianza + 1.5 * (deuda > 100000)
        y = int(rng.random() < _sigmoide(z))           # 1 = necesitó ejecutivo
        filas.append([y, turnos, intentos, molestia, confianza, deuda, antiguedad])
    return filas


def recomendacion(rng):
    n = 6000
    filas = []
    for _ in range(n):
        compras = _poisson(rng, 3)                     # compras anteriores del usuario
        ticket = round(rng.uniform(20000, 400000))     # CLP que suele gastar por pedido
        precio = round(rng.uniform(5000, 800000))      # CLP, precio de lista del producto
        vista = rng.randint(0, 1)                      # 1 = ya miró esa categoría
        ventas = _poisson(rng, 30)                     # unidades vendidas del producto en 30 días
        z = -4.5 + 2.2 * vista + 0.04 * ventas + 0.15 * compras - 1.5 * (precio > 1.5 * ticket)
        y = int(rng.random() < _sigmoide(z))           # 1 = compró
        filas.append([y, compras, ticket, precio, vista, ventas])
    return filas


def _poisson(rng, lam):
    # Algoritmo de Knuth: suficiente para lambdas chicos como estos
    limite, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= limite:
            return k
        k += 1


# Nombre de cada columna de entrada, en el orden exacto en que el modelo las espera (sin la etiqueta).
CASOS = {
    "rfm": {
        "generar": rfm,
        "etiqueta": "alta_calidad",
        "columnas": ["recency", "frequency", "monetary"],
        "ejemplo": [10, 30, 800],
    },
    "fraude": {
        "generar": fraude,
        "etiqueta": "es_fraude",
        "columnas": ["monto_clp", "hora", "intentos_previos_1h", "edad_cuenta_dias",
                     "nivel_kyc", "usuarios_distintos", "emulador", "pais_distinto"],
        "ejemplo": [250000, 3, 4, 12, 0, 3, 1, 1],
    },
    "chatbot": {
        "generar": chatbot,
        "etiqueta": "necesito_ejecutivo",
        "columnas": ["turnos", "intentos_fallidos", "puntaje_molestia", "confianza_intencion",
                     "deuda_vigente", "antiguedad_meses"],
        "ejemplo": [18, 2, 0.7, 0.35, 150000, 24],
    },
    "recomendacion": {
        "generar": recomendacion,
        "etiqueta": "compro",
        "columnas": ["compras_previas", "ticket_promedio", "precio_lista", "categoria_vista", "ventas_30d"],
        "ejemplo": [5, 120000, 90000, 1, 45],
    },
}


def generar(caso):
    """Devuelve las filas del caso, con la etiqueta primero. Siempre las mismas: la semilla es fija."""
    rng = random.Random(SEMILLA)
    return CASOS[caso]["generar"](rng)
