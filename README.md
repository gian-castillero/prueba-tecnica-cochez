# Prueba Técnica: Sistema Híbrido de Recomendación y Personalización de Retail (GenAI + ML)

Prototipo de un sistema que clasifica a los clientes según su riesgo de fuga y luego genera un mensaje de fidelización personalizado, adaptando el tono y la oferta según el segmento del cliente y su historial de compras. El sistema es diseñado con el objetivo de mejorar la retención de clientes mediante campañas de marketing ultra-personalizadas. 

## Contexto del negocio 

Cochez, una empresa líder en materiales de construcción, acabados y ferretería necesita identificar qué clientes están en riesgo de fuga o representan alto valor, y comunicarse con cada cliente de forma distinta, basándose en su segmento, a través de notificaciones push individuales y personalizadas para cada cliente. Este prototipo utiliza data individual de cada cliente (Frecuencia, Recencia, Monto total pagado, Categoría más comprada) para clasificar clientes según su comportamiento de compra, y luego, a través de un LLM, utiliza esa clasificación para generar una notificación personalizada para ese cliente.

## Arquitectura de la solución 

El sistema combina un modelo predictivo (Machine Learning tradicional) y un LLM (GenAI) en un pipeline para generar las notificaciones push: 

1. Los datos crudos se encuentran en una base de datos relacional de con 3 tablas clientes, transacciones y productos. Mínimamente la estructura de estas tablas es: 

- **clientes**: `id_cliente`
- **transacciones**: `id_transaccion`, `id_cliente`, `id_producto`, `fecha`
- **productos**: `id_producto`, `precio`, `categoría`

Nota: Para esta prueba tecnica sin embargo, se entiende que en la práctica esta base de datos tendría columnas adicionales como por ejemplo nombre del cliente, sucursal de transacción, etc.

2. Mediante SQL se construye un dataset con la siguientes variables para cada cliente:

- `ID del cliente`: identificador único del cliente
- `Recencia`: días desde la última compra
- `Frecuencia`: número total de transacciones
- `Monto total gastado`: suma de dinero gastada acumulada en todas las transacciones del cliente
- `Categoría de producto más comprada`: categoría de la cual un cliente ha adquirido más productos

3. Para este prototipo, se genera un dataset sintético de 1000 clientes con estas variables. 

4. Se realiza preprocesamiento de datos normalizando los datos númericos (Recencia, Frecuencia, y Monto total gastado) utilizando StandardScaler. 

5. Se entrena un **Gaussian Mixture Model** (clasificador no supervisado), seleccionando el número óptimo de clusters mediante el **Bayesian Information Criterion** (BIC)

6. Cada cliente es asignado a uno de 4 clusters:

- **Alto valor**
- **Bajo valor**
- **Estándar**
- **Inactivo**

7. A partir del cluster, monto total gastado, y categoría más comprada, se construye un prompt a través de **prompt engineering avanzado** incluyendo asignación de rol y few-shot prompting, en este caso, 2 ejemplos.

8. El prompt se envía al LLM seleccionado, en este caso **GPT 5 mini** a través de la **API de OpenAI** 

Nota: en este prototipo, la respuesta es simulada con un "mock" para evitar gastar créditos. 

9. El modelo devuelve una **respuesta en formato JSON** con la siguiente estructura: 

- **Asunto**
- **Cuerpo del mensaje** 
- **Cupón de descuento sugerido**

Nota: En la práctica, esta respuesta se utiliza para enviar una notificación push personalizada al cliente.

## Estructura del repositorio

```bash
prueba-tecnica-cochez/
│
├── README.md
├── .gitignore
├── Instrucciones - Prueba Técnica.pdf   # contexto, requerimientos e indicaciones
│
└── sistema-recomendacion.ipynb          # Prototipo del sistema
```

## Como correr el código

1. Clonar el repositorio

```bash
git clone https://github.com/gian-castillero/prueba-tecnica-cochez.git
cd prueba-tecnica-cochez
```

2. Instalar las librerias necesarias (en caso de no tenerlas)

```bash
pip install numpy pandas scikit-learn matplotlib openai python-dotenv notebook
```

3. Agregar API key de OpenAI

Crea un archivo llamado `.env` en la misma carpeta del notebook con esta única línea:
 
```
OPENAI_API_KEY=tu-api-key-aqui
```
Nota: Este paso solo es necesario en caso de querer una respuesta real del LLM (`MOCK = False` dentro del notebook) y no utilizar el mock que simula la respuesta.

## Decisiones técnicas 

1. **Estrucura de la base de datos**: 

En la base de datos relacional y, por ende en la consulta SQL, cada fila de **transacciones** representa un producto dentro de una transacción ya que en una transacción se pueden adquirir múltiples productos. Por ende, múltiples filas pueden tener el mismo `id_transaccion`.

2. **Dataset sintético**:

Para este prototipo, el dataset se generó sintéticamente utilizando distribuciones independientes por cliente, con el objetivo de poder implementar el sistema de forma completa. Esto implica una menor segmentación natural en los datos. En un dataset de transacciones reales, los patrones de comportamiento serían más definidos y, por ende, se esperaría una segmentación más marcada. A pesar de esto, el análisis se realizó siguiendo el mismo enfoque que se utilizaría con datos reales.

3. **Modelo seleccionado - Gaussian Mixture Model**:

Elegí un Gaussian Mixture Model para este análisis por 3 principales motivos. 

Primero, el GMM realiza **asignaciones probabilísticas** (soft clustering), es decir, calcula la probabilidad de que el cliente pertenezca a cada uno de los clusters. Esto es importante ya que existe superposición entre los grupos (indicado por el silhouette score), por lo que los grupos no son completamente separables.

Segundo, el GMM es más flexible y **modela formas, tamaños y covarianzas distintas por cluster**. Esto lo adapta mejor a la data de clientes donde no todos se comportan de la misma manera. 

Por último, el modelo permite su evaluación a través de BIC, lo cual permite obtener un modelo que **balancea el fit con complejidad**. Esto permite evitar "overfitting" y elegir el número óptimo de clusters, facilitando la interpretación de resultados. 

4. **Prompt Engineering**:

El prompt combina asignación de rol (experto en e-commerce/marketing de Cochez) con few-shot prompting (dos ejemplos completos de entrada y salida), e instrucciones por segmento y por monto total gastado.

## Resultados

| Segmento | % clientes | Frecuencia promedio | Monto total gastado promedio | Recencia promedio (días) | Interpretación |
|---|---|---|---|---|---|
| Alto valor | 13.3% | 4.04 | $318.72 | 184.8 | Mayor gasto histórico, pero recencia alta indicando alto riesgo de fuga y alta prioridad de reactivación |
| Bajo valor | 32.9% | 3.19 | $39.41 | 196.0 | Menor gasto y frecuencia, clientes ocasionales. Riesgo de fuga alto pero cluster de menor prioridad |
| Estándar | 21.9% | 4.12 | $95.58 | 48.9 | Clientes activos y leales, compran con frecuencia y recientemente. No están en riesgo pero retenerlos debe ser una prioridad |
| Inactivo | 31.9% | 4.34 | $115.09 | 241.7 | Históricamente buenos clientes, pero sin compras recientes. Tienen el mayor riesgo de fuga |
 
Silhouette Score del modelo final: **0.14**, lo cual indica overlap considerable entre segmentos.

Nota: Esto es esperable debido a que el dataset es sintético y no contiene patrones reales de comportamiento diferenciados entre clientes. 

## Como escalaría este sistema en producción

Escalaría este sistema en producción a través de un flujo automático que no dependa del notebook.

Primero, los datos se guardarían y actualizarían en una base de datos real en un data lake o data warehouse. De forma programada y automática, el sistema calcula las variables necesarias y las guarda en una tabla de perfiles de clientes. 

De la misma forma y basado en la tabla de perfiles de clientes, utilizaría el modelo para asignar automáticamente un cluster a cada cliente. Esto se actualizaría en la base de datos. 

Finalmente, desarrollaría una API con la que sistemas como email o notifaciones push de celular puedan acceder al modelo y recibir la información para su uso. 

