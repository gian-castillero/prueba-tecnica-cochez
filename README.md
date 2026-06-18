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

5. Se entrena un Gaussian Mixture Model (clasificador no supervisado), seleccionando el número optimo de clusters mediante el Bayesian Information Criterion (BIC)

6. Cada cliente es asignado a uno de 4 clusters:

- **Alto valor**
- **Bajo valor**
- **Estándar**
- **Inactivo**

7. A partir del cluster, monto total gastado, y categoría más comprada, se construye un prompt a través de prompt engineering avanzado incluyendo few-shot prompting, en este caso, 2 ejemplos.

8. El prompt se envía al LLM seleccionado, en este caso **GPT 5 mini** a través de la API de OpenAI 

Nota: en este prototipo, la respuesta es simulada con un "mock" para evitar gastar créditos. 

9. El modelo devuelve una respuesta en formato JSON con la siguiente estructura: 

- Asunto
- Cuerpo del mensaje 
- Cupón de descuento sugerido

Nota: En la práctica, esta respuesta se utiliza para enviar una notificación push personalizada al cliente.





