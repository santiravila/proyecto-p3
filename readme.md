# Asistente Legal RAG - Infracciones de Tránsito (Colombia)

Proyecto final para la asignatura Programación III. 

Este proyecto implementa un chatbot en Telegram respaldado por una arquitectura RAG (Retrieval-Augmented Generation) 100% local. El bot está diseñado para actuar como un asistente legal que responde preguntas sobre el **Manual de Infracciones de Tránsito de Colombia**.

El objetivo principal del proyecto fue construir un sistema que **no alucine**. Si la información no está explícitamente en el documento proporcionado, el LLM está instruido para admitir que no tiene la respuesta.

## Tech Stack 

Toda la inferencia y almacenamiento ocurre en local. No se utilizan APIs externas (como OpenAI).

* **Orquestación / Interfaz:** Python 3.10+, `python-telegram-bot`.
* **Almacenamiento Vectorial:** Qdrant (Docker).
* **Motor LLM:** Ollama (Docker).
* **Modelos:** 
  * Embeddings: `nomic-embed-text`
  * Generación: `gemma2:2b`
* **Procesamiento de Documentos:** `PyMuPDF` (extracción), `langchain-text-splitters` (chunking).

## Estructura del Proyecto

El código está modularizado para separar la ingesta de datos, la lógica de recuperación y la interfaz de usuario:

* `config.py`: Variables globales, puertos y configuración de los modelos (evita hardcodear variables).
* `ingestor.py`: Script de un solo uso. Extrae el texto del PDF, lo divide en chunks con solapamiento, genera los embeddings y hace el *upsert* a Qdrant usando UUIDs.
* `rag_test.py`: El "cerebro". Maneja el *Retrieval* (búsqueda en Qdrant + Reranking Léxico para mejorar precisión) y la *Generación* (construcción del prompt estricto para Gemma2).
* `bot.py`: Controlador de Telegram. Maneja la asincronía y da feedback visual al usuario mientras el modelo local procesa la respuesta.
* `docker-compose.yml`: Archivo de infraestructura para levantar Qdrant y Ollama.

## Cómo ejecutar el proyecto

### 1. Levantar la Infraestructura (Docker)
Asegúrate de tener Docker Desktop abierto y ejecuta:

```bash
docker-compose up -d
```

### 2. Descargar los Modelos en Ollama
La primera vez que corras el proyecto, necesitas descargar los modelos dentro del contenedor de Ollama:

```bash
docker exec -it ollama_server ollama pull nomic-embed-text
docker exec -it ollama_server ollama pull gemma2:2b
```

### 3. Entorno de Python
Crea tu entorno virtual e instala las dependencias:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Ingesta del Documento (Base de Conocimiento)
Coloca el PDF del manual de tránsito en la carpeta `documentos/` y corre el ingestor. Esto poblará la base de datos vectorial en Qdrant.

```bash
python ingestor.py
```
*(Puedes verificar que los vectores subieron correctamente entrando a `http://localhost:6333/dashboard` en tu navegador).*

### 5. Iniciar el Bot
Finalmente, levanta el proceso que escucha a Telegram:

```bash
python bot.py
```
Ve a Telegram, busca tu bot y envíale `/start`.

## ⚠️ Notas Técnicas y Limitaciones Conocidas

* **Timeouts en Telegram:** Como el LLM corre localmente (y depende del hardware de la máquina host), el tiempo de inferencia puede variar entre 15 y 40 segundos. Se configuraron los *timeouts* de la librería de Telegram a 120 segundos para evitar que la conexión se caiga mientras el LLM genera la respuesta.
* **Pérdida de Contexto por Fragmentación (Chunking):** Debido a que el RAG divide el PDF en fragmentos de 1000 caracteres, hay consultas complejas que pueden fallar si la respuesta está dividida en páginas muy lejanas. Por ejemplo: El valor de la multa (Categoría C) se define en la página 24, pero la regla de tránsito específica está en la página 40. Al recuperar el chunk de la regla, el modelo no tendrá el contexto del precio y responderá "No encontré esa información". Esto es un comportamiento esperado y deseado, ya que priorizamos que el bot **no invente datos legales**.