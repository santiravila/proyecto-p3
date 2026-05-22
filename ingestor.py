import os
import uuid
import logging
import fitz  # PyMuPDF
import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------
# 1. CONFIGURACIÓN GLOBAL (Constantes)
# ---------------------------------------------------------
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "profesor_ingles_rag"  
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_SIZE = 768  # Tamaño del vector para nomic-embed-text
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Configuración del Logger (Mejor práctica en lugar de usar print)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# 2. CLASE: Procesador de Documentos
# ---------------------------------------------------------
class DocumentProcessor:
    """Se encarga de extraer y fragmentar el texto de los PDFs."""
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def extract_text(self, pdf_path: str) -> str:
        if not os.path.exists(pdf_path):
            logger.error(f"Archivo no encontrado: {pdf_path}")
            raise FileNotFoundError(f"Archivo no encontrado: {pdf_path}")

        logger.info(f"Extrayendo texto de: {os.path.basename(pdf_path)}")
        try:
            document = fitz.open(pdf_path)
            full_text = "".join([page.get_text() for page in document])
            return full_text
        except Exception as e:
            logger.error(f"Error al procesar el PDF: {e}")
            raise

    def split_text(self, text: str) -> list[str]:
        logger.info("Fragmentando el texto en chunks...")
        chunks = self.splitter.split_text(text)
        logger.info(f"Texto dividido exitosamente en {len(chunks)} fragmentos.")
        return chunks

# ---------------------------------------------------------
# 3. CLASE: Gestor de Base de Datos Vectorial (Qdrant)
# ---------------------------------------------------------
class VectorDBManager:
    """Se encarga de la conexión a Qdrant y la ingesta de vectores."""
    
    def __init__(self, host: str = QDRANT_HOST, port: int = QDRANT_PORT):
        logger.info(f"Conectando a Qdrant en {host}:{port}...")
        self.client = QdrantClient(host=host, port=port)
        
    def setup_collection(self, collection_name: str):
        if not self.client.collection_exists(collection_name):
            logger.info(f"Creando colección '{collection_name}' en Qdrant...")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
            )
        else:
            logger.info(f"La colección '{collection_name}' ya existe. Se agregarán nuevos datos.")

    def upload_chunks(self, chunks: list[str], collection_name: str, source_filename: str):
        logger.info(f"Generando embeddings usando Ollama (Modelo: {EMBEDDING_MODEL})...")
        
        points = []
        for chunk in chunks:
            try:
                # Generación del embedding (vector de números)
                response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=chunk)
                
                # Creación de un UUID real para no sobreescribir datos de otros PDFs
                point_id = str(uuid.uuid4()) 
                
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=response["embedding"],
                        payload={
                            "source": source_filename,
                            "content": chunk
                        }
                    )
                )
            except Exception as e:
                logger.error(f"Error al generar embedding para un chunk: {e}")
                continue
        
        if points:
            logger.info(f"Subiendo {len(points)} vectores a Qdrant...")
            self.client.upsert(collection_name=collection_name, points=points)
            logger.info("¡Ingesta completada exitosamente!")
        else:
            logger.warning("No se generaron puntos para subir.")

# ---------------------------------------------------------
# 4. PIPELINE PRINCIPAL (Orquestador)
# ---------------------------------------------------------
class RAGIngestionPipeline:
    """Orquesta el flujo completo: Leer -> Fragmentar -> Vectorizar -> Subir"""
    
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.db_manager = VectorDBManager()

    def run(self, pdf_path: str, collection_name: str = COLLECTION_NAME):
        logger.info("=== INICIANDO PIPELINE DE INGESTA ===")
        
        # 1. Preparar Base de Datos
        self.db_manager.setup_collection(collection_name)
        
        # 2. Procesar Documento
        text = self.doc_processor.extract_text(pdf_path)
        chunks = self.doc_processor.split_text(text)
        
        # 3. Subir a Qdrant
        filename = os.path.basename(pdf_path)
        self.db_manager.upload_chunks(chunks, collection_name, filename)
        
        logger.info("=== PIPELINE FINALIZADO ===")

# ---------------------------------------------------------
# PUNTO DE ENTRADA (Main)
# ---------------------------------------------------------
if __name__ == "__main__":
    # Ruta del archivo que vas a leer (Asegúrate de poner un PDF real aquí)
    PDF_FILE_PATH = "./documentos/test.pdf" 
    
    pipeline = RAGIngestionPipeline()
    
    try:
        pipeline.run(PDF_FILE_PATH)
        logger.info("Verifica los datos en: http://localhost:6333/dashboard")
    except Exception as e:
        logger.error(f"El pipeline falló de manera crítica: {e}")