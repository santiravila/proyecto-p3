# ── Base de datos vectorial ──────────────────────────────────
QDRANT_HOST     = "localhost"
QDRANT_PORT     = 6333
COLLECTION_NAME = "rag_collection"
VECTOR_SIZE     = 768         

# ── Modelos Ollama ───────────────────────────────────────────
EMBED_MODEL     = "nomic-embed-text"
LLM_MODEL       = "gemma2:2b"  

# ── Chunking ─────────────────────────────────────────────────
CHUNK_SIZE      = 900
CHUNK_OVERLAP   = 200

# ── Recuperación ─────────────────────────────────────────────
RETRIEVAL_LIMIT = 6       
SCORE_THRESHOLD = 0.55   
LEXICAL_BONUS   = 0.03    