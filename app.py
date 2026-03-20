import io
import os
import re
import json
import time
import math
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import streamlit as st
from openai import OpenAI

# Dependencias opcionales para lectura de documentos
try:
    from docx import Document
except Exception:
    Document = None

try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================
st.set_page_config(page_title="Asistente de Atención al Cliente para Guías", page_icon="📘", layout="wide")

APP_TITLE = "📘 Asistente de Atención al Cliente para Guías"
APP_SUBTITLE = "Consulta las pautas oficiales y responde según el idioma de la pregunta."
DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
MAX_CHUNK_CHARS = 1800
CHUNK_OVERLAP_CHARS = 250
TOP_K = 6


# ============================================================
# UTILIDADES
# ============================================================
@dataclass
class Chunk:
    chunk_id: str
    heading_path: str
    text: str
    source_name: str


def normalize_whitespace(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def file_sha256(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def cosine_similarity_matrix(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query_vec)
    matrix_norm = np.linalg.norm(matrix, axis=1)
    denom = (matrix_norm * query_norm) + 1e-12
    return (matrix @ query_vec) / denom


def safe_get_api_key() -> str:
    secret_key = st.secrets.get("OPENAI_API_KEY", "") if hasattr(st, "secrets") else ""
    if secret_key:
        return secret_key
    return st.session_state.get("manual_api_key", "")


# ============================================================
# LECTURA DE DOCUMENTOS
# ============================================================
def extract_text_from_txt(file_bytes: bytes) -> str:
    for encoding in ["utf-8", "latin-1", "cp1252"]:
        try:
            return file_bytes.decode(encoding)
        except Exception:
            continue
    return file_bytes.decode("utf-8", errors="ignore")


def extract_text_from_pdf(file_bytes: bytes) -> str:
    if PdfReader is None:
        raise RuntimeError("Falta instalar PyPDF2 para procesar archivos PDF.")

    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for idx, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        pages.append(f"\n\n[Página {idx}]\n{page_text}")
    return "".join(pages)


def extract_sections_from_docx(file_bytes: bytes, source_name: str) -> List[Chunk]:
    if Document is None:
        raise RuntimeError("Falta instalar python-docx para procesar archivos Word.")

    doc = Document(io.BytesIO(file_bytes))

    sections: List[Tuple[str, List[str]]] = []
    heading_stack = [source_name]
    current_lines: List[str] = []
    last_heading_path = source_name

    for para in doc.paragraphs:
        text = normalize_whitespace(para.text)
        if not text:
            continue

        style_name = para.style.name if para.style else ""
        heading_match = re.match(r"Heading\s+(\d+)", style_name, re.IGNORECASE)
        titulo_match = re.match(r"T[íi]tulo\s+(\d+)", style_name, re.IGNORECASE)
        level = None

        if heading_match:
            level = int(heading_match.group(1))
        elif titulo_match:
            level = int(titulo_match.group(1))

        if level is not None:
            if current_lines:
                sections.append((last_heading_path, current_lines.copy()))
                current_lines = []

            heading_stack = heading_stack[:level]
            while len(heading_stack) < level:
                heading_stack.append("Sin título")
            if len(heading_stack) == level:
                heading_stack.append(text)
            else:
                heading_stack[level] = text
                heading_stack = heading_stack[: level + 1]

            heading_path = " > ".join([h for h in heading_stack if h])
            last_heading_path = heading_path
        else:
            current_lines.append(text)

    if current_lines:
        sections.append((last_heading_path, current_lines.copy()))

    chunks: List[Chunk] = []
    for heading_path, lines in sections:
        full_text = normalize_whitespace("\n".join(lines))
        if not full_text:
            continue
        chunks.extend(split_text_into_chunks(full_text, heading_path, source_name))

    if not chunks:
        raw_text = "\n".join([p.text for p in doc.paragraphs])
        chunks.extend(split_text_into_chunks(raw_text, source_name, source_name))

    return chunks


def extract_chunks_generic_text(file_bytes: bytes, source_name: str, extension: str) -> List[Chunk]:
    if extension == ".txt":
        raw_text = extract_text_from_txt(file_bytes)
    elif extension == ".pdf":
        raw_text = extract_text_from_pdf(file_bytes)
    else:
        raise ValueError("Formato no soportado en lectura genérica.")

    raw_text = normalize_whitespace(raw_text)
    return split_text_into_chunks(raw_text, source_name, source_name)


# ============================================================
# CHUNKING
# ============================================================
def split_text_into_chunks(text: str, heading_path: str, source_name: str) -> List[Chunk]:
    text = normalize_whitespace(text)
    if not text:
        return []

    chunks: List[Chunk] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + MAX_CHUNK_CHARS, text_len)
        candidate = text[start:end]

        if end < text_len:
            last_break = max(
                candidate.rfind("\n\n"),
                candidate.rfind(". "),
                candidate.rfind("; "),
            )
            if last_break > int(MAX_CHUNK_CHARS * 0.55):
                end = start + last_break + 1
                candidate = text[start:end]

        chunk_text = normalize_whitespace(candidate)
        if chunk_text:
            chunk_id = hashlib.md5(f"{heading_path}|{start}|{chunk_text[:80]}".encode("utf-8")).hexdigest()
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    heading_path=heading_path,
                    text=chunk_text,
                    source_name=source_name,
                )
            )

        if end >= text_len:
            break

        start = max(end - CHUNK_OVERLAP_CHARS, start + 1)

    return chunks


# ============================================================
# EMBEDDINGS
# ============================================================
def build_embeddings(client: OpenAI, chunks: List[Chunk], embedding_model: str) -> np.ndarray:
    texts = [f"Título: {c.heading_path}\n\nContenido: {c.text}" for c in chunks]
    vectors = []

    batch_size = 64
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=embedding_model, input=batch)
        vectors.extend([item.embedding for item in resp.data])

    return np.array(vectors, dtype=np.float32)


def build_query_embedding(client: OpenAI, query: str, embedding_model: str) -> np.ndarray:
    resp = client.embeddings.create(model=embedding_model, input=query)
    return np.array(resp.data[0].embedding, dtype=np.float32)


# ============================================================
# RECUPERACIÓN
# ============================================================
def retrieve_top_chunks(
    client: OpenAI,
    question: str,
    chunks: List[Chunk],
    embeddings: np.ndarray,
    embedding_model: str,
    top_k: int = TOP_K,
) -> List[Tuple[Chunk, float]]:
    query_vec = build_query_embedding(client, question, embedding_model)
    scores = cosine_similarity_matrix(query_vec, embeddings)
    idxs = np.argsort(scores)[::-1][:top_k]
    return [(chunks[i], float(scores[i])) for i in idxs]


# ============================================================
# RESPUESTA DEL MODELO
# ============================================================
def build_context_block(top_chunks: List[Tuple[Chunk, float]]) -> str:
    blocks = []
    for i, (chunk, score) in enumerate(top_chunks, start=1):
        blocks.append(
            f"[FUENTE {i}]\n"
            f"Ruta: {chunk.heading_path}\n"
            f"Documento: {chunk.source_name}\n"
            f"Relevancia: {score:.4f}\n"
            f"Contenido:\n{chunk.text}"
        )
    return "\n\n------------------------------\n\n".join(blocks)


def ask_guidelines_assistant(
    client: OpenAI,
    question: str,
    top_chunks: List[Tuple[Chunk, float]],
    model_name: str,
) -> str:
    context_block = build_context_block(top_chunks)

    instructions = (
        "Eres un asistente interno para guías de una empresa de viajes. "
        "Tu única fuente válida es el contexto documental proporcionado. "
        "No inventes políticas ni procedimientos. "
        "Si la respuesta no está suficientemente respaldada por el contexto, dilo de forma explícita. "
        "Debes responder en el mismo idioma en que el usuario formule la consulta. "
        "Si el usuario pregunta en portugués, responde en portugués; si pregunta en español, responde en español; "
        "si pregunta en inglés, responde en inglés. "
        "Estructura la respuesta de forma clara y operativa para un guía en servicio. "
        "Al final, añade una sección breve llamada 'Base documental' con las rutas o títulos de los fragmentos utilizados."
    )

    user_input = (
        f"CONSULTA DEL GUÍA:\n{question}\n\n"
        f"CONTEXTO DOCUMENTAL DISPONIBLE:\n{context_block}\n\n"
        "Instrucciones adicionales:\n"
        "1. Responde solo con base en lo que aparece en el contexto.\n"
        "2. Si hay ambigüedad, indícalo.\n"
        "3. Prioriza una redacción práctica y accionable.\n"
        "4. No menciones información técnica sobre embeddings, chunks o recuperación."
    )

    response = client.responses.create(
        model=model_name,
        instructions=instructions,
        input=user_input,
    )

    return response.output_text.strip()


# ============================================================
# CACHE DE DOCUMENTO PROCESADO
# ============================================================
def initialize_cache_state() -> None:
    if "doc_cache" not in st.session_state:
        st.session_state.doc_cache = {}
    if "messages" not in st.session_state:
        st.session_state.messages = []


def process_document(client: OpenAI, uploaded_file, embedding_model: str):
    file_bytes = uploaded_file.read()
    file_hash = file_sha256(file_bytes)
    source_name = uploaded_file.name
    extension = os.path.splitext(source_name)[1].lower()

    cache_key = f"{file_hash}|{embedding_model}"
    if cache_key in st.session_state.doc_cache:
        return st.session_state.doc_cache[cache_key]

    if extension == ".docx":
        chunks = extract_sections_from_docx(file_bytes, source_name)
    elif extension in [".pdf", ".txt"]:
        chunks = extract_chunks_generic_text(file_bytes, source_name, extension)
    else:
        raise ValueError("Formato no soportado. Usa .docx, .pdf o .txt")

    if not chunks:
        raise ValueError("No se pudo extraer contenido útil del documento.")

    embeddings = build_embeddings(client, chunks, embedding_model)

    payload = {
        "file_hash": file_hash,
        "source_name": source_name,
        "chunks": chunks,
        "embeddings": embeddings,
        "processed_at": time.time(),
        "extension": extension,
    }
    st.session_state.doc_cache[cache_key] = payload
    return payload


# ============================================================
# INTERFAZ
# ============================================================
def render_sidebar():
    st.sidebar.header("Configuración")

    if not st.secrets.get("OPENAI_API_KEY", ""):
        st.sidebar.text_input(
            "OpenAI API Key",
            type="password",
            key="manual_api_key",
            help="Si no está en secrets.toml, introdúcela aquí.",
        )
    else:
        st.sidebar.success("API key cargada desde secrets.toml")

    model_name = st.sidebar.selectbox(
        "Modelo de respuesta",
        ["gpt-5-mini", "gpt-5", "gpt-4.1-mini"],
        index=0,
    )

    embedding_model = st.sidebar.selectbox(
        "Modelo de embeddings",
        ["text-embedding-3-small", "text-embedding-3-large"],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        "Recomendación operativa: usa el documento maestro en formato .docx. "
        "Si hoy lo tienes en Google Docs, expórtalo a Word para preservar mejor la jerarquía del índice y los títulos."
    )

    return model_name, embedding_model


def render_header():
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)

    with st.expander("Enfoque recomendado", expanded=False):
        st.markdown(
            """
**Qué hace esta app**
- Toma tu documento de pautas como base documental.
- Divide el contenido por secciones y fragmentos.
- Busca los pasajes más relevantes para cada consulta.
- Responde únicamente con base en esos pasajes.
- Contesta en el idioma de la pregunta del guía.

**Formato recomendado del documento**
- Mejor opción: **.docx** con estilos de títulos bien aplicados.
- Alternativa válida: **.pdf**.
- Menos recomendable: texto plano.

**Uso sugerido**
1. Carga el documento maestro.
2. Espera a que la indexación termine.
3. Escribe una consulta como lo haría un guía.
4. Revisa la respuesta y la base documental utilizada.
            """
        )


def main():
    initialize_cache_state()
    render_header()
    model_name, embedding_model = render_sidebar()

    api_key = safe_get_api_key()
    if not api_key:
        st.warning("Introduce tu API key en la barra lateral o configúrala en secrets.toml para continuar.")
        st.stop()

    client = OpenAI(api_key=api_key)

    col1, col2 = st.columns([0.42, 0.58])

    with col1:
        st.subheader("Documento base")
        uploaded_file = st.file_uploader(
            "Carga el documento maestro",
            type=["docx", "pdf", "txt"],
            help="Para máxima calidad de recuperación, usa Word (.docx).",
        )

        if uploaded_file is not None:
            try:
                with st.spinner("Procesando documento e indexando contenido..."):
                    doc_data = process_document(client, uploaded_file, embedding_model)
                st.success("Documento procesado correctamente.")
                st.metric("Fragmentos indexados", len(doc_data["chunks"]))
                st.caption(f"Documento: {doc_data['source_name']}")
            except Exception as e:
                st.error(f"No se pudo procesar el documento: {e}")
                st.stop()
        else:
            st.info("Carga el documento para habilitar el asistente.")

        with st.expander("Ejemplos de consulta", expanded=False):
            st.markdown(
                """
- ¿Qué debo hacer si un pasajero reclama por una excursión no incluida?
- Como devo proceder se um cliente perder a ligação do transfer?
- What should I do if a customer asks for compensation on the spot?
- ¿Qué corresponde informar cuando hay un cambio de hotel de último momento?
                """
            )

    with col2:
        st.subheader("Chat operativo")

        if uploaded_file is None:
            st.stop()

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        question = st.chat_input("Escribe la consulta del guía...")

        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                try:
                    with st.spinner("Consultando la base documental..."):
                        top_chunks = retrieve_top_chunks(
                            client=client,
                            question=question,
                            chunks=doc_data["chunks"],
                            embeddings=doc_data["embeddings"],
                            embedding_model=embedding_model,
                            top_k=TOP_K,
                        )
                        answer = ask_guidelines_assistant(
                            client=client,
                            question=question,
                            top_chunks=top_chunks,
                            model_name=model_name,
                        )

                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                    with st.expander("Fragmentos utilizados", expanded=False):
                        for i, (chunk, score) in enumerate(top_chunks, start=1):
                            st.markdown(f"**{i}. {chunk.heading_path}**")
                            st.caption(f"Relevancia: {score:.4f}")
                            st.write(chunk.text)
                            st.markdown("---")

                except Exception as e:
                    error_msg = f"Se produjo un error al generar la respuesta: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})


if __name__ == "__main__":
    main()

