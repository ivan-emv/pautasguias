import io
import os
import re
import time
import hashlib
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import streamlit as st
from openai import OpenAI

# Dependencias opcionales
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
st.set_page_config(
    page_title="Asistente de Atención al Cliente para Guías",
    page_icon="📘",
    layout="wide",
)

APP_TITLE = "📘 Asistente de Atención al Cliente para Guías"
APP_SUBTITLE = "Consulta las pautas oficiales y recibe respuestas directas, breves y accionables."
DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
MAX_CHUNK_CHARS = 1800
CHUNK_OVERLAP_CHARS = 250
TOP_K = 8


# ============================================================
# MODELOS DE DATOS
# ============================================================
@dataclass
class Chunk:
    chunk_id: str
    heading_path: str
    text: str
    source_name: str


# ============================================================
# UTILIDADES
# ============================================================
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
    try:
        secret_key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        secret_key = ""
    return secret_key


def detect_priority_focus(question: str) -> str:
    q = question.lower()

    if any(term in q for term in [
        "salud", "médic", "medic", "seguro", "hospital", "ambul", "emergenc", "enfermo", "enferma", "doente", "saúde"
    ]):
        return "health"

    if any(term in q for term in [
        "teléfono", "telefono", "phone", "número", "numero", "llamo", "llamar", "contacto", "contactar"
    ]):
        return "phone"

    return "general"


def extract_phone_lines(text: str) -> List[str]:
    phone_patterns = [
        r"\+?\d[\d\s\-()]{6,}\d",
    ]

    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if any(re.search(pattern, line) for pattern in phone_patterns):
            lines.append(line)
            continue
        if any(keyword in line.lower() for keyword in ["tel", "telefono", "teléfono", "phone", "whatsapp", "seguro", "asistencia"]):
            lines.append(line)

    dedup = []
    seen = set()
    for line in lines:
        key = line.lower()
        if key not in seen:
            seen.add(key)
            dedup.append(line)
    return dedup[:8]


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
            last_break = max(candidate.rfind("\n\n"), candidate.rfind(". "), candidate.rfind("; "))
            if last_break > int(MAX_CHUNK_CHARS * 0.55):
                end = start + last_break + 1
                candidate = text[start:end]

        chunk_text = normalize_whitespace(candidate)
        if chunk_text:
            chunk_id = hashlib.md5(f"{heading_path}|{start}|{chunk_text[:80]}".encode("utf-8")).hexdigest()
            chunks.append(Chunk(chunk_id=chunk_id, heading_path=heading_path, text=chunk_text, source_name=source_name))

        if end >= text_len:
            break
        start = max(end - CHUNK_OVERLAP_CHARS, start + 1)

    return chunks


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

            last_heading_path = " > ".join([h for h in heading_stack if h])
        else:
            current_lines.append(text)

    if current_lines:
        sections.append((last_heading_path, current_lines.copy()))

    chunks: List[Chunk] = []
    for heading_path, lines in sections:
        full_text = normalize_whitespace("\n".join(lines))
        if full_text:
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

    return split_text_into_chunks(normalize_whitespace(raw_text), source_name, source_name)


# ============================================================
# EMBEDDINGS Y RECUPERACIÓN
# ============================================================
def build_embeddings(client: OpenAI, chunks: List[Chunk], embedding_model: str) -> np.ndarray:
    texts = [f"Título: {c.heading_path}\n\nContenido: {c.text}" for c in chunks]
    vectors = []
    batch_size = 64

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(model=embedding_model, input=batch)
        vectors.extend([item.embedding for item in resp.data])

    return np.array(vectors, dtype=np.float32)


def build_query_embedding(client: OpenAI, query: str, embedding_model: str) -> np.ndarray:
    resp = client.embeddings.create(model=embedding_model, input=query)
    return np.array(resp.data[0].embedding, dtype=np.float32)


def retrieve_top_chunks(client: OpenAI, question: str, chunks: List[Chunk], embeddings: np.ndarray, embedding_model: str, top_k: int = TOP_K) -> List[Tuple[Chunk, float]]:
    query_vec = build_query_embedding(client, question, embedding_model)
    scores = cosine_similarity_matrix(query_vec, embeddings)
    idxs = np.argsort(scores)[::-1][:top_k]
    return [(chunks[i], float(scores[i])) for i in idxs]


def rerank_for_priority(question: str, retrieved: List[Tuple[Chunk, float]]) -> List[Tuple[Chunk, float]]:
    focus = detect_priority_focus(question)
    reranked = []

    for chunk, score in retrieved:
        bonus = 0.0
        text_low = f"{chunk.heading_path} {chunk.text}".lower()

        if focus == "health":
            for kw in ["seguro", "asistencia", "emergencia", "hospital", "médic", "medic", "ambul", "teléfono", "telefono"]:
                if kw in text_low:
                    bonus += 0.08
        if focus in ["health", "phone"]:
            phone_lines = extract_phone_lines(chunk.text)
            if phone_lines:
                bonus += 0.12

        reranked.append((chunk, score + bonus))

    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked[:TOP_K]


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


def build_priority_notes(question: str, top_chunks: List[Tuple[Chunk, float]]) -> str:
    focus = detect_priority_focus(question)
    phone_lines = []
    for chunk, _ in top_chunks:
        phone_lines.extend(extract_phone_lines(chunk.text))

    unique_phone_lines = []
    seen = set()
    for line in phone_lines:
        key = line.lower()
        if key not in seen:
            seen.add(key)
            unique_phone_lines.append(line)

    notes = []
    if focus == "health":
        notes.append("La consulta parece relacionada con salud o emergencia. Prioriza teléfonos, seguros, asistencia médica y pasos inmediatos.")
    if focus in ["health", "phone"] and unique_phone_lines:
        notes.append("Datos de contacto detectados en el contexto:")
        for line in unique_phone_lines[:8]:
            notes.append(f"- {line}")
    return "\n".join(notes)


def ask_guidelines_assistant(client: OpenAI, question: str, top_chunks: List[Tuple[Chunk, float]], model_name: str) -> str:
    context_block = build_context_block(top_chunks)
    priority_notes = build_priority_notes(question, top_chunks)

    instructions = (
        "Eres un asistente interno para guías de una empresa de viajes. "
        "Tu única fuente válida es el contexto documental proporcionado. "
        "No inventes políticas, teléfonos, procesos ni excepciones. "
        "Debes responder en el mismo idioma en que el usuario formule la consulta. "
        "La respuesta debe ser breve, directa y operativa. "
        "Empieza siempre con la respuesta principal en 1 o 2 frases. "
        "Después, añade solo los pasos indispensables. "
        "Si en el contexto aparece un teléfono, seguro, número de asistencia o contacto aplicable, colócalo al principio de la respuesta. "
        "No desarrolles explicaciones largas salvo que sean necesarias para no omitir algo importante. "
        "Si el contexto no permite identificar un teléfono o una instrucción exacta, dilo claramente. "
        "Al final, añade una sección breve llamada 'Base documental' con las rutas utilizadas."
    )

    user_input = (
        f"CONSULTA DEL GUÍA:\n{question}\n\n"
        f"PRIORIDADES OPERATIVAS:\n{priority_notes if priority_notes else 'Ninguna prioridad adicional detectada.'}\n\n"
        f"CONTEXTO DOCUMENTAL DISPONIBLE:\n{context_block}\n\n"
        "Reglas de salida:\n"
        "1. Máximo preferente de 120 a 170 palabras, salvo que el contexto exija algo más.\n"
        "2. Si hay un dato crítico inmediato, como un teléfono o contacto, colócalo en la primera línea.\n"
        "3. Usa viñetas solo si ayudan a la acción.\n"
        "4. No repitas el contenido del contexto.\n"
        "5. No menciones aspectos técnicos sobre embeddings, chunks o recuperación."
    )

    response = client.responses.create(
        model=model_name,
        instructions=instructions,
        input=user_input,
    )

    return response.output_text.strip()


# ============================================================
# ESTADO Y PROCESAMIENTO
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
def render_header():
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)


def render_document_status(doc_data: dict):
    c1, c2 = st.columns([0.7, 0.3])
    with c1:
        st.info(f"Documento activo: **{doc_data['source_name']}**")
    with c2:
        st.metric("Fragmentos indexados", len(doc_data["chunks"]))


def render_history():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def main():
    initialize_cache_state()
    render_header()

    api_key = safe_get_api_key()
    if not api_key:
        st.error("No se encontró OPENAI_API_KEY en secrets.toml.")
        st.stop()

    client = OpenAI(api_key=api_key)
    model_name = DEFAULT_MODEL
    embedding_model = DEFAULT_EMBEDDING_MODEL

    top_col1, top_col2 = st.columns([0.62, 0.38])

    with top_col1:
        uploaded_file = st.file_uploader(
            "Carga el documento base",
            type=["docx", "pdf", "txt"],
            help="Para mejor calidad, usa Word (.docx) con títulos bien estructurados.",
        )

    with top_col2:
        with st.expander("Información del asistente", expanded=False):
            st.markdown(
                """
- Responde con base en el documento cargado.
- Prioriza respuestas directas y accionables.
- Si la consulta está en portugués o inglés, responde en ese idioma.
                """
            )

    if uploaded_file is None:
        st.info("Carga el documento para comenzar a consultar.")
        st.stop()

    try:
        with st.spinner("Procesando documento..."):
            doc_data = process_document(client, uploaded_file, embedding_model)
    except Exception as e:
        st.error(f"No se pudo procesar el documento: {e}")
        st.stop()

    render_document_status(doc_data)
    st.markdown("---")

    chat_container = st.container()
    with chat_container:
        render_history()

    question = st.chat_input("Escribe la consulta del guía...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Consultando la base documental..."):
                    retrieved = retrieve_top_chunks(
                        client=client,
                        question=question,
                        chunks=doc_data["chunks"],
                        embeddings=doc_data["embeddings"],
                        embedding_model=embedding_model,
                        top_k=TOP_K,
                    )
                    top_chunks = rerank_for_priority(question, retrieved)
                    answer = ask_guidelines_assistant(
                        client=client,
                        question=question,
                        top_chunks=top_chunks,
                        model_name=model_name,
                    )

                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

                with st.expander("Base utilizada", expanded=False):
                    for i, (chunk, score) in enumerate(top_chunks[:5], start=1):
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
