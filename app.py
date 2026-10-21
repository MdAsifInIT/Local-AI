"""Local document RAG. Importing this module does not start Streamlit."""
from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import os
import re
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

# Keep local workloads local, even when the parent shell enables tracing.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ[_name] = "1"

import httpx
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

OLLAMA_URL = "http://host.docker.internal:11434"
LLM_MODEL = "gemma4:e4b"
EMBED_MODEL = "nomic-embed-text"
COLLECTION_NAME = "local_documents_v1"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
BATCH_SIZE = 8
TOP_K = 4
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_UPLOAD_FILES = 10
MAX_PDF_PAGES = 200
MAX_DOCUMENT_CHUNKS = 2000
MAX_LIBRARY_CHUNKS = 10000
MAX_EXTRACTED_CHARS = 1_600_000
MAX_QUESTION_CHARS = 2000
REFUSAL = "The provided documents do not contain enough information to answer this question."
SYSTEM_PROMPT = '''<|think|>
You are a highly accurate, professional research assistant. Your objective is to answer the user's question using ONLY the provided document context.

Strict Rules:
1. Do not rely on your internal training data or outside knowledge.
2. If the context does not contain the answer, you must respond with: "The provided documents do not contain enough information to answer this question." Do not attempt to guess.
3. Base your entire final response strictly on facts extracted from the <context> block below.
4. Keep your final answer concise, direct, and well-structured.

<context>
{retrieved_context}
</context>'''

LOGGER = logging.getLogger(__name__)
_OPERATION_LOCK = threading.Lock()


class AppError(Exception):
    """An actionable application failure suitable for display."""


class BusyError(AppError):
    """Another browser session owns the application operation lock."""


@dataclass(frozen=True)
class Config:
    persist_dir: Path
    timeout: float

    @classmethod
    def from_env(cls) -> Config:
        try:
            timeout = float(os.getenv("OLLAMA_REQUEST_TIMEOUT", "300"))
            if not math.isfinite(timeout) or timeout <= 0:
                raise ValueError
        except ValueError as exc:
            raise AppError("OLLAMA_REQUEST_TIMEOUT must be a finite positive number of seconds.") from exc
        raw_path = os.getenv("CHROMA_PERSIST_DIR", "/data/chroma")
        if not raw_path.strip():
            raise AppError("CHROMA_PERSIST_DIR must be a writable directory path.")
        return cls(Path(raw_path).expanduser().resolve(), timeout)


@dataclass
class AnswerResult:
    answer: str
    sources: list[Document]


def safe_filename(name: str) -> str:
    """Uploaded names are display metadata, never filesystem destinations."""
    return re.sub(r"[\x00-\x1f\x7f]", "", name.replace("\\", "/").rsplit("/", 1)[-1])[:200] or "document"


def validate_uploads(files: list[tuple[str, bytes]]) -> None:
    if not files:
        raise AppError("Select at least one PDF, TXT, or Markdown file.")
    if len(files) > MAX_UPLOAD_FILES:
        raise AppError("Upload at most 10 files at a time.")
    if sum(len(data) for _, data in files) > MAX_UPLOAD_BYTES:
        raise AppError("The upload set exceeds 25 MB. Upload fewer or smaller files.")
    for name, data in files:
        if not data:
            raise AppError(f"{safe_filename(name)} is empty. Choose a document containing text.")
        if len(data) > MAX_FILE_BYTES:
            raise AppError(f"{safe_filename(name)} exceeds 10 MB. Split the document first.")
        if Path(safe_filename(name)).suffix.lower() not in {".pdf", ".txt", ".md"}:
            raise AppError("Supported file types are PDF, TXT, and Markdown (.md).")


def iter_pages(name: str, data: bytes) -> Iterator[tuple[int, str]]:
    """Yield text page by page; text files use page=0 (no pagination)."""
    if Path(name).suffix.lower() != ".pdf":
        try:
            yield 0, data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise AppError(f"{name} is not UTF-8 text. Save it as UTF-8 and upload again.") from exc
        return
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise AppError(f"{name} is encrypted. Upload an unencrypted copy.")
        if len(reader.pages) > MAX_PDF_PAGES:
            raise AppError(f"{name} exceeds 200 PDF pages. Split it into smaller documents.")
        for number, page in enumerate(reader.pages, start=1):
            yield number, page.extract_text() or ""
    except AppError:
        raise
    except Exception as exc:
        raise AppError(f"Cannot read {name}. Export a valid, unencrypted PDF and try again.") from exc


def iter_chunks(name: str, data: bytes) -> Iterator[Document]:
    validate_uploads([(name, data)])
    name = safe_filename(name)
    file_id = hashlib.sha256(data).hexdigest()
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    count = characters = 0
    for page, text in iter_pages(name, data):
        characters += len(text)
        if characters > MAX_EXTRACTED_CHARS:
            raise AppError(f"{name} exceeds 1,600,000 extracted characters. Split it first.")
        for chunk in splitter.split_text(text):
            if not chunk.strip():
                continue
            count += 1
            if count > MAX_DOCUMENT_CHUNKS:
                raise AppError(f"{name} exceeds {MAX_DOCUMENT_CHUNKS:,} chunks. Split it first.")
            yield Document(page_content=chunk, metadata={"file_id": file_id, "filename": name,
                                                          "page": page, "chunk_index": count})
    if count == 0:
        raise AppError(f"{name} contains no extractable text. For a scanned PDF, run OCR externally first.")


def check_ollama(config: Config) -> None:
    """One bounded discovery call; never pull models or retry automatically."""
    try:
        response = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=min(config.timeout, 10), trust_env=False)
        response.raise_for_status()
        payload = response.json()
        models = {item["name"] for item in payload["models"]}
    except httpx.TimeoutException as exc:
        raise AppError("Ollama discovery timed out. Check the Windows Ollama service and firewall.") from exc
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        raise AppError(f"Cannot reach Ollama at {OLLAMA_URL}. Check OLLAMA_HOST, restart Ollama, and check the firewall.") from exc
    missing = [model for model in (LLM_MODEL, EMBED_MODEL) if model not in models and f"{model}:latest" not in models]
    if missing:
        commands = "; ".join(f"ollama pull {model}" for model in missing)
        raise AppError(f"Required models are missing. Run on Windows: {commands}")


def operation_error(exc: Exception) -> AppError:
    if isinstance(exc, AppError):
        return exc
    if isinstance(exc, httpx.TimeoutException):
        return AppError("Ollama request timed out. Try a shorter document/question or increase OLLAMA_REQUEST_TIMEOUT.")
    if isinstance(exc, (httpx.ConnectError, ConnectionError)):
        return AppError(f"Ollama disconnected. Check the Windows service and {OLLAMA_URL}.")
    return AppError(f"Operation failed ({type(exc).__name__}). Check container logs, available disk/RAM, and Ollama; then retry.")


def source_label(doc: Document, number: int) -> str:
    location = f"page {doc.metadata['page']}" if doc.metadata.get("page") else f"chunk {doc.metadata['chunk_index']}"
    return f"Source {number}: {doc.metadata['filename']}, {location}"


def final_text(message) -> str:
    """Use final content only; never read additional_kwargs reasoning fields."""
    content = message.content
    if isinstance(content, list):
        content = "\n".join(block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text")
    if not isinstance(content, str) or not content.strip():
        raise AppError("The model returned no final answer. Retry or check the Ollama model version.")
    # Defensive handling for runtimes that place tagged reasoning in content.
    if "<think>" in content:
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        if "<think>" in content:
            raise AppError("The model returned incomplete reasoning without a final answer. Retry the question.")
    if "<|channel>" in content:
        if "<|channel>final" not in content:
            raise AppError("The model returned no identifiable final answer. Check the Ollama model version.")
        content = content.split("<|channel>final", 1)[1].lstrip("\n :")
        content = content.replace("<channel|>", "").replace("<|message|>", "")
    if not content.strip():
        raise AppError("The model returned no final answer. Retry the question.")
    return content.strip()


class Library:
    """Single-process persistent library with file-level commit visibility."""

    def __init__(self, config: Config, embeddings: Embeddings | None = None):
        self.config = config
        self.lock = _OPERATION_LOCK
        try:
            config.persist_dir.mkdir(parents=True, exist_ok=True)
            self.manifest_path = config.persist_dir / "manifest.json"
            self._read_manifest()
            self.embeddings = embeddings or OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL,
                num_thread=3, keep_alive=0, client_kwargs={"timeout": config.timeout, "trust_env": False})
            self.store = Chroma(collection_name=COLLECTION_NAME, embedding_function=self.embeddings,
                persist_directory=str(config.persist_dir), client_settings=Settings(anonymized_telemetry=False),
                collection_configuration={"hnsw": {"space": "cosine", "num_threads": 1}})
        except Exception as exc:
            raise operation_error(exc) from exc

    @contextmanager
    def _operation(self):
        if not self.lock.acquire(blocking=False):
            raise BusyError("The library is busy with another request. Try again when it finishes.")
        try:
            yield
        except Exception as exc:
            if not isinstance(exc, AppError):
                LOGGER.exception("Library operation failed")
            raise operation_error(exc) from exc
        finally:
            self.lock.release()

    def _read_manifest(self) -> dict:
        if not self.manifest_path.exists():
            return {"version": 1, "files": {}}
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if manifest.get("version") != 1 or not isinstance(manifest.get("files"), dict):
                raise ValueError
            for key, value in manifest["files"].items():
                if not re.fullmatch(r"[0-9a-f]{64}", key) or not isinstance(value, dict):
                    raise ValueError
                if not isinstance(value.get("filename"), str) or type(value.get("chunks")) is not int or not 1 <= value["chunks"] <= MAX_DOCUMENT_CHUNKS:
                    raise ValueError
            if sum(item["chunks"] for item in manifest["files"].values()) > MAX_LIBRARY_CHUNKS:
                raise ValueError
            return manifest
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            raise AppError("The library manifest is unreadable or incompatible. Restore the data volume from backup; do not edit it while the app runs.") from exc

    def _write_manifest(self, manifest: dict) -> None:
        """fsync a same-directory temporary file, then atomically replace."""
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.config.persist_dir,
                                             prefix="manifest-", suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                json.dump(manifest, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.manifest_path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def stats(self) -> tuple[int, int]:
        files = self._read_manifest()["files"]
        return len(files), sum(item["chunks"] for item in files.values())

    def _recover(self, manifest: dict) -> None:
        """Remove vectors left behind before a file's manifest commit."""
        stored = self.store.get(include=["metadatas"])
        orphan_ids = [key for key, meta in zip(stored["ids"], stored["metadatas"])
                      if not meta or meta.get("file_id") not in manifest["files"]]
        for start in range(0, len(orphan_ids), 256):
            self.store.delete(ids=orphan_ids[start:start + 256])

    def index_files(self, files: list[tuple[str, bytes]], progress: Callable[[str], None] | None = None) -> list[str]:
        report = progress or (lambda _: None)
        with self._operation():
            validate_uploads(files)
            manifest = self._read_manifest()
            self._recover(manifest)
            check_ollama(self.config)
            messages = []
            total = sum(item["chunks"] for item in manifest["files"].values())
            for original_name, data in files:
                name = safe_filename(original_name)
                file_id = hashlib.sha256(data).hexdigest()
                if file_id in manifest["files"]:
                    messages.append(f"Skipped duplicate: {name}")
                    report(messages[-1])
                    continue
                count = 0
                batch = []
                try:
                    for chunk in iter_chunks(name, data):
                        count += 1
                        if total + count > MAX_LIBRARY_CHUNKS:
                            raise AppError("The library would exceed 10,000 chunks. Clear the library or upload a smaller document.")
                        batch.append(chunk)
                        if len(batch) == BATCH_SIZE:
                            self._add_batch(file_id, batch)
                            report(f"Indexing {name}: {count:,} chunks embedded")
                            batch = []
                    if batch:
                        self._add_batch(file_id, batch)
                    updated = {"version": 1, "files": {**manifest["files"], file_id: {"filename": name, "chunks": count}}}
                    self._write_manifest(updated)
                except Exception:
                    # Failed cleanup is safe: the uncommitted hash stays invisible.
                    try:
                        self.store._collection.delete(where={"file_id": file_id})
                    except Exception:
                        LOGGER.exception("Deferred uncommitted-vector cleanup for %s", file_id)
                    raise
                manifest = updated
                total += count
                messages.append(f"Indexed {name}: {count:,} chunks")
                report(messages[-1])
            return messages

    def _add_batch(self, file_id: str, batch: list[Document]) -> None:
        ids = [f"{file_id}:{doc.metadata['chunk_index']}" for doc in batch]
        self.store.add_documents(batch, ids=ids)

    def _retrieve(self, question: str, manifest: dict) -> list[Document]:
        if not manifest["files"]:
            return []
        check_ollama(self.config)
        return self.store.similarity_search(question, k=TOP_K,
                                            filter={"file_id": {"$in": list(manifest["files"])}})

    def retrieve(self, question: str) -> list[Document]:
        question = validate_question(question)
        with self._operation():
            return self._retrieve(question, self._read_manifest())

    def answer(self, question: str) -> AnswerResult:
        question = validate_question(question)
        with self._operation():
            docs = self._retrieve(question, self._read_manifest())
            if not docs:
                return AnswerResult(REFUSAL, [])
            context = "\n\n".join(f"[{source_label(doc, index)}]\n{doc.page_content}" for index, doc in enumerate(docs, 1))
            messages = [SystemMessage(content=SYSTEM_PROMPT.replace("{retrieved_context}", context)),
                        HumanMessage(content=question)]
            model = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_URL, temperature=0, num_ctx=4096,
                num_predict=1024, num_thread=3, keep_alive=0, reasoning=True,
                client_kwargs={"timeout": self.config.timeout, "trust_env": False})
            return AnswerResult(final_text(model.invoke(messages)), docs)

    def clear(self) -> None:
        with self._operation():
            # Invalidate visibility first; a crash during deletion cannot expose old data.
            self._read_manifest()
            self._write_manifest({"version": 1, "files": {}})
            self._recover({"version": 1, "files": {}})


def validate_question(question: str) -> str:
    question = question.strip()
    if not question:
        raise AppError("Enter a question about your documents.")
    if len(question) > MAX_QUESTION_CHARS:
        raise AppError("Questions must be at most 2,000 characters.")
    return question


def display_source_label(doc: Document, number: int) -> str:
    """Plain-language UI label; keep prompt source labels unchanged."""
    location = f"page {doc.metadata['page']}" if doc.metadata.get("page") else f"section {doc.metadata['chunk_index']}"
    return f"Source {number} · {doc.metadata['filename']} · {location}"


def display_progress(message: str) -> str:
    """Translate ingestion messages for display without changing the RAG API."""
    if message.startswith("Skipped duplicate: "):
        return "Already in library: " + message.removeprefix("Skipped duplicate: ")
    match = re.fullmatch(r"(Indexing|Indexed) (.+): ([\d,]+) chunks( embedded)?", message)
    if match:
        action, filename, count, embedded = match.groups()
        unit = "section" if count == "1" else "sections"
        return f"{'Adding' if action == 'Indexing' else 'Added'} {filename}: {count} {unit}{' ready' if embedded else ''}"
    return message


def run_ui() -> None:
    import streamlit as st

    st.set_page_config(page_title="Research", page_icon=":material/menu_book:", layout="centered")
    st.session_state.setdefault("pending_action", None)
    if "restore_question" in st.session_state:
        st.session_state["question"] = st.session_state.pop("restore_question")
    if st.session_state.pop("reset_clear_confirmation", False):
        st.session_state["confirm_clear"] = False

    @st.cache_resource(show_spinner=False)
    def cached_library(config: Config) -> Library:
        return Library(config)

    def queue_action(kind: str) -> None:
        # Callbacks run before the next render; no model/storage work runs here.
        if st.session_state["pending_action"] is not None:
            return
        action = {"kind": kind}
        if kind == "ask":
            action["question"] = st.session_state.get("question", "")
        elif kind == "add":
            uploads = st.session_state.get("uploads", [])
            if len(uploads) > MAX_UPLOAD_FILES or sum(item.size for item in uploads) > MAX_UPLOAD_BYTES:
                st.session_state["feedback"] = ("error", "Choose at most 10 files and 25 MB in total.")
                return
            action["files"] = [(item.name, item.getvalue()) for item in uploads]
        if kind != "check":
            st.session_state.pop("result", None)
        st.session_state.pop("feedback", None)
        st.session_state["pending_action"] = action

    st.title("Research", anchor=False)
    st.caption("Ask questions. Explore your sources.")
    try:
        config = Config.from_env()
        library = cached_library(config)
        files, chunks = library.stats()
    except AppError as exc:
        st.error(str(exc))
        st.stop()

    pending = st.session_state["pending_action"]
    busy = pending is not None
    if busy and pending["kind"] == "ask":
        st.session_state["question"] = pending["question"]
    with st.sidebar:
        st.header("Library", anchor=False)
        st.write(f"{files:,} {'document' if files == 1 else 'documents'}")
        uploads = st.file_uploader("Documents", type=["pdf", "txt", "md"], accept_multiple_files=True,
                                   max_upload_size=10, key="uploads", disabled=busy,
                                   help="PDF, TXT, or Markdown. Up to 10 files, 10 MB each and 25 MB total. PDFs must contain text.")
        st.button("Add to library", type="primary", disabled=busy or not uploads, width="stretch",
                  on_click=queue_action, args=("add",))
        st.divider()
        with st.expander("Details"):
            st.caption(f"{chunks:,} text sections")
            st.caption("Runs on your computer. No API key needed.")
            st.caption(f"Answers: {LLM_MODEL}\n\nSearch: {EMBED_MODEL}")
            st.button("Check connection", disabled=busy, width="stretch", on_click=queue_action, args=("check",))
        with st.expander("Clear library"):
            confirmed = st.checkbox("Delete all documents", key="confirm_clear", disabled=busy)
            st.button("Clear library", disabled=busy or not confirmed, width="stretch",
                      on_click=queue_action, args=("clear",))

    # Allocate the same slots every run. A disappearing notice must not move the
    # form's delta path and leave an old faded form behind during slow inference.
    feedback_slot = st.empty()
    guidance_slot = st.empty()
    with st.container(key="question_panel"):
        with st.form("question_form", border=False):
            st.text_area("Question", key="question", max_chars=MAX_QUESTION_CHARS,
                         placeholder="What would you like to know?", height=110, disabled=busy)
            st.form_submit_button("Ask", type="primary", disabled=busy or not files,
                                  on_click=queue_action, args=("ask",))
    status_slot = st.empty()
    results_slot = st.empty()

    if not files:
        guidance_slot.info("Add documents to your Library to start.")
    if feedback := st.session_state.get("feedback"):
        kind, message = feedback
        getattr(feedback_slot, kind)(message)

    if pending is not None:
        # All controls and placeholders now exist before any slow operation.
        feedback_slot.empty()
        results_slot.empty()
        kind = pending["kind"]
        label = {"ask": "Finding an answer…", "add": "Adding documents…",
                 "check": "Checking connection…", "clear": "Clearing library…"}[kind]
        try:
            with status_slot.container():
                with st.spinner(label):
                    if kind == "ask":
                        st.session_state["result"] = library.answer(pending["question"])
                    elif kind == "add":
                        detail = st.empty()
                        messages = library.index_files(pending["files"],
                            lambda message: detail.caption(display_progress(message)))
                        st.session_state["feedback"] = ("success", "\n\n".join(map(display_progress, messages)))
                    elif kind == "clear":
                        library.clear()
                        st.session_state["reset_clear_confirmation"] = True
                        st.session_state["feedback"] = ("success", "Library cleared.")
                    else:
                        with library._operation():
                            check_ollama(config)
                        st.session_state["feedback"] = ("success", "Connected. Both models are ready.")
        except AppError as exc:
            message = str(exc)
            if kind == "add":
                message += " Any documents already added are ready to use. You can retry the remaining files."
            st.session_state["feedback"] = ("error", message)
        finally:
            if kind == "ask":
                # Disabled widget values can be dropped during an immediate
                # rerun; restore the submitted snapshot before widget creation.
                st.session_state["restore_question"] = pending["question"]
            st.session_state["pending_action"] = None
            status_slot.empty()
        # Render enabled controls and current counts after every outcome.
        st.rerun()

    if result := st.session_state.get("result"):
        with results_slot.container():
            st.subheader("Answer", anchor=False)
            st.write(result.answer)
            if result.sources:
                st.subheader("Sources", anchor=False)
                for number, doc in enumerate(result.sources, 1):
                    with st.expander(display_source_label(doc, number)):
                        st.text(doc.page_content)
                st.caption("Check the sources to verify this answer.")


if __name__ == "__main__":
    # Streamlit reruns __main__; use the canonical module so the lock is shared.
    from app import run_ui as _run_ui
    _run_ui()
