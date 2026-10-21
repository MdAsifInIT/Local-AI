# Low-level design

## Configuration, dependencies, and constants

`app.py` is the only application module. Importing it defines the pipeline without launching Streamlit. The script entrypoint imports `app.run_ui` from the canonical module so Streamlit reruns share that module's `_OPERATION_LOCK`. Tracing is forcibly disabled through `LANGSMITH_TRACING=false` and `LANGCHAIN_TRACING_V2=false`. OMP, OpenBLAS, MKL, NumExpr, and Rayon thread variables are set to `1` before integration imports.

`Config(persist_dir: Path, timeout: float)` is a frozen dataclass. `Config.from_env() -> Config` reads `CHROMA_PERSIST_DIR` (default `/data/chroma`, expanded and resolved) and `OLLAMA_REQUEST_TIMEOUT` (default `300`, finite and positive). An empty directory value is rejected. Direct construction is used by controlled tests; environment validation lives in `from_env`.

| Constant | Value |
| --- | --- |
| `OLLAMA_URL` | `http://host.docker.internal:11434` |
| `LLM_MODEL` / `EMBED_MODEL` | `gemma4:e4b` / `nomic-embed-text` |
| `COLLECTION_NAME` | `local_documents_v1` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `120` characters |
| `BATCH_SIZE` / `TOP_K` | `8` / `4` |
| `MAX_FILE_BYTES` / `MAX_UPLOAD_BYTES` | `10 * 1024 * 1024` / `25 * 1024 * 1024` |
| `MAX_UPLOAD_FILES` / `MAX_PDF_PAGES` | `10` / `200` |
| `MAX_DOCUMENT_CHUNKS` / `MAX_LIBRARY_CHUNKS` | `2000` / `10000` |
| `MAX_EXTRACTED_CHARS` / `MAX_QUESTION_CHARS` | `1_600_000` / `2000` |
| `REFUSAL` | `The provided documents do not contain enough information to answer this question.` |

The requirements file pins the resolved dependency set. Primary integrations are Streamlit, LangChain core/Ollama/Chroma/text splitters, ChromaDB, pypdf, and HTTPX. Ollama embeddings are explicitly supplied, so Chroma does not download or run its default embedding model.

Behavior tests include real Chroma persistence, failures at `os.replace`, failed vector rollback/deletion, direct retrieval while orphan vectors still exist, and Streamlit AppTest question/clear flows. External inference is mocked in the automated suite; live model tests are recorded separately.

## Parsing interfaces

- `safe_filename(name: str) -> str`: remove directory components for either slash convention, remove control characters, truncate to 200 characters, and use `document` if empty. This value is metadata only; uploads are never written using supplied names.
- `validate_uploads(files: list[tuple[str, bytes]]) -> None`: require nonempty supported uploads and enforce file/set counts and byte limits. Fail with `AppError` before processing the set.
- `iter_pages(name: str, data: bytes) -> Iterator[tuple[int, str]]`: decode TXT/MD with `utf-8-sig`, using page `0`; for PDFs use `PdfReader(BytesIO(data))`, reject encryption and more than 200 pages, then extract pages lazily with one-based numbers. PDF parser failures become actionable errors.
- `iter_chunks(name: str, data: bytes) -> Iterator[Document]`: validate, hash bytes, extract pages, bound total extracted characters, and split with `RecursiveCharacterTextSplitter`. Skip empty chunks; reject zero chunks or more than 2,000. Overlap may be less than 120 at natural separators and never crosses PDF pages.
- `validate_question(question: str) -> str`: strip whitespace; require nonempty text and at most 2,000 characters.

Each LangChain `Document` contains `page_content: str` plus scalar metadata:

```json
{"file_id":"<64 lowercase SHA-256 hex characters>","filename":"notes.pdf","page":1,"chunk_index":1}
```

`chunk_index` is one-based across the file. `page=0` means unpaginated text. Deterministic Chroma IDs are `<file_id>:<chunk_index>`. Same bytes under another name retain the original completed entry and its display filename.

## Library and manifest

`Library(config: Config, embeddings: Embeddings | None = None)` creates the data directory, validates the existing manifest, and constructs `Chroma` with a persistent path, explicit embedding client, disabled anonymized telemetry, cosine HNSW distance, and one indexing thread. Without an injected test embedding provider it constructs `OllamaEmbeddings` with the fixed URL/model, three threads, `keep_alive=0`, configured timeout, and `trust_env=False` to bypass proxy environment settings for local requests.

The manifest path is `<persist_dir>/manifest.json`. A missing manifest denotes an empty library. Its shape is:

```json
{
  "version": 1,
  "files": {
    "<sha256>": {"filename": "notes.pdf", "chunks": 3}
  }
}
```

- `_read_manifest() -> dict` validates version, dictionary shape, hash syntax, filename/count types, per-document counts, and total chunk count. An unreadable or incompatible manifest fails closed; it is never silently overwritten.
- `_write_manifest(manifest: dict) -> None` writes UTF-8 JSON to a same-directory temporary file, flushes and `fsync`s it, then uses `os.replace`. Temporary files are removed on normal failures; a hard interruption may leave an ignored `manifest-*.tmp` runtime file. This is atomic replacement, not a distributed transaction or a guarantee against storage hardware failure.
- `stats() -> tuple[int, int]` returns completed document/chunk counts from a manifest snapshot.
- `_operation()` is a context manager acquiring the shared lock without blocking, translating failures, and always releasing the lock. `BusyError` signals contention. Reads of immutable manifest snapshots do not require it.
- `_recover(manifest: dict) -> None` obtains stored IDs/metadata, finds hashes absent from the manifest, and deletes orphan IDs in batches of 256. This scan has bounded normal library size plus interrupted work; it does not fetch document text or embeddings.
- `index_files(files: list[tuple[str, bytes]], progress: Callable[[str], None] | None = None) -> list[str]` locks, validates the upload set, cleans orphans, and checks models. Per file: skip completed hashes, consume chunks, enforce library capacity, upsert batches, then replace the manifest. The callback receives descriptive progress strings; results contain per-file success/duplicate messages. If processing fails, delete that hash's vectors and abort the remaining set. If cleanup itself fails, log it and leave vectors invisible until later recovery. Earlier files remain committed.
- `_add_batch(file_id: str, batch: list[Document]) -> None` derives deterministic IDs and calls `store.add_documents`.
- `clear() -> None` locks, validates the manifest, writes an empty manifest, then uses orphan cleanup to delete all vectors. Deletion failures leave an empty searchable library; retry clear/indexing to finish physical cleanup.

Chroma and manifest writes are not one transaction. Visibility is controlled by the manifest, written only after a complete file's vectors. Retrieval filters against its hashes. A deleted/corrupt manifest cannot be reconstructed automatically; stop and restore a coherent backup. The integration's `_collection.delete(where=...)` is deliberately used for targeted failed-file cleanup and covered by tests; review it on LangChain upgrades.

## Retrieval and generation

- `check_ollama(config: Config) -> None`: GET `<fixed URL>/api/tags` with timeout `min(config.timeout, 10)`, proxy inheritance disabled, and HTTP status/schema checks. Require exact model tags, allowing `nomic-embed-text:latest`. Missing-model errors give host-side `ollama pull` commands; no model download occurs.
- `_retrieve(question: str, manifest: dict) -> list[Document]`: return `[]` immediately when empty; otherwise check models, embed the query, and request up to four cosine-nearest chunks with `file_id $in <completed hashes>`. There is no uncalibrated similarity cutoff or reranker.
- `retrieve(question: str) -> list[Document]`: validate and lock around `_retrieve`.
- `source_label(doc: Document, number: int) -> str`: produce `Source N: filename, page P` for PDFs or `Source N: filename, chunk C` for text.
- `answer(question: str) -> AnswerResult`: validate, lock, retrieve, return the exact refusal if no passages, otherwise concatenate `[Source N: ...]\n<text>` blocks with blank lines. Substitute `{retrieved_context}` in `SYSTEM_PROMPT` once and send a `SystemMessage` plus a separate `HumanMessage`. Use `ChatOllama` with temperature `0`, `num_ctx=4096`, `num_predict=1024`, `num_thread=3`, `keep_alive=0`, `reasoning=True`, fixed URL/model, and timeout/proxy settings. Conversation history is never sent.
- `AnswerResult(answer: str, sources: list[Document])` carries final output and retrieved evidence.
- `final_text(message) -> str`: consume content only, selecting text blocks for structured content. Ignore reasoning metadata, remove complete `<think>...</think>` blocks defensively, and select a tagged `final` channel when present. Reject absent or incomplete final output. It cannot identify arbitrary untagged reasoning misreported as final content.

The exact prompt text is defined by `SYSTEM_PROMPT` in source and independently asserted at the outbound message boundary in tests. It begins with `<|think|>`; wrapper `<system_prompt>` tags are excluded. Context is inserted without executing its braces as a formatting template. The 800-character chunk size is not a token guarantee for all languages; 4,096 context tokens and 1,024 generated tokens remain model/runtime budgets.

## UI, errors, and deployment

`run_ui() -> None` configures a centered Streamlit page, resource-caches `Library(Config)`, and renders a sidebar uploader, indexing status, library counts, model connection check, and confirmed clear action. A separate form submits questions. Answer state is removed before a new query; successful indexing/clearing removes the initiating session's old answer. Other sessions can retain their previous displayed answer until their next operation. Confirmation is reset at the beginning of the next rerun, before creating the checkbox.

`AppError` is displayable; `BusyError` is its contention subtype. `operation_error(exc: Exception) -> AppError` preserves known app failures, distinguishes request timeouts/disconnection, and provides a generic error for storage or unexpected failures. Unexpected errors are logged with stack traces; application code does not intentionally log document text, questions, or answers. Discovery/embedding/generation do not retry automatically. HTTP client timeout bounds network phases, not the complete multi-request operation; parsing and Chroma calls have no wall-clock deadline.

The Docker image uses Python 3.12 slim and UID/GID 10001, owns `/data`, installs pinned packages, and includes tests. An allowlisted build context excludes docs and local data. Compose publishes `127.0.0.1:8501`, mounts `rag_data:/data`, limits CPU/memory, drops capabilities, and prevents privilege escalation. Only the two documented application variables are configurable. The image's HTTP health check tests Streamlit, independently of Ollama. Numerical thread/telemetry settings are deployment defaults, not additional user-facing application options.
