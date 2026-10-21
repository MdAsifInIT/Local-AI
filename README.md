# Research

A lightweight document question-answering app for a Windows PC with four CPU cores and 16 GB RAM. Streamlit, LangChain, and local ChromaDB run in a Docker container. Ollama runs **natively on Windows**, using `gemma4:e4b` for answers and `nomic-embed-text` for embeddings.

Upload text-based PDFs, UTF-8 TXT, or Markdown files, index them, then ask questions and inspect the retrieved passages. The library survives container restarts. No API keys, secrets, cloud account, or external search service are required. Internet access is needed initially to download software, Python packages, the container base image, and models.

## Prerequisites

- Windows 10/11 supported by current Docker Desktop and Ollama, with virtualization enabled.
- Docker Desktop running **Linux containers** (WSL 2 backend recommended).
- Current native Windows [Ollama](https://docs.ollama.com/windows), with sufficient disk space for the models and Docker images. The installed E4B model observed during development occupied about 6.6 GB on disk; runtime memory is additional.
- A current Docker Compose v2 plugin (`docker compose version`). Host Python is optional; the container supplies Python 3.12.

## Quickstart (PowerShell)

Configure the **Windows Ollama process**, not the container:

```powershell
[Environment]::SetEnvironmentVariable('OLLAMA_HOST', '0.0.0.0:11434', 'User')
[Environment]::SetEnvironmentVariable('OLLAMA_NUM_PARALLEL', '1', 'User')
[Environment]::SetEnvironmentVariable('OLLAMA_MAX_LOADED_MODELS', '1', 'User')
```

Quit Ollama from its Windows tray icon and relaunch it from the Start menu so it inherits the new settings. If you run `ollama serve` manually, start it from a newly opened terminal; do not run a second server alongside the tray app. Then:

```powershell
ollama pull gemma4:e4b
ollama pull nomic-embed-text
ollama list
docker compose up --build -d
docker compose ps
```

Open [Research](http://localhost:8501). Add documents in the sidebar, select **Add to library**, and wait for completion. Enter a question and select **Ask**. Expand **Sources** to check the evidence.

Every application call to Ollama uses **`http://host.docker.internal:11434`**. Docker Desktop supplies this host bridge; `localhost` inside the container would refer to the container itself. [Docker Desktop networking](https://docs.docker.com/desktop/features/networking/)

Binding Ollama to `0.0.0.0` can make its unauthenticated API reachable beyond the host. Restrict Windows Firewall access to the Docker Desktop/WSL connection as appropriate for your installation; do not expose port 11434 to untrusted networks. Only the Streamlit port is published by Compose, bound to host loopback.

## Inputs and limits

| Input | Limit |
| --- | --- |
| Formats | `.pdf`, `.txt`, `.md` (extensions are case-insensitive) |
| File size | 10 MiB per file; UI calls this 10 MB |
| Upload set | 10 files, 25 MiB total |
| PDF | 200 pages, unencrypted, extractable text |
| Extracted text | 1,600,000 characters per document |
| Chunks | 2,000 per document; 10,000 across the library |
| Question | 2,000 characters after trimming whitespace |

Scanned PDFs require external OCR first. Text files must be UTF-8 (a UTF-8 BOM is accepted). Duplicate bytes are skipped even under another filename; changed content under the same filename is a new document. Chunking uses up to 800 characters with up to 120 characters of overlap, separately per PDF page. Embeddings are processed sequentially in batches of eight.

If a later file fails, earlier completed files remain indexed; retrying the upload skips those duplicates. An unfinished file is excluded from search and cleaned up before the next indexing operation. **Clear library** requires a checkbox confirmation and deletes all indexed content, including pending vectors; individual deletion is outside v1.

## Configuration and flags

The application exposes two environment settings. Compose supplies their defaults; neither is a secret.

| Variable | Default | Purpose |
| --- | --- | --- |
| `CHROMA_PERSIST_DIR` | `/data/chroma` | Chroma storage and `manifest.json`; keep the container path under `/data` to retain persistence and write permissions. |
| `OLLAMA_REQUEST_TIMEOUT` | `300` | Finite positive timeout in seconds per HTTP request/network phase; discovery is capped at 10 seconds. It is not a total job deadline. |

For example, set `$env:OLLAMA_REQUEST_TIMEOUT = '600'` before `docker compose up -d`. Compose reads exported values (and its standard optional `.env` file); the Python app itself does not load `.env`. No `.env` file is needed. Models and the Ollama URL are fixed in code and cannot be overridden through environment variables.

There are no custom application CLI flags. Relevant standard commands and flags:

```powershell
docker compose up --build -d       # Build, then run in the background
docker compose logs --tail 100 rag # Inspect application errors
docker compose restart rag        # Restart; keep the library
docker compose down               # Stop and remove containers; keep the volume
```

The image starts `streamlit run app.py` with `--server.address=0.0.0.0`, `--server.port=8501`, `--server.headless=true`, and `--server.maxUploadSize=10`. Compose limits the app to two CPUs and 3 GB RAM; those limits do not constrain native Ollama. Numerical libraries and Chroma indexing use one thread; Ollama requests use three CPU threads. Both model clients request unload after use (`keep_alive=0`) to reduce memory pressure, at the cost of load latency.

## Outputs, persistence, and health

Research uses a compact slate-accented theme with light/dark support. **Library** shows document counts; **Details** contains section counts, model information, and **Check connection**. Controls are disabled while work runs and restored after completion or an error, with the question preserved. The bundled `.streamlit/config.toml` and fixed UI containers keep the layout consistent during slow requests.

The UI shows an answer and up to four retrieved passages with file/page or chunk labels. These passages are evidence for inspection, not guaranteed sentence-level citations. Separate model reasoning metadata is not displayed. Questions are independent; previous answers are not added to prompts.

Chroma stores extracted chunks, embeddings, and metadata in the Docker named volume `rag_data` (normally prefixed with the Compose project name). `manifest.json` records completed file hashes, original display names, and chunk counts. Original upload files are not archived by the app; Streamlit holds upload bytes in session memory. The most recent answer exists only in browser-session state and is replaced by another query.

The image health check tests Streamlit at `/_stcore/health`. A healthy UI does **not** prove Ollama is reachable; use **Details → Check connection** in the sidebar. Discovery also runs before indexing or querying a nonempty library. No automatic model pulls or retries occur.

The exact supplied grounding prompt is preserved. An empty library deterministically returns the specified refusal through the application API. For nonempty retrieval, insufficient-evidence refusal is model-dependent: prompts do not guarantee factuality or resistance to instructions embedded in documents. Inspect the passages and use trusted documents. CPU inference and reasoning may be slow; memory/latency on the target four-core machine still need measurement.

## Tests and maintenance

Run checks inside the image when Docker is available:

```powershell
docker compose run --rm --no-deps rag python -m unittest discover -s tests -v
docker compose run --rm --no-deps rag python -m pip check
```

For development with host Python 3.12:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests use temporary Chroma storage and fake embeddings; no models are required. The production deployment remains Docker-based. Tests run against a stopped application are recommended because the test cleanup stops shared Chroma clients in that Python process.

Validation passed all 35 automated tests on both Windows and inside the Linux image, dependency/syntax/import checks, and Docker Compose validation/build. The running container is healthy, uses the real host bridge, and completed a RAG smoke question through native Ollama. Two earlier native smoke questions also produced the expected factual answer and exact refusal. Resource use on a representative four-core/16 GB workload still needs measurement; see [verification evidence](docs/context.md#verification-evidence) and [deployment state](docs/context.md#github-synchronization-and-docker-deployment).

`docker compose down --volumes` **permanently deletes this project's library volume**. Use ordinary `down` when retaining data. Stop the container before backing up the complete `/data` volume; restore Chroma and the manifest together.

Supporting engineering documentation: [project state](docs/context.md), [high-level design](docs/hld.md), [low-level design](docs/lld.md), [file index](docs/files.md), and [operation/extension guide](docs/solution.md). Future changes must follow [AGENTS.md](AGENTS.md).
