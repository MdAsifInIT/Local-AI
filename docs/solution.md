# Operation and extension guide

## End-to-end behavior

Use the [README](../README.md) for the canonical installation commands. Native Windows Ollama owns both models; the Python container calls its fixed Docker Desktop bridge URL. Do not move Ollama into Compose or replace the bridge with container `localhost`.

An upload does nothing until **Index documents** is selected. The app validates the set, removes stale uncommitted vectors, checks the model inventory, then processes files serially. It hashes raw bytes so identical uploads do not repeat extraction or embedding once committed. Text is split per page, and only eight chunks at a time are passed to the embedding service and Chroma. Progress reports completed embedding batches and file commits; it is not an estimated percentage or ETA.

Each successful file becomes visible when its manifest entry is atomically written. If the last file fails, the preceding files remain. Refresh to see updated counts, then retry the same set: completed files are skipped, and the failed file is rebuilt. A crash after vector insertion but before manifest commit does not expose partial content. The next indexing run removes those orphan vectors before proceeding.

Questions are independent. The retriever embeds the question and selects up to four nearest completed chunks. The exact supplied system prompt receives only these labeled passages; the question occupies its own user message. The final model content and retrieved evidence are displayed separately. Similarity retrieval can select irrelevant text, and model compliance is not guaranteed. For an unanswerable question, inspect whether the expected refusal appears; do not equate a fluent answer with evidence.

## Usage patterns

- Keep one small, coherent research library. Upload batches below the limits rather than a large archive. No ZIP, directory crawl, web fetch, or background ingestion is included.
- Ask narrow factual questions that fit in a few paragraphs of evidence. The app does not summarize an entire large document merely because it is indexed.
- PDF page labels refer to physical PDF pages, which can differ from printed page numbers. TXT/Markdown sources use file-wide chunk numbers.
- Corrected bytes under the same filename create a second document. To replace an obsolete corpus, clear the library with confirmation and upload the desired set. There is no per-file deletion in v1.
- The latest displayed answer belongs to its browser session. A different session can change the shared library while that old answer remains visible; refresh or submit a new question for current evidence. Avoid simultaneous use beyond this local shared-library model.

## Recovery and persistence

Normal container restart and Compose `down` retain the volume. The volume contains extracted document content and embeddings, not just hashes. Original files are not archived; retain your source documents separately. Treat a volume backup as sensitive as those sources.

For a coherent backup, stop the app and copy/archive the **entire** `/data` volume, including Chroma persistence and the manifest. Restore them together into a writable volume owned by UID/GID 10001. Do not copy a manifest from a different vector store or manually edit it while running.

An unreadable/incompatible manifest fails closed instead of assuming all vectors are committed. Restore a known-good backup. If the library is disposable, remove the project's volume using the documented destructive teardown and rebuild from sources. A clear-library failure after manifest replacement can leave invisible vectors on disk; retry clear or begin another indexing run to finish cleanup. Removing a manifest alone is not a supported recovery strategy.

## Troubleshooting

| Symptom | Cause and next step |
| --- | --- |
| UI does not open | Start Docker Desktop in Linux-container mode; inspect `docker compose ps` and `docker compose logs --tail 100 rag`. Check host port 8501 is free. |
| Docker runs but the CLI is not found | For a per-user installation, check `%LOCALAPPDATA%\Programs\DockerDesktop\resources\bin\docker.exe`; use the full path or add that directory to the current terminal's PATH. |
| UI healthy, Ollama check fails | Health only tests Streamlit. Confirm native Ollama is running, restart after changing `OLLAMA_HOST`, and check Windows Firewall/Docker Desktop DNS. |
| Cannot connect through the host bridge | Run an HTTP request to `http://host.docker.internal:11434/api/tags` from the container, not just Windows `localhost`; see the command below. |
| Model missing | Pull the exact tag on Windows. The app will not pull it automatically or substitute another model. |
| Request timed out | CPU inference/model loading can be slow. Try a short question/small document, inspect native Ollama logs, then increase the documented timeout if necessary. A timeout is per network phase, not a total indexing deadline. |
| No final answer returned | Reasoning may consume the output budget, or the runtime may return an unsupported response shape. Try a shorter question and inspect the Ollama version. Separate reasoning is intentionally not shown. |
| No extractable text | Use a text-based PDF; perform OCR externally for scans. Empty pages are skipped. |
| UTF-8 error | Resave TXT/Markdown as UTF-8, optionally with BOM. |
| File/page/chunk limits exceeded | Split the source or reduce the upload set. Earlier completed files may remain indexed. |
| Library busy | Another operation is running. Wait for it to finish and retry; requests are not queued by the app. |
| Wrong answer or unsupported facts | Inspect retrieved passages; rephrase the question or revise the corpus. Prompting and nearest-neighbor retrieval do not prove factuality. |
| Duplicate retains an old filename | Deduplication uses file bytes; the first committed display name wins. |
| Corrupt manifest or permission error | Stop the app, inspect disk space/ownership, and restore a coherent backup. Keep `CHROMA_PERSIST_DIR` under `/data`. |
| Container exits with OOM / host stalls | Check Docker and native Ollama memory separately. The container limit cannot cap Ollama or guarantee bounded PDF extraction allocations. Use smaller trusted files, close other workloads, and measure before changing limits. |

Container-to-host diagnostic:

```powershell
docker compose exec rag python -c "import urllib.request; print(urllib.request.urlopen('http://host.docker.internal:11434/api/tags', timeout=10).status)"
```

On Windows, inspect native Ollama logs under `%LOCALAPPDATA%\Ollama` where available. Do not paste sensitive document content or entire model requests into shared logs. The app bypasses proxy environment variables for Ollama requests; Docker Desktop/host connectivity must work directly.

## Resource tuning and extensions

Start with the defaults. One native loaded model, serial processing, short chunks, three model threads, one HNSW thread, and unload-after-use reduce contention. They trade throughput and warm-model latency for lower memory pressure. Keep Windows and Docker Desktop overhead in the RAM budget; `mem_limit: 3g` applies only to the container. A 10 MB PDF can decompress to much more memory before extracted-text limits are checked; only ingest trusted files. Dedicated parser isolation would be a separate future change.

To add a file format, extend validation and page extraction together, preserve scalar metadata and generator boundaries, add malformed/oversized fixtures, update dependencies, and synchronize all affected docs. OCR, DOCX, authentication, multi-process serving, individual document deletion, and external search are not included.

Changing chunking or the embedding model requires rebuilding the index; changing embedding dimensionality is incompatible with existing vectors. Treat the collection name and manifest version as compatibility boundaries. Preserve the locked model names and prompt unless the user explicitly changes the requirements. Do not append instructions to the supplied system prompt as an incidental improvement.

For retrieval improvements, first measure known questions against representative documents. Introduce a similarity threshold only after calibration; no universal confidence threshold is assumed. If increasing generation/context budgets to handle reasoning-heavy questions, measure native memory and latency on the target machine and update constants, tests, and docs together.

## Acceptance checks on the target machine

Run unit tests and dependency checks first, then Compose validation/build. Start a clean library, index a tiny text document with a known fact, and ask one answerable and one unanswerable question. Confirm evidence labels, absence of separate reasoning, duplicate skipping, changed-file behavior, restart persistence, and confirmed clearing. Test an invalid/scanned PDF, missing model, disconnected Ollama, and a competing browser session. Watch CPU/RAM and perceived desktop responsiveness during a representative PDF import and a model response. Record results in `context.md`; do not infer four-core performance from tests on another machine.
