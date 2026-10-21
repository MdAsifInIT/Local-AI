# Project memory and state

## Architectural decisions

- Created from an empty repository on 2026-10-21 (Asia/Calcutta). Implementation follows the approved Lightweight Windows RAG plan.
- Native Windows Ollama is mandatory; the Python app alone is containerized. The production URL is fixed to `http://host.docker.internal:11434`.
- Use `gemma4:e4b`, `nomic-embed-text`, Python 3.12, Streamlit, focused LangChain packages, and embedded persistent ChromaDB.
- PDF, UTF-8 TXT, and Markdown support and one persistent shared library were explicitly selected. OCR and multi-user isolation are excluded.
- The supplied `<system_prompt>` tags are delimiters; the actual immutable system message starts with `<|think|>`. Only retrieved context is substituted.
- The documentation-directory rule applies to documentation. The six requested documents plus root `AGENTS.md` are sufficient; no additional design or planning documents are introduced.
- Sequential eight-chunk embedding batches, bounded uploads/extracted text, one process lock, conservative model/context budgets, and unload-after-use favor host responsiveness over throughput.
- A versioned JSON manifest controls completed-file visibility independently of Chroma writes. Failed and interrupted files remain invisible until successfully reindexed. Commit granularity is a file, not a whole upload set.
- Canonical module import at the Streamlit script entrypoint preserves the shared lock across reruns. Multiple processes/replicas sharing the directory are unsupported.

## Current implementation

All application, deployment, test, and requested documentation files have been populated. The UI supports explicit indexing, independent questions, retrieved passages, model checks, and confirmed clearing. The library persists in `/data/chroma` on a named Docker volume. Settings are limited to persistence path and request timeout. No secrets or custom CLI are required.

The implementation uses an exact prompt plus separate user message, cosine retrieval over completed hashes, deterministic chunk IDs, atomic manifest replacement, and bounded batch upserts. It excludes separate reasoning metadata from display. The README is the canonical user entrypoint; low-level signatures and behavior are maintained in `lld.md`.

## Verification evidence

Development uses a project-local virtual environment with bundled Python 3.12.14. The dependency set was resolved and pinned from an initially empty environment. The following results were observed on 2026-10-21. Initial host-only checks were followed by a successful real Docker deployment; representative target-machine performance remains unmeasured.

- Initial test-first run failed because the importable application did not yet exist.
- Initial implementation run: 23 tests passed with real local Chroma, deterministic fake embeddings, and mocked Ollama boundaries.
- A subsequent complete run passed 26 tests, including Streamlit AppTest clear confirmation and answer/evidence rendering; `pip check` reported no dependency conflicts.
- An independent read-only code review found no confirmed production defect and requested direct tests of failed clear, orphan visibility before cleanup, failed rollback cleanup, and atomic replacement failure. Those regression scenarios were added and passed in the final expanded suite: **29 tests, zero failures, no skipped tests** (8.478 seconds in the recorded final run). Expected pypdf diagnostics for malformed fixtures and Streamlit AppTest bare-mode warnings do not indicate test failures.
- `python -m pip check`: no broken requirements. `python -m compileall -q app.py tests`: passed. Plain `import app`: passed without starting Streamlit.
- Linux x86-64 Python 3.12 binary-wheel resolution for every pin in `requirements.txt`: passed using pip dry-run with manylinux target platforms. This verifies package resolution, not Linux runtime execution or a Docker image build.
- Compose YAML parsed and expected service/loopback-port structure checked with PyYAML. The later deployment pass also passed Docker-native `docker compose config --quiet`.
- A temporary native Streamlit server bound only to `127.0.0.1:18501` returned HTTP `200 ok` at `/_stcore/health`, then was shut down automatically. This does not validate the image health check inside Docker.
- A file inventory audit matched all **14 authored files** to `docs/files.md`, with no missing entries or documentation placeholders. The interface detector reported no findings. Application functionality was exercised through Streamlit AppTest rather than screenshot-based browser inspection.
- Native Ollama was discovered outside PATH at its Windows installation path. Read-only inspection reports version `0.40.5`; `gemma4:e4b` and `nomic-embed-text:latest` are installed. No models were downloaded or replaced by this task.
- Initial implementation checks did not find Docker on PATH or at the usual system-wide installation path. A later deployment pass found the running per-user installation under `%LOCALAPPDATA%\Programs\DockerDesktop`; the initial absence was a discovery limitation, not proof that Docker was uninstalled.
- Live native-model smoke test: indexed a temporary `orchard.txt` containing two known facts using real Ollama embeddings and Chroma. `When is the orchard harvest?` returned `The orchard harvest is October.` with one source in 36.33 seconds. `What is the launch mass of Voyager 1?` returned the exact prescribed refusal with one retrieved source in 35.27 seconds. Temporary vectors were cleared and deleted afterward.
- That smoke test mapped `host.docker.internal` to loopback **only within the test process's DNS function**; the host cannot resolve the Docker hostname in this environment. Production code, URL, DNS configuration, Ollama settings, and source files were not changed. These timings include cold-model behavior and are not a four-core performance certification or proof of general grounding.

## GitHub synchronization and Docker deployment

The user requested synchronization with GitHub `main` and local Docker deployment. The configured remote is `https://github.com/MdAsifInIT/Local-AI.git`; authenticated inspection found it empty, with no existing `main` history to merge. The complete project is prepared on local `main` for initial publication, preserving all application and documentation files while excluding the virtual environment and runtime data. No remote history is rewritten.

The running Docker installation reports Desktop `4.94.0`, Engine `29.8.2`, Linux/amd64 context `desktop-linux`, and Compose `v5.5.1`. `docker compose config --quiet` passed. The host suite was rerun successfully: 29 tests in 7.345 seconds, dependency checks clean, syntax compilation passed.

A disposable Python container successfully fetched `http://host.docker.internal:11434/api/tags` with HTTP 200 using the real Docker bridge and no DNS substitutions. Existing native Ollama remains bound to Windows loopback and is reachable through this Docker Desktop connection; no global Ollama settings or firewall rules were changed.

Deployment verification:

- `docker compose up --build -d` succeeded, producing local image `localai-rag:latest` and starting `localai-rag-1` with named volume `localai_rag_data`.
- Container health is `healthy`; the host HTTP endpoint `http://localhost:8501/_stcore/health` returns `200 ok`. Browser inspection confirmed the rendered Local Research page, uploader, question form, and empty-library guidance. The browser tab was retained as a user-facing result.
- Runtime inspection confirmed non-root UID `10001`, 3,221,225,472-byte memory limit, two CPUs, and host-loopback port publishing. The application successfully opened its writable persistent Chroma path and reported zero documents/chunks.
- The full test suite inside the Linux container passed **29 tests** in 8.335 seconds. Container `pip check` reported no broken requirements. Linux-only transitive package `uvloop` resolved to `0.23.0` in this build.
- A complete container RAG smoke test indexed synthetic `smoke.txt` in temporary storage under `/data`, retrieved one source, and returned `The orchard harvest is October.` through native `gemma4:e4b` in 41.32 seconds for the query. No hostname substitution was used. Temporary data was removed; the main library stayed empty.
- Only local Docker deployment is configured. No registry/image publication target was supplied; the local image name is not evidence of a Docker Hub upload.

## Bug history

- UI refinement: renamed the page/tab to Research, sidebar to Library, and actions to Add to library / Ask / Sources. Added a muted slate theme, compact headings, minimal toolbar, and disabled usage statistics in `.streamlit/config.toml`, bundled by Docker.
- The reported faded duplicate form was traced to conditional content shifting the form's delta position while long inference delayed stale-node cleanup. The revised UI queues callbacks, renders disabled controls first, keeps feedback/form/status/results slots stable, and explicitly replaces old output before slow work. In an isolated browser preview with synthetic uploads and 25-second fake answers, inspection confirmed exactly one form, one question field, and one spinner after upload feedback disappeared. A second delayed request cleared the old answer; the simulated timeout restored controls and preserved the question. Desktop light/dark and narrow mobile rendering, including a long filename and expanded source, were visually inspected without changing the real library.
- A regression test caught submitted question text disappearing after disabled widgets and an immediate completion rerun. A separate submitted snapshot now restores text before widget creation. Another test caught filename corruption from unrestricted wording replacements; translations now match message prefixes/count suffixes, preserving filenames. Independent read-only review confirmed the RAG/prompt/storage/lock behavior remains unchanged; its documentation corrections were applied.
- The expanded suite passed 35 tests on Windows and inside the Linux container. Container dependency checks passed. The rebuilt service was healthy and reached native Ollama. The library remained at 5 documents / 69 sections with the same manifest SHA-256 before and after deployment. The final copy/telemetry configuration passed a fresh 35-test run on Windows (11.548 seconds) and in Docker (11.641 seconds), with clean dependency checks. Final runtime inspection confirmed the slate theme, disabled usage statistics, healthy service, native Ollama connectivity, and unchanged library counts/checksum. The production page was refreshed and visibly showed Research, Library, and 5 documents. Temporary preview servers/tabs and synthetic fixture files were removed; the 15-file documentation index and exact system-prompt preservation checks passed.

- Fixed during initial development: resetting `confirm_clear` after creating the checkbox raised `StreamlitWidgetAlreadyInstantiatedError`. A failing UI regression test reproduced the issue. Reset is now deferred to the next rerun before widget creation.
- Test harness correction: Streamlit AppTest resolves relative script paths from the calling test directory, not repository cwd. UI tests now use the absolute application path. This was a test setup issue, not a production parser error.

## Known limitations and open validation

- During GitHub synchronization, one Windows test-suite rerun encountered an intermittent `WinError 32` while cleaning a temporary Chroma SQLite file after `test_invalid_question`; the application assertion itself passed. An isolated rerun of that test passed. This is a Windows test-teardown reliability issue; no production storage behavior was changed for synchronization.

- Target four-core/16 GB latency, peak RAM, and desktop responsiveness need measurement; container and native model budgets are distinct.
- PDF extraction can allocate heavily before character/chunk limits run. Trusted files only; parser subprocess isolation and hard extraction time limits are not included.
- The exact grounding prompt cannot guarantee factuality, refusal on insufficient evidence, or protection from document prompt injection. Retrieved passages need inspection.
- Reasoning can consume the 1,024-token generation budget before final content appears; the UI reports missing final output instead of displaying separate thoughts.
- Character chunk limits do not guarantee a token budget for all languages. The 4,096-token context is intentionally conservative.
- A valid manifest and its matching vector store must be restored together. Multiple processes, replicas, and network filesystems are unsupported. Storage corruption/hardware durability is outside atomic replacement guarantees.
- Uploads are transient in Streamlit session memory; extracted text/metadata/embeddings persist. Session answers can outlive library changes made by another session.
- Docker build, Linux tests, real container-to-host networking, non-root volume access, browser rendering, and one complete container RAG question are confirmed above. Representative workload performance and recovery across an actual populated-volume container restart still need target-host acceptance testing.
