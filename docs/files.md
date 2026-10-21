# Project file index

This is the flat inventory of authored project files. Excluded: `.git/` internals, `.venv/` dependencies, `__pycache__/`, temporary caches/logs, and generated Docker/Chroma data. No generated library files belong in source control.

| File | Purpose |
| --- | --- |
| `.dockerignore` | Allowlist the application, pinned dependencies, and tests into the image build context. |
| `.gitignore` | Exclude local environments, bytecode, secrets, runtime data, caches, and logs. |
| `AGENTS.md` | Enforce locked architecture, exact prompt preservation, verification, and immediate documentation updates. |
| `app.py` | Implement configuration, bounded parsing/indexing, committed-file retrieval, generation, locking, and Streamlit UI. |
| `Dockerfile` | Build a non-root Python 3.12 runtime with resource-related defaults and a Streamlit health check. |
| `docker-compose.yml` | Run the application with host-loopback publishing, a persistent volume, and CPU/RAM limits. |
| `requirements.txt` | Pin the resolved Python dependency set used for verification. |
| `README.md` | Provide the sole user-facing entrypoint for setup, execution, inputs, settings, outputs, and maintenance. |
| `tests/test_app.py` | Exercise parsing, configuration, persistence, recovery, limits, locking, prompt contracts, errors, and UI behavior. |
| `docs/context.md` | Record implementation state, decisions, verification evidence, bug history, limitations, and open validation work. |
| `docs/files.md` | Maintain this flat file inventory and its exclusions. |
| `docs/hld.md` | Describe system boundaries, pipeline, resource controls, persistence, and data flow. |
| `docs/lld.md` | Specify actual application interfaces, constants, metadata, manifest transitions, and error behavior. |
| `docs/solution.md` | Explain operation, recovery, extension patterns, and troubleshooting for maintainers. |
