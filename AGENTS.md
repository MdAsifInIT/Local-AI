# Agent instructions

## Mandatory documentation synchronization

Every code, dependency, configuration, deployment, test, or file-layout change MUST immediately update the corresponding documentation as part of the same change. Documentation is part of the implementation; do not defer it to a later task, commit, or pull request.

- Update `README.md` whenever user behavior, setup, commands/flags, inputs, settings/secrets, outputs, or prerequisites change.
- Update `docs/context.md` for decisions, current state, actual verification results, known issues, bug history, and open questions.
- Update `docs/hld.md` for architecture, boundaries, pipeline, persistence, or data-flow changes.
- Update `docs/lld.md` for function signatures, interfaces, constants, dependencies, metadata, state transitions, or error behavior.
- Update `docs/files.md` immediately when any authored file is added, renamed, moved, or removed. Keep a flat one-line index of every authored file, including itself and tests. Exclude `.git`, installed dependencies, caches, and generated runtime data explicitly.
- Update `docs/solution.md` for workflow, extension, recovery, tuning, or troubleshooting changes.
- Review all six documents for consistency before declaring work complete. Record which checks actually ran and which remain unavailable. Never claim an image was built or a live model was tested without observed results.

## Documentation structure

`README.md` is the single user-facing entry document. Its spelling and case are fixed. All supporting documentation MUST remain inside `docs/`; this root `AGENTS.md` is the sole additional documentation exception. Do not create README duplicates, case variants, PRODUCT.md, DESIGN.md, standalone root plans, or redundant setup guides. Put product/design decisions and execution state in the existing target documents. Add further documents only when the user requests them or explicitly accepts a demonstrated need.

Source, tests, dependency manifests, and deployment configuration remain in their conventional project locations; the documentation-directory rule applies to documentation, not executable files.

## Locked architecture and prompt

- Windows host, four CPU cores, 16 GB RAM. Native Windows Ollama; never add an Ollama container.
- All production Ollama calls MUST use `http://host.docker.internal:11434`.
- LLM `gemma4:e4b`; embeddings `nomic-embed-text`.
- Python 3.12, Streamlit, LangChain, persistent local ChromaDB; Python deployment in Docker.
- Preserve `SYSTEM_PROMPT` exactly, including the opening `<|think|>` and all whitespace/wording. Substitute only `{retrieved_context}`. The user's `<system_prompt>` tags are delimiters, not transmitted text.
- Keep the question in a separate user message. Do not add previous turns, outside context, or separate reasoning output to the UI.
- Preserve bounded uploads/chunking, serial model work, thread limits, and completed-file-only retrieval. Retune limits only with explicit documentation and meaningful tests.

## Implementation and verification

Keep `app.py` importable without launching Streamlit. The script entrypoint imports the canonical `app` module to preserve one process-wide operation lock across Streamlit reruns and sessions. Never run multiple application processes against this data directory: the lock does not coordinate between processes.

Write behavior tests for meaningful changes, including failure/recovery cases. Run `python -m unittest discover -s tests -v`, `python -m pip check`, syntax/import checks, and relevant container checks. Use temporary local Chroma stores and deterministic fake embeddings in automated tests; live model checks must be reported separately. Requirements are pinned; re-resolve and test Python 3.12/Linux compatibility when updating them.

Do not log uploaded contents, questions, answers, or secrets. Do not silently enable tracing, external search, automatic model downloads, or automatic unbounded retries. Report material limitations candidly, especially untested Docker behavior, model grounding, malicious/oversized PDF parsing, memory use, and CPU latency.
