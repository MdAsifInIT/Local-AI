"""Behavior tests; run with python -m unittest discover -s tests -v."""
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

APP_EXISTS = importlib.util.find_spec("app") is not None
if APP_EXISTS:
    import app
    from langchain_core.embeddings import Embeddings
    from langchain_core.messages import AIMessage
    from pypdf import PdfWriter

    class FakeEmbeddings(Embeddings):
        def embed_documents(self, texts):
            return [self.embed_query(text) for text in texts]

        def embed_query(self, text):
            return [1.0, 0.0, 0.0] if "orchard" in text.lower() else [0.0, 1.0, 0.0]


class BootstrapTests(unittest.TestCase):
    def test_application_module_exists(self):
        self.assertTrue(APP_EXISTS, "The importable application has not been implemented")


@unittest.skipUnless(APP_EXISTS, "Application not implemented yet")
class ParsingTests(unittest.TestCase):
    def test_utf8_metadata_and_hash(self):
        chunks = list(app.iter_chunks("../notes.md", "Café orchard".encode()))
        self.assertEqual(chunks[0].page_content, "Café orchard")
        self.assertEqual(chunks[0].metadata["filename"], "notes.md")
        self.assertEqual(chunks[0].metadata["file_id"], hashlib.sha256("Café orchard".encode()).hexdigest())
        self.assertEqual(chunks[0].metadata["chunk_index"], 1)

    def test_long_text_has_bounded_overlapping_chunks(self):
        text = "".join(str(i % 10) for i in range(1800))
        chunks = list(app.iter_chunks("notes.txt", text.encode()))
        self.assertEqual(len(chunks), 3)
        self.assertTrue(all(len(c.page_content) <= 800 for c in chunks))
        self.assertEqual(chunks[0].page_content[-120:], chunks[1].page_content[:120])

    def test_invalid_documents(self):
        for name, data in [("a.txt", b"\xff"), ("a.txt", b"   "), ("a.exe", b"a"), ("a.pdf", b"bad")]:
            with self.subTest(name=name, data=data), self.assertRaises(app.AppError):
                list(app.iter_chunks(name, data))

    def test_file_and_upload_limits(self):
        with self.assertRaisesRegex(app.AppError, "10 MB"):
            list(app.iter_chunks("a.txt", b"x" * (10 * 1024 * 1024 + 1)))
        with self.assertRaisesRegex(app.AppError, "10 files"):
            app.validate_uploads([("a.txt", b"x")] * 11)
        with self.assertRaisesRegex(app.AppError, "25 MB"):
            app.validate_uploads([("a.txt", b"x" * (9 * 1024 * 1024))] * 3)

    def test_chunk_limit(self):
        with patch.object(app, "MAX_DOCUMENT_CHUNKS", 2):
            with self.assertRaisesRegex(app.AppError, "chunks"):
                list(app.iter_chunks("a.txt", b"x" * 2000))

    def test_pdf_text_page_metadata(self):
        # A tiny PDF text stream exercises the real parser without external fixtures.
        from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
        writer = PdfWriter()
        page = writer.add_blank_page(width=300, height=300)
        font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                                 NameObject("/Subtype"): NameObject("/Type1"),
                                 NameObject("/BaseFont"): NameObject("/Helvetica")})
        page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
        stream = DecodedStreamObject()
        stream.set_data(b"BT /F1 12 Tf 10 20 Td (Orchard harvest is October.) Tj ET")
        page[NameObject("/Contents")] = writer._add_object(stream)
        output = io.BytesIO()
        writer.write(output)
        chunks = list(app.iter_chunks("a.pdf", output.getvalue()))
        self.assertIn("October", chunks[0].page_content)
        self.assertEqual(chunks[0].metadata["page"], 1)

    def test_blank_encrypted_and_oversized_pdf(self):
        for mode in ("blank", "encrypted", "pages"):
            writer = PdfWriter()
            for _ in range(201 if mode == "pages" else 1):
                writer.add_blank_page(width=20, height=20)
            if mode == "encrypted":
                writer.encrypt("secret")
            output = io.BytesIO()
            writer.write(output)
            with self.subTest(mode=mode), self.assertRaises(app.AppError):
                list(app.iter_chunks("a.pdf", output.getvalue()))


@unittest.skipUnless(APP_EXISTS, "Application not implemented yet")
class LibraryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.config = app.Config(Path(self.tmp.name) / "chroma", 3.0)
        self.library = app.Library(self.config, FakeEmbeddings())

    def tearDown(self):
        # Chroma keeps Windows SQLite handles open until its shared system stops.
        from chromadb.api.client import SharedSystemClient
        for system in list(SharedSystemClient._identifier_to_system.values()):
            system.stop()
        SharedSystemClient.clear_system_cache()
        self.tmp.cleanup()

    def index(self, name="notes.txt", data=b"The orchard harvest is October."):
        with patch.object(app, "check_ollama"):
            return self.library.index_files([(name, data)])

    def test_persistence_deduplication_and_retrieval(self):
        self.index()
        self.index("renamed.md")
        self.assertEqual(self.library.stats(), (1, 1))
        reopened = app.Library(self.config, FakeEmbeddings())
        with patch.object(app, "check_ollama"):
            docs = reopened.retrieve("When is orchard harvest?")
        self.assertEqual(docs[0].page_content, "The orchard harvest is October.")
        self.assertEqual(docs[0].metadata["filename"], "notes.txt")

    def test_changed_content_same_name_is_new_document(self):
        self.index()
        self.index(data=b"The orchard harvest is November.")
        self.assertEqual(self.library.stats(), (2, 2))

    def test_streamed_batches_are_no_larger_than_eight(self):
        sizes = []
        original = self.library.store.add_documents
        def record(documents, **kwargs):
            sizes.append(len(documents))
            return original(documents, **kwargs)
        with patch.object(self.library.store, "add_documents", side_effect=record):
            self.index(data=b"x" * 15000)
        self.assertEqual(max(sizes), 8)
        self.assertEqual(sum(sizes), self.library.stats()[1])

    def test_failed_index_cleans_vectors_and_allows_retry(self):
        original = self.library.store.add_documents
        calls = 0
        def fail_second(documents, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("embedding stopped")
            return original(documents, **kwargs)
        with patch.object(self.library.store, "add_documents", side_effect=fail_second), self.assertLogs("app", level="ERROR"):
            with self.assertRaises(app.AppError):
                self.index(data=b"x" * 15000)
        self.assertEqual(self.library.stats(), (0, 0))
        self.assertEqual(len(self.library.store.get()["ids"]), 0)
        self.index(data=b"x" * 15000)
        self.assertEqual(self.library.stats()[0], 1)

    def test_uncommitted_vectors_excluded_and_recovered(self):
        pending = list(app.iter_chunks("pending.txt", b"orchard secret"))
        self.library.store.add_documents(pending, ids=["interrupted"])
        with patch.object(app, "check_ollama", side_effect=AssertionError("network called")):
            self.assertEqual(self.library.answer("orchard").answer, app.REFUSAL)
        self.index()
        self.assertNotIn("interrupted", self.library.store.get()["ids"])
        with patch.object(app, "check_ollama"):
            docs = self.library.retrieve("orchard")
        self.assertEqual(len(docs), 1)
        self.assertNotIn("secret", docs[0].page_content)

    def test_failed_rollback_cleanup_leaves_only_invisible_orphans(self):
        self.index()
        original = self.library.store.add_documents
        calls = 0
        def fail_second(documents, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("embedding failed")
            return original(documents, **kwargs)
        with patch.object(self.library.store, "add_documents", side_effect=fail_second), \
             patch.object(self.library.store._collection, "delete", side_effect=OSError("disk unavailable")), \
             self.assertLogs("app", level="ERROR"):
            with self.assertRaises(app.AppError):
                self.index("pending.txt", b"orchard secret " * 1000)
        self.assertGreater(len(self.library.store.get()["ids"]), 1)
        with patch.object(app, "check_ollama"):
            docs = self.library.retrieve("orchard")
        self.assertEqual(len(docs), 1)
        self.assertNotIn("secret", docs[0].page_content)
        self.index()
        self.assertEqual(len(self.library.store.get()["ids"]), 1)

    def test_atomic_replace_failure_preserves_manifest_and_removes_tempfile(self):
        self.index()
        before = self.library.manifest_path.read_bytes()
        with patch.object(app.os, "replace", side_effect=OSError("replace failed")), self.assertLogs("app", level="ERROR"):
            with self.assertRaises(app.AppError):
                self.index("pending.txt", b"orchard secret")
        self.assertEqual(self.library.manifest_path.read_bytes(), before)
        self.assertEqual(list(self.config.persist_dir.glob("manifest-*.tmp")), [])
        self.assertEqual(len(self.library.store.get()["ids"]), 1)

    def test_manifest_write_failure_rolls_back_and_preserves_previous_files(self):
        self.index()
        with patch.object(self.library, "_write_manifest", side_effect=OSError("disk full")), self.assertLogs("app", level="ERROR"):
            with self.assertRaises(app.AppError):
                self.index("new.txt", b"orchard new")
        self.assertEqual(self.library.stats(), (1, 1))
        self.assertEqual(len(self.library.store.get()["ids"]), 1)

    def test_library_limit_rolls_back_current_file(self):
        self.index()
        with patch.object(app, "MAX_LIBRARY_CHUNKS", 1):
            with self.assertRaisesRegex(app.AppError, "library"):
                self.index("new.txt", b"new content")
        self.assertEqual(self.library.stats(), (1, 1))
        self.assertEqual(len(self.library.store.get()["ids"]), 1)

    def test_clear_persists_empty_library(self):
        self.index()
        self.library.clear()
        self.assertEqual(self.library.stats(), (0, 0))
        self.assertEqual(self.library.store.get()["ids"], [])
        self.assertEqual(app.Library(self.config, FakeEmbeddings()).stats(), (0, 0))

    def test_failed_clear_hides_old_vectors_and_retry_removes_them(self):
        self.index()
        with patch.object(self.library.store, "delete", side_effect=OSError("deletion failed")), self.assertLogs("app", level="ERROR"):
            with self.assertRaises(app.AppError):
                self.library.clear()
        self.assertEqual(len(self.library.store.get()["ids"]), 1)
        reopened = app.Library(self.config, FakeEmbeddings())
        with patch.object(app, "check_ollama", side_effect=AssertionError("network called")):
            self.assertEqual(reopened.answer("orchard").answer, app.REFUSAL)
        reopened.clear()
        self.assertEqual(reopened.store.get()["ids"], [])

    def test_busy_operation_fails_without_waiting(self):
        self.library.lock.acquire()
        try:
            with self.assertRaisesRegex(app.BusyError, "busy"):
                self.library.clear()
        finally:
            self.library.lock.release()

    def test_corrupt_manifest_fails_closed(self):
        self.library.manifest_path.write_text("not json", encoding="utf-8")
        with self.assertRaisesRegex(app.AppError, "manifest"):
            self.library.stats()

    def test_empty_library_refuses_without_ollama(self):
        with patch.object(app, "check_ollama", side_effect=AssertionError("network called")):
            result = self.library.answer("When is harvest?")
        self.assertEqual(result.answer, "The provided documents do not contain enough information to answer this question.")
        self.assertEqual(result.sources, [])

    def test_prompt_roles_options_and_final_answer(self):
        self.index()
        model = Mock()
        model.invoke.return_value = AIMessage(content="October.", additional_kwargs={"reasoning_content": "hidden"})
        with patch.object(app, "check_ollama"), patch.object(app, "ChatOllama", return_value=model) as factory:
            result = self.library.answer("When is orchard harvest?")
        self.assertEqual(result.answer, "October.")
        self.assertEqual(len(result.sources), 1)
        messages = model.invoke.call_args.args[0]
        expected = '''<|think|>
You are a highly accurate, professional research assistant. Your objective is to answer the user's question using ONLY the provided document context.

Strict Rules:
1. Do not rely on your internal training data or outside knowledge.
2. If the context does not contain the answer, you must respond with: "The provided documents do not contain enough information to answer this question." Do not attempt to guess.
3. Base your entire final response strictly on facts extracted from the <context> block below.
4. Keep your final answer concise, direct, and well-structured.

<context>
[Source 1: notes.txt, chunk 1]
The orchard harvest is October.
</context>'''
        self.assertEqual(messages[0].type, "system")
        self.assertEqual(messages[0].content, expected)
        self.assertEqual(messages[1].type, "human")
        self.assertEqual(messages[1].content, "When is orchard harvest?")
        options = factory.call_args.kwargs
        self.assertEqual(options["model"], "gemma4:e4b")
        self.assertEqual(options["base_url"], "http://host.docker.internal:11434")
        self.assertEqual((options["temperature"], options["num_ctx"], options["num_predict"], options["num_thread"]), (0, 4096, 1024, 3))

    def test_invalid_question(self):
        for question in (" ", "x" * 2001):
            with self.subTest(question=question[:10]), self.assertRaises(app.AppError):
                self.library.answer(question)

    def test_ui_clear_confirmation_and_empty_state(self):
        from streamlit.testing.v1 import AppTest
        self.index()
        with patch.dict("os.environ", {"CHROMA_PERSIST_DIR": str(self.config.persist_dir), "OLLAMA_REQUEST_TIMEOUT": "3"}):
            ui = AppTest.from_file(str(Path(app.__file__)), default_timeout=20).run()
            self.assertEqual(len(ui.exception), 0)
            self.assertFalse(next(button for button in ui.button if button.label == "Ask").disabled)
            clear = next(button for button in ui.button if button.label == "Clear library")
            self.assertTrue(clear.disabled)
            ui.checkbox[0].check().run()
            next(button for button in ui.button if button.label == "Clear library").click().run()
            self.assertEqual(len(ui.exception), 0)
            self.assertTrue(next(button for button in ui.button if button.label == "Ask").disabled)
            self.assertEqual(self.library.stats(), (0, 0))

    def test_ui_question_displays_answer_and_evidence(self):
        from streamlit.testing.v1 import AppTest
        self.index()
        model = Mock()
        model.invoke.return_value = AIMessage(content="October.")
        with patch.dict("os.environ", {"CHROMA_PERSIST_DIR": str(self.config.persist_dir), "OLLAMA_REQUEST_TIMEOUT": "3"}), \
             patch.object(app, "check_ollama"), patch.object(app, "ChatOllama", return_value=model), \
             patch.object(app.OllamaEmbeddings, "embed_query", return_value=[1., 0., 0.]):
            ui = AppTest.from_file(str(Path(app.__file__)), default_timeout=20).run()
            ui.text_area[0].input("When is orchard harvest?")
            next(button for button in ui.button if button.label == "Ask").click().run()
            self.assertEqual(len(ui.exception), 0)
            self.assertTrue(any("October." in item.value for item in ui.markdown))
            self.assertTrue(any("notes.txt" in item.label for item in ui.expander))

    def test_ui_uses_simple_names_and_one_question_form(self):
        from streamlit.testing.v1 import AppTest
        with patch.dict("os.environ", {"CHROMA_PERSIST_DIR": str(self.config.persist_dir)}):
            ui = AppTest.from_file(str(Path(app.__file__)), default_timeout=20).run()
            self.assertEqual([item.value for item in ui.title], ["Research"])
            self.assertEqual(ui.sidebar.header[0].value, "Library")
            self.assertEqual(len(ui.text_area), 1)
            self.assertTrue(next(button for button in ui.button if button.label == "Ask").disabled)
            self.assertTrue(any(item.label == "Details" for item in ui.expander))
            self.assertTrue(any(button.label == "Add to library" for button in ui.button))

    def test_ui_repeated_questions_and_failure_restore_controls(self):
        from streamlit.testing.v1 import AppTest
        self.index()
        source = list(app.iter_chunks("notes.txt", b"The orchard harvest is October."))
        replies = [app.AnswerResult("First answer", source), app.AppError("Request timed out. Try again."),
                   app.AnswerResult("Second answer", source)]
        with patch.dict("os.environ", {"CHROMA_PERSIST_DIR": str(self.config.persist_dir)}), \
             patch.object(app.Library, "answer", side_effect=replies) as answer:
            ui = AppTest.from_file(str(Path(app.__file__)), default_timeout=20).run()
            ui.session_state["feedback"] = ("success", "Documents added.")
            for index in range(3):
                ui.text_area[0].input(f"Question {index}")
                next(button for button in ui.button if button.label == "Ask").click().run()
                self.assertEqual(len(ui.exception), 0)
                self.assertEqual(len(ui.text_area), 1)
                self.assertEqual(answer.call_args.args[0], f"Question {index}")
                self.assertEqual(answer.call_count, index + 1)
                self.assertEqual(ui.text_area[0].value, f"Question {index}")
                self.assertFalse(ui.text_area[0].disabled)
                self.assertFalse(next(button for button in ui.button if button.label == "Ask").disabled)
                self.assertIsNone(ui.session_state["pending_action"])
                if index == 1:
                    self.assertTrue(any("timed out" in item.value for item in ui.error))
                    self.assertFalse(any("First answer" in item.value for item in ui.markdown))
            self.assertTrue(any("Second answer" in item.value for item in ui.markdown))
            self.assertTrue(any("section 1" in item.label for item in ui.expander))

    def test_ui_busy_error_and_connection_check_recover(self):
        from streamlit.testing.v1 import AppTest
        self.index()
        with patch.dict("os.environ", {"CHROMA_PERSIST_DIR": str(self.config.persist_dir)}), \
             patch.object(app.Library, "answer", side_effect=app.BusyError("The library is busy. Try again.")):
            ui = AppTest.from_file(str(Path(app.__file__)), default_timeout=20).run()
            ui.text_area[0].input("When is harvest?")
            next(button for button in ui.button if button.label == "Ask").click().run()
            self.assertTrue(any("busy" in item.value for item in ui.error))
            self.assertFalse(ui.text_area[0].disabled)
            with patch.object(app, "check_ollama", side_effect=app.AppError("Connection unavailable.")):
                next(button for button in ui.button if button.label == "Check connection").click().run()
            self.assertTrue(any("Connection unavailable" in item.value for item in ui.error))
            with patch.object(app, "check_ollama"):
                next(button for button in ui.button if button.label == "Check connection").click().run()
            self.assertTrue(any("Connected" in item.value for item in ui.success))
            self.assertEqual(len(ui.exception), 0)

    def test_ui_failed_addition_restores_controls_and_updates_counts(self):
        from streamlit.testing.v1 import AppTest
        with patch.dict("os.environ", {"CHROMA_PERSIST_DIR": str(self.config.persist_dir)}):
            ui = AppTest.from_file(str(Path(app.__file__)), default_timeout=20).run()
            ui.session_state["pending_action"] = {"kind": "add", "files": [("bad.txt", b"\xff")]}
            with patch.object(app, "check_ollama"):
                ui.run()
            self.assertTrue(any("UTF-8" in item.value for item in ui.error))
            self.assertIsNone(ui.session_state["pending_action"])
            self.assertFalse(ui.text_area[0].disabled)
            self.assertEqual(self.library.stats(), (0, 0))

    def test_ui_partial_addition_and_failed_clear_refresh_controls(self):
        from streamlit.testing.v1 import AppTest
        with patch.dict("os.environ", {"CHROMA_PERSIST_DIR": str(self.config.persist_dir)}), \
             patch.object(app, "check_ollama"), \
             patch.object(app.OllamaEmbeddings, "embed_documents", side_effect=FakeEmbeddings().embed_documents):
            ui = AppTest.from_file(str(Path(app.__file__)), default_timeout=20).run()
            ui.session_state["pending_action"] = {"kind": "add", "files": [("good.txt", b"orchard"), ("bad.txt", b"\xff")]}
            ui.run()
            self.assertEqual(len(ui.exception), 0)
            self.assertTrue(any("UTF-8" in item.value for item in ui.error))
            self.assertTrue(any(item.value == "1 document" for item in ui.sidebar.markdown))
            ui.checkbox[0].check().run()
            with patch.object(app.Library, "clear", side_effect=app.AppError("Deletion unavailable.")) as clear:
                next(button for button in ui.button if button.label == "Clear library").click().run()
            self.assertEqual(clear.call_count, 1)
            self.assertTrue(any("Deletion unavailable" in item.value for item in ui.error))
            self.assertFalse(next(button for button in ui.button if button.label == "Ask").disabled)
            self.assertIsNone(ui.session_state["pending_action"])


@unittest.skipUnless(APP_EXISTS, "Application not implemented yet")
class ConfigurationAndNetworkTests(unittest.TestCase):
    def test_progress_translation_preserves_filenames(self):
        self.assertEqual(app.display_progress("Indexed notes.txt: 1 chunks"), "Added notes.txt: 1 section")
        self.assertEqual(app.display_progress("Indexed Indexing chunks.txt: 12 chunks"), "Added Indexing chunks.txt: 12 sections")
        self.assertEqual(app.display_progress("Indexing chunks.md: 8 chunks embedded"), "Adding chunks.md: 8 sections ready")
        self.assertEqual(app.display_progress("Skipped duplicate: Indexed chunks.pdf"), "Already in library: Indexed chunks.pdf")

    def test_reasoning_not_displayed(self):
        for content, expected in [("<think>secret</think>October.", "October."),
                                  ("<|channel>thought\nsecret<|channel>final\nOctober.", "October."),
                                  ([{"type": "reasoning", "reasoning": "secret"}, {"type": "text", "text": "October."}], "October.")]:
            with self.subTest(content=content):
                self.assertEqual(app.final_text(Mock(content=content)), expected)
        for content in ("", "<think>secret", "<|channel>thought\nsecret"):
            with self.subTest(content=content), self.assertRaises(app.AppError):
                app.final_text(Mock(content=content))

    def test_timeout_validation(self):
        for value in ("0", "-1", "abc", "nan", "inf"):
            with self.subTest(value=value), patch.dict("os.environ", {"OLLAMA_REQUEST_TIMEOUT": value}):
                with self.assertRaises(app.AppError):
                    app.Config.from_env()

    def test_model_discovery_and_connection_errors(self):
        config = app.Config(Path("unused"), 3)
        responses = [{"models": [{"name": "gemma4:e4b"}, {"name": "nomic-embed-text:latest"}]},
                     {"models": []}]
        for payload in responses:
            response = Mock()
            response.json.return_value = payload
            with patch.object(app.httpx, "get", return_value=response) as request:
                if payload["models"]:
                    app.check_ollama(config)
                    self.assertEqual(request.call_args.args[0], "http://host.docker.internal:11434/api/tags")
                else:
                    with self.assertRaisesRegex(app.AppError, "ollama pull"):
                        app.check_ollama(config)
        for error in (app.httpx.ConnectError("offline"), app.httpx.ReadTimeout("slow")):
            with patch.object(app.httpx, "get", side_effect=error):
                with self.assertRaises(app.AppError):
                    app.check_ollama(config)


if __name__ == "__main__":
    unittest.main()
