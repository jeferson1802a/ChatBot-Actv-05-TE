"""Pruebas de interfaz sin credenciales ni llamadas externas."""
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import MagicMock, patch

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).with_name("app.py"))


class AppTests(TestCase):
    def test_missing_configuration(self):
        app = AppTest.from_file(APP).run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(app.info)

    def test_k_cache_and_history(self):
        ai, mongo = MagicMock(), MagicMock()
        ai.models.embed_content.return_value.embeddings = [SimpleNamespace(values=[0.0] * 3072)]
        ai.models.generate_content_stream.side_effect = lambda **kw: iter([SimpleNamespace(text="Respuesta "), SimpleNamespace(text="respaldada [Fuente 1].")])
        collection = mongo.__getitem__.return_value.__getitem__.return_value
        collection.aggregate.return_value = [{"id": 12, "texto": "Evidencia completa", "score": 0.9}]
        with patch("google.genai.Client", return_value=ai), patch("pymongo.MongoClient", return_value=mongo):
            app = AppTest.from_file(APP)
            app.secrets = {"app": {"GOOGLE_API_KEY": "test", "MONGODB_URI": "mongodb://test"}}
            app.run()
            for k in (3, 10):
                app.selectbox[0].select(k).run()
                app.chat_input[0].set_value("¿Qué es ACAP?").run()
                self.assertEqual(len(app.exception), 0)
                pipeline = collection.aggregate.call_args.args[0]
                self.assertEqual(pipeline[0]["$vectorSearch"]["limit"], k)
                self.assertEqual(app.session_state.messages[-1]["chunks"][0]["id"], 12)
                self.assertIn("respaldada", app.session_state.messages[-1]["text"])
            self.assertEqual(ai.models.embed_content.call_count, 1)
            self.assertEqual(len(app.session_state.messages), 4)


if __name__ == "__main__":
    main()
