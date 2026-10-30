import io
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from bootstrap_corpus import ensure_corpus


class CorpusTests(unittest.TestCase):
    def test_existing_chunks_are_not_embedded_again(self):
        ai, collection = MagicMock(), MagicMock()
        collection.find.return_value = [{'id': 0}, {'id': 1}]
        reader = SimpleNamespace(pages=[SimpleNamespace(extract_text=lambda: 'x'*1500)])
        with patch('bootstrap_corpus.urllib.request.urlopen', return_value=io.BytesIO(b'%PDF-test')), patch('bootstrap_corpus.PdfReader', return_value=reader):
            result = ensure_corpus(ai, collection, 'gemini-embedding-2')
        self.assertEqual(result['count'], 2)
        ai.models.embed_content.assert_not_called()
        collection.bulk_write.assert_not_called()

    def test_batch_inputs_are_separate_content_objects(self):
        ai, collection = MagicMock(), MagicMock()
        collection.find.return_value = []
        ai.models.embed_content.return_value.embeddings = [SimpleNamespace(values=[0.0]*3072)]*2
        reader = SimpleNamespace(pages=[SimpleNamespace(extract_text=lambda: 'x'*1500)])
        with patch('bootstrap_corpus.urllib.request.urlopen', return_value=io.BytesIO(b'%PDF-test')), patch('bootstrap_corpus.PdfReader', return_value=reader), patch('bootstrap_corpus.time.sleep'):
            result = ensure_corpus(ai, collection, 'gemini-embedding-2')
        sent = ai.models.embed_content.call_args.kwargs['contents']
        self.assertEqual(len(sent), 2)
        self.assertTrue(all(hasattr(c, 'parts') for c in sent))
        self.assertEqual(result['count'], 2)
        self.assertEqual(len(collection.bulk_write.call_args.args[0]), 2)

    def test_invalid_dimensions_are_not_saved(self):
        ai, collection = MagicMock(), MagicMock()
        collection.find.return_value = []
        ai.models.embed_content.return_value.embeddings = [SimpleNamespace(values=[1]), SimpleNamespace(values=[2])]
        reader = SimpleNamespace(pages=[SimpleNamespace(extract_text=lambda: 'x'*1500)])
        with patch('bootstrap_corpus.urllib.request.urlopen', return_value=io.BytesIO(b'%PDF-test')), patch('bootstrap_corpus.PdfReader', return_value=reader):
            with self.assertRaises(ValueError):
                ensure_corpus(ai, collection, 'gemini-embedding-2')
        collection.bulk_write.assert_not_called()


if __name__ == '__main__':
    unittest.main()
