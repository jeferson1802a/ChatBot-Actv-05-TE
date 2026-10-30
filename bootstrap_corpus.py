"""Ingesta idempotente del único PDF oficial del proyecto, sin extraer secretos."""
import hashlib
import io
import time
import urllib.request

from google.genai import types
from pypdf import PdfReader
from pymongo import UpdateOne

PDF_URL = 'https://reports.weforum.org/docs/WEF_AI_Agents_in_Action_A_Playbook_for_Trusted_Adoption_Authorization_and_Scaling_2026.pdf'
DOCUMENT = 'wef-ai-agents-2026'


def ensure_corpus(ai, collection, model, progress=lambda message: None):
    progress('Descargando el PDF oficial…')
    with urllib.request.urlopen(PDF_URL, timeout=30) as response:
        data = response.read(20 * 1024 * 1024 + 1)
    if len(data) > 20 * 1024 * 1024 or not data.startswith(b'%PDF'):
        raise ValueError('PDF inválido o demasiado grande')
    checksum = hashlib.sha256(data).hexdigest()
    reader = PdfReader(io.BytesIO(data))
    text = ''.join((p.extract_text() or '') + '\n' for p in reader.pages).strip()
    if len(text) < 1000:
        raise ValueError('Texto insuficiente')
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    present = {r['id'] for r in collection.find(
        {'documento': DOCUMENT, 'source_sha256': checksum, 'embedding_model': model}, {'id': 1})}
    pending = [i for i in range(len(chunks)) if i not in present]
    for start in range(0, len(pending), 10):
        ids = pending[start:start+10]
        progress(f'Creando embeddings: {len(present)} de {len(chunks)} fragmentos…')
        contents = [f'\nRepresenta el siguiente fragmento de un documento\npara recuperación semántica en un sistema RAG.\n\nFragmento:\n\n{chunks[i]}\n' for i in ids]
        for attempt in range(3):
            try:
                result = ai.models.embed_content(model=model, contents=contents,
                    config=types.EmbedContentConfig(output_dimensionality=3072))
                break
            except Exception as exc:
                if str(getattr(exc, 'code', '')) not in {'429', '503'} or attempt == 2:
                    raise
                progress('Gemini está ocupado; reintentando la carga…')
                time.sleep(5 * (attempt+1))
        if len(result.embeddings) != len(ids):
            raise ValueError('Cantidad de vectores incompatible')
        operations = []
        for i, vector in zip(ids, result.embeddings):
            if len(vector.values) != 3072:
                raise ValueError('Dimensiones incompatibles')
            record = {'id': i, 'texto': chunks[i], 'embedding': vector.values,
                      'documento': DOCUMENT, 'source_sha256': checksum,
                      'embedding_model': model, 'fuente': PDF_URL}
            operations.append(UpdateOne({'_id': f'{DOCUMENT}:{i}'}, {'$set': record}, upsert=True))
        collection.bulk_write(operations, ordered=False)
        present.update(ids)
        time.sleep(0.5)
    return {'count': len(chunks), 'pages': len(reader.pages), 'characters': len(text)}
