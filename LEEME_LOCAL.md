# Chatbot de agentes de IA

Esta versión añade respuesta progresiva, selector de k (3, 5, 10), medición de
embedding/búsqueda/generación, caché de embeddings por sesión y fragmentos completos
conservados en el historial. El cliente de Gemini limita las esperas por petición
a 30 segundos y realiza un intento; MongoDB limita las operaciones a 10 segundos.
Esto evita reintentos prolongados, pero no garantiza que un proveedor ocupado
responda más rápido. Los límites de transporte no constituyen un límite global
estricto para toda una respuesta transmitida progresivamente.

## Ejecutar

1. Instala Python 3.12 y las dependencias con `python -m pip install -r requirements.txt`.
2. Copia `.streamlit/secrets.example.toml` a `.streamlit/secrets.toml`.
3. Introduce credenciales vigentes localmente. No uses las expuestas en capturas.
4. Desde esta carpeta ejecuta `python -m streamlit run app.py`.

La aplicación también acepta GOOGLE_API_KEY y MONGODB_URI como variables de entorno.
GENERATION_MODEL permite seleccionar un modelo disponible para tu cuenta sin
editar el programa. Se mantiene el modelo original como predeterminado; cambiar
el modelo de embeddings exige compatibilidad con los vectores ya almacenados.

## Publicar los cambios

Sube app.py y el archivo de compatibilidad `app (3).py` junto con requirements.txt.
El archivo de compatibilidad permite conservar el punto de entrada de Streamlit.
No subas secretos, entornos locales ni resultados con información privada.

## Comprobar la latencia

Con credenciales configuradas, repite una pregunta con k=3, 5 y 10. Anota los
tiempos mostrados: la segunda consulta puede reutilizar el embedding, mientras
que la búsqueda y la generación se realizan nuevamente. Se limita la longitud
de salida; esto puede acortar respuestas respecto del programa original.
No se han medido mejoras de velocidad contra los servicios reales sin credenciales.

Las pruebas `python -m unittest test_app.py` usan servicios simulados para
verificar selección de k, conservación de evidencia, historial y reutilización
del embedding. No sustituyen la evaluación de calidad con el PDF real.
