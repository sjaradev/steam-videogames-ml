# ACIF104 · Semana 6 · Grupo 11

**Mapa de tipologías de videojuegos de Steam**. Prototipo académico reproducible, probado localmente el 8 de noviembre de 2026. Este repositorio **no** constituye por sí solo evidencia de cumplimiento integral de la rúbrica: faltan validación humana, pruebas de usabilidad y carga, despliegue seguro, monitoreo de deriva y publicación en una cuenta de GitHub del grupo.

## Datos originales y reproducibilidad

El CSV original **no viene incluido** en este ZIP. El grupo debe colocar su copia de `app_dataset_20260907_20260918(1).csv` en `data/steam_original.csv`. El SHA-256 utilizado para la ejecución del informe es:

`45930a2ab0eefd8349ae2932ea56bcf0443da65ee632640d7fb467decfefb18d` *(completo; debe coincidir con el archivo de entrada)*.

Esquema confirmado (15 columnas): `appid, name, genres, positive_reviews, total_reviews, tags, coming_soon, release_date, release_date_raw, is_free, currency, initial_price, final_price, discount_percent, fetched_at`.

- Archivo original: 187.115 registros.
- Filtro: lanzados, al menos 10 reseñas y tags válidos del vocabulario entrenado.
- Matriz final: 82.649 videojuegos × 150 tags binarios.
- Partición por semilla 42: 57.853 entrenamiento; 12.397 validación; 12.399 prueba.
- **No** se usan etiquetas de género como verdad de terreno; los nombres de grupos son provisionales.

## Instalación

Python 3.11 o superior; se ejecutó con Linux y PyTorch CPU (CUDA no disponible). En Windows:

```powershell
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

En Linux/macOS:

```bash 
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Las dependencias principales son pandas, numpy, scikit-learn, PyTorch, imbalanced-learn, SHAP, FastAPI, Uvicorn y matplotlib. El frontend HTML integrado no necesita Streamlit; la versión `frontend/app.py` es una alternativa opcional.

## Reproducir resultados desde cero

Copiar el CSV original a `data/steam_original.csv` y ejecutar desde la raíz del proyecto:

```bash
python -m src.prepare --csv data/steam_original.csv --tags-column tags --reviews-column total_reviews --coming-soon-column coming_soon --id-column appid --n-tags 150 --min-reviews 10
python -m src.eda --csv data/steam_original.csv
python -m src.experiment --k 12 --epochs 20 --max-train-dl 10000 --seed 42
python -m src.evaluate_extra
python -m src.refine
python -m src.build_catalog --csv data/steam_original.csv
```

La salida se guarda en `data/`, `artifacts/` y `results/`. Los archivos `results/metrics.json`, `results/eda.json`, `results/refinement.json`, `results/extra.json`, `results/cluster_profiles.json` y sus figuras fueron generados con datos reales. Los artefactos del ZIP permiten además probar la API sin reentrenar.

**Detalles a tener presente:** el vocabulario se calcula solo en train; el jerárquico Ward se entrena con 1.800 juegos y se evalúa en observaciones nuevas mediante **aproximación al centroide**, pues el algoritmo no implementa `predict`. Las tres arquitecturas neuronales se entrenaron con **10.000** juegos durante 20 épocas, frente a 57.853 en los modelos clásicos; la comparación es exploratoria, no una igualdad estricta de presupuestos. Silueta, Davies-Bouldin y Calinski-Harabasz se comparan en la misma submuestra de 1.200 juegos del conjunto de prueba y en el espacio original de 150 tags. Las métricas de clasificación del modelo Random Forest sustituto **no miden exactitud de las tipologías**, sino concordancia con pseudoetiquetas K-Means. SHAP explica ese sustituto, no causalidad ni la función de decisión interna de K-Means.

## Aplicación web

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Abrir en el navegador:
- Interfaz básica: http://127.0.0.1:8000/ui
- Documentación OpenAPI: http://127.0.0.1:8000/docs

La interfaz web contiene exploración de las doce tipologías, lista completa de tags seleccionables, predicción con tags, búsqueda de juegos, opción de ocultar contenido adulto en resultados de búsqueda, mapa PCA estático y monitoreo de consultas. Endpoints disponibles: `GET /health`, `/tags`, `/clusters`, `/profiles`, `/games?query=...`, `/metrics`, `/monitor`, `/map.png`, `/ui` y `POST /predict`.

El monitoreo local registra consultas en SQLite. **No hay autenticación, HTTPS, detección PSI, reentrenamiento protegido ni escalado productivo**; no exponer este prototipo a Internet sin ampliar los controles.

## Pruebas automatizadas

```bash
pytest -q
```

Se comprueba carga del modelo, perfiles y vocabulario, búsqueda, frontend integrado, métricas, mapa, clasificación y validación de datos inválidos. Las pruebas no reemplazan los ensayos RNF de carga, seguridad y usabilidad.

## Estructura

```
api/main.py                Backend FastAPI
frontend/index.html        Interfaz integrada accesible por /ui
frontend/app.py           Frontend Streamlit alternativo sin pruebas en esta ejecución
src/prepare.py            Filtros, particiones y vocabulario
src/eda.py                Calidad de datos, outliers y visualizaciones
src/experiment.py         ML clásico, autoencoders y balanceo del sustituto
src/evaluate_extra.py     Sensibilidad, estabilidad y SHAP
src/refine.py             Ensayo adicional k=12 frente a k=15
src/build_catalog.py      Perfiles y catálogo de consulta
tests/test_api.py         Pruebas locales de API
data/                     CSV indexado y matrices procesadas (original no incluido)
artifacts/                Modelos y pesos entrenados
results/                  Métricas, figuras y perfiles
```

## Pendientes prioritarios antes de la entrega

1. Publicar el código en un GitHub accesible al docente y añadir la URL al informe.
2. Revisar contenido, autoría y conclusiones con todos los integrantes.
3. Validar al menos algunos nombres de clusters con usuarios; documentar capturas de la UI si corresponde.
4. Implementar/comprobar los restantes requisitos de seguridad, usabilidad, escalabilidad y monitoreo.
5. Ajustar el documento al formato institucional descargable de la asignatura si difiere de la maqueta preparada.
