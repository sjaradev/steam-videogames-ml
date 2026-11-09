"""API funcional mínima: predicción y registro de consultas, no equivale al sistema completo de rúbrica."""
from pathlib import Path
from collections import Counter
import json,sqlite3,time
import numpy as np
import joblib
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field
ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'artifacts/kmeans.joblib';VOCAB=ROOT/'data/vocab.json';DB=ROOT/'artifacts/requests.sqlite'
app=FastAPI(title='ACIF104 - Tipologías Steam',version='0.1')
km=None;vocab=[]
class Prediction(BaseModel):
    tags:list[str]=Field(min_length=1,max_length=20)

def init_db():
    DB.parent.mkdir(exist_ok=True)
    with sqlite3.connect(DB) as con:con.execute('CREATE TABLE IF NOT EXISTS calls (ts TEXT, cluster INTEGER, latency_ms REAL, distance REAL)')

@app.on_event('startup')
def startup():
    global km,vocab
    if MODEL.exists() and VOCAB.exists():
        km=joblib.load(MODEL);vocab=json.loads(VOCAB.read_text(encoding='utf-8'))
    init_db()

@app.get('/health')
def health():return {'status':'ok' if km is not None else 'no_model','model_loaded':km is not None}

@app.get('/clusters')
def clusters():
    if km is None:raise HTTPException(503,'Falta entrenar el modelo principal.')
    centers=km.cluster_centers_
    return [{'cluster':int(i),'top_tags':[{'tag':vocab[j],'relative_frequency':round(float(centers[i,j]),3)}
            for j in np.argsort(centers[i])[-8:][::-1]]} for i in range(len(centers))]

@app.post('/predict')
def predict(req:Prediction):
    if km is None:raise HTTPException(503,'Primero ejecutar preparación y experimentos.')
    bad=[t for t in req.tags if t not in vocab]
    if bad:raise HTTPException(422,f'Tags desconocidos o no incluidos en vocabulario: {bad}')
    t=time.perf_counter();x=np.zeros((1,len(vocab)),dtype=np.float32)
    for tag in req.tags:x[0,vocab.index(tag)]=1
    c=int(km.predict(x)[0]);dist=float(np.linalg.norm(x-km.cluster_centers_[c]))
    ms=round((time.perf_counter()-t)*1000,3)
    with sqlite3.connect(DB) as con:con.execute("INSERT INTO calls VALUES(datetime('now'),?,?,?)",(c,ms,dist))
    salient=[{'tag':vocab[j],'cluster_frequency':round(float(km.cluster_centers_[c,j]),3)} for j in np.argsort(km.cluster_centers_[c])[-5:][::-1]]
    return {'cluster':c,'distance_to_centroid':dist,'latency_ms':ms,'cluster_top_tags':salient,
       'warning':'Los tags son descriptores del centroide, no explicaciones causales.'}

@app.get('/monitor')
def monitor():
    with sqlite3.connect(DB) as con:rows=con.execute('SELECT cluster,latency_ms FROM calls').fetchall()
    return {'calls':len(rows),'counts':dict(Counter(str(x[0]) for x in rows)),
        'mean_latency_ms':round(sum(x[1] for x in rows)/len(rows),3) if rows else None}

# Endpoints añadidos después de realizar experimentos verificables.
from fastapi.responses import FileResponse, HTMLResponse
from fastapi import Query
import pandas as pd
import re
import unicodedata
from functools import lru_cache

INDEX=ROOT/'data/catalog_index.csv';PROFILES=ROOT/'results/cluster_profiles.json';METRICS=ROOT/'results/metrics.json';MAP=ROOT/'results/figures/11_mapa_pca.png'

@app.get('/tags')
def all_tags():
    if not vocab:raise HTTPException(503,'Vocabulario no preparado.')
    return {'tags':vocab,'count':len(vocab)}

@app.get('/profiles')
def profiles():
    if not PROFILES.exists():raise HTTPException(503,'Ejecutar el script build_catalog.')
    return json.loads(PROFILES.read_text(encoding='utf8'))

def normalize_name(value):
    value=unicodedata.normalize('NFKD',str(value).casefold())
    return ' '.join(re.sub(r'[^\w\s]', ' ', ''.join(c for c in value if not unicodedata.combining(c))).split())

@lru_cache(maxsize=1)
def load_catalog(modified_ns):
    d=pd.read_csv(INDEX).dropna(subset=['name'])
    d['_search']=d.name.map(normalize_name)
    return d

@app.get('/games')
def games(query:str=Query(min_length=1,max_length=100),limit:int=Query(default=20,ge=1,le=50),hide_adult:bool=False,offset:int=Query(default=0,ge=0)):
    if not INDEX.exists():raise HTTPException(503,'Falta índice de juegos.')
    q=normalize_name(query)
    if not q:return {'matches':0,'results':[],'has_more':False}
    d=load_catalog(INDEX.stat().st_mtime_ns)
    mask=pd.Series(True,index=d.index)
    for word in q.split():mask &= d['_search'].str.contains(word,regex=False,na=False)
    d=d.loc[mask].copy()
    if hide_adult:d=d[~d.adult_content]
    d['_rank']=np.select([d['_search'].eq(q),d['_search'].str.startswith(q),d['_search'].str.contains(q,regex=False)],[0,1,2],default=3)
    d=d.sort_values(['_rank','total_reviews','name'],ascending=[True,False,True])
    page=d.iloc[offset:offset+limit].drop(columns=['_search','_rank'])
    return {'matches':len(d),'results':page.to_dict('records'),'has_more':offset+limit<len(d)}

@app.get('/metrics')
def metrics():
    if not METRICS.exists():raise HTTPException(503,'Faltan experimentos.')
    return json.loads(METRICS.read_text(encoding='utf8'))

@app.get('/map.png')
def map_png():
    if not MAP.exists():raise HTTPException(503,'Mapa todavía no generado.')
    return FileResponse(MAP,media_type='image/png')

@app.get('/ui',response_class=HTMLResponse)
def ui():
    f=ROOT/'frontend/index.html'
    if not f.exists():raise HTTPException(503,'Archivo de interfaz no disponible.')
    return f.read_text(encoding='utf8')
