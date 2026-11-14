"""Comparación adicional sobre dataset completo y validación (k=12 vs 15)."""
import json,time
from pathlib import Path
import joblib,numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from src.experiment import score
ROOT=Path(__file__).resolve().parents[1]
def main():
 z=np.load(ROOT/'data/prepared.npz');Xtr=z['X_train'].astype('float32');Xva=z['X_val'].astype('float32');Xte=z['X_test'].astype('float32')
 old=joblib.load(ROOT/'artifacts/kmeans.joblib')
 t=time.perf_counter();new=KMeans(n_clusters=15,n_init=10,random_state=42).fit(Xtr);elapsed=round(time.perf_counter()-t,2)
 info={'k12':{'validation':score(Xva,old.predict(Xva)),'test':score(Xte,old.predict(Xte))},'k15':{'time_seconds':elapsed,'validation':score(Xva,new.predict(Xva)),'test':score(Xte,new.predict(Xte))},
       'warning':'k15 es experimento de refinamiento; el equipo debe valorar interpretación de los 15 grupos antes de sustituir el modelo de k12.'}
 (ROOT/'results/refinement.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf8');joblib.dump(new,ROOT/'artifacts/kmeans_k15.joblib')
 print(json.dumps(info,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
