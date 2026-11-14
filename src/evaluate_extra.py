"""Análisis adicional de sensibilidad, estabilidad y SHAP, sobre datos reales reservados."""
import json, time
from pathlib import Path
import joblib, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, f1_score
from src.experiment import score
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results';FIG=OUT/'figures';FIG.mkdir(exist_ok=True)

def main():
 np.random.seed(42)
 metrics=json.loads((OUT/'metrics.json').read_text(encoding='utf8'))
 d=np.load(ROOT/'data/prepared.npz')
 Xtr=d['X_train'].astype('float32');Xv=d['X_val'].astype('float32');Xt=d['X_test'].astype('float32')
 km=joblib.load(ROOT/'artifacts/kmeans.joblib');clf=joblib.load(ROOT/'artifacts/surrogate.joblib')
 rng=np.random.default_rng(42); idx=rng.choice(len(Xtr),size=min(15000,len(Xtr)),replace=False)
 S=Xtr[idx]
 sensitivity=[]
 for k in (8,10,12,15,20):
  start=time.perf_counter();model=KMeans(n_clusters=k,n_init=5,random_state=42).fit(S)
  a=score(Xv,model.predict(Xv));sensitivity.append({'k':k,'train_subset_rows':len(S),'validation':a,'seconds':round(time.perf_counter()-start,2)})
 stability=[];labels=[]
 for seed in (42,43,44):
  model=KMeans(n_clusters=12,n_init=5,random_state=seed).fit(S)
  labels.append(model.predict(Xv))
 for a in range(3):
  for b in range(a+1,3):stability.append({'seeds':[42+a,42+b],'ari_on_validation':round(float(adjusted_rand_score(labels[a],labels[b])),4)})
 # SHAP explica solo el modelo supervisado sustituto, no el clustering en sí.
 explain={'status':'no_ejecutado'}
 try:
  import shap
  # mismos 10 juegos retenidos de test, no se usaron para entrenamiento del sustituto.
  x=Xt[:10]; pred=clf.predict(x)
  explainer=shap.TreeExplainer(clf)
  result=explainer.shap_values(x)
  vals=np.asarray(result)
  if vals.ndim==3 and vals.shape[0]==10:
   impacts=vals[0,:,int(pred[0])]
  elif isinstance(result,list) and len(result)>int(pred[0]):
   impacts=np.asarray(result[int(pred[0])])[0]
  else:raise ValueError(f'Formato SHAP inesperado: {vals.shape}')
  vocab=json.loads((ROOT/'data/vocab.json').read_text(encoding='utf8'))
  indices=np.argsort(np.abs(impacts))[-12:][::-1]
  explain={'status':'ejecutado','sample_test_row':0,'sample_test_appid':str(d['id_test'][0]),
   'surrogate_predicted_cluster':int(pred[0]),'kmeans_cluster':int(km.predict(x[:1])[0]),
   'top_features':[{'tag':vocab[j],'shap_value':round(float(impacts[j]),7),'tag_present':bool(x[0,j])} for j in indices],
   'caution':'SHAP explica un RandomForest entrenado con pseudoetiquetas de K-Means; no demuestra causalidad ni verdad de clusters.'}
  fig,ax=plt.subplots(figsize=(8,4.4))
  vals1=[float(impacts[i]) for i in indices][::-1];tags=[vocab[i] for i in indices][::-1]
  ax.barh(tags,vals1);ax.axvline(0,color='gray',lw=.8);ax.set_xlabel('SHAP del sustituto para la clase predicha');ax.set_title('Explicación local de una asignación (juego de test)')
  fig.tight_layout();fig.savefig(FIG/'10_shap_local.png',dpi=180,bbox_inches='tight');plt.close(fig)
 except Exception as ex:explain={'status':'fallo','error':str(ex)}
 # Gráfico común de comparación en espacio binario original
 models={}
 for k,v in metrics['traditional'].items():
  t=v.get('test') or v.get('test_by_nearest_center_approximation');models[k]=t
 for k,v in metrics['deep_learning'].items():models[k]=v['test_original_tags']
 fig,axes=plt.subplots(1,2,figsize=(11,4))
 labs=list(models);sils=[models[l]['silueta'] for l in labs];dbis=[models[l]['davies_bouldin'] for l in labs]
 axes[0].barh(labs[::-1],sils[::-1]);axes[0].set_xlabel('Silueta (mayor es mejor)')
 axes[1].barh(labs[::-1],dbis[::-1]);axes[1].set_xlabel('Davies–Bouldin (menor es mejor)')
 fig.suptitle('Comparación sobre datos de prueba, 1.200 ejemplos, espacio original')
 fig.tight_layout();fig.savefig(FIG/'06_modelos.png',dpi=170,bbox_inches='tight');plt.close(fig)
 fig,ax=plt.subplots(figsize=(7.5,3.8))
 for name,v in metrics['deep_learning'].items():
  h=v['history'];ax.plot([item['epoch'] for item in h],[item['val_loss'] for item in h],label=name+' (validación)',marker='o')
 ax.set_xlabel('Época');ax.set_ylabel('Pérdida de reconstrucción (+ KL en VAE)');ax.legend(fontsize=8);ax.set_title('Convergencia de redes neuronales');fig.tight_layout();fig.savefig(FIG/'07_convergencia.png',dpi=170,bbox_inches='tight');plt.close(fig)
 bal=metrics['balancing_surrogate'];labs=list(bal);fig,ax=plt.subplots(figsize=(8,3.5))
 ax.barh(labs[::-1],[bal[l]['macro_f1'] for l in labs[::-1]]);ax.set_xlim(.8,.93)
 ax.set_xlabel('Macro-F1, fidelidad al K-Means (no exactitud real)');ax.set_title('Estrategias en el clasificador sustituto');fig.tight_layout();fig.savefig(FIG/'08_balanceo.png',dpi=170,bbox_inches='tight');plt.close(fig)
 fig,ax=plt.subplots(figsize=(7.5,3.6));ax.plot([r['k'] for r in sensitivity],[r['validation']['davies_bouldin'] for r in sensitivity],marker='o')
 ax.set_xlabel('Número de grupos k');ax.set_ylabel('Davies–Bouldin, validación');ax.set_title('Sensibilidad a k (muestra de 15.000 juegos)');fig.tight_layout();fig.savefig(FIG/'09_sensibilidad.png',dpi=170,bbox_inches='tight');plt.close(fig)
 # La proyección PCA a 2D es para visualización, no para evaluación cuantitativa.
 idx=rng.choice(len(Xt),size=min(2500,len(Xt)),replace=False)
 p=PCA(n_components=2,random_state=42).fit(Xtr[rng.choice(len(Xtr),5000,replace=False)])
 xy=p.transform(Xt[idx]);cluster=km.predict(Xt[idx])
 fig,ax=plt.subplots(figsize=(7.8,4.8));sc=ax.scatter(xy[:,0],xy[:,1],c=cluster,s=3,alpha=.45,cmap='tab20')
 ax.set_xlabel('Componente principal 1');ax.set_ylabel('Componente principal 2');ax.set_title('Mapa exploratorio 2D de juegos reservados');fig.colorbar(sc,ax=ax,label='Cluster');fig.tight_layout();fig.savefig(FIG/'11_mapa_pca.png',dpi=180,bbox_inches='tight');plt.close(fig)
 output={'sensitivity':sensitivity,'stability_pairs':stability,'shap':explain,'note':'Sensibilidad/estabilidad realizadas en una muestra aleatoria de 15.000 juegos de entrenamiento; informe completo debe indicar este límite.'}
 (OUT/'extra.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf8')
 print(json.dumps({'sensitivity':sensitivity,'stability':stability,'shap_status':explain['status'],'shap_top':explain.get('top_features',[])[:5]},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
