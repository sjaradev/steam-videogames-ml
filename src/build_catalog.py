"""Índice del catálogo, asignaciones del modelo y perfiles a partir del CSV real."""
from pathlib import Path
import pandas as pd, numpy as np, joblib, json
from src.prepare import parse_tags
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]

def main(csv):
 df=pd.read_csv(csv,low_memory=False)
 vocab=json.loads((ROOT/'data/vocab.json').read_text(encoding='utf-8'))
 model=joblib.load(ROOT/'artifacts/kmeans.joblib')
 z=np.load(ROOT/'data/prepared.npz')
 ids=np.concatenate([z['id_train'],z['id_val'],z['id_test']]); X=np.concatenate([z['X_train'],z['X_val'],z['X_test']]).astype('float32')
 labels=model.predict(X)
 base=pd.DataFrame({'appid':ids.astype('int64'),'cluster':labels})
 meta=df[['appid','name','genres','total_reviews','positive_reviews','is_free','initial_price','currency','release_date','tags']]
 final=base.merge(meta,on='appid',how='left',validate='one_to_one')
 final['adult_content']=final.tags.fillna('').str.contains("Sexual Content|Nudity|Hentai",regex=True)
 final[['appid','name','cluster','total_reviews','is_free','adult_content']].to_csv(ROOT/'data/catalog_index.csv',index=False)
 final['positive_percentage']=np.where(final.total_reviews.gt(0),100*final.positive_reviews/final.total_reviews,np.nan)
 final['release_year']=pd.to_numeric(final.release_date.str[:4],errors='coerce')
 centers=model.cluster_centers_;profiles=[]
 for cl,g in final.groupby('cluster'):
  order=np.argsort(centers[int(cl)])[::-1][:8];tags=[{'tag':vocab[j],'frequency_in_cluster':round(float(centers[int(cl),j]),3)} for j in order]
  # Precio USD solamente: evita mezclar divisas
  price=g.loc[g.currency.eq('USD'),'initial_price']
  profiles.append({'id':int(cl),'name_provisional':' / '.join(x['tag'] for x in tags[:3]),'n_games':int(len(g)),'percentage':round(100*len(g)/len(final),2),
   'tags':tags,'percent_free':round(float(100*g.is_free.mean()),1),
   'price_median_USD':round(float(price.median()),2) if price.notna().any() else None,
   'reviews_median':round(float(g.total_reviews.median()),2),'positive_percentage_median':round(float(g.positive_percentage.median()),1),
   'year_median':int(g.release_year.median()) if g.release_year.notna().any() else None,
   'adult_content_share':round(float(100*g.adult_content.mean()),1)})
 (ROOT/'results/cluster_profiles.json').write_text(json.dumps(profiles,ensure_ascii=False,indent=2),encoding='utf8')
 fig,ax=plt.subplots(figsize=(8,4.8));ax.barh([f'Grupo {p["id"]}' for p in profiles][::-1],[p['n_games'] for p in profiles][::-1]);ax.set_title('Tamaño de las tipologías: K-Means entrenado con train');ax.set_xlabel('Juegos');fig.tight_layout();fig.savefig(ROOT/'results/figures/12_tamanos.png',dpi=180,bbox_inches='tight');plt.close(fig)
 print('total',len(final),'size sum',sum(x['n_games'] for x in profiles));print(json.dumps(sorted(profiles,key=lambda x:-x['n_games'])[:3],ensure_ascii=False,indent=2))

if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--csv',required=True);args=ap.parse_args();main(args.csv)
