"""Exploración verificable del CSV original (datos reales, sin simulación)."""
import ast,json
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.prepare import parse_tags
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'; OUT.mkdir(exist_ok=True); FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
SOURCE=ROOT/'data'/'steam_original.csv'

def savefigure(name):
 plt.tight_layout();plt.savefig(FIG/name,dpi=170,bbox_inches='tight');plt.close()

def main(source=SOURCE):
 df=pd.read_csv(source,low_memory=False)
 launched=df[~df.coming_soon].copy()
 filt=launched.loc[launched.total_reviews.ge(10)].copy()
 filt['parsed_tags']=filt.tags.map(parse_tags)
 has_tags=filt[filt.parsed_tags.map(bool)].copy()
 alltags=Counter(t for tags in has_tags.parsed_tags for t in set(tags))
 # Las etiquetas seleccionadas en la preparación se tomaron de TRAIN únicamente.
 vocab=json.loads((ROOT/'data/vocab.json').read_text(encoding='utf8'))
 model=has_tags[has_tags.parsed_tags.map(lambda tags:any(t in set(tags) for t in vocab))].copy()
 dt=pd.to_datetime(df.release_date,errors='coerce'); yr=dt.dt.year
 price=pd.to_numeric(launched.initial_price,errors='coerce'); usd=launched.loc[launched.currency.eq('USD'),'initial_price']
 bad_dates=dt.notna() & (yr.lt(1980)|yr.gt(2026)); bad_reviews=((launched.positive_reviews>launched.total_reviews)|(launched.total_reviews<0)|(launched.positive_reviews<0)).sum()
 report={
  'rows':int(len(df)),'columns':int(df.shape[1]),'names':df.columns.tolist(),
  'duplicate_appid':int(df.appid.duplicated().sum()),'launched':int(len(launched)),'coming_soon':int(df.coming_soon.sum()),
  'reviews_ge_10':int(len(filt)),'nonempty_tags_after_review_filter':int(len(has_tags)),
  'model_rows_after_training_vocabulary':int(len(model)),'n_unique_tags_filtered':int(len(alltags)),
  'n_unique_genres':int(len(set(t for v in launched.genres.dropna().map(parse_tags) for t in v))),
  'missing_raw':{str(c):int(v) for c,v in df.isna().sum().items()},
  'missing_launched':{str(c):int(v) for c,v in launched.isna().sum().items()},
  'invalid_date_outside_1980_2026':int(bad_dates.sum()),
  'unparseable_nonblank_dates':int((df.release_date.notna() & dt.isna()).sum()),
  'negative_or_contradictory_reviews_launched':int(bad_reviews),
  'review_quantiles':{str(k):float(v) for k,v in launched.total_reviews.quantile([0,.25,.5,.75,.9,.95,.99,1]).items()},
  'price_usd_quantiles':{str(k):float(v) for k,v in usd.quantile([0,.25,.5,.75,.9,.95,.99,1]).items()},
  'top_tags':alltags.most_common(20),
  'tags_count_per_game_mean':round(float(model.parsed_tags.map(len).mean()),3),
  'columns_dictionary':[{ 'name':c,'type':str(df[c].dtype),'missing':int(df[c].isna().sum()),'unique':int(df[c].nunique())} for c in df.columns],
  'snapshot_range':[str(df.fetched_at.min()),str(df.fetched_at.max())],
  'model_report_note':'Estadísticas recalculadas directamente desde CSV. La elección del vocabulario de 150 tags se hizo con train y semilla 42; filtrado final igual a fase 1.'
 }
 (OUT/'eda.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
 # F1 conteos anual (solamente fechas válidas y antes de la fecha de captación)
 years=yr[(~df.coming_soon)&yr.between(2015,2026)].value_counts().sort_index()
 fig,ax=plt.subplots(figsize=(8,3.2));ax.bar(years.index.astype(int).astype(str),years.values)
 ax.set_ylabel('Aplicaciones');ax.set_xlabel('Año de lanzamiento');ax.set_title('Juegos lanzados por año (según fecha registrada)')
 plt.xticks(rotation=45);savefigure('01_lanzamientos.png')
 # F2 nulos totales, advertir futuras mezcladas
 missing=df.isna().sum().sort_values(ascending=False)
 fig,ax=plt.subplots(figsize=(9,4.2));ax.barh(missing.index[::-1],missing.values[::-1]);ax.set_xlabel('Registros sin dato')
 ax.set_title('Valores ausentes en las 187.115 filas (incluye próximos lanzamientos)');savefigure('02_nulos.png')
 # F3 tags top 15
 items=alltags.most_common(15)
 fig,ax=plt.subplots(figsize=(9,5));ax.barh([a for a,b in items][::-1],[b for a,b in items][::-1]);ax.set_xlabel('Juegos (con 10+ reseñas y tags)');ax.set_title('Tags más frecuentes');savefigure('03_tags.png')
 # F4 reviews log
 r=model.total_reviews.clip(lower=0)
 fig,ax=plt.subplots(figsize=(8,3.5));ax.hist(np.log10(r+1),bins=35);ax.set_xlabel('log10(reseñas + 1)');ax.set_ylabel('Juegos');ax.set_title('Distribución de reseñas, escala logarítmica');savefigure('04_reseñas.png')
 # F5 correlation; no target label
 C=model[['positive_reviews','total_reviews','initial_price','final_price','discount_percent']].copy();C['positive_reviews']=np.log1p(C['positive_reviews']);C['total_reviews']=np.log1p(C['total_reviews']);C['initial_price']=np.log1p(C['initial_price'].clip(lower=0));C['final_price']=np.log1p(C['final_price'].clip(lower=0));corr=C.corr()
 fig,ax=plt.subplots(figsize=(7,5));im=ax.imshow(corr,vmin=-1,vmax=1,cmap='coolwarm');ax.set_xticks(range(len(C.columns)),C.columns,rotation=38,ha='right',fontsize=8);ax.set_yticks(range(len(C.columns)),C.columns,fontsize=8)
 for i in range(len(corr)):
  for j in range(len(corr)):ax.text(j,i,f'{corr.iloc[i,j]:.2f}',ha='center',va='center',fontsize=8)
 fig.colorbar(im,ax=ax,shrink=.8);ax.set_title('Correlaciones (reviews/precios con log1p)');savefigure('05_correlaciones.png')
 # F6 group original snapshots (for sense of coverage)
 print(json.dumps({k:v for k,v in report.items() if k in ['rows','columns','launched','coming_soon','reviews_ge_10','nonempty_tags_after_review_filter','model_rows_after_training_vocabulary','n_unique_tags_filtered','invalid_date_outside_1980_2026','negative_or_contradictory_reviews_launched']},ensure_ascii=False,indent=2))

if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--csv',default=str(SOURCE)); args=ap.parse_args();main(args.csv)
