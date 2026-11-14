"""Preparación reproducible a partir del CSV ORIGINAL. Requiere nombres exactos de columnas.

Ejemplo de uso:
 python -m src.prepare --csv app_dataset_20260907_20260918.csv --tags-column tags \
   --reviews-column total_reviews --coming-soon-column coming_soon --id-column appid
"""
import argparse, ast, json
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT=Path(__file__).resolve().parents[1]

def parse_tags(v):
    if isinstance(v, (list,tuple,set)): return [str(x).strip() for x in v if str(x).strip()]
    if isinstance(v, dict): return [str(x) for x in v.keys()]
    if pd.isna(v): return []
    s=str(v).strip()
    if not s or s.lower() in ('nan','none','null'): return []
    try:
        obj=json.loads(s)
    except (json.JSONDecodeError,TypeError):
        try: obj=ast.literal_eval(s)
        except (ValueError,SyntaxError): obj=s
    if isinstance(obj,dict):return [str(k).strip() for k in obj if str(k).strip()]
    if isinstance(obj,(list,tuple,set)):return [str(x).strip() for x in obj if str(x).strip()]
    if isinstance(obj,str):
        sep='|' if '|' in obj else ';' if ';' in obj else ',' if ',' in obj else None
        return [t.strip() for t in obj.split(sep) if t.strip()] if sep else [obj.strip()]
    return []

def is_coming_soon(v):
    if pd.isna(v): return False
    return str(v).strip().lower() in ('true','1','yes','si','sí')

def main():
    pa=argparse.ArgumentParser()
    pa.add_argument('--csv',required=True)
    pa.add_argument('--tags-column',required=True)
    pa.add_argument('--reviews-column',required=True)
    pa.add_argument('--coming-soon-column',required=True)
    pa.add_argument('--id-column',required=True)
    pa.add_argument('--n-tags',type=int,default=150)
    pa.add_argument('--min-reviews',type=int,default=10)
    pa.add_argument('--seed',type=int,default=42)
    args=pa.parse_args()
    df=pd.read_csv(args.csv,low_memory=False)
    cols=[args.tags_column,args.reviews_column,args.coming_soon_column,args.id_column]
    missing=[k for k in cols if k not in df.columns]
    if missing:raise ValueError(f'Columnas ausentes {missing}. Columnas disponibles: {df.columns.tolist()}')
    n_raw=len(df)
    df=df.drop_duplicates(subset=args.id_column)
    df=df.loc[~df[args.coming_soon_column].map(is_coming_soon)].copy()
    reviews=pd.to_numeric(df[args.reviews_column],errors='coerce')
    df=df.loc[reviews>=args.min_reviews].copy()
    df['__tags']=df[args.tags_column].map(parse_tags)
    df=df.loc[df['__tags'].map(bool)].copy()
    ids=df[args.id_column].astype(str).to_numpy(dtype='U32')
    idx=np.arange(len(df))
    ix_train,ix_hold=train_test_split(idx,test_size=.3,random_state=args.seed)
    ix_val,ix_test=train_test_split(ix_hold,test_size=.5,random_state=args.seed)
    # Evitar fuga: elegir vocabulario solo con entrenamiento.
    freq=Counter(t for row in df['__tags'].iloc[ix_train] for t in set(row))
    vocab=[t for t,_ in freq.most_common(args.n_tags)]
    if len(vocab)<2:raise ValueError('No hay suficientes tags para modelar.')
    mapping={tag:i for i,tag in enumerate(vocab)}
    X=np.zeros((len(df),len(vocab)),dtype=np.uint8)
    for i,tags in enumerate(df['__tags']):
        for t in set(tags):
            j=mapping.get(t)
            if j is not None:X[i,j]=1
    # juegos con 0 tags del vocabulario: se conservan para auditar; se retiran del entrenamiento.
    tr=ix_train[X[ix_train].sum(axis=1)>0]
    va=ix_val[X[ix_val].sum(axis=1)>0]
    te=ix_test[X[ix_test].sum(axis=1)>0]
    dest=ROOT/'data';dest.mkdir(exist_ok=True)
    np.savez_compressed(dest/'prepared.npz',X_train=X[tr],X_val=X[va],X_test=X[te],id_train=ids[tr],id_val=ids[va],id_test=ids[te])
    (dest/'vocab.json').write_text(json.dumps(vocab,ensure_ascii=False,indent=2),encoding='utf-8')
    manifest={'raw_rows':n_raw,'filtered_pre_vocabulary':len(df),'train':len(tr),'val':len(va),'test':len(te),'features':len(vocab),'seed':args.seed,
        'columns':{'tags':args.tags_column,'reviews':args.reviews_column,'coming_soon':args.coming_soon_column,'id':args.id_column},
        'notes':'Vocabulario ajustado exclusivamente sobre entrenamiento. Auditar cómo vienen los tags en el CSV real.'}
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
