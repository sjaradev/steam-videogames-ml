"""Experimentos comparativos. No contiene resultados predefinidos.

Corre sobre matrices preparadas por src.prepare. Los resultados solo representan el
CSV suministrado y las semillas/computador de la ejecución correspondiente.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import joblib
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, balanced_accuracy_score, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
ROOT=Path(__file__).resolve().parents[1]

def score(X,y,seed=42):
    X=np.asarray(X);y=np.asarray(y)
    unique=np.unique(y)
    if len(unique)<2 or len(unique)>=len(y):return {'silueta':None,'davies_bouldin':None,'calinski_harabasz':None}
    rng=np.random.default_rng(seed); ix=rng.choice(len(X),min(1200,len(X)),replace=False)
    X,y=X[ix],y[ix]
    if len(np.unique(y))<2:return {'silueta':None,'davies_bouldin':None,'calinski_harabasz':None}
    return {'silueta':round(float(silhouette_score(X,y)),5),'davies_bouldin':round(float(davies_bouldin_score(X,y)),5),
            'calinski_harabasz':round(float(calinski_harabasz_score(X,y)),3)}

class AE(nn.Module):
    def __init__(self,p,kind):
        super().__init__();self.kind=kind
        dims=[p,64,16] if kind=='AE_compacto' else [p,128,64,32]
        if kind=='VAE':dims=[p,128,64]
        layers=[]
        for a,b in zip(dims[:-1],dims[1:]):layers.extend([nn.Linear(a,b),nn.ReLU()])
        self.encoder=nn.Sequential(*layers)
        if kind=='VAE':
            self.mu=nn.Linear(64,16);self.logvar=nn.Linear(64,16); dec=[16,64,128,p]
        else:dec=list(reversed(dims))
        layers=[]
        for j,(a,b) in enumerate(zip(dec[:-1],dec[1:])):
            layers.append(nn.Linear(a,b));layers.append(nn.Sigmoid() if j==len(dec)-2 else nn.ReLU())
        self.decoder=nn.Sequential(*layers)
    def forward(self,x):
        h=self.encoder(x)
        if self.kind=='VAE':
            mu,lv=self.mu(h),self.logvar(h)
            z=mu+torch.randn_like(mu)*torch.exp(.5*lv) if self.training else mu
            return self.decoder(z),mu,lv
        return self.decoder(h),h,None
    @torch.no_grad()
    def embedding(self,X):
        self.eval(); x=torch.as_tensor(X,dtype=torch.float32)
        h=self.encoder(x)
        return (self.mu(h) if self.kind=='VAE' else h).numpy()

def train_ae(Xtr,Xva,Xte,kind,epochs,seed,batch_size=512):
    torch.manual_seed(seed);torch.set_num_threads(min(4,torch.get_num_threads()))
    net=AE(Xtr.shape[1],kind);opt=torch.optim.Adam(net.parameters(),lr=1e-3)
    loader=DataLoader(TensorDataset(torch.as_tensor(Xtr,dtype=torch.float32)),batch_size=batch_size,shuffle=True)
    losses=[]
    def get_loss(x,out,mu,lv):
        bce=nn.functional.binary_cross_entropy(out,x,reduction='mean')
        return bce+(.001*(-.5*(1+lv-mu.square()-lv.exp()).sum(dim=1).mean())) if lv is not None else bce
    t0=time.perf_counter()
    for epoch in range(epochs):
        net.train();total=0.;n=0
        for (x,) in loader:
            opt.zero_grad();out,mu,lv=net(x);loss=get_loss(x,out,mu,lv);loss.backward();opt.step()
            total+=float(loss.detach())*len(x);n+=len(x)
        net.eval()
        with torch.no_grad():
            xv=torch.as_tensor(Xva,dtype=torch.float32)
            out,mu,lv=net(xv); val=float(get_loss(xv,out,mu,lv))
        losses.append({'epoch':epoch+1,'train_loss':total/n,'val_loss':val})
    Ztr=net.embedding(Xtr);Zva=net.embedding(Xva);Zte=net.embedding(Xte)
    return Ztr,Zva,Zte,losses,round(time.perf_counter()-t0,2),net

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--k',type=int,default=12)
    ap.add_argument('--epochs',type=int,default=6)
    ap.add_argument('--max-train-dl',type=int,default=10000)
    ap.add_argument('--seed',type=int,default=42)
    args=ap.parse_args()
    np.random.seed(args.seed);torch.manual_seed(args.seed)
    d=np.load(ROOT/'data/prepared.npz');Xtr=d['X_train'].astype('float32');Xva=d['X_val'].astype('float32');Xte=d['X_test'].astype('float32')
    if len(Xtr)<args.k*4:raise ValueError('Se necesitan más filas de entrenamiento para k elegido.')
    (ROOT/'artifacts').mkdir(exist_ok=True);(ROOT/'results').mkdir(exist_ok=True)
    report={'config':{'k':args.k,'seed':args.seed,'epochs':args.epochs,'train_rows':len(Xtr),'val_rows':len(Xva),'test_rows':len(Xte),'metric_sample_size':1200,'protocol':'k fijado por estudio previo, vocabulario calculado solo en entrenamiento; score en el mismo submuestreo aleatorio de 1200 registros en espacio original'},'traditional':{},'deep_learning':{},'balancing_surrogate':{}}
    t=time.perf_counter();km=KMeans(n_clusters=args.k,n_init=10,random_state=args.seed).fit(Xtr)
    report['traditional']['KMeans']={'time_seconds':round(time.perf_counter()-t,2),'validation':score(Xva,km.predict(Xva)),'test':score(Xte,km.predict(Xte)),'train_inertia':float(km.inertia_)}
    joblib.dump(km,ROOT/'artifacts/kmeans.joblib')
    p=min(72,Xtr.shape[1]-1,len(Xtr)-1)
    t=time.perf_counter();pca=PCA(n_components=p,random_state=args.seed).fit(Xtr)
    kmp=KMeans(n_clusters=args.k,n_init=10,random_state=args.seed).fit(pca.transform(Xtr))
    report['traditional']['PCA_KMeans']={'time_seconds':round(time.perf_counter()-t,2),'n_components':p,'validation':score(Xva,kmp.predict(pca.transform(Xva))),'test':score(Xte,kmp.predict(pca.transform(Xte)))}
    joblib.dump(pca,ROOT/'artifacts/pca.joblib');joblib.dump(kmp,ROOT/'artifacts/pca_kmeans.joblib')
    # Ward no admite predict: asignación externa por centroide aproximado, declarada como tal.
    rng=np.random.default_rng(args.seed);sel=rng.choice(len(Xtr),size=min(1800,len(Xtr)),replace=False)
    t=time.perf_counter();ward=AgglomerativeClustering(n_clusters=args.k,linkage='ward').fit(Xtr[sel])
    cent=np.stack([Xtr[sel][ward.labels_==k].mean(axis=0) for k in range(args.k)])
    yte=np.argmin(((Xte[:,None,:]-cent[None,:,:])**2).sum(axis=2),axis=1)
    yva=np.argmin(((Xva[:,None,:]-cent[None,:,:])**2).sum(axis=2),axis=1)
    np.save(ROOT/'artifacts/ward_centroids.npy',cent)
    report['traditional']['Ward_muestra']={'validation_by_nearest_center_approximation':score(Xva,yva),'time_seconds':round(time.perf_counter()-t,2),'train_subset_rows':len(sel),'test_by_nearest_center_approximation':score(Xte,yte),'warning':'Asignación externa aproximada: Ward no tiene predict.'}
    train_dl_idx=rng.choice(len(Xtr),size=min(args.max_train_dl,len(Xtr)),replace=False)
    for kind in ['AE_compacto','AE_profundo','VAE']:
        Ztr,Zva,Zte,history,elapsed,net=train_ae(Xtr[train_dl_idx],Xva,Xte,kind,args.epochs,args.seed)
        torch.save({'kind':kind,'n_features':Xtr.shape[1],'state_dict':net.state_dict()},ROOT/f'artifacts/{kind}.pt')
        mdl=KMeans(n_clusters=args.k,n_init=10,random_state=args.seed).fit(Ztr)
        pred=mdl.predict(Zte)
        report['deep_learning'][kind]={'train_seconds':elapsed,'history':history,'validation_original_tags':score(Xva,mdl.predict(Zva)),'test_original_tags':score(Xte,pred),'test_latent':score(Zte,pred),'n_train':len(Ztr),
            'note':'Métricas en el espacio original y latente; validar criterios de comparación antes de seleccionar.'}
    # 3 estrategias auxiliares: no aplicar a los clusters originales para alterar su distribución.
    pseudo=km.labels_;truth=km.predict(Xte)
    if len(Xtr)>25000:
        ix=rng.choice(len(Xtr),25000,replace=False)
    else: ix=np.arange(len(Xtr))
    Xs,ys=Xtr[ix],pseudo[ix]
    cases={'sobremuestreo_aleatorio':'over','submuestreo_aleatorio':'under','ponderacion_de_clases':'weight'}
    for name,kind in cases.items():
        if kind=='over':XX,yy=RandomOverSampler(random_state=args.seed).fit_resample(Xs,ys); weights=None
        elif kind=='under':XX,yy=RandomUnderSampler(random_state=args.seed).fit_resample(Xs,ys); weights=None
        else:XX,yy=Xs,ys;weights='balanced'
        clf=RandomForestClassifier(n_estimators=100,max_depth=16,n_jobs=-1,class_weight=weights,random_state=args.seed)
        clf.fit(XX,yy);pred=clf.predict(Xte)
        pval=clf.predict(Xva);truth_val=km.predict(Xva)
        report['balancing_surrogate'][name]={
            'validation_macro_f1':float(f1_score(truth_val,pval,average='macro',zero_division=0)),
            'train_rows':len(XX),'macro_f1':float(f1_score(truth,pred,average='macro',zero_division=0)),
            'macro_precision':float(precision_score(truth,pred,average='macro',zero_division=0)),
            'macro_recall':float(recall_score(truth,pred,average='macro',zero_division=0)),
            'balanced_accuracy':float(balanced_accuracy_score(truth,pred)),
            'confusion_matrix':confusion_matrix(truth,pred,labels=list(range(args.k))).tolist(),
            'warning':'Todas estas métricas miden fidelidad a etiquetas de K-Means, NO exactitud de las tipologías.'}
        if kind=='weight':joblib.dump(clf,ROOT/'artifacts/surrogate.joblib')
    joblib.dump({'seed':args.seed,'k':args.k,'train_dl_idx':train_dl_idx.tolist()},ROOT/'artifacts/experiment_config.joblib')
    (ROOT/'results/metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Experimentos guardados en results/metrics.json; modelos en artifacts/. No se aplicó balanceo al clustering principal.')

if __name__=='__main__':main()
