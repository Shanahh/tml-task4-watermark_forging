#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,math
from pathlib import Path
import numpy as np
from PIL import ImageEnhance,ImageFilter,Image
from scipy.fft import dctn
from scipy.signal import fftconvolve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter
from common import *

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--dataset',type=Path,required=True); p.add_argument('--output-dir',type=Path,default=Path('diagnostics_validated')); p.add_argument('--folds',type=int,default=5); p.add_argument('--permutations',type=int,default=100); p.add_argument('--bootstraps',type=int,default=500); p.add_argument('--seed',type=int,default=2026); return p.parse_args()

def model(seed): return Pipeline([('s',StandardScaler()),('c',LogisticRegression(max_iter=4000,class_weight='balanced',C=.25,solver='liblinear',random_state=seed))])
def block_mean(a,grid=8):
    if a.ndim==2:a=a[...,None]
    h,w,_=a.shape; ys=np.linspace(0,h,grid+1,dtype=int); xs=np.linspace(0,w,grid+1,dtype=int); out=[]
    for yi in range(grid):
      for xi in range(grid): out.extend(a[ys[yi]:ys[yi+1],xs[xi]:xs[xi+1]].mean((0,1)).tolist())
    return np.asarray(out,np.float32)

def residual_features(x):
    r=high_pass(x); g=grayscale(r); spec=np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(g))))
    enc=np.repeat(g[...,None],3,2)*.25+.5; sm=resize_np(np.clip(enc,0,1),(64,64))[...,0]; sm=(sm-.5)/.25
    d=dctn(sm,type=2,norm='ortho')[:16,:16]; d[0,0]=0
    return np.concatenate([block_mean(r),block_mean(np.abs(r)),block_mean(spec),d.ravel()]).astype(np.float32)

def channel_features(x,ch):
    a=rgb_to_ycbcr(x)[...,ch]; r=a-gaussian_filter(a,1.5,mode='reflect'); spec=np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(r))))
    return np.concatenate([block_mean(r),block_mean(np.abs(r)),block_mean(spec),[r.mean(),r.std(),np.mean(np.abs(r)),np.percentile(np.abs(r),90)]]).astype(np.float32)

def lsb_features(x):
    u=np.clip(x*255,0,255).round().astype(np.uint8); f=[]
    for c in range(3):
      for bit in range(2):
        p=((u[...,c]>>bit)&1).astype(np.float32); q=p.mean(); ent=-(q*np.log2(q+EPS)+(1-q)*np.log2(1-q+EPS)); f += [q,ent,np.mean(p[:,1:]==p[:,:-1]),np.mean(p[1:]==p[:-1])]
    return np.asarray(f,np.float32)

def dct_features(x):
    g=grayscale(x); h,w=g.shape; g=g[:h-h%8,:w-w%8]-.5; blocks=[]
    for y in range(0,g.shape[0],8):
      for z in range(0,g.shape[1],8): blocks.append(dctn(g[y:y+8,z:z+8],type=2,norm='ortho'))
    c=np.stack(blocks); coords=[(0,1),(1,0),(1,1),(0,2),(2,0),(1,2),(2,1),(2,2),(0,3),(3,0),(1,3),(3,1)]; f=[]
    for a,b in coords:
      v=c[:,a,b]; f += [v.mean(),v.std(),np.mean(np.abs(v)),np.mean(v>0)]
    return np.asarray(f,np.float32)

def spectral_signature(x):
    g=grayscale(high_pass(x)); m=np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(g)))); m=(m-m.mean())/(m.std()+EPS); sm=resize_np(np.repeat(np.clip(m/8+.5,0,1)[...,None],3,2),(64,64))[...,0]; v=((sm-.5)*8).ravel(); v-=v.mean(); return v/(np.linalg.norm(v)+EPS)
def pairwise(vs): return np.asarray([float(vs[i]@vs[j]) for i in range(len(vs)) for j in range(i+1,len(vs))])
def phase_top(xs):
    r=np.stack([grayscale(high_pass(x)) for x in xs]); z=np.fft.fft2(r,axes=(1,2)); coh=np.abs(np.mean(z/(np.abs(z)+EPS),0)); h,w=coh.shape; fy=np.fft.fftfreq(h)[:,None]; fx=np.fft.fftfreq(w)[None,:]; rad=np.sqrt(fx*fx+fy*fy)/.5; vals=coh[(rad>=.02)&(rad<=.95)]; k=max(1,math.ceil(.1*vals.size)); return float(np.partition(vals,-k)[-k:].mean())
def periodicity(x):
    r=grayscale(high_pass(x)); r-=r.mean(); c=fftconvolve(r,r[::-1,::-1],mode='full')/(np.sum(r*r)+EPS); cy,cx=np.array(c.shape)//2; exc=max(3,min(x.shape[:2])//64); mask=np.ones_like(c,dtype=bool); mask[cy-exc:cy+exc+1,cx-exc:cx+exc+1]=False; vals=np.abs(c[mask]); return float(np.percentile(vals,99.9)/(np.sqrt(np.mean(vals*vals))+EPS))
def cv_oof(pos,neg,fn,folds,seed):
    P=np.stack([fn(x) for x in pos]); N=np.stack([fn(x) for x in neg]); X=np.vstack([P,N]); y=np.r_[np.ones(len(P),int),np.zeros(len(N),int)]; sp=StratifiedKFold(n_splits=min(folds,len(pos)),shuffle=True,random_state=seed); pr=np.zeros(len(y))
    for tr,te in sp.split(X,y): m=model(seed); m.fit(X[tr],y[tr]); pr[te]=m.predict_proba(X[te])[:,1]
    return float(roc_auc_score(y,pr)),X,y

def perm_p(X,y,obs,folds,seed,count):
    rng=np.random.default_rng(seed); vals=[]
    for i in range(count):
      yp=rng.permutation(y); sp=StratifiedKFold(n_splits=folds,shuffle=True,random_state=seed+i+1); pr=np.zeros(len(y))
      for tr,te in sp.split(X,yp): m=model(seed+i+1); m.fit(X[tr],yp[tr]); pr[te]=m.predict_proba(X[te])[:,1]
      vals.append(roc_auc_score(yp,pr))
    return float((1+np.sum(np.asarray(vals)>=obs))/(1+count))
def transform(name,x):
    pil=to_pil(x); h,w=x.shape[:2]
    if name=='png': return png_roundtrip(x)
    if name=='jpeg95': return jpeg_roundtrip(x,95)
    if name=='jpeg75': return jpeg_roundtrip(x,75)
    if name=='resize':
      d=max(32,min(h,w)//2); return from_pil(pil.resize((d,d),Image.Resampling.BILINEAR).resize((w,h),Image.Resampling.BILINEAR))
    if name=='blur05': return from_pil(pil.filter(ImageFilter.GaussianBlur(.5)))
    if name=='brightness': return from_pil(ImageEnhance.Brightness(pil).enhance(1.05))
    if name=='flip': return from_pil(pil.transpose(Image.Transpose.FLIP_LEFT_RIGHT))
    raise ValueError(name)
def oof_survival(pos,neg,fn,folds,seed):
    names=['png','jpeg95','jpeg75','resize','blur05','brightness','flip']; ims=pos+neg; y=np.r_[np.ones(len(pos),int),np.zeros(len(neg),int)]; F=np.stack([fn(x) for x in ims]); sp=StratifiedKFold(n_splits=min(folds,len(pos)),shuffle=True,random_state=seed); base=np.zeros(len(y)); out={n:np.zeros(len(y)) for n in names}
    for tr,te in sp.split(F,y):
      m=model(seed); m.fit(F[tr],y[tr]); base[te]=m.predict_proba(F[te])[:,1]
      for n in names: out[n][te]=m.predict_proba(np.stack([fn(transform(n,ims[i])) for i in te]))[:,1]
    idx=np.where(y==1)[0]; bm=np.mean(base[idx]-.5)+EPS; return {n:float(np.mean(out[n][idx]-.5)/bm) for n in names}

def main():
    a=parse_args(); out=a.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True); src,clean=load_dataset(a.dataset); byres={}
    for im in clean.values(): byres.setdefault((im.shape[1],im.shape[0]),[]).append(im)
    rows=[]; rng=np.random.default_rng(a.seed); fns={'residual':residual_features,'Y':lambda x:channel_features(x,0),'Cb':lambda x:channel_features(x,1),'Cr':lambda x:channel_features(x,2),'LSB':lsb_features,'DCT':dct_features}
    for c in CATEGORIES:
      pos=src[c]; neg=byres[(pos[0].shape[1],pos[0].shape[0])]; r={'category':c,'resolution':f'{pos[0].shape[1]}x{pos[0].shape[0]}'}
      for n,fn in fns.items():
        auc,X,y=cv_oof(pos,neg,fn,a.folds,a.seed); r[f'{n}_auc']=auc
        if n in {'residual','Cb','LSB','DCT'}: r[f'{n}_perm_p']=perm_p(X,y,auc,a.folds,a.seed,a.permutations)
      pp=pairwise([spectral_signature(x) for x in pos]); nn=pairwise([spectral_signature(x) for x in neg]); r['spectral_consistency_wm']=float(pp.mean()); r['spectral_consistency_clean']=float(nn.mean()); r['spectral_consistency_delta']=float(pp.mean()-nn.mean()); boots=[float(rng.choice(pp,size=len(pp),replace=True).mean()) for _ in range(a.bootstraps)]; r['spectral_ci_low']=float(np.percentile(boots,2.5)); r['spectral_ci_high']=float(np.percentile(boots,97.5)); r['phase_top_wm']=phase_top(pos); r['phase_top_clean']=phase_top(neg[:len(pos)]); r['phase_top_delta']=r['phase_top_wm']-r['phase_top_clean']; p=np.asarray([periodicity(x) for x in pos]); q=np.asarray([periodicity(x) for x in neg]); r['periodicity_z']=float((p.mean()-q.mean())/(np.sqrt(.5*(p.var()+q.var()))+EPS)); r['transform_survival_residual']=oof_survival(pos,neg,residual_features,a.folds,a.seed); r['provenance_residual_auc']={n:cv_oof([transform(n,x) for x in pos],[transform(n,x) for x in neg],residual_features,a.folds,a.seed)[0] for n in ['png','jpeg95','jpeg75']}; rows.append(r); print(c,r)
    # OOF specificity matrix
    labels=CATEGORIES+['clean']; M=np.zeros((8,9))
    for ri,c in enumerate(CATEGORIES):
      pos=src[c]; neg=byres[(pos[0].shape[1],pos[0].shape[0])]; P=np.stack([residual_features(x) for x in pos]); N=np.stack([residual_features(x) for x in neg]); X=np.vstack([P,N]); y=np.r_[np.ones(len(P),int),np.zeros(len(N),int)]; sp=StratifiedKFold(n_splits=min(a.folds,len(pos)),shuffle=True,random_state=a.seed+ri); pr=np.zeros(len(y)); mods=[]
      for tr,te in sp.split(X,y): m=model(a.seed+ri); m.fit(X[tr],y[tr]); pr[te]=m.predict_proba(X[te])[:,1]; mods.append(m)
      M[ri,ri]=pr[:len(P)].mean(); M[ri,-1]=pr[len(P):].mean()
      for cj,o in enumerate(CATEGORIES):
        if o==c: continue
        F=np.stack([residual_features(x) for x in src[o]]); M[ri,cj]=np.mean([m.predict_proba(F)[:,1].mean() for m in mods])
    write_json(out/'validated_diagnostics.json',{'summary':rows,'specificity_matrix':{'rows':CATEGORIES,'columns':labels,'values':M.tolist()}})
    with (out/'validated_summary.csv').open('w',newline='') as h:
      keys=sorted({k for r in rows for k,v in r.items() if not isinstance(v,dict)}); w=csv.DictWriter(h,fieldnames=keys); w.writeheader(); [w.writerow({k:v for k,v in r.items() if not isinstance(v,dict)}) for r in rows]
    with (out/'oof_specificity_matrix.csv').open('w',newline='') as h:
      w=csv.writer(h); w.writerow(['detector']+labels); [w.writerow([c]+list(M[i])) for i,c in enumerate(CATEGORIES)]
if __name__=='__main__': main()
