#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json, math, re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dctn
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CATEGORIES = {f"WM_{i}": None for i in range(1, 9)}

@dataclass
class Result:
    category: str
    resolution: str
    spatial_consistency: float
    phase_coherence_mean: float
    phase_coherence_top10pct: float
    resize_consistency: float
    crop_consistency: float
    residual_auc_mean: float
    residual_auc_std: float
    residual_accuracy_mean: float
    rgb_auc_mean: float
    rgb_auc_std: float
    rgb_accuracy_mean: float
    recommendation: str
    confidence: str
    notes: str

def args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", type=Path, default=Path("."))
    p.add_argument("--output-json", type=Path, default=Path("watermark_diagnostics.json"))
    p.add_argument("--output-csv", type=Path, default=Path("watermark_diagnostics.csv"))
    p.add_argument("--blur-sigma", type=float, default=1.5)
    p.add_argument("--signature-size", type=int, default=64)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()

def num(path: Path) -> int:
    m = re.search(r"(\d+)$", path.stem)
    if not m:
        raise ValueError(path)
    return int(m.group(1))

def pngs(d: Path): return sorted(d.glob("*.png"), key=num)

def load(path: Path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), np.float32) / 255.0

def size(path: Path):
    with Image.open(path) as im:
        return im.size

def resize(x, wh):
    im = Image.fromarray(np.clip(x * 255, 0, 255).astype(np.uint8))
    return np.asarray(im.resize(wh, Image.Resampling.BILINEAR), np.float32) / 255.0

def hp(x, sigma): return x - gaussian_filter(x, sigma=(sigma, sigma, 0), mode="reflect")

def unit(v):
    v = v.astype(np.float64).reshape(-1)
    v -= v.mean()
    n = np.linalg.norm(v)
    return np.zeros_like(v) if n < 1e-12 else v / n

def signature(x, sigma, s):
    r = hp(x, sigma)
    y = resize(np.clip(r * 4 + .5, 0, 1), (s, s))
    return unit((y - .5) / 4)

def pairwise(sigs):
    m = np.stack(sigs)
    sim = m @ m.T
    return float(sim[np.triu_indices(len(m), 1)].mean())

def phase_stats(images, sigma):
    r = np.stack([hp(x, sigma) for x in images])
    g = .2126*r[...,0] + .7152*r[...,1] + .0722*r[...,2]
    z = np.fft.fft2(g, axes=(1,2))
    u = z / (np.abs(z) + 1e-8)
    c = np.abs(u.mean(0))
    h,w = c.shape
    fy,fx = np.fft.fftfreq(h)[:,None], np.fft.fftfreq(w)[None,:]
    rad = np.sqrt(fx*fx+fy*fy)/.5
    vals = c[(rad>=.02)&(rad<=.95)]
    k=max(1, math.ceil(vals.size*.1))
    return float(vals.mean()), float(np.partition(vals,-k)[-k:].mean())

def resize_consistency(images, sigma, s):
    out=[]
    for x in images:
        h,w=x.shape[:2]; d=max(32,min(h,w)//2)
        xr=resize(resize(x,(d,d)),(w,h))
        out.append(float(signature(x,sigma,s) @ signature(xr,sigma,s)))
    return float(np.mean(out))

def crop_consistency(images, sigma, s):
    out=[]
    for x in images:
        h,w=x.shape[:2]; ch,cw=max(32,int(.75*h)),max(32,int(.75*w))
        y0,x0=(h-ch)//2,(w-cw)//2
        crop=x[y0:y0+ch,x0:x0+cw]
        out.append(float(signature(x,sigma,s) @ signature(crop,sigma,s)))
    return float(np.mean(out))

def block_mean(a, grid=8):
    if a.ndim==2: a=a[...,None]
    h,w,c=a.shape
    ys=np.linspace(0,h,grid+1,dtype=int); xs=np.linspace(0,w,grid+1,dtype=int)
    f=[]
    for yi in range(grid):
        for xi in range(grid):
            f.extend(a[ys[yi]:ys[yi+1],xs[xi]:xs[xi+1]].mean((0,1)).tolist())
    return np.asarray(f,np.float32)

def residual_features(x,sigma):
    r=hp(x,sigma)
    g=.2126*r[...,0]+.7152*r[...,1]+.0722*r[...,2]
    spatial=block_mean(r)
    spatial_abs=block_mean(np.abs(r))
    spec=np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(g))))
    fourier=block_mean(spec)
    rg=resize(np.repeat(g[...,None],3,2)*.25+.5,(64,64))[...,0]
    rg=(rg-.5)/.25
    d=dctn(rg,type=2,norm="ortho")[:16,:16]; d[0,0]=0
    flat=r.reshape(-1,3)
    stats=np.concatenate([flat.mean(0),flat.std(0),
        np.percentile(flat,[5,25,50,75,95],axis=0).reshape(-1),
        [np.mean(np.abs(g)),np.std(g),np.percentile(np.abs(g),90),np.percentile(np.abs(g),99)]])
    return np.concatenate([spatial,spatial_abs,fourier,d.reshape(-1),stats]).astype(np.float32)

def rgb_features(x):
    pooled=block_mean(resize(x,(32,32)),grid=4)
    flat=x.reshape(-1,3)
    stats=np.concatenate([flat.mean(0),flat.std(0),
        np.percentile(flat,[5,25,50,75,95],axis=0).reshape(-1)])
    return np.concatenate([pooled,stats]).astype(np.float32)

def cv(pos,neg,folds,seed):
    X=np.concatenate([pos,neg]); y=np.r_[np.ones(len(pos),int),np.zeros(len(neg),int)]
    n=min(folds,int(min((y==0).sum(),(y==1).sum())))
    splitter=StratifiedKFold(n_splits=n,shuffle=True,random_state=seed)
    aucs=[]; accs=[]
    for tr,te in splitter.split(X,y):
        model=Pipeline([("s",StandardScaler()),
            ("c",LogisticRegression(max_iter=3000,class_weight="balanced",
             C=.25,solver="liblinear",random_state=seed))])
        model.fit(X[tr],y[tr]); p=model.predict_proba(X[te])[:,1]
        aucs.append(roc_auc_score(y[te],p)); accs.append(accuracy_score(y[te],p>=.5))
    return float(np.mean(aucs)),float(np.std(aucs)),float(np.mean(accs))

def choose(spatial,phase,res_auc,rgb_auc,resize_s,crop_s):
    gap=rgb_auc-res_auc
    if spatial>=.20:
        rec="robust_spatial_template"; conf="high" if spatial>=.30 else "medium"
        note="Strong cross-image residual alignment; start with a native-resolution robust median or trimmed-mean template."
    elif phase>=.45:
        rec="phase_coherent_spectral_template"; conf="high" if phase>=.60 else "medium"
        note="Fourier phases align more than spatial residuals; use a phase-coherent spectral template."
    elif res_auc>=.80 and gap<=.12:
        rec="residual_surrogate_detector_pgd"; conf="high" if res_auc>=.90 else "medium"
        note="Residual-domain classification generalizes; use an ensemble surrogate with tight LPIPS/L-infinity constraints."
    else:
        rec="content_adaptive_one_shot_or_learned_residual"; conf="low"
        note="No stable fixed pattern or trustworthy residual detector was found."
    if resize_s<.25: note+=" Resize-sensitive; avoid resizing."
    elif resize_s>.70: note+=" Relatively resize-stable."
    if crop_s<.20: note+=" Likely global; avoid crops."
    elif crop_s>.60: note+=" Possibly local or patch-stable."
    if gap>.15: note+=" RGB is much stronger than residual classification, suggesting content leakage."
    return rec,conf,note

def analyze(cat, root, sigma, sig_size, folds, seed):
    spaths=pngs(root/"watermarked_sources"/cat)
    if len(spaths)!=25: raise RuntimeError(f"{cat}: expected 25 sources, found {len(spaths)}")
    sizes=Counter(size(p) for p in spaths)
    if len(sizes)!=1: raise RuntimeError(f"{cat}: mixed source sizes {sizes}")
    resolution=next(iter(sizes))
    npaths=[p for p in pngs(root/"clean_targets") if size(p)==resolution]
    if len(npaths)<25: raise RuntimeError(f"{cat}: not enough same-resolution clean negatives")
    src=[load(p) for p in spaths]; neg=[load(p) for p in npaths]
    spatial=pairwise([signature(x,sigma,sig_size) for x in src])
    pmean,ptop=phase_stats(src,sigma)
    rs=resize_consistency(src,sigma,sig_size); cs=crop_consistency(src,sigma,sig_size)
    pra=np.stack([residual_features(x,sigma) for x in src])
    nra=np.stack([residual_features(x,sigma) for x in neg])
    prgb=np.stack([rgb_features(x) for x in src]); nrgb=np.stack([rgb_features(x) for x in neg])
    rauc,rstd,racc=cv(pra,nra,folds,seed); gauc,gstd,gacc=cv(prgb,nrgb,folds,seed)
    rec,conf,note=choose(spatial,ptop,rauc,gauc,rs,cs)
    return Result(cat,f"{resolution[0]}x{resolution[1]}",spatial,pmean,ptop,rs,cs,
                  rauc,rstd,racc,gauc,gstd,gacc,rec,conf,note)

def print_table(results):
    head=["Cat","Res","Spatial","PhaseTop","Resize","Crop","ResAUC","RGBAUC","Recommendation"]
    rows=[[r.category,r.resolution,f"{r.spatial_consistency:.3f}",
           f"{r.phase_coherence_top10pct:.3f}",f"{r.resize_consistency:.3f}",
           f"{r.crop_consistency:.3f}",f"{r.residual_auc_mean:.3f}",
           f"{r.rgb_auc_mean:.3f}",r.recommendation] for r in results]
    widths=[max(len(head[i]),max(len(row[i]) for row in rows)) for i in range(len(head))]
    line="  ".join(head[i].ljust(widths[i]) for i in range(len(head)))
    print(line); print("-"*len(line))
    for row in rows: print("  ".join(row[i].ljust(widths[i]) for i in range(len(row))))

def main():
    a=args(); root=a.dataset.resolve()
    if not (root/"clean_targets").is_dir() or not (root/"watermarked_sources").is_dir():
        raise FileNotFoundError("Dataset must contain clean_targets and watermarked_sources")
    results=[]
    for cat in CATEGORIES:
        print(f"Analyzing {cat}...")
        results.append(analyze(cat,root,a.blur_sigma,a.signature_size,a.cv_folds,a.seed))
    print(); print_table(results)
    a.output_json.write_text(json.dumps([asdict(r) for r in results],indent=2))
    with a.output_csv.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(asdict(results[0]).keys()))
        w.writeheader(); [w.writerow(asdict(r)) for r in results]
    print(f"\nSaved JSON: {a.output_json}\nSaved CSV:  {a.output_csv}")
    for r in results:
        print(f"\n{r.category} [{r.resolution}] {r.recommendation} ({r.confidence})\n  {r.notes}")

if __name__=="__main__":
    main()
