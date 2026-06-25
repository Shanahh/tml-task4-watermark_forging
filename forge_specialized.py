#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from scipy.fft import dctn,idctn
from scipy.ndimage import gaussian_filter
from common import *

def parse_args():
 p=argparse.ArgumentParser(); p.add_argument('--dataset',type=Path,required=True); p.add_argument('--output-dir',type=Path,default=Path('specialized_candidates')); p.add_argument('--strength-grid',default='0.0025,0.005,0.01,0.02'); p.add_argument('--wm4-threshold',type=float,default=.45); p.add_argument('--wm6-coeff-count',type=int,default=8); return p.parse_args()
def channel_template(xs,ch):
 rs=[]
 for x in xs:
  c=rgb_to_ycbcr(x)[...,ch]; rs.append(c-gaussian_filter(c,1.5,mode='reflect'))
 t=np.median(np.stack(rs),0); return (t-t.mean())/(t.std()+EPS)
def apply_channel(x,t,ch,s):
 y=rgb_to_ycbcr(x); y[...,ch]=np.clip(y[...,ch]+s*t,0,1); return ycbcr_to_rgb(y)
def phase_template(xs,thr):
 rs=[]
 for x in xs:
  g=grayscale(x); rs.append(g-gaussian_filter(g,1.5,mode='reflect'))
 z=np.fft.fft2(np.stack(rs),axes=(1,2)); unit=z/(np.abs(z)+EPS); mu=unit.mean(0); coh=np.abs(mu); ph=np.angle(mu); mag=np.median(np.abs(z),0); mask=coh>=thr; mask[0,0]=False; t=np.fft.ifft2(mask*mag*np.exp(1j*ph)).real; return (t-t.mean())/(t.std()+EPS)
def apply_luma(x,t,s):
 y=rgb_to_ycbcr(x); y[...,0]=np.clip(y[...,0]+s*t,0,1); return ycbcr_to_rgb(y)
def lsb_template(xs,ch):
 bits=[]
 for x in xs:
  c=np.round(rgb_to_ycbcr(x)[...,ch]*255).astype(np.uint8); bits.append(c&1)
 return (np.mean(np.stack(bits),0)>=0.5).astype(np.uint8)
def apply_lsb(x,template,ch):
 y=rgb_to_ycbcr(x); c=np.round(y[...,ch]*255).astype(np.uint8); c=(c&0xFE)|template; y[...,ch]=c.astype(np.float32)/255.0; return ycbcr_to_rgb(y)
COORDS=[(0,1),(1,0),(1,1),(0,2),(2,0),(1,2),(2,1),(2,2),(0,3),(3,0),(1,3),(3,1),(2,3),(3,2),(3,3)]
def stats(xs):
 d={c:[] for c in COORDS}
 for x in xs:
  g=grayscale(x); h,w=g.shape; g=g[:h-h%8,:w-w%8]-.5
  for y in range(0,g.shape[0],8):
   for z in range(0,g.shape[1],8):
    c=dctn(g[y:y+8,z:z+8],type=2,norm='ortho')
    for q in COORDS:d[q].append(c[q])
 return {q:{'mean':float(np.mean(v)),'std':float(np.std(v)+EPS),'sign':float(np.mean(np.asarray(v)>0))} for q,v in d.items()}
def select(ws,cs,n):
 sc=[]
 for q in COORDS:
  a,b=ws[q],cs[q]; sc.append((abs(a['mean']-b['mean'])/b['std']+abs(a['std']-b['std'])/b['std']+abs(a['sign']-b['sign']),q))
 return [q for _,q in sorted(sc,reverse=True)[:n]]
def apply_dct(x,sel,ws,cs,s):
 ycc=rgb_to_ycbcr(x); y=ycc[...,0].copy(); h,w=y.shape; out=y.copy()
 for by in range(0,h-h%8,8):
  for bx in range(0,w-w%8,8):
   c=dctn(y[by:by+8,bx:bx+8]-.5,type=2,norm='ortho')
   for q in sel:
    z=(c[q]-cs[q]['mean'])/cs[q]['std']; target=ws[q]['mean']+z*ws[q]['std']; c[q]=(1-s)*c[q]+s*target
   out[by:by+8,bx:bx+8]=idctn(c,type=2,norm='ortho')+.5
 ycc[...,0]=np.clip(out,0,1); return ycbcr_to_rgb(ycc)
def main():
 a=parse_args(); src,clean=load_dataset(a.dataset); byres={}
 for im in clean.values():byres.setdefault((im.shape[1],im.shape[0]),[]).append(im)
 t1=channel_template(src['WM_1'],1); t4=phase_template(src['WM_4'],a.wm4_threshold); ws=stats(src['WM_6']); res=(src['WM_6'][0].shape[1],src['WM_6'][0].shape[0]); cs=stats(byres[res]); sel=select(ws,cs,a.wm6_coeff_count); print('WM6 DCT coeffs',sel)
 t5b_lsb=lsb_template(src['WM_5'],1); t5r_lsb=lsb_template(src['WM_5'],2)
 for s in [float(v) for v in a.strength_grid.split(',')]:
  out=a.output_dir/f'strength_{s:g}'; out.mkdir(parents=True,exist_ok=True)
  for i,x in clean.items():
   if 1<=i<=25:y=apply_channel(x,t1,1,s)
   elif 76<=i<=100:y=apply_luma(x,t4,s)
   elif 101<=i<=125:y=apply_lsb(apply_lsb(x,t5b_lsb,1),t5r_lsb,2)
   elif 126<=i<=150:y=apply_dct(x,sel,ws,cs,min(1,s*25))
   else:y=x
   save_rgb(y,out/f'{i}.png')
  print('saved',out)
if __name__=='__main__':main()
