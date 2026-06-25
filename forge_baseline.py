#!/usr/bin/env python3
"""Generic naive baseline: average the high-pass residual across a watermark
group's 25 source images and additively transfer that mean residual onto the
matching clean targets. This is the 'simple averaging' idea from Yang et al.
(NeurIPS 2024) applied in the forging direction: averaging over many images
carrying the same watermark message isolates the (content-independent) common
signal, which can then be re-applied to new clean images.

Acts as the fallback attack for any category with no validated category-specific
signal (WM_2/7/8), and as the ablation baseline for categories that do have a
specialized attack (WM_1/3/4/5/6), per the task's required ablation strategy.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from common import load_dataset,save_rgb,high_pass,rgb_to_ycbcr,ycbcr_to_rgb,CATEGORY_RANGES,CATEGORIES
def parse_args():
 p=argparse.ArgumentParser()
 p.add_argument('--dataset',type=Path,required=True)
 p.add_argument('--output-dir',type=Path,default=Path('baseline_candidates'))
 p.add_argument('--strength-grid',default='0.01,0.02,0.04,0.08')
 p.add_argument('--categories',default=','.join(CATEGORIES))
 return p.parse_args()
def mean_residual_template(xs):
 r=np.mean(np.stack([high_pass(rgb_to_ycbcr(x)) for x in xs]),0)
 return r/(np.std(r)+1e-8)
def apply_template(x,t,s):
 y=rgb_to_ycbcr(x); y[...,0]=np.clip(y[...,0]+s*t[...,0],0,1); y[...,1]=np.clip(y[...,1]+s*t[...,1],0,1); y[...,2]=np.clip(y[...,2]+s*t[...,2],0,1)
 return ycbcr_to_rgb(y)
def main():
 a=parse_args(); src,clean=load_dataset(a.dataset)
 cats=[c.strip() for c in a.categories.split(',')]
 templates={c:mean_residual_template(src[c]) for c in cats}
 for s in [float(v) for v in a.strength_grid.split(',')]:
  out=a.output_dir/f'strength_{s:g}'; out.mkdir(parents=True,exist_ok=True)
  for i,x in clean.items():
   cat=next((c for c,(lo,hi) in CATEGORY_RANGES.items() if lo<=i<=hi),None)
   y=apply_template(x,templates[cat],s) if cat in templates else x
   save_rgb(y,out/f'{i}.png')
  print('saved',out)
if __name__=='__main__':main()
