#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from common import load_dataset,save_rgb
from train_wm3_surrogate import ResidualCNN
def args():
 p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--models',type=Path,required=True);p.add_argument('--output-dir',type=Path,default=Path('wm3_candidates'));p.add_argument('--eps-grid',default='0.0039215686,0.0078431373,0.011764706');p.add_argument('--steps',type=int,default=50);p.add_argument('--step-size',type=float,default=.0009803922);p.add_argument('--mse-weight',type=float,default=10.0);p.add_argument('--tv-weight',type=float,default=.05);return p.parse_args()
def hp(x):
 k=torch.tensor([1,4,6,4,1],dtype=x.dtype,device=x.device);k=k/k.sum();w=(k[:,None]*k[None,:]).expand(x.shape[1],1,5,5);return x-F.conv2d(x,w,padding=2,groups=x.shape[1])
def tv(d):return torch.mean(torch.abs(d[:,:,1:]-d[:,:,:-1]))+torch.mean(torch.abs(d[:,:,:,1:]-d[:,:,:,:-1]))
def load_models(d,dev):
 ms=[]
 for p in sorted(d.glob('wm3_detector_*.pt')):
  q=torch.load(p,map_location='cpu');m=ResidualCNN().to(dev);m.load_state_dict(q['state_dict']);m.eval();[z.requires_grad_(False) for z in m.parameters()];ms.append(m)
 if not ms:raise FileNotFoundError(d)
 return ms
def optimize(im,ms,eps,a,dev):
 o=torch.from_numpy(np.transpose(im,(2,0,1))).unsqueeze(0).float().to(dev);x=o.clone().detach().requires_grad_(True)
 for _ in range(a.steps):
  logit=torch.stack([m(hp(x)) for m in ms]).mean();d=x-o;loss=F.binary_cross_entropy_with_logits(logit,torch.ones_like(logit))+a.mse_weight*torch.mean(d*d)+a.tv_weight*tv(d);g=torch.autograd.grad(loss,x)[0];x=(x.detach()-a.step_size*g.sign());x=torch.max(torch.min(x,o+eps),o-eps).clamp(0,1).requires_grad_(True)
 return np.transpose(x.detach().cpu().numpy()[0],(1,2,0))
def main():
 a=args();dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');ms=load_models(a.models,dev);_,clean=load_dataset(a.dataset)
 for eps in [float(v) for v in a.eps_grid.split(',')]:
  out=a.output_dir/f'eps_{eps:.6f}';out.mkdir(parents=True,exist_ok=True)
  for i,x in clean.items():save_rgb(optimize(x,ms,eps,a,dev) if 51<=i<=75 else x,out/f'{i}.png')
  print('saved',out)
if __name__=='__main__':main()
