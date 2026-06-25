#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
from common import load_dataset,high_pass,CATEGORIES
class DS(Dataset):
 def __init__(self,ims,y):self.ims=ims;self.y=y
 def __len__(self):return len(self.ims)
 def __getitem__(self,i):return torch.from_numpy(np.transpose(high_pass(self.ims[i]),(2,0,1)).astype(np.float32)),torch.tensor(self.y[i],dtype=torch.float32)
class ResidualCNN(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Conv2d(3,24,3,padding=1),nn.BatchNorm2d(24),nn.GELU(),nn.Conv2d(24,24,3,stride=2,padding=1),nn.GELU(),nn.Conv2d(24,48,3,padding=1),nn.BatchNorm2d(48),nn.GELU(),nn.Conv2d(48,48,3,stride=2,padding=1),nn.GELU(),nn.Conv2d(48,96,3,padding=1),nn.GELU(),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(96,1))
 def forward(self,x):return self.net(x).squeeze(1)
def args():
 p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--category',required=True,choices=CATEGORIES);p.add_argument('--output-dir',type=Path,default=Path('surrogates'));p.add_argument('--ensemble-size',type=int,default=5);p.add_argument('--epochs',type=int,default=40);p.add_argument('--batch-size',type=int,default=8);p.add_argument('--lr',type=float,default=1e-3);p.add_argument('--seed',type=int,default=2026);return p.parse_args()
def main():
 a=args();out=a.output_dir/a.category.lower();out.mkdir(parents=True,exist_ok=True);src,clean=load_dataset(a.dataset);pos=src[a.category];neg=[x for x in clean.values() if x.shape[:2]==pos[0].shape[:2]];ims=pos+neg;y=np.r_[np.ones(len(pos),np.float32),np.zeros(len(neg),np.float32)];dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');meta=[]
 for member in range(a.ensemble_size):
  seed=a.seed+member;tr,va=next(StratifiedShuffleSplit(n_splits=1,test_size=.25,random_state=seed).split(np.zeros(len(y)),y));m=ResidualCNN().to(dev);opt=torch.optim.AdamW(m.parameters(),lr=a.lr,weight_decay=1e-4);lossf=nn.BCEWithLogitsLoss();best=-1;state=None
  for e in range(a.epochs):
   m.train()
   for X,Y in DataLoader(DS([ims[i] for i in tr],y[tr]),batch_size=a.batch_size,shuffle=True):
    X,Y=X.to(dev),Y.to(dev);opt.zero_grad(set_to_none=True);loss=lossf(m(X),Y);loss.backward();opt.step()
   m.eval();ys=[];ps=[]
   with torch.no_grad():
    for X,Y in DataLoader(DS([ims[i] for i in va],y[va]),batch_size=a.batch_size):ps += torch.sigmoid(m(X.to(dev))).cpu().tolist();ys += Y.tolist()
   auc=roc_auc_score(ys,ps);print(a.category,member,e+1,auc)
   if auc>best:best=auc;state={k:v.detach().cpu() for k,v in m.state_dict().items()}
  path=out/f'detector_{member}.pt';torch.save({'state_dict':state,'val_auc':best,'seed':seed,'category':a.category},path);meta.append({'path':str(path),'val_auc':best,'seed':seed})
 (out/'metadata.json').write_text(json.dumps(meta,indent=2))
if __name__=='__main__':main()
