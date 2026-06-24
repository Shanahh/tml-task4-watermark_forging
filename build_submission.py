#!/usr/bin/env python3
from __future__ import annotations
import argparse,shutil,zipfile
from pathlib import Path
from common import load_dataset,save_rgb
def args():
 p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--specialized',type=Path);p.add_argument('--wm3',type=Path);p.add_argument('--output-dir',type=Path,default=Path('final_submission'));p.add_argument('--zip',type=Path,default=Path('final_submission.zip'));return p.parse_args()
def main():
 a=args();_,clean=load_dataset(a.dataset);a.output_dir.mkdir(parents=True,exist_ok=True)
 for i,x in clean.items():
  src=None
  if 51<=i<=75 and a.wm3:src=a.wm3/f'{i}.png'
  elif ((1<=i<=25) or (76<=i<=150)) and a.specialized:src=a.specialized/f'{i}.png'
  dst=a.output_dir/f'{i}.png';shutil.copy2(src,dst) if src and src.exists() else save_rgb(x,dst)
 with zipfile.ZipFile(a.zip,'w',zipfile.ZIP_DEFLATED) as z:
  for i in range(1,201):z.write(a.output_dir/f'{i}.png',arcname=f'{i}.png')
 print('saved',a.zip)
if __name__=='__main__':main()
