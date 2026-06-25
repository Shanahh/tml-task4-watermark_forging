#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,shutil,zipfile
from pathlib import Path
from common import load_dataset,save_rgb,category_for_id
def args():
 p=argparse.ArgumentParser()
 p.add_argument('--dataset',type=Path,required=True)
 p.add_argument('--routing',type=Path,required=True,help='JSON file mapping category (WM_1..WM_8) to a candidate directory containing <id>.png files')
 p.add_argument('--output-dir',type=Path,default=Path('final_submission'))
 p.add_argument('--zip',type=Path,default=Path('final_submission.zip'))
 return p.parse_args()
def main():
 a=args();_,clean=load_dataset(a.dataset);a.output_dir.mkdir(parents=True,exist_ok=True)
 routing={k:Path(v) for k,v in json.loads(a.routing.read_text()).items()}
 for i,x in clean.items():
  cat=category_for_id(i);d=routing.get(cat);src=d/f'{i}.png' if d else None
  dst=a.output_dir/f'{i}.png';shutil.copy2(src,dst) if src and src.exists() else save_rgb(x,dst)
 with zipfile.ZipFile(a.zip,'w',zipfile.ZIP_DEFLATED) as z:
  for i in range(1,201):z.write(a.output_dir/f'{i}.png',arcname=f'{i}.png')
 print('saved',a.zip)
if __name__=='__main__':main()
