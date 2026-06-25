from __future__ import annotations
import io,re,json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

CATEGORIES=[f'WM_{i}' for i in range(1,9)]
CATEGORY_RANGES={'WM_1':(1,25),'WM_2':(26,50),'WM_3':(51,75),'WM_4':(76,100),'WM_5':(101,125),'WM_6':(126,150),'WM_7':(151,175),'WM_8':(176,200)}
EPS=1e-8

def numeric_suffix(p:Path)->int:
    m=re.search(r'(\d+)$',p.stem)
    if not m: raise ValueError(p)
    return int(m.group(1))

def sorted_pngs(d:Path): return sorted(d.glob('*.png'),key=numeric_suffix)

def load_rgb(p:Path):
    with Image.open(p) as im: return np.asarray(im.convert('RGB'),np.float32)/255.0

def save_rgb(x,p:Path):
    p.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(np.clip(x*255,0,255).round().astype(np.uint8)).save(p)

def to_pil(x): return Image.fromarray(np.clip(x*255,0,255).round().astype(np.uint8))
def from_pil(im): return np.asarray(im.convert('RGB'),np.float32)/255.0

def resize_np(x,size): return from_pil(to_pil(x).resize(size,Image.Resampling.BILINEAR))
def grayscale(x): return .2126*x[...,0]+.7152*x[...,1]+.0722*x[...,2]
def high_pass(x,sigma=1.5): return x-gaussian_filter(x,sigma=(sigma,sigma,0),mode='reflect')

def rgb_to_ycbcr(x):
    r,g,b=x[...,0],x[...,1],x[...,2]
    y=.299*r+.587*g+.114*b
    cb=-.168736*r-.331264*g+.5*b+.5
    cr=.5*r-.418688*g-.081312*b+.5
    return np.stack([y,cb,cr],-1)

def ycbcr_to_rgb(x):
    y=x[...,0]; cb=x[...,1]-.5; cr=x[...,2]-.5
    r=y+1.402*cr; g=y-.344136*cb-.714136*cr; b=y+1.772*cb
    return np.clip(np.stack([r,g,b],-1),0,1)

def png_roundtrip(x):
    b=io.BytesIO(); to_pil(x).save(b,format='PNG'); b.seek(0); return from_pil(Image.open(b))
def jpeg_roundtrip(x,q=95):
    b=io.BytesIO(); to_pil(x).save(b,format='JPEG',quality=q); b.seek(0); return from_pil(Image.open(b))

def load_dataset(root:Path):
    root=root.resolve(); sr=root/'watermarked_sources'; cr=root/'clean_targets'
    if not sr.is_dir() or not cr.is_dir(): raise FileNotFoundError('dataset needs clean_targets/ and watermarked_sources/')
    src={c:[load_rgb(p) for p in sorted_pngs(sr/c)] for c in CATEGORIES}
    clean={numeric_suffix(p):load_rgb(p) for p in sorted_pngs(cr)}
    return src,clean

def write_json(p:Path,obj): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(obj,indent=2))

def category_for_id(i:int)->str:
    for c,(lo,hi) in CATEGORY_RANGES.items():
        if lo<=i<=hi: return c
    raise ValueError(i)
