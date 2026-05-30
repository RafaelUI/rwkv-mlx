import sys, os, time, math
sys.path.insert(0, os.getcwd())
import mlx.core as mx, mlx.nn as nn
import mlx.optimizers as optim
from mlx.nn.utils import checkpoint as nn_ckpt
from config import RWKVConfig
from model.rwkv7_x070 import RWKV7X070
from model.world_tokenizer import WorldTokenizer
from model.lora import add_lora

mx.set_cache_limit(int(1.5e9)); GB=1e9

cfg = RWKVConfig(n_layer=24, n_embd=2048, vocab_size=65536, head_size=64, ctx_len=256, batch_size=1)
m = RWKV7X070(cfg)
m.load_weights("weights/rwkv7_1.5B_x070.safetensors"); mx.eval(m.parameters())

tok = WorldTokenizer("tokenizer/rwkv_vocab_v20230424.txt")
with open("cleaned.txt","rb") as f: raw=f.read(80000)
ids = tok.encode(raw.decode("utf-8",errors="ignore")[300:])
W=256
xs=[mx.array(ids[i*W:(i+1)*W]).reshape(1,W) for i in range(6)]
ys=[mx.array(ids[i*W+1:(i+1)*W+1]).reshape(1,W) for i in range(6)]

def lf(model, x, y):
    h=model.ln0(model.emb(x)); vf=None
    for blk in model.blocks: h,vf=nn_ckpt(blk)(h,vf)
    lo=model.head(model.ln_out(h)); B,T,V=lo.shape
    return nn.losses.cross_entropy(lo.reshape(B*T,V), y.reshape(B*T)).mean().astype(mx.float32)

base=float(lf(m,xs[0],ys[0])); mx.eval(base)
print(f"baseline (bf16, реальный русский): {base:.4f}")

# QLoRA: квантуем ТОЛЬКО большие замороженные матрицы (FFN/head/emb); low-rank остаются bf16
BIG=("cmix.key","cmix.value","head","emb")
def pred(p,mod): return hasattr(mod,"to_quantized") and any(p.endswith(b) for b in BIG)
nn.quantize(m, group_size=64, bits=4, class_predicate=pred); mx.eval(m.parameters())
m, info = add_lora(m, rank=16, alpha=16.0, quantize_base=4, layers=range(12,24)); mx.eval(m.parameters())
q0=float(lf(m,xs[0],ys[0])); mx.eval(q0)
print(f"baseline (4-бит база, до обучения): {q0:.4f} | обучаемо {info['trainable_params']/1e6:.2f}M")

opt=optim.AdamW(learning_rate=1e-4)
def step(x,y):
    l,g=nn.value_and_grad(m,lf)(m,x,y); g,_=optim.clip_grad_norm(g,1.0); opt.update(m,g); return l
print("overfit реальных окон (верхние 12 слоёв):")
t0=time.time(); seen=0; l=q0
for s in range(61):
    x,y=xs[s%6], ys[s%6]
    l=step(x,y); mx.eval(l,m.state,opt.state); seen+=W
    if s%10==0:
        dt=time.time()-t0
        print(f"  step {s:3d} | loss {float(l):.4f} | {seen/dt:.0f} tok/s | peak {mx.get_peak_memory()/GB:.2f} GB", flush=True)
        t0=time.time(); seen=0
print(f"\nИтог: 4-бит {q0:.3f} -> {float(l):.3f}. Падение = весь продакшен-стек учится на реальной World 1.5B.")
