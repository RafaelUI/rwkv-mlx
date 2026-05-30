import sys, os
sys.path.insert(0, os.getcwd())
import mlx.core as mx, mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
from config import RWKVConfig
from model.rwkv7 import RWKV7, init_weights
from model.lora import (add_lora, merge_lora, save_lora, load_lora,
                        LoRALinear, lora_state)

def to_bf16(m):
    m.update(tree_map(lambda x: x.astype(mx.bfloat16) if isinstance(x, mx.array) else x,
                      m.parameters()))
    return m

mx.random.seed(0)
cfg = RWKVConfig(n_layer=2, n_embd=128, vocab_size=512, ctx_len=64, batch_size=2)
m = to_bf16(init_weights(RWKV7(cfg)))
x = mx.random.randint(0, 512, (2, 64)); y = mx.random.randint(0, 512, (2, 64))
out_before = m(x); mx.eval(out_before)

m, info = add_lora(m, rank=8, alpha=16.0)
print(f"[1] trainable {info['trainable_params']/1e3:.1f}K / {info['total_params']/1e6:.2f}M "
      f"= {info['trainable_pct']:.3f}%  adapters/blk={info['num_adapters']}")

out_after = m(x); mx.eval(out_after)
d = float(mx.abs(out_before - out_after).max())
print(f"[2] forward diff at init (~0): {d:.2e}  {'OK' if d < 1e-3 else 'FAIL'}")

def lf(model, x, y): return model.loss(x, y).astype(mx.float32)
loss, grads = nn.value_and_grad(m, lf)(m, x, y)
mx.eval(loss, grads)
gd = dict(tree_flatten(grads)); keys = list(gd.keys())
only_lora = all(("lora_a" in k or "lora_b" in k) for k in keys)
print(f"[3] grad keys={len(keys)}, только lora_*: {only_lora}  {'OK' if only_lora else 'FAIL'}")
# При B=0 ∂L/∂A=0 (корректно для LoRA), сигнал идёт в lora_b → проверяем именно его
ok3 = True
for proj in ("r_proj", "k_proj", "v_proj", "o_proj"):
    gk = [k for k in keys if proj in k and "lora_b" in k][0]
    gn = float(mx.linalg.norm(gd[gk].astype(mx.float32)))
    flow = "через WKV backward" if proj in ("r_proj","k_proj","v_proj") else "после WKV"
    good = gn > 0; ok3 &= good
    print(f"    {proj:7s} lora_b |g|={gn:.4e} ({flow})  {'OK' if good else 'FAIL(0)'}")
print(f"    → градиент достигает всех адаптеров: {'OK' if ok3 else 'FAIL'}")

b0 = mx.array(m.blocks[0].tmix.r_proj.linear.weight)
a0 = mx.array(m.blocks[0].tmix.r_proj.lora_b)
opt = optim.AdamW(learning_rate=1e-3)
opt.update(m, grads); mx.eval(m.parameters(), opt.state)
db = float(mx.abs(b0 - m.blocks[0].tmix.r_proj.linear.weight).max())
da = float(mx.abs(a0 - m.blocks[0].tmix.r_proj.lora_b).max())
print(f"[4] база Δ={db:.2e} (=0), адаптер Δ={da:.2e} (>0)  {'OK' if db<1e-6 and da>0 else 'FAIL'}")

# save/load: сравниваем именно тензоры адаптера (база может отличаться)
save_lora(m, "/tmp/lora_test.safetensors")
m2, _ = add_lora(to_bf16(init_weights(RWKV7(cfg))), rank=8, alpha=16.0)
m2 = load_lora(m2, "/tmp/lora_test.safetensors")
s1, s2 = lora_state(m), lora_state(m2)
maxd = max(float(mx.abs(s1[k].astype(mx.float32) - s2[k].astype(mx.float32)).max())
           for k in s1)
print(f"[5] save/load адаптеров max diff: {maxd:.2e}  {'OK' if maxd == 0 else 'FAIL'}")

o_pre = m(x); mx.eval(o_pre)
m = merge_lora(m)
is_plain = type(m.blocks[0].tmix.r_proj) is nn.Linear
o_post = m(x); mx.eval(o_post)
d = float(mx.abs(o_pre - o_post).max())
print(f"[6] merge: plain_linear={is_plain}, diff={d:.2e}  {'OK' if is_plain and d<5e-2 else 'FAIL'}")
print("DONE")
