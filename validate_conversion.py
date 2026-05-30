import sys, os, math
sys.path.insert(0, os.getcwd())
import mlx.core as mx
from model.world_tokenizer import WorldTokenizer
from convert_rwkv_pth import build_and_load

PTH = "weights/RWKV-x070-World-1.5B-v3-20250127-ctx4096.pth"
tok = WorldTokenizer("tokenizer/rwkv_vocab_v20230424.txt")

# реальный русский текст из cleaned.txt
with open("cleaned.txt", "rb") as f:
    raw = f.read(20000)
text = raw.decode("utf-8", errors="ignore")
# чистый кусок без обрезанных краёв
text = text[200:200+3000]
print("образец текста:", repr(text[:120]))
ids = tok.encode(text)
print(f"токенов: {len(ids)} | round-trip ок: {tok.decode(ids)[:40] == text[:40]}")

T = min(512, len(ids) - 1)
x = mx.array(ids[:T]).reshape(1, T)
y = mx.array(ids[1:T+1]).reshape(1, T)

m, cfg = build_and_load(PTH)
loss = float(m.loss(x, y)); mx.eval(loss)
print(f"\nloss на реальном русском тексте: {loss:.4f}")
print(f"perplexity: {math.exp(loss):.2f}")
print(f"(random был бы ln(65536)={math.log(65536):.2f})")
print("ВЕРДИКТ:", "✅ конвертация КОРРЕКТНА" if loss < 5.0 else "❌ мусор — баг в конвертации/архитектуре")
