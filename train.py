import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import numpy as np
import math, time, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CONFIGS
from model.rwkv7 import RWKV7, init_weights

CFG_NAME   = "debug"
DTYPE       = None           # dtype входных токенов (оставляем None)
MODEL_DTYPE = mx.bfloat16   # bf16 модель: +5-10% скорость, -12% RAM, качество равное fp32
GRAD_ACCUM  = 1             # gradient accumulation: 1=выкл, 4=эфф.batch×4, 8=×8
TRAIN_BIN  = "data/train.bin"
VAL_BIN    = "data/val.bin"
CKPT_DIR   = "checkpoints"
LOG_EVERY  = 50
EVAL_EVERY = 500
SAVE_EVERY = 1000
MAX_STEPS  = 500_000

class BinDataset:
    def __init__(self, path, ctx_len):
        # np.memmap - не грузит всё в RAM, читает батчи с диска
        self.data    = np.memmap(path, dtype=np.uint16, mode='r')
        self.ctx_len = ctx_len
        self.n       = len(self.data)
        size_gb      = self.n * 2 / 1e9
        print(f"  {path}: {self.n/1e6:.1f}M токенов ({size_gb:.1f} GB)")

    def batch(self, batch_size, step):
        stride = self.ctx_len + 1
        starts = [(step * batch_size + i) * stride % (self.n - stride)
                  for i in range(batch_size)]
        # Читаем только нужные позиции — остальное на диске
        x = mx.array(np.stack([
            self.data[s : s + self.ctx_len].astype(np.int32)
            for s in starts
        ]))
        y = mx.array(np.stack([
            self.data[s+1 : s + self.ctx_len + 1].astype(np.int32)
            for s in starts
        ]))
        return x, y

def lr_schedule(step, cfg):
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, MAX_STEPS - cfg.warmup_steps)
    return cfg.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))

def save_bf16(model, path):
    """MLX save_weights сохраняет в текущем dtype модели.
    Если модель в bf16 → файл 2× меньше автоматически (~70MB вместо ~140MB)."""
    model.save_weights(path)

def loss_fn(model, x, y):
    return model.loss(x, y).astype(mx.float32)

def make_train_step(model, optimizer, grad_accum=1):
    """Шаг обучения с опциональным gradient accumulation.

    grad_accum=1: обычный mx.compile шаг (быстро, без накопления)
    grad_accum=N: накапливаем градиенты N микро-шагов → эфф.batch = batch × N
                  Память не растёт. mx.compile внутри каждого микро-шага.
                  Полезно для 100M+ моделей где batch=1 по памяти.
    """
    from mlx.utils import tree_map

    if grad_accum == 1:
        # Быстрый путь: один compile на весь шаг
        state = [model.state, optimizer.state]
        def _step(x, y):
            loss, grads = mx.value_and_grad(loss_fn)(model, x, y)
            grads, norm = optim.clip_grad_norm(grads, max_norm=1.0)
            optimizer.update(model, grads)
            return loss, norm
        return mx.compile(_step, inputs=state, outputs=state)

    # Gradient accumulation: N микро-шагов, один update
    # КРИТИЧНО: mx.eval(grads) после каждого шага — иначе lazy граф
    # накапливается как a+b+c+... и взрывает память (видели 28GB!)
    def _accum_step(xs, ys):
        """xs, ys — списки из grad_accum микро-батчей"""
        # Первый микро-шаг
        total_loss, total_grads = mx.value_and_grad(loss_fn)(model, xs[0], ys[0])
        mx.eval(total_loss, total_grads)  # ← материализуем сразу

        for i in range(1, grad_accum):
            loss_i, grads_i = mx.value_and_grad(loss_fn)(model, xs[i], ys[i])
            mx.eval(loss_i, grads_i)      # ← материализуем до сложения
            total_loss  = total_loss + loss_i
            total_grads = tree_map(lambda a, b: a + b, total_grads, grads_i)
            mx.eval(total_grads)          # ← материализуем накопленное

        total_grads = tree_map(lambda g: g / grad_accum, total_grads)
        total_loss  = total_loss / grad_accum
        grads, norm = optim.clip_grad_norm(total_grads, max_norm=1.0)
        optimizer.update(model, grads)
        mx.eval(model.state, optimizer.state)
        return total_loss, norm

    return _accum_step

def eval_loss(model, dataset, cfg, n_batches=20):
    losses = []
    for i in range(n_batches):
        x, y = dataset.batch(cfg.batch_size, i)
        l = loss_fn(model, x, y)
        mx.eval(l)
        losses.append(l.item())
    return sum(losses) / len(losses)

def train():
    os.makedirs(CKPT_DIR, exist_ok=True)
    cfg = CONFIGS[CFG_NAME]
    print(f"Конфиг: {CFG_NAME} | layers={cfg.n_layer} embd={cfg.n_embd} vocab={cfg.vocab_size}")

    print("Загрузка данных:")
    train_ds = BinDataset(TRAIN_BIN, cfg.ctx_len)
    val_ds   = BinDataset(VAL_BIN,   cfg.ctx_len)

    print("Инициализация модели...")
    model    = RWKV7(cfg)
    n_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"  Параметры: {n_params/1e6:.1f}M")

    ckpt_path  = os.path.join(CKPT_DIR, f"rwkv7_{CFG_NAME}_latest.npz")

    if not os.path.exists(ckpt_path):
        print("  Инициализация весов RWKV-7...")
        model = init_weights(model)

    # Конвертируем модель в bf16 если задано
    if MODEL_DTYPE is not None:
        from mlx.utils import tree_map
        model.update(tree_map(
            lambda x: x.astype(MODEL_DTYPE) if isinstance(x, mx.array) else x,
            model.parameters()))
        print(f"  Модель в {MODEL_DTYPE} (+5-10% скорость, -12% RAM)")

    optimizer = optim.AdamW(
        learning_rate = cfg.lr,
        betas         = (cfg.beta1, cfg.beta2),
        eps           = cfg.adam_eps,
        weight_decay  = cfg.weight_decay,
    )

    start_step = 0
    step_file  = ckpt_path.replace(".npz", ".step")
    if os.path.exists(ckpt_path):
        model.load_weights(ckpt_path)
        if os.path.exists(step_file):
            start_step = int(open(step_file).read())
        print(f"  Продолжаем с шага {start_step}")

    # Создаём compile-оптимизированный шаг
    train_step = make_train_step(model, optimizer, GRAD_ACCUM)

    eff_batch = cfg.batch_size * GRAD_ACCUM
    accum_str = f" | grad_accum={GRAD_ACCUM} (эфф.batch={eff_batch})" if GRAD_ACCUM > 1 else ""
    print(f"Обучение | ctx={cfg.ctx_len} batch={cfg.batch_size}{accum_str} | mx.compile ✓")
    print("-" * 60)

    t0       = time.time()
    best_val = float("inf")
    losses   = []

    for step in range(start_step, MAX_STEPS or int(1e9)):
        optimizer.learning_rate = lr_schedule(step, cfg)
        if GRAD_ACCUM == 1:
            x, y = train_ds.batch(cfg.batch_size, step)
            if DTYPE is not None:
                x, y = x.astype(DTYPE), y.astype(DTYPE)
            loss, norm = train_step(x, y)
            mx.eval(loss, norm)
        else:
            xs = [train_ds.batch(cfg.batch_size, step * GRAD_ACCUM + i)[0]
                  for i in range(GRAD_ACCUM)]
            ys = [train_ds.batch(cfg.batch_size, step * GRAD_ACCUM + i)[1]
                  for i in range(GRAD_ACCUM)]
            loss, norm = train_step(xs, ys)
        losses.append(loss.item())

        if (step + 1) % LOG_EVERY == 0:
            dt    = time.time() - t0
            avg   = sum(losses[-LOG_EVERY:]) / LOG_EVERY
            tok_s = cfg.batch_size * cfg.ctx_len * GRAD_ACCUM * LOG_EVERY / dt
            t0    = time.time()
            print(f"step {step+1:6d} | loss {avg:.4f} | lr {optimizer.learning_rate:.2e} | norm {norm.item():.2f} | {tok_s:.0f} tok/s")

        if (step + 1) % EVAL_EVERY == 0:
            val = eval_loss(model, val_ds, cfg)
            mark = " <- best" if val < best_val else ""
            print(f"  VAL loss: {val:.4f}{mark}")
            if val < best_val:
                best_val = val
                save_bf16(model, os.path.join(CKPT_DIR, f"rwkv7_{CFG_NAME}_best.npz"))

        if (step + 1) % SAVE_EVERY == 0:
            save_bf16(model, ckpt_path)
            open(step_file, "w").write(str(step + 1))
            print(f"  Чекпоинт: шаг {step+1}")

        if MAX_STEPS and step + 1 >= MAX_STEPS:
            break

    save_bf16(model, ckpt_path)
    print(f"Готово. Лучший val loss: {best_val:.4f}")

if __name__ == "__main__":
    train()
