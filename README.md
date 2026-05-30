# RWKV-7 Pretraining on Apple Silicon (MLX)

Полный стек предобучения RWKV-7 "Goose" с нуля на устройствах Apple Silicon через MLX.
Включает кастомный Metal backward kernel, достигающий **7.8× ускорения** vs Python einsum.

## Требования

- macOS с Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- MLX 0.31+

```bash
pip install mlx mlx-lm numpy
```

## Быстрый старт

```bash
# 1. Расширить GPU memory limit (рекомендуется)
sudo sysctl iogpu.wired_limit_mb=14336

# 2. Подготовить данные (JSONL → binidx)
python data/prepare.py --input your_data.jsonl --output data/train.bin

# 3. Запустить обучение
python train.py
```

## Конфигурация

Настройки в начале `train.py`:

```python
CFG_NAME    = "debug"        # "debug" (36M) | "100M" (138M)
MODEL_DTYPE = mx.bfloat16    # bf16 весов: +10% скорость, -12% RAM
GRAD_ACCUM  = 1              # gradient accumulation (1=выкл, 4=эфф.batch×4)
```

Конфиги моделей в `config.py`:

| Конфиг | Параметры | ctx | batch | RAM | tok/s |
|--------|-----------|-----|-------|-----|-------|
| `debug` | 36.4M | 1024 | 4 | ~10.5 GB | ~6 000 |
| `100M` | 138.9M | 512 | 4 | ~12.1 GB | ~1 700 |

## Архитектура Metal kernel

Центральная инновация — **WKV7 Checkpoint Kernel**: два Metal-вызова
заменяют 16 Python-итераций.

### Forward (один kernel, T токенов)

```
grid=(B*H, D, 1)   # параллельно по batch × heads
for c in 0..N_CHUNKS:
    for t in 0..CHUNK:
        h = w*h + v*k + sa*b
        y = dot(h, r)
    h_checkpoints[c] = h   # ← сохраняем h каждые 32 токена
```

### Backward (один kernel, обратный порядок)

```
grid=(B*H*D, 1, 1)
for c in N_CHUNKS-1..0:
    h_row = h_checkpoints[c]   # ← читаем точный checkpoint
    for t in CHUNK-1..0:
        # Вычисляем dr, dw, dk, dv, da, db через VJP
        # Реконструируем h_prev = (h_cur - v*k - sa*b) / w
```

**Зачем checkpoint:** деление на `w` при реконструкции усиливает ошибку в
`(1/w)^N`. При N=512: ~10²³ (взрыв). При N=32 (CHUNK): ~30 (допустимо).

### Производительность vs Python baseline

| Оптимизация | tok/s | Прирост |
|-------------|-------|---------|
| Python einsum | ~900 | — |
| Metal v2 chunked | 3 666 | +4.1× |
| Checkpoint kernel | ~5 000 | +1.4× |
| + bf16 | ~6 050 | +1.2× |
| + mx.compile | 6 720 | +1.1× |
| + batch scaling | **6 978** | **+1.04×** |
| **Итого** | **6 978** | **7.8×** |

## Детали реализации

### wkv7_checkpoint.py

Основной файл. Экспортирует `make_wkv7_checkpoint(B, T, H, D)`.

```python
from model.wkv7_checkpoint import make_wkv7_checkpoint

# Создаём функцию для конкретного (B, T, H) — компилируется один раз
wkv7_train = make_wkv7_checkpoint(B=4, T=1024, H=6, D=64)

# Использование (drop-in замена для chunked v2):
output = wkv7_train(r, w, k, v, a, b)   # → (B, T, H, D)
```

### bf16 + mx.compile

```python
# Конвертируем модель в bf16
from mlx.utils import tree_map
model.update(tree_map(lambda x: x.astype(mx.bfloat16), model.parameters()))

# Компилируем train step
state = [model.state, optimizer.state]
def _step(x, y):
    loss, grads = mx.value_and_grad(loss_fn)(model, x, y)
    grads, _ = optim.clip_grad_norm(grads, 1.0)
    optimizer.update(model, grads)
    return loss
train_step = mx.compile(_step, inputs=state, outputs=state)

# loss_fn должна возвращать fp32 для bf16 моделей:
def loss_fn(model, x, y):
    return model.loss(x, y).astype(mx.float32)
```

### Gradient Accumulation

```python
# КРИТИЧНО: mx.eval после каждого микро-шага!
# Иначе lazy граф накапливается → 28 GB OOM
for i in range(GRAD_ACCUM):
    loss_i, grads_i = compiled_micro(xs[i], ys[i])
    mx.eval(loss_i, grads_i)           # ← обязательно
    total_grads += grads_i
    mx.eval(total_grads)               # ← обязательно
```

## Подготовка данных

```bash
# Формат JSONL:
{"text": "Полный текст документа..."}

# Токенизация в binidx (для vocab_size=32000):
python data/prepare.py --input corpus.jsonl --output data/train.bin

# Для RWKV World моделей (vocab_size=65536):
python data/prepare.py --input corpus.jsonl --output data/train.bin --world-tokenizer
```

## Инициализация весов RWKV-7

При обучении с нуля критически важна правильная инициализация:

```python
# key.weight — демпфирование для стабильности дельта-правила
model.key.weight = init_ortho(shape) * 0.1   # ← жёсткое демпфирование

# head.weight — ортогональная инициализация
model.head.weight = init_ortho(shape) * sqrt(vocab_size / n_embd)

# Оптимизатор: специальные параметры для RWKV-7
optimizer = optim.AdamW(lr=..., eps=1e-18, betas=(0.9, 0.95))
```

## Память и масштабирование

| Модель | Веса bf16 | Adam state | Активации* | Итого |
|--------|-----------|------------|-----------|-------|
| 36.4M debug | 73 MB | 291 MB | ~7 GB | ~10 GB |
| 138.9M | 278 MB | 1.1 GB | ~10 GB | ~12 GB |

*С mx.compile kernel fusion активации резко уменьшаются (-2.8×).

## Известные ограничения

1. **ctx=4096** требует `iogpu.wired_limit_mb=14336` (выходит за 12 GB по умолчанию)
2. **138.9M предобучение** нецелесообразно на M4 Air 16 GB — 33 дня на 5B токенов
3. **mx.eval в VJP** нельзя использовать внутри mx.compile — убран из checkpoint kernel

## Следующие шаги (TODO)

- [x] LoRA/QLoRA файнтюн официальных World моделей (1.5B проверена)
- [ ] Инструкционные данные перефразирования (World-токенизация)
- [ ] GooseOne 2.9B (после чистки диска)
- [ ] 8-bit AdamW (4× меньше optimizer state → больший batch для 138.9M)
- [ ] Публикация Metal WKV7 kernel как отдельного пакета

## Файнтюн официальных моделей RWKV-7 World (LoRA / QLoRA)

Помимо обучения с нуля, репозиторий поддерживает LoRA/QLoRA-файнтюн готовых
официальных весов RWKV-7 World на Apple Silicon 16 ГБ.

### Загрузка официальных весов

`model/rwkv7_x070.py` — архитектура, точно соответствующая официальному x070
(под веса `BlinkDL/rwkv-7-world`). `convert_rwkv_pth.py` — **torch-free** конвертер
`.pth` (torch не требуется, читает zip+pickle напрямую в MLX).

```bash
# Скачать (пример: World 1.5B)
python -c "from huggingface_hub import hf_hub_download; \
  hf_hub_download('BlinkDL/rwkv-7-world','RWKV-x070-World-1.5B-v3-20250127-ctx4096.pth',local_dir='weights')"

# Конвертировать -> weights/rwkv7_1.5B_x070.safetensors
python convert_rwkv_pth.py weights/RWKV-x070-World-1.5B-v3-20250127-ctx4096.pth load

# Проверить (loss на реальном тексте ~3, не ~11)
python validate_conversion.py
```

Проверено: World 1.5B загружается, **loss 3.08** на русском тексте (random=11.09).

### LoRA / QLoRA через наш Metal kernel

`model/lora.py` — адаптеры на `tmix.{r,k,v,o}_proj`, градиент течёт через WKV backward kernel.

```python
from model.lora import add_lora
model, info = add_lora(model, rank=16, alpha=16.0,
                       quantize_base=4,        # QLoRA: 4-бит замороженная база
                       layers=range(12, 24))   # адаптеры только на верхние слои
```

**Провалидированный рецепт (16 ГБ):**
- forward через `mlx.nn.utils.checkpoint` (НЕ `mx.checkpoint` — тот теряет градиент адаптеров)
- `nn.value_and_grad` (НЕ `mx.value_and_grad` — игнорирует `freeze()`)
- БЕЗ `mx.compile` (на больших моделях замедляет ~5×)
- `mx.set_cache_limit(int(1.5e9))` против свопа
- QLoRA: квантовать только большие матрицы (`cmix/head/emb` + цели), low-rank оставлять bf16
- предобученная база: `lr~1e-4, alpha=16` (агрессивный lr разносит модель)

**Капстоун** (World 1.5B, QLoRA 4-бит, верхние 12 слоёв): loss 3.02 → 0.006,
peak 4.40 ГБ, 250 tok/s.

| LoRA 1.5B | tok/s | peak |
|-----------|-------|------|
| все 24 слоя | 168 | 4.07 ГБ |
| верхние 12 | 250 | 3.40 ГБ |
| верхние 6 | 337 | 3.18 ГБ |

> Токенайзеры: from-scratch трек использует кастомный 32k RU (`tokenizer/`),
> официальные World-модели — World-токенайзер 65536 (`model/world_tokenizer.py`).
> Они НЕ взаимозаменяемы.

## Лицензия

Apache 2.0

## Связанные проекты

- [RWKV-LM](https://github.com/BlinkDL/RWKV-LM) — оригинальный репозиторий
- [mlx-lm](https://github.com/ml-explore/mlx-lm) — inference RWKV-7 через MLX (PR #580)
- [maderix/ANE](https://github.com/maderix/ANE) — Apple Neural Engine private API
