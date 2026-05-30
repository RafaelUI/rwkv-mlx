# RWKV-7 конфигурация для обучения с нуля на русском

class RWKVConfig:
    vocab_size: int  = 32000
    n_layer:    int  = 12
    n_embd:     int  = 768
    head_size:  int  = 64

    @property
    def n_head(self):
        return self.n_embd // self.head_size

    # ctx_len=4096 — по рекомендации автора RWKV-7
    ctx_len:      int   = 4096
    batch_size:   int   = 3
    lr:           float = 6e-4
    weight_decay: float = 0.1
    beta1:        float = 0.9
    beta2:        float = 0.95
    adam_eps:     float = 1e-18
    grad_clip:    float = 1.0
    warmup_steps: int   = 20

    data_path:      str = "data/train.bin"
    tokenizer_path: str = "tokenizer/rwkv_ru.model"

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


CONFIGS = {
    # ~36M параметров — старт, быстрая проверка
    "debug": RWKVConfig(n_layer=6,  n_embd=384,  vocab_size=32000,
                        ctx_len=1024, batch_size=12),

    # ~100M параметров — основная цель
    "100M":  RWKVConfig(n_layer=12, n_embd=768,  vocab_size=32000,
                        ctx_len=1024, batch_size=4),

    # ~300M параметров
    "300M":  RWKVConfig(n_layer=24, n_embd=1024, vocab_size=32000,
                        ctx_len=4096, batch_size=1),
}
