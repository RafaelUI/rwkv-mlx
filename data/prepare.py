"""
Подготовка данных для обучения RWKV-7.
Токенизирует cleaned.txt и сохраняет в бинарный формат uint16.
"""
import sentencepiece as spm
import numpy as np
import os
import sys

CORPUS       = "/Users/s/Develop/rwkv-mlx/cleaned.txt"
TOKENIZER    = "/Users/s/Develop/rwkv-mlx/tokenizer/rwkv_ru.model"
OUT_TRAIN    = "/Users/s/Develop/rwkv-mlx/data/train.bin"
OUT_VAL      = "/Users/s/Develop/rwkv-mlx/data/val.bin"

VAL_RATIO    = 0.001   # 0.1% на валидацию
CHUNK_LINES  = 10_000  # строк за раз в памяти
BOS_ID       = 2
EOS_ID       = 3


def prepare():
    if not os.path.exists(TOKENIZER):
        print("Токенайзер не найден. Сначала запусти tokenizer/train.py")
        sys.exit(1)

    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER)
    print(f"Токенайзер загружен. Vocab: {sp.get_piece_size()}")

    # Считаем строки для прогресса
    print("Считаем строки...")
    total_lines = sum(1 for _ in open(CORPUS, encoding="utf-8"))
    print(f"Всего строк: {total_lines:,}")

    val_tokens  = []
    train_buf   = []
    written     = 0
    val_written = 0
    processed   = 0
    val_done    = False

    train_f = open(OUT_TRAIN, "wb")

    with open(CORPUS, "r", encoding="utf-8") as f:
        chunk = []
        for line in f:
            line = line.strip()
            if len(line) < 5:
                continue
            chunk.append(line)

            if len(chunk) >= CHUNK_LINES:
                ids = _process_chunk(sp, chunk)
                chunk = []
                processed += CHUNK_LINES

                # Первые VAL_RATIO токенов — валидация
                if not val_done:
                    val_tokens.extend(ids)
                    if len(val_tokens) >= 500_000:
                        val_done = True
                else:
                    arr = np.array(ids, dtype=np.uint16)
                    arr.tofile(train_f)
                    written += len(ids)

                if processed % 500_000 == 0:
                    gb = written * 2 / 1e9
                    print(f"  {processed:,} строк | {written/1e6:.1f}M токенов | {gb:.2f} GB")

        # Остаток
        if chunk:
            ids = _process_chunk(sp, chunk)
            if val_done:
                arr = np.array(ids, dtype=np.uint16)
                arr.tofile(train_f)
                written += len(ids)

    train_f.close()

    # Сохраняем валидацию
    np.array(val_tokens[:500_000], dtype=np.uint16).tofile(OUT_VAL)

    print("\n✓ Готово:")
    print(f"  train.bin: {written/1e6:.1f}M токенов ({written*2/1e9:.2f} GB)")
    print(f"  val.bin:   {min(len(val_tokens),500_000)/1e6:.1f}M токенов")


def _process_chunk(sp, lines):
    ids = []
    for line in lines:
        toks = sp.encode(line)
        if len(toks) < 3:
            continue
        ids.append(BOS_ID)
        ids.extend(toks)
        ids.append(EOS_ID)
    return ids


def info():
    """Показывает статистику готовых данных."""
    for name, path in [("train", OUT_TRAIN), ("val", OUT_VAL)]:
        if os.path.exists(path):
            size    = os.path.getsize(path)
            n_toks  = size // 2
            print(f"{name}.bin: {n_toks/1e6:.1f}M токенов ({size/1e9:.2f} GB)")
        else:
            print(f"{name}.bin: не найден")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "info":
        info()
    else:
        prepare()
