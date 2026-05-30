"""
Обучение BPE токенайзера на русской прозе.
Оптимизирован под кириллицу, художественный текст, морфологию русского языка.
"""
import sentencepiece as spm
import os
import random

CORPUS      = "/Users/s/Develop/rwkv-mlx/cleaned.txt"
MODEL_PATH  = "/Users/s/Develop/rwkv-mlx/tokenizer/rwkv_ru"
SAMPLE_PATH = "/Users/s/Develop/rwkv-mlx/tokenizer/sample.txt"

VOCAB_SIZE   = 32000
SAMPLE_GB    = 3.0
SAMPLE_LINES = 8_000_000


def make_sample():
    if os.path.exists(SAMPLE_PATH):
        size_gb = os.path.getsize(SAMPLE_PATH) / 1e9
        print(f"Выборка уже есть: {size_gb:.2f} GB — пропускаем")
        return
    print(f"Создаём выборку {SAMPLE_GB} GB из корпуса...")
    target_bytes = int(SAMPLE_GB * 1e9)
    written = 0
    count   = 0
    with open(CORPUS, "r", encoding="utf-8") as fin,          open(SAMPLE_PATH, "w", encoding="utf-8") as fout:
        for line in fin:
            if len(line.strip()) < 20:
                continue
            if random.random() > 0.12:
                continue
            fout.write(line)
            written += len(line.encode("utf-8"))
            count   += 1
            if written >= target_bytes or count >= SAMPLE_LINES:
                break
    print(f"Выборка готова: {written/1e9:.2f} GB, {count:,} строк")


def train():
    make_sample()
    print(f"Обучаем токенайзер (vocab={VOCAB_SIZE})...")
    spm.SentencePieceTrainer.train(
        input              = SAMPLE_PATH,
        model_prefix       = MODEL_PATH,
        vocab_size         = VOCAB_SIZE,
        model_type         = "bpe",
        character_coverage = 0.9999,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        pad_piece="[PAD]", unk_piece="[UNK]",
        bos_piece="[BOS]", eos_piece="[EOS]",
        user_defined_symbols=["--","—","...","«","»","!..","?..","?!"],
        max_sentencepiece_length = 16,
        split_by_unicode_script  = True,
        split_by_number          = True,
        split_digits             = True,
        max_sentence_length      = 8192,
        input_sentence_size      = SAMPLE_LINES,
        shuffle_input_sentence   = True,
        num_threads              = 8,
        byte_fallback            = True,
    )
    print(f"Токенайзер сохранён: {MODEL_PATH}.model")


def test():
    sp = spm.SentencePieceProcessor()
    sp.load(f"{MODEL_PATH}.model")
    examples = [
        "Михаил Иванович нехотя взял трубку.",
        "-- Ты уверен? -- тихо спросил он.",
        "перефразирование перефразировал перефразирует",
        "«Странное дело», — подумал он...",
    ]
    print("Проверка токенайзера:")
    for text in examples:
        tokens = sp.encode(text)
        pieces = sp.encode(text, out_type=str)
        print(f"  {len(tokens)} токенов: {pieces}")

    print("Морфология:")
    for w in ["перефразировать","перефразировал","перефразирование"]:
        print(f"  {w:30s} -> {sp.encode(w, out_type=str)}")


if __name__ == "__main__":
    train()
    test()
