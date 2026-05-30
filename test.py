#!/usr/bin/env python3
"""RWKV тестовый скрипт.

Этот скрипт загружает токенайзер из заданного пути
и выполняет один прямой проход через модель RWKV.
"""

import argparse
import os

import sentencepiece as spm
import mlx.core as mx

from config import CONFIGS, RWKVConfig
from model.rwkv7 import RWKV7, init_weights

TOKENIZER_MODEL = "/Users/s/Develop/rwkv-mlx/tokenizer/rwkv_ru.model"


def load_tokenizer(path: str = TOKENIZER_MODEL):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Tokenizer model not found: {path}")
    sp = spm.SentencePieceProcessor()
    sp.load(path)
    return sp


def select_config(config_name: str | None, weights_path: str | None):
    if config_name is not None:
        if config_name not in CONFIGS:
            raise ValueError(f"Unknown config: {config_name}. Available: {', '.join(CONFIGS)}")
        return CONFIGS[config_name]

    if weights_path is not None:
        for name in CONFIGS:
            if name in os.path.basename(weights_path):
                print(f"Detected config '{name}' from weights path")
                return CONFIGS[name]

    return RWKVConfig()


def build_model(weights_path: str | None = None, config_name: str | None = None):
    cfg = select_config(config_name, weights_path)
    model = RWKV7(cfg)
    if weights_path:
        if os.path.exists(weights_path):
            print(f"Loading RWKV weights from: {weights_path}")
            model.load_weights(weights_path)
        else:
            print(f"Weight file not found: {weights_path}")
            print("Using randomly initialized weights instead.")
            model = init_weights(model)
    else:
        print("No weights path provided. Using randomly initialized RWKV model.")
        model = init_weights(model)
    mx.eval(model.parameters())
    return model


def infer(model, tokenizer, text: str):
    token_ids = tokenizer.encode(text)
    if len(token_ids) == 0:
        raise ValueError("Prompt is empty after tokenization")

    prompt_pieces = tokenizer.encode(text, out_type=str)
    print("Prompt:", text)
    print("Token IDs:", token_ids)
    print("Pieces:", prompt_pieces)

    if len(token_ids) > model.config.ctx_len:
        token_ids = token_ids[-model.config.ctx_len:]
        print(f"Prompt truncated to last {model.config.ctx_len} tokens")

    x = mx.array([token_ids], dtype=mx.int32)
    logits = model(x)
    mx.eval(logits)
    print("Logits shape:", logits.shape)

    last_logits = logits[0, -1]
    try:
        top_id = int(mx.argmax(last_logits, axis=-1).item())
        top_piece = tokenizer.id_to_piece(top_id)
        top_text = tokenizer.decode([top_id])
        print("Top predicted next token:", top_id, top_piece)
        print("Decoded next token:", repr(top_text))
    except Exception:
        print("Could not compute top predicted token with mx.argmax")

    try:
        top10 = last_logits[:10].tolist()
        print("First 10 logits for last position:", top10)
    except Exception:
        print("Could not print first logits values")

    return logits


def greedy_generate(model, tokenizer, prompt: str, max_tokens: int = 20):
    token_ids = tokenizer.encode(prompt)
    if len(token_ids) == 0:
        raise ValueError("Prompt is empty after tokenization")

    for _ in range(max_tokens):
        if len(token_ids) > model.config.ctx_len:
            token_ids = token_ids[-model.config.ctx_len:]
        x = mx.array([token_ids], dtype=mx.int32)
        logits = model(x)
        mx.eval(logits)
        next_id = int(mx.argmax(logits[0, -1], axis=-1).item())
        token_ids.append(next_id)
        if next_id == tokenizer.eos_id() if hasattr(tokenizer, 'eos_id') else -1:
            break

    return tokenizer.decode(token_ids)


def parse_args():
    parser = argparse.ArgumentParser(description="RWKV smoke test script")
    parser.add_argument("--prompt", type=str, default="Привет, как дела?",
                        help="Текст для инференса")
    parser.add_argument("--weights", type=str, default=None,
                        help="Опциональный путь к файлу весов RWKV (.npz)")
    parser.add_argument("--config", type=str, default=None, choices=list(CONFIGS),
                        help="Имя конфигурации модели из config.py")
    parser.add_argument("--gen_len", type=int, default=20,
                        help="Сколько токенов сгенерировать")
    parser.add_argument("--tokenizer", type=str, default=TOKENIZER_MODEL,
                        help="Путь к файлу SentencePiece токенизатора")
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer = load_tokenizer(args.tokenizer)
    model = build_model(args.weights, args.config)
    infer(model, tokenizer, args.prompt)
    print("\nGenerating reply from model...")
    generated = greedy_generate(model, tokenizer, args.prompt, max_tokens=args.gen_len)
    print("Generated text:", generated)


if __name__ == "__main__":
    main()
