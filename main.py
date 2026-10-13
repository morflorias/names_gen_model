from __future__ import annotations

import argparse
import sys
from pathlib import Path


def cmd_train(args):
    from src.train import train
    train(
        corpus_path=args.corpus,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_len=args.max_len,
        lr=args.lr,
        patience=args.patience,
    )


def _load_model_and_tokenizer(checkpoint_path: str, tokenizer_path: str):
    import torch
    from src.model import NameGenerator
    from src.tokenizer import Tokenizer

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    tokenizer = Tokenizer.load(tokenizer_path)

    model = NameGenerator(
        vocab_size=ckpt["vocab_size"],
        n_categories=ckpt["n_categories"],
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, tokenizer, ckpt["cat_to_id"]


def _parse_categories(raw: str | None, legacy: str | None) -> str:
    """--categories "cosmetics,food" / "cosmetics+food" (+ legacy --category)."""
    if raw:
        return raw
    if legacy:
        return legacy
    print("Pass --categories, e.g. --categories cosmetics,food")
    sys.exit(2)


def cmd_generate(args):
    from src.generate import (
        normalize_category_ids,
        resolve_prefix_ids,
        temperature_generate,
        beam_search_generate,
    )

    model, tokenizer, cat_to_id = _load_model_and_tokenizer(
        args.checkpoint, args.tokenizer
    )

    if args.list_start_tokens:
        for tok, cnt in tokenizer.start_tokens(top_k=args.top_n):
            print(f"{tok}\t{cnt}")
        return

    raw_cats = _parse_categories(args.categories, args.category)
    try:
        cat_ids = normalize_category_ids(raw_cats, cat_to_id)
    except ValueError as e:
        print(e)
        sys.exit(1)
    if len(cat_ids) < 2:
        print(
            "Choose at least 2 categories, e.g. --categories cosmetics,food "
            "(hybrid product belongs to both at once)."
        )
        sys.exit(1)

    try:
        prefix = resolve_prefix_ids(args.start_token, tokenizer)
    except ValueError as e:
        print(e)
        sys.exit(1)

    label = "+".join(sorted(
        c for c in
        ([k for k, v in cat_to_id.items() if v in cat_ids])
    ))
    print(f"Categories: {label}   Start token: {args.start_token or '—'}")

    if args.beam:
        beams = beam_search_generate(
            model, tokenizer, cat_ids, prefix_ids=prefix,
            beam_width=args.beam,
        )
        for text, score in beams[:args.n]:
            print("%s  (%.3f)" % (text, score))
    else:
        for _ in range(args.n):
            name = temperature_generate(
                model, tokenizer, cat_ids, prefix_ids=prefix,
                temperature=args.temperature,
                top_k=args.top_k,
            )
            print(name)


def cmd_eval(args):
    import torch
    from src.model import NameGenerator
    from src.tokenizer import Tokenizer
    from src.evaluate import evaluate

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    tokenizer = Tokenizer.load(args.tokenizer)

    model = NameGenerator(
        vocab_size=ckpt["vocab_size"],
        n_categories=ckpt["n_categories"],
    )
    model.load_state_dict(ckpt["model"])
    model.eval()

    scores = evaluate(model, tokenizer, args.corpus, ckpt["cat_to_id"])
    for k, v in scores.items():
        print("%s: %.4f" % (k, v))


def cmd_tsne(args):
    import torch
    from src.model import NameGenerator
    from src.tokenizer import Tokenizer
    from src.visualize import visualize

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    tokenizer = Tokenizer.load(args.tokenizer)

    model = NameGenerator(
        vocab_size=ckpt["vocab_size"],
        n_categories=ckpt["n_categories"],
    )
    model.load_state_dict(ckpt["model"])
    model.eval()

    from src.dataset import resolve_corpus
    path = visualize(model, tokenizer, resolve_corpus(args.corpus), args.output)
    print("Saved: %s" % path)


def cmd_web(args):
    import subprocess
    app = str(Path(__file__).parent / "app" / "streamlit_app.py")
    subprocess.run(["streamlit", "run", app, "--server.port", str(args.port)])


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid product name generator (2+ categories + start token)"
    )
    sub = parser.add_subparsers(dest="command")

    p_train = sub.add_parser("train", help="Train the model")
    p_train.add_argument("--corpus", default=None,
                         help="CSV corpus (default: data/corpus_pairs.csv, fallback: legacy)")
    p_train.add_argument("--checkpoint-dir", default="checkpoints")
    p_train.add_argument("--epochs", type=int, default=40)
    p_train.add_argument("--batch-size", type=int, default=32)
    p_train.add_argument("--max-len", type=int, default=15)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--patience", type=int, default=5)

    p_gen = sub.add_parser("generate", help="Generate hybrid names")
    p_gen.add_argument("--checkpoint", default="checkpoints/best.pt")
    p_gen.add_argument("--tokenizer", default="checkpoints/tokenizer.json")
    p_gen.add_argument("--categories", default=None,
                       help="2+ categories, e.g. 'cosmetics,food' or 'cosmetics+food'")
    p_gen.add_argument("--category", default=None,
                       help="Legacy alias for a single category")
    p_gen.add_argument("--start-token", default=None,
                       help="Initial token from the tokenizer (see --list-start-tokens)")
    p_gen.add_argument("--list-start-tokens", action="store_true",
                       help="Print available initial tokens and exit")
    p_gen.add_argument("--top-n", type=int, default=50)
    p_gen.add_argument("--n", type=int, default=5)
    p_gen.add_argument("--temperature", type=float, default=0.9)
    p_gen.add_argument("--top-k", type=int, default=30)
    p_gen.add_argument("--beam", type=int, default=0, help="Beam width (0=disabled)")

    p_eval = sub.add_parser("eval", help="Evaluate model")
    p_eval.add_argument("--checkpoint", default="checkpoints/best.pt")
    p_eval.add_argument("--tokenizer", default="checkpoints/tokenizer.json")
    p_eval.add_argument("--corpus", default=None)

    p_tsne = sub.add_parser("tsne", help="t-SNE visualization")
    p_tsne.add_argument("--checkpoint", default="checkpoints/best.pt")
    p_tsne.add_argument("--tokenizer", default="checkpoints/tokenizer.json")
    p_tsne.add_argument("--corpus", default=None)
    p_tsne.add_argument("--output", default="outputs/tsne.png")

    p_web = sub.add_parser("web", help="Launch Streamlit UI")
    p_web.add_argument("--port", type=int, default=8501)

    args = parser.parse_args()
    if args.command == "train":
        cmd_train(args)
    elif args.command == "generate":
        cmd_generate(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "tsne":
        cmd_tsne(args)
    elif args.command == "web":
        cmd_web(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
