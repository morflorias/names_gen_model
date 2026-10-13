from __future__ import annotations

import torch
from collections import Counter
from typing import Sequence

from .dataset import load_corpus, resolve_corpus
from .generate import normalize_category_ids
from .model import NameGenerator
from .tokenizer import Tokenizer
from .vocab import Vocab


def ngrams(tokens: Sequence[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def bleu_score(candidate: str, references: list[str], max_n: int = 4) -> float:
    cand_tokens = candidate.split()
    if not cand_tokens:
        return 0.0

    total_score = 0.0
    for n in range(1, max_n + 1):
        cand_ngrams = Counter(ngrams(cand_tokens, n))
        if not cand_ngrams:
            break

        clipped = 0
        total_count = 0
        for ng, count in cand_ngrams.items():
            max_ref = max(ref.count(" ".join(ng)) for ref in references)
            clipped += min(count, max_ref)
            total_count += count

        if total_count == 0:
            break
        total_score += (clipped / total_count) ** (1 / n)

    avg = total_score / max_n
    brevity = min(1.0, len(cand_tokens) / max(1, min(len(r.split()) for r in references)))
    return avg * brevity


def jaccard_similarity(a: str, b: str) -> float:
    set_a = set(a.split())
    set_b = set(b.split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


@torch.no_grad()
def generate_samples(
    model: NameGenerator,
    tokenizer: Tokenizer,
    cat_to_id: dict[str, int],
    n_per_cat: int = 50,
    temperature: float = 0.9,
    top_k: int = 30,
    device: torch.device = torch.device("cpu"),
    labels: Sequence[str] | None = None,
) -> dict[str, list[str]]:
    """Sample names per (pair) label, e.g. ``cosmetics+food``."""
    model.eval()
    if labels is None:
        labels = sorted(cat_to_id.keys())
    results: dict[str, list[str]] = {}

    for label in labels:
        pair_ids = normalize_category_ids(label, cat_to_id)
        cat_tensor = torch.tensor([pair_ids] * n_per_cat, dtype=torch.long, device=device)
        tokens_batch = model.generate(
            cat_tensor, max_len=15, temperature=temperature, top_k=top_k
        )
        results[label] = [tokenizer.decode(t) for t in tokens_batch]

    return results


def evaluate(
    model: NameGenerator,
    tokenizer: Tokenizer,
    corpus_path: str | None,
    cat_to_id: dict[str, int],
    n_per_cat: int = 50,
    device: torch.device = torch.device("cpu"),
) -> dict[str, float]:
    corpus_path = resolve_corpus(corpus_path)
    data = load_corpus(corpus_path)
    real_names: dict[str, list[str]] = {}
    for item in data:
        real_names.setdefault(item["label"], []).append(item["name"])

    generated = generate_samples(
        model, tokenizer, cat_to_id, n_per_cat,
        device=device, labels=sorted(real_names.keys()),
    )

    bleu_scores = []
    jaccard_scores = []
    all_gen_tokens: list[str] = []
    all_real_tokens: list[str] = []

    for cat, gen_names in generated.items():
        refs = real_names.get(cat, [])
        for name in gen_names:
            bleu_scores.append(bleu_score(name, refs[:100]))
            if refs:
                jaccard_scores.append(max(jaccard_similarity(name, r) for r in refs[:50]))
            all_gen_tokens.extend(name.split())

    for names in real_names.values():
        for name in names:
            all_real_tokens.extend(name.split())

    gen_set = set(all_gen_tokens)
    real_set = set(all_real_tokens)
    cosine_sim = (
        len(gen_set & real_set)
        / ((len(gen_set) ** 0.5) * (len(real_set) ** 0.5))
        if gen_set and real_set
        else 0.0
    )

    novelty_count = 0
    novelty_total = 0
    for cat, gen_names in generated.items():
        refs = real_names.get(cat, [])
        for name in gen_names:
            novelty_total += 1
            if name not in refs:
                novelty_count += 1

    return {
        "bleu": sum(bleu_scores) / max(len(bleu_scores), 1),
        "jaccard": sum(jaccard_scores) / max(len(jaccard_scores), 1),
        "novelty": novelty_count / max(novelty_total, 1),
        "cosine": cosine_sim,
    }
