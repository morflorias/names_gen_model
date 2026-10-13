from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, DataLoader

from .tokenizer import Tokenizer
from .vocab import Vocab

# New pair corpus (each name belongs to 2 categories) + legacy single-category file.
PAIRS_CORPUS = "data/corpus_pairs.csv"
LEGACY_CORPUS = "data/corpus_lmstudio.csv"


def resolve_corpus(path: Path | str | None = None) -> str:
    """Return the corpus file to use.

    Explicit *path* wins. Otherwise prefer the new pair corpus
    (``data/corpus_pairs.csv``) and fall back to the legacy
    single-category file so old setups keep working.
    """
    if path:
        return str(path)
    if Path(PAIRS_CORPUS).exists():
        return PAIRS_CORPUS
    return LEGACY_CORPUS


def pair_label(categories: list[str]) -> str:
    """Canonical label for a (pair) of categories, e.g. ``cosmetics+food``."""
    return "+".join(sorted(categories))


def parse_row_categories(row: dict[str, str]) -> list[str]:
    """Extract category list from a corpus row.

    Supports all known formats:
    - new: ``category_a,category_b[, ...]`` (extra ``category_c`` allowed)
    - alt: ``categories`` with ``+``/``,``-separated values
    - legacy: single ``category``
    """
    cats: list[str] = []
    for key in ("category_a", "category_b", "category_c"):
        if row.get(key):
            cats.append(row[key].strip())
    if not cats and row.get("categories"):
        raw = row["categories"].replace(",", "+")
        cats = [c.strip() for c in raw.split("+") if c.strip()]
    if not cats and row.get("category"):
        raw = row["category"].strip()
        if "+" in raw:
            cats = [c.strip() for c in raw.split("+") if c.strip()]
        elif raw:
            cats = [raw]
    return [c for c in cats if c]


def load_corpus(path: Path | str) -> list[dict[str, str]]:
    """Load corpus and normalize every row to ``name / categories / label``.

    ``categories`` is a list of 1+ base category names, ``label`` is the
    canonical pair key (``cosmetics+food``) or the single category name for
    legacy rows. The raw ``category*`` fields are preserved untouched.
    """
    rows: list[dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = (row.get("name") or "").strip()
            if not name:
                continue
            cats = parse_row_categories(row)
            if not cats:
                continue
            row["name"] = name
            row["categories"] = cats  # type: ignore[assignment]
            row["label"] = pair_label(cats)
            rows.append(row)
    return rows


def split_data(
    data: list[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    rng = random.Random(seed)
    shuffled = data.copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return (
        shuffled[:n_train],
        shuffled[n_train : n_train + n_val],
        shuffled[n_train + n_val :],
    )


class ProductNameDataset(Dataset):
    def __init__(
        self,
        data: list[dict[str, str]],
        tokenizer: Tokenizer,
        category_to_id: dict[str, int],
        max_len: int | None = None,
    ) -> None:
        self.data = data
        self.tokenizer = tokenizer
        self.category_to_id = category_to_id
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def _category_ids(self, item: dict[str, str]) -> list[int]:
        cats = item.get("categories") or ([item["category"]] if item.get("category") else [])
        ids = [self.category_to_id[c] for c in cats if c in self.category_to_id]
        if not ids:
            ids = [0]
        if len(ids) == 1:
            # Single-category rows: duplicate so the conditioning tensor
            # always has width 2 (mean pooling of identical vectors = same).
            ids = ids * 2
        return ids[:3]

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.data[idx]

        full_ids = self.tokenizer.encode(item["name"], add_bos=True, add_eos=True)

        if self.max_len and len(full_ids) > self.max_len + 1:
            full_ids = full_ids[: self.max_len + 1]

        input_ids = full_ids[:-1]
        target_ids = full_ids[1:]
        cat_ids = self._category_ids(item)

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long),
            torch.tensor(cat_ids, dtype=torch.long),
        )


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    pad_id: int = Vocab.PAD_ID,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs, targets, categories = zip(*batch)
    max_len = max(len(x) for x in inputs)

    padded_inputs = torch.full((len(inputs), max_len), pad_id, dtype=torch.long)
    padded_targets = torch.full((len(targets), max_len), pad_id, dtype=torch.long)

    for i, (inp, tgt) in enumerate(zip(inputs, targets)):
        padded_inputs[i, : len(inp)] = inp
        padded_targets[i, : len(tgt)] = tgt

    # Category tensors always share the same width (see _category_ids).
    return padded_inputs, padded_targets, torch.stack(categories)


def create_dataloaders(
    corpus_path: Path | str | None = None,
    tokenizer: Tokenizer | None = None,
    batch_size: int = 32,
    max_len: int = 20,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, int], Tokenizer]:
    corpus_path = resolve_corpus(corpus_path)
    data = load_corpus(corpus_path)
    base_cats: set[str] = set()
    for r in data:
        base_cats.update(r["categories"])
    categories = sorted(base_cats)
    cat_to_id = {c: i for i, c in enumerate(categories)}

    train_data, val_data, test_data = split_data(data, seed=seed)

    if tokenizer is None:
        tokenizer = Tokenizer()
        tokenizer.fit([r["name"] for r in train_data])

    train_ds = ProductNameDataset(train_data, tokenizer, cat_to_id, max_len)
    val_ds = ProductNameDataset(val_data, tokenizer, cat_to_id, max_len)
    test_ds = ProductNameDataset(test_data, tokenizer, cat_to_id, max_len)

    make_collate = lambda b: collate_fn(b, pad_id=Vocab.PAD_ID)

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=make_collate
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=make_collate
    )
    test_dl = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, collate_fn=make_collate
    )

    return train_dl, val_dl, test_dl, cat_to_id, tokenizer
