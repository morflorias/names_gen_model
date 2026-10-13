from __future__ import annotations

import torch
import numpy as np
from pathlib import Path

from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .model import NameGenerator
from .tokenizer import Tokenizer
from .dataset import load_corpus
from .vocab import Vocab


def collect_embeddings(
    model: NameGenerator,
    tokenizer: Tokenizer,
    corpus_path: str,
    device: torch.device = torch.device("cpu"),
) -> tuple[np.ndarray, list[str], list[str]]:
    data = load_corpus(corpus_path)

    all_names = [item["name"] for item in data]
    all_categories = [item["label"] for item in data]

    base_cats: set[str] = set()
    for item in data:
        base_cats.update(item["categories"])
    cat_to_id = {cat: i for i, cat in enumerate(sorted(base_cats))}

    all_embeddings: list[np.ndarray] = []
    batch_size = 128
    model.eval()

    for start in range(0, len(all_names), batch_size):
        batch_names = all_names[start : start + batch_size]
        batch_items = data[start : start + batch_size]

        inputs = []
        cats = []
        for name, item in zip(batch_names, batch_items):
            ids = tokenizer.encode(name, add_bos=True, add_eos=False)
            inputs.append(torch.tensor(ids, dtype=torch.long))
            ids_pair = [cat_to_id[c] for c in item["categories"] if c in cat_to_id] or [0]
            if len(ids_pair) == 1:
                ids_pair = ids_pair * 2
            cats.append(ids_pair[:3])

        max_len = max(len(x) for x in inputs)
        padded = torch.full((len(inputs), max_len), Vocab.PAD_ID, dtype=torch.long)
        for i, ids in enumerate(inputs):
            padded[i, : len(ids)] = ids

        cats_tensor = torch.tensor(cats, dtype=torch.long, device=device)
        padded = padded.to(device)

        with torch.no_grad():
            te = model.token_emb(padded)
            ce = model.category_vector(cats_tensor).unsqueeze(1).expand(-1, te.size(1), -1)
            x = torch.cat([te, ce], dim=-1)
            out, _ = model.lstm(x)
            hidden = out[:, -1, :]

        all_embeddings.append(hidden.cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    return embeddings, all_names, all_categories


def plot_tsne(
    embeddings: np.ndarray,
    labels: list[str],
    categories: list[str],
    output_path: str = "outputs/tsne.png",
    perplexity: int = 30,
    seed: int = 42,
) -> str:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    n_samples = len(embeddings)
    actual_perplexity = min(perplexity, n_samples - 1)

    tsne = TSNE(
        n_components=2,
        perplexity=actual_perplexity,
        random_state=seed,
        max_iter=1000,
    )
    coords = tsne.fit_transform(embeddings)

    unique_cats = sorted(set(categories))
    try:
        cmap = plt.get_cmap("tab10").resampled(len(unique_cats))
    except AttributeError:  # matplotlib < 3.9
        cmap = plt.cm.get_cmap("tab10", len(unique_cats))
    color_map = {cat: cmap(i) for i, cat in enumerate(unique_cats)}

    fig, ax = plt.subplots(figsize=(12, 8))
    for cat in unique_cats:
        mask = [c == cat for c in categories]
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=[color_map[cat]],
            label=cat,
            alpha=0.6,
            s=15,
        )

    ax.legend(fontsize=10)
    ax.set_title("t-SNE: Name Embeddings by Category", fontsize=14)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def visualize(
    model: NameGenerator,
    tokenizer: Tokenizer,
    corpus_path: str,
    output_path: str = "outputs/tsne.png",
    device: torch.device = torch.device("cpu"),
) -> str:
    embeddings, names, categories = collect_embeddings(
        model, tokenizer, corpus_path, device
    )
    return plot_tsne(embeddings, names, categories, output_path)
