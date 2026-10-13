from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .model import NameGenerator
from .dataset import create_dataloaders
from .vocab import Vocab

HISTORY_FILE = "checkpoints/training_history.json"


def train_epoch(
    model: nn.Module,
    dataloader,
    optimizer,
    criterion,
    device: torch.device,
) -> tuple[float, list[float], float, float]:
    model.train()
    total_loss = 0.0
    n_tokens = 0
    correct = 0
    top5_correct = 0
    batch_losses: list[float] = []
    for inputs, targets, cats in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        cats = cats.to(device)

        optimizer.zero_grad()
        logits, _ = model(inputs, cats)
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        mask = targets != Vocab.PAD_ID
        bsz = mask.sum().item()
        total_loss += loss.item() * bsz
        n_tokens += bsz
        batch_losses.append(loss.item())

        with torch.no_grad():
            flat_logits = logits.reshape(-1, logits.size(-1))
            flat_targets = targets.reshape(-1)
            flat_mask = mask.reshape(-1)
            masked_logits = flat_logits[flat_mask]
            masked_targets = flat_targets[flat_mask]
            if masked_logits.numel() > 0:
                top1 = masked_logits.argmax(dim=-1)
                correct += (top1 == masked_targets).sum().item()
                top5 = masked_logits.topk(min(5, masked_logits.size(-1)), dim=-1).indices
                top5_correct += (top5 == masked_targets.unsqueeze(-1)).any(dim=-1).sum().item()

    acc = correct / max(n_tokens, 1)
    top5_acc = top5_correct / max(n_tokens, 1)
    return total_loss / max(n_tokens, 1), batch_losses, acc, top5_acc


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    dataloader,
    criterion,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    n_tokens = 0
    correct = 0
    top5_correct = 0
    for inputs, targets, cats in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        cats = cats.to(device)

        logits, _ = model(inputs, cats)
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )

        mask = targets != Vocab.PAD_ID
        bsz = mask.sum().item()
        total_loss += loss.item() * bsz
        n_tokens += bsz

        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_targets = targets.reshape(-1)
        flat_mask = mask.reshape(-1)
        masked_logits = flat_logits[flat_mask]
        masked_targets = flat_targets[flat_mask]
        if masked_logits.numel() > 0:
            top1 = masked_logits.argmax(dim=-1)
            correct += (top1 == masked_targets).sum().item()
            top5 = masked_logits.topk(min(5, masked_logits.size(-1)), dim=-1).indices
            top5_correct += (top5 == masked_targets.unsqueeze(-1)).any(dim=-1).sum().item()

    acc = correct / max(n_tokens, 1)
    top5_acc = top5_correct / max(n_tokens, 1)
    return total_loss / max(n_tokens, 1), acc, top5_acc


def train(
    corpus_path: str | None = None,
    checkpoint_dir: str = "checkpoints",
    epochs: int = 40,
    batch_size: int = 32,
    max_len: int = 15,
    lr: float = 1e-3,
    patience: int = 5,
) -> list[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_dl, val_dl, test_dl, cat_to_id, tokenizer = create_dataloaders(
        corpus_path, batch_size=batch_size, max_len=max_len
    )
    tokenizer.save(f"{checkpoint_dir}/tokenizer.json")

    model = NameGenerator(
        vocab_size=tokenizer.vocab.vocab_size,
        n_categories=len(cat_to_id),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")
    print(f"Vocab: {tokenizer.vocab.vocab_size}")
    print(f"Categories: {cat_to_id}")
    print(f"Train: {len(train_dl)} batches, Val: {len(val_dl)} batches")

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss(ignore_index=Vocab.PAD_ID)

    ckpt_path = Path(checkpoint_dir)
    ckpt_path.mkdir(exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0
    history: list[dict] = []
    history_path = Path(checkpoint_dir) / "training_history.json"

    if history_path.exists():
        history_path.unlink()
    history_path.write_text("[]", encoding="utf-8")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, batch_losses, train_acc, train_top5 = train_epoch(
            model, train_dl, optimizer, criterion, device
        )
        val_loss, val_acc, val_top5 = eval_epoch(model, val_dl, criterion, device)
        elapsed = time.time() - t0

        train_ppl = torch.exp(torch.tensor(train_loss)).item()
        val_ppl = torch.exp(torch.tensor(val_loss)).item()
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        epoch_data = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "train_ppl": round(train_ppl, 4),
            "val_ppl": round(val_ppl, 4),
            "train_acc": round(train_acc, 4),
            "val_acc": round(val_acc, 4),
            "train_top5": round(train_top5, 4),
            "val_top5": round(val_top5, 4),
            "lr": current_lr,
            "time": round(elapsed, 2),
            "batch_losses": [round(b, 4) for b in batch_losses],
            "samples": {},
            "saved": False,
        }

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train_loss={train_loss:.4f} ppl={train_ppl:.2f} acc={train_acc:.1%} top5={train_top5:.1%} | "
            f"val_loss={val_loss:.4f} ppl={val_ppl:.2f} acc={val_acc:.1%} top5={val_top5:.1%} | "
            f"{elapsed:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            epoch_data["saved"] = True
            torch.save(
                {
                    "model": model.state_dict(),
                    "vocab_size": tokenizer.vocab.vocab_size,
                    "n_categories": len(cat_to_id),
                    "cat_to_id": cat_to_id,
                    "conditioning": "mean-pool",
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_top5": val_top5,
                    "epoch": epoch,
                },
                ckpt_path / "best.pt",
            )
            print(f"  -> saved best (val_loss={val_loss:.4f})")

            from .generate import normalize_category_ids, temperature_generate
            samples: dict[str, list[str]] = {}
            # Pair corpus → sample a few pair labels to keep history small.
            id_to_cat = {v: k for k, v in cat_to_id.items()}
            sample_labels = sorted({r["label"] for r in train_dl.dataset.data})[:6]
            if not sample_labels:
                sample_labels = sorted(cat_to_id.keys())
            for label in sample_labels:
                try:
                    pair_ids = normalize_category_ids(label, cat_to_id)
                except ValueError:
                    continue
                names: list[str] = []
                for _ in range(2):
                    names.append(temperature_generate(
                        model, tokenizer, pair_ids,
                        temperature=0.9, top_k=30, device=device,
                    ))
                samples[label] = names
            epoch_data["samples"] = samples
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        history.append(epoch_data)
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
    return history


if __name__ == "__main__":
    train()
