from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_prefix(prefix_ids: Sequence[int] | Sequence[Sequence[int]] | torch.Tensor | None) -> list[int] | list[list[int]] | None:
    """Pass through prefix ids; model.generate broadcasts a single list."""
    if prefix_ids is None:
        return None
    if isinstance(prefix_ids, torch.Tensor):
        return prefix_ids.tolist()  # type: ignore[return-value]
    return list(prefix_ids)  # type: ignore[arg-type]


class NameGenerator(nn.Module):
    """LSTM name generator conditioned on 1+ categories + optional prefix.

    Conditioning is mean-pooled over the category embeddings, so a name can
    belong to two (or more) categories at once — e.g. coconut oil is both
    ``food`` and ``cosmetics``. A single category behaves exactly like the
    old single-embedding model.
    """

    def __init__(
        self,
        vocab_size: int,
        n_categories: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.n_categories = n_categories
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.token_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.cat_emb = nn.Embedding(n_categories, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim * 2,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, vocab_size)

    def category_vector(self, categories: torch.Tensor) -> torch.Tensor:
        """(B,) or (B,K) category ids → (B, embed_dim) via mean pooling."""
        emb = self.cat_emb(categories)
        if emb.dim() == 3:
            emb = emb.mean(dim=1)
        return emb

    def forward(
        self,
        tokens: torch.Tensor,
        categories: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        te = self.token_emb(tokens)
        ce = self.category_vector(categories).unsqueeze(1).expand(-1, te.size(1), -1)
        x = torch.cat([te, ce], dim=-1)
        out, hidden = self.lstm(x, hidden)
        logits = self.head(out)
        return logits, hidden

    def _prepare_batch(
        self,
        categories: torch.Tensor,
        prefix_ids: Sequence[int] | None,
    ) -> tuple[torch.Tensor, list[int], torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None, list[list[int]]]:
        """Warm up the LSTM on ``[<bos>] + prefix`` (teacher forcing).

        Returns (categories, prefix_list, next_input, hidden, prefix_columns).
        """
        from .vocab import Vocab

        batch_size = categories.size(0)
        device = categories.device
        prefix = list(prefix_ids or [])
        if prefix:
            forced = torch.full(
                (batch_size, len(prefix) + 1), Vocab.BOS_ID,
                dtype=torch.long, device=device,
            )
            forced[:, 1:] = torch.tensor(prefix, dtype=torch.long, device=device).unsqueeze(0)
            with torch.no_grad():
                _, hidden = self.forward(forced, categories)
            next_input = torch.tensor([prefix[-1]], dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, 1)
            prefix_cols = [torch.full((batch_size, 1), t, dtype=torch.long, device=device) for t in prefix]
        else:
            hidden = None
            next_input = torch.full((batch_size, 1), Vocab.BOS_ID, dtype=torch.long, device=device)
            prefix_cols = []
        return categories, prefix, next_input, hidden, prefix_cols

    @torch.no_grad()
    def generate(
        self,
        categories: torch.Tensor,
        prefix_ids: Sequence[int] | None = None,
        max_len: int = 15,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> list[list[int]]:
        """Sample continuations for a batch.

        *categories* is (B,) or (B,K). *prefix_ids* is a single forced
        start (broadcast to the batch) — the returned sequences include
        the prefix tokens. At most ``max_len`` tokens total after <bos>.
        """
        from .vocab import Vocab

        batch_size = categories.size(0)
        device = categories.device

        _, prefix, input_tokens, hidden, all_tokens = self._prepare_batch(categories, prefix_ids)
        remaining = max(0, max_len - len(prefix))
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(remaining):
            logits, hidden = self.forward(input_tokens, categories, hidden)
            logits = logits[:, -1, :] / temperature

            if top_k > 0:
                topk_vals, _ = logits.topk(top_k, dim=-1)
                logits[logits < topk_vals[:, -1:]] = float("-inf")

            if top_p < 1.0:
                sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                mask = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[mask] = float("-inf")
                logits.scatter_(1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            all_tokens.append(next_token)

            # Track which sequences hit EOS
            finished = finished | (next_token.squeeze(-1) == Vocab.EOS_ID)
            if finished.all():
                break

            # Only continue for unfinished sequences
            input_tokens = next_token

        if not all_tokens:
            return [[] for _ in range(batch_size)]
        return torch.cat(all_tokens, dim=1).tolist()

    @torch.no_grad()
    def trace_generate(
        self,
        categories: torch.Tensor | Sequence[int],
        prefix_ids: Sequence[int] | None = None,
        max_len: int = 15,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> dict:
        """Single-sequence generation with a full activation trace.

        *categories* is one id, a list of ids, or a (1,) / (1,K) tensor.
        Forced prefix steps are recorded with ``forced: True``.
        """
        from .vocab import Vocab
        if not isinstance(categories, torch.Tensor):
            ids = [categories] if isinstance(categories, int) else list(categories)
            device = next(self.parameters()).device
            cat_batch = torch.tensor([ids], dtype=torch.long, device=device)
        else:
            cat_batch = categories.view(1, -1) if categories.dim() == 1 and categories.numel() > 1 else categories.view(1) if categories.dim() == 0 else categories
            if cat_batch.dim() == 1:
                cat_batch = cat_batch.unsqueeze(0)
            device = cat_batch.device
        cat_emb_step = self.category_vector(cat_batch)

        prefix = list(prefix_ids or [])
        forced_seq = [Vocab.BOS_ID] + prefix

        steps: list[dict] = []
        hidden = None

        # Forced warmup: BOS + prefix tokens (no sampling).
        input_token = torch.tensor([[forced_seq[0]]], dtype=torch.long, device=device)
        for forced_id in forced_seq[1:]:
            tok_emb = self.token_emb(input_token)
            x = torch.cat([tok_emb, cat_emb_step.unsqueeze(1)], dim=-1)
            out, hidden = self.lstm(x, hidden)
            logits = self.head(out[:, -1, :])
            probs = F.softmax(logits / temperature, dim=-1)
            steps.append({
                "step": len(steps),
                "input_token_id": input_token.item(),
                "chosen_token_id": forced_id,
                "forced": True,
                "token_emb": tok_emb.squeeze().cpu().numpy(),
                "logits": logits.squeeze().cpu().numpy(),
                "probs": probs.squeeze().cpu().numpy(),
                "hidden": [h.squeeze().cpu().numpy() for h in hidden[0]],  # h_n
                "cell": [c.squeeze().cpu().numpy() for c in hidden[1]],    # c_n
            })
            input_token = torch.tensor([[forced_id]], dtype=torch.long, device=device)

        remaining = max(0, max_len - len(prefix))
        for _ in range(remaining):
            tok_emb = self.token_emb(input_token)
            x = torch.cat([tok_emb, cat_emb_step.unsqueeze(1)], dim=-1)

            out, hidden = self.lstm(x, hidden)

            logits = self.head(out[:, -1, :])
            scaled_logits = logits / temperature

            if top_k > 0:
                topk_vals, _ = scaled_logits.topk(top_k, dim=-1)
                scaled_logits[scaled_logits < topk_vals[:, -1:]] = float("-inf")

            if top_p < 1.0:
                sorted_logits, sorted_idx = scaled_logits.sort(dim=-1, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                mask = cum_probs > top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = False

                sorted_logits[mask] = float("-inf")
                scaled_logits.scatter_(1, sorted_idx, sorted_logits)

            probs = F.softmax(scaled_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            chosen_id = next_token.item()

            steps.append({
                "step": len(steps),
                "input_token_id": input_token.item(),
                "chosen_token_id": chosen_id,
                "forced": False,
                "token_emb": tok_emb.squeeze().cpu().numpy(),
                "logits": logits.squeeze().cpu().numpy(),
                "probs": probs.squeeze().cpu().numpy(),
                "hidden": [h.squeeze().cpu().numpy() for h in hidden[0]],  # h_n
                "cell": [c.squeeze().cpu().numpy() for c in hidden[1]],    # c_n
            })

            if chosen_id == Vocab.EOS_ID:
                break
            input_token = next_token

        return {
            "category_ids": cat_batch.squeeze(0).tolist(),
            "category_emb": cat_emb_step.squeeze().cpu().numpy(),
            "steps": steps,
        }
