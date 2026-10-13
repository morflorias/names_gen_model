from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .dataset import pair_label
from .model import NameGenerator
from .tokenizer import Tokenizer
from .vocab import Vocab


def normalize_category_ids(
    category_ids: int | str | Sequence[int] | Sequence[str],
    cat_to_id: dict[str, int] | None = None,
) -> list[int]:
    """Accept int / [ints] / names / pair strings → list of category ids.

    Examples: ``3`` → ``[3]``; ``[0, 2]`` → ``[0, 2]``;
    ``"cosmetics+food"`` / ``"cosmetics,food"`` → ``[2, 4]`` (needs cat_to_id).
    """
    if isinstance(category_ids, int):
        return [category_ids]
    if isinstance(category_ids, str):
        if cat_to_id is None:
            raise ValueError("cat_to_id is required to resolve category names")
        parts = [p.strip() for p in category_ids.replace(",", "+").split("+") if p.strip()]
        try:
            return [cat_to_id[p] for p in parts]
        except KeyError as e:
            raise ValueError(
                f"Unknown category: {e}. Available: {', '.join(sorted(cat_to_id))}"
            ) from e
    ids: list[int] = []
    for c in category_ids:
        if isinstance(c, str):
            if cat_to_id is None:
                raise ValueError("cat_to_id is required to resolve category names")
            if c not in cat_to_id:
                raise ValueError(
                    f"Unknown category: {c!r}. Available: {', '.join(sorted(cat_to_id))}"
                )
            ids.append(cat_to_id[c])
        else:
            ids.append(int(c))
    if not ids:
        raise ValueError("At least one category is required")
    return ids


def resolve_prefix_ids(
    prefix: str | Sequence[int] | int | None,
    tokenizer: Tokenizer,
) -> list[int]:
    """Accept None / token string / id / id list → validated list of ids.

    A string is treated as ONE initial token from the tokenizer (not free
    text) and resolved via ``resolve_start_token``.
    """
    if prefix is None:
        return []
    if isinstance(prefix, int):
        return [prefix]
    if isinstance(prefix, str):
        return [tokenizer.resolve_start_token(prefix)]
    return [int(t) for t in prefix]


@dataclass
class BeamCandidate:
    """One candidate path in beam search."""
    seq: list[int]
    score: float
    hidden: tuple[torch.Tensor, torch.Tensor] | None = None

    def clone_hidden(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.hidden is None:
            return None
        return (self.hidden[0].clone(), self.hidden[1].clone())


@dataclass
class BeamStepRecord:
    """All candidates at one beam search step (for visualization)."""
    step: int
    candidates: list[dict]   # {seq, score, token_id, token, pruned}
    surviving: list[int]     # indices of surviving candidates
    topk_log_probs: list[float] | None = None
    topk_token_ids: list[int] | None = None


@dataclass
class BeamSearchResult:
    """Full beam search output with trace data."""
    results: list[tuple[str, float]]
    steps: list[BeamStepRecord]


def _category_tensor(
    category_ids: int | str | Sequence[int] | Sequence[str],
    cat_to_id: dict[str, int] | None,
    device: torch.device,
    batch: int = 1,
) -> torch.Tensor:
    ids = normalize_category_ids(category_ids, cat_to_id)
    return torch.tensor([ids] * batch, dtype=torch.long, device=device)


def greedy_generate(
    model: NameGenerator,
    tokenizer: Tokenizer,
    category_ids: int | str | Sequence[int] | Sequence[str],
    prefix_ids: str | Sequence[int] | int | None = None,
    max_len: int = 15,
    device: torch.device = torch.device("cpu"),
    cat_to_id: dict[str, int] | None = None,
) -> str:
    model.eval()
    cat_tensor = _category_tensor(category_ids, cat_to_id, device)
    prefix = resolve_prefix_ids(prefix_ids, tokenizer)
    with torch.no_grad():
        tokens = model.generate(cat_tensor, prefix_ids=prefix, max_len=max_len, temperature=1.0)
    return tokenizer.decode(tokens[0])


def temperature_generate(
    model: NameGenerator,
    tokenizer: Tokenizer,
    category_ids: int | str | Sequence[int] | Sequence[str],
    prefix_ids: str | Sequence[int] | int | None = None,
    max_len: int = 15,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    device: torch.device = torch.device("cpu"),
    cat_to_id: dict[str, int] | None = None,
) -> str:
    model.eval()
    cat_tensor = _category_tensor(category_ids, cat_to_id, device)
    prefix = resolve_prefix_ids(prefix_ids, tokenizer)
    with torch.no_grad():
        tokens = model.generate(
            cat_tensor,
            prefix_ids=prefix,
            max_len=max_len,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
    return tokenizer.decode(tokens[0])


def beam_search_generate(
    model: NameGenerator,
    tokenizer: Tokenizer,
    category_ids: int | str | Sequence[int] | Sequence[str],
    prefix_ids: str | Sequence[int] | int | None = None,
    max_len: int = 15,
    beam_width: int = 5,
    length_penalty: float = 0.6,
    device: torch.device = torch.device("cpu"),
    return_trace: bool = False,
    cat_to_id: dict[str, int] | None = None,
) -> list[tuple[str, float]] | BeamSearchResult:
    """Beam search with hidden state caching and optional forced prefix.

    Each beam carries its own hidden state. At each step only the last
    token is fed to the LSTM — O(L) complexity instead of O(L²).

    If *return_trace* is True, returns a :class:`BeamSearchResult` that
    also contains per-step candidate data for visualization.
    """
    model.eval()
    cat_tensor = _category_tensor(category_ids, cat_to_id, device)
    prefix = resolve_prefix_ids(prefix_ids, tokenizer)
    inv_vocab = {v: k for k, v in tokenizer.vocab.token_to_id.items()}

    # Warm up the shared hidden state on [<bos>] + prefix (teacher forcing).
    start_seq = [Vocab.BOS_ID] + prefix
    if prefix:
        forced = torch.tensor([start_seq], dtype=torch.long, device=device)
        with torch.no_grad():
            _, warmed_hidden = model(forced, cat_tensor)
    else:
        warmed_hidden = None

    def _clone_hidden(h):
        if h is None:
            return None
        return (h[0].clone(), h[1].clone())

    beams: list[BeamCandidate] = [
        BeamCandidate(seq=start_seq, score=0.0, hidden=_clone_hidden(warmed_hidden))
    ]

    step_records: list[BeamStepRecord] = []
    remaining = max(0, max_len - len(prefix))

    for step_i in range(remaining):
        all_candidates: list[BeamCandidate] = []
        raw_scores: list[tuple[int, int, float]] = []  # (parent_idx, token_id, log_prob)

        for bi, beam in enumerate(beams):
            if beam.seq[-1] == Vocab.EOS_ID:
                all_candidates.append(BeamCandidate(seq=beam.seq, score=beam.score))
                continue

            input_token = torch.tensor([[beam.seq[-1]]], dtype=torch.long, device=device)
            cat_input = cat_tensor  # shape (1, K)
            with torch.no_grad():
                logits, new_hidden = model(input_token, cat_input, beam.hidden)

            log_probs = F.log_softmax(logits[0, -1, :], dim=-1)
            topk_log_probs, topk_ids = log_probs.topk(beam_width)

            for k_i in range(beam_width):
                tok_id = topk_ids[k_i].item()
                tok_score = topk_log_probs[k_i].item()
                candidate = BeamCandidate(
                    seq=beam.seq + [tok_id],
                    score=beam.score + tok_score,
                    hidden=new_hidden,
                )
                all_candidates.append(candidate)
                raw_scores.append((bi, tok_id, tok_score))

        # keep top beam_width
        all_candidates.sort(key=lambda c: c.score, reverse=True)
        prev_beams = beams
        beams = all_candidates[:beam_width]

        # determine surviving vs pruned indices
        surviving_set = set()
        prev_to_surviving: dict[int, list[int]] = {}
        for new_i, cand in enumerate(all_candidates):
            if new_i < beam_width:
                surviving_set.add(new_i)
                # find parent
                for pi, beam in enumerate(prev_beams):
                    if cand.seq[:-1] == beam.seq:
                        prev_to_surviving.setdefault(pi, []).append(new_i)
                        break

        # build step record
        if return_trace:
            inv_v = {v: k for k, v in tokenizer.vocab.token_to_id.items()}
            candidates_data = []
            for ci, cand in enumerate(all_candidates):
                tok_id = cand.seq[-1] if cand.seq else -1
                tok_str = inv_v.get(tok_id, "?") if tok_id >= 0 else ""
                candidates_data.append({
                    "seq": cand.seq[:],
                    "score": cand.score,
                    "token_id": tok_id,
                    "token": tok_str,
                    "pruned": ci >= beam_width,
                })
            step_records.append(BeamStepRecord(
                step=step_i,
                candidates=candidates_data,
                surviving=list(range(min(beam_width, len(all_candidates)))),
            ))

        if all(c.seq[-1] == Vocab.EOS_ID for c in beams):
            break

    # format results
    results: list[tuple[str, float]] = []
    for beam in beams:
        length = max(len(beam.seq) - 1, 1)
        normalized_score = beam.score / (length ** length_penalty)
        text = tokenizer.decode(beam.seq)
        results.append((text, normalized_score))
    results.sort(key=lambda x: x[1], reverse=True)

    if return_trace:
        return BeamSearchResult(results=results, steps=step_records)
    return results


# ── backward-compatible wrapper used by main app ──────────────────────────────

def beam_search_generate_compat(
    model: NameGenerator,
    tokenizer: Tokenizer,
    category_ids: int | str | Sequence[int] | Sequence[str],
    max_len: int = 15,
    beam_width: int = 5,
    length_penalty: float = 0.6,
    device: torch.device = torch.device("cpu"),
    cat_to_id: dict[str, int] | None = None,
) -> list[tuple[str, float]]:
    """Same as beam_search_generate but always returns plain list."""
    result = beam_search_generate(
        model, tokenizer, category_ids,
        max_len=max_len, beam_width=beam_width,
        length_penalty=length_penalty, device=device,
        return_trace=False, cat_to_id=cat_to_id,
    )
    assert isinstance(result, list)
    return result
