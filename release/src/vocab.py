from __future__ import annotations

import json
from pathlib import Path
from collections import Counter


class Vocab:
    PAD = "<pad>"
    BOS = "<bos>"
    EOS = "<eos>"
    UNK = "<unk>"
    SPECIALS = [PAD, BOS, EOS, UNK]
    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    UNK_ID = 3

    def __init__(self) -> None:
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self.vocab_size: int = 0

    def build_from_tokens(
        self, tokens: list[str], min_freq: int = 1
    ) -> None:
        counter = Counter(tokens)
        self.token_to_id = {
            special: i for i, special in enumerate(self.SPECIALS)
        }
        idx = len(self.SPECIALS)
        for token, freq in counter.most_common():
            if freq >= min_freq and token not in self.token_to_id:
                self.token_to_id[token] = idx
                idx += 1
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.vocab_size = len(self.token_to_id)

    def token2id(self, token: str) -> int:
        return self.token_to_id.get(token, self.UNK_ID)

    def id2token(self, id_: int) -> str:
        return self.id_to_token.get(id_, self.UNK)

    def save(self, path: Path | str, start_counts: dict[str, int] | None = None) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "token_to_id": self.token_to_id,
                    "vocab_size": self.vocab_size,
                    "start_counts": start_counts or {},
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def load(cls, path: Path | str) -> tuple[Vocab, dict[str, int]]:
        """Returns (vocab, start_counts). Old files lack start_counts → {}."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        v = cls()
        v.token_to_id = data["token_to_id"]
        v.id_to_token = {id_: token for token, id_ in v.token_to_id.items()}
        v.vocab_size = data["vocab_size"]
        return v, data.get("start_counts") or {}
