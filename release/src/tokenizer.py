from __future__ import annotations

import re
from collections import Counter

from .vocab import Vocab

_BRAND_APOSTROPHE = "\u2019"
_TOKEN_RE = re.compile(
    r"[0-9]+(?:[.,][0-9]+)?[%‰µ°]?|[%‰µ°]+"
    r"|[a-zа-яё]+(?:[" + re.escape(_BRAND_APOSTROPHE) + r"'][a-zа-яё]+)?"
    r"|[a-zа-яё]+"
)


class Tokenizer:
    def __init__(
        self,
        vocab: Vocab | None = None,
        lowercase: bool = True,
    ) -> None:
        self.vocab = vocab or Vocab()
        self.lowercase = lowercase
        # First-token frequencies over the texts seen in fit().
        # Powers the "начальный токен" feature: the user picks the first
        # word of the name from actually observed options instead of
        # generating from an empty (bare <bos>) prefix.
        self.start_counts: Counter[str] = Counter()

    def tokenize(self, text: str) -> list[str]:
        if text is None:
            return []
        s = str(text)
        if self.lowercase:
            s = s.lower()
        s = s.replace(_BRAND_APOSTROPHE, "\u2019")
        s = re.sub(r"[(){}\[\]\"`]", " ", s)
        s = re.sub(r"[,;:!?/\\|+=*@#^~]", " ", s)
        s = re.sub(r"['\u2019](?=[a-zа-яё])", "\u2019", s)
        s = re.sub(r"[^a-zа-яё0-9\s" + re.escape(_BRAND_APOSTROPHE) + r"%‰µ°\.\-_/]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return _TOKEN_RE.findall(s)

    def fit(self, texts: list[str], min_freq: int = 1) -> None:
        all_tokens: list[str] = []
        self.start_counts = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            all_tokens.extend(tokens)
            if tokens:
                self.start_counts[tokens[0]] += 1
        self.vocab.build_from_tokens(all_tokens, min_freq)

    def start_tokens(
        self, top_k: int | None = None, min_freq: int = 1
    ) -> list[tuple[str, int]]:
        """Available initial tokens sorted by frequency (token, count).

        Falls back to the whole vocabulary (alphabetical) when no
        start statistics were saved — e.g. checkpoints trained before
        this feature existed.
        """
        items = [
            (tok, cnt)
            for tok, cnt in self.start_counts.most_common()
            if cnt >= min_freq and tok in self.vocab.token_to_id
        ]
        if not items:
            items = sorted(
                (t, 0)
                for t in self.vocab.token_to_id
                if t not in Vocab.SPECIALS
            )
        if top_k is not None:
            items = items[:top_k]
        return items

    def resolve_start_token(self, token: str) -> int:
        """Map a user-picked initial token to its vocab id.

        Raises ValueError with the available options if the token is
        unknown or never starts a name — the prefix must come from the
        tokenizer's observed initial tokens, not free text. (Checkpoints
        trained before this feature have no start statistics; then any
        vocab token is accepted.)
        """
        key = token.lower() if self.lowercase else token
        if key not in self.vocab.token_to_id:
            raise ValueError(
                f"Unknown start token: {token!r}. "
                f"Pick one of: {', '.join(t for t, _ in self.start_tokens(20))}..."
            )
        if self.start_counts and key not in self.start_counts:
            raise ValueError(
                f"Token {token!r} never starts a name in the training corpus. "
                f"Pick one of: {', '.join(t for t, _ in self.start_tokens(20))}..."
            )
        return self.vocab.token2id(key)

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        tokens = self.tokenize(text)
        ids = [self.vocab.token2id(t) for t in tokens]
        if add_bos:
            ids = [Vocab.BOS_ID] + ids
        if add_eos:
            ids = ids + [Vocab.EOS_ID]
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        tokens: list[str] = []
        for id_ in ids:
            token = self.vocab.id2token(id_)
            if skip_special and token in Vocab.SPECIALS:
                if token == Vocab.EOS:
                    break
                continue
            tokens.append(token)
        return " ".join(tokens)

    def save(self, path: str) -> None:
        self.vocab.save(path, start_counts=dict(self.start_counts))

    @classmethod
    def load(cls, path: str) -> Tokenizer:
        vocab, start_counts = Vocab.load(path)
        tok = cls(vocab=vocab)
        tok.start_counts = Counter(start_counts or {})
        return tok
