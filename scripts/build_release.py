"""
Build script: assembles a self-contained release/ folder with
the trained model, tokenizer, minimal src modules, and a
generation-only Streamlit UI.

Usage:
    python scripts/build_release.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
RELEASE = ROOT / "release"

# Source files to copy into release/src/
# (generate.py needs dataset.py for pair labels)
SRC_MODULES = [
    "__init__.py",
    "vocab.py",
    "tokenizer.py",
    "model.py",
    "dataset.py",
    "generate.py",
]

# Checkpoint files to copy into release/checkpoints/
CHECKPOINT_FILES = [
    "best.pt",
    "tokenizer.json",
]


def build() -> None:
    print(f"Building release in {RELEASE} ...")

    # Clean previous build — only generated dirs (app.py, run scripts,
    # requirements.txt are committed and must survive a rebuild).
    RELEASE.mkdir(parents=True, exist_ok=True)
    for _d in ("src", "app", "checkpoints"):
        _p = RELEASE / _d
        if _p.exists():
            shutil.rmtree(_p)
        _p.mkdir(parents=True)

    # Copy src modules
    src_dir = ROOT / "src"
    for mod in SRC_MODULES:
        src_file = src_dir / mod
        if not src_file.exists():
            print(f"  [WARN] {src_file} not found, skipping")
            continue

        if mod == "__init__.py":
            # Write a minimal __init__.py for release (no train/evaluate/visualize)
            content = (
                "\n"
                "from .vocab import Vocab\n"
                "from .tokenizer import Tokenizer\n"
                "from .model import NameGenerator\n"
                "from .generate import temperature_generate, beam_search_generate\n"
                "\n"
            )
            (RELEASE / "src" / "__init__.py").write_text(content, encoding="utf-8")
            print(f"  [GEN]  src/__init__.py (minimal)")
        else:
            dst = RELEASE / "src" / mod
            shutil.copy2(src_file, dst)
            print(f"  [COPY] src/{mod}")

    # Copy checkpoints
    ckpt_dir = ROOT / "checkpoints"
    for f in CHECKPOINT_FILES:
        src_file = ckpt_dir / f
        if not src_file.exists():
            print(f"  [WARN] checkpoints/{f} not found, skipping")
            continue
        dst = RELEASE / "checkpoints" / f
        shutil.copy2(src_file, dst)
        print(f"  [COPY] checkpoints/{f}")

    # Per-pair start-token manifest for the release UI dropdown
    # (release has no corpus file, so token options ship precomputed).
    try:
        sys.path.insert(0, str(ROOT))
        from src.dataset import load_corpus
        from src.tokenizer import Tokenizer
        from collections import Counter, defaultdict

        rows = load_corpus(ROOT / "data" / "corpus_pairs.csv")
        _tok = Tokenizer()
        _by_pair: dict[str, list] = defaultdict(list)
        for _r in rows:
            _toks = _tok.tokenize(_r["name"])
            if _toks:
                _by_pair["+".join(sorted(_r["categories"]))].append((_toks[0], _r["name"]))
        _manifest: dict = {"pairs": sorted(_by_pair), "starts": {}}
        for _pair in sorted(_by_pair):
            _cnt = Counter(t for t, _ in _by_pair[_pair])
            _ex: dict[str, Counter] = {}
            for _t, _n in _by_pair[_pair]:
                _ex.setdefault(_t, Counter())[_n] += 1
            _manifest["starts"][_pair] = [
                [_t, _cnt[_t], _ex[_t].most_common(1)[0][0]]
                for _t, _ in _cnt.most_common(25)
            ]
        (RELEASE / "checkpoints" / "pair_starts.json").write_text(
            json.dumps(_manifest, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(f"  [GEN]  checkpoints/pair_starts.json "
              f"({len(_manifest['pairs'])} pairs)")
    except Exception as e:
        print(f"  [WARN] pair_starts.json not generated: {e}")

    # Release UI is release/app.py (standalone, generation only).
    app_path = RELEASE / "app.py"
    if not app_path.exists():
        print("  [WARN] release/app.py not found — create it before using the release")
    else:
        print("  [OK]   app.py")

    print()
    print("Done! To run the release UI:")
    print(f"  cd {RELEASE}")
    print("  streamlit run app.py")
    print()
    print("Or use the launch script:")
    print(f"  {RELEASE / 'run.bat'}   (Windows)")
    print(f"  {RELEASE / 'run.sh'}    (Linux/Mac)")


if __name__ == "__main__":
    build()
