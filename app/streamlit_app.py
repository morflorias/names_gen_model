from __future__ import annotations

import json
import shutil
import sys
import threading
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.model import NameGenerator
from src.tokenizer import Tokenizer
from src.generate import (
    normalize_category_ids,
    resolve_prefix_ids,
    temperature_generate,
    beam_search_generate,
)
from src.dataset import load_corpus, create_dataloaders, resolve_corpus, pair_label
from src.evaluate import evaluate, generate_samples
from src.visualize import collect_embeddings
from src.vocab import Vocab

ROOT = Path(__file__).parent.parent
CHECKPOINT_DIR = ROOT / "checkpoints"
LEGACY_CORPUS_PATH = ROOT / "data" / "corpus_lmstudio.csv"
PAIRS_CORPUS_PATH = ROOT / "data" / "corpus_pairs.csv"


def resolve_corpus_path() -> Path:
    if PAIRS_CORPUS_PATH.exists():
        return PAIRS_CORPUS_PATH
    return LEGACY_CORPUS_PATH


CORPUS_PATH = resolve_corpus_path()
HISTORY_FILE = CHECKPOINT_DIR / "training_history.json"
TSNE_OUTPUT = ROOT / "outputs" / "tsne.png"


def load_model():
    ckpt_path = CHECKPOINT_DIR / "best.pt"
    tok_path = CHECKPOINT_DIR / "tokenizer.json"
    if not ckpt_path.exists():
        return None, None, None
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    tokenizer = Tokenizer.load(str(tok_path))
    model = NameGenerator(
        vocab_size=ckpt["vocab_size"],
        n_categories=ckpt["n_categories"],
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, tokenizer, ckpt["cat_to_id"]


def load_history() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    try:
        text = HISTORY_FILE.read_text(encoding="utf-8").strip()
        if not text:
            return []
        return json.loads(text)
    except (json.JSONDecodeError, OSError):
        return []


def load_corpus_df() -> pd.DataFrame:
    path = resolve_corpus_path()
    if path.exists():
        data = load_corpus(path)
        df = pd.DataFrame(data)
        if "label" not in df.columns:
            df["label"] = df.get("category", "")
        return df
    return pd.DataFrame(columns=["category", "label", "name"])


def available_start_tokens(tokenizer: Tokenizer, top_k: int = 100) -> list[tuple[str, int]]:
    return tokenizer.start_tokens(top_k=top_k)


@st.cache_data
def corpus_first_tokens(path: str, mtime: float) -> pd.DataFrame:
    """First token + example name per corpus row (for the start-token picker)."""
    from src.tokenizer import Tokenizer as _Tok
    data = load_corpus(path)
    t = _Tok()
    rows = []
    for r in data:
        toks = t.tokenize(r["name"])
        if toks:
            rows.append({
                "cats": "+".join(sorted(r["categories"])),
                "first": toks[0],
                "name": r["name"],
            })
    return pd.DataFrame(rows, columns=["cats", "first", "name"])


def start_options_for(
    tokenizer: Tokenizer,
    categories: list[str],
    top_k: int = 25,
) -> tuple[list[tuple[str, int, str, str]], list[str]]:
    """Start tokens STRICTLY for the chosen categories.

    Returns (options, pairs_covered), where each option is
    (token, count, example_name, pair_tag). Counts and examples are computed
    only over corpus rows whose pair lies inside the selected categories —
    nothing from other pairs leaks into the list.
    """
    try:
        path = resolve_corpus_path()
        df = corpus_first_tokens(str(path), path.stat().st_mtime)
    except (OSError, ValueError):
        df = pd.DataFrame(columns=["cats", "first", "name"])

    sel = set(categories)
    if not df.empty and sel:
        sub = df[df["cats"].apply(lambda c: set(c.split("+")) <= sel)]
        if sub.empty:  # unseen pair combo → any overlap with selection
            sub = df[df["cats"].apply(lambda c: bool(set(c.split("+")) & sel))]
    else:
        sub = df

    vocab = tokenizer.vocab.token_to_id
    if sub.empty:
        # Fallback: global tokenizer stats (e.g. no corpus file).
        return ([(tok, cnt, "", "") for tok, cnt in available_start_tokens(tokenizer, top_k)], [])

    sub = sub[sub["first"].isin(vocab)]
    pairs_covered = sorted(sub["cats"].unique().tolist())
    counts = sub["first"].value_counts().head(top_k)
    examples = sub.groupby("first")["name"].agg(lambda s: s.value_counts().idxmax())
    top_pair = sub.groupby("first")["cats"].agg(lambda s: s.value_counts().idxmax())
    multi = len(pairs_covered) > 1
    options = [
        (tok, int(counts[tok]), str(examples[tok]),
         str(top_pair[tok]) if multi else "")
        for tok in counts.index
    ]
    return options, pairs_covered


def _short_example(name: str, limit: int = 30) -> str:
    name = " ".join(str(name).split())
    return name if len(name) <= limit else name[:limit].rstrip() + "…"


def start_token_selectbox(
    tokenizer: Tokenizer,
    categories: list[str],
    key: str,
    top_k: int = 25,
) -> str | None:
    """Dropdown of start tokens strictly for the chosen categories.

    Each option shows the fixed token, how many corpus names start with it
    and a real example (brand fragments like ``mi``/``h`` become readable
    through the example: ``mi… — Mi&Ko Масло…``).
    Returns the picked token or None (no prefix).
    """
    options, pairs_covered = start_options_for(tokenizer, categories, top_k)
    labels = ["— без префикса —"] + [
        f"{tok}…  ({cnt})"
        + (f"  [{pair}]" if pair else "")
        + (f"  — {_short_example(ex)}" if ex else "")
        for tok, cnt, ex, pair in options
    ]
    if pairs_covered:
        st.caption(f"Токены для пар: {', '.join(pairs_covered)} — всего вариантов: {len(options)}.")
    elif not options:
        st.caption("Нет данных корпуса — показан весь словарь токенизатора.")
    # Reset the choice whenever categories change (options list changes too).
    dyn_key = f"{key}__{'__'.join(sorted(categories))}"
    choice = st.selectbox(
        "Start token",
        labels,
        key=dyn_key,
        help="Фиксируется только ПЕРВЫЙ токен (показан до …), продолжение придумывает модель. "
             "В скобках — сколько названий корпуса начинается с этого токена, дальше пример.",
    )
    if choice == "— без префикса —":
        return None
    return options[labels.index(choice) - 1][0]


def pair_default_options(cat_to_id: dict[str, int]) -> list[str]:
    """Pair labels seen in corpus, else all C(n,2) combos of base categories."""
    import itertools
    df = load_corpus_df()
    if not df.empty and "label" in df.columns:
        labels = sorted(df["label"].dropna().unique().tolist())
        if labels:
            return labels
    cats = sorted(cat_to_id.keys())
    return ["+".join(p) for p in itertools.combinations(cats, 2)]


def tab_training():
    st.header("Training")

    col_cfg, col_status = st.columns([1, 1])

    with col_cfg:
        st.subheader("Configuration")
        epochs = st.number_input("Epochs", min_value=1, max_value=5000, value=40, step=1,
                                  help="Сколько раз модель пройдёт по всему корпусу. Больше = дольше обучение, но риск переобучения.")
        batch_size = st.number_input("Batch size", min_value=8, max_value=128, value=32, step=8,
                                     help="Сколько имён обрабатывается за один шаг. Больше = быстрее, но больше памяти GPU. Меньше = градиенты шумят, но лучше обобщает.")
        lr = st.number_input("Learning rate", min_value=1e-5, max_value=1e-1, value=1e-3, format="%.1e", step=1e-4,
                             help="Шаг обновления весов. Слишком высокий → модель расходится. Слишком низкий → сходится очень медленно. 1e-3 — хороший старт для Adam.")
        patience = st.number_input("Early stopping patience", min_value=1, max_value=30, value=5, step=1,
                                   help="Остановить обучение, если val loss не улучшался столько эпох. Защита от переобучения. 5 = ждём 5 эпох и стоп.")
        max_len = st.number_input("Max sequence length", min_value=5, max_value=30, value=15, step=1,
                                  help="Максимум токенов в одном имени. Длинные обрезаются при обучении. Также ограничивает длину при генерации.")
        corpus_options = []
        if PAIRS_CORPUS_PATH.exists():
            corpus_options.append("data/corpus_pairs.csv")
        if LEGACY_CORPUS_PATH.exists():
            corpus_options.append("data/corpus_lmstudio.csv")
        if not corpus_options:
            corpus_options = ["data/corpus_pairs.csv"]
        corpus_file = st.selectbox(
            "Corpus",
            corpus_options,
            index=0,
            help="Парный корпус: CSV со столбцами 'category_a,category_b,name' (гибрид принадлежит обеим категориям). Legacy: 'name,category'.",
        )

    with col_status:
        st.subheader("Status")
        history = load_history()
        ckpt_exists = (CHECKPOINT_DIR / "best.pt").exists()

        if ckpt_exists:
            ckpt = torch.load(CHECKPOINT_DIR / "best.pt", map_location="cpu", weights_only=False)
            st.success(f"Model saved: epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")
        else:
            st.warning("No trained model found.")

        if history:
            last = history[-1]
            st.info(
                f"History: {len(history)} epochs | "
                f"Last: train={last['train_loss']:.4f} val={last['val_loss']:.4f}"
            )

    st.divider()

    if "training_running" not in st.session_state:
        st.session_state.training_running = False
    if "training_log" not in st.session_state:
        st.session_state.training_log = []

    col_start, col_stop = st.columns([1, 1])
    with col_start:
        start_disabled = st.session_state.training_running
        if st.button("Start Training", disabled=start_disabled, type="primary"):
            st.session_state.training_running = True
            st.session_state.training_done = False
            st.session_state.training_log = []
            st.rerun()
    with col_stop:
        if st.button("Stop Training", disabled=not st.session_state.training_running):
            st.session_state.training_running = False
            st.rerun()

    if st.session_state.training_running and not st.session_state.get("training_done", False):
        log_placeholder = st.empty()
        progress_bar = st.progress(0, text="Training...")

        if "train_poll_count" not in st.session_state:
            st.session_state.train_poll_count = 0
        st.session_state.training_done = False

        def run_training():
            from src.train import train as train_fn
            try:
                history_out = train_fn(
                    corpus_path=str(ROOT / corpus_file),
                    checkpoint_dir=str(CHECKPOINT_DIR),
                    epochs=epochs,
                    batch_size=batch_size,
                    max_len=max_len,
                    lr=lr,
                    patience=patience,
                )
                st.session_state.training_log = history_out
            except Exception as e:
                st.session_state.training_log = [{"error": str(e)}]

        thread = threading.Thread(target=run_training, daemon=True)
        thread.start()

        while thread.is_alive():
            time.sleep(2.0)
            fresh_history = load_history()
            if fresh_history:
                last_ep = fresh_history[-1]
                progress = min(fresh_history[-1]["epoch"] / max(epochs, 1), 1.0)
                progress_bar.progress(
                    progress,
                    text=f"Epoch {last_ep['epoch']}/{epochs} | "
                         f"train={last_ep['train_loss']:.4f} val={last_ep['val_loss']:.4f}",
                )
                poll_key = st.session_state.train_poll_count
                st.session_state.train_poll_count += 1
                with log_placeholder.container():
                    df_h = pd.DataFrame(fresh_history)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df_h["epoch"], y=df_h["train_loss"],
                        name="Train loss", mode="lines+markers",
                    ))
                    fig.add_trace(go.Scatter(
                        x=df_h["epoch"], y=df_h["val_loss"],
                        name="Val loss", mode="lines+markers",
                    ))
                    fig.update_layout(
                        height=300, margin=dict(l=0, r=0, t=30, b=0),
                        xaxis_title="Epoch", yaxis_title="Loss",
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"train_live_{poll_key}")

        st.session_state.training_running = False
        st.session_state.training_done = True
        progress_bar.progress(1.0, text="Training complete!")
        st.rerun()

    history = load_history()
    if history:
        st.subheader("Training History")
        df_h = pd.DataFrame(history)
        cols_to_show = [
            c for c in ["epoch", "train_loss", "val_loss", "train_ppl", "val_ppl",
                        "train_acc", "val_acc", "train_top5", "val_top5",
                        "lr", "time", "saved"]
            if c in df_h.columns
        ]
        st.dataframe(
            df_h[cols_to_show],
            use_container_width=True,
            hide_index=True,
        )

    st.divider()
    st.subheader("Export to release")
    release_dir = ROOT / "release"
    release_ckpt = release_dir / "checkpoints" / "best.pt"
    release_tok = release_dir / "checkpoints" / "tokenizer.json"

    if not release_dir.exists():
        st.warning("Folder `release/` not found. Create it first (see AGENTS.md).")
    elif not ckpt_exists:
        st.warning("No trained model to export.")
    else:
        if st.button("Выгрузить веса в release", type="primary",
                      help="Copy best.pt, tokenizer.json and the pair_starts manifest into release/checkpoints/"):
            try:
                release_ckpt.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(CHECKPOINT_DIR / "best.pt", release_ckpt)
                shutil.copy2(CHECKPOINT_DIR / "tokenizer.json", release_tok)
                # Refresh the start-token manifest so the release dropdown
                # matches the exported vocab/corpus.
                from collections import Counter, defaultdict
                _rows = load_corpus(resolve_corpus_path())
                _t = Tokenizer()
                _by_pair: dict[str, list] = defaultdict(list)
                for _r in _rows:
                    _toks = _t.tokenize(_r["name"])
                    if _toks:
                        _by_pair["+".join(sorted(_r["categories"]))].append((_toks[0], _r["name"]))
                _manifest: dict = {"pairs": sorted(_by_pair), "starts": {}}
                for _pair in sorted(_by_pair):
                    _cnt = Counter(t for t, _ in _by_pair[_pair])
                    _ex: dict[str, Counter] = {}
                    for _t2, _n in _by_pair[_pair]:
                        _ex.setdefault(_t2, Counter())[_n] += 1
                    _manifest["starts"][_pair] = [
                        [_t3, _cnt[_t3], _ex[_t3].most_common(1)[0][0]]
                        for _t3, _ in _cnt.most_common(25)
                    ]
                _manifest_path = release_ckpt.parent / "pair_starts.json"
                _manifest_path.write_text(
                    json.dumps(_manifest, ensure_ascii=False, indent=1), encoding="utf-8")
                st.success(
                    f"Exported to `release/checkpoints/`:\n"
                    f"- best.pt ({release_ckpt.stat().st_size / 1024:.0f} KB)\n"
                    f"- tokenizer.json ({release_tok.stat().st_size / 1024:.0f} KB)\n"
                    f"- pair_starts.json ({len(_manifest['pairs'])} pairs)"
                )
            except Exception as e:
                st.error(f"Export failed: {e}")


def _layer_selection(step: dict, n_layers: int, top_n: int) -> list[dict]:
    layers_data: list[dict] = []
    layers_data.append({"name": "Input", "values": np.asarray(step["token_emb"]).flatten()})
    for li in range(n_layers):
        layers_data.append({
            "name": f"LSTM L{li+1}",
            "values": np.asarray(step["hidden"][li]).flatten(),
        })
    layers_data.append({"name": "Output", "values": np.asarray(step["logits"]).flatten()})

    selected: list[dict] = []
    for layer in layers_data:
        v = layer["values"]
        abs_v = np.abs(v)
        if abs_v.max() == 0:
            top_idx = np.arange(min(top_n, len(v)))
        else:
            top_idx = abs_v.argsort()[-top_n:][::-1]
        selected.append({
            "name": layer["name"],
            "values": v,
            "indices": [int(i) for i in top_idx],
            "activations": [float(v[i]) for i in top_idx],
        })
    return selected


def _with_opacity(color: str, opacity: float) -> str:
    if color.startswith("#"):
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        return f"rgba({r}, {g}, {b}, {opacity})"
    import re as _re
    m = _re.match(r"rgba?\((\d+),\s*(\d+),\s*(\d+)", color)
    if m:
        return f"rgba({m.group(1)}, {m.group(2)}, {m.group(3)}, {opacity})"
    return color


def _node_color(act: float, layer_name: str) -> str:
    if layer_name == "Input":
        return "#4ECDC4"
    if layer_name == "Output":
        return "#FFD93D"
    intensity = min(1.0, abs(act) / 4.0)
    return f"rgba(255, 165, 80, {0.3 + 0.7 * intensity})"


def _build_sankey(step: dict, n_layers: int, top_n: int,
                  inv_vocab: dict[int, str]) -> go.Sankey:
    selected = _layer_selection(step, n_layers, top_n)
    n_layers_total = len(selected)

    node_labels: list[str] = []
    node_colors: list[str] = []
    node_x: list[float] = []
    node_y: list[float] = []

    layer_offsets: list[int] = []
    for li, layer in enumerate(selected):
        layer_offsets.append(len(node_labels))
        n_nodes = len(layer["indices"])
        for ni, (idx, act) in enumerate(zip(layer["indices"], layer["activations"])):
            token_label = ""
            if layer["name"] == "Input":
                token_label = inv_vocab.get(idx, "")
                token_label = f"'{token_label}' "
            elif layer["name"] == "Output":
                token_label = inv_vocab.get(idx, "")
                token_label = f"'{token_label}' "
            node_labels.append(f"{token_label}#{idx}<br>{act:+.3f}")
            node_colors.append(_node_color(act, layer["name"]))
            node_x.append(li / max(n_layers_total - 1, 1))
            node_y.append(ni / max(n_nodes - 1, 1) if n_nodes > 1 else 0.5)

    source: list[int] = []
    target: list[int] = []
    value: list[float] = []
    link_colors: list[str] = []

    for li in range(n_layers_total - 1):
        src = selected[li]
        dst = selected[li + 1]
        src_off = layer_offsets[li]
        dst_off = layer_offsets[li + 1]
        for si, sa in enumerate(src["activations"]):
            for di, da in enumerate(dst["activations"]):
                flow = abs(sa * da)
                if flow < 1e-4:
                    continue
                source.append(src_off + si)
                target.append(dst_off + di)
                value.append(flow)
                link_colors.append(_with_opacity(node_colors[src_off + si], 0.3))

    return go.Sankey(
        arrangement="fixed",
        node=dict(
            pad=18, thickness=18,
            line=dict(color="rgba(80,80,80,0.4)", width=0.5),
            label=node_labels, color=node_colors,
            x=node_x, y=node_y,
        ),
        link=dict(
            source=source, target=target, value=value,
            color=link_colors,
        ),
    )


def build_network_diagram(
    steps: list[dict],
    n_layers: int,
    token_to_id: dict[str, int],
    inv_vocab: dict[int, str] | None = None,
    step_index: int = -1,
    top_n_neurons: int = 6,
) -> go.Figure:
    if not steps:
        return go.Figure()
    step = steps[step_index if step_index >= 0 else len(steps) - 1]
    if inv_vocab is None:
        inv_vocab = {v: k for k, v in token_to_id.items()}

    sankey = _build_sankey(step, n_layers, top_n_neurons, inv_vocab)
    in_label = inv_vocab.get(step["input_token_id"], "?")
    out_label = inv_vocab.get(step["chosen_token_id"], "?")

    fig = go.Figure(sankey)
    fig.update_layout(
        height=460,
        margin=dict(l=20, r=20, t=50, b=30),
        title=dict(
            text=f"Activation path — step {step_index + 1}: "
                 f"<span style='color:#2A8B85'>{in_label}</span> → "
                 f"<span style='color:#B89A1F'>{out_label}</span>",
            x=0.5, xanchor="center", font=dict(size=13),
        ),
    )
    return fig


def build_animated_network_diagram(
    steps: list[dict],
    n_layers: int,
    token_to_id: dict[str, int],
    inv_vocab: dict[int, str] | None = None,
    top_n_neurons: int = 6,
    frame_duration_ms: int = 800,
) -> go.Figure:
    if not steps:
        return go.Figure()
    if inv_vocab is None:
        inv_vocab = {v: k for k, v in token_to_id.items()}

    sankey0 = _build_sankey(steps[0], n_layers, top_n_neurons, inv_vocab)
    fig = go.Figure(sankey0)

    frames: list[go.Frame] = []
    for i, step in enumerate(steps):
        sk = _build_sankey(step, n_layers, top_n_neurons, inv_vocab)
        in_label = inv_vocab.get(step["input_token_id"], "?")
        out_label = inv_vocab.get(step["chosen_token_id"], "?")
        frames.append(go.Frame(
            data=[sk],
            name=f"step{i}",
            layout=dict(
                title=dict(
                    text=f"Activation path — step {i+1}/{len(steps)}: "
                         f"<span style='color:#2A8B85'>{in_label}</span> → "
                         f"<span style='color:#B89A1F'>{out_label}</span>",
                    x=0.5, xanchor="center", font=dict(size=13),
                ),
            ),
        ))
    fig.frames = frames

    in0 = inv_vocab.get(steps[0]["input_token_id"], "?")
    out0 = inv_vocab.get(steps[0]["chosen_token_id"], "?")
    fig.update_layout(
        height=460,
        margin=dict(l=20, r=20, t=60, b=30),
        title=dict(
            text=f"Activation path — step 1/{len(steps)}: "
                 f"<span style='color:#2A8B85'>{in0}</span> → "
                 f"<span style='color:#B89A1F'>{out0}</span>",
            x=0.5, xanchor="center", font=dict(size=13),
        ),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=1.12, x=0.0, xanchor="left",
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, {"frame": {"duration": frame_duration_ms, "redraw": True},
                                  "fromcurrent": True, "transition": {"duration": 200}}]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate", "transition": {"duration": 0}}]),
                dict(label="⏮ Reset", method="animate",
                     args=[[f"step0"], {"frame": {"duration": 0, "redraw": True},
                                        "mode": "immediate", "transition": {"duration": 0}}]),
            ],
        )],
        sliders=[dict(
            active=0,
            steps=[dict(
                args=[[f"step{i}"],
                      {"frame": {"duration": 0, "redraw": True},
                       "mode": "immediate", "transition": {"duration": 0}}],
                label=f"{i+1}",
                method="animate",
            ) for i in range(len(steps))],
            x=0.0, y=-0.05, xanchor="left", len=0.95,
            currentvalue=dict(prefix="Step: ", visible=True, xanchor="right"),
            transition=dict(duration=0),
        )],
    )
    return fig


# ── Beam Search Diagram 1: probability heatmap ───────────────────────────────

def build_beam_heatmap(
    step_records: list,
    inv_vocab: dict[int, str],
    beam_width: int = 5,
    top_display: int = 8,
) -> go.Figure:
    """Heatmap: X = step, Y = token, color = log-probability.

    Shows the top-K log-probabilities at each beam search step.
    """
    if not step_records:
        return go.Figure()

    all_tokens: set[str] = set()
    cell_data: dict[tuple[int, str], float] = {}
    pruned_tokens: dict[tuple[int, str], bool] = {}

    for rec in step_records:
        for cand in rec.candidates:
            tok = cand["token"]
            if tok in ("<bos>", "<pad>"):
                continue
            key = (rec.step, tok)
            cell_data[key] = cand["score"]
            pruned_tokens[key] = cand["pruned"]
            all_tokens.add(tok)

    if not all_tokens:
        return go.Figure()

    steps = [r.step for r in step_records]
    tokens = sorted(all_tokens)

    z: list[list[float]] = []
    text: list[list[str]] = []
    hover: list[list[str]] = []
    for tok in tokens:
        row_z = []
        row_text = []
        row_hover = []
        for step in steps:
            val = cell_data.get((step, tok), float("nan"))
            row_z.append(val)
            pruned = pruned_tokens.get((step, tok), False)
            status = " (pruned)" if pruned else ""
            row_text.append(f"{tok}{status}")
            row_hover.append(
                f"Step {step+1}: '{tok}'<br>log-prob: {val:.3f}{status}"
            )
        z.append(row_z)
        text.append(row_text)
        hover.append(row_hover)

    fig = go.Figure(go.Heatmap(
        z=z,
        x=[f"Step {s+1}" for s in steps],
        y=tokens,
        text=text,
        hovertext=hover,
        hoverinfo="text",
        colorscale="RdYlGn",
        colorbar=dict(title="log P"),
    ))
    fig.update_layout(
        height=max(300, 40 * len(tokens)),
        margin=dict(l=20, r=20, t=50, b=30),
        title=dict(
            text="Beam Search — token probabilities per step",
            x=0.5, xanchor="center", font=dict(size=13),
        ),
        xaxis=dict(title="Step"),
        yaxis=dict(title="Token"),
    )
    return fig


# ── Beam Search Diagram 2: tree (Sankey) of beam paths ───────────────────────

def build_beam_tree(
    step_records: list,
    inv_vocab: dict[int, str],
    beam_width: int = 5,
) -> go.Figure:
    """Sankey diagram of beam search paths.

    Surviving beams in teal, pruned beams in red.  Link width = score magnitude.
    """
    if not step_records:
        return go.Figure()

    # node_id → (step, token, pruned, score)
    nodes: list[tuple[int, str, bool, float]] = []
    node_labels: list[str] = []
    node_colors: list[str] = []

    def _add_node(step: int, tok: str, pruned: bool, score: float) -> int:
        idx = len(nodes)
        nodes.append((step, tok, pruned, score))
        short = tok if len(tok) <= 8 else tok[:7] + "…"
        node_labels.append(f"{short}  {score:.2f}")
        node_colors.append(
            "rgba(220, 80, 80, 0.55)" if pruned else "#4ECDC4"
        )
        return idx

    # root BOS
    bos_idx = _add_node(-1, "<bos>", False, 0.0)

    # per-step maps: token → node_index  (only surviving)
    step_maps: list[dict[str, int]] = [{}, {}]
    step_maps[0] = {}  # step 0 will map from BOS

    source: list[int] = []
    target: list[int] = []
    value: list[float] = []
    link_colors: list[str] = []

    for ri, rec in enumerate(step_records):
        curr_map: dict[str, int] = {}
        prev_map = step_maps[ri] if ri < len(step_maps) else step_maps[-1]

        for cand in rec.candidates:
            tok = cand["token"]
            if tok in ("<bos>", "<pad>"):
                continue
            child_idx = _add_node(rec.step, tok, cand["pruned"], cand["score"])
            curr_map[tok] = child_idx

            # determine parent node
            if ri == 0:
                parent = bos_idx
            else:
                # parent = previous step's token
                if len(cand["seq"]) >= 2:
                    parent_tok = inv_vocab.get(cand["seq"][-2], "?")
                else:
                    parent_tok = "<bos>"
                parent = prev_map.get(parent_tok, bos_idx)

            flow = abs(cand["score"]) + 1.0
            source.append(parent)
            target.append(child_idx)
            value.append(flow)
            link_colors.append(
                "rgba(220, 80, 80, 0.25)" if cand["pruned"]
                else "rgba(78, 205, 196, 0.35)"
            )

        if len(step_maps) <= ri + 1:
            step_maps.append(curr_map)
        else:
            step_maps[ri + 1] = curr_map

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=15, thickness=14,
            line=dict(color="rgba(80,80,80,0.4)", width=0.5),
            label=node_labels, color=node_colors,
        ),
        link=dict(
            source=source, target=target, value=value,
            color=link_colors,
        ),
    ))
    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=50, b=20),
        title=dict(
            text="Beam Search — candidate paths  "
                 "<span style='color:#4ECDC4'>■ survived</span>  "
                 "<span style='color:#DC5050'>■ pruned</span>",
            x=0.5, xanchor="center", font=dict(size=13),
        ),
    )
    return fig


def tab_generation():
    GEN_MAX_TOKENS = 10
    st.header("Name Generation")

    model, tokenizer, cat_to_id = load_model()
    if model is None:
        st.error("No trained model found. Run training first.")
        return

    col_ctrl, col_out = st.columns([1, 1])

    with col_ctrl:
        st.subheader("Parameters")
        categories = st.multiselect(
            "Categories (выбери 2–3)",
            sorted(cat_to_id.keys()),
            default=sorted(cat_to_id.keys())[:2],
            max_selections=3,
            help="Гибридный товар принадлежит всем выбранным категориям сразу "
                 "(например, кокосовое масло — это и еда, и косметика).",
        )
        prefix_tok = start_token_selectbox(tokenizer, categories, key="gen_start", top_k=30)
        method = st.radio("Method", ["Temperature", "Beam search"], horizontal=True)

        # defaults so rendering code never hits NameError when the
        # opposite method's slider wasn't instantiated
        temperature: float = 0.9
        top_k: int = 30
        beam_width: int = st.session_state.get("beam_width", 5)

        if method == "Temperature":
            temperature = st.slider("Temperature", 0.1, 2.0, 0.9, 0.05)
            top_k = st.slider("Top-k", 0, 100, 30, 5)
            n_names = st.slider("Count", 1, 20, 5)
        else:
            beam_width = st.slider("Beam width", 1, 10, 5, key="gen_beam_width")
            n_names = st.slider("Count", 1, 10, 5)

        if st.button("Generate", type="primary", key="gen_btn"):
            if len(categories) < 2:
                st.error("Выбери минимум 2 категории — генерируем гибрид.")
                st.stop()
            cat_ids = normalize_category_ids(list(categories), cat_to_id)
            try:
                prefix = resolve_prefix_ids(prefix_tok, tokenizer)
            except ValueError as e:
                st.error(str(e))
                st.stop()
            pair_name = "+".join(sorted(categories))
            inv_vocab = {v: k for k, v in tokenizer.vocab.token_to_id.items()}
            t0 = time.perf_counter()
            with st.spinner("Generating..."):
                names: list[str] = []
                trace_steps = None
                trace_labels: list[str] = []
                if method == "Temperature":
                    # First name via trace_generate → Sankey matches it
                    trace = model.trace_generate(
                        cat_ids,
                        prefix_ids=prefix,
                        max_len=GEN_MAX_TOKENS,
                        temperature=temperature,
                        top_k=top_k,
                    )
                    trace_steps = trace["steps"]
                    trace_labels = ["<BOS>"]
                    trace_ids: list[int] = []
                    for s in trace_steps:
                        trace_labels.append(inv_vocab.get(s["chosen_token_id"], "?"))
                        if s["chosen_token_id"] != Vocab.EOS_ID:
                            trace_ids.append(s["chosen_token_id"])
                    first_name = tokenizer.decode(trace_ids)
                    names.append(first_name)

                    # Remaining n-1 names via batch generate
                    if n_names > 1:
                        cat_tensor = torch.tensor([cat_ids] * (n_names - 1))
                        extra_ids = model.generate(
                            cat_tensor,
                            prefix_ids=prefix,
                            max_len=GEN_MAX_TOKENS,
                            temperature=temperature,
                            top_k=top_k,
                        )
                        names.extend(tokenizer.decode(e) for e in extra_ids)
                else:
                    bs_result = beam_search_generate(
                        model, tokenizer, cat_ids,
                        prefix_ids=prefix,
                        max_len=GEN_MAX_TOKENS,
                        beam_width=beam_width,
                        return_trace=True,
                    )
                    names = [b[0] for b in bs_result.results[:n_names]]
                    st.session_state["beam_steps"] = bs_result.steps
                    st.session_state["beam_width"] = beam_width
            elapsed = time.perf_counter() - t0

            st.session_state.generated_names = names
            st.session_state.gen_method = method
            st.session_state.gen_run_id = st.session_state.get("gen_run_id", 0) + 1
            st.session_state["gen_elapsed"] = elapsed
            st.session_state["gen_category"] = pair_name
            st.session_state["gen_prefix"] = prefix_tok or "—"
            if trace_steps is not None:
                st.session_state["gen_trace_steps"] = trace_steps
                st.session_state["gen_trace_labels"] = trace_labels

    with col_out:
        st.subheader("Results")
        if "generated_names" in st.session_state:
            run_id = st.session_state.get("gen_run_id", 0)
            names_list = st.session_state.generated_names
            n_total = len(names_list)
            lengths = [len(n) for n in names_list] if n_total else [0]
            n_unique = len(set(names_list)) if n_total else 0
            avg_len = sum(lengths) / n_total if n_total else 0.0
            min_len = min(lengths) if lengths else 0
            max_len_actual = max(lengths) if lengths else 0
            elapsed = st.session_state.get("gen_elapsed", 0.0)
            cat_used = st.session_state.get("gen_category", "?")
            prefix_used = st.session_state.get("gen_prefix", "—")

            st.markdown("##### Overall Result")
            m1, m2, m3 = st.columns(3)
            m1.metric("Total", n_total)
            m2.metric("Unique", n_unique)
            m3.metric("Avg length", f"{avg_len:.1f}")
            m4, m5, m6 = st.columns(3)
            m4.metric("Min length", min_len)
            m5.metric("Max length", max_len_actual)
            m6.metric("Time", f"{elapsed*1000:.0f} ms")
            st.caption(
                f"Categories: **{cat_used}** · Start: **{prefix_used}** · "
                f"Method: **{st.session_state.get('gen_method', method)}**"
            )
            st.code(" | ".join(names_list) if names_list else "(empty)", language=None)

            st.markdown("##### Per-name")
            for i, name in enumerate(names_list, 1):
                st.text_input(f"#{i}", name, key=f"gen_{run_id}_{i}", disabled=True)
        else:
            st.info("Configure parameters and click Generate.")

        if st.session_state.get("generated_names"):
            gen_method = st.session_state.get("gen_method", "Temperature")

            # ── Temperature: Network Diagram ──
            if gen_method == "Temperature":
                has_trace = bool(st.session_state.get("gen_trace_steps"))
                if has_trace:
                    st.markdown("##### Network Diagram (Temperature)")
                    animate = st.checkbox(
                        "Animate (step through generation)",
                        value=True, key="net_animate",
                        help="Show all steps as an animated sequence with Play / Pause / Reset and a slider.",
                    )

                    if animate:
                        fig_net = build_animated_network_diagram(
                            st.session_state["gen_trace_steps"],
                            model.n_layers,
                            tokenizer.vocab.token_to_id,
                            inv_vocab=None,
                            top_n_neurons=6,
                            frame_duration_ms=800,
                        )
                        st.plotly_chart(
                            fig_net, use_container_width=True,
                            key=f"net_anim_{run_id}",
                            config={"displayModeBar": True, "responsive": True},
                        )
                        st.caption(
                            f"Animated trace over {len(st.session_state['gen_trace_steps'])} steps. "
                            "Use the ▶ Play / ⏸ Pause / ⏮ Reset buttons or drag the slider."
                        )
                    else:
                        step_choice = st.selectbox(
                            "Step",
                            options=list(range(len(st.session_state.get("gen_trace_steps", [])))),
                            index=len(st.session_state.get("gen_trace_steps", [])) - 1,
                            key="net_step",
                            format_func=lambda i: (
                                f"Step {i+1}: {st.session_state.get('gen_trace_labels', ['?'])[i]}"
                                if i < len(st.session_state.get("gen_trace_labels", []))
                                else f"Step {i+1}"
                            ),
                        )
                        fig_net = build_network_diagram(
                            st.session_state["gen_trace_steps"],
                            model.n_layers,
                            tokenizer.vocab.token_to_id,
                            inv_vocab=None,
                            step_index=step_choice,
                        )
                        st.plotly_chart(fig_net, use_container_width=True,
                                        key=f"net_{run_id}_{step_choice}",
                                        config={"displayModeBar": True, "responsive": True})
                        st.caption(
                            "Sankey flow width = |source activation × dest activation|. "
                            "Teal = input, orange = LSTM, gold = output. "
                            "Hover a link to see the exact flow value."
                        )

            # ── Beam Search: diagrams ──
            if gen_method == "Beam search" and st.session_state.get("beam_steps"):
                st.markdown("##### Beam Search Visualization")
                inv_vocab = {v: k for k, v in tokenizer.vocab.token_to_id.items()}
                bs_steps = st.session_state["beam_steps"]
                bw = int(st.session_state.get("beam_width", beam_width if "beam_width" in locals() else 5))

                bs_tab1, bs_tab2 = st.tabs(["📊 Probabilities heatmap", "🌳 Beam paths tree"])

                with bs_tab1:
                    fig_heat = build_beam_heatmap(bs_steps, inv_vocab, bw)
                    st.plotly_chart(fig_heat, use_container_width=True,
                                    key=f"beam_heat_{run_id}",
                                    config={"displayModeBar": True, "responsive": True})
                    st.caption(
                        "Each cell = log-probability of a token at a given step. "
                        "Darker green = higher probability.  "
                        "Hover for details."
                    )

                with bs_tab2:
                    fig_tree = build_beam_tree(bs_steps, inv_vocab, bw)
                    st.plotly_chart(fig_tree, use_container_width=True,
                                    key=f"beam_tree_{run_id}",
                                    config={"displayModeBar": True, "responsive": True})
                    st.caption(
                        "Teal = surviving beams, red = pruned candidates.  "
                        "Link width = score magnitude."
                    )

    st.divider()
    st.subheader("Model Info")
    total_params = sum(p.numel() for p in model.parameters())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Parameters", f"{total_params:,}")
    c2.metric("Vocab size", tokenizer.vocab.vocab_size)
    c3.metric("Base categories", len(cat_to_id))
    ckpt = torch.load(CHECKPOINT_DIR / "best.pt", map_location="cpu", weights_only=False)
    c4.metric("Best epoch", ckpt["epoch"])
    st.caption(
        f"Conditioning: mean-pooled CategoryEmbedding "
        f"({ckpt.get('conditioning', 'single')}) · "
        f"Start tokens available: {len(tokenizer.start_tokens())} · "
        f"Pairs: {', '.join(pair_default_options(cat_to_id)[:6])}"
        f"{'…' if len(pair_default_options(cat_to_id)) > 6 else ''}"
    )


def tab_benchmarks():
    st.header("Benchmarks & Evaluation")

    model, tokenizer, cat_to_id = load_model()
    if model is None:
        st.error("No trained model found. Run training first.")
        return

    n_per_cat = st.slider("Samples per category pair", 10, 200, 50, 10)

    if st.button("Run Evaluation", type="primary"):
        with st.spinner("Evaluating... this may take a moment"):
            scores = evaluate(
                model, tokenizer,
                str(resolve_corpus_path()),
                cat_to_id,
                n_per_cat=n_per_cat,
            )
            st.session_state.eval_scores = scores

            generated = generate_samples(
                model, tokenizer, cat_to_id,
                n_per_cat=n_per_cat,
                labels=pair_default_options(cat_to_id),
            )
            data = load_corpus(resolve_corpus_path())
            real_names: dict[str, list[str]] = {}
            for item in data:
                real_names.setdefault(item["label"], []).append(item["name"])
            st.session_state.eval_generated = generated
            st.session_state.eval_real = real_names

    if "eval_scores" in st.session_state:
        scores = st.session_state.eval_scores

        st.subheader("Overall Metrics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("BLEU", f"{scores['bleu']:.4f}")
        c2.metric("Jaccard", f"{scores['jaccard']:.4f}")
        c3.metric("Novelty", f"{scores['novelty']:.1%}")
        c4.metric("Cosine", f"{scores['cosine']:.4f}")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=scores["novelty"] * 100,
            title={"text": "Novelty %"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#4CAF50"}},
        ))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

        if "eval_generated" in st.session_state:
            st.subheader("Generated Samples by Category")
            generated = st.session_state.eval_generated
            for cat, names in sorted(generated.items()):
                with st.expander(f"{cat} ({len(names)} names)"):
                    for n in names[:10]:
                        st.write(f"  {n}")

            st.subheader("Category Breakdown")
            cat_data = []
            real = st.session_state.eval_real
            for cat in sorted(generated.keys()):
                gen_count = len(generated[cat])
                real_count = len(real.get(cat, []))
                unique = len(set(generated[cat]))
                cat_data.append({
                    "Category": cat,
                    "Generated": gen_count,
                    "Real": real_count,
                    "Unique": unique,
                })
            st.dataframe(pd.DataFrame(cat_data), use_container_width=True, hide_index=True)


def tab_activations():
    st.header("Network Activations")
    st.caption("Visualize the activation chain during name generation — embeddings, hidden states, probabilities")

    model, tokenizer, cat_to_id = load_model()
    if model is None:
        st.error("No trained model found. Run training first.")
        return

    col_cfg, col_meta = st.columns([1, 2])

    with col_cfg:
        st.subheader("Run")
        act_categories = st.multiselect(
            "Categories (2–3)", sorted(cat_to_id.keys()),
            default=sorted(cat_to_id.keys())[:2],
            max_selections=3, key="act_cats",
        )
        act_prefix_tok = start_token_selectbox(tokenizer, act_categories, key="act_start", top_k=30)
        temperature = st.slider("Temperature", 0.1, 2.0, 0.8, 0.05, key="act_temp")
        top_k = st.slider("Top-k", 0, 100, 30, 5, key="act_topk")
        max_len = st.slider("Max length", 5, 25, 15, 1, key="act_maxlen")

        if st.button("Trace Generation", type="primary"):
            if len(act_categories) < 2:
                st.error("Выбери минимум 2 категории.")
                st.stop()
            with st.spinner("Tracing forward pass..."):
                act_ids = normalize_category_ids(list(act_categories), cat_to_id)
                try:
                    act_prefix = resolve_prefix_ids(act_prefix_tok, tokenizer)
                except ValueError as e:
                    st.error(str(e))
                    st.stop()
                trace = model.trace_generate(
                    act_ids,
                    prefix_ids=act_prefix,
                    max_len=max_len,
                    temperature=temperature,
                    top_k=top_k,
                )
            st.session_state.last_trace = trace
            st.session_state.last_trace_cat = "+".join(sorted(act_categories))
            st.session_state.last_trace_prefix = act_prefix_tok or "—"

    with col_meta:
        st.subheader("Architecture")
        st.markdown(
            f"""
            - **Embedding** `{tokenizer.vocab.vocab_size}×64`
            - **CategoryEmbedding** `{len(cat_to_id)}×64`
            - **LSTM** `2 layers × 128 hidden` (dropout=0.3)
            - **Linear head** `128 → {tokenizer.vocab.vocab_size}`
            - **Total params** `{sum(p.numel() for p in model.parameters()):,}`
            """
        )

    if "last_trace" not in st.session_state:
        st.info("Click **Trace Generation** to inspect the forward pass.")
        return

    trace = st.session_state.last_trace
    steps = trace["steps"]
    if not steps:
        st.warning("No steps generated.")
        return

    inv_vocab = {v: k for k, v in tokenizer.vocab.token_to_id.items()}
    token_labels = ["<BOS>"]
    chosen_ids: list[int] = []
    for s in steps:
        token_labels.append(inv_vocab.get(s["chosen_token_id"], "<unk>"))
        chosen_ids.append(s["chosen_token_id"])
    full_text = " ".join(t for t in token_labels if t not in Vocab.SPECIALS)

    st.divider()
    st.subheader("Generated Sequence")
    st.markdown(f"### `{full_text}`")

    st.divider()

    st.subheader("Per-Token Probability Distribution")
    st.caption("Probability over the vocabulary at each step. The chosen token is highlighted.")

    step_idx = st.slider("Step", 0, len(steps) - 1, 0, key="prob_step")
    cur = steps[step_idx]
    probs = cur["probs"]
    top_n = 15
    top_indices = probs.argsort()[-top_n:][::-1]

    prob_df = pd.DataFrame({
        "token": [inv_vocab.get(int(i), "<unk>") for i in top_indices],
        "probability": [float(probs[i]) for i in top_indices],
        "chosen": [int(i) == cur["chosen_token_id"] for i in top_indices],
    })
    prob_df["color"] = prob_df["chosen"].map({True: "#FF6B6B", False: "#4ECDC4"})

    fig_prob = go.Figure()
    fig_prob.add_trace(go.Bar(
        x=prob_df["probability"],
        y=prob_df["token"],
        orientation="h",
        marker_color=prob_df["color"],
        text=[f"{p:.3f}" for p in prob_df["probability"]],
        textposition="outside",
    ))
    fig_prob.update_layout(
        title=f"Step {step_idx + 1}: input={inv_vocab.get(cur['input_token_id'], '?')} → chosen={inv_vocab.get(cur['chosen_token_id'], '?')}",
        xaxis_title="Probability",
        yaxis_title="Token",
        height=400,
        yaxis=dict(autorange="reversed"),
        margin=dict(l=80, r=20, t=60, b=40),
    )
    st.plotly_chart(fig_prob, use_container_width=True)

    prob_path = [float(probs[c]) for probs, c in zip([s["probs"] for s in steps], chosen_ids)]
    prob_df_path = pd.DataFrame({
        "step": list(range(1, len(prob_path) + 1)),
        "chosen_token": [inv_vocab.get(c, "?") for c in chosen_ids],
        "probability": prob_path,
    })
    fig_path = px.line(
        prob_df_path, x="step", y="probability",
        title="Confidence in Chosen Token Across Steps",
        markers=True, hover_data=["chosen_token"],
        labels={
            "step": "Шаг генерации (позиция токена в названии)",
            "probability": "P(выбранный токен)",
            "chosen_token": "Токен",
        },
    )
    fig_path.update_layout(height=300, yaxis_title="P(выбранный токен)")
    st.plotly_chart(fig_path, use_container_width=True)

    st.divider()
    st.subheader("Hidden State Activations")
    st.caption("LSTM hidden state per layer. Rows = hidden dim, columns = generation step. Bright = active.")

    layer_idx = st.radio("Layer", [0, 1], horizontal=True, key="hidden_layer")
    hidden_matrix = np.array([s["hidden"][layer_idx] for s in steps])
    hidden_matrix = hidden_matrix.T

    fig_hid = px.imshow(
        hidden_matrix,
        x=[f"{i+1}:{inv_vocab.get(chosen_ids[i], '?')}" for i in range(len(steps))],
        color_continuous_scale="RdBu_r",
        aspect="auto",
        title=f"Layer {layer_idx + 1} Hidden States (128 dim × {len(steps)} steps)",
        labels={"x": "Step: Token", "y": "Hidden dim", "color": "Activation"},
    )
    fig_hid.update_layout(height=450, margin=dict(l=0, r=0, t=60, b=80))
    fig_hid.update_xaxes(tickangle=45)
    st.plotly_chart(fig_hid, use_container_width=True)

    st.divider()
    st.subheader("Cell State (LSTM Memory)")
    st.caption("Long-term memory cell. Patterns here indicate what the LSTM 'remembers' at each step.")

    cell_layer = st.radio("Layer", [0, 1], horizontal=True, key="cell_layer")
    cell_matrix = np.array([s["cell"][cell_layer] for s in steps]).T

    fig_cell = px.imshow(
        cell_matrix,
        x=[f"{i+1}:{inv_vocab.get(chosen_ids[i], '?')}" for i in range(len(steps))],
        color_continuous_scale="Viridis",
        aspect="auto",
        title=f"Layer {cell_layer + 1} Cell States",
        labels={"x": "Step: Token", "y": "Cell dim", "color": "Value"},
    )
    fig_cell.update_layout(height=450, margin=dict(l=0, r=0, t=60, b=80))
    fig_cell.update_xaxes(tickangle=45)
    st.plotly_chart(fig_cell, use_container_width=True)

    st.divider()
    st.subheader("Activation Dynamics")
    st.caption("How strongly each layer activates at each step (L2 norm).")

    norms: dict[str, list[float]] = {}
    for layer_idx in range(model.n_layers):
        norms[f"Layer {layer_idx + 1} hidden"] = [
            float(np.linalg.norm(s["hidden"][layer_idx])) for s in steps
        ]
        norms[f"Layer {layer_idx + 1} cell"] = [
            float(np.linalg.norm(s["cell"][layer_idx])) for s in steps
        ]
    norms["Token embedding"] = [float(np.linalg.norm(s["token_emb"])) for s in steps]
    norms["Logits"] = [float(np.linalg.norm(s["logits"])) for s in steps]
    fig_norms = go.Figure()
    for name, values in norms.items():
        fig_norms.add_trace(go.Scatter(
            x=list(range(1, len(values) + 1)),
            y=values,
            name=name,
            mode="lines+markers",
        ))
    fig_norms.update_layout(
        xaxis_title="Step",
        yaxis_title="L2 norm",
        height=350,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig_norms, use_container_width=True)

    st.divider()
    st.subheader("Logit Distribution (Raw)")
    st.caption("Logits before softmax. High positive = strong preference, negative = suppressed.")

    logit_step = st.slider("Step", 0, len(steps) - 1, 0, key="logit_step")
    logits = steps[logit_step]["logits"]
    top_logits_idx = logits.argsort()[-20:][::-1]
    logit_df = pd.DataFrame({
        "token": [inv_vocab.get(int(i), "?") for i in top_logits_idx],
        "logit": [float(logits[i]) for i in top_logits_idx],
    })
    fig_logits = px.bar(
        logit_df, x="logit", y="token", orientation="h",
        title=f"Top-20 Logits at Step {logit_step + 1}",
        labels={
            "logit": "Логит (сырой скор до softmax)",
            "token": "Токен",
        },
    )
    fig_logits.update_layout(
        height=450, yaxis=dict(autorange="reversed"),
        margin=dict(l=80, r=20, t=60, b=40),
    )
    st.plotly_chart(fig_logits, use_container_width=True)

    st.divider()
    st.subheader("Neural Network Grid")
    st.caption("Classic neuron grid view. Each column is a layer, each row a neuron. Brightness = activation magnitude.")

    grid_step = st.slider("Generation step", 0, len(steps) - 1, 0, key="grid_step", help="Which timestep of the trace to inspect")
    cur_grid = steps[grid_step]

    grid_cols = st.columns([1, 1, 1])
    with grid_cols[0]:
        st.markdown("**Input token**")
        st.code(inv_vocab.get(cur_grid["input_token_id"], "?"), language=None)
        st.metric("Token embedding ‖·‖", f"{np.linalg.norm(cur_grid['token_emb']):.2f}")

    with grid_cols[1]:
        st.markdown("**Category**")
        st.code(st.session_state.get("last_trace_cat", "?"), language=None)
        st.metric("Category embedding ‖·‖", f"{np.linalg.norm(trace['category_emb']):.2f}")

    with grid_cols[2]:
        st.markdown("**Chosen token**")
        st.code(inv_vocab.get(cur_grid["chosen_token_id"], "?"), language=None)
        st.metric("Logits ‖·‖", f"{np.linalg.norm(cur_grid['logits']):.2f}")

    layer_names = ["Input Embedding", "LSTM Layer 1", "LSTM Layer 2", "Output Logits"]
    layer_data = [
        cur_grid["token_emb"],
        cur_grid["hidden"][0],
        cur_grid["hidden"][1] if model.n_layers > 1 else cur_grid["hidden"][0],
        cur_grid["logits"],
    ]
    n_cols = len(layer_names)
    grid_w = 16
    cell_h = 18
    fig_grid = make_subplots(
        rows=1, cols=n_cols,
        subplot_titles=layer_names,
        horizontal_spacing=0.04,
    )

    for li, (name, vec) in enumerate(zip(layer_names, layer_data)):
        flat = np.asarray(vec).flatten()
        if flat.size % grid_w == 0:
            reshaped = flat.reshape(grid_w, -1)
        else:
            trim = flat[: grid_w * (flat.size // grid_w)]
            reshaped = trim.reshape(grid_w, -1)
        vmax = float(np.max(np.abs(reshaped))) or 1.0
        normalized = reshaped / vmax

        heatmap = go.Heatmap(
            z=normalized,
            colorscale="RdBu_r",
            zmin=-1, zmax=1,
            showscale=(li == n_cols - 1),
            colorbar=dict(title="Activation", x=1.02) if li == n_cols - 1 else None,
            hovertemplate=f"{name}<br>row=%{{y}} col=%{{x}}<br>val=%{{z:.2f}}<extra></extra>",
        )
        fig_grid.add_trace(heatmap, row=1, col=li + 1)

    for _col in range(1, n_cols + 1):
        fig_grid.update_xaxes(showticklabels=False, row=1, col=_col)
        fig_grid.update_yaxes(showticklabels=False, autorange="reversed", row=1, col=_col)
    fig_grid.update_layout(
        height=cell_h * grid_w + 80,
        margin=dict(l=0, r=80, t=40, b=20),
        title_text=f"Step {grid_step + 1}: activations flowing through the network",
    )
    st.plotly_chart(fig_grid, use_container_width=True)

    st.caption("Red = positive activation, Blue = negative. The activation magnitude tells you which neurons are 'firing'.")

    st.subheader("Top Active Neurons per Layer")
    top_k_neurons = 8
    for li, (name, vec) in enumerate(zip(layer_names, layer_data)):
        with st.expander(f"{name} — top {top_k_neurons} active neurons"):
            vec_flat = np.asarray(vec).flatten()
            abs_vals = np.abs(vec_flat)
            top_idx = abs_vals.argsort()[-top_k_neurons:][::-1]
            df_neurons = pd.DataFrame({
                "neuron": [int(i) for i in top_idx],
                "activation": [float(vec_flat[i]) for i in top_idx],
                "abs": [float(abs_vals[i]) for i in top_idx],
            })
            st.dataframe(df_neurons, use_container_width=True, hide_index=True)

    st.subheader("Activation Path Through Network")
    st.caption("Sum of activations (signal strength) at each layer step.")

    signal_layers: list[list[float]] = []
    for s in steps:
        signal_layers.append([
            float(np.mean(np.abs(s["token_emb"]))),
            float(np.mean(np.abs(s["hidden"][0]))),
            float(np.mean(np.abs(s["hidden"][1]))) if model.n_layers > 1 else 0.0,
            float(np.mean(np.abs(s["logits"]))),
        ])

    fig_path = go.Figure()
    layer_legend = ["Input Emb", "LSTM L1", "LSTM L2", "Logits"]
    for li in range(4):
        fig_path.add_trace(go.Scatter(
            x=list(range(1, len(signal_layers) + 1)),
            y=[s[li] for s in signal_layers],
            name=layer_legend[li],
            mode="lines+markers",
            line=dict(width=2),
            marker=dict(size=8),
        ))
    fig_path.update_layout(
        xaxis_title="Generation step",
        yaxis_title="Mean |activation|",
        height=350,
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig_path, use_container_width=True)


def tab_charts():
    st.header("Charts & Visualization")

    model, tokenizer, cat_to_id = load_model()
    chart_type = st.radio(
        "Chart type",
        [
            "Training curves",
            "Per-batch loss",
            "Epoch comparison",
            "Generation snapshots",
            "t-SNE",
            "Corpus stats",
            "Token frequency",
        ],
        horizontal=True,
    )

    if chart_type == "Training curves":
        history = load_history()
        if not history:
            st.info("No training history. Run training first.")
            return

        df_h = pd.DataFrame(history)

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Loss", "Perplexity", "Learning Rate", "Epoch Time"),
        )

        fig.add_trace(
            go.Scatter(x=df_h["epoch"], y=df_h["train_loss"], name="Train", mode="lines+markers"),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=df_h["epoch"], y=df_h["val_loss"], name="Val", mode="lines+markers"),
            row=1, col=1,
        )

        fig.add_trace(
            go.Scatter(x=df_h["epoch"], y=df_h["train_ppl"], name="Train PPL", mode="lines+markers", showlegend=False),
            row=1, col=2,
        )
        fig.add_trace(
            go.Scatter(x=df_h["epoch"], y=df_h["val_ppl"], name="Val PPL", mode="lines+markers", showlegend=False),
            row=1, col=2,
        )

        fig.add_trace(
            go.Scatter(x=df_h["epoch"], y=df_h["lr"], name="LR", mode="lines+markers", showlegend=False, line=dict(color="green")),
            row=2, col=1,
        )

        fig.add_trace(
            go.Bar(x=df_h["epoch"], y=df_h["time"], name="Time", showlegend=False, marker_color="orange"),
            row=2, col=2,
        )

        fig.update_layout(height=600, margin=dict(l=40, r=20, t=40, b=40))
        fig.update_xaxes(title_text="Epoch", row=2, col=1)
        fig.update_xaxes(title_text="Epoch", row=2, col=2)
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="PPL", row=1, col=2)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "t-SNE":
        if model is None:
            st.error("No trained model found.")
            return
        corpus_now = resolve_corpus_path()
        if not corpus_now.exists():
            st.error("Corpus not found.")
            return

        col1, col2, col3 = st.columns(3)
        with col1:
            method = st.selectbox("Method", ["UMAP", "t-SNE", "PCA"], index=0)
        with col2:
            n_sample = st.slider("Sample size", 100, 3000, 1000, 100)
        with col3:
            search_query = st.text_input("Search", placeholder="filter names…")

        with st.spinner(f"Computing {method}…"):
            all_embeddings, all_names, all_categories = collect_embeddings(
                model, tokenizer, str(corpus_now),
            )
            embeddings = all_embeddings

            if len(embeddings) > n_sample:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(embeddings), n_sample, replace=False)
                embeddings = embeddings[idx]
                all_names = [all_names[i] for i in idx]
                all_categories = [all_categories[i] for i in idx]

            if method == "UMAP":
                import umap
                reducer = umap.UMAP(
                    n_components=2,
                    n_neighbors=min(15, len(embeddings) - 1),
                    min_dist=0.1,
                    random_state=42,
                )
            elif method == "t-SNE":
                from sklearn.manifold import TSNE
                perplexity = min(30, len(embeddings) - 1)
                reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
            else:
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=2, random_state=42)

            coords = reducer.fit_transform(embeddings)

        # Оси проекции: у t-SNE/UMAP/PCA нет физических единиц —
        # подписываем честно (латентные измерения), смысл несёт только
        # взаимное расположение точек.
        dim1 = f"{method}, измерение 1 (усл. ед.)"
        dim2 = f"{method}, измерение 2 (усл. ед.)"

        df_emb = pd.DataFrame({
            "x": coords[:, 0],
            "y": coords[:, 1],
            "category": all_categories,
            "name": all_names,
        })

        if search_query:
            mask = df_emb["name"].str.contains(search_query, case=False, na=False)
            df_emb = df_emb[mask].copy()
            if df_emb.empty:
                st.warning(f"No names match '{search_query}'.")
                return

        fig = px.scatter(
            df_emb, x="x", y="y", color="category",
            hover_data={"name": True, "x": ":.2f", "y": ":.2f", "category": True},
            labels={
                "x": dim1,
                "y": dim2,
                "category": "Пара категорий",
                "name": "Название",
            },
            title=f"{method}: проекция last hidden state LSTM (точка = одно название)",
            opacity=0.75,
            category_orders={"category": sorted(df_emb["category"].unique())},
        )
        fig.update_traces(marker=dict(size=7, line=dict(width=0.5, color="white")))
        fig.update_layout(
            height=600,
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            f"Показано точек: {len(df_emb)}. Наведи курсор — название + пара категорий. "
            f"Оси ({dim1} / {dim2}) — латентные измерения без физических единиц: "
            "абсолютные значения не интерпретируются, важен только взаимный разброс — "
            "близкие точки означают похожие скрытые состояния модели."
        )

        st.subheader("Cluster Centroids (per category)")
        centroids = df_emb.groupby("category").agg({"x": "mean", "y": "mean", "name": "count"}).reset_index()
        centroids.columns = ["Category", "x", "y", "count"]
        fig_cent = px.scatter(
            centroids, x="x", y="y", text="Category", size="count",
            color="Category", size_max=40,
            labels={
                "x": dim1,
                "y": dim2,
                "count": "Названий в паре",
                "Category": "Пара категорий",
            },
            title="Центроиды пар категорий (среднее проекции; размер звезды = число названий)",
        )
        fig_cent.update_traces(textposition="top center", marker=dict(symbol="star"))
        fig_cent.update_layout(height=400, margin=dict(l=0, r=0, t=40, b=0), showlegend=False)
        st.plotly_chart(fig_cent, use_container_width=True)

        st.subheader("Per-Category Sample")
        cat_choice = st.selectbox("Pick a category to inspect", sorted(df_emb["category"].unique()))
        sample = df_emb[df_emb["category"] == cat_choice].head(15)
        st.dataframe(
            sample[["name", "x", "y"]].rename(columns={"name": "Название", "x": dim1, "y": dim2}),
            use_container_width=True, hide_index=True,
        )

    elif chart_type == "Corpus stats":
        df = load_corpus_df()
        if df.empty:
            st.info("No corpus found.")
            return

        col1, col2 = st.columns(2)

        group_col = "label" if "label" in df.columns else "category"
        with col1:
            counts = df[group_col].value_counts().sort_values()
            fig_bar = px.bar(
                x=counts.values, y=counts.index, orientation="h",
                title="Names per Category Pair",
                labels={"x": "Count", "y": "Pair"},
                color=counts.index,
            )
            fig_bar.update_layout(height=350, margin=dict(l=0, r=0, t=40, b=0), showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            df["word_count"] = df["name"].apply(lambda x: len(x.split()))
            df["char_count"] = df["name"].apply(len)
            fig_box = px.box(
                df, x=group_col, y="word_count",
                title="Word Count Distribution",
                labels={"word_count": "Words", group_col: ""},
            )
            fig_box.update_layout(height=350, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_box, use_container_width=True)

        st.subheader("Corpus Preview")
        preview_cols = [c for c in ["category_a", "category_b", "label", "category", "name", "word_count", "char_count"] if c in df.columns]
        st.dataframe(
            df[preview_cols].head(50),
            use_container_width=True, hide_index=True,
        )

    elif chart_type == "Token frequency":
        df = load_corpus_df()
        if df.empty:
            st.info("No corpus found.")
            return

        all_tokens = []
        for name in df["name"]:
            all_tokens.extend(name.split())
        counter = Counter(all_tokens)
        top_n = st.slider("Top N tokens", 10, 200, 50, 10)
        most_common = counter.most_common(top_n)

        tokens, counts = zip(*most_common) if most_common else ([], [])
        fig = go.Figure(go.Bar(
            x=list(reversed(counts)),
            y=list(reversed(tokens)),
            orientation="h",
        ))
        fig.update_layout(
            height=max(300, top_n * 18),
            title=f"Top {top_n} Most Frequent Tokens",
            xaxis_title="Frequency",
            yaxis_title="Token",
            margin=dict(l=100, r=20, t=40, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Unique tokens", len(counter))
        with col2:
            st.metric("Total tokens", len(all_tokens))

    elif chart_type == "Per-batch loss":
        history = load_history()
        if not history:
            st.info("No training history. Run training first.")
            return

        rows = []
        for ep in history:
            for bi, bl in enumerate(ep.get("batch_losses", [])):
                rows.append({
                    "epoch": ep["epoch"],
                    "batch": bi + 1,
                    "global_step": (ep["epoch"] - 1) * len(ep.get("batch_losses", [1])) + bi,
                    "loss": bl,
                })
        if not rows:
            st.info("No per-batch data. History entries lack batch_losses.")
            return
        df_b = pd.DataFrame(rows)

        fig = go.Figure()
        for ep in history:
            ep_rows = [r for r in rows if r["epoch"] == ep["epoch"]]
            if ep_rows:
                fig.add_trace(go.Scatter(
                    x=[r["batch"] for r in ep_rows],
                    y=[r["loss"] for r in ep_rows],
                    mode="lines+markers",
                    name=f"epoch {ep['epoch']}",
                    opacity=0.7,
                ))
        fig.update_layout(
            title="Per-Batch Training Loss (per epoch)",
            xaxis_title="Batch",
            yaxis_title="Loss",
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Global View (all batches in sequence)")
        fig_g = px.line(
            df_b, x="global_step", y="loss", color="epoch",
            title="Batch Loss Across All Epochs",
            labels={
                "global_step": "Сквозной индекс батча (эпоха × батч)",
                "loss": "Loss (кросс-энтропия)",
                "epoch": "Эпоха",
            },
        )
        fig_g.update_layout(height=400)
        st.plotly_chart(fig_g, use_container_width=True)

    elif chart_type == "Epoch comparison":
        history = load_history()
        if not history:
            st.info("No training history. Run training first.")
            return

        df_h = pd.DataFrame(history)
        best_idx = int(df_h["val_loss"].idxmin())
        best = df_h.iloc[best_idx]

        st.subheader("Key Metrics Comparison")
        cols = st.columns(min(len(history), 6))
        for i, ep in enumerate(history):
            with cols[i % len(cols)]:
                delta_loss = ep["val_loss"] - best["val_loss"]
                st.metric(
                    f"Epoch {ep['epoch']}",
                    f"{ep['val_loss']:.4f}",
                    delta=f"{delta_loss:+.4f}",
                    delta_color="inverse",
                )

        st.divider()

        epochs_to_show = st.multiselect(
            "Select epochs to compare",
            options=[ep["epoch"] for ep in history],
            default=[ep["epoch"] for ep in history if ep.get("saved")],
        )

        if epochs_to_show:
            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=("Train Loss", "Val Loss", "Perplexity"),
            )
            for ep in history:
                if ep["epoch"] in epochs_to_show:
                    color = "gold" if ep.get("saved") else None
                    width = 3 if ep.get("saved") else 1.5
                    fig.add_trace(go.Scatter(
                        x=[ep["epoch"]], y=[ep["train_loss"]],
                        mode="markers+text",
                        marker=dict(size=14, color=color, line=dict(width=width, color="black")),
                        text=[f"E{ep['epoch']}"], textposition="top center",
                        name=f"Epoch {ep['epoch']}",
                        showlegend=False,
                    ), row=1, col=1)
                    fig.add_trace(go.Scatter(
                        x=[ep["epoch"]], y=[ep["val_loss"]],
                        mode="markers+text",
                        marker=dict(size=14, color=color, line=dict(width=width, color="black")),
                        text=[f"E{ep['epoch']}"], textposition="top center",
                        showlegend=False,
                    ), row=1, col=2)
                    fig.add_trace(go.Scatter(
                        x=[ep["epoch"]], y=[ep["val_ppl"]],
                        mode="markers+text",
                        marker=dict(size=14, color=color, line=dict(width=width, color="black")),
                        text=[f"E{ep['epoch']}"], textposition="top center",
                        showlegend=False,
                    ), row=1, col=3)
            fig.update_layout(height=400)
            for _col in (1, 2, 3):
                fig.update_xaxes(title_text="Эпоха", row=1, col=_col)
            fig.update_yaxes(title_text="Loss (кросс-энтропия)", row=1, col=1)
            fig.update_yaxes(title_text="Loss (кросс-энтропия)", row=1, col=2)
            fig.update_yaxes(title_text="Perplexity", row=1, col=3)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Gold dots = best checkpoint (saved)")

        st.divider()
        st.subheader("Detailed Comparison Table")
        df_show = df_h[["epoch", "train_loss", "val_loss", "train_ppl", "val_ppl", "lr", "time", "saved"]].copy()
        df_show["delta_val"] = df_show["val_loss"].diff().fillna(0).round(4)
        st.dataframe(df_show, use_container_width=True, hide_index=True)

    elif chart_type == "Generation snapshots":
        history = load_history()
        if not history:
            st.info("No training history. Run training first.")
            return

        snapshots = [ep for ep in history if ep.get("samples")]
        if not snapshots:
            st.info("No generation snapshots yet. Snapshots are saved at best checkpoints.")
            return

        st.caption("Sample generations saved at each new best checkpoint.")
        for ep in snapshots:
            with st.expander(
                f"Epoch {ep['epoch']} (val_loss={ep['val_loss']:.4f})",
                expanded=(ep == snapshots[-1]),
            ):
                samples = ep.get("samples", {})
                cols = st.columns(len(samples)) if samples else []
                for col, (cat, names) in zip(cols, samples.items()):
                    with col:
                        st.markdown(f"**{cat}**")
                        for n in names:
                            st.code(n, language=None)


def main():
    st.set_page_config(
        page_title="Name Generator",
        page_icon="🧠",
        layout="wide",
    )
    st.title("Product Name Generator")
    st.caption("LSTM + Beam-Search | Train, Generate, Evaluate, Visualize")

    tabs = st.tabs(["Training", "Generation", "Benchmarks", "Activations", "Charts"])

    with tabs[0]:
        tab_training()
    with tabs[1]:
        tab_generation()
    with tabs[2]:
        tab_benchmarks()
    with tabs[3]:
        tab_activations()
    with tabs[4]:
        tab_charts()


if __name__ == "__main__":
    main()
