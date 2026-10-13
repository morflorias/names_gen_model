"""
Name Generator — standalone release UI (generation only).
Hybrid names: 2-3 categories at once + optional start token from the tokenizer.
Includes a Sankey activation-path diagram.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from src.model import NameGenerator
from src.tokenizer import Tokenizer
from src.vocab import Vocab
from src.generate import (
    normalize_category_ids,
    resolve_prefix_ids,
    beam_search_generate,
)

CHECKPOINT_DIR = _HERE / "checkpoints"
CATEGORIES = ["electronics", "clothing", "food", "cosmetics", "toys", "household"]


@st.cache_data
def load_starts_manifest() -> dict:
    """Per-pair start tokens: {pairs: [...], starts: {pair: [[tok, cnt, ex]]}}."""
    path = CHECKPOINT_DIR / "pair_starts.json"
    if not path.exists():
        return {"pairs": [], "starts": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def start_options_for_release(
    tokenizer: Tokenizer,
    manifest: dict,
    categories: list[str],
    top_k: int = 25,
) -> tuple[list[tuple[str, int, str, str]], list[str]]:
    """Start tokens strictly for the chosen categories (no corpus needed)."""
    sel = set(categories)
    pairs = manifest.get("pairs", [])
    covered = [p for p in pairs if set(p.split("+")) <= sel]
    if not covered:
        covered = [p for p in pairs if set(p.split("+")) & sel]
    vocab = tokenizer.vocab.token_to_id
    agg: dict[str, list] = {}
    for p in covered:
        for tok, cnt, ex in manifest.get("starts", {}).get(p, []):
            if tok not in vocab:
                continue
            if tok in agg:
                prev_cnt, prev_ex, prev_pair = agg[tok]
                agg[tok] = [prev_cnt + cnt,
                            prev_ex if prev_cnt >= cnt else ex,
                            prev_pair if prev_cnt >= cnt else p]
            else:
                agg[tok] = [cnt, ex, p]
    ranked = sorted(agg.items(), key=lambda kv: kv[1][0], reverse=True)[:top_k]
    multi = len(covered) > 1
    options = [(tok, c, ex, (pair if multi else ""))
               for tok, (c, ex, pair) in ranked]
    return options, covered


def _short(name: str, limit: int = 30) -> str:
    name = " ".join(str(name).split())
    return name if len(name) <= limit else name[:limit].rstrip() + "…"

# ── cached model/tokenizer loaders ────────────────────────────────────────────

@st.cache_resource
def load_model():
    ckpt_path = CHECKPOINT_DIR / "best.pt"
    tok_path = CHECKPOINT_DIR / "tokenizer.json"
    if not ckpt_path.exists() or not tok_path.exists():
        return None, None, None

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    tokenizer = Tokenizer.load(str(tok_path))

    model = NameGenerator(
        vocab_size=ckpt["vocab_size"],
        n_categories=ckpt["n_categories"],
    )
    model.load_state_dict(ckpt["model"])
    model.eval()

    cat_to_id = ckpt.get("cat_to_id", {c: i for i, c in enumerate(CATEGORIES)})
    return model, tokenizer, cat_to_id

# ── Sankey activation-path builder ────────────────────────────────────────────

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
                token_label = f"'{inv_vocab.get(idx, '')}' "
            elif layer["name"] == "Output":
                token_label = f"'{inv_vocab.get(idx, '')}' "
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


def build_network_diagram(steps, n_layers, inv_vocab, step_index=-1, top_n=6):
    if not steps:
        return go.Figure()
    step = steps[step_index if step_index >= 0 else len(steps) - 1]
    sankey = _build_sankey(step, n_layers, top_n, inv_vocab)
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


def build_animated_network_diagram(steps, n_layers, inv_vocab, top_n=6, frame_ms=800):
    if not steps:
        return go.Figure()
    sankey0 = _build_sankey(steps[0], n_layers, top_n, inv_vocab)
    fig = go.Figure(sankey0)

    frames: list[go.Frame] = []
    for i, step in enumerate(steps):
        sk = _build_sankey(step, n_layers, top_n, inv_vocab)
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
            type="buttons", showactive=False,
            y=1.12, x=0.0, xanchor="left",
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, {"frame": {"duration": frame_ms, "redraw": True},
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
                label=f"{i+1}", method="animate",
            ) for i in range(len(steps))],
            x=0.0, y=-0.05, xanchor="left", len=0.95,
            currentvalue=dict(prefix="Step: ", visible=True, xanchor="right"),
            transition=dict(duration=0),
        )],
    )
    return fig

# ── beam search with hidden state caching ─────────────────────────────────────

# ── beam search diagram 1: probability heatmap ────────────────────────────────

def build_beam_heatmap(step_records, inv_vocab, beam_width, top_display=8):
    if not step_records:
        return go.Figure()

    all_tokens: set[str] = set()
    cell_data: dict[tuple[int, str], float] = {}
    pruned_map: dict[tuple[int, str], bool] = {}

    for rec in step_records:
        for cand in rec.candidates:
            tok = cand["token"]
            if tok in ("<bos>", "<pad>"):
                continue
            key = (rec.step, tok)
            cell_data[key] = cand["score"]
            pruned_map[key] = cand["pruned"]
            all_tokens.add(tok)

    if not all_tokens:
        return go.Figure()

    steps = [r.step for r in step_records]
    tokens = sorted(all_tokens)
    z, txt, hov = [], [], []
    for tok in tokens:
        rz, rt, rh = [], [], []
        for step in steps:
            val = cell_data.get((step, tok), float("nan"))
            rz.append(val)
            pruned = pruned_map.get((step, tok), False)
            status = " (pruned)" if pruned else ""
            rt.append(f"{tok}{status}")
            rh.append(f"Step {step+1}: '{tok}'<br>log-prob: {val:.3f}{status}")
        z.append(rz); txt.append(rt); hov.append(rh)

    fig = go.Figure(go.Heatmap(
        z=z, x=[f"Step {s+1}" for s in steps], y=tokens,
        text=txt, hovertext=hov, hoverinfo="text",
        colorscale="RdYlGn", colorbar=dict(title="log P"),
    ))
    fig.update_layout(
        height=max(300, 40 * len(tokens)),
        margin=dict(l=20, r=20, t=50, b=30),
        title=dict(text="Beam Search — token probabilities per step",
                   x=0.5, xanchor="center", font=dict(size=13)),
        xaxis=dict(title="Step"), yaxis=dict(title="Token"),
    )
    return fig


# ── beam search diagram 2: tree (Sankey) of beam paths ───────────────────────

def build_beam_tree(step_records, inv_vocab, beam_width):
    if not step_records:
        return go.Figure()

    node_labels: list[str] = []
    node_colors: list[str] = []
    source, target, value, link_colors = [], [], [], []

    def _add(tok, step, score, pruned):
        idx = len(node_labels)
        short = tok if len(tok) <= 8 else tok[:7] + "…"
        node_labels.append(f"{short}  {score:.2f}")
        node_colors.append("rgba(220, 80, 80, 0.55)" if pruned else "#4ECDC4")
        return idx

    bos_idx = _add("<bos>", -1, 0.0, False)
    step_maps: list[dict[str, int]] = [{}]

    for ri, rec in enumerate(step_records):
        curr_map: dict[str, int] = {}
        prev_map = step_maps[ri]
        for cand in rec.candidates:
            tok = cand["token"]
            if tok in ("<bos>", "<pad>"):
                continue
            child_idx = _add(tok, rec.step, cand["score"], cand["pruned"])
            curr_map[tok] = child_idx
            if ri == 0:
                parent = bos_idx
            else:
                parent_tok = inv_vocab.get(cand["seq"][-2], "?") if len(cand["seq"]) >= 2 else "<bos>"
                parent = prev_map.get(parent_tok, bos_idx)
            source.append(parent)
            target.append(child_idx)
            value.append(abs(cand["score"]) + 1.0)
            link_colors.append("rgba(220, 80, 80, 0.25)" if cand["pruned"]
                               else "rgba(78, 205, 196, 0.35)")
        step_maps.append(curr_map)

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(pad=15, thickness=14,
                  line=dict(color="rgba(80,80,80,0.4)", width=0.5),
                  label=node_labels, color=node_colors),
        link=dict(source=source, target=target, value=value, color=link_colors),
    ))
    fig.update_layout(
        height=520, margin=dict(l=10, r=10, t=50, b=20),
        title=dict(
            text="Beam Search — candidate paths  "
                 "<span style='color:#4ECDC4'>■ survived</span>  "
                 "<span style='color:#DC5050'>■ pruned</span>",
            x=0.5, xanchor="center", font=dict(size=13),
        ),
    )
    return fig


# ── UI ────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Name Generator", page_icon="🏷️", layout="wide")
    st.title("🏷️ Name Generator")
    st.caption("LSTM · гибрид 2–3 категорий + стартовый токен · word-level генерация")

    model, tokenizer, cat_to_id = load_model()
    if model is None:
        st.error("Model not found. Place `best.pt` and `tokenizer.json` in `checkpoints/`.")
        return

    inv_vocab = {v: k for k, v in tokenizer.vocab.token_to_id.items()}
    manifest = load_starts_manifest()

    # ── sidebar ──
    with st.sidebar:
        st.header("Parameters")
        categories = st.multiselect(
            "Categories (2–3)",
            sorted(cat_to_id.keys()),
            default=sorted(cat_to_id.keys())[:2],
            max_selections=3,
            help="Гибрид принадлежит всем выбранным категориям сразу "
                 "(например, кокосовое масло — еда и косметика).",
        )

        options, pairs_covered = start_options_for_release(tokenizer, manifest, categories)
        if pairs_covered:
            st.caption(f"Токены для пар: {', '.join(pairs_covered)} — вариантов: {len(options)}.")
        labels = ["— без префикса —"] + [
            f"{tok}…  ({cnt})"
            + (f"  [{pair}]" if pair else "")
            + (f"  — {_short(ex)}" if ex else "")
            for tok, cnt, ex, pair in options
        ]
        start_choice = st.selectbox(
            "Start token",
            labels,
            key=f"rel_start__{'__'.join(sorted(categories))}",
            help="Фиксируется только ПЕРВЫЙ токен, продолжение придумывает модель. "
                 "Список — только для выбранных категорий.",
        )
        prefix_tok = None if start_choice == "— без префикса —" else options[labels.index(start_choice) - 1][0]

        method = st.radio("Method", ["Temperature", "Beam search"], horizontal=True)

        temperature: float = 0.9
        top_k: int = 30
        beam_width: int = int(st.session_state.get("beam_width", 5))

        if method == "Temperature":
            temperature = st.slider("Temperature", 0.1, 2.0, 0.9, 0.05)
            top_k = st.slider("Top-k", 0, 100, 30, 5)
            n_names = st.slider("Count", 1, 20, 5)
        else:
            beam_width = st.slider("Beam width", 1, 10, 5, key="rel_beam_width")
            n_names = st.slider("Count", 1, 10, 5)

        st.divider()
        total_params = sum(p.numel() for p in model.parameters())
        st.metric("Parameters", f"{total_params:,}")
        st.metric("Vocab size", tokenizer.vocab.vocab_size)
        st.metric("Base categories", len(cat_to_id))
        st.metric("Pairs in manifest", len(manifest.get("pairs", [])))

    # ── generate ──
    if st.button("Generate", type="primary", use_container_width=True):
        if len(categories) < 2:
            st.error("Выбери минимум 2 категории — генерируем гибрид.")
            st.stop()
        try:
            cat_ids = normalize_category_ids(list(categories), cat_to_id)
            prefix = resolve_prefix_ids(prefix_tok, tokenizer)
        except ValueError as e:
            st.error(str(e))
            st.stop()
        GEN_MAX_TOKENS = 10
        t0 = time.perf_counter()

        names: list[str] = []
        trace_steps = None

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
            trace_ids: list[int] = []
            for s in trace_steps:
                if s["chosen_token_id"] == Vocab.EOS_ID:
                    break
                trace_ids.append(s["chosen_token_id"])
            names.append(tokenizer.decode(trace_ids))

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

        st.session_state["names"] = names
        st.session_state["trace_steps"] = trace_steps
        st.session_state["elapsed"] = elapsed
        st.session_state["category"] = "+".join(sorted(categories))
        st.session_state["prefix"] = prefix_tok or "—"
        st.session_state["method"] = method
        st.session_state["run_id"] = st.session_state.get("run_id", 0) + 1

    # ── results ──
    if "names" in st.session_state:
        names_list = st.session_state["names"]
        n_total = len(names_list)
        lengths = [len(n) for n in names_list] if n_total else [0]
        n_unique = len(set(names_list)) if n_total else 0
        avg_len = sum(lengths) / n_total if n_total else 0.0
        elapsed = st.session_state.get("elapsed", 0.0)

        st.subheader("Results")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total", n_total)
        c2.metric("Unique", n_unique)
        c3.metric("Avg length", f"{avg_len:.1f}")
        c4.metric("Min / Max", f"{min(lengths)} / {max(lengths)}")
        c5.metric("Time", f"{elapsed * 1000:.0f} ms")

        st.caption(
            f"Categories: **{st.session_state.get('category', '')}** · "
            f"Start: **{st.session_state.get('prefix', '—')}** · "
            f"Method: **{st.session_state.get('method', '')}**"
        )

        st.code(" | ".join(names_list) if names_list else "(empty)", language=None)

        run_id = st.session_state.get("run_id", 0)
        for i, name in enumerate(names_list, 1):
            st.text_input(f"#{i}", name, key=f"gen_{run_id}_{i}", disabled=True)

    # ── diagrams by method ──
    gen_method = st.session_state.get("method", "Temperature")
    run_id = st.session_state.get("run_id", 0)

    # ── Temperature: Network Diagram ──
    if gen_method == "Temperature" and st.session_state.get("trace_steps"):
        steps = st.session_state["trace_steps"]
        st.markdown("##### Network Diagram (Temperature)")
        animate = st.checkbox(
            "Animate (step through generation)", value=True, key="net_animate",
        )

        if animate:
            fig = build_animated_network_diagram(steps, model.n_layers, inv_vocab)
            st.plotly_chart(fig, use_container_width=True,
                            key=f"net_anim_{run_id}",
                            config={"displayModeBar": True, "responsive": True})
            st.caption(
                f"Animated trace over {len(steps)} steps. "
                "Use ▶ Play / ⏸ Pause / ⏮ Reset or drag the slider."
            )
        else:
            step_choice = st.selectbox(
                "Step",
                options=list(range(len(steps))),
                index=len(steps) - 1,
                key="net_step",
                format_func=lambda i: f"Step {i + 1}",
            )
            fig = build_network_diagram(steps, model.n_layers, inv_vocab,
                                        step_index=step_choice)
            st.plotly_chart(fig, use_container_width=True,
                            key=f"net_{run_id}_{step_choice}",
                            config={"displayModeBar": True, "responsive": True})
            st.caption(
                "Sankey flow width = |source × dest activation|. "
                "Teal = input, orange = LSTM, gold = output."
            )

    # ── Beam Search: diagrams ──
    if gen_method == "Beam search" and st.session_state.get("beam_steps"):
        st.markdown("##### Beam Search Visualization")
        bs_steps = st.session_state["beam_steps"]
        bw = int(st.session_state.get("beam_width", 5))

        bs_tab1, bs_tab2 = st.tabs(["📊 Probabilities heatmap", "🌳 Beam paths tree"])

        with bs_tab1:
            fig_heat = build_beam_heatmap(bs_steps, inv_vocab, bw)
            st.plotly_chart(fig_heat, use_container_width=True,
                            key=f"beam_heat_{run_id}",
                            config={"displayModeBar": True, "responsive": True})
            st.caption(
                "Each cell = log-probability of a token at a given step. "
                "Darker green = higher probability."
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


if __name__ == "__main__":
    main()
