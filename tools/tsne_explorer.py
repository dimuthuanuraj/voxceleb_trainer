#!/usr/bin/env python3
"""Interactive t-SNE explorer for the Sinhala/Tamil speaker-embedding sets.

    streamlit run tools/tsne_explorer.py --server.port 8501

Then open http://localhost:8501 (VS Code forwards the port automatically over
an SSH remote; otherwise use `ssh -L 8501:localhost:8501 <host>`).

What you are looking at
-----------------------
Each point is one utterance, positioned by t-SNE over its 192-d ECAPA speaker
embedding, using COSINE distance -- the same geometry the verification scoring
uses. Colour is the speaker by default, so:

  * tight, well-separated colour blobs  -> clean speaker labels
  * one colour split into two far blobs -> that "speaker" may be two people,
                                           or one person across two channels
  * two colours overlapping             -> the model cannot tell them apart;
                                           check them in the mislabel shortlist

t-SNE preserves LOCAL neighbourhoods. Distances *between* well-separated blobs
carry no meaning -- do not read "these two clusters are far apart" as a claim
about how different those speakers are. Use the UMAP explorer for a view with
more global structure, and trust conclusions that hold in both.

Perplexity is roughly "how many neighbours each point tries to keep". Low values
fragment; high values merge. Anything you conclude should survive a sweep.
"""
from __future__ import annotations

import json
import os

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EMB_DIR = os.path.join(REPO, "data", "_qc", "embeddings")
AUDIT_DIR = os.path.join(REPO, "data", "_qc", "label_audit")

st.set_page_config(page_title="t-SNE explorer — SL_SPV", layout="wide")


@st.cache_data(show_spinner=False)
def list_datasets(emb_dir):
    if not os.path.isdir(emb_dir):
        return []
    return sorted(f[:-4] for f in os.listdir(emb_dir) if f.endswith(".npz"))


@st.cache_data(show_spinner=False)
def load_emb(emb_dir, name):
    z = np.load(os.path.join(emb_dir, f"{name}.npz"), allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d["key"] = np.array([f"{l}/{s}" for l, s in zip(d["lang"], d["spk"])])
    return d


@st.cache_data(show_spinner=False)
def load_shortlist(name):
    p = os.path.join(AUDIT_DIR, f"{name}_shortlist.json")
    if not os.path.isfile(p):
        return set()
    with open(p) as fh:
        return {r["path"] for r in json.load(fh)}


@st.cache_data(show_spinner=True)
def run_tsne(emb, perplexity, seed, init, learning_rate, n_iter):
    from sklearn.manifold import TSNE
    n = len(emb)
    perp = float(min(perplexity, max(5.0, (n - 1) / 3)))
    ts = TSNE(n_components=2, metric="cosine", init=init, perplexity=perp,
              learning_rate=learning_rate, max_iter=n_iter, random_state=seed)
    return ts.fit_transform(emb), perp


# --------------------------------------------------------------------------- #
st.title("t-SNE explorer — speaker embeddings")
st.caption("192-d SpeechBrain ECAPA-TDNN embeddings, cosine metric. "
           "One point = one utterance.")

datasets = list_datasets(EMB_DIR)
if not datasets:
    st.error(f"No embeddings found in {EMB_DIR}. "
             "Run: tools/noderun.sh 3 python tools/extract_embeddings.py --all")
    st.stop()

with st.sidebar:
    st.header("Data")
    name = st.selectbox("Dataset", datasets)
    d = load_emb(EMB_DIR, name)
    all_spk = sorted(set(d["key"].tolist()))
    st.caption(f"{len(d['emb']):,} utterances · {len(all_spk):,} speakers")

    n_spk = st.slider("Speakers to show", 2, min(40, len(all_spk)),
                      min(15, len(all_spk)))
    min_utts = st.slider("Min utterances per speaker", 2, 40, 8)
    seed = st.number_input("Seed", value=42, step=1)
    max_points = st.slider("Max points", 200, 4000, 1800, step=100)

    st.header("t-SNE parameters")
    perplexity = st.slider("Perplexity", 5.0, 100.0, 30.0, step=1.0,
                           help="Effective number of neighbours. Sweep it: "
                                "conclusions should survive the sweep.")
    n_iter = st.slider("Iterations", 250, 3000, 1000, step=250)
    lr = st.select_slider("Learning rate", ["auto", 10, 50, 200, 500, 1000],
                          value="auto")
    init = st.selectbox("Init", ["pca", "random"], index=0,
                        help="PCA init is deterministic and preserves more "
                             "global structure than random.")

    st.header("Display")
    colour_by = st.selectbox("Colour by",
                             ["speaker", "session", "gender", "duration", "language"])
    mark_outliers = st.checkbox("Ring the mislabel shortlist", value=True,
                                help="Utterances flagged by the Layer-3 "
                                     "leave-one-out centroid audit.")
    point_size = st.slider("Point size", 3, 16, 7)

rng = np.random.default_rng(int(seed))
key = d["key"]
labs, counts = np.unique(key, return_counts=True)
eligible = labs[counts >= min_utts]
if len(eligible) == 0:
    st.warning(f"No speaker has >= {min_utts} utterances; showing all speakers.")
    eligible = labs
chosen = (rng.choice(eligible, min(n_spk, len(eligible)), replace=False)
          if len(eligible) > n_spk else eligible)
mask = np.isin(key, chosen)
idx = np.where(mask)[0]
if len(idx) > max_points:
    idx = rng.choice(idx, max_points, replace=False)
idx = np.sort(idx)

if len(idx) < 10:
    st.error("Fewer than 10 points selected — loosen the filters.")
    st.stop()

emb = d["emb"][idx]
xy, perp_used = run_tsne(emb, perplexity, int(seed), init, lr, n_iter)

colour = {
    "speaker": d["key"][idx], "session": d["sess"][idx],
    "gender": d["gender"][idx], "language": d["lang"][idx],
    "duration": d["dur"][idx],
}[colour_by]

hover = np.array([
    f"{p}<br>spk={s}<br>sess={ss}<br>{dd:.2f}s"
    for p, s, ss, dd in zip(d["path"][idx], d["key"][idx], d["sess"][idx],
                            d["dur"][idx])])

c1, c2 = st.columns([3, 1])
with c1:
    if colour_by == "duration":
        fig = px.scatter(x=xy[:, 0], y=xy[:, 1], color=colour,
                         color_continuous_scale="Viridis",
                         hover_name=hover, labels={"color": "seconds"})
    else:
        fig = px.scatter(x=xy[:, 0], y=xy[:, 1], color=colour.astype(str),
                         hover_name=hover, labels={"color": colour_by})
    fig.update_traces(marker=dict(size=point_size,
                                  line=dict(width=0.4, color="rgba(0,0,0,.35)")))

    if mark_outliers:
        short = load_shortlist(name)
        if short:
            flag = np.array([p in short for p in d["path"][idx]])
            if flag.any():
                fig.add_trace(go.Scatter(
                    x=xy[flag, 0], y=xy[flag, 1], mode="markers",
                    name="mislabel shortlist",
                    marker=dict(size=point_size + 8, color="rgba(0,0,0,0)",
                                line=dict(width=2, color="crimson")),
                    hoverinfo="skip"))

    fig.update_layout(height=760, xaxis_title="t-SNE 1", yaxis_title="t-SNE 2",
                      legend=dict(itemsizing="constant"),
                      margin=dict(l=10, r=10, t=30, b=10))
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("This view")
    st.metric("Points", f"{len(idx):,}")
    st.metric("Speakers", f"{len(set(d['key'][idx].tolist())):,}")
    st.metric("Perplexity used", f"{perp_used:.0f}")
    st.caption(f"Requested {perplexity:.0f}; capped at (n-1)/3 when the "
               f"selection is small.")

    # a quick quantitative companion to the eyeball test
    from sklearn.metrics import silhouette_score
    try:
        sil = silhouette_score(emb, d["key"][idx], metric="cosine")
        st.metric("Silhouette (cosine, 192-d)", f"{sil:.3f}")
        st.caption("Computed on the ORIGINAL embeddings, not the 2-D "
                   "projection — the projection is for looking at, not for "
                   "measuring.")
    except Exception:
        pass

    st.divider()
    st.markdown(
        "**Reading it**\n\n"
        "- one colour in two far blobs → possible two people under one id, "
        "or one person across two channels\n"
        "- two colours overlapping → the model cannot separate them\n"
        "- red rings → flagged by the leave-one-out centroid audit\n\n"
        "t-SNE preserves *local* neighbourhoods. Gaps between blobs are not "
        "distances. Check anything important in the UMAP explorer too."
    )

with st.expander("Method and mathematics"):
    st.markdown(r"""
t-SNE converts pairwise distances into conditional probabilities

$$p_{j|i}=\frac{\exp(-\lVert x_i-x_j\rVert^2/2\sigma_i^2)}{\sum_{k\neq i}\exp(-\lVert x_i-x_k\rVert^2/2\sigma_i^2)},\qquad
p_{ij}=\frac{p_{j|i}+p_{i|j}}{2N}$$

with each $\sigma_i$ chosen so the perplexity $2^{H(P_i)}$ matches the target.
The low-dimensional layout $Y$ uses a Student-t kernel with one degree of freedom

$$q_{ij}=\frac{(1+\lVert y_i-y_j\rVert^2)^{-1}}{\sum_{k\neq l}(1+\lVert y_k-y_l\rVert^2)^{-1}}$$

and is optimised by minimising

$$KL(P\Vert Q)=\sum_{i\neq j}p_{ij}\log\frac{p_{ij}}{q_{ij}}$$

The heavy tail of the Student-t kernel is what keeps distinct clusters from
collapsing into each other, and is also why *between*-cluster distances in the
picture do not correspond to distances in the embedding space.

Here $\lVert\cdot\rVert$ is cosine distance $d(a,b)=1-\frac{\langle a,b\rangle}{\lVert a\rVert\lVert b\rVert}$,
matching how trials are scored. Embeddings are already L2-normalised, so the
cosine similarity is a plain inner product.
""")
