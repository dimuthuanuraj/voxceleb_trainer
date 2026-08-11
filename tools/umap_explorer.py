#!/usr/bin/env python3
"""Interactive UMAP explorer for the Sinhala/Tamil speaker-embedding sets.

    streamlit run tools/umap_explorer.py --server.port 8502

Then open http://localhost:8502 (VS Code forwards the port automatically over an
SSH remote; otherwise `ssh -L 8502:localhost:8502 <host>`).

Companion to tools/tsne_explorer.py. Run both: they fail differently, and a
structure that appears in only one of them is usually an artefact of the
projection rather than a property of the data.

Where UMAP differs from t-SNE
-----------------------------
  * it keeps more GLOBAL structure, so relative positions of distant clusters
    carry some meaning (t-SNE's do not)
  * it is much faster, so bigger subsamples are practical
  * `n_neighbors` trades local detail (low) against global shape (high)
  * `min_dist` controls how tightly points may pack; it is purely cosmetic for
    cluster *membership* but changes how separated things look, so do not read
    tight packing as strong evidence

Neither view is evidence on its own. The quantitative answers live in
Layer 3 (`data/_qc/label_audit/`); these plots are for finding *what* to
go and listen to.
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

st.set_page_config(page_title="UMAP explorer — SL_SPV", layout="wide")


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
def run_umap(emb, n_neighbors, min_dist, seed, metric, n_epochs, spread):
    import umap
    n = len(emb)
    nn = int(min(n_neighbors, max(2, n - 1)))
    red = umap.UMAP(n_components=2, metric=metric, n_neighbors=nn,
                    min_dist=float(min_dist), spread=float(spread),
                    n_epochs=int(n_epochs), random_state=int(seed))
    return red.fit_transform(emb), nn


# --------------------------------------------------------------------------- #
st.title("UMAP explorer — speaker embeddings")
st.caption("192-d SpeechBrain ECAPA-TDNN embeddings, cosine metric. "
           "One point = one utterance.")

try:
    import umap  # noqa: F401
except Exception as exc:
    st.error(f"umap-learn is not importable: {exc}\n\npip install umap-learn")
    st.stop()

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

    n_spk = st.slider("Speakers to show", 2, min(60, len(all_spk)),
                      min(15, len(all_spk)))
    min_utts = st.slider("Min utterances per speaker", 2, 40, 8)
    seed = st.number_input("Seed", value=42, step=1)
    max_points = st.slider("Max points", 200, 8000, 2500, step=100,
                           help="UMAP is fast; larger subsamples are practical "
                                "here than with t-SNE.")

    st.header("UMAP parameters")
    n_neighbors = st.slider("n_neighbors", 2, 200, 15,
                            help="Low = local detail and more fragments. "
                                 "High = global shape, blurred boundaries.")
    min_dist = st.slider("min_dist", 0.0, 0.99, 0.10, step=0.01,
                         help="How tightly points may pack. Cosmetic for "
                              "cluster membership.")
    spread = st.slider("spread", 0.5, 3.0, 1.0, step=0.1)
    n_epochs = st.select_slider("Epochs", [100, 200, 500, 1000], value=200)
    metric = st.selectbox("Metric", ["cosine", "euclidean", "correlation"],
                          index=0)

    st.header("Display")
    colour_by = st.selectbox("Colour by",
                             ["speaker", "session", "gender", "duration", "language"])
    mark_outliers = st.checkbox("Ring the mislabel shortlist", value=True)
    point_size = st.slider("Point size", 3, 16, 6)

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
xy, nn_used = run_umap(emb, n_neighbors, min_dist, int(seed), metric,
                       n_epochs, spread)

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

    fig.update_layout(height=760, xaxis_title="UMAP 1", yaxis_title="UMAP 2",
                      legend=dict(itemsizing="constant"),
                      margin=dict(l=10, r=10, t=30, b=10))
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("This view")
    st.metric("Points", f"{len(idx):,}")
    st.metric("Speakers", f"{len(set(d['key'][idx].tolist())):,}")
    st.metric("n_neighbors used", f"{nn_used}")

    from sklearn.metrics import silhouette_score
    try:
        sil = silhouette_score(emb, d["key"][idx], metric="cosine")
        st.metric("Silhouette (cosine, 192-d)", f"{sil:.3f}")
        st.caption("Computed on the ORIGINAL embeddings, not the projection.")
    except Exception:
        pass

    st.divider()
    st.markdown(
        "**Cross-check with t-SNE**\n\n"
        "Structure that shows up here *and* in the t-SNE view is probably real. "
        "Structure in only one is probably an artefact.\n\n"
        "Unlike t-SNE, relative positions of distant clusters here carry some "
        "meaning — but `min_dist` still changes how separated things *look* "
        "without changing membership."
    )

with st.expander("Method and mathematics"):
    st.markdown(r"""
UMAP models the data as a fuzzy simplicial set. For each point it finds the $k$
nearest neighbours, sets $\rho_i$ to the distance to the nearest neighbour (the
local connectivity correction, which guarantees every point is connected to
something), and picks $\sigma_i$ so that

$$\sum_{j=1}^{k}\exp\!\left(-\frac{\max(0,\; d(x_i,x_j)-\rho_i)}{\sigma_i}\right)=\log_2 k$$

giving directed memberships

$$\mu_{i\to j}=\exp\!\left(-\frac{\max(0,\; d(x_i,x_j)-\rho_i)}{\sigma_i}\right)$$

symmetrised as $\mu_{ij}=\mu_{i\to j}+\mu_{j\to i}-\mu_{i\to j}\mu_{j\to i}$.

In the low-dimensional space the membership is

$$\nu_{ij}=\left(1+a\lVert y_i-y_j\rVert^{2b}\right)^{-1}$$

with $a,b$ fitted from `min_dist` and `spread`. The layout minimises the fuzzy
cross-entropy

$$C=\sum_{i\neq j}\Big[\mu_{ij}\log\frac{\mu_{ij}}{\nu_{ij}}+(1-\mu_{ij})\log\frac{1-\mu_{ij}}{1-\nu_{ij}}\Big]$$

optimised by stochastic gradient descent with negative sampling. The second term
is the repulsive one that t-SNE lacks in this form, and it is why UMAP retains
more global structure.

Distance $d$ is cosine, matching how trials are scored; the embeddings are
L2-normalised so cosine similarity is a plain inner product.
""")
