#!/usr/bin/env python
"""
ESM-2 Viral Metagenomic Atlas (ESM-Atlas style).

Fits UMAP on a subsample of VirusHostDB reference genome embeddings, then
transforms a subsample of MG08 environmental contig embeddings into the same
2D space. Exports a standalone interactive Plotly HTML:
  - reference points colored categorically by taxonomic ORDER (family on hover)
  - MG08 unknown contigs as grey points (predicted family + confidence on hover)

Strategy: memory-map the 15/18 GB .npy files so only sampled rows enter RAM.
"""
import os
import time
from collections import Counter

import numpy as np
import pandas as pd
import umap
import plotly.express as px
import plotly.graph_objects as go

# ----------------------------- config -----------------------------
BASE       = "/project/jespinoza_1537"
N_REF      = int(os.environ.get("N_REF", "150000"))   # reference points to fit UMAP on
N_MG08     = int(os.environ.get("N_MG08", "150000"))  # MG08 points to transform + overlay
OUT_HTML   = os.environ.get("OUT_HTML", os.path.join(BASE, "MG08_umap_atlas.html"))
N_TOP_ORD  = 30           # distinct colors for the most common orders; rest -> "other"
SEED       = 42
N_JOBS     = int(os.environ.get("SLURM_CPUS_PER_TASK", "16"))

# MG08 confidence colorscale. Monotonically increasing luminance so that on the
# black canvas low-confidence points stay dim and recede into the background
# haze, while high-confidence points glow warm/bright and pop out. Reads as
# "hotter = more confident" — the black-background-correct form of the intuitive
# "deeper color = more confident". Distinct from the reference categorical hues.
CONF_SCALE = [
    [0.00, "#3a3a4d"],   # low confidence  -> dim slate, recedes on black
    [0.30, "#5b4b8a"],   #                 -> violet
    [0.55, "#9b3f9b"],   #                 -> magenta
    [0.75, "#e0563f"],   #                 -> orange-red
    [1.00, "#ffe14d"],   # high confidence -> bright warm yellow, pops
]

# MG08 confidence tiers. The single MG08 layer is split into these bins, each
# emitted as its own trace under a shared legend group, so it toggles like any
# reference-order entry: double-click a tier to isolate it (e.g. view ONLY the
# high-confidence calls). Thresholds are [lo, hi) on confidence; defaults sit at
# the observed median (0.33) and a selective high cut (0.66), env-overridable.
CONF_T_LO = float(os.environ.get("CONF_T_LO", "0.33"))  # low | medium boundary
CONF_T_HI = float(os.environ.get("CONF_T_HI", "0.66"))  # medium | high boundary

rng = np.random.default_rng(SEED)
_t0 = time.time()
def log(msg):
    print(f"[{time.time()-_t0:8.1f}s] {msg}", flush=True)


def load_ids(path):
    with open(path) as fh:
        return np.array([ln.strip() for ln in fh])


def sample_rows(npy_path, n, seed_rng):
    """mmap an .npy, pick n sorted random row indices, materialize those rows."""
    arr = np.load(npy_path, mmap_mode="r")
    n = min(n, arr.shape[0])
    idx = np.sort(seed_rng.choice(arr.shape[0], size=n, replace=False))
    X = np.asarray(arr[idx], dtype=np.float32)   # fancy-index copies only sampled rows
    return idx, X, arr.shape[0]


# ------------------------- reference load -------------------------
log("Loading reference IDs")
ref_ids = load_ids(f"{BASE}/reference_genome_ids.txt")

log("Sampling reference embeddings")
ref_idx, ref_X, n_ref_total = sample_rows(f"{BASE}/reference_genome_embeddings.npy", N_REF, rng)
assert len(ref_ids) == n_ref_total, (len(ref_ids), n_ref_total)
ref_sample_ids = ref_ids[ref_idx]
log(f"  reference: {ref_X.shape} (of {n_ref_total:,})")

log("Loading reference taxonomy")
tax = pd.read_csv(f"{BASE}/reference_genome_taxonomy.csv",
                  usecols=["genome_id", "order", "family"], dtype=str)
tax = tax.drop_duplicates("genome_id").set_index("genome_id")
ref_order  = tax["order"].reindex(ref_sample_ids).fillna("").to_numpy()
ref_family = tax["family"].reindex(ref_sample_ids).fillna("").to_numpy()
ref_order  = np.array([o if o else "unclassified" for o in ref_order])
ref_family = np.array([f if f else "unclassified" for f in ref_family])

# ---------------------------- fit UMAP ----------------------------
log(f"Fitting UMAP (2D, cosine, n_jobs={N_JOBS}) on {ref_X.shape[0]:,} reference points")
reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                    metric="cosine", low_memory=True, n_jobs=N_JOBS, verbose=True)
ref_2d = reducer.fit_transform(ref_X)
log("UMAP fit complete")

# ------------------------- MG08 transform -------------------------
log("Loading MG08 IDs")
mg_ids = load_ids(f"{BASE}/MG08_genome_ids.txt")

log("Sampling MG08 embeddings")
mg_idx, mg_X, n_mg_total = sample_rows(f"{BASE}/MG08_genome_embeddings.npy", N_MG08, rng)
assert len(mg_ids) == n_mg_total, (len(mg_ids), n_mg_total)
mg_sample_ids = mg_ids[mg_idx]
log(f"  MG08: {mg_X.shape} (of {n_mg_total:,})")

log("Transforming MG08 into reference UMAP space")
mg_2d = reducer.transform(mg_X)

log("Loading MG08 predictions")
pred = pd.read_csv(f"{BASE}/MG08_predictions.csv",
                   dtype={"genome_id": str, "predicted_family": str, "confidence": float})
pred = pred.drop_duplicates("genome_id").set_index("genome_id")
mg_family = pred["predicted_family"].reindex(mg_sample_ids).fillna("NA").to_numpy()
mg_conf   = pred["confidence"].reindex(mg_sample_ids).fillna(0.0).to_numpy()

# --------------------------- coloring -----------------------------
# Catch-all bucket: mostly VirusHostDB genomes with NO ICTV `order` assigned at
# all (~50% of the reference — order is a high rank ICTV often leaves blank), plus
# a small tail of genuinely rare orders (rank 31+). It is NOT a taxonomic group,
# so it is labeled honestly and dimmed to near-background so it recedes into the
# black haze instead of dominating the atlas as a bright grey mass.
OTHER_LABEL = "no ICTV order"
counts = Counter(o for o in ref_order if o != "unclassified")
top_orders = [o for o, _ in counts.most_common(N_TOP_ORD)]
top_set = set(top_orders)
ref_order_plot = np.array([o if o in top_set else OTHER_LABEL for o in ref_order])

palette = px.colors.qualitative.Dark24 + px.colors.qualitative.Light24
color_map = {o: palette[i % len(palette)] for i, o in enumerate(top_orders)}
color_map[OTHER_LABEL] = "#1c1c1c"   # very dim near-black grey -> faint haze on black bg
log(f"{len(top_orders)} distinct orders colored; {len(counts)} orders total")

# ----------------------------- plot -------------------------------
log("Building figure")
fig = go.Figure()

# MG08 underlay (drawn first so reference sits on top), colored by prediction
# confidence: dim = uncertain "dark matter" haze, bright = confident calls.
# Split into confidence tiers, each its OWN standalone legend entry (like the
# reference-order entries) — NOT a shared legendgroup. A shared group looks like
# a drop-down but makes Plotly toggle/isolate the whole group at once; standalone
# entries let you double-click a single tier to view only those points (e.g. the
# high-confidence calls on their own). Names are prefixed "MG08" and legendrank'd
# so the three sit together at the top of the legend, reading most-confident
# first. Points keep the continuous CONF_SCALE coloring with fixed cmin/cmax=0..1,
# so a given confidence maps to the same color in every tier (gradient preserved).
mg_cd = np.array(list(zip(mg_sample_ids.tolist(), mg_family.tolist(), mg_conf.tolist())),
                 dtype=object)

# draw order low -> high so confident points sit on top of the haze; legendrank
# reorders the legend to read most-confident first.
mg_tiers = [
    dict(key="low",    lo=0.0,       hi=CONF_T_LO, rank=3,
         label="low conf (<%.2f)" % CONF_T_LO),
    dict(key="medium", lo=CONF_T_LO, hi=CONF_T_HI, rank=2,
         label="medium conf (%.2f-%.2f)" % (CONF_T_LO, CONF_T_HI)),
    dict(key="high",   lo=CONF_T_HI, hi=1.01,      rank=1,
         label="high conf (>=%.2f)" % CONF_T_HI),
]
for t in mg_tiers:
    t["mask"] = (mg_conf >= t["lo"]) & (mg_conf < t["hi"])

for t in mg_tiers:
    m = t["mask"]
    if not m.any():
        continue
    marker = dict(size=2.5, opacity=0.60,
                  color=mg_conf[m], colorscale=CONF_SCALE, cmin=0.0, cmax=1.0)
    if t["key"] == "high":
        # one shared colorbar, on the high tier so it survives isolating it
        marker["colorbar"] = dict(
            title=dict(text="MG08 prediction confidence", side="top",
                       font=dict(color="#cccccc", size=11)),
            orientation="h", thickness=12, len=0.26,
            x=0.01, xanchor="left", y=0.02, yanchor="bottom",
            tickvals=[0.0, 0.25, 0.5, 0.75, 1.0],
            tickfont=dict(color="#cccccc", size=9),
            outlinecolor="#333333", outlinewidth=1,
            bgcolor="rgba(20,20,20,0.45)",
        )
    else:
        marker["showscale"] = False
    fig.add_trace(go.Scattergl(
        x=mg_2d[m, 0], y=mg_2d[m, 1], mode="markers",
        name=f"MG08 {t['label']} ({int(m.sum()):,})",
        legendrank=t["rank"],
        marker=marker,
        customdata=mg_cd[m],
        hovertemplate=(f"<b>MG08 contig ({t['key']} conf)</b>"
                       "<br>%{customdata[0]}"
                       "<br>pred family: %{customdata[1]}"
                       "<br>confidence: %{customdata[2]:.2f}<extra></extra>"),
    ))

# reference, one trace per order for a toggleable legend
for o in top_orders + [OTHER_LABEL]:
    mask = ref_order_plot == o
    if not mask.any():
        continue
    cd = np.stack([ref_sample_ids[mask], ref_family[mask], ref_order[mask]], axis=1)
    fig.add_trace(go.Scattergl(
        x=ref_2d[mask, 0], y=ref_2d[mask, 1], mode="markers",
        name=f"{o} ({int(mask.sum()):,})",
        # the "no ICTV order" catch-all is hidden by default (one legend tap to show)
        # so the atlas opens on the classified signal only; still one tap from full view.
        visible=("legendonly" if o == OTHER_LABEL else True),
        marker=dict(size=2.5, color=color_map[o]),
        customdata=cd,
        hovertemplate=("<b>%{customdata[2]}</b><br>%{customdata[0]}"
                       "<br>family: %{customdata[1]}<extra></extra>"),
    ))

fig.update_layout(
    title=dict(text="ESM-2 Viral Metagenomic Atlas — MG08 vs VirusHostDB reference",
               font=dict(color="#dddddd", size=16), x=0.01),
    paper_bgcolor="black", plot_bgcolor="black",
    font=dict(color="#cccccc"),
    legend=dict(font=dict(size=9), itemsizing="constant", bgcolor="rgba(0,0,0,0)",
                bordercolor="#333333", borderwidth=1),
    xaxis=dict(visible=False), yaxis=dict(visible=False),
    autosize=True,                                   # fill the browser tab, resize with the window
    margin=dict(l=0, r=230, t=40, b=0),
)
fig.update_yaxes(scaleanchor="x", scaleratio=1)  # equal aspect -> circular cloud

log(f"Writing {OUT_HTML}")
# default_*="100vw/100vh" + responsive -> the plot fills a generic Chrome tab (no zoom, no white edges)
fig.write_html(OUT_HTML, include_plotlyjs=True, full_html=True,
               default_width="100vw", default_height="100vh",
               config={"responsive": True})
# strip the default 8px <body> margin (would show as a white border on the black atlas)
with open(OUT_HTML, "r") as fh:
    _html = fh.read()
_html = _html.replace("<body>", '<body style="margin:0;background:#000;overflow:hidden">', 1)
with open(OUT_HTML, "w") as fh:
    fh.write(_html)
log("Done")
