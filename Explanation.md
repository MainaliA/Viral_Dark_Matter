# Explanation — Viral Dark Matter, end to end

A teaching document for the ESM-2 viral metagenomic classification + UMAP atlas
project. Written so you can **explain the technical depth to anyone** — a
labmate, Dr. Espinoza, or a poster-session judge — without hand-waving. It walks
the whole pipeline in order, then drills into the two things people always ask
about: **where the logistic-regression classifier comes from**, and **where the
confidence score comes from**.

Read it top to bottom once; after that it works as a reference.

---

## 0. The one-paragraph version (memorize this)

We take unknown environmental viral DNA fragments (contigs), find the genes on
them, and turn each gene's protein into a 1280-number "meaning vector" using
**ESM-2**, a protein language model. We average a genome's gene-vectors into one
vector per genome (**mean pooling**). We train a **logistic-regression**
classifier on *reference* viruses whose families we already know, then ask it to
guess the family of each unknown genome and how sure it is (**confidence**).
Finally we squash all these 1280-dimensional vectors down to a 2-D map with
**UMAP** so we can *see* where the unknowns fall relative to known virus
families. Most unknowns land in low-confidence regions — that low confidence is
the actual scientific finding, the "viral dark matter."

---

## 1. The biological problem

Environmental samples (soil, water, gut, etc.) are full of viruses we have never
catalogued. When you sequence such a sample you get millions of short DNA
fragments. Standard tools like **Kraken2** work by matching your DNA against a
database of known sequences (essentially exact/k-mer matching). On environmental
**viral** metagenomes that database-matching approach classifies only ~**1–3%**
of sequences — because the viruses simply are not in any database. The other
97–99% is **viral dark matter**: real biological sequence that no reference
recognizes.

**Our bet:** instead of matching *DNA letters*, compare the *proteins* in a
learned "meaning space." Two viruses can have very different DNA but proteins
that do the same job (homologs) and therefore sit close together in a protein
language model's embedding space. That should let us place unknowns near their
relatives even when exact matching fails.

**The honest caveat (important):** we have **not yet benchmarked** against
Kraken2 on the same data. So we do **not** claim to beat it. We claim we can
*embed and organize* the dark matter and *quantify* how novel it looks. Beating
Kraken2 is a future, measured comparison.

---

## 2. The pipeline, stage by stage

```
contigs  →  Prodigal ORFs  →  ESM-2 650M embeddings  →  mean pooling
        →  logistic-regression classifier  →  UMAP atlas
```

### 2.1 Contigs (the raw material)
A **contig** is a *contig*uous stretch of DNA that the assembler stitched
together from many short sequencing reads. One contig is our unit of "a genome
or genome fragment." Our working environmental sample is **MG08**; a second
sample **MG09** is planned once MG08 is validated.

### 2.2 Prodigal → ORFs (find the genes)
DNA is not all genes. **Prodigal** is a gene-prediction program that scans each
contig and finds **ORFs** (Open Reading Frames) — stretches that look like
protein-coding genes (start codon, no internal stop, plausible codon
statistics). Each ORF is translated into a **protein sequence** (a string of
amino acids). This is why the project talks about *proteins*, not raw DNA, from
here on.

- MG08 yields **3,955,083 proteins (ORFs)**. That is the number ESM-2 embedded.

### 2.3 ESM-2 650M → per-protein embeddings (the core idea)
**ESM-2** is a **protein language model** from Meta AI. Same idea as a text
language model (BERT/GPT), but the "language" is protein: the "words" are the 20
amino acids. It was trained on hundreds of millions of natural protein sequences
with a **masked-token objective** — hide some amino acids, predict them from
context. To do that well it has to internalize protein "grammar": which residues
co-vary, structural motifs, functional families. The "650M" is the model size
(~650 million parameters). Output vector width is **1280**.

For each protein we run it through ESM-2 and take the model's internal
representation, producing one **1280-dimensional embedding vector** per protein.
Geometrically: proteins with similar structure/function end up **near each other**
in this 1280-D space, even if their amino-acid strings differ. That "similar
function ⇒ nearby vector" property is the entire reason the downstream steps work.

- Reference protein embeddings live in `GENOMESDB/virus_embeddings_batches/`
  (61,755 parquet files); MG08's in `MG08_embeddings/` (619 parquet files).

### 2.4 Mean pooling → one vector per genome
A genome has many proteins, so many 1280-D vectors. The classifier and the map
need **one vector per genome**, not per protein. **Mean pooling** = average all
of a genome's protein vectors element-wise into a single 1280-D vector (the
centroid of its proteins). Simple, cheap, and it captures a genome's overall
"protein composition."

- MG08: 3,955,083 proteins collapse to **3,701,871 genome/contig vectors**.
  (Fewer genomes than proteins because a genome has many proteins; the small
  gap from "every contig" is contigs that lost all ORFs / failed pooling.)
- Reference: **3,052,046 genome vectors**.

> **Keep these three counts straight — they get conflated constantly:**
> - **3,955,083** = MG08 *proteins* (what ESM-2 embedded)
> - **3,701,871** = MG08 *genome vectors* after pooling (what the classifier scored)
> - **3,052,046** = *reference* genome vectors

Files (Laguna, `/project/jespinoza_1537/`):
- `MG08_genome_embeddings.npy` — 3,701,871 × 1280 (18 GB), row-aligned to `MG08_genome_ids.txt`
- `reference_genome_embeddings.npy` — 3,052,046 × 1280 (15 GB), row-aligned to `reference_genome_ids.txt`

### 2.5 The reference and its labels
To train a classifier you need examples with **known** answers. That is
**VirusHostDB** — a curated database of viruses whose taxonomy is known, tagged
with **ICTV** (International Committee on Taxonomy of Viruses) ranks. We use 15
ranks (realm → … → species) in `reference_genome_taxonomy.csv`; the classifier
targets the **family** rank. There are **220 families** with enough examples to
learn from.

### 2.6 Classifier → predicted family + confidence
Covered in depth in §3–4. In one line: a logistic-regression model trained on
reference genome vectors predicts, for every MG08 genome vector, the most likely
family and a confidence (`MG08_predictions.csv`).

### 2.7 UMAP → the 2-D atlas
Covered in §5. We can't look at a 1280-D cloud, so **UMAP** projects it to 2-D
preserving neighborhood structure, and we render an interactive atlas.

---

## 3. Where the logistic regression comes from

### 3.1 What "logistic regression" is
Logistic regression is a **linear classifier**. For an input vector **x** (here
a 1280-D genome embedding) and a class *k* (a virus family), it computes a score
that is just a weighted sum of the input features plus a bias:

```
score_k(x) = w_k · x + b_k        (a dot product: w_k is a 1280-length weight vector)
```

Each of the **220 families** gets its **own** weight vector `w_k` and bias `b_k`.
So the model is a `220 × 1280` weight matrix plus a length-220 bias vector. The
family with the highest score is the prediction. "Training" means finding the
weights that best separate the families on the labeled reference data.

Because it is linear, logistic regression draws **flat decision boundaries
(hyperplanes)** between families in the 1280-D embedding space. That is fine here
because ESM-2 has already done the hard non-linear work — it arranged proteins so
that families are *roughly linearly separable* in embedding space. The
embedding is the intelligence; the classifier is a cheap, interpretable readout
on top. This is a standard and deliberate design: **big pretrained encoder +
simple linear head** (a "linear probe").

### 3.2 How it is trained here (the exact recipe)
- **Training data:** a **200K** random subsample of the 3,052,046 reference
  genome vectors — split **160K train / 40K test** (80/20).
  - We subsample because 3M × 1280 is large and 200K is plenty to fit a linear
    model; the 40K held-out test set is what the accuracy number is measured on.
- **Model:** scikit-learn `LogisticRegression`.
- **Solver:** `lbfgs` — a quasi-Newton optimizer (limited-memory BFGS). It is a
  good default for multiclass logistic regression on dense features; it uses
  gradient + an approximate curvature (Hessian) estimate to converge fast.
- **`max_iter=2000`** — cap on optimizer iterations (how long it is allowed to
  keep improving the weights before stopping).
- **220 classes**, multinomial (softmax) formulation — all families scored
  jointly, not one-vs-rest.
- **16 CPUs** for the fit.
- **Result: ~75% accuracy on the 40K test set.** Meaning: on *reference* viruses
  the model had never seen during training, it names the exact correct family
  75% of the time out of 220 possibilities (random guessing ≈ 1/220 ≈ 0.45%).
  75% is strong for 220-way classification and confirms the embedding space is
  genuinely family-structured.

Artifacts on Laguna:
- `classifier.pkl` — the trained `LogisticRegression` object (weights + biases).
- `label_encoder.pkl` — a scikit-learn `LabelEncoder` mapping integer class
  indices ↔ family name strings (the model works in integers; this decodes them).
- `classification_report.csv` — per-family precision / recall / F1 on the test set.

### 3.3 Applying it to MG08 (inference)
Every one of the **3,701,871** MG08 genome vectors is fed through the trained
model. For each we record the **argmax family** and its **confidence** (§4) into
`MG08_predictions.csv` (`genome_id, predicted_family, confidence`).

### 3.4 Why "predicted family" for MG08 must be read carefully
The classifier is **forced to choose** one of the 220 *known* families. It has no
"none of the above / novel" option. So for a genuinely novel environmental virus
it still emits its *nearest* known family — a best guess, not a certainty. That
is exactly why the confidence score matters: it is how we tell "confident, likely
correct" from "forced guess about something the model has never seen."

---

## 4. Where the confidence score comes from

### 4.1 From scores to probabilities: the softmax
The raw `score_k(x) = w_k · x + b_k` values are unbounded real numbers, not
probabilities. **Softmax** converts the 220 scores into 220 numbers in [0, 1]
that sum to 1:

```
P(family_k | x) = exp(score_k(x)) / Σ_j exp(score_j(x))       (j runs over all 220 families)
```

Exponentiate every score (makes them positive, amplifies the largest), then
normalize so they sum to 1. This is what scikit-learn's `predict_proba` returns.

### 4.2 The confidence number itself
```
confidence(x) = max_k P(family_k | x)      # the probability of the winning family
```

It is the **maximum softmax probability** — how much probability mass the model
puts on its single best guess. `predicted_family` is the `argmax`; `confidence`
is the `max` of the very same probability vector. They come from **one** softmax
pass; the family is *which* is largest, the confidence is *how* large.

- Range: **1/220 ≈ 0.0045** (model spreads belief evenly = maximally unsure) up
  to **1.0** (all belief on one family = maximally sure).
- Observed in MG08: **0.049 to 1.0**.

### 4.3 What the MG08 confidences actually look like — and why
Across all 3.7M MG08 predictions:
- **mean ≈ 0.38, median ≈ 0.33** — low.
- Only ~**23%** clear 0.5; ~**9%** clear 0.7; ~**1.9%** clear 0.9.
- Top predicted families: *jeanschmidtviridae* (~403K), *flaviviridae* (~344K),
  *peduoviridae* (~255K).

**This low confidence is the finding, not a bug.** On the 40K reference test set
the same model is ~75% accurate and typically confident — because those viruses
resemble its training families. MG08 is mostly novel, so the model is *correctly
uncertain*: it is telling us "this doesn't clearly match any known family."
Low, diffuse confidence across an environmental sample is the quantitative
signature of **viral dark matter**.

### 4.4 What confidence is NOT
- It is **not** calibrated ground-truth probability of correctness. A raw softmax
  max tends to be **overconfident**; 0.8 does not guarantee 80% correctness
  without calibration (e.g. temperature scaling), which we have not done.
- It is **not** a novelty detector by construction — but empirically low
  confidence tracks novelty well here, so we use it as a practical proxy.
- High confidence on MG08 means "confidently resembles a known family," **not**
  "definitely that family" — still a hypothesis to check.

---

## 5. The UMAP atlas (seeing 1280-D)

### 5.1 What UMAP does
**UMAP** (Uniform Manifold Approximation and Projection) is a **dimensionality
reduction** method: it takes high-dimensional vectors and produces low-D (here
2-D) coordinates that keep **local neighborhood structure** — points that were
neighbors in 1280-D stay neighbors in 2-D, so clusters and gradients survive. It
does this by building a fuzzy nearest-neighbor graph in high-D and then laying
that graph out in 2-D so the two graphs match as closely as possible. It is the
same family of tool as t-SNE, but generally faster and better at preserving
larger-scale structure.

Caveat to state out loud: **UMAP axes are meaningless** and absolute distances
are only roughly meaningful. Read it as "what clusters near what," not as a
quantitative coordinate system.

### 5.2 Our specific approach (and the memory trick)
The reference (15 GB) and MG08 (18 GB) embedding matrices are far too big to load
and co-embed in RAM. So `umap_atlas.py` does:

1. **Memory-map** both `.npy` files (`mmap_mode="r"`) — the array stays on disk;
   only the rows we actually touch are read into RAM.
2. **Subsample the reference** (default **150K** rows, seeded `SEED=42`) and
   **fit** UMAP on just those, with `metric="cosine"` (cosine because embedding
   *direction* carries the meaning; magnitude matters less).
3. **Subsample MG08** (default **150K**) and **`transform`** it into the
   *already-fitted* reference space — crucially we do **not** refit. Both sets
   now share one coordinate system, so overlaying them is meaningful.
4. Merge in reference **taxonomy** and MG08 **predictions**, render one
   standalone interactive **Plotly** HTML (WebGL `Scattergl` for smooth
   pan/zoom at 300K points).

> Note the 150K + 150K atlas subsample is drawn **fresh and independently** of
> the 200K the classifier used. They are different random draws for different
> purposes (train/evaluate vs. visualize).

UMAP knobs currently: `n_neighbors=15`, `min_dist=0.1`, `metric="cosine"`,
2 components. `n_neighbors` trades local vs. global structure (higher = more
global); `min_dist` controls how tightly points pack. Tuning these (e.g.
`n_neighbors=50`) is on the roadmap.

### 5.3 What you see in the atlas
- **Reference viruses** are colored by taxonomic **order** (the top 30 of ~72
  orders get distinct colors; the rest collapse to grey `"other"`). Hover shows
  the family. These colored clouds are the "known-world" map.
- **MG08 unknown contigs** are the overlay. **Each MG08 point is now colored by
  its prediction confidence** via a sequential colorscale: **dim = uncertain,
  bright/warm = confident** (see §6). Hover shows the contig ID, predicted
  family, and confidence.
- Reading it: MG08 points that fall **inside** a colored reference cluster *and*
  glow bright are the model's confident, plausible calls. The vast dim haze
  sitting in the gaps between known families is the dark matter.

---

## 6. The confidence coloring (what changed, and the design choice)

Originally the MG08 layer was flat grey. It now encodes **confidence as color**.

**The design tension worth understanding.** The intuitive encoding is "darker =
more confident." But the atlas background is **black**. On black, dark points
vanish and pale/bright points dominate the eye — so literal "darker = more
confident" would *hide* the confident points (the interesting ones) and make the
low-confidence noise the loudest thing on screen. Backwards.

**The fix** is the same idea inverted for a dark canvas: **more confident = more
intense / hotter / brighter**. The colorscale (`CONF_SCALE` in `umap_atlas.py`)
climbs monotonically in luminance:

```
low confidence   dim slate  → violet → magenta → orange-red → bright warm yellow   high confidence
0.0              (#3a3a4d)                                    (#ffe14d)             1.0
```

Consequences, all intentional:
- Low-confidence points stay dim and **recede into the background haze** — this
  even preserves the grey-cloud look the professor liked.
- The rare high-confidence points **glow and pop out** — your eye goes straight
  to the signal.
- A **horizontal colorbar** (bottom-left) makes the mapping explicit, and
  `cmin=0, cmax=1` keeps the scale anchored to the true probability range so the
  color is honest, not stretched.

Because MG08 is a single Plotly trace, you can click it in the legend to
toggle it, or hide reference orders, to study one layer at a time.

---

## 7. Anticipated questions (rapid-fire answers)

**Why not just BLAST / Kraken2?** Those match sequence *letters*; on novel
viruses there is nothing to match. Embeddings compare *function*, which
generalizes past exact matches.

**Why mean pooling and not something fancier?** It is a strong, cheap baseline
that needs no training and captures overall protein content. Attention pooling or
per-protein classification are possible refinements, but pooling already gives a
family-structured space (75% test accuracy proves it).

**Why logistic regression and not a deep net / XGBoost?** The heavy lifting is
already in ESM-2. A linear probe on top is fast, interpretable, hard to overfit,
and standard practice for evaluating pretrained embeddings. **KNN with FAISS** is
on the roadmap as an alternative that may scale better at this size.

**Is 75% good?** For **220-way** classification where chance is 0.45%, yes —
very. It validates the embedding, not the environmental predictions themselves.

**So did we beat Kraken2's 1–3%?** **Not claimed.** That is an apples-to-apples
benchmark we still have to run. Say "we can embed and organize the dark matter
and quantify its novelty," not "we beat Kraken2."

**Why is MG08 confidence so low — did something break?** No. It is the expected
viral-dark-matter signal: novel viruses that do not resemble the 220 known
training families, so the forced-choice classifier is correctly uncertain.

**Can I trust a confidence of 0.9 on an MG08 contig?** Treat it as a strong
*hypothesis* worth following up, not proof. The scores are uncalibrated and can
be overconfident; and even a correct family is a coarse label.

---

## 8. Glossary (say these correctly)

- **Contig** — an assembled continuous DNA sequence; our per-genome unit.
- **ORF** — Open Reading Frame; a predicted protein-coding gene (from Prodigal).
- **ESM-2** — Meta's protein language model; turns a protein into a 1280-D vector.
- **Embedding** — the vector; nearby vectors ⇒ similar structure/function.
- **Mean pooling** — averaging a genome's protein vectors into one genome vector.
- **VirusHostDB / ICTV** — the labeled reference viruses / their taxonomy authority.
- **Family** — the taxonomic rank we classify into (220 of them).
- **Logistic regression** — linear classifier: `w_k·x + b_k` per family, softmax on top.
- **Softmax** — turns raw scores into a probability distribution summing to 1.
- **Confidence** — the max softmax probability = probability of the winning family.
- **UMAP** — dimensionality reduction to 2-D that preserves local neighborhoods.
- **Cosine metric** — similarity by vector *angle/direction*, ignoring magnitude.
- **Viral dark matter** — environmental viral sequence unrecognized by references;
  here shows up as pervasively low classifier confidence.

---

## 9. Where everything lives (quick map)

Repo:
- `umap_atlas.py` — builds the atlas (fit reference, transform MG08, render HTML).
- `run_umap.slurm` — Slurm job to run it on Laguna.
- `MG08_umap_atlas.html` — the rendered atlas.
- `README.md` — method + usage. `Explanation.md` — this document.

Laguna (`/project/jespinoza_1537/`):
- `*_genome_embeddings.npy` / `*_genome_ids.txt` — pooled vectors + row-aligned IDs.
- `reference_genome_taxonomy.csv` — ICTV ranks for the reference.
- `MG08_predictions.csv` — genome_id, predicted_family, confidence.
- `classifier.pkl`, `label_encoder.pkl`, `classification_report.csv` — the model + its report.
- `MG08_embeddings/`, `GENOMESDB/virus_embeddings_batches/` — protein-level embeddings.
- `MG_assemblies/` — the assembled contigs (FASTA).
