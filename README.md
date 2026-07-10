# Viral Dark Matter — ESM-2 Metagenomic Atlas

Interactive UMAP visualization of viral metagenomic sequences in ESM-2 protein
language model embedding space, in the style of Meta's ESM Metagenomic Atlas.

Reference viruses (VirusHostDB) are colored by taxonomic **order**; unknown
environmental contigs from sample **MG08** are overlaid as grey points, each
carrying its classifier-predicted family and confidence on hover.

## What this repo contains

| File | Purpose |
|------|---------|
| `umap_atlas.py` | Builds the atlas: fits UMAP on reference embeddings, transforms MG08 into the same space, exports a standalone interactive Plotly HTML. |
| `run_umap.slurm` | Slurm batch script to run `umap_atlas.py` on USC CARC Laguna. |
| `MG08_umap_atlas.html` | Rendered output — standalone WebGL Plotly atlas (open in any browser). |

## Method

The pipeline classifies unknown environmental viral metagenomic sequences using
ESM-2 (650M) protein embeddings, then visualizes them against a labeled
reference in a shared 2D UMAP space.

**Embedding space strategy (memory-safe).** The reference (~3.05M genomes,
15 GB) and MG08 (~3.70M genomes, 18 GB) embedding matrices are too large to
co-embed in RAM. Instead:

1. Memory-map both `.npy` matrices (`mmap_mode="r"`) so only sampled rows enter RAM.
2. Randomly subsample the reference and **fit** UMAP on it (`metric="cosine"`).
3. Subsample MG08 and **transform** it into the already-fitted reference space
   (`reducer.transform`), so both datasets share coordinates.
4. Merge reference taxonomy and MG08 predictions, export one Plotly HTML.

**Visualization.** Plotly `Scattergl` (WebGL) for smooth pan/zoom at scale.
Reference points are colored by taxonomic order (top ~30 orders colored, rarer
orders grouped as `other`); MG08 points are grey with predicted family +
confidence on hover. Black background, equal aspect ratio.

Default render: **150K reference + 150K MG08 = 300K points**.

## Configuration

`umap_atlas.py` reads these environment variables (with defaults):

| Var | Default | Meaning |
|-----|---------|---------|
| `N_REF` | `150000` | reference points to fit UMAP on |
| `N_MG08` | `150000` | MG08 points to transform + overlay |
| `OUT_HTML` | `.../MG08_umap_atlas.html` | output path |
| `SLURM_CPUS_PER_TASK` | `16` | UMAP `n_jobs` |

## Running on Laguna (USC CARC)

```bash
# activate the shared conda env
source ~/.bashrc && module load conda/25.11.0
conda activate /project/jespinoza_1537/envs/metagenome

# submit
sbatch run_umap.slurm          # full 150K x 150K render, ~10 min on 16 CPUs

# quick smoke test (seconds)
N_REF=2000 N_MG08=2000 OUT_HTML=/tmp/_smoke.html python umap_atlas.py
```

Requires `umap-learn`, `plotly`, plus `numpy`, `pandas`, `scikit-learn`,
`pyarrow` (already in the `metagenome` env).

## Input files (on Laguna, `/project/jespinoza_1537/`)

- `reference_genome_embeddings.npy` — 3,052,046 × 1280 reference vectors
- `reference_genome_ids.txt` / `reference_genome_taxonomy.csv` — IDs + 15 ICTV ranks
- `MG08_genome_embeddings.npy` — 3,701,871 × 1280 MG08 vectors
- `MG08_genome_ids.txt` / `MG08_predictions.csv` — IDs + predicted family/confidence

## Project context

Part of a viral metagenomics classification pipeline: assembled contigs →
Prodigal ORFs → ESM-2 embeddings → genome-level mean pooling → logistic
regression family classifier (220 families) → this atlas. Developed during a
SURE-IRP summer research project at Keck Graduate Institute (KGI).
