# Simulation-Free Experiment-Trained Deep Learning for Generative Inverse Design of 4D-Printed Shape-Morphing Materials

Forward and inverse Pix2Pix conditional GAN models for the pattern &lt;-&gt;
shape mapping of a 4D-printed bilayer hydrogel system, trained entirely
on physically fabricated, hydration-actuated experimental data (no
finite element or other simulated training data is used).

This repository accompanies the manuscript submission and reproduces the
forward model training, inverse model training, and cycle-consistency
evaluation reported there.

## Contents

A single script, `pix2pix_pipeline.py`, contains the full pipeline:

- Dataset loading and the train/test split
- Shared Generator (U-Net) / Discriminator (PatchGAN) architectures
- Forward model training and evaluation (Pattern -> Experiment)
- Inverse model training and cycle-consistency evaluation (Experiment -> Pattern)

Running the script executes both stages in sequence, forward model first
(producing the checkpoint the inverse model's cycle-consistency
evaluation depends on), then the inverse model.

## Dataset

The dataset consists of 150 paired samples (128 train / 22 test), each
pair comprising:

- a 2D precursor pattern image, and
- the corresponding experimentally measured, hydration-actuated shape image,

both physically fabricated and imaged (no simulated data).

The full dataset is deposited at Mendeley Data: 10.17632/5jws9g65yf.1.

Local folder structure after downloading:

```
data/
  Pattern_256x256/       precursor pattern images
  Experiment_256x256/    corresponding experimental shape images
  new_targets/           (optional) previously unseen target shapes for
                         inverse design / discovery predictions
```

Set the `DATASET_DIR` environment variable to point to this folder, or
edit `BASE_INPUT` directly near the top of `pix2pix_pipeline.py`.

## Setup

```bash
pip install -r requirements.txt
```

Training and evaluation require a CUDA-capable GPU for reasonable
runtime, but will fall back to CPU automatically if none is available.

## Usage

```bash
python pix2pix_pipeline.py
```

All outputs (trained model weights, per-epoch training logs, per-sample
evaluation metrics, and the figure-panel CSVs/plots used in the
manuscript) are written under `./outputs/`.

## Reproducibility notes

- A fixed random seed (`SEED = 113`, near the top of the script) is used
  for the train/test split and model initialization. Exact numerical
  reproduction of the reported metrics may still vary slightly between
  hardware/library versions due to non-deterministic GPU operations,
  which is expected and normal for GAN training.
- The random seed used for scatter-plot jitter in the Figure 2 panel
  (d) visualization (`np.random.seed(42)` inside `export_panel_d()`) is
  cosmetic only and does not affect any reported metric. It exists
  solely to make the plotted point positions reproducible.

## Scope note

This repository covers only the deep learning pipeline (forward model, inverse model, and cycle-consistency). The two auxiliary steps mentioned in the manuscript (generating geometries in OpenSCAD and exporting STL/mesh print files) are handled by separate fabrication scripts that are not included here.
