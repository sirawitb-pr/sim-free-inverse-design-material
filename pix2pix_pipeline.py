"""
Simulation-Free Generative Deep Learning for Inverse Design of
4D-Printed Shape-Morphing Materials.

Trains and evaluates paired Pix2Pix conditional GAN models for a
4D-printed bilayer hydrogel system, using data from physically
fabricated, hydration-actuated samples only (no simulated/FEA data):

  - Forward model  (G_for): 2D precursor pattern  -> experimental shape
  - Inverse model  (G_inv): experimental shape     -> 2D precursor pattern

The inverse model is evaluated via cycle consistency against the
pretrained forward model, and can additionally be applied to new,
previously unseen target shapes (the "autonomous discovery" demonstration).

Usage
-----
    python pix2pix_pipeline.py

This runs the forward training + evaluation, then the inverse training +
cycle-consistency evaluation, in sequence (matching the original
two-part notebook this script was consolidated from). All outputs are
written under ./outputs/ (see CONFIGURATION below).

This is a structural reorganization of the original two-notebook-cell
implementation into one linear script. No computational logic
(architecture, hyperparameters, training loop, or evaluation metrics)
has been changed; only notebook-specific commands (package installs,
IPython download links), hardcoded Kaggle paths, and unused imports/
variables have been removed. See the "Scope note" at the bottom of this
docstring.

Scope note
----------
Two auxiliary steps referenced in the manuscript, (1) STL/mesh export of
the physical print files, and (2) OpenSCAD-based geometry generation,
are handled by separate fabrication scripts not included here. This
script covers only the deep learning training and evaluation pipeline.
"""

import os
import re
import time
import random
import zipfile

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

import lpips
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


# =============================================================================
# CONFIGURATION
# =============================================================================

# ---- Data paths ----
# Update BASE_INPUT to point to your local copy of the dataset (see README).
# Expected structure:
#   <BASE_INPUT>/Pattern_256x256/      2D precursor pattern images
#   <BASE_INPUT>/Experiment_256x256/   Corresponding experimental shape images
BASE_INPUT = os.environ.get("DATASET_DIR", "./data")
PATTERN_DIR = os.path.join(BASE_INPUT, "Pattern_256x256")
EXP_DIR = os.path.join(BASE_INPUT, "Experiment_256x256")

# Optional: directory of new, previously unseen target images for inverse
# design / discovery predictions (Figure 4e-style demonstration). Skipped
# gracefully at runtime if this directory does not exist.
NEW_EXP_DIR = os.environ.get("NEW_TARGETS_DIR", "./data/new_targets")

# ---- Output paths ----
OUT_DIR = "./outputs/pix2pix_single_run"
INVERSE_OUT_DIR = "./outputs/inverse_predictions"
for _dir in [OUT_DIR, INVERSE_OUT_DIR]:
    os.makedirs(_dir, exist_ok=True)

# ---- Device ----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Training hyperparameters (unchanged from original experiments) ----
BATCH_SIZE = 4
LEARNING_RATE = 2e-4
LAMBDA_L1 = 100
EPOCHS = 85
TEST_RATIO = 0.15
IMG_SIZE = 256
CLIP_VALUE = 0.5
SEED = 113
GEN_FEATURES = 96


# =============================================================================
# UTILITIES
# =============================================================================

def set_seed(seed):
    """Seed all relevant RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def denorm(tensor):
    """Convert a normalized (-1, 1) CHW tensor to a (0, 1) HWC numpy array."""
    return ((tensor.cpu().numpy().transpose(1, 2, 0) + 1) / 2).clip(0, 1)


# =============================================================================
# DATASET
# =============================================================================

class PairedHydrogelDataset(Dataset):
    """
    Loads matched (pattern, experimental shape) image pairs by filename.
    Each sample returns (pattern_tensor, experiment_tensor, filename).
    """

    def __init__(self, pattern_dir, exp_dir, augment=False):
        self.patterns = sorted(
            f for f in os.listdir(pattern_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        self.exp = sorted(
            f for f in os.listdir(exp_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        common = set(self.patterns) & set(self.exp)
        self.patterns = [f for f in self.patterns if f in common]
        self.exp = [f for f in self.exp if f in common]

        self.pattern_dir = pattern_dir
        self.exp_dir = exp_dir
        self.augment = augment

        self.base_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        if augment:
            self.aug_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
            ])
        else:
            self.aug_transform = None

    def __len__(self):
        return len(self.patterns)

    def __getitem__(self, idx):
        pat_path = os.path.join(self.pattern_dir, self.patterns[idx])
        exp_path = os.path.join(self.exp_dir, self.exp[idx])
        pat = Image.open(pat_path).convert("RGB")
        exp = Image.open(exp_path).convert("RGB")

        if self.augment and self.aug_transform:
            seed = random.randint(0, 2**32)
            torch.manual_seed(seed)
            pat = self.aug_transform(pat)
            torch.manual_seed(seed)
            exp = self.aug_transform(exp)

        pat = self.base_transform(pat)
        exp = self.base_transform(exp)
        return pat, exp, self.patterns[idx]


class InverseDataset(Dataset):
    """Wraps a PairedHydrogelDataset subset, swapping (pattern, exp) order
    so the inverse model is trained to map experiment -> pattern."""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        pat, exp, fname = self.dataset[self.indices[idx]]
        return exp, pat, fname  # swapped


def get_splits(seed=SEED):
    """
    Deterministic train/test split by sample ID extracted from filenames
    (falls back to a hash of the filename if no digits are found).
    Splitting is done on unique sample IDs rather than raw indices to
    ensure no data leakage between train and test sets.
    """
    set_seed(seed)
    full = PairedHydrogelDataset(PATTERN_DIR, EXP_DIR, augment=False)

    ids = []
    for f in full.patterns:
        nums = re.findall(r"\d+", f)
        ids.append(int(nums[0]) if nums else hash(f) % 10000)

    unique = sorted(set(ids))
    random.shuffle(unique)
    n_test = int(len(unique) * TEST_RATIO)
    test_ids = set(unique[:n_test])
    train_ids = set(unique[n_test:])

    def get_idx(ids_set):
        return [
            i for i, f in enumerate(full.patterns)
            if int(re.findall(r"\d+", f)[0]) in ids_set
        ]

    train_idx = get_idx(train_ids)
    test_idx = get_idx(test_ids)

    train_ds = PairedHydrogelDataset(PATTERN_DIR, EXP_DIR, augment=False)
    test_ds = PairedHydrogelDataset(PATTERN_DIR, EXP_DIR, augment=False)

    return (
        torch.utils.data.Subset(train_ds, train_idx),
        torch.utils.data.Subset(test_ds, test_idx),
    )


# =============================================================================
# MODELS
# =============================================================================

class CNNBlock(nn.Module):
    """Single downsampling conv block: Conv2d -> InstanceNorm -> LeakyReLU."""

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, 1,
                      bias=False, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)


class Generator(nn.Module):
    """U-Net generator with skip connections (standard Pix2Pix generator).
    Used as both the forward generator (G_for) and inverse generator (G_inv)."""

    def __init__(self, in_channels=3, features=96):
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        self.down1 = CNNBlock(features, features * 2, 2)
        self.down2 = CNNBlock(features * 2, features * 4, 2)
        self.down3 = CNNBlock(features * 4, features * 8, 2)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1), nn.ReLU()
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(features * 8, features * 8, 4, 2, 1),
            nn.InstanceNorm2d(features * 8), nn.ReLU(),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(features * 16, features * 4, 4, 2, 1),
            nn.InstanceNorm2d(features * 4), nn.ReLU(),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(features * 8, features * 2, 4, 2, 1),
            nn.InstanceNorm2d(features * 2), nn.ReLU(),
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(features * 4, features, 4, 2, 1),
            nn.InstanceNorm2d(features), nn.ReLU(),
        )
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, in_channels, 4, 2, 1), nn.Tanh()
        )

    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        bn = self.bottleneck(d4)
        u1 = self.up1(bn)
        u2 = self.up2(torch.cat([u1, d4], dim=1))
        u3 = self.up3(torch.cat([u2, d3], dim=1))
        u4 = self.up4(torch.cat([u3, d2], dim=1))
        return self.final_up(torch.cat([u4, d1], dim=1))


class Discriminator(nn.Module):
    """PatchGAN discriminator conditioned on the paired input image."""

    def __init__(self, in_channels=3, features=(64, 128, 256, 512)):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels * 2, features[0], 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        layers = []
        in_ch = features[0]
        for f in features[1:]:
            stride = 1 if f == features[-1] else 2
            layers.append(CNNBlock(in_ch, f, stride=stride))
            in_ch = f
        layers.append(nn.Conv2d(in_ch, 1, 4, 1, 1, padding_mode="reflect"))
        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        return self.model(self.initial(torch.cat([x, y], dim=1)))


# =============================================================================
# PART 1: FORWARD MODEL (Pattern -> Experiment)
# =============================================================================

def train_forward(seed):
    """Train the forward Pix2Pix model and log per-epoch train/test loss."""
    set_seed(seed)
    train_ds, test_ds = get_splits(seed)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    gen = Generator(features=GEN_FEATURES).to(DEVICE)
    disc = Discriminator().to(DEVICE)
    opt_g = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    opt_d = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()

    model_path = os.path.join(OUT_DIR, f"model_seed_{seed}.pth")

    epochs_list, train_D_list, train_G_list = [], [], []
    test_D_list, test_G_list = [], []

    print(f"\n--- Training forward model (seed {seed}) on {DEVICE} ---")
    for epoch in range(1, EPOCHS + 1):
        gen.train()
        disc.train()
        g_loss_sum, d_loss_sum = 0, 0
        for pat, exp, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            pat, exp = pat.to(DEVICE), exp.to(DEVICE)
            fake = gen(pat)

            real = disc(pat, exp)
            fake_d = disc(pat, fake.detach())
            d_loss = (bce(real, torch.ones_like(real))
                      + bce(fake_d, torch.zeros_like(fake_d))) / 2
            opt_d.zero_grad()
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(disc.parameters(), CLIP_VALUE)
            opt_d.step()

            adv = bce(disc(pat, fake), torch.ones_like(disc(pat, fake)))
            l1 = l1_loss(fake, exp) * LAMBDA_L1
            g_loss = adv + l1
            opt_g.zero_grad()
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(gen.parameters(), CLIP_VALUE)
            opt_g.step()

            g_loss_sum += g_loss.item()
            d_loss_sum += d_loss.item()

        avg_train_D = d_loss_sum / len(train_loader)
        avg_train_G = g_loss_sum / len(train_loader)

        gen.eval()
        disc.eval()
        test_g_loss, test_d_loss = 0, 0
        with torch.no_grad():
            for pat, exp, _ in test_loader:
                pat, exp = pat.to(DEVICE), exp.to(DEVICE)
                fake = gen(pat)
                real_pred = disc(pat, exp)
                fake_pred = disc(pat, fake)
                d_loss_test = (bce(real_pred, torch.ones_like(real_pred))
                               + bce(fake_pred, torch.zeros_like(fake_pred))) / 2
                test_d_loss += d_loss_test.item()
                adv_test = bce(disc(pat, fake), torch.ones_like(disc(pat, fake)))
                l1_test = l1_loss(fake, exp) * LAMBDA_L1
                test_g_loss += (adv_test + l1_test).item()

        avg_test_D = test_d_loss / len(test_loader)
        avg_test_G = test_g_loss / len(test_loader)

        epochs_list.append(epoch)
        train_D_list.append(avg_train_D)
        train_G_list.append(avg_train_G)
        test_D_list.append(avg_test_D)
        test_G_list.append(avg_test_G)

        print(f"Epoch {epoch:3d} | Train D: {avg_train_D:.4f}  Train G: {avg_train_G:.4f} "
              f"| Test D: {avg_test_D:.4f}  Test G: {avg_test_G:.4f}")

    torch.save(gen.state_dict(), model_path)

    metrics_df = pd.DataFrame({
        "epoch": epochs_list, "train_D_loss": train_D_list, "train_G_loss": train_G_list,
        "test_D_loss": test_D_list, "test_G_loss": test_G_list,
    })
    csv_path = os.path.join(OUT_DIR, "training_metrics.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"Training metrics saved to {csv_path}")

    gen.eval()
    test_l1 = 0
    with torch.no_grad():
        for pat, exp, _ in test_loader:
            pat, exp = pat.to(DEVICE), exp.to(DEVICE)
            test_l1 += l1_loss(gen(pat), exp).item()
    test_l1 /= len(test_loader)
    print(f"Final Test L1 for seed {seed}: {test_l1:.4f}")

    return gen, model_path, test_loader, test_l1, metrics_df


def visualise_forward(model, loader, seed, max_display=15):
    """Save a grid of (input, target, prediction) visualizations."""
    model.eval()
    samples = list(loader)
    n = min(len(samples), max_display)
    fig, axes = plt.subplots(n, 3, figsize=(9, n * 2.5))
    if n == 1:
        axes = axes.reshape(1, -1)
    l1_loss = nn.L1Loss()
    losses = []
    with torch.no_grad():
        for i in range(n):
            pat, exp, fname = samples[i]
            pat, exp = pat.to(DEVICE), exp.to(DEVICE)
            pred = model(pat)
            losses.append(l1_loss(pred, exp).item())
            axes[i, 0].imshow(denorm(pat[0])); axes[i, 0].set_title(f"Input\n{fname[0]}", fontsize=8); axes[i, 0].axis("off")
            axes[i, 1].imshow(denorm(exp[0])); axes[i, 1].set_title("Target", fontsize=8); axes[i, 1].axis("off")
            axes[i, 2].imshow(denorm(pred[0])); axes[i, 2].set_title(f"Prediction (L1={losses[-1]:.4f})", fontsize=8); axes[i, 2].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"test_visualisation_seed_{seed}.png"), dpi=150)
    plt.close(fig)
    print(f"Seed {seed} - Per-image L1: min={min(losses):.4f}, max={max(losses):.4f}, mean={np.mean(losses):.4f}")


def evaluate_forward(model, loader, split_name="test"):
    """Compute L1, PSNR, SSIM, and LPIPS for each sample in loader."""
    model.eval()
    l1_loss = nn.L1Loss()
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    lpips_fn = lpips.LPIPS(net="alex").to(DEVICE)

    l1_vals, psnr_vals, ssim_vals, lpips_vals, fnames, times = [], [], [], [], [], []
    with torch.no_grad():
        for pat, exp, fname in tqdm(loader, desc=f"Evaluating {split_name}"):
            pat, exp = pat.to(DEVICE), exp.to(DEVICE)
            start = time.time()
            pred = model(pat)
            times.append(time.time() - start)

            l1_vals.append(l1_loss(pred, exp).item())
            pred_scaled = (pred + 1) / 2
            exp_scaled = (exp + 1) / 2
            psnr_vals.append(psnr(pred_scaled, exp_scaled).item())
            ssim_vals.append(ssim(pred_scaled, exp_scaled).item())
            lpips_vals.append(lpips_fn(pred_scaled, exp_scaled).item())
            fnames.append(fname[0])

    df = pd.DataFrame({"filename": fnames, "L1": l1_vals, "PSNR": psnr_vals,
                        "SSIM": ssim_vals, "LPIPS": lpips_vals})
    summary = {
        "Metric": ["L1", "PSNR (dB)", "SSIM", "LPIPS"],
        "Mean": [np.mean(l1_vals), np.mean(psnr_vals), np.mean(ssim_vals), np.mean(lpips_vals)],
        "Std": [np.std(l1_vals), np.std(psnr_vals), np.std(ssim_vals), np.std(lpips_vals)],
        "Min": [np.min(l1_vals), np.min(psnr_vals), np.min(ssim_vals), np.min(lpips_vals)],
        "Max": [np.max(l1_vals), np.max(psnr_vals), np.max(ssim_vals), np.max(lpips_vals)],
        "Median": [np.median(l1_vals), np.median(psnr_vals), np.median(ssim_vals), np.median(lpips_vals)],
    }
    return df, pd.DataFrame(summary), np.mean(times) * 1000


def save_comparison_images(model, loader, split_name, max_samples=None):
    """Save (pattern, prediction, ground truth) comparison images per sample."""
    model.eval()
    comp_dir = os.path.join(OUT_DIR, f"comparison_images_{split_name}")
    os.makedirs(comp_dir, exist_ok=True)
    count = 0
    with torch.no_grad():
        for pat, exp, fname in tqdm(loader, desc=f"Saving {split_name} images"):
            if max_samples is not None and count >= max_samples:
                break
            pat, exp = pat.to(DEVICE), exp.to(DEVICE)
            pred = model(pat)
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(denorm(pat[0])); axes[0].set_title("Pattern (Input)"); axes[0].axis("off")
            axes[1].imshow(denorm(pred[0])); axes[1].set_title("Predicted Experiment"); axes[1].axis("off")
            axes[2].imshow(denorm(exp[0])); axes[2].set_title("Ground Truth"); axes[2].axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(comp_dir, f"{split_name}_{count:04d}_{fname[0]}.png"), dpi=150)
            plt.close(fig)
            count += 1
    print(f"Saved {count} comparison images for {split_name} set.")


def export_panel_d(metrics_df):
    """
    Save the Figure 2 panel (d) raw metrics CSV and a publication-style
    SSIM/L1 distribution plot.
    """
    panel_d_raw_df = metrics_df[["filename", "SSIM", "L1"]].copy()
    panel_d_csv_path = os.path.join(OUT_DIR, "figure2_panel_d_raw_metrics.csv")
    panel_d_raw_df.to_csv(panel_d_csv_path, index=False)
    print(f"Panel (d) raw distribution dataset saved to: {panel_d_csv_path}")

    sns.set_style("whitegrid")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.8), dpi=300)

    sns.histplot(metrics_df["SSIM"], kde=True, color="#1f77b4", ax=ax1, bins=7, alpha=0.45, edgecolor="black")
    np.random.seed(42)  # cosmetic only: fixes scatter jitter positions, does not affect any metric
    y_jitter1 = np.random.uniform(0.2, 0.8, len(metrics_df))
    ax1.scatter(metrics_df["SSIM"], y_jitter1, color="#0d3b66", alpha=0.75, s=28, zorder=5,
                label=f"Real Test Samples ($N={len(metrics_df)}$)")
    mean_ssim = metrics_df["SSIM"].mean()
    ax1.axvline(mean_ssim, color="#d62728", linestyle="--", linewidth=1.8, label=f"Mean: {mean_ssim:.3f}")
    ax1.set_xlabel("Structural Similarity (SSIM)", fontweight="bold", fontsize=10)
    ax1.set_ylabel("Frequency", fontweight="bold", fontsize=10)
    ax1.set_title("Structural Overlap (SSIM)", fontweight="bold", fontsize=11)
    ax1.legend(loc="upper left", fontsize=8.5, frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.6)

    sns.histplot(metrics_df["L1"], kde=True, color="#2ca02c", ax=ax2, bins=7, alpha=0.45, edgecolor="black")
    y_jitter2 = np.random.uniform(0.2, 0.8, len(metrics_df))
    ax2.scatter(metrics_df["L1"], y_jitter2, color="#1b4332", alpha=0.75, s=28, zorder=5,
                label=f"Real Test Samples ($N={len(metrics_df)}$)")
    mean_l1 = metrics_df["L1"].mean()
    ax2.axvline(mean_l1, color="#d62728", linestyle="--", linewidth=1.8, label=f"Mean: {mean_l1:.4f}")
    ax2.set_xlabel("L1 Reconstruction Error", fontweight="bold", fontsize=10)
    ax2.set_ylabel("Frequency", fontweight="bold", fontsize=10)
    ax2.set_title("Pixel Reconstruction Error (L1)", fontweight="bold", fontsize=11)
    ax2.legend(loc="upper right", fontsize=8.5, frameon=True)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    panel_d_plot_path = os.path.join(OUT_DIR, "figure2_panel_d_preview.png")
    plt.savefig(panel_d_plot_path, dpi=300)
    plt.close(fig)
    print(f"Panel (d) preview plot saved to: {panel_d_plot_path}")
    return panel_d_csv_path, panel_d_plot_path


def run_forward_pipeline():
    """Runs forward model training + full evaluation. Returns the loaded
    forward model (for reuse in the inverse cycle-consistency evaluation)."""
    forward_gen, forward_model_path, forward_test_loader, _, _ = train_forward(SEED)

    visualise_forward(forward_gen, forward_test_loader, SEED)

    forward_model = Generator(features=GEN_FEATURES).to(DEVICE)
    forward_model.load_state_dict(torch.load(forward_model_path, map_location=DEVICE))
    forward_model.eval()

    _, test_ds = get_splits(SEED)
    test_loader_eval = DataLoader(test_ds, batch_size=1, shuffle=False)
    train_ds, _ = get_splits(SEED)
    train_loader_eval = DataLoader(train_ds, batch_size=1, shuffle=False)

    save_comparison_images(forward_model, train_loader_eval, "train")
    save_comparison_images(forward_model, test_loader_eval, "test")

    metrics_df, summary_df, avg_time = evaluate_forward(forward_model, test_loader_eval, "test")
    print("\n===== FORWARD SUMMARY METRICS (TEST SET) =====")
    print(summary_df.to_string(index=False))
    print(f"\nAvg inference time: {avg_time:.2f} ms per image")

    report_dir = os.path.join(OUT_DIR, "forward_report")
    os.makedirs(report_dir, exist_ok=True)
    metrics_df.to_csv(os.path.join(report_dir, "per_sample_metrics.csv"), index=False)
    summary_df.to_csv(os.path.join(report_dir, "summary_metrics.csv"), index=False)

    _, panel_d_plot_path = export_panel_d(metrics_df)

    pdf_path = os.path.join(report_dir, "forward_evaluation_report.pdf")
    with PdfPages(pdf_path) as pdf:
        fig_title, ax = plt.subplots(figsize=(11, 8.5))
        ax.text(0.5, 0.8, "Pix2Pix Forward Evaluation", fontsize=24, ha="center")
        ax.text(0.5, 0.6, f"Seed={SEED}, Features={GEN_FEATURES}", fontsize=16, ha="center")
        ax.text(0.5, 0.5, f"Test set size: {len(metrics_df)}", fontsize=14, ha="center")
        ax.text(0.5, 0.4, f"Avg inference time: {avg_time:.2f} ms", fontsize=14, ha="center")
        ax.axis("off")
        pdf.savefig(fig_title); plt.close(fig_title)

        fig, ax = plt.subplots(figsize=(10, 4)); ax.axis("tight"); ax.axis("off")
        table = ax.table(cellText=summary_df.values, colLabels=summary_df.columns,
                          loc="center", cellLoc="center",
                          colColours=["#f0f0f0"] * len(summary_df.columns))
        table.auto_set_font_size(False); table.set_fontsize(12)
        pdf.savefig(fig); plt.close(fig)

        img = plt.imread(panel_d_plot_path)
        fig, ax = plt.subplots(figsize=(10, 5)); ax.imshow(img); ax.axis("off")
        pdf.savefig(fig); plt.close(fig)
    print(f"Forward PDF report saved to {pdf_path}")
    print("\nForward training and evaluation complete.")

    return forward_model


# =============================================================================
# PART 2: INVERSE MODEL & CYCLE CONSISTENCY (Experiment -> Pattern)
# =============================================================================

def train_inverse(seed):
    """Train the inverse Pix2Pix model and log per-epoch train/test loss."""
    set_seed(seed)
    train_ds, test_ds = get_splits(seed)

    train_inv = InverseDataset(train_ds, list(range(len(train_ds))))
    test_inv = InverseDataset(test_ds, list(range(len(test_ds))))
    train_loader = DataLoader(train_inv, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_inv, batch_size=1, shuffle=False)

    gen = Generator(features=GEN_FEATURES).to(DEVICE)
    disc = Discriminator().to(DEVICE)
    opt_g = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    opt_d = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()

    model_path = os.path.join(OUT_DIR, f"inverse_model_seed_{seed}.pth")

    epochs_list, train_D_list, train_G_list = [], [], []
    test_D_list, test_G_list = [], []
    train_l1_losses, test_l1_losses = [], []

    print(f"\n--- Training inverse model (seed {seed}) on {DEVICE} ---")
    for epoch in range(1, EPOCHS + 1):
        gen.train()
        disc.train()
        g_loss_sum, d_loss_sum = 0, 0
        for exp, pat, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            exp, pat = exp.to(DEVICE), pat.to(DEVICE)
            fake_pat = gen(exp)

            real = disc(exp, pat)
            fake_d = disc(exp, fake_pat.detach())
            d_loss = (bce(real, torch.ones_like(real))
                      + bce(fake_d, torch.zeros_like(fake_d))) / 2
            opt_d.zero_grad()
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(disc.parameters(), CLIP_VALUE)
            opt_d.step()

            adv = bce(disc(exp, fake_pat), torch.ones_like(disc(exp, fake_pat)))
            l1 = l1_loss(fake_pat, pat) * LAMBDA_L1
            g_loss = adv + l1
            opt_g.zero_grad()
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(gen.parameters(), CLIP_VALUE)
            opt_g.step()

            g_loss_sum += g_loss.item()
            d_loss_sum += d_loss.item()

        avg_train_G = g_loss_sum / len(train_loader)
        avg_train_D = d_loss_sum / len(train_loader)

        gen.eval()
        disc.eval()
        test_g_loss, test_d_loss, test_l1 = 0, 0, 0
        with torch.no_grad():
            for exp, pat, _ in test_loader:
                exp, pat = exp.to(DEVICE), pat.to(DEVICE)
                fake_pat = gen(exp)
                real_pred = disc(exp, pat)
                fake_pred = disc(exp, fake_pat)
                d_loss_test = (bce(real_pred, torch.ones_like(real_pred))
                               + bce(fake_pred, torch.zeros_like(fake_pred))) / 2
                test_d_loss += d_loss_test.item()
                adv_test = bce(disc(exp, fake_pat), torch.ones_like(disc(exp, fake_pat)))
                l1_test_val = l1_loss(fake_pat, pat)
                test_l1 += l1_test_val.item()
                test_g_loss += (adv_test + l1_test_val * LAMBDA_L1).item()

        avg_test_G = test_g_loss / len(test_loader)
        avg_test_D = test_d_loss / len(test_loader)
        avg_test_l1 = test_l1 / len(test_loader)

        epochs_list.append(epoch)
        train_D_list.append(avg_train_D)
        train_G_list.append(avg_train_G)
        test_D_list.append(avg_test_D)
        test_G_list.append(avg_test_G)
        train_l1_losses.append(avg_train_G)
        test_l1_losses.append(avg_test_l1)

        print(f"Epoch {epoch:3d} | Train D: {avg_train_D:.4f}  Train G: {avg_train_G:.4f} "
              f"| Test D: {avg_test_D:.4f}  Test G: {avg_test_G:.4f}")

    torch.save(gen.state_dict(), model_path)

    log_df = pd.DataFrame({"epoch": epochs_list, "train_l1": train_l1_losses, "test_l1": test_l1_losses})
    log_df.to_csv(os.path.join(OUT_DIR, f"inverse_training_log_seed_{seed}.csv"), index=False)

    fig3_b_df = pd.DataFrame({"epoch": epochs_list, "train_G_loss": train_G_list, "train_D_loss": train_D_list})
    fig3_b_df.to_csv(os.path.join(OUT_DIR, "figure3_panel_b_training_dynamics.csv"), index=False)
    print("Figure 3 (b) training dynamics CSV saved.")

    fig3_c_df = pd.DataFrame({"epoch": epochs_list, "train_G_loss": train_G_list, "test_G_loss": test_G_list})
    fig3_c_df.to_csv(os.path.join(OUT_DIR, "figure3_panel_c_generalization.csv"), index=False)
    print("Figure 3 (c) generalization trajectory CSV saved.")

    return gen, model_path, test_loader, log_df


def evaluate_cycle(inv_model, fwd_model, loader):
    """
    Cycle-consistency evaluation: shape -> inverse model -> predicted
    pattern -> forward model -> reconstructed shape, compared against
    the original input shape.
    """
    inv_model.eval()
    fwd_model.eval()
    l1_loss = nn.L1Loss()
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    lpips_fn = lpips.LPIPS(net="alex").to(DEVICE)

    l1_vals, psnr_vals, ssim_vals, lpips_vals, fnames, times = [], [], [], [], [], []
    comp_dir = os.path.join(OUT_DIR, "cycle_comparison_images")
    os.makedirs(comp_dir, exist_ok=True)

    with torch.no_grad():
        for idx, (exp, pat, fname) in enumerate(tqdm(loader, desc="Cycle evaluation")):
            exp, pat = exp.to(DEVICE), pat.to(DEVICE)
            start = time.time()
            pred_pat = inv_model(exp)
            recon_exp = fwd_model(pred_pat)
            times.append(time.time() - start)

            l1_vals.append(l1_loss(recon_exp, exp).item())
            recon_scaled = (recon_exp + 1) / 2
            exp_scaled = (exp + 1) / 2
            psnr_vals.append(psnr(recon_scaled, exp_scaled).item())
            ssim_vals.append(ssim(recon_scaled, exp_scaled).item())
            lpips_vals.append(lpips_fn(recon_scaled, exp_scaled).item())
            fnames.append(fname[0])

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(denorm(exp[0])); axes[0].set_title("Original Experiment"); axes[0].axis("off")
            axes[1].imshow(denorm(pred_pat[0])); axes[1].set_title("Predicted Pattern"); axes[1].axis("off")
            axes[2].imshow(denorm(recon_exp[0])); axes[2].set_title("Reconstructed Experiment"); axes[2].axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(comp_dir, f"cycle_test_{idx:04d}_{fname[0]}.png"), dpi=150)
            plt.close(fig)

    df = pd.DataFrame({"filename": fnames, "L1": l1_vals, "PSNR": psnr_vals,
                        "SSIM": ssim_vals, "LPIPS": lpips_vals})
    summary = {
        "Metric": ["L1", "PSNR (dB)", "SSIM", "LPIPS"],
        "Mean": [np.mean(l1_vals), np.mean(psnr_vals), np.mean(ssim_vals), np.mean(lpips_vals)],
        "Std": [np.std(l1_vals), np.std(psnr_vals), np.std(ssim_vals), np.std(lpips_vals)],
        "Min": [np.min(l1_vals), np.min(psnr_vals), np.min(ssim_vals), np.min(lpips_vals)],
        "Max": [np.max(l1_vals), np.max(psnr_vals), np.max(ssim_vals), np.max(lpips_vals)],
        "Median": [np.median(l1_vals), np.median(psnr_vals), np.median(ssim_vals), np.median(lpips_vals)],
    }
    return df, pd.DataFrame(summary), np.mean(times) * 1000


def predict_cycle_on_new(exp_dir, inv_model, fwd_model, out_dir):
    """Run the trained inverse+forward cycle on new, previously unseen
    target images (e.g., the autonomous discovery demonstration set)."""
    if not os.path.exists(exp_dir):
        print(f"Directory not found: {exp_dir}. Skipping new image prediction.")
        return
    files = [f for f in os.listdir(exp_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not files:
        print(f"No images in {exp_dir}. Skipping.")
        return

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    inv_model.eval()
    fwd_model.eval()

    def to_numpy(t):
        arr = t.cpu().squeeze(0).numpy().transpose(1, 2, 0)
        arr = (arr + 1) / 2 * 255
        return np.clip(arr, 0, 255).astype(np.uint8)

    with torch.no_grad():
        for fname in tqdm(files, desc="Predicting new images"):
            img = Image.open(os.path.join(exp_dir, fname)).convert("RGB")
            inp = transform(img).unsqueeze(0).to(DEVICE)
            pred_pat = inv_model(inp)
            recon_exp = fwd_model(pred_pat)

            pat_np = to_numpy(pred_pat)
            recon_np = to_numpy(recon_exp)
            Image.fromarray(pat_np).save(os.path.join(out_dir, f"predicted_pattern_{fname}"))
            Image.fromarray(recon_np).save(os.path.join(out_dir, f"reconstructed_experiment_{fname}"))

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(np.array(img)); axes[0].set_title("Experiment (Input)"); axes[0].axis("off")
            axes[1].imshow(pat_np); axes[1].set_title("Predicted Pattern"); axes[1].axis("off")
            axes[2].imshow(recon_np); axes[2].set_title("Reconstructed Experiment"); axes[2].axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"cycle_comparison_{fname}"), dpi=150)
            plt.close(fig)
    print(f"All predictions saved to {out_dir}")


def run_inverse_pipeline(forward_model):
    """Runs inverse model training + cycle-consistency evaluation, using
    the already-trained forward_model passed in from run_forward_pipeline()."""
    inverse_gen, inverse_model_path, inverse_test_loader, _ = train_inverse(SEED)

    cycle_df, cycle_summary, cycle_avg_time = evaluate_cycle(inverse_gen, forward_model, inverse_test_loader)

    print("\n===== CYCLE SUMMARY METRICS =====")
    print(cycle_summary.to_string(index=False))
    print(f"\nAvg cycle inference time: {cycle_avg_time:.2f} ms per image")

    cycle_report_dir = os.path.join(OUT_DIR, "cycle_report")
    os.makedirs(cycle_report_dir, exist_ok=True)
    cycle_df.to_csv(os.path.join(cycle_report_dir, "cycle_per_sample_metrics.csv"), index=False)
    cycle_summary.to_csv(os.path.join(cycle_report_dir, "cycle_summary_metrics.csv"), index=False)

    fig3_d_raw_df = cycle_df[["filename", "SSIM", "L1"]].copy()
    fig3_d_csv_path = os.path.join(OUT_DIR, "figure3_panel_d_raw_metrics.csv")
    fig3_d_raw_df.to_csv(fig3_d_csv_path, index=False)
    print(f"Figure 3 (d) raw distribution dataset saved to: {fig3_d_csv_path}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].hist(cycle_df["L1"], bins=20, alpha=0.7, color="green", edgecolor="black")
    axes[0, 0].set_title("Cycle L1 Histogram")
    axes[0, 1].boxplot(cycle_df[["L1", "PSNR", "SSIM", "LPIPS"]].values,
                        labels=["L1", "PSNR", "SSIM", "LPIPS"])
    axes[0, 1].set_title("Cycle Metrics Boxplot")
    axes[1, 0].bar(range(len(cycle_df)), cycle_df["L1"], alpha=0.7, color="green")
    axes[1, 0].set_title("Per-sample Cycle L1"); axes[1, 0].set_xlabel("Sample index"); axes[1, 0].set_ylabel("L1")
    axes[1, 1].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(cycle_report_dir, "cycle_plots.png"), dpi=150)
    plt.close(fig)

    pdf_cycle = os.path.join(cycle_report_dir, "cycle_evaluation_report.pdf")
    with PdfPages(pdf_cycle) as pdf:
        fig_title, ax = plt.subplots(figsize=(11, 8.5))
        ax.text(0.5, 0.8, "Cycle Consistency Evaluation", fontsize=24, ha="center")
        ax.text(0.5, 0.6, f"Seed={SEED}, Features={GEN_FEATURES}", fontsize=16, ha="center")
        ax.text(0.5, 0.5, f"Test set size: {len(cycle_df)}", fontsize=14, ha="center")
        ax.text(0.5, 0.4, f"Avg cycle inference time: {cycle_avg_time:.2f} ms", fontsize=14, ha="center")
        ax.axis("off")
        pdf.savefig(fig_title); plt.close(fig_title)

        fig, ax = plt.subplots(figsize=(10, 4)); ax.axis("tight"); ax.axis("off")
        table = ax.table(cellText=cycle_summary.values, colLabels=cycle_summary.columns,
                          loc="center", cellLoc="center",
                          colColours=["#f0f0f0"] * len(cycle_summary.columns))
        table.auto_set_font_size(False); table.set_fontsize(12)
        pdf.savefig(fig); plt.close(fig)

        img = plt.imread(os.path.join(cycle_report_dir, "cycle_plots.png"))
        fig, ax = plt.subplots(figsize=(10, 8)); ax.imshow(img); ax.axis("off")
        pdf.savefig(fig); plt.close(fig)
    print(f"Cycle PDF report saved to {pdf_cycle}")

    predict_cycle_on_new(NEW_EXP_DIR, inverse_gen, forward_model, INVERSE_OUT_DIR)

    zip_path = os.path.join(OUT_DIR, "inverse_cycle_results.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(INVERSE_OUT_DIR):
            for f in files:
                zipf.write(os.path.join(root, f), arcname=os.path.join("inverse_predictions", f))
        for d in [cycle_report_dir, os.path.join(OUT_DIR, "cycle_comparison_images")]:
            if os.path.exists(d):
                for root, _, files in os.walk(d):
                    for f in files:
                        zipf.write(os.path.join(root, f), arcname=os.path.join(os.path.basename(d), f))
        for f in [inverse_model_path,
                  os.path.join(OUT_DIR, f"inverse_training_log_seed_{SEED}.csv"),
                  os.path.join(OUT_DIR, "figure3_panel_b_training_dynamics.csv"),
                  os.path.join(OUT_DIR, "figure3_panel_c_generalization.csv"),
                  os.path.join(OUT_DIR, "figure3_panel_d_raw_metrics.csv"),
                  os.path.join(cycle_report_dir, "cycle_per_sample_metrics.csv"),
                  os.path.join(cycle_report_dir, "cycle_summary_metrics.csv"), pdf_cycle]:
            if os.path.exists(f):
                zipf.write(f, arcname=os.path.basename(f))
    print(f"\nAll inverse and cycle results zipped to {zip_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    trained_forward_model = run_forward_pipeline()
    run_inverse_pipeline(trained_forward_model)
