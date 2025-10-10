import os
import math
import random
from dataclasses import dataclass
from typing import Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms

# ----------------------------
# Config
# ----------------------------

@dataclass
class Config:
    data_root: str = "./data"
    seed: int = 1337
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset / bias
    epsilon: float = 0.1           # fraction of bias-conflicting minority samples
    train_size_per_class: int = 5000  # number of (digit==0) and (digit==1) images used to form biased train
    test_size_per_class: int = 1000   # for balanced test set per class

    # Backbone pretrain (on plain MNIST 0/1)
    out_dim: int = 128
    pretrain_backbone_epochs: int = 2
    pretrain_batch_size: int = 512
    pretrain_lr: float = 5e-3

    # Head training (biased Colored MNIST)
    head_lr: float = 1e-2          # constant step size h
    head_steps: int = 40000        # T steps
    log_every: int = 100
    print_every: int = 10000

    # Loader sizes (for near full-batch, make this very large)
    train_batch_size: int = 4096
    eval_batch_size: int = 4096

    # Output
    out_dir: str = "./runs/colored_mnist_eps_0p1"


# ----------------------------
# Utils
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# ----------------------------
# Dataset Construction Helper
# ----------------------------

class ColoredMNISTBiased(Dataset):
    """
    2-label task (y in {0,1}) from MNIST digits:
      y = 0 if digit==0 else 1 if digit==1
    Spurious feature = background color (red/green):
      Majority (bias-aligned): y=0 -> red bg; y=1 -> green bg
      Minority (bias-conflicting): flip background with prob epsilon
    We track group label: 'maj' (aligned) or 'min' (conflicting).
    """

    def __init__(self, xs: torch.Tensor, ys: torch.Tensor, epsilon: float, img_size: int = 28):
        """
        xs: (N,1,28,28) grayscale images, values in [0,1]
        ys: (N,) labels 0 or 1 (digit 0 vs digit 1)
        epsilon: prob. to flip the color mapping (creates minority)
        """
        assert xs.ndim == 4 and xs.shape[1] == 1 and xs.shape[2] == img_size and xs.shape[3] == img_size
        assert ys.ndim == 1
        self.x_gray = xs.clone()
        self.y = ys.clone().long()
        self.epsilon = float(epsilon)
        self.img_size = img_size
        # group 0=majority (aligned color), 1=minority (conflicting color)
        self.group = torch.zeros_like(self.y)

    def __len__(self):
        return self.y.shape[0]

    @staticmethod
    def _colorize_background(x_gray: torch.Tensor, y: int, flip: bool) -> torch.Tensor:
        """
        x_gray: (1,H,W) in [0,1], digits are ~white strokes, background ~black.
        We fill background with a color (red or green) depending on label y and flip flag.
        """
        # Create RGB
        H, W = x_gray.shape[1], x_gray.shape[2]
        rgb = torch.zeros(3, H, W)

        # Decide target color based on y and flip
        # majority mapping: y=0 -> red, y=1 -> green
        use_red = (y == 0)
        if flip:
            use_red = not use_red

        # Background mask: where grayscale is dark (tune threshold)
        bg = (x_gray < 0.2).float()
        fg = 1.0 - bg

        if use_red:
            # red background
            rgb[0] = bg  # R
            rgb[1] = 0.0
            rgb[2] = 0.0
        else:
            # green background
            rgb[0] = 0.0
            rgb[1] = bg  # G
            rgb[2] = 0.0

        # Draw foreground (digit) in white on top
        for c in range(3):
            rgb[c] = torch.clamp(rgb[c] + fg * x_gray[0], 0.0, 1.0)

        return rgb

    def __getitem__(self, idx: int):
        xg = self.x_gray[idx]  # (1,28,28)
        y = int(self.y[idx].item())
        # Flip mapping with prob epsilon
        flip = (random.random() < self.epsilon)
        group = 1 if flip else 0  # 1=minority, 0=majority
        rgb = self._colorize_background(xg, y=y, flip=flip)
        return rgb, y, group


def _filter_mnist_01(dataset: datasets.MNIST, max_per_class: int) -> Tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    counts = {0: 0, 1: 0}
    for img, label in dataset:
        if label in [0, 1] and counts[int(label)] < max_per_class:
            xs.append(img)  # PIL Image -> transform later
            ys.append(0 if label == 0 else 1)
            counts[int(label)] += 1
        if counts[0] >= max_per_class and counts[1] >= max_per_class:
            break
    to_tensor = transforms.ToTensor()  # [0,1], shape (1,28,28)
    xs = torch.stack([to_tensor(img) for img in xs], dim=0)
    ys = torch.tensor(ys, dtype=torch.long)
    return xs, ys

def make_colored_mnist_biased(cfg: Config):
    # Base MNIST
    base_train = datasets.MNIST(cfg.data_root, train=True, download=True)
    base_test = datasets.MNIST(cfg.data_root, train=False, download=True)

    # (A) Pretrain backbone data: plain MNIST (0/1), balanced
    pretrain_xs, pretrain_ys = _filter_mnist_01(base_train, max_per_class=cfg.train_size_per_class)
    # (B) Biased training data: reuse those xs/ys but colorize with epsilon flips
    train_ds = ColoredMNISTBiased(pretrain_xs, pretrain_ys, epsilon=cfg.epsilon)

    # (C) Balanced test set (colorized with 50/50 groups to match paper’s balanced eval)
    test_xs, test_ys = _filter_mnist_01(base_test, max_per_class=cfg.test_size_per_class)
    # Build two test splits: aligned and conflicting, then concatenate to get 50/50
    test_aligned = ColoredMNISTBiased(test_xs, test_ys, epsilon=0.0)  # all aligned
    test_conflict = ColoredMNISTBiased(test_xs, test_ys, epsilon=1.0) # all flipped
    # Half-and-half balanced
    half_n = min(len(test_aligned), len(test_conflict))
    # Slice equal halves
    bal_test = torch.utils.data.ConcatDataset([
        torch.utils.data.Subset(test_aligned, list(range(half_n))),
        torch.utils.data.Subset(test_conflict, list(range(half_n))),
    ])

    return train_ds, bal_test, (pretrain_xs, pretrain_ys)


# ----------------------------
# Models
# ----------------------------

class SmallBackbone(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        # Expect RGB (3x28x28)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14x14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 7x7
        )
        self.fc = nn.Linear(64 * 7 * 7, out_dim)

    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        z = self.fc(h)  # (B, out_dim)
        return z

class LinearHead(nn.Module):
    def __init__(self, in_dim=128, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, feats):
        return self.fc(feats)


# ----------------------------
# Backbone pretraining
# ----------------------------

def pretrain_backbone_on_plain_mnist(cfg: Config, backbone: SmallBackbone, xs: torch.Tensor, ys: torch.Tensor):
    """
    Pretrain the backbone for a couple of epochs on *plain grayscale* MNIST 0/1.
    Convert grayscale to pseudo-RGB by repeating channels; use a small linear head for pretraining, then discard head.
    """
    print("[Pretrain] backbone on plain MNIST (0/1)")
    device = cfg.device
    backbone = backbone.to(device)
    head = LinearHead(in_dim=backbone.fc.out_features, num_classes=2).to(device)

    # Build pretrain loader (repeat grayscale to 3 channels)
    class PlainMNIST01(Dataset):
        def __init__(self, xs, ys):
            self.xs = xs
            self.ys = ys
        def __len__(self): return len(self.ys)
        def __getitem__(self, i):
            x = self.xs[i].repeat(3, 1, 1)  # (3,28,28)
            y = self.ys[i]
            return x, y

    pre_ds = PlainMNIST01(xs, ys)
    pre_loader = DataLoader(pre_ds, batch_size=cfg.pretrain_batch_size, shuffle=True, num_workers=2, pin_memory=True)

    opt = torch.optim.SGD(list(backbone.parameters()) + list(head.parameters()), lr=cfg.pretrain_lr, momentum=0.9)
    for ep in range(cfg.pretrain_backbone_epochs):
        backbone.train(); head.train()
        running = 0.0
        for x, y in pre_loader:
            x = x.to(device); y = y.to(device)
            z = head(backbone(x))
            loss = F.cross_entropy(z, y)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item() * x.size(0)
        print(f"[Pretrain] epoch {ep+1}/{cfg.pretrain_backbone_epochs} loss={running/len(pre_ds):.4f}")

    # Freeze backbone
    for p in backbone.parameters():
        p.requires_grad = False
    self_pre_head = head  # keep the trained linear probe
    for p in self_pre_head.parameters(): p.requires_grad = False
    return backbone, self_pre_head
    
@torch.no_grad()
def cache_features(backbone, pre_head, dataset, device, batch_size=8192, num_workers=4):
    backbone.eval(); pre_head.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True, persistent_workers=True)

    R_list, S_list, Y_list, G_list = [], [], [], []
    for x, y, g in loader:
        x = x.to(device, non_blocking=True)
        r_vec = backbone(x)                                # (B, d_r)
        rg = (x[:,0] - x[:,1]).unsqueeze(1)                # (B,1,H,W)
        s_vec = F.adaptive_avg_pool2d(rg, (4,4)).view(x.size(0), -1)  # (B,16)
        R_list.append(r_vec.detach().cpu())
        S_list.append(s_vec.detach().cpu())
        Y_list.append(y.cpu()); G_list.append(g.cpu())

    R = torch.cat(R_list).to(device)
    S = torch.cat(S_list).to(device)
    R = (R - R.mean(0, keepdim=True)) / (R.std(0, keepdim=True) + 1e-8)
    S = (S - S.mean(0, keepdim=True)) / (S.std(0, keepdim=True) + 1e-8)

    Feat = torch.cat([R, S], dim=1)  
    Y = torch.cat(Y_list).to(device)
    G = torch.cat(G_list).to(device)
    return Feat, Y, G

    

# ----------------------------
# Visualization utils
# ----------------------------

import matplotlib.pyplot as plt

def show_colored_mnist_examples(dataset, cfg, n_per_group=5):
    """Display examples of majority (g=0) and minority (g=1) samples."""
    # Collect indices by group
    idx_major = [i for i, (_, _, g) in enumerate(dataset) if g == 0][:n_per_group]
    idx_minor = [i for i, (_, _, g) in enumerate(dataset) if g == 1][:n_per_group]

    fig, axes = plt.subplots(2, n_per_group, figsize=(2.5*n_per_group, 5))
    for row, idx_list in enumerate([idx_major, idx_minor]):
        for col, idx in enumerate(idx_list):
            x, y, g = dataset[idx]
            img = x.permute(1, 2, 0).numpy()       # (H,W,3) for plotting
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)  # normalize for display

            axes[row, col].imshow(img)
            axes[row, col].set_title(f"Label={y}, Group={g}")
            axes[row, col].axis("off")

    axes[0, 0].set_ylabel("Majority (g=0)", fontsize=12)
    axes[1, 0].set_ylabel("Minority (g=1)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, "ds_sample.png"), dpi=180)
    plt.show()

import re, json, glob
import numpy as np
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize


def parse_epsilon_from_dir(run_dir: str) -> float:
    # expects ".../notebook_{epsilon}_{outdim}"
    m = re.search(r'notebook_([0-9.]+)_[0-9]+$', run_dir.rstrip("/"))
    if not m: raise ValueError(f"Could not parse epsilon from {run_dir}")
    return float(m.group(1))

def load_all_runs(root="./runs"):
    runs = {}
    for p in sorted(glob.glob(os.path.join(root, "notebook_*_*"))):
        log_path = os.path.join(p, "logs.json")
        if not os.path.exists(log_path): 
            continue
        eps = parse_epsilon_from_dir(p)
        with open(log_path) as f:
            logs = json.load(f)
        runs[eps] = {
            "dir": p,
            "t": np.asarray(logs["t"], dtype=float),
            "one_minus_pmin": np.asarray(logs["one_minus_pmin"], dtype=float),
            "one_minus_pmaj": np.asarray(logs["one_minus_pmaj"], dtype=float),
            "balanced_test_loss": np.asarray(logs["balanced_test_loss"], dtype=float),
        }
    if not runs:
        raise RuntimeError("No runs found under ./runs/notebook_*_*")
    return dict(sorted(runs.items()))  # sort by epsilon

def plot_1_over_t_decay(runs, H, cfg, group="min", t_min=5e5, cmap_name="viridis"):
    """
    Plots 1 - p_y for all eps, focusing on the tail (t >= t_min).
    Colors encode epsilon;
    Adds the theory ~ 1 / [4 * (...) * h * t] line on the same tail region.
    """
    plt.figure(figsize=(7.2, 4.8))

    # color map over epsilon
    eps_vals = np.array(sorted(runs.keys()))
    norm = Normalize(vmin=eps_vals.min(), vmax=eps_vals.max())
    cmap = get_cmap(cmap_name)

    # draw curves (thin, alpha for overlap)
    for eps in eps_vals:
        R = runs[eps]
        t = np.asarray(R["t"])
        y = np.asarray(R["one_minus_pmin"] if group == "min" else R["one_minus_pmaj"])

        mask = t >= t_min
        if not np.any(mask):
            continue
        t_tail = t[mask]
        y_tail = np.maximum(y[mask], 1e-12)

        plt.semilogy(t_tail, y_tail, lw=1.2, alpha=0.9, color=cmap(norm(eps)))

    # reference theory on the same tail window
    # pick a representative epsilon (median) just for the slope/scale line
    eps_ref = float(np.median(eps_vals))
    t_ref = next(iter(runs.values()))["t"]
    t_mask = t_ref >= t_min
    if np.any(t_mask):
        t_th = t_ref[t_mask]
        if group == "min":
            theory = 1.0 / (4.0 * eps_ref * H * t_th)
            label = rf"Theory $\sim \frac{{1}}{{4\,\varepsilon\,h\,t}}$ ($\varepsilon={eps_ref:g}$)"
        else:
            theory = 1.0 / (4.0 * (1.0 - eps_ref) * H * t_th)
            label = rf"Theory $\sim \frac{{1}}{{4(1-\varepsilon)\,h\,t}}$ ($\varepsilon={eps_ref:g}$)"
        plt.semilogy(t_th, theory, "k--", lw=2.0, label=label)

    # axes/labels
    plt.xlabel("t (steps)")
    plt.ylabel(f"1 - p_y ({'minority' if group=='min' else 'majority'})")
    plt.title(f"Decay of 1 - p_y ({group}) across $\\varepsilon$")
    # colorbar for epsilon
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # fixes ValueError
    cbar = plt.colorbar(sm, ax=plt.gca(), pad=0.01)
    cbar.set_label(r"$\varepsilon$")
    # one compact legend only for theory line
    plt.legend(loc="upper right", frameon=False)

    plt.tight_layout()
    out_name = f"Error_decay_{group}ority_group_tail_tmin_{int(t_min)}.png"
    plt.savefig(os.path.join(cfg.out_dir, out_name), dpi=180)
    plt.show()

    
def plot_speed_vs_epsilon_at_last(runs, H, cfg):
    eps_arr, qmin, qmaj = [], [], []
    t_star = None

    for eps, R in runs.items():
        t = R["t"]; 
        t_star = t[-1] if t_star is None else min(t_star, t[-1])  # ensure common t* if grids differ

    for eps, R in runs.items():
        t = R["t"]
        # interpolate (or pick closest) to t_star
        idx = np.argmin(np.abs(t - t_star))
        om_min = R["one_minus_pmin"][idx]
        om_maj = R["one_minus_pmaj"][idx]
        eps_arr.append(eps)
        qmin.append( (om_min) * (4.0 * H * t[idx]) )
        qmaj.append( (om_maj) * (4.0 * H * t[idx]) )

    eps_arr = np.array(eps_arr)
    qmin = np.array(qmin)
    qmaj = np.array(qmaj)

    # Theory curves: 1/ε and 1/(1-ε)
    egrid = np.linspace(max(1e-3, eps_arr.min()), min(0.999, eps_arr.max()), 400)
    theory_min = 2.4 / egrid
    theory_maj = 6.2 / (1.0 - egrid)

    plt.figure(figsize=(7,5))
    plt.scatter(eps_arr, qmin, s=25, label="Empirical (minority)")
    plt.plot(egrid, theory_min, "k--", lw=2, label=r"Theory $\propto \frac{1}{\varepsilon}$")
    plt.xlabel(r"$\varepsilon$"); plt.ylabel(r"$(1 - p_{min}) \cdot (4 h T^*)$")
    plt.title(r"Minority speed scaling vs $\varepsilon$")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, "Minority_speed_scaling_vs_eps"), dpi=180)
    plt.show()

    plt.figure(figsize=(7,5))
    plt.scatter(eps_arr, qmaj, s=25, label="Empirical (majority)")
    plt.plot(egrid, theory_maj, "k--", lw=2, label=r"Theory $\propto \frac{1}{1-\varepsilon}$")
    plt.xlabel(r"$\varepsilon$"); plt.ylabel(r"$(1 - p_{maj}) \cdot (4 h T^*)$")
    plt.title(r"Majority speed scaling vs $\varepsilon$")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, "Majority_speed_scaling_vs_eps"), dpi=180)
    plt.show()

def plot_balanced_loss_over_time(runs, H, cfg, n_eps_to_show=10, t_min=None, t_max=None, cmap_name="plasma"):
    """
    Balanced loss vs time across epsilons.
      - Shows at most n_eps_to_show curves evenly sampled across epsilons.
      - Optionally trims to [t_min, t_max].
      - Single theory reference (∝ 1/t) with median epsilon.
    """
    all_eps = np.array(sorted(runs.keys()))
    # pick ε evenly across the available runs
    if n_eps_to_show >= len(all_eps):
        sel_eps = all_eps
    else:
        idx = np.linspace(0, len(all_eps)-1, n_eps_to_show).round().astype(int)
        sel_eps = all_eps[idx]

    # color mapping for ε
    norm = Normalize(vmin=all_eps.min(), vmax=all_eps.max())
    cmap = get_cmap(cmap_name)

    plt.figure(figsize=(7.5, 4.8))
    
    # empirical curves
    for eps in sel_eps:
        R = runs[float(eps)]
        t = np.asarray(R["t"])
        L = np.asarray(R["balanced_test_loss"])
        if t_min is not None:
            m = t >= t_min
            t, L = t[m], L[m]
        if t_max is not None:
            m = t <= t_max
            t, L = t[m], L[m]
        if len(t) == 0:
            continue
        plt.plot(t, L, color=cmap(norm(eps)), lw=1.6, alpha=0.9)

    # single theory reference (~1/t)
    eps_ref = float(np.median(all_eps))
    t_ref = next(iter(runs.values()))["t"]
    if t_min is not None:
        t_ref = t_ref[t_ref >= t_min]
    if t_max is not None:
        t_ref = t_ref[t_ref <= t_max]
    theory = 1.0 / (8.0 * eps_ref * (1.0 - eps_ref) * H * t_ref)
    plt.plot(t_ref, theory, "k--", lw=2.0, label=rf"Theory $\propto\frac{{1}}{{t}}$")

    plt.xlabel("t (steps)")
    plt.ylabel("Balanced CE loss")
    plt.title("Balanced loss over time (empirical vs theory)")
    if t_min or t_max:
        plt.xlim(left=t_min if t_min else None, right=t_max if t_max else None)

    # colorbar for ε instead of legend
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), pad=0.01)
    cbar.set_label(r"$\varepsilon$")
    plt.legend(loc="upper right", frameon=False)

    plt.tight_layout()
    out_name = f"Balanced_loss_over_time.png"
    plt.savefig(os.path.join(cfg.out_dir, out_name), dpi=180)
    plt.show()
    
def plot_balanced_loss_normalized_at_last(runs, H, cfg):
    eps_arr, L_last = [], []
    t_star = None

    # Find the common last step among all runs
    for eps, R in runs.items():
        t = R["t"]
        t_star = t[-1] if t_star is None else min(t_star, t[-1])

    # Collect final balanced losses
    for eps, R in runs.items():
        t = np.asarray(R["t"])
        L = np.asarray(R["balanced_test_loss"])
        idx = np.argmin(np.abs(t - t_star))
        eps_arr.append(eps)
        L_last.append(L[idx])

    eps_arr = np.array(eps_arr)
    L_last = np.array(L_last)

    # Build the theoretical curve 1 / [eps * (1 - eps)]
    egrid = np.linspace(max(1e-3, eps_arr.min()), min(0.999, eps_arr.max()), 400)
    theory = 1.0 / (egrid * (1.0 - egrid))
    theory /= theory.max()  # optional: normalize so both fit in scale
    L_last_scaled = L_last / L_last.max()

    plt.figure(figsize=(7,5))
    plt.scatter(eps_arr, L_last_scaled, s=35, label="Empirical (scaled)")
    plt.plot(egrid, theory, "k--", lw=2, label=r"Theory $\propto \frac{1}{\varepsilon(1-\varepsilon)}$")
    plt.xlabel(r"$\varepsilon$")
    plt.ylabel(r"Balanced CE loss (scaled)")
    plt.title(r"Balanced loss across $\varepsilon$ at large $T^*$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, "Balanced_loss_vs_eps.png"), dpi=180)
    plt.show()
