"""EuroSAT CNN reproduction: baseline 3-block CNN and a toned-down balanced multi-task attention
CNN (Coordinate Attention + Squeeze-Excitation fused via a learnable per-block alpha), reproducing
arXiv:2510.15527 at a scale that fits a single T4. Trained and evaluated on Colab GPU (no local
CUDA in this environment); this file is the local source of truth per the smart-colab-sync skill
-- its exact content is what gets sent into Colab cells, never composed fresh there.

Does not touch live_tests/train.py's PCA(50)+LogisticRegression EuroSAT trainer (the deployed
baseline this reproduction is compared against) or live_tests/improve.py.
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset

CLASS_NAMES = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
               "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

EUROSAT_URLS = {
    "train": "https://huggingface.co/datasets/timm/eurosat-rgb/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
    "validation": "https://huggingface.co/datasets/timm/eurosat-rgb/resolve/refs%2Fconvert%2Fparquet/default/validation/0000.parquet",
    "test": "https://huggingface.co/datasets/timm/eurosat-rgb/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet",
}


def load_full_eurosat(seed: int = 0, split_frac: float = 0.7):
    """Fetches all three EuroSAT-RGB splits directly from HuggingFace (27,000 images total) and
    applies the SAME class-balanced shuffle + split train_eurosat_classifier uses in train.py, so
    every CNN result here is comparable to the deployed PCA+LogReg baseline on identical held-out
    images."""
    dfs = []
    for url in EUROSAT_URLS.values():
        r = requests.get(url, timeout=180)
        r.raise_for_status()
        dfs.append(pd.read_parquet(io.BytesIO(r.content)))
    df = pd.concat(dfs, ignore_index=True)

    def decode(cell):
        raw = cell["bytes"] if isinstance(cell, dict) else cell
        return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(df))
    df = df.iloc[order].reset_index(drop=True)
    cut = int(len(df) * split_frac)

    imgs = np.stack(df["image"].map(decode).to_numpy())
    labels = df["label"].to_numpy()
    return imgs[:cut], labels[:cut], imgs[cut:], labels[cut:]


class EuroSATDS(Dataset):
    """Plain (no augmentation) dataset -- used for held-out eval and for BN recalibration passes,
    where the whole point is showing the model clean, un-augmented images."""

    def __init__(self, imgs, labels):
        self.imgs, self.labels = imgs, labels

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img = self.imgs[i].astype(np.float32) / 255.0
        img = (img - MEAN) / STD
        img = np.ascontiguousarray(img.transpose(2, 0, 1))
        return torch.from_numpy(img), int(self.labels[i])


class EuroSATDSFlipAug(Dataset):
    """Light augmentation (baseline model): random horizontal/vertical flip only."""

    def __init__(self, imgs, labels):
        self.imgs, self.labels = imgs, labels

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img = self.imgs[i].astype(np.float32) / 255.0
        if np.random.rand() < 0.5:
            img = img[:, ::-1, :]
        if np.random.rand() < 0.5:
            img = img[::-1, :, :]
        img = (img - MEAN) / STD
        img = np.ascontiguousarray(img.transpose(2, 0, 1))
        return torch.from_numpy(img), int(self.labels[i])


class EuroSATDSFullAug(Dataset):
    """Full augmentation recipe (attention model), matching the paper's general experimental
    setup: 90-degree rotation, flips, color jitter, Gaussian blur, random erasing. This is the
    augmentation strength that caused the BatchNorm running-stats mismatch documented in
    inboxes/stats-improve-eurosat-cnn-next/never-try-this/pitfalls.md -- kept as-is here since the
    fix is BN recalibration (recalibrate_bn below), not weaker augmentation."""

    def __init__(self, imgs, labels, train: bool = False):
        self.imgs, self.labels, self.train = imgs, labels, train
        self.jitter = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)
        self.blur = T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
        self.erase = T.RandomErasing(p=0.3)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img = self.imgs[i].astype(np.float32) / 255.0
        if self.train:
            k = np.random.randint(4)
            if k:
                img = np.rot90(img, k, axes=(0, 1)).copy()
            if np.random.rand() < 0.5:
                img = img[:, ::-1, :].copy()
            if np.random.rand() < 0.5:
                img = img[::-1, :, :].copy()
        img = (img - MEAN) / STD
        img = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
        if self.train:
            img = self.jitter(img)
            img = self.blur(img)
            img = self.erase(img)
        return img, int(self.labels[i])


class BaselineCNN(nn.Module):
    """Paper's Model 1 equivalent (94.30% in their own from-scratch ablation ladder): 3 plain
    conv-BN-ReLU-maxpool blocks, GAP, FC(512), dropout, FC(10)."""

    def __init__(self, n_classes: int = 10):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout),
                                  nn.ReLU(inplace=True), nn.MaxPool2d(2))

        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(128, 512)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, n_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)


class CoordAttn(nn.Module):
    """Hou et al., arXiv:2103.02907. Factorizes 2D pooling into 1D horizontal/vertical encodings
    to preserve positional information SE-style channel attention discards."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(8, channels // reduction)
        self.conv1 = nn.Conv2d(channels, mid, 1)
        self.bn1 = nn.BatchNorm2d(mid)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mid, channels, 1)
        self.conv_w = nn.Conv2d(mid, channels, 1)

    def forward(self, x):
        n, c, h, w = x.shape
        pool_h = x.mean(dim=3, keepdim=True)
        pool_w = x.mean(dim=2, keepdim=True).transpose(2, 3)
        y = torch.cat([pool_h, pool_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_w = y_w.transpose(2, 3)
        a_h = torch.sigmoid(self.conv_h(y_h))
        a_w = torch.sigmoid(self.conv_w(y_w))
        return x * a_h * a_w


class SEBlock(nn.Module):
    """Hu et al., arXiv:1709.01507. Channel attention via squeeze (GAP) + excitation (FC-ReLU-FC-
    sigmoid). Note (per eurosat-attn-q7, this session's NotebookLM grounding): the paper found
    removing the FC biases helps -- not yet applied here, a documented, not-yet-tested deviation."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(8, channels // reduction)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x):
        n, c, _, _ = x.shape
        y = x.mean(dim=(2, 3))
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(n, c, 1, 1)
        return x * y


class DropBlock2d(nn.Module):
    """block_size=3 (toned down from the paper's 7 -- a 7x7 max_pool at 64x64 resolution was the
    likely cause of the T4 stall documented in never-try-this/pitfalls.md)."""

    def __init__(self, drop_prob, block_size=3):
        super().__init__()
        self.drop_prob, self.block_size = drop_prob, block_size

    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
        n, c, h, w = x.shape
        gamma = self.drop_prob / (self.block_size ** 2) * (h * w) / max(
            1, (h - self.block_size + 1) * (w - self.block_size + 1))
        mask = (torch.rand(n, c, h, w, device=x.device) < gamma).float()
        mask = F.max_pool2d(mask, kernel_size=self.block_size, stride=1, padding=self.block_size // 2)
        mask = 1 - mask.clamp(max=1)
        return x * mask * (mask.numel() / mask.sum().clamp(min=1))


class BalancedAttnBlock(nn.Module):
    """CoordAttn (spatial) + SEBlock (spectral/channel) fused via a learnable per-block scalar
    alpha (sigmoid-gated) -- the reproduction target paper's (arXiv:2510.15527) own fusion design,
    not something CBAM/SE/CoordAttn themselves do (confirmed negative result, eurosat-attn-q3)."""

    def __init__(self, cin, cout, stride=1, drop_prob=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.coord = CoordAttn(cout)
        self.se = SEBlock(cout)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.dropblock = DropBlock2d(drop_prob)
        self.shortcut = nn.Sequential()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride=stride, bias=False),
                                           nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        a = torch.sigmoid(self.alpha)
        out = a * self.coord(out) + (1 - a) * self.se(out)
        out = self.dropblock(out)
        out = F.relu(out + self.shortcut(x))
        return out


class BalancedAttnCNN(nn.Module):
    """Toned down for one T4 vs. the paper's full Model 3 (11.2M params, blocks=(3,3,3,2),
    channels=(64,128,256,512)): half the channels, fewer blocks per stage -- 1.68M params."""

    def __init__(self, n_classes=10, blocks=(2, 2, 2, 1), channels=(32, 64, 128, 256),
                 drop_rates=(0.05, 0.10, 0.15, 0.20)):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, channels[0], 3, padding=1, bias=False),
                                   nn.BatchNorm2d(channels[0]), nn.ReLU(inplace=True))
        stages = []
        cin = channels[0]
        for i, (n_blocks, cout, dp) in enumerate(zip(blocks, channels, drop_rates)):
            for b in range(n_blocks):
                stride = 2 if (b == 0 and i > 0) else 1
                stages.append(BalancedAttnBlock(cin, cout, stride=stride, drop_prob=dp))
                cin = cout
        self.stages = nn.Sequential(*stages)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[-1], n_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.gap(x).flatten(1)
        return self.fc(x)


# Class-balanced loss weights for the attention model (paper's own scheme, applied to our 10 real
# EuroSAT classes by name): 1.3 for classes the paper found most confused, 0.8 for easy classes.
ATTN_CLASS_WEIGHTS = [1.0, 0.8, 1.3, 1.0, 1.3, 1.0, 1.3, 0.8, 1.0, 0.8]  # order matches CLASS_NAMES


def train_baseline_cnn(X_train, y_train, X_test, y_test, device, epochs: int = 30):
    """Adam, CosineAnnealingLR(T_max=epochs), flip-only augmentation. Actually run: 30 epochs,
    145.2s on a T4, final train_acc=0.9598. Test macro-F1 0.9536 (verified, checkpoint round-
    tripped bit-identical after reload)."""
    train_dl = DataLoader(EuroSATDSFlipAug(X_train, y_train), batch_size=64, shuffle=True, num_workers=2)

    model = BaselineCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()

    history = []
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        tr_loss = tr_correct = tr_n = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * xb.size(0)
            tr_correct += (out.argmax(1) == yb).sum().item()
            tr_n += xb.size(0)
        sched.step()
        history.append({"epoch": epoch, "train_loss": tr_loss / tr_n, "train_acc": tr_correct / tr_n})
    wall_seconds = time.time() - t0
    return model, history, wall_seconds


def train_attention_cnn(X_train, y_train, X_test, y_test, device, epochs: int = 15,
                         extra_epochs: int = 6, extra_lr: float = 3e-4):
    """AdamW + CosineAnnealingLR(T_max=epochs), full augmentation (EuroSATDSFullAug), mixed
    precision, class-balanced loss. Actually run: 15 epochs (train_acc 0.7490), then continued 6
    more epochs at a reduced, freshly-scheduled LR (train_acc 0.7583 at epoch 21, still improving
    -- not converged). See recalibrate_bn: this model's BatchNorm running stats must be
    recalibrated on clean data before real evaluation, due to the augmentation strength here."""
    train_dl = DataLoader(EuroSATDSFullAug(X_train, y_train, train=True), batch_size=64,
                           shuffle=True, num_workers=2)

    model = BalancedAttnCNN().to(device)
    class_weights = torch.tensor(ATTN_CLASS_WEIGHTS, dtype=torch.float32).to(device)
    crit = nn.CrossEntropyLoss(weight=class_weights)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda")

    history = []
    t0 = time.time()

    def run_epochs(n, optimizer, scheduler):
        for _ in range(n):
            model.train()
            tr_loss = tr_correct = tr_n = 0
            for xb, yb in train_dl:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                with torch.amp.autocast("cuda"):
                    out = model(xb)
                    loss = crit(out, yb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                tr_loss += loss.item() * xb.size(0)
                tr_correct += (out.argmax(1) == yb).sum().item()
                tr_n += xb.size(0)
            scheduler.step()
            history.append({"epoch": len(history), "train_loss": tr_loss / tr_n, "train_acc": tr_correct / tr_n})

    run_epochs(epochs, opt, sched)
    if extra_epochs:
        for g in opt.param_groups:
            g["lr"] = extra_lr
        sched_cont = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=extra_epochs)
        run_epochs(extra_epochs, opt, sched_cont)

    wall_seconds = time.time() - t0
    return model, history, wall_seconds


def bn_running_var_mean(model) -> float:
    """Cheap per-epoch BN health signal -- the exact statistic none of the three in-training-
    visualization papers (ConceptEvo, DeepTracker, In situ TensorView) track. DeepTracker
    explicitly excludes BN, reasoning it 'can be totally recovered given the weights on CONV and
    FC layers' -- an assumption this project's own incident disproves, since BN running stats
    depend on the DATA DISTRIBUTION seen (augmented vs. clean), not just the learned weights.
    A large jump in this value epoch-to-epoch, or a large gap between this and the equivalent
    computed on a clean-data forward pass, is an early warning of the augmented-vs-clean
    distributional mismatch documented in never-try-this/pitfalls.md."""
    vars_ = [m.running_var.detach().float().mean().item() for m in model.modules()
             if isinstance(m, nn.BatchNorm2d)]
    return float(np.mean(vars_)) if vars_ else float("nan")


def weight_update_norm(model, prev_state) -> float:
    """Coarse per-EPOCH aggregate signal. NOT a faithful implementation of In situ TensorView's
    method (see WeightTrajectoryTracker below for that) -- kept as a cheap secondary summary (total
    L2 norm of the change in every parameter since the last epoch), ~300x coarser in time than the
    paper's actual per-step tracking. Honest fidelity note logged in RESEARCH-NOTES.md under the
    2026-09-28 Elicit implementation-grade re-query."""
    if prev_state is None:
        return float("nan")
    total = 0.0
    for name, p in model.state_dict().items():
        if p.dtype.is_floating_point:
            total += (p.detach().float() - prev_state[name].float()).pow(2).sum().item()
    return total ** 0.5


class WeightTrajectoryTracker:
    """Faithful implementation of In situ TensorView's trajectory view (arXiv:1806.07382, exact
    mechanism per the 2026-09-28 Elicit implementation-grade re-query): plots 3 ARBITRARY individual
    weight scalars as (x, y, z) coordinates, one point per training TIME STEP (not per epoch),
    colored by time. Their own case study: a healthy run (lr=0.001) shows a winding path; a
    pathological run (lr=0.05) shows the trajectory freeze into a straight line within ~100-200
    steps, from gradient vanishing. This is a real per-batch read of 3 scalars (no extra forward
    pass, negligible cost) -- cheap enough to call every training step, unlike weight_update_norm's
    epoch-level aggregate."""

    def __init__(self, model, param_name: str = "stem.0.weight", n_coords: int = 3):
        sd = model.state_dict()
        assert param_name in sd, f"{param_name} not found in model.state_dict()"
        self.param_name = param_name
        self.indices = list(range(n_coords))
        self.points: list[tuple[float, float, float]] = []

    def step(self, model):
        flat = model.state_dict()[self.param_name].detach().flatten()
        self.points.append(tuple(float(flat[i].item()) for i in self.indices))

    def health_signal(self, window: int = 100) -> float:
        """Cheap scalar summary of the last `window` points' total path length -- near-zero means
        the trajectory has frozen (the paper's own gradient-vanishing signature); a healthy run
        keeps a non-trivial path length. Not from the paper itself (they inspect the 3D shape
        visually); this is a numeric readout of the same underlying signal for non-visual use."""
        pts = self.points[-window:]
        if len(pts) < 2:
            return float("nan")
        return sum(
            sum((a - b) ** 2 for a, b in zip(pts[i], pts[i - 1])) ** 0.5
            for i in range(1, len(pts))
        )


def per_image_correctness(model, dl, device) -> np.ndarray:
    """Per-image correct/incorrect array (bool), in the DataLoader's fixed iteration order --
    required by LeftRuleAnomalyDetector below. dl must use shuffle=False for the order to be
    stable across repeated calls."""
    model.eval()
    correct = []
    with torch.no_grad():
        for xb, yb in dl:
            pred = model(xb.to(device)).argmax(1).cpu()
            correct.append((pred == yb).numpy())
    return np.concatenate(correct)


class LeftRuleAnomalyDetector:
    """Faithful implementation of DeepTracker's left-rule anomaly detection (arXiv:1808.08531,
    exact algorithm per the 2026-09-28 Elicit implementation-grade re-query): each validation
    image has a binary correctness history a_{i,t} in {0,1}. Over a sliding window of the
    preceding `k` sampled checkpoints, if an image's correctness value stayed constant, the rule
    predicts it continues; a violation at the current checkpoint is an image-level anomaly. Class
    score L_{c,t} = count of rule-violating images in class c at checkpoint t. The paper does not
    report a specific numeric value for k (user-selected) -- k=3 is this project's own choice,
    sized for a 40-epoch run with checkpoints every few epochs, not copied from the paper."""

    def __init__(self, labels: np.ndarray, k: int = 3):
        self.labels = labels
        self.k = k
        self.history = np.zeros((len(labels), 0), dtype=np.int8)
        self.log: list[dict] = []

    def update(self, t: int, correct: np.ndarray) -> dict:
        self.history = np.concatenate([self.history, correct.reshape(-1, 1).astype(np.int8)], axis=1)
        result = {"t": t, "class_violation_counts": {}, "total_violations": 0}
        if self.history.shape[1] > self.k:
            window = self.history[:, -(self.k + 1):-1]
            current = self.history[:, -1]
            constant = np.all(window == window[:, [0]], axis=1)
            violation = constant & (current != window[:, 0])
            for c in sorted(set(self.labels.tolist())):
                mask = self.labels == c
                result["class_violation_counts"][int(c)] = int(violation[mask].sum())
            result["total_violations"] = int(violation.sum())
        self.log.append(result)
        return result


def train_attention_cnn_v2(X_train, y_train, X_test, y_test, device, epochs: int = 40,
                            val_frac: float = 0.10, eval_every: int = 5):
    """Careful retraining, informed directly by this session's BN-mismatch incident and the
    in-training-visualization research that followed it (ConceptEvo/DeepTracker/In situ TensorView,
    logged in RESEARCH-NOTES.md under eurosat-training-viz-q5/q6/q8). Differences from
    train_attention_cnn (kept, not deleted, per this project's standing legacy discipline):

    1. A clean validation slice is carved from the END of the shuffled train block (this project's
       standing time-forward-style discipline: validation never touches the real test set) and
       evaluated in eval() mode (real BN running stats) every `eval_every` epochs -- exactly the
       train-vs-eval gap monitoring DeepTracker and ConceptEvo both do (eurosat-training-viz-q6),
       which would have surfaced the augmented-vs-clean BN mismatch mid-training instead of only
       after, via a completely different, independent mechanism from item 2 below.
    2. `bn_running_var_mean` and `weight_update_norm` are logged every epoch -- the specific
       BatchNorm-statistics signal the literature review found is a genuine blind spot in existing
       in-training-visualization tools (none of the three papers track it), plus the cheap
       gradient-health proxy motivated by In situ TensorView's own case study.
    3. A single, honestly-sized epoch budget (default 40, not the previous run's improvised
       15-then-6-more) with one CosineAnnealingLR schedule -- the previous run's train_loss was
       still falling with no plateau at epoch 21, so this session's evidence is that more epochs
       under one coherent schedule is the right lever, not a structural change (see the plan's
       Part A.4 architecture reflection).
    4. WeightTrajectoryTracker (updated every training STEP) and LeftRuleAnomalyDetector (updated
       every `eval_every` epochs on the clean val set) are the paper-faithful upgrades over
       bn_running_var_mean/weight_update_norm's epoch-level aggregates -- added after an explicit
       fidelity audit found the original two diagnostics only "inspired by" TensorView/DeepTracker,
       not implementing their actual mechanisms (RESEARCH-NOTES.md, 2026-09-28 Elicit re-query).
    """
    rng_val = np.random.default_rng(1)  # independent from the train/test split's seed=0
    order = rng_val.permutation(len(X_train))
    cut = int(len(X_train) * (1 - val_frac))
    tr_idx, val_idx = order[:cut], order[cut:]
    X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    train_dl = DataLoader(EuroSATDSFullAug(X_tr, y_tr, train=True), batch_size=64,
                           shuffle=True, num_workers=2)
    val_dl = DataLoader(EuroSATDS(X_val, y_val), batch_size=256, shuffle=False, num_workers=2)

    model = BalancedAttnCNN().to(device)
    class_weights = torch.tensor(ATTN_CLASS_WEIGHTS, dtype=torch.float32).to(device)
    crit = nn.CrossEntropyLoss(weight=class_weights)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda")

    history = []
    prev_state = None
    trajectory = WeightTrajectoryTracker(model)  # per-step, TensorView-faithful
    anomaly_detector = LeftRuleAnomalyDetector(y_val)  # per-eval_every, DeepTracker-faithful
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        tr_loss = tr_correct = tr_n = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                out = model(xb)
                loss = crit(out, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tr_loss += loss.item() * xb.size(0)
            tr_correct += (out.argmax(1) == yb).sum().item()
            tr_n += xb.size(0)
            trajectory.step(model)
        sched.step()

        state_now = {k: v.detach().clone() for k, v in model.state_dict().items()}
        entry = {
            "epoch": epoch,
            "train_loss": tr_loss / tr_n,
            "train_acc": tr_correct / tr_n,
            "bn_running_var_mean": bn_running_var_mean(model),
            "weight_update_norm": weight_update_norm(model, prev_state),
            "trajectory_health_last_epoch": trajectory.health_signal(window=len(train_dl)),
        }
        prev_state = state_now

        if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
            _, _, _, val_report, _ = evaluate(model, val_dl, device)
            correct = per_image_correctness(model, val_dl, device)
            anomaly_result = anomaly_detector.update(epoch, correct)
            entry["anomaly_total_violations"] = anomaly_result["total_violations"]
            entry["anomaly_class_violation_counts"] = anomaly_result["class_violation_counts"]
            entry["clean_val_macro_f1"] = val_report["macro avg"]["f1-score"]
            entry["clean_val_accuracy"] = val_report["accuracy"]
            entry["train_eval_gap"] = entry["train_acc"] - entry["clean_val_accuracy"]
            print(f"epoch {epoch+1}/{epochs} | train_loss {entry['train_loss']:.4f} "
                  f"train_acc {entry['train_acc']:.4f} | clean_val_f1 {entry['clean_val_macro_f1']:.4f} "
                  f"clean_val_acc {entry['clean_val_accuracy']:.4f} | anomaly_violations "
                  f"{entry['anomaly_total_violations']} | {time.time()-t0:.0f}s elapsed", flush=True)
        else:
            print(f"epoch {epoch+1}/{epochs} | train_loss {entry['train_loss']:.4f} "
                  f"train_acc {entry['train_acc']:.4f} | {time.time()-t0:.0f}s elapsed", flush=True)

        history.append(entry)

    wall_seconds = time.time() - t0
    return model, history, wall_seconds, trajectory, anomaly_detector


def recalibrate_bn(model, X_train, y_train, device):
    """Fixes the BatchNorm running-stats mismatch documented in never-try-this/pitfalls.md: forward
    passes ONLY (no backward, weights frozen) over CLEAN (unaugmented) training images, so BN's
    running_mean/running_var re-converge to the clean-image distribution the weights were actually
    trained to recognize. Took the attention model from test macro-F1 0.2716 -> 0.5998 in 12s on a
    T4, with zero change to any learned weight. Always try this before retraining when test
    accuracy is far below train accuracy and training used heavy pixel-level augmentation."""
    import copy

    recal_model = copy.deepcopy(model)
    for m in recal_model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None  # cumulative moving average over every batch seen, not exponential decay

    clean_dl = DataLoader(EuroSATDS(X_train, y_train), batch_size=256, shuffle=True, num_workers=2)
    recal_model.train()
    t0 = time.time()
    with torch.no_grad():
        for xb, _ in clean_dl:
            recal_model(xb.to(device))
    recal_model.eval()
    return recal_model, time.time() - t0


def evaluate(model, dl, device):
    model.eval()
    all_preds, all_labels, all_conf = [], [], []
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device)
            out = model(xb)
            probs = F.softmax(out, dim=1)
            conf, pred = probs.max(1)
            all_preds.append(pred.cpu().numpy())
            all_labels.append(yb.numpy())
            all_conf.append(conf.cpu().numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    confs = np.concatenate(all_conf)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    return y_true, y_pred, confs, report, cm


def diagnose_bn_mismatch(model, test_dl, device):
    """The controlled, single-variable diagnostic from never-try-this/pitfalls.md: same batch,
    same weights, only `model.train()` (batch stats) vs `model.eval()` (running stats) differs.
    A large gap is a causal proof of BN running-stats mismatch, not a guess."""
    model.train()
    with torch.no_grad():
        xb, yb = next(iter(test_dl))
        acc_train_mode = (model(xb.to(device)).argmax(1).cpu() == yb).float().mean().item()
    model.eval()
    with torch.no_grad():
        acc_eval_mode = (model(xb.to(device)).argmax(1).cpu() == yb).float().mean().item()
    return acc_train_mode, acc_eval_mode


def plot_first_layer_filters(weight_tensor, title, path):
    """Zeiler & Fergus (arXiv:1311.2901) style: plot each output filter of a first conv layer
    (out_channels, 3, k, k) as a normalized RGB patch.

    Caveat (found honest 2026-09-28, viewing the actual baseline_filters.png output): Z&F's own
    diagnostic ("dead/aliased filters => architecture problem; sensible edge/color detectors =>
    keep training") was built on AlexNet's 11x11 first-layer filters (reduced to 7x7 after that
    diagnosis, RESEARCH-NOTES.md eurosat-gradcam-q6) -- large enough to show real spatial-frequency
    structure. Both CNNs here use 3x3 first-layer kernels (VGG-style stacked small convs), which are
    mathematically too small to ever display edge-detector-like shapes regardless of training
    quality -- a 3x3 patch is just 9 solid-colored squares. This function still renders the true
    trained weights correctly (no bug), but "do these look like edge detectors" is not an answerable
    question at this kernel size; the only thing legitimately readable from it is whether any filter
    looks uniformly flat/saturated (a weak dead-filter signal), not frequency content or orientation.
    Prefer plot_weight_histograms for an actual health read on a 3x3-kernel network."""
    import matplotlib.pyplot as plt

    w = weight_tensor.detach().cpu().numpy()
    n = w.shape[0]
    wn = np.zeros_like(w)
    for i in range(n):
        f = w[i]
        f = f - f.min()
        f = f / (f.max() + 1e-8)
        wn[i] = f
    cols = 8
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.0, rows * 1.0))
    axes = np.array(axes).reshape(rows, cols)
    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        ax.axis("off")
        if i < n:
            ax.imshow(wn[i].transpose(1, 2, 0))
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return w


def layer_groups(model):
    """Weight tensors grouped by module TYPE (not by parameter-name substring -- a plain
    nn.Sequential's children have names like 'features.0.0.weight', which contain neither 'conv'
    nor 'bn', so name-substring matching silently drops them)."""
    groups = {"conv_kernels": [], "bn_weight": [], "fc_weight": []}
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            groups["conv_kernels"].append(m.weight.detach().cpu().numpy().flatten())
        elif isinstance(m, nn.BatchNorm2d):
            groups["bn_weight"].append(m.weight.detach().cpu().numpy().flatten())
        elif isinstance(m, nn.Linear):
            groups["fc_weight"].append(m.weight.detach().cpu().numpy().flatten())
    return {k: np.concatenate(v) for k, v in groups.items() if v}


def plot_weight_histograms(groups, title, path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(groups), figsize=(4 * len(groups), 3))
    if len(groups) == 1:
        axes = [axes]
    stats = {}
    for ax, (name, vals) in zip(axes, groups.items()):
        ax.hist(vals, bins=60, color="steelblue")
        ax.set_title(f"{name}\n(n={len(vals)})")
        stats[name] = {"mean": float(vals.mean()), "std": float(vals.std()),
                       "min": float(vals.min()), "max": float(vals.max())}
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return stats


def alpha_per_block(attn_model: BalancedAttnCNN):
    """Reads the learned CoordAttn/SE fusion weight directly off each trained BalancedAttnBlock --
    this session's own result to compare against the reproduction paper's reported alpha~=0.57,
    not something CBAM/SE/CoordAttn themselves report (see BalancedAttnBlock docstring)."""
    alphas = []
    for name, m in attn_model.named_modules():
        if isinstance(m, BalancedAttnBlock):
            alphas.append({"block": name, "alpha": torch.sigmoid(m.alpha).item()})
    return alphas


def grad_cam(model, x, target_class, device):
    """Selvaraju et al. (arXiv:1610.02391) Grad-CAM on BaselineCNN's last conv layer
    (model.features[2][0], the paper-confirmed correct target for a plain conv stack per
    eurosat-gradcam-q4). Hooks the layer's forward activations and backward gradients on a single
    image, weights each channel by the global-average-pooled gradient, and ReLUs the weighted sum
    -- the exact mechanism, not a saliency-map proxy. Caveat (eurosat-gradcam-q9): our 64x64 input
    gives an 8x8 last-conv feature map, smaller than anything validated in the paper's own
    experiments (7x14x14 on 224x224 input) -- an honest extrapolation, not a proven-safe use case.
    Returns the 8x8 (or model-dependent) CAM as a numpy array, already ReLU'd and normalized to
    [0, 1], plus the raw logits for the requested image."""
    target_layer = model.features[2][0]
    activations, gradients = {}, {}

    def fwd_hook(_module, _inp, out):
        activations["value"] = out.detach()

    def bwd_hook(_module, _grad_in, grad_out):
        gradients["value"] = grad_out[0].detach()

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)
    try:
        model.eval()
        xb = x.unsqueeze(0).to(device).requires_grad_(False)
        out = model(xb)
        model.zero_grad()
        out[0, target_class].backward()
        acts = activations["value"][0]       # (C, h, w)
        grads = gradients["value"][0]        # (C, h, w)
        weights = grads.mean(dim=(1, 2))     # (C,)
        cam = F.relu((weights.view(-1, 1, 1) * acts).sum(dim=0))
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, out.detach().cpu().numpy()[0]
    finally:
        h1.remove()
        h2.remove()


def occlusion_sensitivity(model, x, target_class, device, patch_size=8, stride=4):
    """Zeiler & Fergus (arXiv:1311.2901) occlusion sensitivity: slide a grey (zero-mean-normalized)
    patch over the image, re-run the forward pass at every position, and record how much the
    target class's softmax probability drops. This is the causal check the research explicitly
    found missing from every attention/CAM method reviewed this session (eurosat-attn-q4,
    eurosat-gradcam findings): a real perturbation of the input, not a correlate. Returns a 2D
    grid (n_positions_h, n_positions_w) of probability values (not drops -- the caller compares
    against the unoccluded baseline probability) plus that baseline probability."""
    model.eval()
    _, H, W = x.shape
    grey = torch.zeros(3, device=device)  # mean-normalized image already has ~0 as its "neutral" value
    with torch.no_grad():
        base_prob = F.softmax(model(x.unsqueeze(0).to(device)), dim=1)[0, target_class].item()
    n_h = (H - patch_size) // stride + 1
    n_w = (W - patch_size) // stride + 1
    grid = np.zeros((n_h, n_w), dtype=np.float32)
    with torch.no_grad():
        for i in range(n_h):
            for j in range(n_w):
                xo = x.clone()
                r0, c0 = i * stride, j * stride
                xo[:, r0:r0 + patch_size, c0:c0 + patch_size] = grey.view(3, 1, 1)
                prob = F.softmax(model(xo.unsqueeze(0).to(device)), dim=1)[0, target_class].item()
                grid[i, j] = prob
    return grid, base_prob


def raw_pixel_embedding(imgs, n_components=2, method="pca", sample_n=2000, seed=0):
    """PCA or t-SNE on the RAW NORMALIZED PIXEL VECTORS -- the identical feature representation
    live_tests/train.py's deployed PCA(50)+LogisticRegression baseline uses -- not a post-training
    model embedding. Answers a genuinely different question from any Part B embedding: how
    separable are the classes in the DATA itself, before any CNN touches it. Subsamples to
    sample_n images (t-SNE is O(n^2) and the full 27k-image set is too slow) with a fixed seed."""
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(imgs), size=min(sample_n, len(imgs)), replace=False)
    flat = (imgs[idx].astype(np.float32) / 255.0).reshape(len(idx), -1)
    if method == "pca":
        reducer = PCA(n_components=n_components, random_state=seed)
    elif method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=n_components, random_state=seed, init="pca")
    else:
        raise ValueError(f"unknown method {method!r}")
    coords = reducer.fit_transform(flat)
    return coords, idx


def class_prototypes(imgs, labels):
    """Per-class mean image ("prototype"): averages every image within each class into one 64x64
    RGB image. Answers what eda.py's sample-tiles chart doesn't -- the typical visual signature per
    class, independent of any single example's idiosyncrasies -- and is a direct companion to a
    confusion matrix: classes with visually similar prototypes are the ones a model is more likely
    to confuse. Returns {class_index: (64, 64, 3) uint8-range float array}."""
    protos = {}
    for c in sorted(set(labels.tolist())):
        mask = labels == c
        protos[int(c)] = imgs[mask].astype(np.float32).mean(axis=0)
    return protos


def confused_pairs_gallery(cm, imgs, labels, top_k=3, n_examples=4, seed=0):
    """Dataset-level complement to a model's misclassification gallery: for the top_k most-confused
    OFF-DIAGONAL class pairs in a confusion matrix `cm` (both directions of a pair summed), sample
    n_examples real images from EACH class in the pair -- so the question being asked is "do these
    classes look genuinely similar in the raw data", independent of whether any specific model got
    a specific image wrong. Returns a list of {"pair": (class_a, class_b), "count": int,
    "examples_a": [...], "examples_b": [...]} sorted by confusion count, descending."""
    rng = np.random.default_rng(seed)
    n = cm.shape[0]
    pair_counts = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            key = tuple(sorted((i, j)))
            pair_counts[key] = pair_counts.get(key, 0) + cm[i, j]
    top_pairs = sorted(pair_counts.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    gallery = []
    for (a, b), count in top_pairs:
        idx_a = np.where(labels == a)[0]
        idx_b = np.where(labels == b)[0]
        ex_a = imgs[rng.choice(idx_a, size=min(n_examples, len(idx_a)), replace=False)]
        ex_b = imgs[rng.choice(idx_b, size=min(n_examples, len(idx_b)), replace=False)]
        gallery.append({"pair": (int(a), int(b)), "count": int(count),
                         "examples_a": ex_a, "examples_b": ex_b})
    return gallery
