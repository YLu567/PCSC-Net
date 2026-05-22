# train_juice_fusion.py
# -*- coding: utf-8 -*-

import os
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ==================== 自动解析列名，确定分量和频率 ====================
def parse_columns_and_build_matrix(df: pd.DataFrame, label_col: str = "kind"):
    """
    自动解析列名，格式：分量_频率，例如 Cp_100, Cs_150, X_2000
    返回：
        X: float32 [N, n_comp, n_freq]
        y: int64   [N]
        colmap: dict[(comp,freq)] -> column_name
        comps: list of component names
        freqs: list of frequencies
    """
    import re
    pat = re.compile(r"^([A-Za-z]+)_(\d+)$")
    colmap = {}
    for col in df.columns:
        if col == label_col:
            continue
        m = pat.match(str(col))
        if m:
            comp = m.group(1)
            freq = int(m.group(2))
            colmap[(comp, freq)] = col

    if not colmap:
        raise ValueError("没有找到符合 '分量_数字' 模式的列，请检查列名格式")

    comps = sorted(set(comp for comp, _ in colmap.keys()))
    freqs = sorted(set(freq for _, freq in colmap.keys()))

    missing = []
    for comp in comps:
        for freq in freqs:
            if (comp, freq) not in colmap:
                missing.append((comp, freq))
    if missing:
        raise ValueError(f"缺失以下组合的列：{missing}")

    N = len(df)
    X = np.zeros((N, len(comps), len(freqs)), dtype=np.float32)
    for i, comp in enumerate(comps):
        for j, freq in enumerate(freqs):
            col_name = colmap[(comp, freq)]
            X[:, i, j] = df[col_name].to_numpy(np.float32)

    y = df[label_col].to_numpy(np.int64)
    return X, y, colmap, comps, freqs


class BeefDataset(Dataset):
    def __init__(self, X, y, raw_X_for_save=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.raw = None
        if raw_X_for_save is not None:
            self.raw = torch.tensor(raw_X_for_save, dtype=torch.float32)

    def __len__(self):
        return self.y.numel()

    def __getitem__(self, idx):
        if self.raw is None:
            return self.X[idx], self.y[idx]
        return self.X[idx], self.y[idx], self.raw[idx]


# ----------------------------
# 每分量1D小卷积（groups=分量数）
# ----------------------------
class PerComponentConv(nn.Module):
    """
    输入:  [B, n_comp, n_freq]
    输出:  [B, n_comp, n_freq]
    每个分量独立做1D卷积（groups=n_comp）
    """
    def __init__(self, n_comp, hidden_per_comp=8, k=3):
        super().__init__()
        self.conv_expand = nn.Conv1d(
            in_channels=n_comp,
            out_channels=n_comp * hidden_per_comp,
            kernel_size=k,
            padding=k // 2,
            groups=n_comp,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(n_comp * hidden_per_comp)

        self.conv_reduce = nn.Conv1d(
            in_channels=n_comp * hidden_per_comp,
            out_channels=n_comp,
            kernel_size=1,
            groups=n_comp,
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(n_comp)

    def forward(self, x):
        x = self.conv_expand(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.conv_reduce(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        return x


# ----------------------------
# 框架图同款调色函数
# ----------------------------
def palette_map_torch(t: torch.Tensor) -> torch.Tensor:
    """
    t: [B,1,H,W] in [0,1]
    return: [B,3,H,W]
    深海军蓝 -> 青蓝 -> 浅钢蓝 -> 米金 -> 棕金
    """
    anchors = torch.tensor([
        [31, 72, 103],    # #1F4867 深海军蓝
        [35, 124, 133],   # #237C85 青蓝
        [123, 172, 196],  # #7BACC4 浅钢蓝
        [221, 200, 156],  # #DDC89C 米金
        [188, 137, 89],   # #BC8959 棕金
    ], dtype=t.dtype, device=t.device) / 255.0

    n_seg = anchors.shape[0] - 1
    x = torch.clamp(t, 0.0, 1.0) * n_seg
    idx0 = torch.floor(x).long().clamp(min=0, max=n_seg - 1)
    idx1 = (idx0 + 1).clamp(max=n_seg)
    w = x - idx0.float()

    c0 = anchors[idx0.squeeze(1)]
    c1 = anchors[idx1.squeeze(1)]

    rgb = c0 * (1.0 - w.squeeze(1).unsqueeze(-1)) + c1 * w.squeeze(1).unsqueeze(-1)
    rgb = rgb.permute(0, 3, 1, 2).contiguous()
    return rgb


def palette_map_np(t: np.ndarray) -> np.ndarray:
    """
    t: [H,W] in [0,1]
    return: [H,W,3]
    """
    anchors = np.array([
        [31, 72, 103],    # #1F4867
        [35, 124, 133],   # #237C85
        [123, 172, 196],  # #7BACC4
        [221, 200, 156],  # #DDC89C
        [188, 137, 89],   # #BC8959
    ], dtype=np.float32) / 255.0

    n_seg = len(anchors) - 1
    x = np.clip(t, 0.0, 1.0) * n_seg
    idx0 = np.floor(x).astype(np.int64)
    idx0 = np.clip(idx0, 0, n_seg - 1)
    idx1 = np.clip(idx0 + 1, 0, n_seg)

    w = (x - idx0)[..., None]
    c0 = anchors[idx0]
    c1 = anchors[idx1]
    rgb = c0 * (1.0 - w) + c1 * w
    return np.clip(rgb, 0.0, 1.0)


# ----------------------------
# 伪图生成器：频率×分量高度图 -> 框架图色系伪图
# accuracy优先：refine 直接输出最终伪图
# ----------------------------
class PseudoImageGenerator(nn.Module):
    """
    输入:  height map [B,1, n_freq, n_comp]
    输出:  pseudo image [B,3,S,S]
    """
    def __init__(self, out_size=64, learnable_refine=True):
        super().__init__()
        self.out_size = out_size
        self.learnable_refine = learnable_refine

        if learnable_refine:
            self.refine = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 3, 3, padding=1, bias=True),
                nn.Sigmoid()
            )
        else:
            self.refine = None

    def forward(self, h):
        h_min = h.amin(dim=(2, 3), keepdim=True)
        h_max = h.amax(dim=(2, 3), keepdim=True)
        h_norm = (h - h_min) / (h_max - h_min + 1e-6)

        dzdy = h[:, :, 2:, :] - h[:, :, :-2, :]
        dzdx = h[:, :, :, 2:] - h[:, :, :, :-2]
        dzdy = F.pad(dzdy, (0, 0, 1, 1), mode='replicate')
        dzdx = F.pad(dzdx, (1, 1, 0, 0), mode='replicate')

        slope = torch.sqrt(dzdx.pow(2) + dzdy.pow(2) + 1e-8)
        slope_max = slope.amax(dim=(2, 3), keepdim=True)
        slope_norm = slope / (slope_max + 1e-6)

        # 固定色系映射
        base = palette_map_torch(h_norm)

        # 轻微明暗调制
        shade = 0.82 + 0.18 * (1.0 - slope_norm)
        img = torch.clamp(base * shade, 0.0, 1.0)

        img = F.interpolate(
            img,
            size=(self.out_size, self.out_size),
            mode='bilinear',
            align_corners=False
        )

        # accuracy优先：refine 直接接管最终输出
        if self.refine is not None:
            img = self.refine(img)

        return img


# ----------------------------
# 自适应噪声模块（训练时开启）
# ----------------------------
class AdaptiveNoise(nn.Module):
    def __init__(self, channels=3, max_sigma=0.15):
        super().__init__()
        self.max_sigma = max_sigma
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(channels, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, channels),
            nn.Softplus()
        )

    def forward(self, x):
        if not self.training:
            return x
        sigma = self.mlp(x).view(x.size(0), x.size(1), 1, 1)
        sigma = torch.clamp(sigma, 0.0, self.max_sigma)
        eps = torch.randn_like(x)
        return x + sigma * eps


# ----------------------------
# 判别网络：伪图特征 + 原谱特征 融合后分类
# ----------------------------
class BeefNet(nn.Module):
    def __init__(self,
                 n_comp,
                 n_freq,
                 num_classes=3,
                 pseudo_size=64,
                 spec_feat_dim=128,
                 img_feat_dim=128,
                 fusion_dropout=0.2):
        super().__init__()
        self.n_comp = n_comp
        self.n_freq = n_freq
        self.per_comp = PerComponentConv(n_comp=n_comp, hidden_per_comp=8, k=3)
        self.pseudo = PseudoImageGenerator(out_size=pseudo_size, learnable_refine=True)
        self.noise = AdaptiveNoise(channels=3, max_sigma=0.15)

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, img_feat_dim),
            nn.ReLU(inplace=True),
        )

        self.spec_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_comp * n_freq, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=fusion_dropout),
            nn.Linear(128, spec_feat_dim),
            nn.ReLU(inplace=True),
        )

        fusion_dim = img_feat_dim + spec_feat_dim
        self.classifier = nn.Sequential(
            nn.Dropout(p=fusion_dropout),
            nn.Linear(fusion_dim, num_classes)
        )

    def forward(self, x):
        x_in = x

        x_proc = self.per_comp(x_in)
        h = x_proc.transpose(1, 2).unsqueeze(1)
        img = self.pseudo(h)
        img = self.noise(img)

        feat_img = self.backbone(img)
        feat_spec = self.spec_encoder(x_in)

        feat = torch.cat([feat_img, feat_spec], dim=1)
        logits = self.classifier(feat)

        return logits, img, h


# ==================== 可视化函数 ====================
def height_to_normalmap_np(height_np: np.ndarray):
    """
    height_np: [n_freq, n_comp]
    return: RGB in [0,1], shape [n_freq, n_comp, 3]
    """
    h = height_np.astype(np.float32)

    h_min = np.min(h)
    h_max = np.max(h)
    h_norm = (h - h_min) / (h_max - h_min + 1e-6)

    gy, gx = np.gradient(h)
    slope = np.sqrt(gx ** 2 + gy ** 2)
    slope_norm = slope / (np.max(slope) + 1e-6)

    base = palette_map_np(h_norm)
    shade = 0.82 + 0.18 * (1.0 - slope_norm)
    img = base * shade[..., None]

    img = np.clip(img, 0.0, 1.0)
    return img


def save_surface_3d(height_np: np.ndarray, save_path: Path, comps, freqs):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    h = height_np.astype(np.float32)
    x = np.arange(h.shape[1])
    y = np.arange(h.shape[0])
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, h, rstride=1, cstride=1, linewidth=0, antialiased=True)

    ax.set_xlabel("Component")
    ax.set_ylabel("Frequency index")
    ax.set_zlabel("Value")

    ax.set_xticks(np.arange(len(comps)))
    ax.set_xticklabels(list(comps))
    ax.set_yticks(np.arange(len(freqs)))
    ax.set_yticklabels([str(f) for f in freqs])

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def save_normalmap_with_axes(nm_rgb: np.ndarray, save_path: Path, comps, freqs, dpi=200):
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.imshow(nm_rgb, origin="lower", aspect="auto", interpolation="nearest")

    ax.set_xticks(np.arange(len(comps)))
    ax.set_xticklabels(list(comps))
    ax.set_yticks(np.arange(len(freqs)))
    ax.set_yticklabels([str(f) for f in freqs])

    ax.set_xlabel("Component (分量)")
    ax.set_ylabel("Frequency (Hz) (频率)")

    ax.set_xticks(np.arange(-0.5, len(comps), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(freqs), 1), minor=True)
    ax.grid(which="minor", linestyle="--", linewidth=0.5, alpha=0.4)

    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)


def plot_confusion_matrix(cm, class_names, save_path: Path, normalize=False):
    if normalize:
        cm = cm.astype(np.float32)
        row_sum = cm.sum(axis=1, keepdims=True) + 1e-12
        cm = cm / row_sum

    fig = plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=30)
    plt.yticks(tick_marks, class_names)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() * 0.6
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            plt.text(
                j, i, format(val, fmt),
                ha="center", va="center",
                color="white" if val > thresh else "black"
            )

    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_curves(train_losses, val_losses, train_accs, val_accs, save_path: Path):
    fig = plt.figure(figsize=(8, 5))
    x = np.arange(1, len(train_losses) + 1)
    plt.plot(x, train_losses, label="train_loss")
    plt.plot(x, val_losses, label="val_loss")
    plt.plot(x, train_accs, label="train_acc")
    plt.plot(x, val_accs, label="val_acc")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


# =========================
# 统计工具
# =========================
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def estimate_param_memory_mb(model: nn.Module) -> float:
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return total_bytes / (1024 ** 2)


def file_size_mb(path: Path) -> float:
    if path is None or (not Path(path).exists()):
        return 0.0
    return Path(path).stat().st_size / (1024 ** 2)


def compute_conv_linear_macs_flops(model: nn.Module, example_input: torch.Tensor, macs_to_flops_factor: int = 2):
    macs_by_module = {}
    handles = []
    was_training = model.training

    def add_macs(name: str, macs: int):
        macs_by_module[name] = macs_by_module.get(name, 0) + int(macs)

    def make_hook(name: str):
        def hook(m: nn.Module, inputs, output):
            out = output
            if isinstance(m, nn.Conv1d):
                B = out.shape[0]
                Cout = out.shape[1]
                Lout = out.shape[2]
                Cin = m.in_channels
                groups = m.groups
                k = m.kernel_size[0]
                macs = B * Lout * Cout * (Cin // groups) * k
                add_macs(name, macs)

            elif isinstance(m, nn.Conv2d):
                B = out.shape[0]
                Cout = out.shape[1]
                Hout = out.shape[2]
                Wout = out.shape[3]
                Cin = m.in_channels
                groups = m.groups
                kH, kW = m.kernel_size
                macs = B * Hout * Wout * Cout * (Cin // groups) * kH * kW
                add_macs(name, macs)

            elif isinstance(m, nn.Linear):
                out_features = m.out_features
                batch_like = out.numel() // out_features
                macs = batch_like * m.in_features * out_features
                add_macs(name, macs)

        return hook

    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            handles.append(m.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        _ = model(example_input)

    for h in handles:
        h.remove()

    if was_training:
        model.train()

    total_macs = int(sum(macs_by_module.values()))
    total_flops = int(total_macs * macs_to_flops_factor)

    return {
        "macs_total": total_macs,
        "flops_total": total_flops,
        "macs_to_flops_factor": macs_to_flops_factor,
        "macs_by_module": macs_by_module
    }


# ----------------------------
# 评估
# ----------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    if isinstance(device, str):
        device = torch.device(device)

    model.eval()
    all_true, all_pred = [], []
    total_loss = 0.0
    ce = nn.CrossEntropyLoss()

    if device.type == 'cuda':
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    for batch in loader:
        if len(batch) == 2:
            xb, yb = batch
        else:
            xb, yb, _ = batch
        xb = xb.to(device)
        yb = yb.to(device)

        logits, _, _ = model(xb)
        loss = ce(logits, yb).item()
        total_loss += loss * yb.size(0)

        pred = logits.argmax(dim=1)
        all_true.append(yb.cpu().numpy())
        all_pred.append(pred.cpu().numpy())

    if device.type == 'cuda':
        torch.cuda.synchronize()
    t_end = time.perf_counter()
    total_time = t_end - t_start
    total_samples = sum(len(t) for t in all_true)
    avg_inference_time_ms = (total_time / total_samples) * 1000.0

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    acc = accuracy_score(y_true, y_pred)
    avg_loss = total_loss / len(y_true)
    f1m = f1_score(y_true, y_pred, average="macro")
    return avg_loss, acc, f1m, y_true, y_pred, avg_inference_time_ms


def train_one_fold(
    fold_id: int,
    X_all, y_all,
    train_idx, val_idx,
    out_dir: Path,
    comps, freqs,
    num_classes, class_names,
    epochs=50,
    batch_size=64,
    lr=1e-3,
    weight_decay=1e-4,
    pseudo_size=64,
    patience=10,
    save_pseudo_max_per_class=30,
    seed=42,
    spec_feat_dim=128,
    img_feat_dim=128,
    fusion_dropout=0.2
):
    fold_dir = out_dir / f"fold_{fold_id}"
    ckpt_dir = fold_dir / "checkpoints"
    vis_dir = fold_dir / "pseudo_images"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    scaler = StandardScaler()
    X_train = X_all[train_idx]
    X_val = X_all[val_idx]

    n_comp = len(comps)
    n_freq = len(freqs)
    X_train_flat = X_train.reshape(len(train_idx), -1)
    X_val_flat = X_val.reshape(len(val_idx), -1)

    X_train_s = scaler.fit_transform(X_train_flat).reshape(-1, n_comp, n_freq).astype(np.float32)
    X_val_s = scaler.transform(X_val_flat).reshape(-1, n_comp, n_freq).astype(np.float32)

    raw_val = X_val.copy().astype(np.float32)

    train_ds = BeefDataset(X_train_s, y_all[train_idx])
    val_ds = BeefDataset(X_val_s, y_all[val_idx], raw_X_for_save=raw_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=max(128, batch_size), shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BeefNet(
        n_comp=n_comp,
        n_freq=n_freq,
        num_classes=num_classes,
        pseudo_size=pseudo_size,
        spec_feat_dim=spec_feat_dim,
        img_feat_dim=img_feat_dim,
        fusion_dropout=fusion_dropout
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    ce = nn.CrossEntropyLoss()

    best_acc = -1.0
    best_path = ckpt_dir / "best.pt"
    bad_epochs = 0

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    if device == "cuda":
        torch.cuda.synchronize()
    t_train_start = time.perf_counter()

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)
            logits, _, _ = model(xb)
            loss = ce(logits, yb)
            loss.backward()
            opt.step()

            total_loss += loss.item() * yb.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.numel()

        tr_loss = total_loss / total
        tr_acc = correct / total

        va_loss, va_acc, va_f1m, _, _, va_infer_ms = evaluate(model, val_loader, device)
        scheduler.step(va_acc)

        train_losses.append(tr_loss)
        val_losses.append(va_loss)
        train_accs.append(tr_acc)
        val_accs.append(va_acc)

        if va_acc > best_acc:
            best_acc = va_acc
            bad_epochs = 0
            torch.save({
                "model": model.state_dict(),
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
                "val_acc": best_acc,
                "epoch": ep,
                "seed": seed,
                "spec_feat_dim": spec_feat_dim,
                "img_feat_dim": img_feat_dim,
                "fusion_dropout": fusion_dropout,
                "comps": comps,
                "freqs": freqs,
            }, best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if device == "cuda":
        torch.cuda.synchronize()
    t_train_end = time.perf_counter()
    train_time_sec = float(t_train_end - t_train_start)

    plot_curves(train_losses, val_losses, train_accs, val_accs, fold_dir / "curves.png")

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    va_loss, va_acc, va_f1m, y_true, y_pred, va_infer_ms = evaluate(model, val_loader, device)

    param_count = count_parameters(model)
    param_mem_mb = estimate_param_memory_mb(model)
    best_ckpt_mb = file_size_mb(best_path)

    dummy = torch.zeros(1, n_comp, n_freq, dtype=torch.float32, device=device)
    flops_info = compute_conv_linear_macs_flops(model, dummy, macs_to_flops_factor=2)

    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    plot_confusion_matrix(cm, class_names, fold_dir / "confusion_matrix.png", normalize=False)
    plot_confusion_matrix(cm, class_names, fold_dir / "confusion_matrix_norm.png", normalize=True)

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    (fold_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    metrics = {
        "fold": fold_id,
        "best_val_acc": float(best_acc),
        "val_loss": float(va_loss),
        "val_acc": float(va_acc),
        "val_macro_f1": float(va_f1m),
        "val_inference_time_ms_per_sample": float(va_infer_ms),
        "epochs_ran": len(train_losses),
        "train_time_sec": float(train_time_sec),
        "param_count": int(param_count),
        "param_memory_mb_est": float(param_mem_mb),
        "best_ckpt_file_mb": float(best_ckpt_mb),
        "flops_per_forward_batch1": int(flops_info["flops_total"]),
        "macs_per_forward_batch1": int(flops_info["macs_total"]),
        "flops_detail": {
            "macs_to_flops_factor": int(flops_info["macs_to_flops_factor"]),
            "macs_by_module": flops_info["macs_by_module"],
        }
    }
    (fold_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[Fold {fold_id}] Train time: {train_time_sec:.3f} s")
    print(f"[Fold {fold_id}] Params: {param_count:,} | ParamMem~ {param_mem_mb:.3f} MB | BestCkpt: {best_ckpt_mb:.3f} MB")
    print(f"[Fold {fold_id}] FLOPs(batch=1 forward): {flops_info['flops_total']:,}  (MACs: {flops_info['macs_total']:,})")
    print(f"[Fold {fold_id}] Avg inference time per sample: {va_infer_ms:.3f} ms")

    saved_count = {c: 0 for c in range(num_classes)}
    model.eval()
    sample_global = 0
    with torch.no_grad():
        for batch in val_loader:
            xb, yb, rawb = batch
            xb = xb.to(device)
            logits, _, h = model(xb)
            pred = logits.argmax(dim=1).cpu().numpy()

            yb_np = yb.numpy()
            rawb_np = rawb.numpy()

            for i in range(len(yb_np)):
                gt = int(yb_np[i])
                if saved_count[gt] >= save_pseudo_max_per_class:
                    continue

                height_2d = rawb_np[i].T
                nm = height_to_normalmap_np(height_2d)

                sid = f"val_{sample_global:05d}_gt{gt}_pred{int(pred[i])}"
                save_normalmap_with_axes(nm, vis_dir / f"{sid}_normal.png", comps, freqs)
                save_surface_3d(height_2d, vis_dir / f"{sid}_surface3d.png", comps, freqs)

                saved_count[gt] += 1
                sample_global += 1

            if all(saved_count[c] >= save_pseudo_max_per_class for c in saved_count):
                break

    return metrics, (val_idx, y_true, y_pred)


# ----------------------------
# 主函数
# ----------------------------
def main(
    excel_path: str,
    out_dir: str = "run/juice",
    seed: int = 42,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    pseudo_size: int = 64,
    patience: int = 10,
    save_pseudo_max_per_class: int = 30,
    spec_feat_dim: int = 128,
    img_feat_dim: int = 128,
    fusion_dropout: float = 0.2,
):
    seed_everything(seed)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "excel_path": excel_path,
        "out_dir": str(out_dir),
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "pseudo_size": pseudo_size,
        "patience": patience,
        "save_pseudo_max_per_class": save_pseudo_max_per_class,
        "spec_feat_dim": spec_feat_dim,
        "img_feat_dim": img_feat_dim,
        "fusion_dropout": fusion_dropout,
        "flops_convention": "FLOPs = 2*MACs (mul+add), count Conv/Linear only"
    }
    (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    df = pd.read_excel(excel_path)
    X_all, y_all, colmap, comps, freqs = parse_columns_and_build_matrix(df, label_col="kind")

    unique_labels = np.unique(y_all)
    num_classes = len(unique_labels)
    class_names = [f"class_{lab}" for lab in sorted(unique_labels)]

    print(f"检测到分量: {comps}")
    print(f"检测到频率: {freqs}")
    print(f"类别数: {num_classes} ({class_names})")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    fold_metrics = []
    overall_pred = np.zeros_like(y_all)
    overall_true = y_all.copy()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_cv0 = time.perf_counter()

    for fold_id, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all), start=1):
        metrics, (v_idx, y_true, y_pred) = train_one_fold(
            fold_id=fold_id,
            X_all=X_all,
            y_all=y_all,
            train_idx=train_idx,
            val_idx=val_idx,
            out_dir=out_dir,
            comps=comps,
            freqs=freqs,
            num_classes=num_classes,
            class_names=class_names,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            pseudo_size=pseudo_size,
            patience=patience,
            save_pseudo_max_per_class=save_pseudo_max_per_class,
            seed=seed,
            spec_feat_dim=spec_feat_dim,
            img_feat_dim=img_feat_dim,
            fusion_dropout=fusion_dropout,
        )
        fold_metrics.append(metrics)
        overall_pred[v_idx] = y_pred

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_cv1 = time.perf_counter()
    total_cv_time_sec = float(t_cv1 - t_cv0)

    accs = [m["val_acc"] for m in fold_metrics]
    f1s = [m["val_macro_f1"] for m in fold_metrics]
    infer_times = [m["val_inference_time_ms_per_sample"] for m in fold_metrics]

    summary = {
        "acc_each_fold": accs,
        "macro_f1_each_fold": f1s,
        "inference_time_ms_each_fold": infer_times,
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "macro_f1_mean": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s)),
        "inference_time_ms_mean": float(np.mean(infer_times)),
        "inference_time_ms_std": float(np.std(infer_times)),
        "train_time_sec_each_fold": [float(m["train_time_sec"]) for m in fold_metrics],
        "total_cv_time_sec": total_cv_time_sec
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    cm_all = confusion_matrix(overall_true, overall_pred, labels=unique_labels)
    plot_confusion_matrix(cm_all, class_names, out_dir / "overall_confusion_matrix.png", normalize=False)
    plot_confusion_matrix(cm_all, class_names, out_dir / "overall_confusion_matrix_norm.png", normalize=True)

    report_all = classification_report(overall_true, overall_pred, target_names=class_names, digits=4)
    (out_dir / "overall_classification_report.txt").write_text(report_all, encoding="utf-8")

    import csv
    with open(out_dir / "fold_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fold_metrics[0].keys()))
        w.writeheader()
        for row in fold_metrics:
            w.writerow(row)

    print("Done. Results saved to:", out_dir.resolve())
    print("5-fold ACC:", accs)
    print("ACC mean±std:", summary["acc_mean"], "±", summary["acc_std"])
    print("5-fold inference time (ms) per sample:", infer_times)
    print("Inference time mean±std: {:.3f} ± {:.3f} ms".format(summary["inference_time_ms_mean"], summary["inference_time_ms_std"]))
    print("[CV] Total time (5 folds):", total_cv_time_sec, "sec")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel_path", type=str, default="D:/1DCNN/Beef-Milk-Juice/strong/juice.xlsx", help="Excel文件路径")
    parser.add_argument("--out_dir", type=str, default="D:/1DCNN/Beef-Milk-Juice/strong/juice_ours", help="输出目录")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--pseudo_size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--save_pseudo_max_per_class", type=int, default=30)
    parser.add_argument("--spec_feat_dim", type=int, default=128)
    parser.add_argument("--img_feat_dim", type=int, default=128)
    parser.add_argument("--fusion_dropout", type=float, default=0.2)

    args = parser.parse_args()

    main(
        excel_path=args.excel_path,
        out_dir=args.out_dir,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        pseudo_size=args.pseudo_size,
        patience=args.patience,
        save_pseudo_max_per_class=args.save_pseudo_max_per_class,
        spec_feat_dim=args.spec_feat_dim,
        img_feat_dim=args.img_feat_dim,
        fusion_dropout=args.fusion_dropout,
    )