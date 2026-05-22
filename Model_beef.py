# train_beef_3class_fusion.py
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


# ----------------------------
# 0) 固定随机种子
# ----------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------
# 1) 读 Excel -> [N,6,8]
# ----------------------------
def parse_columns_and_build_matrix(df: pd.DataFrame,
                                   label_col: str = "类型",
                                   components=("CP", "CS", "Z", "Φ", "X", "R"),
                                   freqs=(100, 500, 1000, 3000, 8000, 15000, 50000, 200000)):
    """
    兼容中文全角括号：CP（100） 或普通括号：CP(100)
    返回：
      X: float32 [N,6,8]
      y: int64   [N]
      colmap: dict[(comp,freq)] -> column_name
    """
    cols = list(df.columns)
    if label_col not in cols:
        raise ValueError(f"Excel中找不到标签列：{label_col}，当前列：{cols[:10]}...")

    import re
    colmap = {}
    pat = re.compile(r"^(.*?)[(（]\s*(\d+)\s*[)）]\s*$")

    for c in cols:
        if c == label_col:
            continue
        m = pat.match(str(c))
        if not m:
            continue
        comp = m.group(1).strip()
        freq = int(m.group(2).strip())
        colmap[(comp, freq)] = c

    missing = []
    for comp in components:
        for f in freqs:
            if (comp, f) not in colmap:
                missing.append((comp, f))
    if missing:
        raise ValueError(
            "缺少以下分量-频率列（请检查Excel列名是否一致，如 Φ 是否写成phi/Φ 等）：\n"
            + "\n".join([f"{a}-{b}" for a, b in missing[:50]])
            + ("\n...(more)" if len(missing) > 50 else "")
        )

    N = len(df)
    X = np.zeros((N, len(components), len(freqs)), dtype=np.float32)
    for i, comp in enumerate(components):
        for j, f in enumerate(freqs):
            X[:, i, j] = df[colmap[(comp, f)]].to_numpy(np.float32)

    y = df[label_col].to_numpy(np.int64)
    return X, y, colmap


class BeefDataset(Dataset):
    def __init__(self, X, y, raw_X_for_save=None):
        self.X = torch.tensor(X, dtype=torch.float32)      # [N,6,8] (标准化后)
        self.y = torch.tensor(y, dtype=torch.long)         # [N]
        self.raw = None
        if raw_X_for_save is not None:
            self.raw = torch.tensor(raw_X_for_save, dtype=torch.float32)  # [N,6,8] (原始未标准化)

    def __len__(self):
        return self.y.numel()

    def __getitem__(self, idx):
        if self.raw is None:
            return self.X[idx], self.y[idx]
        return self.X[idx], self.y[idx], self.raw[idx]


# ----------------------------
# 2) 模型模块：每分量1D小卷积（groups=6）
# ----------------------------
class PerComponentConv(nn.Module):
    """
    输入:  [B,6,8]
    输出:  [B,6,8]
    每个分量(通道)独立做1D卷积（groups=6），实现“每个分量分别小卷积处理”
    """
    def __init__(self, hidden_per_comp=8, k=3):
        super().__init__()
        self.conv_expand = nn.Conv1d(
            in_channels=6,
            out_channels=6 * hidden_per_comp,
            kernel_size=k,
            padding=k // 2,
            groups=6,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(6 * hidden_per_comp)

        self.conv_reduce = nn.Conv1d(
            in_channels=6 * hidden_per_comp,
            out_channels=6,
            kernel_size=1,
            groups=6,
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(6)

    def forward(self, x):
        x = self.conv_expand(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.conv_reduce(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        return x

def palette_map_torch(t: torch.Tensor) -> torch.Tensor:
    """
    t: [B,1,H,W] in [0,1]
    return: [B,3,H,W]
    按照你给的框架图配色做分段插值：
    深海军蓝 -> 青蓝 -> 浅钢蓝 -> 米金 -> 棕金
    """
    anchors = torch.tensor([
        [31, 72, 103],    # #1F4867  深海军蓝
        [35, 124, 133],   # #237C85  青蓝
        [123, 172, 196],  # #7BACC4  浅钢蓝
        [221, 200, 156],  # #DDC89C  米金
        [188, 137, 89],   # #BC8959  棕金
    ], dtype=t.dtype, device=t.device) / 255.0

    n_seg = anchors.shape[0] - 1
    x = torch.clamp(t, 0.0, 1.0) * n_seg
    idx0 = torch.floor(x).long().clamp(min=0, max=n_seg - 1)
    idx1 = (idx0 + 1).clamp(max=n_seg)

    w = (x - idx0.float())

    c0 = anchors[idx0.squeeze(1)]   # [B,H,W,3]
    c1 = anchors[idx1.squeeze(1)]   # [B,H,W,3]

    rgb = c0 * (1.0 - w.squeeze(1).unsqueeze(-1)) + c1 * w.squeeze(1).unsqueeze(-1)
    rgb = rgb.permute(0, 3, 1, 2).contiguous()   # [B,3,H,W]
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

    c0 = anchors[idx0]   # [H,W,3]
    c1 = anchors[idx1]   # [H,W,3]
    rgb = c0 * (1.0 - w) + c1 * w
    return np.clip(rgb, 0.0, 1.0)

# ----------------------------
# 3) 伪图生成器：8×6 高度图 -> normal map (RGB)
# ----------------------------
class PseudoImageGenerator(nn.Module):
    """
    输入:  height map [B,1,8,6]
    输出:  pseudo image [B,3,S,S]，默认 S=64

    用法线贴图(normal map)表征3D surface起伏：
      - 先计算dz/dx, dz/dy
      - 拼接法向量 [-dzdx, -dzdy, 1]
      - 归一化后映射到RGB
    再上采样 + 可学习refine（小2D卷积）实现端到端。
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
        # h: [B,1,8,6]
        h_min = h.amin(dim=(2, 3), keepdim=True)
        h_max = h.amax(dim=(2, 3), keepdim=True)
        h_norm = (h - h_min) / (h_max - h_min + 1e-6)  # [0,1]

        dzdy = h[:, :, 2:, :] - h[:, :, :-2, :]  # [B,1,6,6]
        dzdx = h[:, :, :, 2:] - h[:, :, :, :-2]  # [B,1,8,4]
        dzdy = F.pad(dzdy, (0, 0, 1, 1), mode='replicate')
        dzdx = F.pad(dzdx, (1, 1, 0, 0), mode='replicate')

        slope = torch.sqrt(dzdx.pow(2) + dzdy.pow(2) + 1e-8)
        slope_max = slope.amax(dim=(2, 3), keepdim=True)
        slope_norm = slope / (slope_max + 1e-6)  # [0,1]

        # 1) 先按高度值做“框架图同款色系”映射
        base = palette_map_torch(h_norm)  # [B,3,8,6]

        # 2) 再用坡度做轻微明暗调制，让纹理更自然
        shade = 0.82 + 0.18 * (1.0 - slope_norm)  # [B,1,8,6]
        img = torch.clamp(base * shade, 0.0, 1.0)

        # 3) 双线性上采样
        img = F.interpolate(
            img,
            size=(self.out_size, self.out_size),
            mode='bilinear',
            align_corners=False
        )

        # 4) 可学习细化
        if self.refine is not None:
            refined = self.refine(img)
            img = 0.75 * img + 0.25 * refined

        img = torch.clamp(img, 0.0, 1.0)
        return img


# ----------------------------
# 4) 自适应噪声模块（训练时开启）
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
# 5) 判别网络：伪图特征 + 原谱特征 融合后分类
# ----------------------------
class BeefNet(nn.Module):
    """
    你原来的端到端流程保留：
      x -> per_comp -> height map -> pseudo image -> backbone -> feature_img

    新增“原谱特征分支”：
      x(原谱, 标准化后) -> flatten -> MLP -> feature_spec

    融合：
      concat([feature_img, feature_spec]) -> classifier
    """
    def __init__(self,
                 num_classes=3,
                 pseudo_size=64,
                 spec_feat_dim=128,
                 img_feat_dim=128,
                 fusion_dropout=0.2):
        super().__init__()
        self.per_comp = PerComponentConv(hidden_per_comp=8, k=3)
        self.pseudo = PseudoImageGenerator(out_size=pseudo_size, learnable_refine=True)
        self.noise = AdaptiveNoise(channels=3, max_sigma=0.15)

        # --- 伪图CNN分支（输出 img_feat_dim）
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /2
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /4
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, img_feat_dim),
            nn.ReLU(inplace=True),
        )

        # --- 原谱数据分支（输入 6*8=48 -> spec_feat_dim）
        self.spec_encoder = nn.Sequential(
            nn.Flatten(),                # [B,48]
            nn.Linear(48, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=fusion_dropout),
            nn.Linear(128, spec_feat_dim),
            nn.ReLU(inplace=True),
        )

        # --- 融合分类器
        fusion_dim = img_feat_dim + spec_feat_dim
        self.classifier = nn.Sequential(
            nn.Dropout(p=fusion_dropout),
            nn.Linear(fusion_dim, num_classes)
        )

    def forward(self, x):
        """
        x: [B,6,8] (标准化后的原谱数据)
        return:
          logits: [B,C]
          img:    [B,3,S,S] (伪图)
          h:      [B,1,8,6] (高度图)
        """
        x_in = x  # 保留“原谱数据”分支输入

        # 伪图分支：先做每分量小卷积，再转高度图生成伪图
        x_proc = self.per_comp(x_in)                  # [B,6,8]
        h = x_proc.transpose(1, 2).unsqueeze(1)       # [B,1,8,6]
        img = self.pseudo(h)                          # [B,3,S,S]
        img = self.noise(img)                         # train only

        feat_img = self.backbone(img)                 # [B,img_feat_dim]
        feat_spec = self.spec_encoder(x_in)           # [B,spec_feat_dim]

        feat = torch.cat([feat_img, feat_spec], dim=1)
        logits = self.classifier(feat)

        return logits, img, h


# ----------------------------
# 6) 可视化：保存伪图(normal map) + 真实3D surface
# ----------------------------
def height_to_normalmap_np(height_8x6: np.ndarray):
    """
    height_8x6: [8,6]
    return: pseudo color RGB in [0,1], shape [8,6,3]

    配色改成与你给的框架图一致的蓝-青-浅蓝-米金-棕金系
    """
    h = height_8x6.astype(np.float32)

    h_min = np.min(h)
    h_max = np.max(h)
    h_norm = (h - h_min) / (h_max - h_min + 1e-6)

    gy, gx = np.gradient(h)
    slope = np.sqrt(gx ** 2 + gy ** 2)
    slope_norm = slope / (np.max(slope) + 1e-6)

    # 按高度映射到框架图色系
    base = palette_map_np(h_norm)   # [8,6,3]

    # 用坡度做轻微阴影调节
    shade = 0.82 + 0.18 * (1.0 - slope_norm)
    img = base * shade[..., None]

    img = np.clip(img, 0.0, 1.0)
    return img


def save_surface_3d(height_8x6: np.ndarray, save_path: Path,
                    freqs=(100, 500, 1000, 3000, 8000, 15000, 50000, 200000),
                    comps=("CP", "CS", "Z", "Φ", "X", "R")):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    h = height_8x6.astype(np.float32)
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
            plt.text(j, i, format(val, fmt),
                     ha="center", va="center",
                     color="white" if val > thresh else "black")

    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
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
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


# =========================
# 统计工具：模型大小 / FLOPs / 推理时间
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


def compute_conv_linear_macs_flops(model: nn.Module,
                                  example_input: torch.Tensor,
                                  macs_to_flops_factor: int = 2):
    """
    用 forward hook 统计 Conv1d/Conv2d/Linear 的 MACs。
    默认 FLOPs = 2 * MACs（一次乘法+一次加法）。
    说明：
      - 不计 BN/ReLU/Pool/Interpolate/Normalize 等（占比通常较小）
      - 适合做粗略对比
    """
    macs_by_module = {}
    handles = []
    was_training = model.training

    def add_macs(name: str, macs: int):
        macs_by_module[name] = macs_by_module.get(name, 0) + int(macs)

    def make_hook(name: str):
        def hook(m: nn.Module, inputs, output):
            out = output
            if isinstance(m, nn.Conv1d):
                # out: [B, Cout, Lout]
                B = out.shape[0]
                Cout = out.shape[1]
                Lout = out.shape[2]
                Cin = m.in_channels
                groups = m.groups
                k = m.kernel_size[0]
                macs = B * Lout * Cout * (Cin // groups) * k
                add_macs(name, macs)

            elif isinstance(m, nn.Conv2d):
                # out: [B, Cout, Hout, Wout]
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

    model.eval()  # 避免 AdaptiveNoise 在 train 里生效
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


# =========================
# 新增：推理时间测量函数
# =========================
def measure_inference_time(model, loader, device, num_runs=5):
    """
    测量模型在给定数据加载器上的平均推理时间（每个样本的毫秒数）。
    包含数据从CPU到GPU的传输和前向计算，重复 num_runs 次取平均。
    """
    model.eval()
    total_samples = len(loader.dataset)
    total_time = 0.0

    # 预热（2个epoch）
    with torch.no_grad():
        for _ in range(2):
            for batch in loader:
                if len(batch) == 2:
                    xb, _ = batch
                else:
                    xb, _, _ = batch
                xb = xb.to(device)
                _ = model(xb)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # 正式测量
    for run in range(num_runs):
        start = time.perf_counter()
        with torch.no_grad():
            for batch in loader:
                if len(batch) == 2:
                    xb, _ = batch
                else:
                    xb, _, _ = batch
                xb = xb.to(device)
                _ = model(xb)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        total_time += (end - start)

    avg_time_per_sample = total_time / (num_runs * total_samples) * 1000  # 毫秒
    return avg_time_per_sample


# ----------------------------
# 7) 训练/验证
# ----------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_true, all_pred = [], []
    total_loss = 0.0
    ce = nn.CrossEntropyLoss()

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

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    acc = accuracy_score(y_true, y_pred)
    avg_loss = total_loss / len(y_true)
    f1m = f1_score(y_true, y_pred, average="macro")
    return avg_loss, acc, f1m, y_true, y_pred


def train_one_fold(fold_id: int,
                   X_all, y_all,
                   train_idx, val_idx,
                   out_dir: Path,
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
                   fusion_dropout=0.2):
    fold_dir = out_dir / f"fold_{fold_id}"
    ckpt_dir = fold_dir / "checkpoints"
    vis_dir = fold_dir / "pseudo_images"

    # ✅ 新增：训练伪图保存目录
    train_vis_dir = out_dir / "train-pseudo_images"
    train_vis_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    # --- 标准化：只用训练集fit，避免数据泄漏
    scaler = StandardScaler()
    X_train = X_all[train_idx]
    X_val = X_all[val_idx]

    X_train_flat = X_train.reshape(len(train_idx), -1)
    X_val_flat = X_val.reshape(len(val_idx), -1)

    X_train_s = scaler.fit_transform(X_train_flat).reshape(-1, 6, 8).astype(np.float32)
    X_val_s = scaler.transform(X_val_flat).reshape(-1, 6, 8).astype(np.float32)

    # 保留 raw 用于可视化（保存伪图/3D surface）
    raw_val = X_val.copy().astype(np.float32)

    train_ds = BeefDataset(X_train_s, y_all[train_idx])
    val_ds = BeefDataset(X_val_s, y_all[val_idx], raw_X_for_save=raw_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=max(128, batch_size), shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BeefNet(
        num_classes=3,
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

            # ✅ 获取伪图
            logits, pseudo_img, h = model(xb)
            loss = ce(logits, yb)
            loss.backward()
            opt.step()

            total_loss += loss.item() * yb.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.numel()

        tr_loss = total_loss / total
        tr_acc = correct / total

        va_loss, va_acc, va_f1m, _, _ = evaluate(model, val_loader, device)
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
            }, best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if device == "cuda":
        torch.cuda.synchronize()
    t_train_end = time.perf_counter()
    train_time_sec = float(t_train_end - t_train_start)

    # 画训练曲线
    plot_curves(train_losses, val_losses, train_accs, val_accs, fold_dir / "curves.png")

    # --- 加载最佳模型并评估 ---
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    va_loss, va_acc, va_f1m, y_true, y_pred = evaluate(model, val_loader, device)

    # ===== 新增：测量推理时间 =====
    inf_time_ms = measure_inference_time(model, val_loader, device, num_runs=5)
    print(f"[Fold {fold_id}] Inference time per sample: {inf_time_ms:.3f} ms")

    # ===== 原有统计 =====
    param_count = count_parameters(model)
    param_mem_mb = estimate_param_memory_mb(model)
    best_ckpt_mb = file_size_mb(best_path)

    dummy = torch.zeros(1, 6, 8, dtype=torch.float32, device=device)
    flops_info = compute_conv_linear_macs_flops(model, dummy, macs_to_flops_factor=2)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    class_names = ["fresh(0)", "carrageenan(1)", "compound(2)"]
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
        "epochs_ran": len(train_losses),

        # 模型大小 & FLOPs
        "train_time_sec": float(train_time_sec),
        "param_count": int(param_count),
        "param_memory_mb_est": float(param_mem_mb),
        "best_ckpt_file_mb": float(best_ckpt_mb),
        "flops_per_forward_batch1": int(flops_info["flops_total"]),
        "macs_per_forward_batch1": int(flops_info["macs_total"]),
        "flops_detail": {
            "macs_to_flops_factor": int(flops_info["macs_to_flops_factor"]),
            "macs_by_module": flops_info["macs_by_module"],
        },

        # 新增：推理时间
        "inference_time_ms_per_sample": float(inf_time_ms),
    }
    (fold_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[Fold {fold_id}] Train time: {train_time_sec:.3f} s")
    print(f"[Fold {fold_id}] Params: {param_count:,} | ParamMem~ {param_mem_mb:.3f} MB | BestCkpt: {best_ckpt_mb:.3f} MB")
    print(f"[Fold {fold_id}] FLOPs(batch=1 forward): {flops_info['flops_total']:,}  (MACs: {flops_info['macs_total']:,})")

    # --- 保存部分样本的伪图 ---
    saved_count = {0: 0, 1: 0, 2: 0}

    def save_normalmap_with_axes(nm_8x6_rgb: np.ndarray,
                                 save_path: Path,
                                 comps=("CP", "CS", "Z", "Φ", "X", "R"),
                                 freqs=(100, 500, 1000, 3000, 8000, 15000, 50000, 200000),
                                 dpi=200):
        fig, ax = plt.subplots(figsize=(6.5, 4.8))
        ax.imshow(nm_8x6_rgb, origin="lower", aspect="auto", interpolation="nearest")

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

    model.eval()
    sample_global = 0
    with torch.no_grad():
        for batch in val_loader:
            xb, yb, rawb = batch
            xb = xb.to(device)
            logits, pseudo_img, h = model(xb)  # pseudo_img: [B,3,S,S], h: [B,1,8,6]
            pred = logits.argmax(dim=1).cpu().numpy()

            yb_np = yb.numpy()
            rawb_np = rawb.numpy()  # [B,6,8] raw

            for i in range(len(yb_np)):
                gt = int(yb_np[i])
                if saved_count[gt] >= save_pseudo_max_per_class:
                    continue

                height_8x6 = rawb_np[i].T  # [8,6]
                nm = height_to_normalmap_np(height_8x6)  # [8,6,3]

                sid = f"val_{sample_global:05d}_gt{gt}_pred{int(pred[i])}"
                save_normalmap_with_axes(
                    nm,
                    vis_dir / f"{sid}_normal.png",
                    comps=("CP", "CS", "Z", "Φ", "X", "R"),
                    freqs=(100, 500, 1000, 3000, 8000, 15000, 50000, 200000),
                )

                save_surface_3d(height_8x6, vis_dir / f"{sid}_surface3d.png")

                saved_count[gt] += 1
                sample_global += 1

            if all(saved_count[c] >= save_pseudo_max_per_class for c in saved_count):
                break

    return metrics, (val_idx, y_true, y_pred)


# ----------------------------
# 8) 主函数：5折交叉验证 + 保存 run/beef_ours
# ----------------------------
def main(
    excel_path: str,
    out_dir: str = "run/beef_ours",
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
    X_all, y_all, _ = parse_columns_and_build_matrix(df)

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
    inf_times = [m["inference_time_ms_per_sample"] for m in fold_metrics]   # 新增

    summary = {
        "acc_each_fold": accs,
        "macro_f1_each_fold": f1s,
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "macro_f1_mean": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s)),
        "train_time_sec_each_fold": [float(m["train_time_sec"]) for m in fold_metrics],
        "total_cv_time_sec": total_cv_time_sec,
        # 新增推理时间汇总
        "inference_time_ms_per_sample_each_fold": inf_times,
        "inference_time_ms_mean": float(np.mean(inf_times)),
        "inference_time_ms_std": float(np.std(inf_times)),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    cm_all = confusion_matrix(overall_true, overall_pred, labels=[0, 1, 2])
    class_names = ["fresh(0)", "carrageenan(1)", "compound(2)"]
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
    print("Inference time per sample (ms) mean±std:", summary["inference_time_ms_mean"], "±", summary["inference_time_ms_std"])
    print("[CV] Total time (5 folds):", total_cv_time_sec, "sec")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel_path", type=str, default="C:/Users/YLu/Desktop/beef(XR).xlsx")
    parser.add_argument("--out_dir", type=str, default="D:/1DCNN/Beef-Milk-Juice/strong/beef-ours-1")
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
