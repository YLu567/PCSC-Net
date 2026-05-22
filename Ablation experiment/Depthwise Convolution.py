# train_beef_percomp.py
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
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)      # [N,6,8] (标准化后)
        self.y = torch.tensor(y, dtype=torch.long)         # [N]

    def __len__(self):
        return self.y.numel()

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


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


# ----------------------------
# 3) 简化分类网络（仅使用 per-component 卷积）
# ----------------------------
class BeefNetPerComp(nn.Module):
    """
    仅使用每分量小卷积提取特征，然后展平后分类。
    """
    def __init__(self, num_classes=3, dropout=0.2):
        super().__init__()
        self.per_comp = PerComponentConv(hidden_per_comp=8, k=3)
        self.classifier = nn.Sequential(
            nn.Flatten(),                     # [B, 6*8=48]
            nn.Dropout(p=dropout),
            nn.Linear(48, num_classes)
        )

    def forward(self, x):
        x = self.per_comp(x)                  # [B,6,8]
        logits = self.classifier(x)           # [B, num_classes]
        return logits


# ----------------------------
# 4) 可视化工具
# ----------------------------
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
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_curves(train_losses, val_losses, train_accs, val_accs, val_noise_accs, save_path: Path):
    fig = plt.figure(figsize=(10, 5))
    x = np.arange(1, len(train_losses) + 1)
    plt.plot(x, train_losses, label="train_loss")
    plt.plot(x, val_losses, label="val_loss")
    plt.plot(x, train_accs, label="train_acc")
    plt.plot(x, val_accs, label="val_acc (clean)")
    plt.plot(x, val_noise_accs, label="val_acc (noisy)", linestyle='--')
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
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
    用 forward hook 统计 Conv1d/Linear 的 MACs。
    默认 FLOPs = 2 * MACs（一次乘法+一次加法）。
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

            elif isinstance(m, nn.Linear):
                out_features = m.out_features
                batch_like = out.numel() // out_features
                macs = batch_like * m.in_features * out_features
                add_macs(name, macs)

        return hook

    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv1d, nn.Linear)):
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


def measure_inference_time(model, loader, device, num_runs=5, noise_std=0.0):
    """
    测量模型在给定数据加载器上的平均推理时间（每个样本的毫秒数）。
    包含数据从CPU到GPU的传输和前向计算，可添加噪声。
    """
    model.eval()
    total_samples = len(loader.dataset)
    total_time = 0.0

    # 预热
    with torch.no_grad():
        for _ in range(2):
            for batch in loader:
                xb, _ = batch
                xb = xb.to(device)
                if noise_std > 0:
                    xb = xb + torch.randn_like(xb) * noise_std
                _ = model(xb)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # 正式测量
    for run in range(num_runs):
        start = time.perf_counter()
        with torch.no_grad():
            for batch in loader:
                xb, _ = batch
                xb = xb.to(device)
                if noise_std > 0:
                    xb = xb + torch.randn_like(xb) * noise_std
                _ = model(xb)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        total_time += (end - start)

    avg_time_per_sample = total_time / (num_runs * total_samples) * 1000  # 毫秒
    return avg_time_per_sample


# ----------------------------
# 5) 训练/验证（支持添加噪声）
# ----------------------------
@torch.no_grad()
def evaluate(model, loader, device, noise_std=0.0):
    model.eval()
    all_true, all_pred = [], []
    total_loss = 0.0
    ce = nn.CrossEntropyLoss()

    for batch in loader:
        xb, yb = batch
        xb = xb.to(device)
        if noise_std > 0:
            xb = xb + torch.randn_like(xb) * noise_std
        yb = yb.to(device)

        logits = model(xb)
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
                   patience=10,
                   seed=42,
                   dropout=0.2,
                   noise_std_val=0.1):          # 验证时添加的噪声标准差
    fold_dir = out_dir / f"fold_{fold_id}"
    ckpt_dir = fold_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- 标准化：只用训练集fit
    scaler = StandardScaler()
    X_train = X_all[train_idx]
    X_val = X_all[val_idx]

    X_train_flat = X_train.reshape(len(train_idx), -1)
    X_val_flat = X_val.reshape(len(val_idx), -1)

    X_train_s = scaler.fit_transform(X_train_flat).reshape(-1, 6, 8).astype(np.float32)
    X_val_s = scaler.transform(X_val_flat).reshape(-1, 6, 8).astype(np.float32)

    train_ds = BeefDataset(X_train_s, y_all[train_idx])
    val_ds = BeefDataset(X_val_s, y_all[val_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=max(128, batch_size), shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BeefNetPerComp(num_classes=3, dropout=dropout).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    ce = nn.CrossEntropyLoss()

    best_acc = -1.0
    best_path = ckpt_dir / "best.pt"
    bad_epochs = 0

    train_losses, val_losses = [], []
    train_accs, val_accs, val_noise_accs = [], [], []

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
            logits = model(xb)
            loss = ce(logits, yb)
            loss.backward()
            opt.step()

            total_loss += loss.item() * yb.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.numel()

        tr_loss = total_loss / total
        tr_acc = correct / total

        # 正常验证（无噪声）
        va_loss, va_acc, va_f1m, _, _ = evaluate(model, val_loader, device, noise_std=0.0)
        # 带噪声验证
        va_noise_loss, va_noise_acc, va_noise_f1m, _, _ = evaluate(model, val_loader, device, noise_std=noise_std_val)

        scheduler.step(va_acc)

        train_losses.append(tr_loss)
        val_losses.append(va_loss)
        train_accs.append(tr_acc)
        val_accs.append(va_acc)
        val_noise_accs.append(va_noise_acc)

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
                "dropout": dropout,
            }, best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if device == "cuda":
        torch.cuda.synchronize()
    t_train_end = time.perf_counter()
    train_time_sec = float(t_train_end - t_train_start)

    # 画训练曲线（包含噪声验证准确率）
    plot_curves(train_losses, val_losses, train_accs, val_accs, val_noise_accs, fold_dir / "curves.png")

    # --- 加载最佳模型并评估 ---
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    # 最终评估：无噪声和带噪声
    va_loss, va_acc, va_f1m, y_true, y_pred = evaluate(model, val_loader, device, noise_std=0.0)
    va_noise_loss, va_noise_acc, va_noise_f1m, _, _ = evaluate(model, val_loader, device, noise_std=noise_std_val)

    # 测量推理时间（无噪声和带噪声）
    inf_time_clean_ms = measure_inference_time(model, val_loader, device, num_runs=5, noise_std=0.0)
    inf_time_noise_ms = measure_inference_time(model, val_loader, device, num_runs=5, noise_std=noise_std_val)

    # 统计
    param_count = count_parameters(model)
    param_mem_mb = estimate_param_memory_mb(model)
    best_ckpt_mb = file_size_mb(best_path)

    dummy = torch.zeros(1, 6, 8, dtype=torch.float32, device=device)
    flops_info = compute_conv_linear_macs_flops(model, dummy, macs_to_flops_factor=2)

    # 混淆矩阵
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
        "val_noise_loss": float(va_noise_loss),
        "val_noise_acc": float(va_noise_acc),
        "val_noise_macro_f1": float(va_noise_f1m),
        "epochs_ran": len(train_losses),

        "train_time_sec": float(train_time_sec),
        "param_count": int(param_count),
        "param_memory_mb_est": float(param_mem_mb),
        "best_ckpt_file_mb": float(best_ckpt_mb),
        "flops_per_forward_batch1": int(flops_info["flops_total"]),
        "macs_per_forward_batch1": int(flops_info["macs_total"]),

        "inference_time_ms_per_sample_clean": float(inf_time_clean_ms),
        "inference_time_ms_per_sample_noisy": float(inf_time_noise_ms),
    }
    (fold_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[Fold {fold_id}] Train time: {train_time_sec:.3f} s")
    print(f"[Fold {fold_id}] Params: {param_count:,} | ParamMem~ {param_mem_mb:.3f} MB | BestCkpt: {best_ckpt_mb:.3f} MB")
    print(f"[Fold {fold_id}] FLOPs(batch=1 forward): {flops_info['flops_total']:,}")
    print(f"[Fold {fold_id}] Clean val acc: {va_acc:.4f} | Noisy val acc: {va_noise_acc:.4f} (noise std={noise_std_val})")

    return metrics, (val_idx, y_true, y_pred)


# ----------------------------
# 6) 主函数：5折交叉验证 + 保存 results/beef-percomp
# ----------------------------
def main(
    excel_path: str,
    out_dir: str = "results/beef-percomp",
    seed: int = 42,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
    dropout: float = 0.2,
    noise_std_val: float = 0.5,          # 验证时添加的噪声标准差
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
        "patience": patience,
        "dropout": dropout,
        "noise_std_val": noise_std_val,
        "model": "BeefNetPerComp (only per-component conv)",
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
            patience=patience,
            seed=seed,
            dropout=dropout,
            noise_std_val=noise_std_val,
        )
        fold_metrics.append(metrics)
        overall_pred[v_idx] = y_pred

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_cv1 = time.perf_counter()
    total_cv_time_sec = float(t_cv1 - t_cv0)

    # 汇总统计
    accs = [m["val_acc"] for m in fold_metrics]
    noise_accs = [m["val_noise_acc"] for m in fold_metrics]
    f1s = [m["val_macro_f1"] for m in fold_metrics]
    noise_f1s = [m["val_noise_macro_f1"] for m in fold_metrics]
    inf_clean = [m["inference_time_ms_per_sample_clean"] for m in fold_metrics]
    inf_noisy = [m["inference_time_ms_per_sample_noisy"] for m in fold_metrics]

    summary = {
        "acc_each_fold": accs,
        "noise_acc_each_fold": noise_accs,
        "macro_f1_each_fold": f1s,
        "noise_macro_f1_each_fold": noise_f1s,
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "noise_acc_mean": float(np.mean(noise_accs)),
        "noise_acc_std": float(np.std(noise_accs)),
        "macro_f1_mean": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s)),
        "noise_macro_f1_mean": float(np.mean(noise_f1s)),
        "noise_macro_f1_std": float(np.std(noise_f1s)),
        "train_time_sec_each_fold": [float(m["train_time_sec"]) for m in fold_metrics],
        "total_cv_time_sec": total_cv_time_sec,
        "inference_time_ms_clean_mean": float(np.mean(inf_clean)),
        "inference_time_ms_clean_std": float(np.std(inf_clean)),
        "inference_time_ms_noisy_mean": float(np.mean(inf_noisy)),
        "inference_time_ms_noisy_std": float(np.std(inf_noisy)),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # 整体混淆矩阵
    cm_all = confusion_matrix(overall_true, overall_pred, labels=[0, 1, 2])
    class_names = ["fresh(0)", "carrageenan(1)", "compound(2)"]
    plot_confusion_matrix(cm_all, class_names, out_dir / "overall_confusion_matrix.png", normalize=False)
    plot_confusion_matrix(cm_all, class_names, out_dir / "overall_confusion_matrix_norm.png", normalize=True)

    report_all = classification_report(overall_true, overall_pred, target_names=class_names, digits=4)
    (out_dir / "overall_classification_report.txt").write_text(report_all, encoding="utf-8")

    # 保存每个fold的详细指标为CSV
    import csv
    with open(out_dir / "fold_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fold_metrics[0].keys()))
        w.writeheader()
        for row in fold_metrics:
            w.writerow(row)

    print("Done. Results saved to:", out_dir.resolve())
    print("5-fold ACC (clean):", accs)
    print("ACC mean±std (clean):", summary["acc_mean"], "±", summary["acc_std"])
    print("5-fold ACC (noisy):", noise_accs)
    print("ACC mean±std (noisy):", summary["noise_acc_mean"], "±", summary["noise_acc_std"])
    print("Inference time per sample (clean) mean±std (ms):", summary["inference_time_ms_clean_mean"], "±", summary["inference_time_ms_clean_std"])
    print("Inference time per sample (noisy) mean±std (ms):", summary["inference_time_ms_noisy_mean"], "±", summary["inference_time_ms_noisy_std"])
    print("[CV] Total time (5 folds):", total_cv_time_sec, "sec")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel_path", type=str, default="D:/1DCNN/Beef-Milk-Juice/data/beef_data.xlsx")
    parser.add_argument("--out_dir", type=str, default="D:/1DCNN/Beef-Milk-Juice/results/beef-percomp")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--noise_std_val", type=float, default=0.6, help="验证时添加的高斯噪声标准差")

    args = parser.parse_args()

    main(
        excel_path=args.excel_path,
        out_dir=args.out_dir,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        dropout=args.dropout,
        noise_std_val=args.noise_std_val,
    )