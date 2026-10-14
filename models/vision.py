"""PyTorch image classification for vision competitions.

Mixed precision (bf16 autocast on Ampere+/Blackwell such as the RTX 5070 Ti; fp16 + GradScaler on older
GPUs), torchvision augmentation, cosine LR with warmup, early stopping on validation loss, per-epoch
checkpoints so an interrupted fold resumes, and test-time horizontal-flip averaging. Backbones come from
torchvision; if `timm` is installed, any timm model name works too (pretrained weights are only used when
the competition rules allow pretrained models).
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("kaggle_agent.vision")

BACKBONES = {"resnet18": "resnet18", "resnet50": "resnet50", "efficientnet_b0": "efficientnet_b0",
             "convnext_tiny": "convnext_tiny", "vit_b_16": "vit_b_16"}


@dataclass
class VisionConfig:
    backbone: str = "resnet18"
    num_classes: int = 2
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: float = 0.5
    patience: int = 3
    pretrained: bool = False
    num_workers: int = 0          # Windows DataLoader workers re-import the module; 0 is the safe default
    seed: int = 42


def image_paths(data_dir: Path, vcfg: dict[str, Any], test: bool) -> list[Path]:
    import pandas as pd
    from competitions.profiler import read_table
    table = read_table(data_dir / vcfg["test_table" if test else "train_table"]) if vcfg.get("train_table") \
        else None
    image_dir = data_dir / vcfg["test_image_dir" if test else "image_dir"]
    if table is None:
        return sorted(image_dir.glob(f"*{vcfg.get('extension', '.jpg')}"))
    col = vcfg["image_column"]
    ext = vcfg.get("extension", "")
    return [image_dir / f"{v}{ext}" for v in pd.Series(table[col]).astype(str)]


def build_model(cfg: VisionConfig):
    import torch.nn as nn
    try:
        import timm
        return timm.create_model(cfg.backbone, pretrained=cfg.pretrained, num_classes=cfg.num_classes)
    except ImportError:
        pass
    import torchvision.models as tvm
    if cfg.backbone not in BACKBONES:
        raise ValueError(f"Unknown backbone {cfg.backbone!r} (install timm for more): {sorted(BACKBONES)}")
    model = getattr(tvm, BACKBONES[cfg.backbone])(weights="DEFAULT" if cfg.pretrained else None)
    if hasattr(model, "fc"):
        model.fc = nn.Linear(model.fc.in_features, cfg.num_classes)
    elif hasattr(model, "classifier"):
        last = model.classifier[-1]
        model.classifier[-1] = nn.Linear(last.in_features, cfg.num_classes)
    elif hasattr(model, "heads"):
        model.heads.head = nn.Linear(model.heads.head.in_features, cfg.num_classes)
    return model


class ImageDataset:
    def __init__(self, paths, labels, transform):
        self.paths, self.labels, self.transform = list(paths), labels, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        img = Image.open(self.paths[i]).convert("RGB")
        x = self.transform(img)
        return x, (int(self.labels[i]) if self.labels is not None else -1)


def _transforms(size: int, train: bool):
    from torchvision import transforms as T
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if train:
        return T.Compose([T.RandomResizedCrop(size, scale=(0.7, 1.0)), T.RandomHorizontalFlip(),
                          T.ColorJitter(0.2, 0.2, 0.2), T.ToTensor(), norm])
    return T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor(), norm])


def _predict(model, loader, device, amp_dtype):
    import torch
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x).float() + model(torch.flip(x, dims=[3])).float()  # hflip TTA
            out.append(torch.softmax(logits / 2, dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 0))


def train_fold(cfg: VisionConfig, train_paths, y_train, val_paths, y_val, test_paths, device: str = "gpu",
               ckpt_dir: Path | None = None) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    torch.manual_seed(cfg.seed)
    dev = torch.device("cuda" if device == "gpu" and torch.cuda.is_available() else "cpu")
    amp_dtype = None
    if dev.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        torch.backends.cudnn.benchmark = True
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)

    model = build_model(cfg).to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    pin = dev.type == "cuda"
    tl = DataLoader(ImageDataset(train_paths, y_train, _transforms(cfg.image_size, True)), batch_size=cfg.batch_size,
                    shuffle=True, num_workers=cfg.num_workers, pin_memory=pin, drop_last=len(train_paths) > cfg.batch_size)
    vl = DataLoader(ImageDataset(val_paths, y_val, _transforms(cfg.image_size, False)), batch_size=cfg.batch_size * 2,
                    num_workers=cfg.num_workers, pin_memory=pin)
    total_steps = max(1, cfg.epochs * len(tl))
    warmup = max(1, int(cfg.warmup_epochs * len(tl)))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps))))
    loss_fn = torch.nn.CrossEntropyLoss()

    start_epoch, best_loss, bad_epochs = 0, float("inf"), 0
    last, best_path = None, None
    if ckpt_dir:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        last, best_path = ckpt_dir / "last.pt", ckpt_dir / "best.pt"
        if last.exists():
            state = torch.load(last, map_location=dev, weights_only=False)
            model.load_state_dict(state["model"])
            opt.load_state_dict(state["opt"])
            sched.load_state_dict(state["sched"])
            start_epoch, best_loss, bad_epochs = state["epoch"] + 1, state["best_loss"], state["bad_epochs"]
            log.info("resuming fold from epoch %d", start_epoch)

    for epoch in range(start_epoch, cfg.epochs):
        if bad_epochs >= cfg.patience:
            break
        model.train()
        t0, running = time.time(), 0.0
        for x, yb in tl:
            x = x.to(dev, non_blocking=True).to(memory_format=torch.channels_last)
            yb = yb.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                loss = loss_fn(model(x), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item() * len(yb)
        val_proba = _predict(model, vl, dev, amp_dtype)
        val_loss = float(-np.log(np.clip(val_proba[np.arange(len(y_val)), y_val], 1e-15, 1)).mean())
        log.info("epoch %d train_loss %.4f val_loss %.4f (%.1fs)", epoch, running / max(1, len(tl.dataset)),
                 val_loss, time.time() - t0)
        if val_loss < best_loss:
            best_loss, bad_epochs = val_loss, 0
            if best_path:
                torch.save(model.state_dict(), best_path)
        else:
            bad_epochs += 1
        if last:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "epoch": epoch, "best_loss": best_loss, "bad_epochs": bad_epochs}, last)

    if best_path and best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=dev, weights_only=True))
    test_loader = DataLoader(ImageDataset(test_paths, None, _transforms(cfg.image_size, False)),
                             batch_size=cfg.batch_size * 2, num_workers=cfg.num_workers, pin_memory=pin)
    out = {"val_proba": _predict(model, vl, dev, amp_dtype), "test_proba": _predict(model, test_loader, dev, amp_dtype),
           "best_val_loss": best_loss,
           "peak_gpu_mb": torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else 0.0}
    del model, opt
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return out
