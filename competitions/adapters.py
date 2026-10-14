"""Task adapters: the seam between competition data and model training.

TabularAdapter is the production path (onboarding -> profiler -> experiments.trainer). VisionAdapter wraps
models.vision for image classification: it trains on the same frozen folds and writes the same artifacts
(oof.npy / test.npy / result.json shapes), so tracking, ensembling and submissions can consume it.
Not yet done: onboarding does not detect image competitions, and the worker/scheduler still call the
tabular trainer directly; routing experiments through get_adapter() is the next integration step.
"""
from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class TaskAdapter(Protocol):
    name: str

    def model_families(self) -> list[str]: ...
    def run_cv(self, ctx: Any, spec: dict[str, Any], device: str, save_dir) -> Any: ...


class TabularAdapter:
    name = "tabular"

    def model_families(self) -> list[str]:
        from models.tabular import FAMILIES
        return sorted(FAMILIES)

    def run_cv(self, ctx, spec, device, save_dir):
        from experiments.trainer import cross_validate
        return cross_validate(ctx, spec, spec.get("params", {}), device=device, save_dir=save_dir)


class VisionAdapter:
    """Image classification: config['vision'] = {image_dir, image_column, extension, test_image_dir}."""

    name = "vision"

    def model_families(self) -> list[str]:
        from models.vision import BACKBONES
        return sorted(BACKBONES)

    def run_cv(self, ctx, spec, device, save_dir):
        from evaluation.validation import fold_ids, fold_split
        from experiments.trainer import CVResult
        from models.vision import VisionConfig, image_paths, train_fold

        vcfg = ctx.config["vision"]
        train_paths = image_paths(ctx.ws.data_dir, vcfg, test=False)
        test_paths = image_paths(ctx.ws.data_dir, vcfg, test=True)
        y, k = ctx.data.y, max(ctx.data.n_classes, 2)
        cfg = VisionConfig(**{**spec.get("params", {}), "backbone": spec["model"], "num_classes": k,
                              "seed": spec["seed"]})
        oof = np.full((len(y), k), np.nan)
        test_sum = np.zeros((len(test_paths), k))
        scores = []
        folds = fold_ids(ctx.folds)
        for f in folds:
            tr, va = fold_split(ctx.folds, f, ctx.scheme["strategy"])
            out = train_fold(cfg, [train_paths[i] for i in tr], y[tr], [train_paths[i] for i in va], y[va],
                             test_paths, device=device, ckpt_dir=save_dir / f"fold_{f}")
            oof[va] = out["val_proba"]
            test_sum += out["test_proba"]
            scores.append(ctx.score(y[va], out["val_proba"][:, 1] if ctx.task == "binary" else out["val_proba"]))
        mask = ~np.isnan(oof).any(axis=1)
        oof_out = oof[:, 1] if ctx.task == "binary" else oof
        test_out = test_sum / len(folds)
        test_out = test_out[:, 1] if ctx.task == "binary" else test_out
        cv = ctx.score(y[mask], oof_out[mask])
        return CVResult(oof_out, test_out, None, scores, float(cv), float(np.std(scores)), None, 0.5, [], {},
                        0, cfg.__dict__)


ADAPTERS = {"tabular": TabularAdapter, "vision": VisionAdapter}


def get_adapter(config: dict[str, Any]) -> TaskAdapter:
    return ADAPTERS[config.get("task_adapter", "tabular")]()
