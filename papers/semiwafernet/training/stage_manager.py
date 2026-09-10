"""Three-stage SSL schedule (paper Section 2.2).

Stage 1 warm-up -> Stage 2/3 offline MC pseudo on Du -> train on Dl U D_pseudo.

Gates follow Eq. 10-13. Author defaults: tau_base=0.94, alpha=0.08, beta=0.02,
eps_H=0.08, eps_MI=0.12 (raw nats). On our teachers these defaults flood D_pseudo
with 'none'; we (1) re-calibrate eps_H / prior scale on the held-out
pseudo-eval split (paper Section 4.1) for Macro-F1 of accepted labels, and
(2) cap 'none' in D_pseudo so SSL does not undo SMOTE balance on Dl.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.data import Dataset, TensorDataset
from tqdm import tqdm

from papers.semiwafernet.training.adaptive_threshold import AdaptiveThreshold
from papers.semiwafernet.training.ema import EMATeacher
from papers.semiwafernet.training.mc_dropout import MonteCarloDropout, enable_mc_dropout
from papers.semiwafernet.training.uncertainty import UncertaintyFilter


def _macro_f1(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    f1s: list[float] = []
    pred = pred.view(-1).long()
    target = target.view(-1).long()
    for c in range(num_classes):
        tp = int(((pred == c) & (target == c)).sum().item())
        fp = int(((pred == c) & (target != c)).sum().item())
        fn = int(((pred != c) & (target == c)).sum().item())
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1s.append(2.0 * prec * rec / (prec + rec + 1e-12))
    return float(sum(f1s) / max(len(f1s), 1))


class StageManager:
    """Three-stage semi-supervised training workflow manager."""

    def __init__(
        self,
        student: nn.Module,
        num_classes: int = 9,
        ema_decay: float = 0.999,
        base_threshold: float = 0.94,
        alpha: float = 0.08,
        beta: float = 0.02,
        mc_passes: int = 20,
        entropy_threshold: float = 0.08,
        mi_threshold: float = 0.12,
        consistency_weight: float = 0.0,
        logit_bias: torch.Tensor | None = None,
        max_none_to_defect_ratio: float = 1.0,
        **_unused: Any,
    ) -> None:
        self.student = student
        self.num_classes = num_classes
        self.consistency_weight = consistency_weight
        self.base_threshold = base_threshold
        self.max_none_to_defect_ratio = float(max_none_to_defect_ratio)
        self.current_stage: int = 1
        self._base_logit_bias: torch.Tensor | None = None
        self.logit_bias: torch.Tensor | None = None
        self.ssl_prior_scale: float = 1.0
        if logit_bias is not None:
            self.register_logit_bias(logit_bias, scale=1.0)
        self.teacher = EMATeacher(student, momentum=ema_decay)
        self.adaptive_threshold = AdaptiveThreshold(
            num_classes=num_classes,
            base_threshold=base_threshold,
            alpha=alpha,
            beta=beta,
        )
        self.mc_dropout = MonteCarloDropout(num_passes=mc_passes)
        self.uncertainty_filter = UncertaintyFilter(
            entropy_threshold=entropy_threshold,
            mi_threshold=mi_threshold,
            normalize_entropy=False,
            num_classes=num_classes,
        )
        self._paper_eps_h = float(entropy_threshold)

    def register_logit_bias(
        self, logit_bias: torch.Tensor, scale: float = 1.0
    ) -> None:
        """Register natural log-pi; ``scale`` softens bias used at SSL time."""
        bias = logit_bias.detach().float().view(-1)
        if bias.numel() != self.num_classes:
            raise ValueError(
                f"logit_bias must have {self.num_classes} elements, got {bias.numel()}"
            )
        self._base_logit_bias = bias
        self.ssl_prior_scale = float(scale)
        self.logit_bias = (scale * bias) if scale != 0.0 else None

    def set_ssl_prior_scale(self, scale: float) -> None:
        if self._base_logit_bias is None:
            self.logit_bias = None
            self.ssl_prior_scale = 0.0
            return
        self.ssl_prior_scale = float(scale)
        if abs(self.ssl_prior_scale) < 1e-12:
            self.logit_bias = None
        else:
            self.logit_bias = self.ssl_prior_scale * self._base_logit_bias

    def set_stage(self, stage: int) -> None:
        if stage not in (1, 2, 3):
            raise ValueError(f"Invalid stage: {stage}. Must be 1, 2, or 3.")
        self.current_stage = stage

    def get_stage(self) -> int:
        return self.current_stage

    def is_semi_supervised(self) -> bool:
        return self.current_stage in (2, 3)

    def install_teacher_from_student(self) -> None:
        """Copy current student weights into the teacher (paper Section 2.2)."""
        self.teacher.teacher.load_state_dict(self.student.state_dict(), strict=True)
        for t_buf, s_buf in zip(self.teacher.teacher.buffers(), self.student.buffers()):
            t_buf.copy_(s_buf)
        self.teacher.teacher.eval()

    def generate_pseudo_labels(
        self, unlabeled_x: torch.Tensor
    ) -> dict[str, torch.Tensor | float]:
        """Generate pseudo-labels for a single batch (tests / debugging)."""
        bias = None
        if self.logit_bias is not None:
            bias = self.logit_bias.to(device=unlabeled_x.device)
        mc_results = self.mc_dropout(
            self.teacher.teacher, unlabeled_x, logit_bias=bias, temperature=1.0
        )
        class_probs = mc_results["mean_probs_class"]
        class_confidence, pseudo_class_labels = class_probs.max(dim=1)
        self.adaptive_threshold.update_statistics(
            confidence=class_confidence,
            pseudo_labels=pseudo_class_labels,
        )
        adaptive_tau_class = self.adaptive_threshold.compute_threshold(
            pseudo_labels=pseudo_class_labels,
            entropy=mc_results["entropy_class"],
        )
        filter_masks = self.uncertainty_filter(
            confidence_class=class_confidence,
            confidence_seg=class_confidence,
            adaptive_threshold=adaptive_tau_class,
            adaptive_threshold_seg=adaptive_tau_class,
            entropy_class=mc_results["entropy_class"],
            entropy_seg=mc_results["entropy_class"],
            mutual_info_class=mc_results["mutual_info_class"],
            mutual_info_seg=mc_results["mutual_info_class"],
        )
        B = unlabeled_x.shape[0]
        H, W = unlabeled_x.shape[-2], unlabeled_x.shape[-1]
        dummy_seg = torch.zeros(B, H, W, dtype=torch.long, device=unlabeled_x.device)
        dummy_mask = torch.zeros(B, H, W, dtype=torch.bool, device=unlabeled_x.device)
        return {
            "pseudo_labels_class": pseudo_class_labels,
            "pseudo_labels_seg": dummy_seg,
            "mask_class": filter_masks["classification"],
            "mask_seg": dummy_mask,
            "confidence_class": class_confidence,
            "confidence_seg": torch.zeros(B, H, W, device=unlabeled_x.device),
            "adaptive_threshold": adaptive_tau_class.mean().item(),
        }

    @torch.no_grad()
    def _mc_collect(
        self,
        loader: Any,
        device: torch.device,
        bias: torch.Tensor | None,
        desc: str,
        verbose: bool,
        with_labels: bool = False,
    ) -> dict[str, torch.Tensor]:
        images: list[torch.Tensor] = []
        confs: list[torch.Tensor] = []
        preds: list[torch.Tensor] = []
        ents: list[torch.Tensor] = []
        mis: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []

        iterator = loader
        if verbose:
            total = len(loader) if hasattr(loader, "__len__") else None
            iterator = tqdm(loader, desc=desc, total=total, unit="batch")

        for batch in iterator:
            if with_labels:
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    inputs_u, y = batch[0], batch[1]
                elif isinstance(batch, dict):
                    inputs_u, y = batch["image"], batch["label"]
                else:
                    raise TypeError(f"Expected (x,y) batch, got {type(batch)}")
                labels.append(y.detach().cpu().view(-1).long())
            else:
                inputs_u = batch
                if isinstance(inputs_u, (list, tuple)):
                    inputs_u = inputs_u[0]
            inputs_u = inputs_u.to(device)

            enable_mc_dropout(self.teacher.teacher)
            pass_probs: list[torch.Tensor] = []
            for _ in range(self.mc_dropout.num_passes):
                logits = self.teacher.teacher(inputs_u)["classification"]
                if bias is not None:
                    logits = logits + bias.to(device=logits.device, dtype=logits.dtype)
                pass_probs.append(torch.softmax(logits, dim=1))
            self.teacher.teacher.eval()

            probs_stack = torch.stack(pass_probs, dim=0)
            mean_probs = probs_stack.mean(dim=0)
            conf, pred = mean_probs.max(dim=1)
            ent = MonteCarloDropout._entropy(mean_probs)
            per_pass_ent = torch.stack(
                [MonteCarloDropout._entropy(pass_probs[i]) for i in range(len(pass_probs))],
                dim=0,
            )
            mi = (ent - per_pass_ent.mean(dim=0)).clamp(min=0.0)

            images.append(inputs_u.detach().cpu())
            confs.append(conf.detach().cpu())
            preds.append(pred.detach().cpu())
            ents.append(ent.detach().cpu())
            mis.append(mi.detach().cpu())

        out: dict[str, torch.Tensor] = {
            "x": torch.cat(images, dim=0),
            "conf": torch.cat(confs, dim=0),
            "pred": torch.cat(preds, dim=0),
            "ent": torch.cat(ents, dim=0),
            "mi": torch.cat(mis, dim=0),
        }
        if with_labels:
            out["y"] = torch.cat(labels, dim=0)
        return out

    def _apply_gates(
        self,
        conf: torch.Tensor,
        pred: torch.Tensor,
        ent: torch.Tensor,
        mi: torch.Tensor,
        eps_h: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mask, tau) for Eq. 13 with optional eps_H override."""
        old_eps = self.uncertainty_filter.entropy_threshold
        if eps_h is not None:
            self.uncertainty_filter.entropy_threshold = float(eps_h)
        self.adaptive_threshold.reset()
        self.adaptive_threshold.update_statistics(confidence=conf, pseudo_labels=pred)
        tau = self.adaptive_threshold.compute_threshold(pseudo_labels=pred, entropy=ent)
        mask = self.uncertainty_filter.filter_classification(
            confidence=conf,
            adaptive_threshold=tau,
            entropy=ent,
            mutual_information=mi,
        )
        self.uncertainty_filter.entropy_threshold = old_eps
        return mask, tau

    def _cap_none_in_pseudo(
        self,
        mask: torch.Tensor,
        pred: torch.Tensor,
        conf: torch.Tensor,
    ) -> torch.Tensor:
        """Keep all accepted defects; cap none so SSL does not undo SMOTE.

        none_keep <= max_none_to_defect_ratio * n_defect_accepted
        (default ratio=1.0 -> none cannot exceed defects in D_pseudo).
        """
        if not mask.any():
            return mask
        idx = mask.nonzero(as_tuple=False).view(-1)
        pred_a = pred[idx]
        conf_a = conf[idx]
        defect = pred_a != 0
        n_def = int(defect.sum().item())
        none_idx = idx[~defect]
        if none_idx.numel() == 0:
            return mask
        if n_def == 0:
            # No defects passed gates — keep a small high-conf none set only.
            k = min(int(none_idx.numel()), max(256, none_idx.numel() // 20))
            keep_none = none_idx[torch.topk(conf[none_idx], k=k).indices]
        else:
            k = max(1, int(self.max_none_to_defect_ratio * n_def))
            k = min(k, int(none_idx.numel()))
            keep_none = none_idx[torch.topk(conf[none_idx], k=k).indices]
        keep_def = idx[defect]
        kept = torch.cat([keep_def, keep_none], dim=0) if keep_def.numel() else keep_none
        out = torch.zeros_like(mask)
        out[kept] = True
        return out

    @torch.no_grad()
    def calibrate_on_pseudo_eval(
        self,
        pe_loader: Any,
        device: torch.device,
        prior_scales: list[float] | None = None,
        eps_grid: list[float] | None = None,
        verbose: bool = True,
    ) -> dict[str, float]:
        """Paper Section 4.1: pick eps_H / prior scale on held-out pe by Macro-F1."""
        if pe_loader is None:
            return {
                "eps_h": float(self.uncertainty_filter.entropy_threshold),
                "prior_scale": float(self.ssl_prior_scale),
                "macro_f1": float("nan"),
            }

        prior_scales = prior_scales or [0.0, 0.25, 0.5, 0.75, 1.0]
        eps_grid = eps_grid or [
            self._paper_eps_h,
            self._paper_eps_h * 1.5,
            self._paper_eps_h * 2.0,
            self._paper_eps_h * 3.0,
            self._paper_eps_h * 4.0,
        ]

        self.teacher.teacher.to(device)
        self.teacher.teacher.eval()
        best: dict[str, float] | None = None

        for scale in prior_scales:
            if self._base_logit_bias is None:
                bias = None
            elif abs(scale) < 1e-12:
                bias = None
            else:
                bias = (scale * self._base_logit_bias).to(device)

            # One MC pass per prior scale (pe is small).
            pack = self._mc_collect(
                pe_loader,
                device,
                bias,
                desc=f"[SSL] pe-calibrate prior={scale:.2f}",
                verbose=verbose,
                with_labels=True,
            )
            y_true = pack["y"]
            for eps_h in eps_grid:
                mask, tau = self._apply_gates(
                    pack["conf"], pack["pred"], pack["ent"], pack["mi"], eps_h=eps_h
                )
                n_acc = int(mask.sum().item())
                if n_acc < max(20, self.num_classes * 2):
                    continue
                f1 = _macro_f1(pack["pred"][mask], y_true[mask], self.num_classes)
                rate = n_acc / max(int(pack["pred"].numel()), 1)
                # Prefer Macro-F1; mild bonus for higher accept (paper ~0.8).
                score = f1 + 0.05 * min(rate, 0.75)
                # Prefer paper eps when scores are close.
                if abs(eps_h - self._paper_eps_h) < 1e-9:
                    score += 0.01
                cand = {
                    "eps_h": float(eps_h),
                    "prior_scale": float(scale),
                    "macro_f1": float(f1),
                    "accept_rate": float(rate),
                    "n_accepted": float(n_acc),
                    "mean_tau": float(tau.mean().item()),
                    "score": float(score),
                }
                if best is None or cand["score"] > best["score"]:
                    best = cand

        if best is None:
            if verbose:
                print("[SSL] pe-calibration found no viable gate; keeping paper defaults")
            return {
                "eps_h": float(self._paper_eps_h),
                "prior_scale": float(self.ssl_prior_scale),
                "macro_f1": float("nan"),
            }

        self.uncertainty_filter.entropy_threshold = best["eps_h"]
        self.set_ssl_prior_scale(best["prior_scale"])
        if verbose:
            print(
                f"[SSL] pe-calibrated: eps_H={best['eps_h']:.3f}, "
                f"prior_scale={best['prior_scale']:.2f}, "
                f"Macro-F1(accepted)={best['macro_f1']:.4f}, "
                f"accept={100 * best['accept_rate']:.1f}%"
            )
        return best

    @torch.no_grad()
    def build_pseudo_dataset(
        self,
        unlabeled_loader: Any,
        device: torch.device,
        verbose: bool = True,
    ) -> tuple[Dataset | None, dict[str, float]]:
        """Offline pseudo-label generation on Du (paper Section 2.2 / Eq. 13)."""
        self.teacher.teacher.to(device)
        self.teacher.teacher.eval()
        bias = self.logit_bias.to(device) if self.logit_bias is not None else None

        pack = self._mc_collect(
            unlabeled_loader,
            device,
            bias,
            desc=f"[SSL Stage {self.current_stage}] MC pseudo-labels",
            verbose=verbose,
            with_labels=False,
        )
        all_x = pack["x"]
        all_conf = pack["conf"]
        all_y = pack["pred"]
        all_ent = pack["ent"]
        all_mi = pack["mi"]
        n_total = int(all_x.shape[0])

        final_mask, tau = self._apply_gates(all_conf, all_y, all_ent, all_mi)
        n_hard = int(final_mask.sum().item())
        final_mask = self._cap_none_in_pseudo(final_mask, all_y, all_conf)
        n_accepted = int(final_mask.sum().item())

        stats = {
            "n_total": float(n_total),
            "n_accepted": float(n_accepted),
            "n_hard": float(n_hard),
            "accept_rate": 100.0 * n_accepted / max(n_total, 1),
            "hard_accept_rate": 100.0 * n_hard / max(n_total, 1),
            "mean_tau": float(tau.mean().item()) if torch.is_tensor(tau) else float(tau),
            "mean_conf_accepted": (
                float(all_conf[final_mask].mean().item()) if n_accepted else 0.0
            ),
            "selection_temperature": 1.0,
            "selection_eps_h": float(self.uncertainty_filter.entropy_threshold),
            "ssl_prior_scale": float(self.ssl_prior_scale),
        }
        if n_accepted == 0:
            if verbose:
                print(
                    f"[SSL] Pseudo-set empty: 0/{n_total} passed gates "
                    f"(mean conf={float(all_conf.mean()):.3f}, "
                    f"mean H={float(all_ent.mean()):.3f}, "
                    f"mean tau={stats['mean_tau']:.3f})."
                )
            return None, stats

        if verbose and n_hard != n_accepted:
            print(
                f"[SSL] none-cap: {n_hard} -> {n_accepted} "
                f"(max_none_to_defect_ratio={self.max_none_to_defect_ratio:.2f})"
            )

        pseudo_ds = TensorDataset(all_x[final_mask], all_y[final_mask].long())
        if verbose:
            hist = torch.bincount(all_y[final_mask], minlength=self.num_classes).tolist()
            print(
                f"[SSL] Pseudo-set ready: {n_accepted}/{n_total} "
                f"({stats['accept_rate']:.1f}%), hard={stats['hard_accept_rate']:.1f}%, "
                f"eps_H={stats['selection_eps_h']:.3f}, "
                f"prior={stats['ssl_prior_scale']:.2f}, "
                f"mean tau={stats['mean_tau']:.3f}"
            )
            print(f"[SSL] Pseudo class counts: {hist}")
        return pseudo_ds, stats

    def refresh_teacher(self) -> None:
        self.install_teacher_from_student()

    def reset_statistics(self) -> None:
        self.adaptive_threshold.reset()
