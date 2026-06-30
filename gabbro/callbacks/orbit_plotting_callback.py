"""Schema-neutral ORBIT plotting callback."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import lightning as L
import numpy as np
import awkward as ak
import fastjet
import vector

from gabbro.plotting.orbit import (
    close_figure,
    collect_reconstruction_histograms,
    collect_physical_reconstruction_histograms,
    physical_reconstruction_plots,
    plot_codebook_histogram,
    plot_feature_histograms,
    plot_residual_histograms,
)
from gabbro.utils.pylogger import get_pylogger

logger = get_pylogger("OrbitPlottingCallback")
vector.register_awkward()


def _delta_r(particles, jets):
    jets = ak.unflatten(ak.flatten(jets), counts=1)
    return particles.deltaR(jets)


class OrbitPlottingCallback(L.Callback):
    """Plot masked reconstruction summaries for particle or jet ORBIT parquet batches."""

    def __init__(
        self,
        image_path: str | None = None,
        image_filetype: str = "png",
        save_histograms: bool = True,
        no_trainer_info_in_filename: bool = False,
        enable_physics_plots: bool = True,
        include_all_ratios: bool = True,
        jet_radius: float = 0.8,
        include_codebook_histogram: bool = True,
    ):
        super().__init__()
        self.image_path = image_path
        self.image_filetype = image_filetype
        self.save_histograms = save_histograms
        self.no_trainer_info_in_filename = no_trainer_info_in_filename
        self.enable_physics_plots = enable_physics_plots
        self.include_all_ratios = include_all_ratios
        self.jet_radius = jet_radius
        self.include_codebook_histogram = include_codebook_histogram

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        pl_module.concat_validation_loop_predictions()
        self.plot(trainer, pl_module, stage="val")

    def on_test_epoch_end(self, trainer, pl_module):
        pl_module.concat_test_loop_predictions()
        self.plot(trainer, pl_module, stage="test")

    def _feature_names(self, trainer, pl_module) -> list[str]:
        datamodule_hparams = trainer.datamodule.hparams
        if "selected_features" in datamodule_hparams:
            return list(datamodule_hparams.selected_features)
        if "dataset_kwargs_common" in datamodule_hparams:
            return list(datamodule_hparams.dataset_kwargs_common.feature_dict.keys())
        return [f"feature_{i}" for i in range(pl_module.val_x_original_concat.shape[-1])]

    @staticmethod
    def _transformed_feature_alias(feature_name: str, feature_index: int) -> str:
        aliases = {
            "L1T_PUPPIPart_Eta": "eta_scaled",
            "L1T_PUPPIPart_Phi_cos": "cos_phi",
            "L1T_PUPPIPart_Phi_sin": "sin_phi",
            "L1T_PUPPIPart_PT": "log_pt_shifted",
        }
        return aliases.get(
            feature_name,
            feature_name.replace("/", "_").replace(" ", "_") or f"feature_{feature_index}",
        )

    def _plot_dir(self, trainer) -> Path:
        plot_dir = (
            Path(self.image_path)
            if self.image_path is not None
            else Path(trainer.default_root_dir) / "plots"
        )
        plot_dir.mkdir(parents=True, exist_ok=True)
        return plot_dir

    def _figure_name(self, trainer, stage: str, name: str) -> str:
        if self.no_trainer_info_in_filename:
            return f"{stage}_{name}.{self.image_filetype}"
        if stage == "val":
            return (
                f"val_epoch{trainer.current_epoch}_"
                f"gstep{trainer.global_step}_{name}.{self.image_filetype}"
            )
        return f"test_{name}.{self.image_filetype}"

    @staticmethod
    def _wandb_figure_key(name: str) -> str:
        parts = [part.strip().replace(" ", "_") for part in name.split("/") if part.strip()]
        if len(parts) >= 3 and parts[0] in {"val", "test"}:
            stage, group, plot_name = parts[0], parts[1], "/".join(parts[2:])
            return f"{stage}_plots/{group}/{plot_name}"
        return f"plots/{'_'.join(parts)}"

    def _log_figure(self, trainer, path: Path, name: str) -> None:
        logged = False
        wandb_key = self._wandb_figure_key(name)
        for lightning_logger in trainer.loggers:
            if isinstance(lightning_logger, L.pytorch.loggers.CometLogger):
                lightning_logger.experiment.log_image(
                    str(path),
                    name=name,
                    step=trainer.global_step,
                )
                logged = True
            elif isinstance(lightning_logger, L.pytorch.loggers.WandbLogger):
                try:
                    import wandb

                    lightning_logger.experiment.log(
                        {
                            wandb_key: wandb.Image(
                                str(path),
                                caption=name,
                            )
                        },
                        commit=False,
                    )
                    logged = True
                except Exception as exc:
                    logger.warning(f"Failed to log {path} to W&B: {exc}")
        if logged:
            logger.info(f"Logged ORBIT figure {name} from {path}")
        else:
            logger.info(f"Saved ORBIT figure {name} to {path}; no image logger was active")

    def _log_metrics(
        self,
        trainer,
        stage: str,
        group_name: str,
        metrics: dict[str, float | int | None],
    ) -> None:
        prefix = f"{stage}_metrics" if group_name == "all" else f"{stage}_metrics/{group_name}"
        scalar_metrics = {}
        for name, value in metrics.items():
            if value is None:
                continue
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, (float, int)):
                scalar_metrics[f"{prefix}/{name}"] = value

        if not scalar_metrics:
            return

        for lightning_logger in trainer.loggers:
            lightning_logger.log_metrics(scalar_metrics)
        logger.info(
            f"Logged {len(scalar_metrics)} ORBIT scalar metrics for {stage}/{group_name}"
        )

    def _num_codes(self, pl_module) -> int | None:
        vqlayer = getattr(pl_module.model, "vqlayer", None)
        if vqlayer is None:
            return None
        return getattr(vqlayer, "num_codes", None)

    def _data_level(self, trainer) -> str:
        sequence_type = getattr(trainer.datamodule.hparams, "sequence_type", "particle")
        return "particle" if sequence_type == "particle" else "jet"

    @staticmethod
    def _jet_radius_from_sequence_type(sequence_type: str | None, default: float) -> float:
        if sequence_type and "ak4" in sequence_type.lower():
            return 0.4
        if sequence_type and "ak8" in sequence_type.lower():
            return 0.8
        return default

    def _jet_radii_for_group(
        self,
        trainer,
        group_name: str,
        labels: np.ndarray,
    ) -> tuple[np.ndarray, str]:
        class_specs = getattr(trainer.datamodule.hparams, "class_specs", None)
        if not class_specs:
            return np.full(labels.shape[0], self.jet_radius), f"{self.jet_radius:g}"

        class_specs = dict(class_specs)
        if group_name != "all":
            class_spec = dict(class_specs).get(group_name, {})
            sequence_type = class_spec.get("eval_sequence_type") or class_spec.get("sequence_type")
            radius = self._jet_radius_from_sequence_type(sequence_type, self.jet_radius)
            return np.full(labels.shape[0], radius), f"{radius:g}"

        class_to_label = getattr(trainer.datamodule.hparams, "class_to_label", {})
        label_to_class = {int(label): name for name, label in dict(class_to_label).items()}
        radii = np.full(labels.shape[0], self.jet_radius)
        for label, class_name in label_to_class.items():
            class_spec = dict(class_specs).get(class_name, {})
            sequence_type = class_spec.get("eval_sequence_type") or class_spec.get("sequence_type")
            radii[labels == label] = self._jet_radius_from_sequence_type(
                sequence_type,
                self.jet_radius,
            )
        return radii, "class-specific"

    def _to_physical_features(self, x: np.ndarray) -> np.ndarray:
        """Convert `[eta/3, cos(phi), sin(phi), log(pT)-1.8]` to `[eta, phi, pT]`."""
        eta = x[..., 0] * 3.0
        phi = np.arctan2(x[..., 2], x[..., 1])
        pt = np.exp(x[..., 3] + 1.8) - 1e-8
        pt = np.clip(pt, a_min=0.0, a_max=None)
        return np.stack([eta, phi, pt], axis=-1)

    def _collect_missing_transverse_energy(
        self,
        x_physical: np.ndarray,
        x_reco_physical: np.ndarray,
        mask: np.ndarray,
    ):
        def missing_et(events):
            pts = np.where(mask, events[..., 2], 0.0)
            phis = events[..., 1]
            px = np.sum(pts * np.cos(phis), axis=1)
            py = np.sum(pts * np.sin(phis), axis=1)
            return np.hypot(px, py)

        return missing_et(x_physical), missing_et(x_reco_physical)

    def _collect_direct_jet_metrics(
        self,
        x_physical: np.ndarray,
        x_reco_physical: np.ndarray,
        mask: np.ndarray,
    ):
        true_jet_pts, reco_jet_pts = [], []
        true_jet_etas, reco_jet_etas = [], []
        true_jet_phis, reco_jet_phis = [], []
        for i in range(x_physical.shape[0]):
            event_mask = mask[i]
            true_vals = x_physical[i, event_mask]
            reco_vals = x_reco_physical[i, event_mask]
            n_match = min(len(true_vals), len(reco_vals))
            if n_match:
                true_jet_pts.extend(true_vals[:n_match, 2])
                reco_jet_pts.extend(reco_vals[:n_match, 2])
                true_jet_etas.extend(true_vals[:n_match, 0])
                reco_jet_etas.extend(reco_vals[:n_match, 0])
                true_jet_phis.extend(true_vals[:n_match, 1])
                reco_jet_phis.extend(reco_vals[:n_match, 1])
        return (
            true_jet_pts,
            reco_jet_pts,
            true_jet_etas,
            reco_jet_etas,
            true_jet_phis,
            reco_jet_phis,
        )

    def _reconstruct_event_jets(
        self,
        pt: np.ndarray,
        eta: np.ndarray,
        phi: np.ndarray,
        jet_radius: float,
    ):
        pt = np.asarray(pt, dtype=np.float64)
        eta = np.asarray(eta, dtype=np.float64)
        phi = np.asarray(phi, dtype=np.float64)
        if len(pt) < 3 or np.sum(pt) <= 0:
            return {
                "pt": np.array([]),
                "jet_mass": 0.0,
                "tau32": 0.0,
                "jet_n_constituents": len(pt),
                "jet_count": 0,
            }

        particles = ak.zip(
            {"pt": [pt], "eta": [eta], "phi": [phi], "mass": [np.zeros_like(pt)]},
            with_name="Momentum4D",
        )
        particles_sum = ak.sum(particles, axis=1)
        jetdef = fastjet.JetDefinition(fastjet.kt_algorithm, jet_radius)
        cluster = fastjet.ClusterSequence(particles, jetdef)
        inclusive_jets = cluster.inclusive_jets(min_pt=0.0)
        d0 = ak.sum(particles.pt * jet_radius, axis=1)

        exclusive_jets_1 = cluster.exclusive_jets(n_jets=1)
        exclusive_jets_2 = cluster.exclusive_jets(n_jets=2)
        exclusive_jets_3 = cluster.exclusive_jets(n_jets=3)

        dr_1i = _delta_r(particles, exclusive_jets_1[:, :1])
        tau1 = ak.sum(particles.pt * dr_1i, axis=1) / d0

        dr_1i_t2 = _delta_r(particles, exclusive_jets_2[:, :1])
        dr_2i_t2 = _delta_r(particles, exclusive_jets_2[:, 1:2])
        min_dr_t2 = ak.min(
            ak.concatenate(
                [dr_1i_t2[..., np.newaxis], dr_2i_t2[..., np.newaxis]],
                axis=-1,
            ),
            axis=-1,
        )
        tau2 = ak.sum(particles.pt * min_dr_t2, axis=1) / d0

        dr_1i_t3 = _delta_r(particles, exclusive_jets_3[:, :1])
        dr_2i_t3 = _delta_r(particles, exclusive_jets_3[:, 1:2])
        dr_3i_t3 = _delta_r(particles, exclusive_jets_3[:, 2:3])
        min_dr_t3 = ak.min(
            ak.concatenate(
                [
                    dr_1i_t3[..., np.newaxis],
                    dr_2i_t3[..., np.newaxis],
                    dr_3i_t3[..., np.newaxis],
                ],
                axis=-1,
            ),
            axis=-1,
        )
        tau3 = ak.sum(particles.pt * min_dr_t3, axis=1) / d0
        tau32 = np.nan_to_num(float((tau3 / (tau2 + 1e-8))[0]))

        return {
            "pt": (
                np.asarray(inclusive_jets.pt[0])
                if len(inclusive_jets[0]) > 0
                else np.array([])
            ),
            "jet_mass": float(particles_sum.mass[0]),
            "tau32": tau32,
            "jet_n_constituents": len(pt),
            "jet_count": len(inclusive_jets[0]),
        }

    def _collect_particle_jet_metrics(
        self,
        x_physical: np.ndarray,
        x_reco_physical: np.ndarray,
        mask: np.ndarray,
        jet_radii: np.ndarray,
    ):
        true_jet_pts = []
        reco_jet_pts = []
        true_jet_masses = []
        reco_jet_masses = []
        true_tau32s = []
        reco_tau32s = []
        true_jet_counts = []
        reco_jet_counts = []

        for i in range(x_physical.shape[0]):
            event_mask = mask[i]
            jet_radius = float(jet_radii[i])
            true_jets = self._reconstruct_event_jets(
                x_physical[i, event_mask, 2],
                x_physical[i, event_mask, 0],
                x_physical[i, event_mask, 1],
                jet_radius,
            )
            reco_jets = self._reconstruct_event_jets(
                x_reco_physical[i, event_mask, 2],
                x_reco_physical[i, event_mask, 0],
                x_reco_physical[i, event_mask, 1],
                jet_radius,
            )
            true_jet_counts.append(true_jets["jet_count"])
            reco_jet_counts.append(reco_jets["jet_count"])
            n_match = min(len(true_jets["pt"]), len(reco_jets["pt"]))
            if n_match:
                true_jet_pts.extend(true_jets["pt"][:n_match])
                reco_jet_pts.extend(reco_jets["pt"][:n_match])
            if (
                true_jets["jet_n_constituents"] >= 3
                and reco_jets["jet_n_constituents"] >= 3
            ):
                true_jet_masses.append(true_jets["jet_mass"])
                reco_jet_masses.append(reco_jets["jet_mass"])
                true_tau32s.append(true_jets["tau32"])
                reco_tau32s.append(reco_jets["tau32"])

        return (
            true_jet_pts,
            reco_jet_pts,
            true_jet_masses,
            reco_jet_masses,
            true_tau32s,
            reco_tau32s,
            true_jet_counts,
            reco_jet_counts,
        )

    def _plot_jet_count_difference(
        self,
        true_jet_counts,
        reco_jet_counts,
        title: str,
    ):
        true_jet_counts = np.asarray(true_jet_counts, dtype=int)
        reco_jet_counts = np.asarray(reco_jet_counts, dtype=int)
        diff = reco_jet_counts - true_jet_counts
        max_count = int(max(true_jet_counts.max(initial=0), reco_jet_counts.max(initial=0)))
        count_bins = np.arange(-0.5, max_count + 1.5, 1)
        diff_abs = int(max(abs(diff.min(initial=0)), abs(diff.max(initial=0))))
        diff_bins = np.arange(-diff_abs - 0.5, diff_abs + 1.5, 1)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(title)
        axes[0].hist(
            true_jet_counts,
            bins=count_bins,
            histtype="stepfilled",
            alpha=0.35,
            label="Original",
        )
        axes[0].hist(
            reco_jet_counts,
            bins=count_bins,
            histtype="step",
            linewidth=2,
            label="Reconstructed",
        )
        axes[0].set_xlabel("Reclustered jet count")
        axes[0].set_ylabel("Events")
        axes[0].legend()
        axes[1].hist(
            diff,
            bins=diff_bins,
            histtype="stepfilled",
            alpha=0.45,
        )
        axes[1].axvline(0, color="black", linestyle="--", alpha=0.5)
        axes[1].set_xlabel("Reconstructed - original jet count")
        axes[1].set_ylabel("Events")
        fig.tight_layout()
        return fig

    def plot(self, trainer, pl_module, stage: str) -> None:
        if stage == "val" and not hasattr(pl_module, "val_x_original_concat"):
            logger.info("No validation predictions found. Skipping ORBIT plots.")
            return
        if stage == "test" and not hasattr(pl_module, "test_x_original_concat"):
            logger.info("No test predictions found. Skipping ORBIT plots.")
            return

        if stage == "val":
            x_original = pl_module.val_x_original_concat
            x_reco = pl_module.val_x_reco_concat
            mask = pl_module.val_mask_concat.astype(bool)
            code_idx = pl_module.val_code_idx_concat
        else:
            x_original = pl_module.test_x_original_concat
            x_reco = pl_module.test_x_reco_concat
            mask = pl_module.test_mask_concat.astype(bool)
            labels = pl_module.test_labels_concat
            code_idx = pl_module.test_code_idx_concat

        if stage == "val":
            labels = pl_module.val_labels_concat

        groups = [("all", "all", np.ones(labels.shape[0], dtype=bool))]
        class_to_label = getattr(trainer.datamodule.hparams, "class_to_label", None)
        if class_to_label:
            label_to_class = {int(label): name for name, label in dict(class_to_label).items()}
            for label, class_name in sorted(label_to_class.items()):
                groups.append((class_name, class_name, labels == label))

        for group_name, display_name, event_selector in groups:
            if np.any(event_selector):
                self._plot_arrays(
                    trainer=trainer,
                    pl_module=pl_module,
                    stage=stage,
                    group_name=group_name,
                    display_name=display_name,
                    x_original=x_original[event_selector],
                    x_reco=x_reco[event_selector],
                    mask=mask[event_selector],
                    labels=labels[event_selector],
                    code_idx=code_idx[event_selector],
                    preserve_legacy_names=(group_name == "all"),
                )

    def _plot_arrays(
        self,
        trainer,
        pl_module,
        stage: str,
        group_name: str,
        display_name: str,
        x_original: np.ndarray,
        x_reco: np.ndarray,
        mask: np.ndarray,
        labels: np.ndarray,
        code_idx: np.ndarray,
        preserve_legacy_names: bool,
    ) -> None:
        if not np.any(mask):
            logger.warning(f"No valid tokens found for {stage}/{group_name} ORBIT plots.")
            return

        feature_names = self._feature_names(trainer, pl_module)
        plot_dir = self._plot_dir(trainer)
        histograms = collect_reconstruction_histograms(
            feature_names=feature_names,
            x_np=x_original,
            x_hat_np=x_reco,
            mask_np=mask,
        )
        mse_per_feature = np.sum(
            ((x_reco - x_original) ** 2) * mask[..., None],
            axis=(0, 1),
        )
        mse_per_feature /= np.clip(np.sum(mask), a_min=1, a_max=None)
        num_codes = self._num_codes(pl_module)
        active_codes = len(np.unique(code_idx[mask]))
        metrics = {
            f"metrics/mse_{feature_name.replace('/', '_').replace(' ', '_')}": float(mse)
            for feature_name, mse in zip(feature_names, mse_per_feature)
        }
        for feature_index, (feature_name, mse) in enumerate(zip(feature_names, mse_per_feature)):
            alias = self._transformed_feature_alias(feature_name, feature_index)
            metrics[f"metrics/transformed_mse_{alias}"] = float(mse)
            metrics[f"metrics/transformed_rmse_{alias}"] = float(np.sqrt(max(mse, 0.0)))
        metrics.update(
            {
                "metrics/mse_total": float(np.mean(mse_per_feature)),
                "metrics/transformed_mse_total": float(np.mean(mse_per_feature)),
                "metrics/transformed_rmse_mean": float(
                    np.mean(np.sqrt(np.clip(mse_per_feature, a_min=0.0, a_max=None)))
                ),
                "metrics/active_codes_total": int(active_codes),
                "metrics/utilization_total": (
                    float(active_codes / num_codes) if num_codes else None
                ),
                "metrics/total_codebook_size": int(num_codes) if num_codes else None,
            }
        )

        figures = {
            f"{stage}/{group_name}/orbit_reconstruction_features": plot_feature_histograms(
                histograms,
                feature_names,
                mse_per_feature=mse_per_feature,
                title=f"{stage.capitalize()} {display_name} ORBIT reconstruction features",
            ),
            f"{stage}/{group_name}/orbit_reconstruction_residuals": plot_residual_histograms(
                histograms,
                feature_names,
                title=f"{stage.capitalize()} {display_name} ORBIT reconstruction residuals",
            ),
        }
        if self.include_codebook_histogram:
            figures[f"{stage}/{group_name}/orbit_codebook_usage"] = plot_codebook_histogram(
                code_idx[mask],
                num_codes=num_codes,
            )
        if self.enable_physics_plots:
            data_level = self._data_level(trainer)
            x_physical = self._to_physical_features(x_original)
            x_reco_physical = self._to_physical_features(x_reco)
            x_physical_flat = x_physical[mask]
            x_reco_physical_flat = x_reco_physical[mask]
            finite = np.all(np.isfinite(x_physical_flat), axis=1) & np.all(
                np.isfinite(x_reco_physical_flat),
                axis=1,
            )
            x_physical_flat = x_physical_flat[finite]
            x_reco_physical_flat = x_reco_physical_flat[finite]
            if x_physical_flat.size:
                physics_feature_names = ["Eta", "Phi", "pT"]
                physics_mse = np.mean((x_reco_physical_flat - x_physical_flat) ** 2, axis=0)
                if data_level == "particle":
                    jet_radii, jet_radius_label = self._jet_radii_for_group(
                        trainer,
                        group_name,
                        labels,
                    )
                    (
                        true_jet_pts,
                        reco_jet_pts,
                        true_jet_masses,
                        reco_jet_masses,
                        true_tau32s,
                        reco_tau32s,
                        true_jet_counts,
                        reco_jet_counts,
                    ) = self._collect_particle_jet_metrics(
                        x_physical,
                        x_reco_physical,
                        mask,
                        jet_radii=jet_radii,
                    )
                    missing_ets = self._collect_missing_transverse_energy(
                        x_physical, x_reco_physical, mask,
                    )
                    jet_count_diff = np.asarray(reco_jet_counts) - np.asarray(true_jet_counts)
                    physics_histograms = collect_physical_reconstruction_histograms(
                        physics_feature_names,
                        x_physical_flat,
                        x_reco_physical_flat,
                        true_jet_pts=true_jet_pts,
                        reco_jet_pts=reco_jet_pts,
                        true_jet_masses=true_jet_masses,
                        reco_jet_masses=reco_jet_masses,
                        true_tau32s=true_tau32s,
                        reco_tau32s=reco_tau32s,
                        true_missing_ets=missing_ets[0],
                        reco_missing_ets=missing_ets[1],
                        data_level=data_level,
                    )
                    physics_histograms["jet_count_orig"] = np.asarray(true_jet_counts)
                    physics_histograms["jet_count_reco"] = np.asarray(reco_jet_counts)
                    physics_histograms["jet_count_diff"] = jet_count_diff
                    metrics["metrics/jet_count_diff_mean"] = float(np.mean(jet_count_diff))
                    metrics["metrics/jet_count_diff_abs_mean"] = float(
                        np.mean(np.abs(jet_count_diff))
                    )
                    figures[f"{stage}/{group_name}/orbit_jet_count_difference"] = (
                        self._plot_jet_count_difference(
                            true_jet_counts,
                            reco_jet_counts,
                            title=(
                                f"{stage.capitalize()} {display_name} "
                                f"reclustered jet counts (R={jet_radius_label})"
                            ),
                        )
                    )
                else:
                    (
                        true_jet_pts, reco_jet_pts,
                        true_jet_etas, reco_jet_etas,
                        true_jet_phis, reco_jet_phis,
                    ) = self._collect_direct_jet_metrics(x_physical, x_reco_physical, mask)
                    physics_histograms = collect_physical_reconstruction_histograms(
                        physics_feature_names,
                        x_physical_flat,
                        x_reco_physical_flat,
                        true_jet_pts=true_jet_pts,
                        reco_jet_pts=reco_jet_pts,
                        true_jet_etas=true_jet_etas,
                        reco_jet_etas=reco_jet_etas,
                        true_jet_phis=true_jet_phis,
                        reco_jet_phis=reco_jet_phis,
                        data_level=data_level,
                    )
                histograms.update(
                    {f"physical_{key}": value for key, value in physics_histograms.items()}
                )
                physical_mse_metrics = {}
                physical_aliases = {
                    "Eta": ["Eta", "eta"],
                    "Phi": ["Phi", "phi"],
                    "pT": ["pT"],
                }
                for name, value in zip(physics_feature_names, physics_mse):
                    value = float(value)
                    physical_mse_metrics[f"metrics/physical_mse_{name}"] = value
                    for alias in physical_aliases.get(name, [name]):
                        physical_mse_metrics[f"metrics/mse_{alias}"] = value
                metrics.update(physical_mse_metrics)
                metrics["metrics/physical_mse_total"] = float(np.mean(physics_mse))
                figures.update(
                    {
                        f"{stage}/{group_name}/orbit_{name}": figure
                        for name, figure in physical_reconstruction_plots(
                            physics_feature_names,
                            physics_mse,
                            physics_histograms,
                            data_level=data_level,
                            include_all_ratios=self.include_all_ratios,
                        ).items()
                    }
                )

        self._log_metrics(trainer, stage, group_name, metrics)

        for name, fig in figures.items():
            filename_stem = name.split("/")[-1]
            if not preserve_legacy_names:
                filename_stem = f"{group_name}_{filename_stem}"
            filename = self._figure_name(trainer, stage, filename_stem)
            path = plot_dir / filename
            fig.savefig(path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved ORBIT figure {name} to {path}")
            self._log_figure(trainer, path, name)
            close_figure(fig)

        if self.save_histograms:
            histogram_dir = Path(trainer.default_root_dir) / "saved_histograms"
            histogram_dir.mkdir(parents=True, exist_ok=True)
            histogram_prefix = (
                f"{stage}_orbit" if preserve_legacy_names else f"{stage}_{group_name}_orbit"
            )
            histogram_path = (
                histogram_dir / f"{histogram_prefix}_hists_step_{trainer.global_step}.npz"
            )
            np.savez_compressed(histogram_path, **histograms)
            logger.info(f"Saved ORBIT histograms to {histogram_path}")

            metrics_dir = Path(trainer.default_root_dir) / "saved_metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = (
                metrics_dir / f"{histogram_prefix}_metrics_step_{trainer.global_step}.json"
            )
            with metrics_path.open("w") as file:
                json.dump(metrics, file, indent=2, sort_keys=True)
            logger.info(f"Saved ORBIT metrics to {metrics_path}")
