"""Schema-neutral ORBIT plotting callback."""

from __future__ import annotations

import json
from pathlib import Path

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
    ):
        super().__init__()
        self.image_path = image_path
        self.image_filetype = image_filetype
        self.save_histograms = save_histograms
        self.no_trainer_info_in_filename = no_trainer_info_in_filename
        self.enable_physics_plots = enable_physics_plots
        self.include_all_ratios = include_all_ratios
        self.jet_radius = jet_radius

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

    def _log_figure(self, trainer, path: Path, name: str) -> None:
        for lightning_logger in trainer.loggers:
            if isinstance(lightning_logger, L.pytorch.loggers.CometLogger):
                lightning_logger.experiment.log_image(
                    str(path),
                    name=name,
                    step=trainer.global_step,
                )
            elif isinstance(lightning_logger, L.pytorch.loggers.WandbLogger):
                try:
                    import wandb

                    lightning_logger.experiment.log(
                        {name: wandb.Image(str(path))},
                        step=trainer.global_step,
                    )
                except Exception as exc:
                    logger.warning(f"Failed to log {path} to W&B: {exc}")

    def _num_codes(self, pl_module) -> int | None:
        vqlayer = getattr(pl_module.model, "vqlayer", None)
        if vqlayer is None:
            return None
        return getattr(vqlayer, "num_codes", None)

    def _data_level(self, trainer) -> str:
        sequence_type = getattr(trainer.datamodule.hparams, "sequence_type", "particle")
        return "particle" if sequence_type == "particle" else "jet"

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
        return true_jet_pts, reco_jet_pts, true_jet_etas, reco_jet_etas, true_jet_phis, reco_jet_phis

    def _reconstruct_event_jets(self, pt: np.ndarray, eta: np.ndarray, phi: np.ndarray):
        pt = np.asarray(pt, dtype=np.float64)
        eta = np.asarray(eta, dtype=np.float64)
        phi = np.asarray(phi, dtype=np.float64)
        if len(pt) < 3 or np.sum(pt) <= 0:
            return {
                "pt": np.array([]),
                "jet_mass": 0.0,
                "tau32": 0.0,
                "jet_n_constituents": len(pt),
            }

        particles = ak.zip(
            {"pt": [pt], "eta": [eta], "phi": [phi], "mass": [np.zeros_like(pt)]},
            with_name="Momentum4D",
        )
        particles_sum = ak.sum(particles, axis=1)
        jetdef = fastjet.JetDefinition(fastjet.kt_algorithm, self.jet_radius)
        cluster = fastjet.ClusterSequence(particles, jetdef)
        inclusive_jets = cluster.inclusive_jets(min_pt=0.0)
        d0 = ak.sum(particles.pt * self.jet_radius, axis=1)

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
        }

    def _collect_particle_jet_metrics(
        self,
        x_physical: np.ndarray,
        x_reco_physical: np.ndarray,
        mask: np.ndarray,
    ):
        true_jet_pts = []
        reco_jet_pts = []
        true_jet_masses = []
        reco_jet_masses = []
        true_tau32s = []
        reco_tau32s = []

        for i in range(x_physical.shape[0]):
            event_mask = mask[i]
            true_jets = self._reconstruct_event_jets(
                x_physical[i, event_mask, 2],
                x_physical[i, event_mask, 0],
                x_physical[i, event_mask, 1],
            )
            reco_jets = self._reconstruct_event_jets(
                x_reco_physical[i, event_mask, 2],
                x_reco_physical[i, event_mask, 0],
                x_reco_physical[i, event_mask, 1],
            )
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
        )

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
            code_idx = pl_module.test_code_idx_concat

        if not np.any(mask):
            logger.warning(f"No valid tokens found for {stage} ORBIT plots.")
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
        metrics.update(
            {
                "metrics/mse_total": float(np.mean(mse_per_feature)),
                "metrics/active_codes_total": int(active_codes),
                "metrics/utilization_total": (
                    float(active_codes / num_codes) if num_codes else None
                ),
                "metrics/total_codebook_size": int(num_codes) if num_codes else None,
            }
        )

        figures = {
            f"{stage}/orbit_reconstruction_features": plot_feature_histograms(
                histograms,
                feature_names,
                mse_per_feature=mse_per_feature,
                title=f"{stage.capitalize()} ORBIT reconstruction features",
            ),
            f"{stage}/orbit_reconstruction_residuals": plot_residual_histograms(
                histograms,
                feature_names,
                title=f"{stage.capitalize()} ORBIT reconstruction residuals",
            ),
            f"{stage}/orbit_codebook_usage": plot_codebook_histogram(
                code_idx[mask],
                num_codes=num_codes,
            ),
        }
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
                    (
                        true_jet_pts, reco_jet_pts,
                        true_jet_masses, reco_jet_masses,
                        true_tau32s, reco_tau32s,
                    ) = self._collect_particle_jet_metrics(x_physical, x_reco_physical, mask)
                    missing_ets = self._collect_missing_transverse_energy(
                        x_physical, x_reco_physical, mask,
                    )
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
                histograms.update({f"physical_{key}": value for key, value in physics_histograms.items()})
                metrics.update(
                    {
                        f"metrics/physical_mse_{name}": float(value)
                        for name, value in zip(physics_feature_names, physics_mse)
                    }
                )
                metrics["metrics/physical_mse_total"] = float(np.mean(physics_mse))
                figures.update(
                    {
                        f"{stage}/orbit_{name}": figure
                        for name, figure in physical_reconstruction_plots(
                            physics_feature_names,
                            physics_mse,
                            physics_histograms,
                            data_level=data_level,
                            include_all_ratios=self.include_all_ratios,
                        ).items()
                    }
                )

        for name, fig in figures.items():
            filename = self._figure_name(trainer, stage, name.split("/")[-1])
            path = plot_dir / filename
            fig.savefig(path, dpi=300, bbox_inches="tight")
            self._log_figure(trainer, path, name)
            close_figure(fig)

        if self.save_histograms:
            histogram_dir = Path(trainer.default_root_dir) / "saved_histograms"
            histogram_dir.mkdir(parents=True, exist_ok=True)
            histogram_path = histogram_dir / f"{stage}_orbit_hists_step_{trainer.global_step}.npz"
            np.savez_compressed(histogram_path, **histograms)
            logger.info(f"Saved ORBIT histograms to {histogram_path}")

            metrics_dir = Path(trainer.default_root_dir) / "saved_metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = metrics_dir / f"{stage}_orbit_metrics_step_{trainer.global_step}.json"
            with metrics_path.open("w") as file:
                json.dump(metrics, file, indent=2, sort_keys=True)
            logger.info(f"Saved ORBIT metrics to {metrics_path}")
