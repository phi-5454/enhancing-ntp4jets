"""Schema-neutral ORBIT plotting callback."""

from __future__ import annotations

import json
from pathlib import Path

import lightning as L
import numpy as np

from gabbro.plotting.orbit import (
    close_figure,
    collect_reconstruction_histograms,
    plot_codebook_histogram,
    plot_feature_histograms,
    plot_residual_histograms,
)
from gabbro.utils.pylogger import get_pylogger

logger = get_pylogger("OrbitPlottingCallback")


class OrbitPlottingCallback(L.Callback):
    """Plot masked reconstruction summaries for particle or jet ORBIT parquet batches."""

    def __init__(
        self,
        image_path: str | None = None,
        image_filetype: str = "png",
        save_histograms: bool = True,
        no_trainer_info_in_filename: bool = False,
    ):
        super().__init__()
        self.image_path = image_path
        self.image_filetype = image_filetype
        self.save_histograms = save_histograms
        self.no_trainer_info_in_filename = no_trainer_info_in_filename

    def on_validation_epoch_end(self, trainer, pl_module):
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
