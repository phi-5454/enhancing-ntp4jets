"""Iterable parquet loaders for ORBIT particle and jet sequences."""

import os
import sys
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Optional

import awkward as ak
import lightning as L
import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch
from torch.utils.data import ChainDataset, DataLoader, IterableDataset, get_worker_info


SEQUENCE_SCHEMAS = {
    "particle": {
        "prefix": "L1T_PUPPIPart",
        "mask_column": "L1T_PUPPIPart_PuppiW",
        "mask_min_value": 0.05,
        "max_sequence_length": 128,
    },
    "jet_ak4": {
        "prefix": "L1T_JetAK4",
        "mask_column": None,
        "mask_min_value": 0.0,
        "max_sequence_length": 14,
    },
    "jet_ak8": {
        "prefix": "L1T_JetAK8",
        "mask_column": None,
        "mask_min_value": 0.0,
        "max_sequence_length": 7,
    },
    "jet_puppi_ak4": {
        "prefix": "L1T_JetPuppiAK4",
        "mask_column": None,
        "mask_min_value": 0.0,
        "max_sequence_length": 14,
    },
    "jet_puppi_ak8": {
        "prefix": "L1T_JetPuppiAK8",
        "mask_column": None,
        "mask_min_value": 0.0,
        "max_sequence_length": 7,
    },
}


MANIFEST_SUFFIXES = {".txt", ".list", ".lst"}
PARQUET_SUFFIXES = {".parquet"}


def _read_path_manifest(path: Path, seen: set[Path] | None = None) -> list[str]:
    """Read ORBIT-style text manifests containing one parquet path per line."""
    path = path.resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError(f"Recursive parquet path manifest detected: {path}")
    seen.add(path)

    files = []
    with path.open() as manifest:
        for line in manifest:
            value = line.strip()
            if not value or value.lstrip().startswith("#"):
                continue
            value = os.path.expandvars(os.path.expanduser(value))
            value_path = Path(value)
            if not value_path.is_absolute():
                value_path = path.parent / value_path
            files.extend(_expand_path_entry(value_path, seen=seen))
    return files


def _expand_path_entry(path, seen: set[Path] | None = None) -> list[str]:
    path = Path(os.path.expandvars(os.path.expanduser(str(path))))
    if path.suffix.lower() in MANIFEST_SUFFIXES or (
        path.is_file() and path.suffix.lower() not in PARQUET_SUFFIXES
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Parquet path manifest does not exist: {path}")
        return _read_path_manifest(path, seen=seen)
    return [str(path)]


def _as_paths(paths) -> list[str]:
    if paths is None:
        return []
    if isinstance(paths, (str, Path)):
        paths = [paths]
    expanded = []
    for path in paths:
        expanded.extend(_expand_path_entry(path))
    return expanded


def _dataset_files(paths) -> list[str]:
    paths = _as_paths(paths)
    if not paths:
        return []
    files = []
    for path in paths:
        files.extend(ds.dataset(path, format="parquet").files)
    return sorted(set(files))


def _limit_files(files: list[str], max_files: Optional[int]) -> list[str]:
    if max_files is None:
        return files
    if max_files < 1:
        raise ValueError("max_files_per_class values must be positive")
    return files[:max_files]


def deterministic_file_split(
    parquet_files,
    train_fraction: float = 0.8,
    split_seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Shuffle parquet files deterministically and return disjoint train and val lists."""
    files = _dataset_files(parquet_files)
    if len(files) < 2:
        raise ValueError(
            "Automatic train/val splitting requires at least two parquet files. "
            "For a smoke test, provide parquet_files_train and parquet_files_val explicitly."
        )
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be strictly between 0 and 1")

    rng = np.random.default_rng(split_seed)
    files = [str(path) for path in np.asarray(files)[rng.permutation(len(files))]]
    split_index = min(max(int(len(files) * train_fraction), 1), len(files) - 1)
    return files[:split_index], files[split_index:]


def _log_worker_exception(message: str) -> None:
    worker_info = get_worker_info()
    worker = "main" if worker_info is None else f"{worker_info.id}/{worker_info.num_workers}"
    print(
        f"[ORBIT_PARQUET_WORKER_ERROR] pid={os.getpid()} worker={worker} {message}",
        file=sys.stderr,
        flush=True,
    )
    traceback.print_exc(file=sys.stderr)
    sys.stderr.flush()


class OrbitPreprocessor:
    """Apply absolute-coordinate preprocessing without centering on a jet axis."""

    def __init__(self, feature_prefix: str, epsilon: float = 1e-8):
        self.eta_column = f"{feature_prefix}_Eta"
        self.phi_column = f"{feature_prefix}_Phi"
        self.pt_column = f"{feature_prefix}_PT"
        self.output_features = [
            self.eta_column,
            f"{self.phi_column}_cos",
            f"{self.phi_column}_sin",
            self.pt_column,
        ]
        self.epsilon = epsilon

    @property
    def input_features(self) -> list[str]:
        return [self.eta_column, self.phi_column, self.pt_column]

    def forward(self, array: ak.Array) -> ak.Array:
        array = ak.with_field(array, array[self.eta_column] / 3, self.eta_column)
        array = ak.with_field(array, np.cos(array[self.phi_column]), f"{self.phi_column}_cos")
        array = ak.with_field(array, np.sin(array[self.phi_column]), f"{self.phi_column}_sin")
        return ak.with_field(
            array,
            np.log(array[self.pt_column] + self.epsilon) - 1.8,
            self.pt_column,
        )


class OrbitParquetDataset(IterableDataset):
    """Stream pre-batched, padded ORBIT events from parquet row groups."""

    def __init__(
        self,
        parquet_files,
        sequence_type: str = "particle",
        max_sequence_length: Optional[int] = None,
        batch_size: int = 32,
        shuffle_row_groups: bool = False,
        shuffle_seed: int = 42,
        mask_column: Optional[str] = None,
        mask_min_value: Optional[float] = None,
        jet_type_label: int = 0,
        min_pt: Optional[float] = None,
        event_filter_sequence_type: Optional[str] = None,
        event_filter_min_pt: Optional[float] = None,
        max_events: Optional[int] = None,
    ):
        super().__init__()
        if sequence_type not in SEQUENCE_SCHEMAS:
            raise ValueError(
                f"Unknown sequence_type {sequence_type!r}. "
                f"Expected one of {sorted(SEQUENCE_SCHEMAS)}"
            )
        if (
            event_filter_sequence_type is not None
            and event_filter_sequence_type not in SEQUENCE_SCHEMAS
        ):
            raise ValueError(
                f"Unknown event_filter_sequence_type {event_filter_sequence_type!r}. "
                f"Expected one of {sorted(SEQUENCE_SCHEMAS)}"
            )
        parquet_files = _dataset_files(parquet_files)
        if not parquet_files:
            raise ValueError("parquet_files must contain at least one parquet file or directory")

        schema = SEQUENCE_SCHEMAS[sequence_type]
        self.preprocessor = OrbitPreprocessor(schema["prefix"])
        self.features = list(self.preprocessor.input_features)
        self.mask_column = schema["mask_column"] if mask_column is None else mask_column
        self.mask_min_value = (
            schema["mask_min_value"] if mask_min_value is None else mask_min_value
        )
        if self.mask_column is not None:
            self.features.append(self.mask_column)
        self.event_filter_pt_column = None
        if event_filter_sequence_type is not None:
            event_filter_prefix = SEQUENCE_SCHEMAS[event_filter_sequence_type]["prefix"]
            self.event_filter_pt_column = f"{event_filter_prefix}_PT"
            if self.event_filter_pt_column not in self.features:
                self.features.append(self.event_filter_pt_column)

        self.row_groups = []
        for file_path in parquet_files:
            parquet_file = pq.ParquetFile(file_path)
            self.row_groups.extend(
                (file_path, row_group_idx)
                for row_group_idx in range(parquet_file.num_row_groups)
            )

        self.sequence_type = sequence_type
        self.output_features = self.preprocessor.output_features
        self.max_sequence_length = (
            schema["max_sequence_length"]
            if max_sequence_length is None
            else max_sequence_length
        )
        self.batch_size = batch_size
        self.shuffle_row_groups = shuffle_row_groups
        self.shuffle_seed = shuffle_seed
        self.jet_type_label = jet_type_label
        self.min_pt = min_pt
        self.event_filter_min_pt = event_filter_min_pt
        if max_events is not None and max_events < 1:
            raise ValueError("max_events must be positive")
        self.max_events = max_events
        self._iteration = 0

    def __iter__(self):
        yielded_events = 0
        try:
            worker_info = get_worker_info()
            row_groups = list(self.row_groups)
            if self.shuffle_row_groups:
                rng = np.random.default_rng(self.shuffle_seed + self._iteration)
                rng.shuffle(row_groups)
                self._iteration += 1
            if worker_info is not None:
                row_groups = row_groups[worker_info.id :: worker_info.num_workers]
        except Exception:
            _log_worker_exception("failed during dataset iteration setup")
            raise

        for file_path, row_group_idx in row_groups:
            batch_idx = None
            try:
                parquet_file = pq.ParquetFile(file_path)
                batches = parquet_file.iter_batches(
                    row_groups=[row_group_idx],
                    columns=self.features,
                    batch_size=self.batch_size,
                    use_threads=True,
                )
                for batch_idx, batch in enumerate(batches):
                    for converted in self._convert_batch(ak.from_arrow(batch)):
                        if self.max_events is not None:
                            remaining = self.max_events - yielded_events
                            if remaining <= 0:
                                return
                            batch_events = converted["part_features"].shape[0]
                            if batch_events > remaining:
                                converted = {
                                    key: value[:remaining]
                                    for key, value in converted.items()
                                }
                                batch_events = remaining
                        yielded_events += converted["part_features"].shape[0]
                        yield converted
            except Exception:
                _log_worker_exception(
                    "failed while reading parquet "
                    f"file={file_path!r} row_group={row_group_idx} "
                    f"sequence_type={self.sequence_type!r} columns={self.features!r} "
                    f"batch_size={self.batch_size} "
                    f"max_sequence_length={self.max_sequence_length} "
                    f"last_batch_idx={batch_idx}"
                )
                raise

    def _convert_batch(self, batch: ak.Array):
        event_selector = ak.ones_like(ak.num(batch[self.preprocessor.pt_column]), dtype=bool)
        if self.event_filter_pt_column is not None:
            filter_pts = batch[self.event_filter_pt_column]
            if self.event_filter_min_pt is None:
                event_selector = ak.num(filter_pts) > 0
            else:
                event_selector = ak.any(filter_pts >= self.event_filter_min_pt, axis=1)

        sequence_mask = ak.ones_like(batch[self.preprocessor.pt_column], dtype=bool)
        if self.min_pt is not None:
            sequence_mask = sequence_mask & (batch[self.preprocessor.pt_column] >= self.min_pt)
        batch = self.preprocessor.forward(batch)
        if self.mask_column is not None:
            sequence_mask = sequence_mask & (batch[self.mask_column] > self.mask_min_value)

        stacked = ak.concatenate(
            [batch[field][sequence_mask][:, :, np.newaxis] for field in self.output_features],
            axis=-1,
        )
        event_lengths = ak.num(stacked, axis=1)
        non_empty_events = (event_lengths > 0) & event_selector
        if not ak.any(non_empty_events):
            return

        stacked = stacked[non_empty_events]
        event_lengths = event_lengths[non_empty_events]
        padded = ak.pad_none(stacked, self.max_sequence_length, axis=1, clip=True)
        filled = ak.fill_none(padded, [0.0] * len(self.output_features), axis=1)
        part_features = torch.from_numpy(ak.to_numpy(filled).astype(np.float32, copy=False))

        lengths = torch.from_numpy(
            np.minimum(ak.to_numpy(event_lengths), self.max_sequence_length).astype(
                np.int64, copy=False
            )
        )
        part_mask = torch.arange(self.max_sequence_length).unsqueeze(0) < lengths.unsqueeze(1)
        jet_type_labels = torch.full(
            (part_features.shape[0],),
            self.jet_type_label,
            dtype=torch.long,
        )

        yield {
            "part_features": part_features,
            "part_mask": part_mask,
            "jet_type_labels": jet_type_labels,
        }


class WeightedClassDataset(IterableDataset):
    """Interleave pre-batched class datasets with configurable class weights."""

    def __init__(
        self,
        datasets: Mapping[str, IterableDataset],
        weights: Mapping[str, float],
        seed: int,
    ):
        super().__init__()
        self.datasets = dict(datasets)
        self.weights = {name: float(weights[name]) for name in self.datasets}
        self.seed = seed
        self._iteration = 0

    def __iter__(self):
        iterators = {name: iter(dataset) for name, dataset in self.datasets.items()}
        active = list(iterators)
        rng = np.random.default_rng(self.seed + self._iteration)
        self._iteration += 1

        while active:
            weights = np.asarray([self.weights[name] for name in active], dtype=np.float64)
            if np.any(weights < 0) or np.sum(weights) <= 0:
                raise ValueError("class_sampling_weights must be non-negative with positive sum")
            probabilities = weights / np.sum(weights)
            name = str(rng.choice(active, p=probabilities))
            try:
                yield next(iterators[name])
            except StopIteration:
                active.remove(name)


class OrbitParquetDataModule(L.LightningDataModule):
    """Separate Lightning data module for ORBIT particle or jet parquet sequences.

    Two modes for train/val data:

    Single-class mode (original):
        Pass ``parquet_files_train_val`` (auto-split) or explicit
        ``parquet_files_train`` / ``parquet_files_val``.  All samples receive
        ``jet_type_label`` as their class index.

    Multi-class mode:
        Pass ``parquet_files_train_val_per_class`` — a mapping of
        ``{class_name: paths_or_manifest}`` or
        ``{class_name: {paths, weight, eval_sequence_type, eval_min_pt}}``.
        Class names are sorted alphabetically to assign deterministic integer labels (0, 1, …).
        Each class is split independently (class-aware train/val split), then
        the resulting datasets are chained.  The ``class_to_label`` mapping is
        recorded in ``hparams`` for reproducibility.
    """

    def __init__(
        self,
        parquet_files_train_val=None,
        parquet_files_train_val_per_class: Optional[dict] = None,
        parquet_files_test_per_class: Optional[dict] = None,
        parquet_files_test=None,
        parquet_files_train=None,
        parquet_files_val=None,
        sequence_type: str = "particle",
        max_sequence_length: Optional[int] = None,
        batch_size: int = 32,
        num_workers: int = 0,
        train_fraction: float = 0.8,
        split_seed: int = 42,
        shuffle_train: bool = True,
        shuffle_seed: int = 42,
        mask_column: Optional[str] = None,
        mask_min_value: Optional[float] = None,
        jet_type_label: int = 0,
        min_pt: Optional[float] = None,
        class_sequence_types: Optional[dict] = None,
        class_min_pts: Optional[dict] = None,
        class_eval_sequence_types: Optional[dict] = None,
        class_eval_min_pts: Optional[dict] = None,
        class_sampling_weights: Optional[dict] = None,
        max_files_per_class: Optional[dict] = None,
        max_train_events_per_class: Optional[dict] = None,
        max_val_events_per_class: Optional[dict] = None,
        max_test_events_per_class: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__()
        if sequence_type not in SEQUENCE_SCHEMAS:
            raise ValueError(
                f"Unknown sequence_type {sequence_type!r}. "
                f"Expected one of {sorted(SEQUENCE_SCHEMAS)}"
            )

        if isinstance(batch_size, int):
            self.batch_size_train = batch_size
            self.batch_size_val = batch_size
            self.batch_size_test = batch_size
        else:
            required_splits = {"train", "val", "test"}
            if not required_splits.issubset(batch_size):
                raise ValueError(
                    "If batch_size is a mapping, it must include train, val, and test"
                )
            self.batch_size_train = batch_size["train"]
            self.batch_size_val = batch_size["val"]
            self.batch_size_test = batch_size["test"]

        if parquet_files_train_val_per_class:
            if parquet_files_train_val or parquet_files_train or parquet_files_val:
                raise ValueError(
                    "Use either parquet_files_train_val_per_class or the single-class "
                    "params (parquet_files_train_val / parquet_files_train+val), not both."
                )
            class_names = sorted(parquet_files_train_val_per_class.keys(), key=str.lower)
            self._class_to_label: dict[str, int] = {name: i for i, name in enumerate(class_names)}
            self._train_files_per_class: dict[str, list[str]] = {}
            self._val_files_per_class: dict[str, list[str]] = {}
            self._test_files_per_class: dict[str, list[str]] = {}
            self._class_specs = self._normalize_class_specs(
                parquet_files_train_val_per_class,
                sequence_type=sequence_type,
                min_pt=min_pt,
                class_sequence_types=class_sequence_types,
                class_min_pts=class_min_pts,
                class_eval_sequence_types=class_eval_sequence_types,
                class_eval_min_pts=class_eval_min_pts,
                class_sampling_weights=class_sampling_weights,
                max_files_per_class=max_files_per_class,
                max_train_events_per_class=max_train_events_per_class,
                max_val_events_per_class=max_val_events_per_class,
                max_test_events_per_class=max_test_events_per_class,
            )
            self._test_class_specs = self._normalize_class_specs(
                parquet_files_test_per_class or {},
                sequence_type=sequence_type,
                min_pt=min_pt,
                class_sequence_types=class_sequence_types,
                class_min_pts=class_min_pts,
                class_eval_sequence_types=class_eval_sequence_types,
                class_eval_min_pts=class_eval_min_pts,
                class_sampling_weights=class_sampling_weights,
                max_files_per_class=max_files_per_class,
                max_train_events_per_class=max_train_events_per_class,
                max_val_events_per_class=max_val_events_per_class,
                max_test_events_per_class=max_test_events_per_class,
            )
            if parquet_files_test_per_class:
                if sorted(parquet_files_test_per_class.keys(), key=str.lower) != class_names:
                    raise ValueError(
                        "parquet_files_test_per_class must contain the same class names "
                        "as parquet_files_train_val_per_class"
                    )
                if parquet_files_test:
                    raise ValueError(
                        "Use either parquet_files_test_per_class or parquet_files_test, not both."
                    )
            for name in class_names:
                train_f, val_f = deterministic_file_split(
                    self._class_specs[name]["paths"],
                    train_fraction=train_fraction,
                    split_seed=split_seed,
                )
                self._train_files_per_class[name] = _limit_files(
                    train_f, self._class_specs[name]["max_files"]
                )
                self._val_files_per_class[name] = _limit_files(
                    val_f, self._class_specs[name]["max_files"]
                )
                if parquet_files_test_per_class:
                    self._test_files_per_class[name] = _limit_files(
                        _dataset_files(self._test_class_specs[name]["paths"]),
                        self._test_class_specs[name]["max_files"],
                    )
            self.parquet_files_train = None
            self.parquet_files_val = None
            self._multi_class = True
        else:
            if parquet_files_train_val:
                if parquet_files_train or parquet_files_val:
                    raise ValueError(
                        "Use either parquet_files_train_val or explicit "
                        "parquet_files_train/parquet_files_val, not both."
                    )
                parquet_files_train, parquet_files_val = deterministic_file_split(
                    parquet_files_train_val,
                    train_fraction=train_fraction,
                    split_seed=split_seed,
                )
            if not parquet_files_train or not parquet_files_val:
                raise ValueError(
                    "Provide parquet_files_train_val, parquet_files_train_val_per_class, "
                    "or explicit parquet_files_train and parquet_files_val."
                )
            self.parquet_files_train = parquet_files_train
            self.parquet_files_val = parquet_files_val
            self._class_to_label = {}
            self._class_specs = {}
            self._test_class_specs = {}
            self._test_files_per_class = {}
            self._multi_class = False

        self.parquet_files_test = parquet_files_test
        self._max_sequence_length = (
            max_sequence_length
            if max_sequence_length is not None
            else SEQUENCE_SCHEMAS[sequence_type]["max_sequence_length"]
        )
        self.selected_features = OrbitPreprocessor(
            SEQUENCE_SCHEMAS[sequence_type]["prefix"]
        ).output_features
        self.save_hyperparameters(ignore=["parquet_files_train", "parquet_files_val"])
        self.hparams["selected_features"] = self.selected_features
        self.hparams["max_sequence_length"] = self._max_sequence_length
        if self._multi_class:
            self.hparams["class_to_label"] = self._class_to_label
            self.hparams["class_specs"] = self._class_specs

    def data_split_summary(self) -> dict:
        """Return resolved split metadata for logging and reproducibility."""

        def split_row(
            split: str,
            class_name: str,
            label: int,
            files: list[str],
            spec: dict,
            event_count: Optional[int],
            batch_size: int,
        ) -> dict:
            return {
                "split": split,
                "class": class_name,
                "label": int(label),
                "file_count": len(files),
                "event_count": event_count,
                "batch_size": int(batch_size),
                "sequence_type": spec.get("sequence_type"),
                "min_pt": spec.get("min_pt"),
                "eval_sequence_type": spec.get("eval_sequence_type"),
                "eval_min_pt": spec.get("eval_min_pt"),
                "sampling_weight": spec.get("weight"),
            }

        if self._multi_class:
            rows = []
            for class_name, label in self._class_to_label.items():
                train_spec = self._class_specs[class_name]
                test_spec = self._test_class_specs.get(class_name, train_spec)
                rows.append(
                    split_row(
                        "train",
                        class_name,
                        label,
                        self._train_files_per_class[class_name],
                        train_spec,
                        train_spec["max_train_events"],
                        self.batch_size_train,
                    )
                )
                rows.append(
                    split_row(
                        "val",
                        class_name,
                        label,
                        self._val_files_per_class[class_name],
                        train_spec,
                        train_spec["max_val_events"],
                        self.batch_size_val,
                    )
                )
                rows.append(
                    split_row(
                        "test",
                        class_name,
                        label,
                        self._test_files_per_class.get(class_name, []),
                        test_spec,
                        test_spec.get("max_test_events"),
                        self.batch_size_test,
                    )
                )
            return {
                "mode": "multi_class",
                "train_fraction": float(self.hparams.train_fraction),
                "split_seed": int(self.hparams.split_seed),
                "class_to_label": dict(self._class_to_label),
                "rows": rows,
            }

        rows = [
            {
                "split": "train",
                "class": "all",
                "label": int(self.hparams.jet_type_label),
                "file_count": len(self.parquet_files_train),
                "event_count": None,
                "batch_size": int(self.batch_size_train),
                "sequence_type": self.hparams.sequence_type,
                "min_pt": self.hparams.min_pt,
                "eval_sequence_type": None,
                "eval_min_pt": None,
                "sampling_weight": 1.0,
            },
            {
                "split": "val",
                "class": "all",
                "label": int(self.hparams.jet_type_label),
                "file_count": len(self.parquet_files_val),
                "event_count": None,
                "batch_size": int(self.batch_size_val),
                "sequence_type": self.hparams.sequence_type,
                "min_pt": self.hparams.min_pt,
                "eval_sequence_type": None,
                "eval_min_pt": None,
                "sampling_weight": 1.0,
            },
            {
                "split": "test",
                "class": "all",
                "label": int(self.hparams.jet_type_label),
                "file_count": len(_dataset_files(self.parquet_files_test or [])),
                "event_count": None,
                "batch_size": int(self.batch_size_test),
                "sequence_type": self.hparams.sequence_type,
                "min_pt": self.hparams.min_pt,
                "eval_sequence_type": None,
                "eval_min_pt": None,
                "sampling_weight": 1.0,
            },
        ]
        return {
            "mode": "single_class",
            "train_fraction": float(self.hparams.train_fraction),
            "split_seed": int(self.hparams.split_seed),
            "class_to_label": {"all": int(self.hparams.jet_type_label)},
            "rows": rows,
        }

    @staticmethod
    def _normalize_class_specs(
        raw_specs,
        sequence_type: str,
        min_pt: Optional[float],
        class_sequence_types: Optional[dict],
        class_min_pts: Optional[dict],
        class_eval_sequence_types: Optional[dict],
        class_eval_min_pts: Optional[dict],
        class_sampling_weights: Optional[dict],
        max_files_per_class: Optional[dict],
        max_train_events_per_class: Optional[dict],
        max_val_events_per_class: Optional[dict],
        max_test_events_per_class: Optional[dict],
    ) -> dict[str, dict]:
        specs = {}
        class_sequence_types = class_sequence_types or {}
        class_min_pts = class_min_pts or {}
        class_eval_sequence_types = class_eval_sequence_types or {}
        class_eval_min_pts = class_eval_min_pts or {}
        class_sampling_weights = class_sampling_weights or {}
        max_files_per_class = max_files_per_class or {}
        max_train_events_per_class = max_train_events_per_class or {}
        max_val_events_per_class = max_val_events_per_class or {}
        max_test_events_per_class = max_test_events_per_class or {}

        for name, value in raw_specs.items():
            if isinstance(value, Mapping) and any(
                key in value for key in ("paths", "files", "manifest")
            ):
                paths = value.get("paths", value.get("files", value.get("manifest")))
                spec_sequence_type = value.get(
                    "sequence_type", class_sequence_types.get(name, sequence_type)
                )
                spec_min_pt = value.get("min_pt", class_min_pts.get(name, min_pt))
                eval_sequence_type = value.get(
                    "eval_sequence_type", class_eval_sequence_types.get(name)
                )
                eval_min_pt = value.get("eval_min_pt", class_eval_min_pts.get(name))
                spec_weight = value.get("weight", class_sampling_weights.get(name, 1.0))
                spec_max_files = value.get("max_files", max_files_per_class.get(name))
                max_train_events = value.get(
                    "max_train_events", max_train_events_per_class.get(name)
                )
                max_val_events = value.get("max_val_events", max_val_events_per_class.get(name))
                max_test_events = value.get(
                    "max_test_events", max_test_events_per_class.get(name)
                )
            else:
                paths = value
                spec_sequence_type = class_sequence_types.get(name, sequence_type)
                spec_min_pt = class_min_pts.get(name, min_pt)
                eval_sequence_type = class_eval_sequence_types.get(name)
                eval_min_pt = class_eval_min_pts.get(name)
                spec_weight = class_sampling_weights.get(name, 1.0)
                spec_max_files = max_files_per_class.get(name)
                max_train_events = max_train_events_per_class.get(name)
                max_val_events = max_val_events_per_class.get(name)
                max_test_events = max_test_events_per_class.get(name)

            if spec_sequence_type not in SEQUENCE_SCHEMAS:
                raise ValueError(
                    f"Unknown sequence_type {spec_sequence_type!r} for class {name!r}. "
                    f"Expected one of {sorted(SEQUENCE_SCHEMAS)}"
                )
            if eval_sequence_type is not None and eval_sequence_type not in SEQUENCE_SCHEMAS:
                raise ValueError(
                    f"Unknown eval_sequence_type {eval_sequence_type!r} for class {name!r}. "
                    f"Expected one of {sorted(SEQUENCE_SCHEMAS)}"
                )
            specs[name] = {
                "paths": paths,
                "sequence_type": spec_sequence_type,
                "min_pt": None if spec_min_pt is None else float(spec_min_pt),
                "eval_sequence_type": eval_sequence_type,
                "eval_min_pt": None if eval_min_pt is None else float(eval_min_pt),
                "weight": float(spec_weight),
                "max_files": None if spec_max_files is None else int(spec_max_files),
                "max_train_events": (
                    None if max_train_events is None else int(max_train_events)
                ),
                "max_val_events": None if max_val_events is None else int(max_val_events),
                "max_test_events": None if max_test_events is None else int(max_test_events),
            }
        return specs

    def _dataset(
        self,
        parquet_files,
        batch_size: int,
        shuffle_row_groups: bool,
        jet_type_label: Optional[int] = None,
        sequence_type: Optional[str] = None,
        min_pt: Optional[float] = None,
        event_filter_sequence_type: Optional[str] = None,
        event_filter_min_pt: Optional[float] = None,
        max_events: Optional[int] = None,
    ):
        return OrbitParquetDataset(
            parquet_files=parquet_files,
            sequence_type=sequence_type or self.hparams.sequence_type,
            max_sequence_length=self._max_sequence_length,
            batch_size=batch_size,
            shuffle_row_groups=shuffle_row_groups,
            shuffle_seed=self.hparams.shuffle_seed,
            mask_column=self.hparams.mask_column,
            mask_min_value=self.hparams.mask_min_value,
            jet_type_label=(
                jet_type_label if jet_type_label is not None else self.hparams.jet_type_label
            ),
            min_pt=min_pt if min_pt is not None else self.hparams.min_pt,
            event_filter_sequence_type=event_filter_sequence_type,
            event_filter_min_pt=event_filter_min_pt,
            max_events=max_events,
        )

    def _loader(self, dataset, persistent_workers: bool = True):
        kwargs = {
            "batch_size": None,
            "num_workers": self.hparams.num_workers,
            "pin_memory": torch.cuda.is_available(),
            "persistent_workers": persistent_workers and self.hparams.num_workers > 0,
        }
        if self.hparams.num_workers > 0:
            kwargs["prefetch_factor"] = 4
        return DataLoader(dataset, **kwargs)

    def train_dataloader(self):
        if self._multi_class:
            datasets = {
                name: self._dataset(
                    self._train_files_per_class[name],
                    batch_size=self.batch_size_train,
                    shuffle_row_groups=self.hparams.shuffle_train,
                    jet_type_label=label,
                    sequence_type=self._class_specs[name]["sequence_type"],
                    min_pt=self._class_specs[name]["min_pt"],
                    max_events=self._class_specs[name]["max_train_events"],
                )
                for name, label in self._class_to_label.items()
            }
            return self._loader(
                WeightedClassDataset(
                    datasets,
                    weights={name: self._class_specs[name]["weight"] for name in datasets},
                    seed=self.hparams.shuffle_seed,
                )
            )
        return self._loader(
            self._dataset(
                self.parquet_files_train,
                batch_size=self.batch_size_train,
                shuffle_row_groups=self.hparams.shuffle_train,
            )
        )

    def val_dataloader(self):
        if self._multi_class:
            datasets = [
                self._dataset(
                    self._val_files_per_class[name],
                    batch_size=self.batch_size_val,
                    shuffle_row_groups=False,
                    jet_type_label=label,
                    sequence_type=self._class_specs[name]["sequence_type"],
                    min_pt=self._class_specs[name]["min_pt"],
                    event_filter_sequence_type=self._class_specs[name]["eval_sequence_type"],
                    event_filter_min_pt=self._class_specs[name]["eval_min_pt"],
                    max_events=self._class_specs[name]["max_val_events"],
                )
                for name, label in self._class_to_label.items()
            ]
            return self._loader(ChainDataset(datasets))
        return self._loader(
            self._dataset(
                self.parquet_files_val,
                batch_size=self.batch_size_val,
                shuffle_row_groups=False,
            )
        )

    def test_dataloader(self):
        if self._multi_class and self._test_files_per_class:
            datasets = [
                self._dataset(
                    self._test_files_per_class[name],
                    batch_size=self.batch_size_test,
                    shuffle_row_groups=False,
                    jet_type_label=label,
                    sequence_type=self._test_class_specs[name]["sequence_type"],
                    min_pt=self._test_class_specs[name]["min_pt"],
                    event_filter_sequence_type=self._test_class_specs[name][
                        "eval_sequence_type"
                    ],
                    event_filter_min_pt=self._test_class_specs[name]["eval_min_pt"],
                    max_events=self._test_class_specs[name]["max_test_events"],
                )
                for name, label in self._class_to_label.items()
            ]
            return self._loader(ChainDataset(datasets), persistent_workers=False)
        if not self.parquet_files_test:
            raise ValueError("Provide parquet_files_test from a separate test directory.")
        return self._loader(
            self._dataset(
                self.parquet_files_test,
                batch_size=self.batch_size_test,
                shuffle_row_groups=False,
            ),
            persistent_workers=False,
        )
