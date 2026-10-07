import os
import time
import torch
import inspect
import datetime
import traceback
import typing as t
import itertools as it
import numpy as np
import pytorch_lightning as pl

from bisect import bisect_left
from pathlib import Path
from collections.abc import MutableMapping

DATA_DIR = Path("data")
EXPERIMENTS_DIR = Path(os.environ.get("NMLN_EXPERIMENTS_DIR", DATA_DIR / "experiments"))

EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)


class SafeMLFlowLogger(pl.loggers.mlflow.MLFlowLogger):
    """
    Subclassing `pl.loggers.mlflow.MLFlowLogger` and overriding metrics methods to handle unsucesful uploads.
    """

    def log_metrics(
        self, metrics: t.Dict[str, float], step: t.Optional[int] = None
    ) -> None:
        # It sometimes happens that connection errors out.
        try:
            super().log_metrics(metrics=metrics, step=step)

        except Exception:
            print("Warning: Metric logging skipped because of connection error.")

    def log_metric(
        self, key: str, value: float, step: t.Optional[int] = None
    ) -> None:
        self.log_metrics({key: value}, step=step)

    def log_artifact(self, path: str, artifact_path: str | None = None) -> None:
        # It sometimes happens that connection errors out.
        try:
            self.experiment.log_artifact(self.run_id, path, artifact_path)

        except Exception:
            print("Warning: Artifact logging skipped because of connection error.")


class MLFlowModelCheckpoint(pl.callbacks.ModelCheckpoint):
    """
    Subclassing `pl.callbacks.ModelCheckpoint` and overriding `save_checkpoint` to upload checkpoints to MLflow.
    See https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.ModelCheckpoint.html#modelcheckpoint.
    """

    def __init__(self, logger_mlflow: pl.loggers.mlflow.MLFlowLogger, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)

        self.logger_mlflow = logger_mlflow

    def _safe_log(self, path: str) -> None:
        for _ in range(3):
            try:
                self.logger_mlflow.experiment.log_artifact(
                    self.logger_mlflow.run_id, path
                )

            except Exception:
                print("Failed to log artifact:", traceback.format_exc())
                print(f"Recover your artifact at {path}.")
                time.sleep(3.14)

            else:
                break

    def save_checkpoint(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().save_checkpoint(*args, **kwargs)

        if self.best_model_path:
            self._safe_log(self.best_model_path)

        if self.last_model_path:
            print("Logging last model path:", self.last_model_path)
            self._safe_log(self.last_model_path)


class RangeBisection(MutableMapping):
    """
    Map ranges to values

    Lookups are done in O(logN) time. There are no limits set on the upper or
    lower bounds of the ranges, but ranges must not overlap.
    """

    def __init__(self, map=None):
        self._upper = []
        self._lower = []
        self._values = []
        if map is not None:
            self.update(map)

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, point_or_range):
        if isinstance(point_or_range, tuple):
            low, high = point_or_range
            i = bisect_left(self._upper, high)
            point = low

        else:
            point = point_or_range
            i = bisect_left(self._upper, point)

        if i >= len(self._values) or self._lower[i] > point:
            raise IndexError(point_or_range)

        return self._values[i]

    def __setitem__(self, r, value):
        lower, upper = r
        i = bisect_left(self._upper, upper)
        if i < len(self._values) and self._lower[i] < upper:
            raise IndexError("No overlaps permitted")
        self._upper.insert(i, upper)
        self._lower.insert(i, lower)
        self._values.insert(i, value)

    def __delitem__(self, r):
        lower, upper = r
        i = bisect_left(self._upper, upper)
        if self._upper[i] != upper or self._lower[i] != lower:
            raise IndexError("Range not in map")
        del self._upper[i]
        del self._lower[i]
        del self._values[i]

    def __iter__(self):
        yield from zip(self._lower, self._upper)


def get_next_version_directory(
    name: str, autocreate: bool = True, versioning: bool = True
) -> t.Tuple[str, Path]:
    version = 0

    while True:
        exp_name = (
            (f"{name} - {datetime.datetime.today().strftime('%d-%m-%Y')} - {version}")
            if versioning
            else name
        )
        output_dir = EXPERIMENTS_DIR / exp_name

        if not versioning or not os.path.isdir(output_dir):
            break

        version += 1

    if autocreate:
        output_dir.mkdir(parents=True, exist_ok=True)

    return exp_name, output_dir


def tf_gather(
    params, indices, batch_dims: t.Optional[int] = None, axis: t.Optional[int] = None
):
    # Giuseppe implementation of
    # `tensorflow.gather(params=params, indices=indices, batch_dims=batch_dims)`
    # and
    # `tf.gather(params=params, indices=indices, axis=axis)`
    assert (
        sum([bool(batch_dims), bool(axis)]) == 1
    ), "Provide batch_dims or axis, not both."

    if axis is not None:
        new_size = tuple([*params.shape[:axis], *indices.shape])
        out = params.index_select(index=indices.view(-1), dim=axis).view(new_size)

    if batch_dims is not None:
        assert list(indices.shape[:batch_dims]) == list(
            params.shape[:batch_dims]
        ), f"{indices.shape=} {params.shape=}"
        batch_dims_shape = list(params.shape[:batch_dims])
        param_dims_shape = list(params.shape[batch_dims:])
        indices_dims_shape = list(indices.shape[batch_dims:])

        prod_batch = np.prod(batch_dims_shape)
        batch_range = np.reshape(np.arange(prod_batch), [-1, 1])
        indices = indices.reshape([prod_batch, -1])
        params = params.reshape([-1] + param_dims_shape)

        out = params[batch_range, indices]
        out = out.reshape(batch_dims_shape + indices_dims_shape + param_dims_shape[1:])

    return out


# According to o1-preview, a faster implementation of `tf_gather`.
def tf_gather_o1(
    params, indices, batch_dims: t.Optional[int] = None, axis: t.Optional[int] = None
):
    # Efficient implementation of
    # `tf.gather(params=params, indices=indices, batch_dims=batch_dims)`
    # and
    # `tf.gather(params=params, indices=indices, axis=axis)` in PyTorch
    assert (
        sum([batch_dims is not None, axis is not None]) == 1
    ), "Provide batch_dims or axis, not both."

    if axis is not None:
        new_size = tuple([*params.shape[:axis], *indices.shape])
        out = params.index_select(index=indices.view(-1), dim=axis).view(new_size)
        return out

    if batch_dims is not None:
        indices = indices.long()
        assert list(indices.shape[:batch_dims]) == list(
            params.shape[:batch_dims]
        ), f"{indices.shape[:batch_dims]} != {params.shape[:batch_dims]}"

        batch_dims_shape = params.shape[:batch_dims]
        param_dims_shape = params.shape[batch_dims:]
        indices_dims_shape = indices.shape[batch_dims:]

        batch_size = int(torch.tensor(batch_dims_shape).prod().item())

        # Flatten batch dimensions
        params = params.reshape(batch_size, *param_dims_shape)
        indices = indices.reshape(batch_size, -1)

        # Create batch indices for advanced indexing
        batch_indices = torch.arange(batch_size, device=params.device).unsqueeze(1)
        batch_indices = batch_indices.expand(-1, indices.size(1))

        # Gather using advanced indexing
        out = params[batch_indices, indices]
        # Reshape to original dimensions
        out_shape = batch_dims_shape + indices_dims_shape + param_dims_shape[1:]
        out = out.reshape(out_shape)
        return out


def shapecheck(**tensors: t.Dict[str, t.Tuple[int, ...]]):
    """
    Decorator factory that will shape-check tensors on runtime.

    Example:

    ```
    @shapecheck(
        x=(..., 4),  # ... indicates arbitary value, e.g. batch dimension.
        y=(3, 5),
    )
    def forward(self, x, y):
        ...
    ```

    You can disable checking with env variable SKIP_SHAPECHECK.
    """

    def decorator(function):
        def wrapper(*args, **kwargs):
            # Get names of arguments from the function (if they are not provided as kwargs, then we need this)/
            argspec = inspect.getfullargspec(function)

            # Iterate first positional arguments with their names from argspec and then kwargs.
            if os.environ.get("SKIP_SHAPECHECK") is None:
                for name, value in it.chain(zip(argspec.args, args), kwargs.items()):
                    # If the argument is not in shapecheck dictionary, skip it.
                    if name not in tensors:
                        continue

                    failed = False

                    if not hasattr(value, "shape"):
                        raise RuntimeError(
                            f"Argument {name} does not have attribute shape."
                        )

                    # If shape has different length, is obviously incorrect.
                    if len(value.shape) != len(tensors[name]):
                        failed = True

                    for real_dim, expected_dim in zip(value.shape, tensors[name]):
                        # ... indicates anything, e.g. batch size.
                        if expected_dim == ...:
                            continue

                        if expected_dim != real_dim:
                            failed = True
                            break

                    if failed:
                        raise RuntimeError(
                            f"Invalid shape {value.shape} for tensor {name}, expected {tensors[name]}."
                        )

            return function(*args, **kwargs)

        return wrapper

    return decorator
