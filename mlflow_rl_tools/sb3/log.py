from typing import Any

import numpy as np
import mlflow
from stable_baselines3.common.logger import KVWriter


class MLflowOutputFormat(KVWriter):
    """
    Dumps key/value pairs into MLflow's numeric format.
    """

    def write(
        self,
        key_values: dict[str, Any],
        key_excluded: dict[str, str | tuple[str, ...]],
        step: int = 0,
    ) -> None:
        """
        Write key/value pairs to MLflow.

        Parameters
        ----------
        key_values: dict[str, Any]
            The key/value pairs to write.
        key_excluded: dict[str, str | tuple[str, ...]]
            The keys to exclude from writing.
        step: int = 0
            The step number to associate with the key/value pairs.
        """

        for (key, value), (_, excluded) in zip(
            sorted(key_values.items()), sorted(key_excluded.items())
        ):

            if excluded is not None and "mlflow" in excluded:
                continue

            if isinstance(value, np.ScalarType):
                if not isinstance(value, str):
                    mlflow.log_metric(key, value, step)
