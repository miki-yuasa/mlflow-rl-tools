"""
The ``mlflow_rl_tools.sb3`` module provides an API for logging and loading Stable-Baselines3 models. 
This module exports Stable-Baselines3 models with the following flavors:

PyTorch (native) format
    This is the main flavor that can be loaded back into Stable-Baselines3.
:py:mod:`mlflow.pyfunc`
    Produced for use by generic pyfunc-based deployment tools and batch inference.
"""

from typing import Any

from mlflow.models import ModelSignature, Model
from mlflow.models.model import ModelInfo
from mlflow.models.utils import ModelInputExample
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
import stable_baselines3
from stable_baselines3.common.base_class import SelfBaseAlgorithm

import mlflow_rl_tools
from mlflow_rl_tools._sb3_base import load_model as base_load_model
from mlflow_rl_tools._sb3_base import _load_pyfunc as _base_load_pyfunc
from mlflow_rl_tools._sb3_base import save_model as base_save_model
from mlflow_rl_tools._sb3_base import MLflowModel
import mlflow_rl_tools.sb3

FLAVOR_NAME = "sb3"


def log_model(
    sb3_model: SelfBaseAlgorithm,
    artifact_path: str,
    conda_env: dict[str, Any] | None = None,
    code_paths: list[str] | None = None,
    registered_model_name: str | None = None,
    signature: ModelSignature | None = None,
    input_example: ModelInputExample | None = None,
    await_registration_for: int = DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    extra_files: list[str] | None = None,
    pip_requirements: str | list[str] | None = None,
    extra_pip_requirements: str | list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs,
) -> ModelInfo:
    """
    Log a PyTorch model as an MLflow artifact for the current run.

    .. warning:: Log the model with a signature to avoid inference errors.
        If the model is logged without a signature, the MLflow Model Server relies on the
        default inferred data type from NumPy. However, PyTorch often expects different
        defaults, particularly when parsing floats. You must include the signature to ensure
        that the model is logged with the correct data type so that the MLflow model server
        can correctly provide valid input.

    Parameters
    ----------
    sb3_model: SelfBaseAlgorithm
        Stable-Baselines3 model to be saved.

    artifact_path: str
        Run-relative artifact path.

    conda_env: dict[str, Any] | None = None
        If provided, this dictionary is included in the model's MLmodel file.
        The dictionary should contain a valid Conda environment.
        This parameter is used to specify a custom environment for the model.

    code_paths: list[str] | None = None
        A list of local filesystem paths to Python file dependencies (or directories containing file dependencies).
        These files are *prepended* to the system path when the model is loaded.
        Files declared as dependencies for a given model should have relative imports declared from a common root path
        if multiple files are defined with import dependencies between them to avoid import errors when loading the model.

    registered_model_name: str | None = None
        If given, create a model version under ``registered_model_name``,
        also create a registered model if one with the given name does not exist.

    signature: ModelSignature | None = None
        An instance of the :py:class:`ModelSignature <mlflow.models.ModelSignature>` class
        that describes the model's inputs and outputs.
        If not specified but an ``input_example`` is supplied, a signature will be
        automatically inferred based on the supplied input example and model.
        To disable automatic signature inference when providing an input example, set ``signature`` to ``False``.
        To manually infer a model signature, call :py`infer_signature() <mlflow.models.infer_signature>`
        on datasets with valid model inputs, such as a training dataset with the target columnomitted,
        and valid model outputs, like model predictions made on the trainingdataset, for example:
        ```python
        from mlflow.models import infer_signature

        train = df.drop_column("target_label")
        predictions = ...  # compute model predictions
        signature = infer_signature(train, predictions)
        ```

    input_example: ModelInputExample | None = None
        One or several instances of valid model input. The input example is used as a hint of what data to feed the model.
        It will be converted to a Pandas DataFrame and then serialized to json using the Pandas split-oriented format,
        or a numpy array where the example will be serialized to json by converting it to a list.
        Bytes are base64-encoded. When the ``signature`` parameter is ``None``, the input example is
        used to infer a model signature.

    await_registration_for: int = DEFAULT_AWAIT_MAX_SLEEP_SECONDS
        Number of seconds to wait for the model version to finish  being created and is in ``READY`` status.
        By default, the function waits for five minutes.
        Specify 0 or None to skip waiting.

    extra_files: list[str] | None = None
        A list of local filesystem paths to supplementary files that should be packaged with the model.
        These files are copied to the same location as the model when it is saved.
        For example, consider the following ``extra_files`` list.
        In this case, the ``"my_file1 & my_file2"`` extra file is downloaded from S3:
        ```python
        extra_files = ["s3://my-bucket/path/to/my_file1", "s3://my-bucket/path/to/my_file2"]
        ```

    pip_requirements: str | list[str] | None = None
        Either an iterable of pip requirement strings
        (e.g. ``["{{ package_name }}", "-r requirements.txt", "-c constraints.txt"]``) or the string path
        to a pip requirements file on the local filesystem (e.g. ``"requirements.txt"``).
        If provided, this describes the environment this model should be run in.
        If ``None``, a default list of requirements is inferred by `mlflow.models.infer_pip_requirements`
        from the current software environment.
        If the requirement inference fails, it falls back to using `get_default_pip_requirements`.
        Both requirements and constraints are automatically parsed and written to ``requirements.txt`` and
        ``constraints.txt`` files, respectively, and stored as part of the model.
        Requirements are alsoritten to the ``pip`` section of the model's conda environment (``conda.yaml``) file.

    extra_pip_requirements: str | list[str] | None = None
        Either an iterable of pip requirement strings
        (e.g. ``["{{ package_name }}", "-r requirements.txt", "-c constraints.txt"]``)
        or the string path to a pip requirements file on the local filesystem (e.g. ``"requirements.txt"``).
        If provided, this describes additional requirements for the model that are not included
        in the main ``pip_requirements``.
        If ``None``, no extra requirements are added to the model.
        Both requirements and constraints are automatically parsed and written to ``requirements.txt`` and
        ``constraints.txt`` files, respectively, and stored as part of the model.
        Requirements are also written to the ``pip`` section of the model's conda environment (``conda.yaml``) file.

        warning:
        The following arguments can't be specified at the same time:
        - `conda_env`
        - `pip_requirements`
        - `extra_pip_requirements`

    metadata: dict[str, Any] | None = None
        Custom metadata dictionary passed to the model and stored in the MLmodel file.

    kwargs:
        kwargs to pass to ``stable_baselines3.{algorithm}.save`` method.


    Returns
    -------
    model_info : `ModelInfo <mlflow.models.ModelInfo>`
        A `ModelInfo <mlflow.models.model.ModelInfo>` instance that contains the metadata of the logged model.

    Examples
    --------
    ```python

        import numpy as np
        import torch
        import mlflow
        from mlflow import MlflowClient
        from mlflow.models import infer_signature

        # Define model, loss, and optimizer
        model = nn.Linear(1, 1)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

        # Create training data with relationship y = 2X
        X = torch.arange(1.0, 26.0).reshape(-1, 1)
        y = X * 2

        # Training loop
        epochs = 250
        for epoch in range(epochs):
            # Forward pass: Compute predicted y by passing X to the model
            y_pred = model(X)

            # Compute the loss
            loss = criterion(y_pred, y)

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Create model signature
        signature = infer_signature(X.numpy(), model(X).detach().numpy())

        # Log the model
        with mlflow.start_run() as run:
            mlflow.pytorch.log_model(model, "model")

            # convert to scripted model and log the model
            scripted_pytorch_model = torch.jit.script(model)
            mlflow.pytorch.log_model(scripted_pytorch_model, "scripted_model")

        # Fetch the logged model artifacts
        print(f"run_id: {run.info.run_id}")
        for artifact_path in ["model/data", "scripted_model/data"]:
            artifacts = [
                f.path for f in MlflowClient().list_artifacts(run.info.run_id, artifact_path)
            ]
            print(f"artifacts: {artifacts}")
    ```

    Output
    ```text
        run_id: 1a1ec9e413ce48e9abf9aec20efd6f71
        artifacts: ['model/data/model.pth']
        artifacts: ['scripted_model/data/model.pth']
    ```
    """
    return Model.log(
        artifact_path=artifact_path,
        flavor=mlflow_rl_tools.sb3,
        sb3_model=sb3_model,
        conda_env=conda_env,
        code_paths=code_paths,
        registered_model_name=registered_model_name,
        signature=signature,
        input_example=input_example,
        await_registration_for=await_registration_for,
        extra_files=extra_files,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        metadata=metadata,
        **kwargs,
    )


def save_model(
    sb3_model: SelfBaseAlgorithm,
    path: str,
    conda_env: dict[str, Any] | None = None,
    mlflow_model: MLflowModel | None = None,
    code_paths: list[str] | None = None,
    signature: ModelSignature | bool | None = None,
    input_example: ModelInputExample | None = None,
    extra_files: list[str] | None = None,
    pip_requirements: str | list[str] | None = None,
    extra_pip_requirements: str | list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs,
) -> None:
    """
    Save a PyTorch model to a path on the local file system.

    Parameters
    ----------
    sb3_model: SelfBaseAlgorithm
        Stable-Baselines3 model to be saved.

    path: str
        Local path where the model is to be saved.

    conda_env: dict[str, Any] | None = None
        If provided, this dictionary is included in the model's MLmodel file.
        The dictionary should contain a valid Conda environment.
        This parameter is used to specify a custom environment for the model.

    mlflow_model: MLflowModel | None = None
        MLflow model `mlflow.models.Model` this flavor is being added to.

    code_paths: list[str] | None = None
        A list of local filesystem paths to Python file dependencies (or directories containing file dependencies).
        These files are *prepended* to the system path when the model is loaded.
        Files declared as dependencies for a given model should have relative imports declared from a common root path
        if multiple files are defined with import dependencies between them to avoid import errors when loading the model.

    signature: ModelSignature | None = None
        An instance of the :py:class:`ModelSignature <mlflow.models.ModelSignature>` class
        that describes the model's inputs and outputs.
        If not specified but an ``input_example`` is supplied, a signature will be
        automatically inferred based on the supplied input example and model.
        To disable automatic signature inference when providing an input example, set ``signature`` to ``False``.
        To manually infer a model signature, call :py`infer_signature() <mlflow.models.infer_signature>`
        on datasets with valid model inputs, such as a training dataset with the target columnomitted,
        and valid model outputs, like model predictions made on the trainingdataset, for example:
        ```python
        from mlflow.models import infer_signature

        train = df.drop_column("target_label")
        predictions = ...  # compute model predictions
        signature = infer_signature(train, predictions)
        ```

    input_example: ModelInputExample | None = None
        One or several instances of valid model input. The input example is used as a hint of what data to feed the model.
        It will be converted to a Pandas DataFrame and then serialized to json using the Pandas split-oriented format,
        or a numpy array where the example will be serialized to json by converting it to a list.
        Bytes are base64-encoded. When the ``signature`` parameter is ``None``, the input example is
        used to infer a model signature.

    extra_files: list[str] | None = None
        A list of local filesystem paths to supplementary files that should be packaged with the model.
        These files are copied to the same location as the model when it is saved.
        For example, consider the following ``extra_files`` list.
        In this case, the ``"my_file1 & my_file2"`` extra file is downloaded from S3:
        ```python
        extra_files = ["s3://my-bucket/path/to/my_file1", "s3://my-bucket/path/to/my_file2"]
        ```

    pip_requirements: str | list[str] | None = None
        Either an iterable of pip requirement strings
        (e.g. ``["{{ package_name }}", "-r requirements.txt", "-c constraints.txt"]``) or the string path
        to a pip requirements file on the local filesystem (e.g. ``"requirements.txt"``).
        If provided, this describes the environment this model should be run in.
        If ``None``, a default list of requirements is inferred by `mlflow.models.infer_pip_requirements`
        from the current software environment.
        If the requirement inference fails, it falls back to using `get_default_pip_requirements`.
        Both requirements and constraints are automatically parsed and written to ``requirements.txt`` and
        ``constraints.txt`` files, respectively, and stored as part of the model.
        Requirements are alsoritten to the ``pip`` section of the model's conda environment (``conda.yaml``) file.

    extra_pip_requirements: str | list[str] | None = None
        Either an iterable of pip requirement strings
        (e.g. ``["{{ package_name }}", "-r requirements.txt", "-c constraints.txt"]``)
        or the string path to a pip requirements file on the local filesystem (e.g. ``"requirements.txt"``).
        If provided, this describes additional requirements for the model that are not included
        in the main ``pip_requirements``.
        If ``None``, no extra requirements are added to the model.
        Both requirements and constraints are automatically parsed and written to ``requirements.txt`` and
        ``constraints.txt`` files, respectively, and stored as part of the model.
        Requirements are also written to the ``pip`` section of the model's conda environment (``conda.yaml``) file.

        warning:
        The following arguments can't be specified at the same time:
        - `conda_env`
        - `pip_requirements`
        - `extra_pip_requirements`

    metadata: dict[str, Any] | None = None
        Custom metadata dictionary passed to the model and stored in the MLmodel file.

    kwargs:
        kwargs to pass to ``stable_baselines3.{algorithm}.save`` method.

    Examples
    --------
    ```python

    import os
    import mlflow
    import torch


    model = nn.Linear(1, 1)

    # Save PyTorch models to current working directory
    with mlflow.start_run() as run:
        mlflow.pytorch.save_model(model, "model")

        # Convert to a scripted model and save it
        scripted_pytorch_model = torch.jit.script(model)
        mlflow.pytorch.save_model(scripted_pytorch_model, "scripted_model")

    # Load each saved model for inference
    for model_path in ["model", "scripted_model"]:
        model_uri = f"{os.getcwd()}/{model_path}"
        loaded_model = mlflow.pytorch.load_model(model_uri)
        print(f"Loaded {model_path}:")
        for x in [6.0, 8.0, 12.0, 30.0]:
            X = torch.Tensor([[x]])
            y_pred = loaded_model(X)
            print(f"predict X: {x}, y_pred: {y_pred.data.item():.2f}")
        print("--")
    ```

    Output
    ```text
        Loaded model:
        predict X: 6.0, y_pred: 11.90
        predict X: 8.0, y_pred: 15.92
        predict X: 12.0, y_pred: 23.96
        predict X: 30.0, y_pred: 60.13
        --
        Loaded scripted_model:
        predict X: 6.0, y_pred: 11.90
        predict X: 8.0, y_pred: 15.92
        predict X: 12.0, y_pred: 23.96
        predict X: 30.0, y_pred: 60.13
    ```

    """
    return base_save_model(
        sb3_model,
        path,
        FLAVOR_NAME,
        conda_env,
        mlflow_model,
        code_paths,
        signature,
        input_example,
        extra_files,
        pip_requirements,
        extra_pip_requirements,
        metadata,
        **kwargs,
    )


def load_model(
    model_uri: str,
    dst_path: str | None = None,
    **kwargs,
) -> SelfBaseAlgorithm:
    """
    Load a PyTorch model from a local file or a run.

    Parameters
    ----------
    model_uri: str
        The location, in URI format, of the MLflow model, for example:

            - ``/Users/me/path/to/local/model``
            - ``relative/path/to/local/model``
            - ``s3://my_bucket/path/to/model``
            - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
            - ``models:/<model_name>/<model_version>``
            - ``models:/<model_name>/<stage>``

            For more information about supported URI schemes, see `Referencing Artifacts \
            <https://www.mlflow.org/docs/latest/concepts.html#artifact-locations>`_.

    dst_path: str | None = None
        The local filesystem path to which to download the model artifact.
        This directory must already exist. If unspecified, a local output path will be created.

    kwargs:
        kwargs to pass to `stable_baselines3.load` method.

    Returns
    -------
    sb3_model: SelfBaseAlgorithm
        The loaded SB3 model.

    Examples

    ```python
    :caption: Example

    import torch
    import mlflow.pytorch


    model = nn.Linear(1, 1)

    # Log the model
    with mlflow.start_run() as run:
        mlflow.pytorch.log_model(model, "model")

    # Inference after loading the logged model
    model_uri = f"runs:/{run.info.run_id}/model"
    loaded_model = mlflow.pytorch.load_model(model_uri)
    for x in [4.0, 6.0, 30.0]:
        X = torch.Tensor([[x]])
        y_pred = loaded_model(X)
        print(f"predict X: {x}, y_pred: {y_pred.data.item():.2f}")
    ```

    Output
    ```text
    predict X: 4.0, y_pred: 7.57
    predict X: 6.0, y_pred: 11.64
    predict X: 30.0, y_pred: 60.48
    ```
    """

    return base_load_model(
        model_uri, stable_baselines3, FLAVOR_NAME, dst_path, **kwargs
    )


_load_pyfunc = _base_load_pyfunc
