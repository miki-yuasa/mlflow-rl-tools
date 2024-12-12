"""
The ``mlflow_rl_tools.sb3`` module provides an API for logging and loading Stable-Baselines3 models. 
This module exports Stable-Baselines3 models with the following flavors:

PyTorch (native) format
    This is the main flavor that can be loaded back into Stable-Baselines3.
:py:mod:`mlflow.pyfunc`
    Produced for use by generic pyfunc-based deployment tools and batch inference.
"""

import os
import posixpath
import shutil
from typing import Any, TypeVar
import warnings

import mlflow
from mlflow import pyfunc
from mlflow.environment_variables import MLFLOW_DEFAULT_PREDICTION_DEVICE
from mlflow.exceptions import MlflowException
from mlflow.ml_package_versions import _ML_PACKAGE_VERSIONS
from mlflow.models import Model, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import _infer_signature_from_input_example
from mlflow.models.utils import ModelInputExample, _save_example, _Example
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.autologging_utils import autologging_integration, safe_patch
from mlflow.utils.checkpoint_utils import download_checkpoint_artifact
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
    _CONDA_ENV_FILE_NAME,
    _CONSTRAINTS_FILE_NAME,
    _PYTHON_ENV_FILE_NAME,
    _REQUIREMENTS_FILE_NAME,
    _mlflow_conda_env,
    _process_conda_env,
    _process_pip_requirements,
    _PythonEnv,
    _validate_env_arguments,
)
from mlflow.utils.file_utils import (
    TempDir,
    get_total_file_size,
    write_to,
)
from mlflow.utils.model_utils import (
    _add_code_from_conf_to_system_path,
    _get_flavor_configuration,
    _validate_and_copy_code_paths,
    _validate_and_prepare_target_save_path,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement
import stable_baselines3
from stable_baselines3.common.base_class import SelfBaseAlgorithm, BaseAlgorithm
import torch
import yaml

from mlflow_rl_tools.sb3.wrapper import ModelWrapper

FLAVOR_NAME = "sb3"

_SERIALIZED_SB3_MODEL_FILE_NAME = "model.zip"
_SB3_STATE_DICT_FILE_NAME = "state_dict.pth"
_SB3_ALGO_CLASS_FILE_NAME = "algo_name.txt"
_EXTRA_FILES_KEY = "extra_files"
_SB3_CPU_DEVICE_NAME = "cpu"
_SB3_DEFAULT_GPU_DEVICE_NAME = "cuda"

_MODEL_DATA_SUBPATH = "data"

MLflowModel = TypeVar("MLflowModel", bound=Model)


def get_default_pip_requirements():
    """
    Get default pip requirements for MLflow Models produced by this flavor.

    Returns
    -------
    default_pip_requirements : list[str]
        List of default pip requirements for MLflow Models produced by this flavor.
        Calls to `save_model()` and `log_model()` produce a pip environment.
        This pip environment, at minimum, contains these requirements.
    """
    default_requirements: list[str] = list(
        map(_get_pinned_requirement, ["stable-baselines3", "gymnasium", "torch"])
    )

    return default_requirements


def get_default_conda_env():
    """
    Returns:
        The default Conda environment as a dictionary for MLflow Models produced by calls to
        :func:`save_model()` and :func:`log_model()`.

    .. code-block:: python
        :caption: Example

        import mlflow

        # Log PyTorch model
        with mlflow.start_run() as run:
            mlflow.pytorch.log_model(model, "model", signature=signature)

        # Fetch the associated conda environment
        env = mlflow.pytorch.get_default_conda_env()
        print(f"conda env: {env}")

    .. code-block:: text
        :caption: Output

        conda env {'name': 'mlflow-env',
                   'channels': ['conda-forge'],
                   'dependencies': ['python=3.8.15',
                                    {'pip': ['torch==1.5.1',
                                             'mlflow',
                                             'cloudpickle==1.6.0']}]}
    """
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements())


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
):
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
        flavor=mlflow.pytorch,
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
    flavor_name: str = FLAVOR_NAME,
    mlmodel_file_name: str = MLMODEL_FILE_NAME,
    model_data_subpath: str = _MODEL_DATA_SUBPATH,
    _serialized_sb3_model_file_name: str = _SERIALIZED_SB3_MODEL_FILE_NAME,
    _sb3_algo_class_file_name: str = _SB3_ALGO_CLASS_FILE_NAME,
    _conda_env_file_name: str = _CONDA_ENV_FILE_NAME,
    _python_env_file_name: str = _PYTHON_ENV_FILE_NAME,
    _extra_files_key: str = _EXTRA_FILES_KEY,
    _requirements_file_name: str = _REQUIREMENTS_FILE_NAME,
    _constraints_file_name: str = _CONSTRAINTS_FILE_NAME,
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

    flavor_name: str = "sb3"
        The name of the flavor that is being added to the model.

    mlmodel_file_name: str = "MLmodel"
        The name of the MLmodel file.

    model_data_subpath: str = "data"
        The subdirectory within the model's root directory where data is stored.

    _serialized_sb3_model_file_name: str = "model.zip"
        The name of the serialized SB3 model file.

    _sb3_algo_class_file_name: str = "algo_name.txt"
        The name of the file that contains the SB3 algorithm class name.

    _conda_env_file_name: str = "conda.yaml"
        The name of the Conda environment file.

    _python_env_file_name: str = "python.yaml"
        The name of the Python environment file.

    _extra_files_key: str = "extra_files"
        The key in the MLmodel file's flavor configuration that specifies the paths to extra files.

    _requirements_file_name: str = "requirements.txt"
        The name of the pip requirements file.

    _constraints_file_name: str = "constraints.txt"
        The name of the pip constraints file.

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
    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

    if not isinstance(sb3_model, BaseAlgorithm):
        raise TypeError(
            f"sb3_model must be an instance of Stable-Baselines3's BaseAlgorithm class. "
            f"Received {type(sb3_model)}."
        )

    if mlflow_model is None:
        mlflow_model = Model()
    saved_example: _Example | None = _save_example(input_example, input_example, path)

    if signature is None and saved_example is not None:
        wrapped_model = ModelWrapper(sb3_model)
        signature: ModelSignature | None = _infer_signature_from_input_example(
            saved_example, wrapped_model
        )
    elif signature is False:
        signature = None
    else:
        pass

    if signature is not None:
        mlflow_model.signature = signature
    else:
        pass

    if metadata is not None:
        mlflow_model.metadata = metadata
    else:
        pass

    code_dir_subpath: str = _validate_and_copy_code_paths(code_paths, path)

    model_data_path = os.path.join(path, model_data_subpath)
    os.makedirs(model_data_path)

    # Persist the algo class name as a file in the model's `data` directory.
    # This is necessary because the `data` directory is the only available parameter to `_load_pyfunc`,
    # and it does not contain the MLmodel configuration;
    # therefore, it is not sufficient to place the module name in the MLmodel
    #
    # TODO: Stop persisting this information to the filesystem once we have a mechanism for
    # supplying the MLmodel configuration to `mlflow.pytorch._load_pyfunc`
    algo_name_path = os.path.join(model_data_path, _sb3_algo_class_file_name)
    with open(algo_name_path, "w") as f:
        f.write(sb3_model.__class__.__name__)

    # Save SB3 model
    model_path = os.path.join(model_data_path, _serialized_sb3_model_file_name)
    sb3_model.save(model_path, **kwargs)

    sb3serve_artifacts_config = {}

    if extra_files:
        sb3serve_artifacts_config[_extra_files_key] = []
        if not isinstance(extra_files, list):
            raise TypeError("Extra files argument should be a list")

        with TempDir() as tmp_extra_files_dir:
            for extra_file in extra_files:
                _download_artifact_from_uri(
                    artifact_uri=extra_file, output_path=tmp_extra_files_dir.path()
                )
                rel_path = posixpath.join(
                    _extra_files_key, os.path.basename(extra_file)
                )
                sb3serve_artifacts_config[_extra_files_key].append({"path": rel_path})
            shutil.move(
                tmp_extra_files_dir.path(),
                posixpath.join(path, _extra_files_key),
            )

    mlflow_model.add_flavor(
        flavor_name,
        model_data=model_data_subpath,
        sb3_version=str(stable_baselines3.__version__),
        code=code_dir_subpath,
        **sb3serve_artifacts_config,
    )
    pyfunc.add_to_model(
        mlflow_model,
        loader_module="mlflow_rl_tools.sb3",
        data=model_data_subpath,
        code=code_dir_subpath,
        conda_env=_conda_env_file_name,
        python_env=_python_env_file_name,
        model_config={"device": None},
    )
    if size := get_total_file_size(path):
        mlflow_model.model_size_bytes = size
    else:
        pass

    mlflow_model.save(os.path.join(path, mlmodel_file_name))

    if conda_env is None:
        if pip_requirements is None:
            default_reqs = get_default_pip_requirements()
            # To ensure `_load_pyfunc` can successfully load the model during the dependency
            # inference, `mlflow_model.save` must be called beforehand to save an MLmodel file.
            inferred_reqs = mlflow.models.infer_pip_requirements(
                model_data_path,
                flavor=flavor_name,
                fallback=default_reqs,
            )
            default_reqs = sorted(set(inferred_reqs).union(default_reqs))
        else:
            default_reqs = None

        conda_env, pip_requirements, pip_constraints = _process_pip_requirements(
            default_reqs,
            pip_requirements,
            extra_pip_requirements,
        )
    else:
        conda_env, pip_requirements, pip_constraints = _process_conda_env(conda_env)

    with open(os.path.join(path, _conda_env_file_name), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    # Save `constraints.txt` if necessary
    if pip_constraints:
        write_to(os.path.join(path, _constraints_file_name), "\n".join(pip_constraints))
    else:
        pass

    write_to(os.path.join(path, _requirements_file_name), "\n".join(pip_requirements))

    _PythonEnv.current().to_yaml(os.path.join(path, _python_env_file_name))


def _load_model(
    path: str,
    _sb3_algo_class_file_name: str = _SB3_ALGO_CLASS_FILE_NAME,
    **kwargs,
) -> SelfBaseAlgorithm:
    """
    Load a PyTorch model from a local file.

    Parameters
    ----------
    path: str
        Local filesystem path to the model.
    _sb3_algo_class_file_name: str = _SB3_ALGO_CLASS_FILE_NAME
        The name of the file that contains the SB3 algorithm class name.
    kwargs:
        Additional kwargs to pass to the SB3 model's `load` method.

    Returns
    -------
    sb3_model: SelfBaseAlgorithm
        The loaded SB3 model.
    """
    import torch

    if os.path.isdir(path):
        # `path` is a directory containing a serialized PyTorch model and a text file containing
        # information about the pickle module that should be used by PyTorch to load it
        model_path = os.path.join(path, "model.zip")
    else:
        model_path = path

    algo_name_path = os.path.join(
        os.path.dirname(model_path), _sb3_algo_class_file_name
    )
    with open(algo_name_path, "r") as f:
        algo_name = f.read().strip()

    sb3_model: SelfBaseAlgorithm = getattr(stable_baselines3, algo_name).load(
        model_path, **kwargs
    )

    return sb3_model


def load_model(model_uri: str, dst_path: str | None = None, **kwargs):
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

    local_model_path = _download_artifact_from_uri(
        artifact_uri=model_uri, output_path=dst_path
    )
    sb3_conf = _get_flavor_configuration(
        model_path=local_model_path, flavor_name=FLAVOR_NAME
    )
    _add_code_from_conf_to_system_path(local_model_path, sb3_conf)

    if stable_baselines3.__version__ != sb3_conf["sb3_version"]:
        warnings.warn(
            "Stored model version '%s' does not match installed Stable Baselines3 version '%s'"
            % (sb3_conf["sb3_version"], stable_baselines3.__version__),
        )

    sb3_model_artifacts_path = os.path.join(local_model_path, sb3_conf["model_data"])
    return _load_model(path=sb3_model_artifacts_path, **kwargs)


def _load_pyfunc(path, model_config: dict[str, Any] = None):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_model``.

    Parameters
    ----------
    path : str
        The path to the MLflow model.
    model_config : dict
        The model configuration to load SB3 models.

    Returns
    -------
    ModelWrapper
        A PyFunc model instance.
    """

    pyfunc_model = ModelWrapper(_load_model(path, **model_config))

    return pyfunc_model
