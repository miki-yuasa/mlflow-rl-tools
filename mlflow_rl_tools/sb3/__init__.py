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
from types import ModuleType
from typing import Any, TypeVar

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
from mlflow.pytorch import pickle_module as mlflow_pytorch_pickle_module
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
import yaml

from mlflow_rl_tools.sb3 import sb3_pickle_module
from mlflow_rl_tools.sb3.wrapper import ModelWrapper

FLAVOR_NAME = "sb3"

_SERIALIZED_SB3_MODEL_FILE_NAME = "model.zip"
_SB3_STATE_DICT_FILE_NAME = "state_dict.pth"
_PICKLE_MODULE_INFO_FILE_NAME = "pickle_module_info.txt"
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
        Calls to :func:`save_model()` and :func:`log_model()` produce a pip environment.
        This pip environment, at minimum, contains these requirements.
    """
    default_requirements: list[str] = list(
        map(
            _get_pinned_requirement,
            [
                "stable-baselines3",
                # We include CloudPickle in the default environment because
                # it's required by the default pickle module used by `save_model()`
                # and `log_model()`: `mlflow.pytorch.pickle_module`.
                "cloudpickle",
            ],
        )
    )

    return default_requirements


def log_model(
    sb3_model: SelfBaseAlgorithm,
    artifact_path: str,
    conda_env: dict[str, Any] | None = None,
    code_paths: list[str] | None = None,
    pickle_module=None,
): ...


def save_model(
    sb3_model: SelfBaseAlgorithm,
    path: str,
    conda_env: dict[str, Any] | None = None,
    mlflow_model: MLflowModel | None = None,
    code_paths: list[str] | None = None,
    pickle_module: ModuleType | None = None,
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
    _pickle_module_info_file_name: str = _PICKLE_MODULE_INFO_FILE_NAME,
    _conda_env_file_name: str = _CONDA_ENV_FILE_NAME,
    _python_env_file_name: str = _PYTHON_ENV_FILE_NAME,
    _extra_files_key: str = _EXTRA_FILES_KEY,
    _requirements_file_name: str = _REQUIREMENTS_FILE_NAME,
    _constraints_file_name: str = _CONSTRAINTS_FILE_NAME,
    **kwargs,
) -> None:
    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

    pickle_module = pickle_module or sb3_pickle_module

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

    # Persist the pickle module name as a file in the model's `data` directory. This is necessary
    # because the `data` directory is the only available parameter to `_load_pyfunc`, and it
    # does not contain the MLmodel configuration; therefore, it is not sufficient to place
    # the module name in the MLmodel
    #
    # TODO: Stop persisting this information to the filesystem once we have a mechanism for
    # supplying the MLmodel configuration to `mlflow.sb3._load_pyfunc`
    pickle_module_path = os.path.join(model_data_path, _pickle_module_info_file_name)
    with open(pickle_module_path, "w") as f:
        f.write(pickle_module.__name__)
    # Save SB3 model
    model_path = os.path.join(model_data_path, _serialized_sb3_model_file_name)
    sb3_model.save(model_path)

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
        pytorch_version=str(stable_baselines3.__version__),
        code=code_dir_subpath,
        **sb3serve_artifacts_config,
    )
    pyfunc.add_to_model(
        mlflow_model,
        loader_module="mlflow_rl_tools.sb3",
        data=model_data_subpath,
        pickle_module_name=pickle_module.__name__,
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
