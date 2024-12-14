import pytest

import gymnasium as gym
import mlflow
from mlflow.models.model import ModelInfo
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.base_class import SelfBaseAlgorithm
from sb3_contrib import TQC

from mlflow_rl_tools import sb3_contrib

# Before running the tests, make sure MLflow server is running with the following command:
# mlflow server --port 5000

REGISTERED_MODEL_NAME = "test_sb3_contrib_log_model"
REGISTERED_MODEL_NAME_VEC = "test_sb3_contrib_log_model_vec"

registered_model_names: list[str] = [REGISTERED_MODEL_NAME, REGISTERED_MODEL_NAME_VEC]

envs: list[gym.Env] = [
    gym.make("MountainCarContinuous-v0"),
    SubprocVecEnv([lambda: gym.make("MountainCarContinuous-v0") for _ in range(4)]),
]

models: list[TQC] = [TQC("MlpPolicy", envs, verbose=1) for envs in envs]


def setup_module(module):
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("test_sb3_contrib")


@pytest.mark.parametrize("model, model_name", zip(models, registered_model_names))
def test_sb3_contrib_log_model(model: TQC, model_name: str):
    with mlflow.start_run():
        model_info: ModelInfo = sb3_contrib.log_model(
            model, "model", registered_model_name=model_name
        )
        active_run_id = mlflow.active_run().info.run_id

    assert isinstance(model_info, ModelInfo)
    assert model_info.run_id == active_run_id


@pytest.mark.parametrize("model_name, env", zip(registered_model_names, envs))
def test_sb3_contrib_load_model(model_name: str, env: gym.Env):
    model: SelfBaseAlgorithm = sb3_contrib.load_model(
        f"models:/{model_name}/latest", env=env
    )
    assert model is not None

    if isinstance(env, SubprocVecEnv):
        obs = env.reset()
        action, _ = model.predict(obs)
        # Check the size of the action
        assert action.size == env.num_envs
    else:
        obs, _ = env.reset()
        action, _ = model.predict(obs)
        # Check the size of the action
        assert action.shape == env.action_space.shape
