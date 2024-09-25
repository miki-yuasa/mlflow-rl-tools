# mlflow-rl-tools
RL-related tools for MLflow
## Installation
This package uses Poetry as the package manager and is pip-installable.
You can simply run `pip install` to add to your dependency or `poetry install` to contribute.

## Usage
Currently, `mlflow-rl-tools` provides a logger and a wrapper for Stable-Baselines3.
You can use `mlflow_tl_tools.sb3.log.MLflowOutputFormat` to log model metrics in MLflow and `mlflow_tl_tools.sb3.wrapper.ModelWrapper` to wrap your SB3 models.

## Examples
Refer to this [repo](https://github.com/Tran-Research-Group/mlflow-tutorial) to see how to use this package.

## Contribution
### Documentation
Documentation and docstring styles should follow the PyTorch style.

### Coding Styles
Type annotations should be used whenever possible.

### Git Flow
- `main`: released versions.
- `dev`: development branch.
- `feat/{issue #}_{3_word_description_of_the_issue}`: feature branch, branched from and merged to `dev`. 
- `hotfix/{issue #}_{3_word_description_of_the_issue}`: hotfix branch to fix urgent bugs, branched from and merged to `dev`.