from typing import Any
from mlflow.pyfunc import PythonModel, PythonModelContext
from numpy.typing import NDArray
from stable_baselines3.common.base_class import SelfBaseAlgorithm


class ModelWrapper(PythonModel):
    """
    Model wrapper for Stable-Baselines3 models.
    """

    def __init__(self, model: SelfBaseAlgorithm):
        """
        Initialize the model wrapper.

        Parameters
        ----------
        model: SelfBaseAlgorithm
            The Stable-Baselines3 model instance to wrap.
        """
        self.model: SelfBaseAlgorithm = model

    def predict(
        self, context: PythonModelContext, model_input: NDArray | dict[str, NDArray]
    ) -> tuple[NDArray, tuple[NDArray] | None]:
        """
        Make predictions using the wrapped model.

        Parameters
        ----------
        context: PythonModelContext
            The context object for the model. Automatically passed by MLflow.
        model_input: NDArray | dict[str, NDArray]
            The input observation(s) to make predictions on.

        Returns
        -------
        action: NDArray
            The predicted action(s).
        next_hidden_state: tuple[NDArray] | None
            The next hidden state(s) of the model used in recurrent policy, if applicable.
        """
        return self.model.predict(model_input)

    # def load_context(self, context: PythonModelContext) -> None:
    #     """
    #     Load the model context.

    #     Parameters
    #     ----------
    #     context: PythonModelContext
    #         The context object for the model. Automatically passed by MLflow.
    #     """
    #     self.model.load(context.artifacts["model"])

    def call_model_method(self, method_name: str, *args, **kwargs) -> Any:
        """
        Call a method of the wrapped model.

        Parameters
        ----------
        method_name: str
            The name of the method to call.
        *args
            Positional arguments to pass to the method.
        **kwargs
            Keyword arguments to pass to the method.

        Returns
        -------
        Any
            The return value of the method call.
        """
        return getattr(self.model, method_name)(*args, **kwargs)
