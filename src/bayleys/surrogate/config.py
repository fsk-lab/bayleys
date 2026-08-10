from dataclasses import dataclass

from .base import Model, ModelArchitecture, TrainingConfig
from .gp import GaussianProcessModel
from .nn import NeuralNetworkModel
from .laplace import LaplaceNeuralNetworkModel
from .nn_ensemble import NeuralNetworkEnsembleModel
from .random_forest import RandomForestModel

from ..utils import AbstractConfig, non_abstract_subclasses


MODEL_FACTORY = {cls.__name__: cls for cls in non_abstract_subclasses(Model)}
MODEL_ARCHITECTURE_FACTORY = {cls.__name__: cls.default_architecture.__class__ for cls in non_abstract_subclasses(Model)}


@dataclass
class ModelConfig(AbstractConfig):

    model_type: str
    model_architecture: ModelArchitecture
    train_config: TrainingConfig

    static_embeddings: bool = True
    inference_batch_size: int = 1024

    def __post_init__(self):
        """
        Validates the surrogate model configuration after initialization by performing the following checks:
          - Ensures that the specified model_type exists in the SUBCLASS_FACTORY.
        """
        if self.model_type not in MODEL_FACTORY:
            raise ValueError(f"Surrogate model type '{self.model_type}' is not recognized!")

    def get_model(self, encoder: any) -> Model:
        """
        Instantiates and returns the surrogate model based on the configuration.

        Args:
            encoder: The molecule encoder to be used by the surrogate model.

        Returns:
            An instance of the specified surrogate model.
        """
        model_class = MODEL_FACTORY[self.model_type]
        model = model_class(
            encoder=encoder,
            architecture=self.model_architecture,
            training_config=self.train_config,
            static_embeddings=self.static_embeddings,
        )
        return model

    @classmethod
    def from_dict(cls, config_dict: dict):
        """
        Creates a ModelConfig instance from a dictionary representation. Overrides the base method to handle
        model-specific configuration classes.

        Args:
            config_dict (dict): A dictionary containing the configuration parameters.

        Returns:
            ModelConfig: An instance of ModelConfig.
        """
        model_type = config_dict["model_type"]
        if model_type not in MODEL_FACTORY:
            raise ValueError(f"Surrogate model type '{model_type}' is not recognized!")

        architecture = MODEL_ARCHITECTURE_FACTORY[model_type].from_dict(config_dict.get("model_architecture", {}))

        train_config_dict = config_dict.get("train_config")
        if train_config_dict is None:
            train_config = MODEL_FACTORY[model_type].default_training_config
        else:
            train_config = TrainingConfig.from_dict(config_dict.get("train_config"))

        return cls(
            model_type=model_type,
            model_architecture=architecture,
            train_config=train_config,
            static_embeddings=config_dict.get("static_embeddings", True),
            inference_batch_size=config_dict.get("inference_batch_size", 1024),
        )
