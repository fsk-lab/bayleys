from dataclasses import dataclass

from .acquisition_function import ucb, ei, logEI, random
from .base import Acquirer
from .top_k_acquisition import TopKAcquisition
from .ensemble_acquisition import EnsembleAcquisition
from .diversity_acquisition import DiversityAcquisition

from ..utils import AbstractConfig, non_abstract_subclasses


ACQUIRER_FACTORY = {cls.__name__: cls for cls in non_abstract_subclasses(Acquirer)}
ACQF_FACTORY = {"ucb": ucb, "ei": ei, "logEI": logEI, "random": random}


@dataclass
class AcquisitionConfig(AbstractConfig):
    """
    Configuration class for acquisition strategies in virtual library search.
    """
    type: str
    acquisition_function: str
    acquisition_function_kwargs: dict = None
    batch_size: int = 1
    max_exploration: float = 1.0
    diversity_weight: float = 0.5

    def __post_init__(self):
        """
        Validates the acquisition configuration after initialization by performing the following checks:
          - Ensures that the specified acquisition type exists in the ACQUIRER_FACTORY.
          - Initializes acquisition_function_kwargs to an empty dictionary if it is None.
        """
        if self.type not in ACQUIRER_FACTORY:
            raise ValueError(f"Acquisition type '{self.type}' is not recognized!")

        if self.acquisition_function not in ACQF_FACTORY:
            raise ValueError(f"Acquisition function '{self.acquisition_function}' is not recognized!")

        if self.acquisition_function_kwargs is None:
            self.acquisition_function_kwargs = {}

    def get_acquirer(self) -> Acquirer:
        """
        Instantiates and returns the acquirer based on the configuration.

        Returns:
            An instance of the specified acquirer.
        """
        acquirer_class = ACQUIRER_FACTORY[self.type]

        # hacky way to pass additional args to specific acquirers, should be cleaned up later
        additional_kwargs = {}
        if "max_exploration" in acquirer_class.__init__.__code__.co_varnames:
            additional_kwargs["max_exploration"] = self.max_exploration
        if "diversity_weight" in acquirer_class.__init__.__code__.co_varnames:
            additional_kwargs["diversity_weight"] = self.diversity_weight

        acquisition_function = ACQF_FACTORY[self.acquisition_function]
        acquirer = acquirer_class(
            acquisition_function=acquisition_function,
            acquisition_function_kwargs=self.acquisition_function_kwargs,
            batch_size=self.batch_size,
            **additional_kwargs
        )
        return acquirer


