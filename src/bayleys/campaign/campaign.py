import copy
import time
from logging import getLogger

from .config import OptimizationConfig

from ..molecule_library import MoleculeLibrary
from ..utils import set_random_seeds
from ..acquisition import get_seed_experiment_indices
from ..surrogate import SurrogateModel


class OptimizationCampaign(object):
    """
    Class to represent a single optimization campaign.
    """
    def __init__(
            self,
            config: OptimizationConfig,
            random_seed: int = 42,
            acquired_candidates: MoleculeLibrary = None,
            read_only: bool = False
    ):

        self.config = config
        self.random_seed = random_seed
        self.read_only = read_only

        if not read_only:
            self.surrogate_model = SurrogateModel(config.encoder_config, config.model_config)
            self.acquisition = config.acquisition_config.get_acquirer()
        else:
            self.surrogate_model, self.acquisition = None, None

        self.acquired_candidates = acquired_candidates

        self.logger = getLogger("bayleys")

    def run(self, library: MoleculeLibrary):
        """
        Run the optimization campaign on the given molecule library.

        Args:
            library: MoleculeLibrary object containing the candidate molecules.
        """
        if self.read_only:
            raise RuntimeError("Cannot run optimization campaign in read-only mode.")

        if self.acquired_candidates is not None:
            self.logger.error("Acquired candidates already exist. Cannot run campaign again.")
            raise ValueError("Acquired candidates already exist. Cannot run campaign again.")

        self.logger.info(f"{self.__class__.__name__}: Starting optimization campaign with the following settings:"
                         f"{self.config.to_dict()}")

        start_time = time.time()
        set_random_seeds(self.random_seed)

        virtual_library = copy.deepcopy(library)
        virtual_library.shuffle()

        self.acquired_candidates = MoleculeLibrary.new(
            dataset_name=f"{library.dataset_config.name}_acquired_candidates_seed{self.random_seed}",
            objective_name=library.dataset_config.label_name,
            objective_direction=library.dataset_config.objective_direction,
            smiles_dtype=library.smiles.dtype
        )

        seed_indices = get_seed_experiment_indices(
            library=virtual_library,
            num_experiments=self.config.acquisition_config.batch_size,
            strategy=self.config.seeding_strategy,
            base_seed=self.random_seed
        )
        smiles, _, embedding, label = virtual_library.pop(seed_indices)
        self.acquired_candidates.extend(smiles=smiles, embeddings=embedding, labels=label)

        self.logger.debug(f"{self.__class__.__name__}: Initial experiments completed with "
                          f"{len(self.acquired_candidates)} molecules.")

        while len(self.acquired_candidates) < self.config.budget:
            try:
                self.surrogate_model.train_model(self.acquired_candidates)
                self.acquisition.update(self.acquired_candidates)
                mean, var, embeddings = self.surrogate_model.predict(virtual_library)
                acquired_indices = self.acquisition.acquire(mean, var, virtual_library.embeddings)
            except Exception as e:
                self.logger.error(f"{self.__class__.__name__}: Exception in model training or acquisition function"
                                  f" optimization: {e}. Picking random candidates instead.")
                acquired_indices = get_seed_experiment_indices(
                    library=virtual_library,
                    num_experiments=self.config.acquisition_config.batch_size,
                    strategy=self.config.seeding_strategy,
                    base_seed=self.random_seed + len(self.acquired_candidates)
                )

            smiles, _, embedding, label = virtual_library.pop(acquired_indices)
            self.acquired_candidates.extend(smiles=smiles, embeddings=embedding, labels=label)

        self.logger.info(f"{self.__class__.__name__}: Optimization campaign completed in {time.time() - start_time:.2f}"
                         f" seconds.")
