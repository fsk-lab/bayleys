from typing import Optional
from logging import getLogger
import numpy as np
import torch

from .config import BenchmarkConfig
from .campaign import OptimizationCampaign

from ..utils import JSONSerializable
from ..utils.cache import load_cached_embeddings, cache_embeddings
from ..molecule_library import MoleculeLibrary, MoleculeLibraryConfig


class BenchmarkCampaign(JSONSerializable):
    """
    Class to wrap a full benchmarking campaign (consisting of multiple independent optimization campaigns).

    Args:
        library (MoleculeLibrary): MoleculeLibrary object containing the candidate molecules for benchmarking. Must have
                                   associated labels for each molecule.
        config (BenchmarkConfig): Configuration for the benchmarking campaign.
        campaigns (list[OptimizationCampaign]): List of pre-initialized optimization campaigns to use for
                                                          benchmarking. If None, new campaigns will be created based on
                                                          the provided config.
        read_only (bool): Whether to set the optimization campaigns to read-only mode. In this case, library and models
                          will not be loaded explicitly, and existing data will be used.
    """
    def __init__(
            self,
            library: MoleculeLibrary,
            config: BenchmarkConfig,
            campaigns: Optional[list[OptimizationCampaign]] = None,
            read_only: bool = False,
    ):

        if library.labels is None and read_only is False:
            raise ValueError("Library must have labels for benchmarking campaigns.")
        self.library = library

        self.config = config
        self.read_only = read_only

        if campaigns is not None:
            self.campaigns = campaigns
        else:
            self.campaigns = [
                OptimizationCampaign(config=self.config.optimization_config, random_seed=i * config.random_seed, read_only=read_only)
                for i in range(self.config.num_repeats)
            ]

        self.logger = getLogger("bayleys")

    def run(self):
        """
        Run the full benchmarking campaign across all optimization campaigns.
        """
        if self.read_only:
            raise RuntimeError("Cannot run benchmarking campaign in read-only mode.")

        self.logger.info(f"{self.__class__.__name__}: Starting benchmarking with {len(self.campaigns)} independent "
                         f"optimization campaigns.")

        if self.config.optimization_config.model_config.static_embeddings is True:
            self.precompute_embeddings()

        for campaign in self.campaigns:
            campaign.run(self.library)

    def precompute_embeddings(self):
        """
        Precompute and store molecular embeddings for the entire library. Useful for static embedding scenarios.
        First, checks if the requested embeddings (encoder + library) are already cached in `cache_dir`.
        """
        if self.read_only:
            raise RuntimeError("Cannot precompute embeddings in read-only mode.")

        try:
            embeddings = load_cached_embeddings(self.campaigns[0].surrogate_model.encoder.name, self.library.name)
            self.library.embeddings = embeddings
            self.logger.info(f"{self.__class__.__name__}: Loaded cached embeddings from cache for "
                             f"encoder {self.campaigns[0].surrogate_model.encoder.name} and library "
                             f"{self.library.name}.")
        except FileNotFoundError:
            self.library.embeddings = self.campaigns[0].surrogate_model.compute_embeddings(self.library)
            cache_embeddings(
                self.library.embeddings,
                self.campaigns[0].surrogate_model.encoder.name,
                self.library.name
            )
            self.logger.info(f"{self.__class__.__name__}: Cached embeddings for encoder "
                             f"{self.campaigns[0].surrogate_model.encoder.name} and library {self.library.name}.")

        embedding_nan_mask = torch.isnan(self.library.embeddings)
        num_affected_rows = torch.any(embedding_nan_mask, dim=1).sum().item()
        if num_affected_rows > 0:
            self.logger.warning(f"{self.__class__.__name__}: Found {num_affected_rows} rows with NaN values in the "
                                f"precomputed embeddings for library {self.library.name}. These values will be replaced"
                                f" with zeros.")
            self.library.embeddings[embedding_nan_mask] = 0.0

        # For consistency, ensure all campaigns have the same embedding dimension attribute
        for campaign in self.campaigns:
            if campaign.surrogate_model.encoder.embedding_dim != self.library.embeddings.shape[1]:
                campaign.surrogate_model.encoder.embedding_dim = self.library.embeddings.shape[1]

    def to_dict(self) -> dict:
        """
        Serialize the benchmarking campaign configuration to a dictionary (json-compatible).
        """
        return {
            "library": self.library.to_dict(include_data=False),
            "config": self.config.to_dict(),
            "runs": {
                campaign.random_seed: campaign.acquired_candidates.to_dict(include_data=True)
                for campaign in self.campaigns
            },
        }

    @classmethod
    def from_dict(cls, data: dict, read_only: bool = False, **kwargs) -> "BenchmarkCampaign":
        """
        Deserialize a benchmarking campaign from a dictionary.

        Args:
            data (dict): Dictionary containing the serialized benchmarking campaign.
            read_only (bool): Whether to set the campaign to read-only mode.

        Returns:
            BenchmarkCampaign: Deserialized BenchmarkCampaign object.
        """
        local_dir = kwargs.get("local_dir", None)

        library_config = MoleculeLibraryConfig.from_dict(data["library"]["config"])
        library = MoleculeLibrary.from_config(config=library_config, local_dir=local_dir, read_only=read_only)
        config = BenchmarkConfig.from_dict(data["config"])

        campaigns = []
        for seed, run_data in data["runs"].items():
            acquired_candidates = MoleculeLibrary.from_dict(run_data)
            campaign = OptimizationCampaign(
                config=config.optimization_config,
                random_seed=int(seed),
                acquired_candidates=acquired_candidates,
                read_only=read_only
            )
            campaigns.append(campaign)

        return cls(library=library, config=config, campaigns=campaigns, read_only=read_only)

    @property
    def all_results(self) -> dict[str, any]:
        """
        Aggregate results from all optimization campaigns in the benchmarking campaign, including
            - SMILES strings of all acquired candidates.
            - Their corresponding labels from the library.

        Returns:
            dict[str, any]: Dictionary containing aggregated results.
        """
        all_smiles = [campaign.acquired_candidates.smiles for campaign in self.campaigns]
        all_labels = [campaign.acquired_candidates.labels for campaign in self.campaigns]

        return {
            "smiles": np.stack(all_smiles, axis=0),
            "labels": np.concatenate(all_labels, axis=1).T
        }

    @property
    def name(self) -> str:
        """
        Returns the name of the benchmarking campaign, derived from the library and configuration.

        Returns:
            str: Name of the benchmarking campaign.
        """
        surrogate_name = self.campaigns[0].surrogate_model.name
        acquisition_name = self.campaigns[0].acquisition.name

        return (f"{self.library.name}_{surrogate_name}_{acquisition_name}_{self.config.optimization_config.budget}_"
                f"{self.config.num_repeats}")
