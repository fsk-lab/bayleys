from typing import Optional, Literal
from dataclasses import dataclass, replace
import argparse
import json

from ..config import BayleysConfig
from ..utils import AbstractConfig
from ..encoders import MoleculeEncoderConfig, ENCODER_FACTORY
from ..surrogate.config import ModelConfig, MODEL_FACTORY
from ..acquisition import AcquisitionConfig, ACQUIRER_FACTORY, ACQF_FACTORY


@dataclass
class OptimizationConfig(AbstractConfig):
    """
    Configuration class for optimization campaigns.
    """
    encoder_config: MoleculeEncoderConfig
    model_config: ModelConfig
    acquisition_config: AcquisitionConfig

    budget: int
    seeding_strategy: Literal["random"] = "random"  # can be extended later


@dataclass
class BenchmarkConfig(AbstractConfig):
    """
    Configuration class for benchmarking virtual library search campaigns.
    """
    optimization_config: OptimizationConfig
    bayleys_config: BayleysConfig = None
    num_repeats: int = 20
    random_seed: int = 42

    def __post_init__(self):
        if self.bayleys_config is not None:
            self.bayleys_config.apply()

    @classmethod
    def from_cli(cls) -> "BenchmarkConfig":
        """
        Creates a BenchmarkConfig instance from command-line interface (CLI) arguments.

        Returns:
            BenchmarkConfig: An instance of BenchmarkConfig populated with CLI arguments.
        """
        parser = argparse.ArgumentParser(description="Benchmarks for Virtual Library Search with BAYLEYS")
        parser.add_argument("--random_seed", type=int, default=42, help="Base random seed for reproducibility.")
        parser.add_argument("--num_repeats", type=int, default=20, help="Number of benchmark repeats.")
        parser.add_argument("--budget", type=int, default=1000, help="Optimization budget for each campaign.")
        parser.add_argument("--seed_strategy", type=str, default="random", help="Strategy for selecting initial experiments (specified in `bayleys.utils.random`).")
        parser.add_argument("--batch_size", type=int, default=50, help="Batch size for acquisition.")
        parser.add_argument("--inference_batch_size", type=int, default=1024, help="Batch size for model inference during acquisition.")
        parser.add_argument("--encoder", type=str, default="MorganFingerprintEncoder", help="Type of molecule encoder to use (specified in `bayleys.encoders`).")
        parser.add_argument("--encoder_kwargs", type=str, default="{}", help="JSON string of keyword arguments for the molecule encoder.")
        parser.add_argument("--static_embeddings", type=bool, default=True, help="Whether to use static embeddings for the surrogate model.")
        parser.add_argument("--model", type=str, default="GaussianProcessModel", help="Type of surrogate model to use (specified in `bayleys.surrogate`).")
        parser.add_argument("--model_architecture", type=str, default="{}", help="JSON string of model architecture parameters.")
        parser.add_argument("--train_config", type=str, default="{}", help="JSON string of training configuration parameters.")
        parser.add_argument("--acquisition", type=str, default="TopKAcquisition", help="Type of acquisition strategy to use (specified in `bayleys.acquisition`).")
        parser.add_argument("--acquisition_function", type=str, default="log_ei", help="Acquisition function to use (specified in `bayleys.acquisition.acquisition_functions`).")
        parser.add_argument("--exploration", type=float, default=0.1, help="Exploration parameter for the acquisition function.")
        parser.add_argument("--log_level", type=str, default="INFO", help="Logging level for BAYLEYS.")
        parser.add_argument("--log_file", type=str, default=None, help="Log file path for BAYLEYS.")
        parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for BAYLEYS.")
        parser.add_argument("--device", type=str, default=None, help="Computation device for BAYLEYS (e.g., 'cpu' or 'cuda').")

        args = parser.parse_args()

        if args.model not in MODEL_FACTORY:
            raise ValueError(f"Surrogate model type '{args.model}' is not recognized!")
        if args.encoder not in ENCODER_FACTORY:
            raise ValueError(f"Molecule encoder type '{args.encoder}' is not recognized!")
        if args.acquisition not in ACQUIRER_FACTORY:
            raise ValueError(f"Acquisition type '{args.acquisition}' is not recognized!")
        if args.acquisition_function not in ACQF_FACTORY:
            raise ValueError(f"Acquisition function '{args.acquisition_function}' is not recognized!")

        default_architecture = MODEL_FACTORY[args.model].default_architecture
        model_architecture = replace(default_architecture, **json.loads(args.model_architecture))
        default_training_config = MODEL_FACTORY[args.model].default_training_config
        train_config = replace(default_training_config, **json.loads(args.train_config))

        benchmark_config = cls(
            num_repeats=args.num_repeats,
            random_seed=args.random_seed,
            optimization_config=OptimizationConfig(
                budget=args.budget,
                seeding_strategy=args.seed_strategy,
                encoder_config=MoleculeEncoderConfig(
                    encoder_type=args.encoder,
                    encoder_kwargs=json.loads(args.encoder_kwargs)
                ),
                model_config=ModelConfig(
                    model_type=args.model,
                    model_architecture=model_architecture,
                    train_config=train_config,
                    static_embeddings=args.static_embeddings,
                    inference_batch_size=args.inference_batch_size,
                ),
                acquisition_config=AcquisitionConfig(
                    type=args.acquisition,
                    acquisition_function=args.acquisition_function,
                    acquisition_function_kwargs={"exploration": args.exploration},
                    batch_size=args.batch_size
                )
            ),
            bayleys_config=BayleysConfig(
                log_level=args.log_level,
                log_file=args.log_file,
                cache_dir=args.cache_dir,
                device=args.device
            )
        )

        return benchmark_config
