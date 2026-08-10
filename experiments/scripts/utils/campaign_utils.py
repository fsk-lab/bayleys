from typing import Optional
from pathlib import Path

from bayleys.molecule_library import MoleculeLibrary
from bayleys.encoders import MoleculeEncoderConfig
from bayleys.surrogate import ModelConfig
from bayleys.acquisition import AcquisitionConfig
from bayleys.campaign import OptimizationConfig, BenchmarkConfig, BenchmarkCampaign


def run_campaign(
        tag: str,
        library: MoleculeLibrary,
        encoder_config: MoleculeEncoderConfig,
        model_config: ModelConfig,
        acquisition_config: AcquisitionConfig,
        num_repeats: int,
        budget: int,
        data_dir: Path
):
    """
    Runs a full optimization campaign on the given library with the specified configurations.

    Args:
        tag (str): A tag to identify the campaign (used for logging and saving results).
        library (MoleculeLibrary): The molecule library to optimize over.
        encoder_config (MoleculeEncoderConfig): Configuration for the molecule encoder.
        model_config (ModelConfig): Configuration for the surrogate model.
        acquisition_config (AcquisitionConfig): Configuration for the acquisition function.
        num_repeats (int): Number of times to repeat the optimization campaign for statistical significance.
        budget (int): The optimization budget (number of iterations).
        data_dir (Path): The directory where results should be saved.
    """
    benchmark_config = BenchmarkConfig(
        num_repeats=num_repeats,
        random_seed=42,
        optimization_config=OptimizationConfig(
            budget=budget,
            seeding_strategy="random",
            encoder_config=encoder_config,
            model_config=model_config,
            acquisition_config=acquisition_config
        )
    )

    # Run the Benchmark Campaign
    campaign = BenchmarkCampaign(library=library, config=benchmark_config)

    campaign.library.get_quantile(0.99)
    campaign.library.get_top_k_threshold(10)
    campaign.library.get_top_k_threshold(100)
    campaign.library.get_top_k_threshold(1000)
    campaign.library.get_top_k_threshold(10000)

    campaign.run()

    campaign.to_json(data_dir / f"{tag}.json")


def get_tag(
        library_name: str,
        encoder_name: str,
        finetuning: Optional[str],
        finetuning_label: Optional[str],
        use_projections: bool,
        model_name: str,
        acquisition_name: str,
        budget: int,
        num_repeats: int
) -> str:
    """
    Generates a unique tag for the campaign based on the provided parameters. This tag can be used for logging and
    saving results.
    """
    tag = f"{library_name}_{encoder_name}"
    if finetuning is not None:
        tag += f"-Finetuning-{finetuning}"
        if finetuning_label is not None:
            tag += f"-{finetuning_label}"
    tag += f"-{'Proj' if use_projections is True else 'Enc'}_"
    tag += f"{model_name}_{acquisition_name}"
    tag += f"_{budget}_{num_repeats}"
    return tag