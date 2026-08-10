import argparse
import logging
import os
import json
from pathlib import Path
from bayleys.config import BayleysConfig
from bayleys.molecule_library import MoleculeLibraryConfig, MoleculeLibrary
from bayleys.encoders import MoleculeEncoderConfig, PreTrainingConfig
from bayleys.surrogate import ModelConfig
from bayleys.acquisition import AcquisitionConfig

from utils.campaign_utils import run_campaign, get_tag


os.environ["TOKENIZERS_PARALLELISM"] = "false"
NUM_REPEATS = 20


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Perform full optimization campaigns on the library.")
    parser.add_argument("--library", type=str, required=True)
    parser.add_argument("--encoder", type=str, required=True)
    parser.add_argument("--finetuning", type=str, default=None)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--acquisition", type=str, required=True)
    parser.add_argument("--use_gpu", action="store_true", help="Whether to use GPU for training if available.")
    args = parser.parse_args()

    # Load the configurations of the working directory
    with open(Path(__file__).parent / "working_directories.json", "r") as f:
        working_directories = json.load(f)

    data_dir = Path(working_directories["data"])
    encoder_config_dir = Path(working_directories["encoder_configs"])
    training_config_dir = Path(working_directories["training_configs"])
    model_config_dir = Path(working_directories["model_configs"])
    acquisition_config_dir = Path(working_directories["acquisition_configs"])
    results_dir = Path(working_directories["results"])

    # Set the BAYLEYS configuration
    bayleys_config = BayleysConfig(
        log_level="DEBUG",
        log_file=None,
        cache_dir=data_dir / "cache",
        device="gpu" if args.use_gpu else "cpu"
    )
    bayleys_config.apply()
    logger = logging.getLogger("bayleys")

    # Load the Library
    library_config = MoleculeLibraryConfig.from_json(data_dir / "libraries" / f"{args.library}.json")
    library = MoleculeLibrary.from_config(library_config, local_dir=data_dir)
    logger.info(f"Loaded library {library.name} with {len(library)} molecules.")

    # Load the Encoder Config
    encoder_config_file = encoder_config_dir / f"{args.encoder}.json"
    if not encoder_config_file.exists():
        raise ValueError(f"Encoder configuration file {encoder_config_file} not found.")
    encoder_config = MoleculeEncoderConfig.from_json(encoder_config_file)

    # Check if the encoder is a trainable LLM encoder
    if encoder_config.encoder_type == "TrainableBERTEncoder":
        model_dir = data_dir / "cache" / "models" / f"{args.encoder}-{library.name}"
        if not model_dir.exists():
            raise ValueError(f"Pre-trained model directory {model_dir} not found for the encoder {args.encoder} and "
                             f"library {library.name}. Please make sure to pre-train the encoder before running the "
                             f"campaign.")
        encoder_config.encoder_kwargs["model_id"] = str(model_dir)

        finetuning_label = None

    # Load the fine-tuning configuration for self-supervised fine-tuning
    elif args.finetuning is not None:
        finetuning_config_file = training_config_dir / f"{args.finetuning}.json"
        if not finetuning_config_file.exists():
            print(f"Fine-tuning configuration file {finetuning_config_file} not found.")
        finetuning_config = PreTrainingConfig.from_json(finetuning_config_file)

        finetuning_label = None
        if finetuning_config.projection_loss == "mse":
            finetuning_label = "MorganFingerprint-2-1024"

        lora_adaptor_name = (f"{args.encoder}-{args.library}-{finetuning_config.projection_loss}-"
                             f"{finetuning_label}-{finetuning_config.projection_loss_weight}-"
                             f"{finetuning_config.unique_id}")
        encoder_config.encoder_kwargs["lora_adaptor"] = lora_adaptor_name
    else:
        finetuning_label = None

    encoder_config.encoder_kwargs["use_projections"] = False

    # Load the Model Config
    model_config_file = model_config_dir / f"{args.model}.json"
    if not model_config_file.exists():
        raise ValueError(f"Model configuration file {model_config_file} not found.")
    model_config = ModelConfig.from_json(model_config_file)

    # Load the Acquisition Config
    acquisition_config_file = acquisition_config_dir / f"{args.acquisition}.json"
    if not acquisition_config_file.exists():
        raise ValueError(f"Acquisition configuration file {acquisition_config_file} not found.")
    acquisition_config = AcquisitionConfig.from_json(acquisition_config_file)

    # Run a benchmark campaign in a virtual screening scenario (20,000 molecules budget, batch size 1,000)

    budget = 20_000

    tag = get_tag(
        library_name=library.name,
        encoder_name=args.encoder,
        finetuning=None,
        finetuning_label=None,
        use_projections=False,
        model_name=args.model,
        acquisition_name=args.acquisition,
        budget=budget,
        num_repeats=NUM_REPEATS
    )

    acquisition_config.batch_size = 1000

    run_campaign(
        tag=tag,
        library=library,
        encoder_config=encoder_config,
        model_config=model_config,
        acquisition_config=acquisition_config,
        budget=budget,
        num_repeats=NUM_REPEATS,
        data_dir=results_dir
    )

    # Run a benchmark campaign in an experimental scenario (1,000 molecules budget, batch size 50)

    budget = 1000

    tag = get_tag(
        library_name=library.name,
        encoder_name=args.encoder,
        finetuning=args.finetuning,
        finetuning_label=finetuning_label,
        use_projections=False,
        model_name=args.model,
        acquisition_name=args.acquisition,
        budget=budget,
        num_repeats=NUM_REPEATS
    )

    acquisition_config.batch_size = 50

    run_campaign(
        tag=tag,
        library=library,
        encoder_config=encoder_config,
        model_config=model_config,
        acquisition_config=acquisition_config,
        budget=budget,
        num_repeats=NUM_REPEATS,
        data_dir=results_dir
    )

