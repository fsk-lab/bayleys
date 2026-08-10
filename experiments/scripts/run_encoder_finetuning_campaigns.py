import argparse
import logging
import os
import time
import json
from pathlib import Path
from bayleys.config import BayleysConfig
from bayleys.molecule_library import MoleculeLibraryConfig, MoleculeLibrary
from bayleys.encoders import MoleculeEncoderConfig, PreTrainingConfig, LLMMoleculeEncoder, MorganFingerprintEncoder
from bayleys.surrogate import ModelConfig
from bayleys.acquisition import AcquisitionConfig
from bayleys.utils.cache import cache_embeddings, load_cached_embeddings

from utils.campaign_utils import run_campaign, get_tag


os.environ["TOKENIZERS_PARALLELISM"] = "false"
NUM_REPEATS = 20


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Fine-tune an encoder and perform full optimization campaigns on the library.")
    parser.add_argument("--library", type=str, required=True)
    parser.add_argument("--encoder", type=str, required=True)
    parser.add_argument("--finetuning", type=str, required=True)
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
    tmp_dir = Path(working_directories["tmp"])

    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id is not None:
        tmp_dir = tmp_dir / slurm_job_id
        tmp_dir.mkdir(parents=True, exist_ok=True)
    else:
        tmp_dir = None

    # Set the BAYLEYS configuration
    bayleys_config = BayleysConfig(
        log_level="DEBUG",
        log_file=None,
        cache_dir=data_dir / "cache",
        tmp_dir=tmp_dir,
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
    encoder_config.encoder_kwargs["use_projections"] = False

    encoder = encoder_config.get_encoder()
    if not isinstance(encoder, LLMMoleculeEncoder):
        raise ValueError(f"Only LLM encoders can be fine-tuned.")
    logger.info(f"Loaded encoder {encoder.name} for fine-tuning.")

    # Load the fine-tuning configuration
    finetuning_config_file = training_config_dir / f"{args.finetuning}.json"
    if not finetuning_config_file.exists():
        print(f"Fine-tuning configuration file {finetuning_config_file} not found.")
    finetuning_config = PreTrainingConfig.from_json(finetuning_config_file)
    if "Laser" in args.library:
        finetuning_config.train_batch_size = finetuning_config.train_batch_size // 2
    if "Kraken" in args.library:
        finetuning_config.train_batch_size = finetuning_config.train_batch_size // 4

    finetuning_label = None
    if finetuning_config.projection_loss == "mse":
        finetuning_label = "MorganFingerprint-2-1024"
        aux_encoder = MorganFingerprintEncoder(radius=2, n_bits=1024, batch_size=2048)
        try:
            library.embeddings = load_cached_embeddings(aux_encoder.name, library.name)
            logger.info(f"Loaded cached embeddings for the auxiliary encoder {aux_encoder.name} and library "
                        f"{library.name}.")
        except FileNotFoundError:
            logger.debug("Starting to generate embeddings for the auxiliary encoder (Morgan Fingerprint).")
            start_time = time.time()
            library.embeddings = aux_encoder.encode(library.smiles)
            cache_embeddings(library.embeddings, aux_encoder.name, library.name)
            logger.info(f"Generated embeddings for auxiliary encoder {aux_encoder.name} and library {library.name} in "
                        f"{time.time() - start_time:.1f}.")
        library.embedding_name = aux_encoder.name

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

    # Fine tune the encoder with the specified fine-tuning configuration
    encoder.fine_tune(finetuning_config, library)
    encoder_config.encoder_kwargs["lora_adaptor"] = encoder.lora_name

    # Calculate and cache the new embeddings for the entire library after fine-tuning
    encoder.use_projections = False
    embeddings = encoder.encode(library.smiles)
    cache_embeddings(embeddings, encoder.name, library.name)

    # Run a benchmark campaign in a virtual screening scenario (20,000 molecules budget, batch size 1,000)

    budget = 20_000

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
