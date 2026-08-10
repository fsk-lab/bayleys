import argparse
import os
from pathlib import Path
import json
from bayleys.config import BayleysConfig
from bayleys.molecule_library import MoleculeLibraryConfig, MoleculeLibrary
from bayleys.encoders import MoleculeEncoderConfig, MoleculeEncoder, LLMMoleculeEncoder
from bayleys.utils.cache import cache_embeddings

os.environ["TOKENIZERS_PARALLELISM"] = "false"


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Precompute static embeddings for a molecule library.")
    parser.add_argument("--library", type=str, required=True)
    parser.add_argument("--encoder", type=str, required=True)
    parser.add_argument("--use_gpu", action="store_true", help="Whether to use GPU for training if available.")
    args = parser.parse_args()

    # Load the configurations of the working directory
    with open(Path(__file__).parent / "working_directories.json", "r") as f:
        working_directories = json.load(f)

    data_dir = Path(working_directories["data"])
    encoder_config_dir = Path(working_directories["encoder_configs"])

    # Set the BAYLEYS configuration
    bayleys_config = BayleysConfig(
        log_level="INFO",
        log_file=None,
        cache_dir=data_dir / "cache",
        device="gpu" if args.use_gpu else "cpu"
    )
    bayleys_config.apply()

    library_config = MoleculeLibraryConfig.from_json(data_dir / "libraries" / f"{args.library}.json")
    library = MoleculeLibrary.from_config(library_config, local_dir=data_dir)

    encoder_config_file = encoder_config_dir / f"{args.encoder}.json"

    # Check if the configuration of the base encoder exists, then load it
    if not encoder_config_file.exists():
        raise ValueError(f"Encoder configuration file {encoder_config_file} not found.")

    encoder_config = MoleculeEncoderConfig.from_json(encoder_config_file)
    encoder: MoleculeEncoder = encoder_config.get_encoder()

    if isinstance(encoder, LLMMoleculeEncoder):
        encoder.max_length = library.max_smiles_length

    embeddings = encoder.encode(library.smiles)
    cache_embeddings(embeddings, encoder.name, library.name)
