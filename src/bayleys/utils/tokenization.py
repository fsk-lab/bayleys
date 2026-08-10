from pathlib import Path
import time
from logging import getLogger, ERROR
import numpy as np
from numpy.lib.format import open_memmap
import torch
from tokenizers import BertWordPieceTokenizer
getLogger("deepchem").setLevel(ERROR)
from deepchem.feat.smiles_tokenizer import BasicSmilesTokenizer

from ..molecule_library import MoleculeLibrary
from ..molecule_library import MemoryMappedDataset, MemoryMappedFixedSizeColumn, MemoryMappedVariableSizeColumn
from ..utils.smiles_utils import randomize_smiles_generator, canonicalize_smiles_batch, randomize_smiles_batch, SMILESPairSampler


def train_smiles_tokenizer(
    library: MoleculeLibrary,
    tokenizer_dir: Path,
    num_views_per_molecule: int = 1
):
    """
    Trains a BertWordPieceTokenizer on the SMILES strings in the provided molecule library, and saves the tokenizer to
    disk.

     1. Tokenizes the SMILES strings in the library using a BasicSmilesTokenizer.
     2. Trains a BertWordPieceTokenizer on the pre-tokenized SMILES.
     3. Saves the trained tokenizer to disk.

    Args:
        library: MoleculeLibrary containing the SMILES strings to train the tokenizer on.
        tokenizer_dir: Directory to save the trained tokenizer to.
        num_views_per_molecule (int): The number of different tokenized views to generate for each molecule in the
                                      library. If >1, generates multiple random views per SMILES.
    """
    logger = getLogger("bayleys")

    tokenizer_dir.mkdir(exist_ok=True, parents=True)
    tmp_file = tokenizer_dir / "smiles_tokenization_results.txt"

    # 1. Tokenize the SMILES strings in the library using a BasicSmilesTokenizer.
    base_tokenizer = BasicSmilesTokenizer()
    with open(tmp_file, "w", encoding="utf-8") as f:
        for smiles in library.smiles:
            for smi in randomize_smiles_generator(smiles, n_random=num_views_per_molecule):
                try:
                    tokens = base_tokenizer.tokenize(smiles)
                    reconstructed_smiles = "".join(tokens)
                    if reconstructed_smiles != smiles:
                        raise ValueError(f"Tokenization error: reconstructed SMILES '{reconstructed_smiles}' does not "
                                         f"match original '{smiles}'")
                    f.write(" ".join(tokens) + "\n")
                except ValueError as e:
                    logger.warning(f"Error tokenizing SMILES '{smiles}': {e}")

    # 2. Train a BertWordPieceTokenizer on the pre-tokenized SMILES.
    tokenizer = BertWordPieceTokenizer(
        lowercase=False,
        clean_text=False,
        handle_chinese_chars=False,
        strip_accents=False
    )

    tokenizer.train(
        files=[str(tokenizer_dir / "smiles_tokenization_results.txt")],
        vocab_size=2048,
        min_frequency=2,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        wordpieces_prefix="##"
    )

    # 3. Save the trained tokenizer to disk.
    tokenizer.save_model(str(tokenizer_dir))
    tmp_file.unlink()


def tokenize_full_library(
    library: MoleculeLibrary,
    tokenizer,
    tmp_dir: Path,
    model_max_length: int = 256,
    contrastive_learning: bool = False,
    num_examples_per_molecule: int = 2,
    include_smiles: bool = False,
    include_embeddings: bool = False,
    include_labels: bool = False
) -> tuple[MemoryMappedDataset, int]:
    """
    Tokenizes the SMILES strings in the provided molecule library using the provided tokenizer, and saves the tokenized
    data to disk in a memory-mapped format. Requires two forward passes through the tokenizer:
      1. First pass to determine the maximum sequence length and total number of tokens across the dataset, which are
         needed to pre-allocate the memory-mapped array for the tokenized input IDs.
      2. Second pass to actually tokenize the SMILES strings and save the tokenized input IDs to the memory-mapped
         array.

    Args:
        library: MoleculeLibrary containing the SMILES strings to tokenize.
        tokenizer: Tokenizer to use for tokenizing the SMILES strings.
        tmp_dir: Directory to save the tokenized data to.
        model_max_length: The maximum sequence length to use for tokenization (allowed by the model).
        contrastive_learning: Whether to generate additional tokenized views for contrastive learning.
        num_examples_per_molecule: The number of positive and negative examples to sample for each molecule in the
                                   library for contrastive learning. Only used if contrastive_learning is True.
        include_smiles: Whether to include the original SMILES strings in the returned dataset.
        include_embeddings: Whether to include the original embeddings in the returned dataset.
        include_labels: Whether to include the labels in the returned dataset.

    Returns:
        MemoryMappedDataset: A MemoryMappedDataset containing the tokenized input IDs, and optionally the original
                             SMILES strings, embeddings, and labels depending on the input flags.
        int: The maximum sequence length across the tokenized SMILES strings.
    """
    tmp_dir.mkdir(exist_ok=True, parents=True)
    logger = getLogger("bayleys")

    max_length = 0
    columns = []

    canonical_smiles = canonicalize_smiles_batch(library.smiles)
    smiles_views = [canonical_smiles]
    names = [""]

    if contrastive_learning:
        smiles_views += [randomize_smiles_batch(canonical_smiles), np.random.permutation(canonical_smiles)]
        names += ["_positive_fallback", "_negative_fallback"]

    for name, smiles_view in zip(names, smiles_views):

        start_time = time.time()

        total_tokens_, max_length_ = 0, 0
        offsets, lengths = np.zeros(len(library), dtype=np.int64), np.zeros(len(library), dtype=np.int32)

        for batch_start_idx in range(0, len(library), 1024):

            encoded = tokenizer(
                smiles_view[batch_start_idx: batch_start_idx + 1024].tolist(),
                add_special_tokens=True,
                truncation=True,
                padding=False,
                max_length=model_max_length,
                return_attention_mask=False,
                return_special_tokens_mask=False,
            )

            for idx, ids in enumerate(encoded["input_ids"]):
                sequence_length = len(ids)
                if sequence_length > max_length_:
                    max_length_ = sequence_length
                    if max_length_ > max_length:
                        max_length = max_length_
                offsets[batch_start_idx + idx] = total_tokens_
                lengths[batch_start_idx + idx] = sequence_length
                total_tokens_ += sequence_length

        input_ids = open_memmap(
            tmp_dir / f"input_ids{name}.npy",
            mode="w+",
            dtype=np.uint16,
            shape=(total_tokens_,)
        )

        special_tokens_mask = open_memmap(
            tmp_dir / f"special_tokens_mask{name}.npy",
            mode="w+",
            dtype=np.uint16,
            shape=(total_tokens_,)
        )

        for batch_start_idx in range(0, len(library), 1024):
            encoded = tokenizer(
                smiles_view[batch_start_idx: batch_start_idx + 1024].tolist(),
                add_special_tokens=True,
                truncation=True,
                padding=False,
                max_length=max_length,
                return_attention_mask=False,
                return_special_tokens_mask=True,
            )
            for idx, _ in enumerate(encoded["input_ids"]):
                start = offsets[batch_start_idx + idx]
                end = start + lengths[batch_start_idx + idx]
                input_ids[start:end] = np.asarray(encoded["input_ids"][idx], dtype=np.uint16)
                special_tokens_mask[start:end] = np.asarray(encoded["special_tokens_mask"][idx], dtype=np.uint16)

        input_ids.flush(), special_tokens_mask.flush()

        np.save(tmp_dir / f"offsets{name}.npy", offsets)
        np.save(tmp_dir / f"lengths{name}.npy", lengths)

        input_id_column = MemoryMappedVariableSizeColumn(
            column_name=f"input_ids" if name == "" else f"_input_ids{name}",
            dtype=torch.long,
            data_file=tmp_dir / f"input_ids{name}.npy",
            offsets_file=tmp_dir / f"offsets{name}.npy",
            lengths_file=tmp_dir / f"lengths{name}.npy",
        )
        special_tokens_mask_column = MemoryMappedVariableSizeColumn(
            column_name=f"special_tokens_mask" if name == "" else f"_special_tokens_mask{name}",
            dtype=torch.long,
            data_file=tmp_dir / f"special_tokens_mask{name}.npy",
            offsets_file=tmp_dir / f"offsets{name}.npy",
            lengths_file=tmp_dir / f"lengths{name}.npy",
        )
        columns.append(input_id_column)
        columns.append(special_tokens_mask_column)

        logger.info(f"Tokenized view '{name}' in {time.time() - start_time:.1f} seconds.")

    if contrastive_learning:
        start_time = time.time()
        sampler = SMILESPairSampler(
            library.smiles,
            cache_dir=tmp_dir / library.name,
            fp_size=2048,
            fp_radius=2,
            batch_size=50_000,
            num_permutations=64,
            band_size=4,
            max_bucket_size=1000,
            max_candidates_per_anchor_bucket=100,
            max_negative_samples_per_anchor=1024,
            similarity_threshold=0.5,
            dissimilarity_threshold=0.1,
            seed=42
        )
        logger.info(f"Initialized SMILESPairSampler in {time.time() - start_time:.1f} seconds.")

        start_time = time.time()
        pos_indices_file = sampler.sample_positive_examples(num_examples=num_examples_per_molecule)
        logger.info(f"Sampled positive examples in {time.time() - start_time:.1f} seconds.")

        start_time = time.time()
        neg_indices_file = sampler.sample_negative_examples(num_examples=num_examples_per_molecule)
        logger.info(f"Sampled negative examples in {time.time() - start_time:.1f} seconds.")

        pos_column = MemoryMappedFixedSizeColumn(
            column_name="_positive_indices",
            dtype=torch.float,
            file=pos_indices_file
        )

        neg_column = MemoryMappedFixedSizeColumn(
            column_name="_negative_indices",
            dtype=torch.float,
            file=neg_indices_file
        )

        columns.append(pos_column)
        columns.append(neg_column)

    dataset = MemoryMappedDataset(*columns)

    if include_smiles:
        smiles_column = MemoryMappedFixedSizeColumn.from_array(
            column_name="smiles",
            dtype=str,
            array=library.smiles,
            data_dir=tmp_dir
        )
        dataset.add_column(smiles_column)

    if include_embeddings and library.embeddings is not None:
        embeddings_column = MemoryMappedFixedSizeColumn.from_array(
            column_name="embeddings",
            dtype=torch.float,
            array=library.embeddings,
            data_dir=tmp_dir
        )
        dataset.add_column(embeddings_column)

    if include_labels and library.labels is not None:
        labels_column = MemoryMappedFixedSizeColumn.from_array(
            column_name="labels",
            dtype=torch.float,
            array=library.labels,
            data_dir=tmp_dir
        )
        dataset.add_column(labels_column)

    if contrastive_learning:
        dataset.loss_type = "contrastive"

    return dataset, max_length
