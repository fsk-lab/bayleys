from abc import ABC, abstractmethod
from typing import Optional, Literal
from logging import getLogger
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import PreTrainedModel
from peft import PeftModel

from .base import MoleculeEncoder
from .pretraining import PreTrainingConfig

from ..config import get_config
from ..molecule_library import MoleculeLibrary, MemoryMappedDataset
from ..utils import TensorLike
from ..utils.tokenization import tokenize_full_library


class LLMMoleculeEncoder(MoleculeEncoder, torch.nn.Module, ABC):
    """
    Abstract base class for molecule encoders that use large language models (LLMs).
    """

    learnable = True
    max_length: int = 512

    lora_levels = {
        0: [],
        1: [],
        2: [],
        3: "all-linear"
    }

    def __init__(
            self,
            lora_adaptor: Optional[str] = None,
            use_projections: bool = False,
            batch_size: int = 256,
            **kwargs
    ):
        """
        Initializes the LLMMoleculeEncoder.

        Args:
            lora_adaptor: Optional LoRA adaptor to load for the base model.
            use_projections: Whether to use the projection head for encoding. If True, the `encode` method will return
                             the latent space of the projection head instead of the encoder output.
            **kwargs: Additional keyword arguments.
        """
        torch.nn.Module.__init__(self)
        MoleculeEncoder.__init__(self)

        if use_projections and lora_adaptor is None:
            raise ValueError(f"Cannot use projections without a LoRA adaptor. Please specify a LoRA adaptor to load or "
                             f"set `use_projections` to False.")

        self.lora_name = lora_adaptor if lora_adaptor is not None else "None"
        self.use_projections = use_projections
        self.batch_size = batch_size

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.logger = getLogger("bayleys")

    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Returns the internal, unique name of the underlying LLM model. Used for logging and model identification (when
        storing embeddings or LoRA adaptors).

        Returns:
            str: Name of the LLM model.
        """
        raise NotImplementedError

    @property
    def name(self) -> str:
        """
        Returns the name of the encoder, including the model and LoRA adaptor if applicable.

        Returns:
            str: Name of the encoder.
        """
        if self.lora_name == "None":
            return self.model_name
        else:
            proj_tag = "Proj" if self.use_projections else "Enc"
            return f"{self.model_name}_LoRA-[{self.lora_name}-{proj_tag}]"

    def tokenize_batch(self, smiles_batch: list[str] | Dataset, **tokenization_kwargs) -> dict:
        """
        Tokenizes a Huggingface dataset or list of SMILES strings for input into the LLM model.

        Args:
            smiles_batch (list[str] | Dataset): List of SMILES strings to tokenize.
            **tokenization_kwargs: Additional keyword arguments for the tokenizer.

        Returns:
            dict: Tokenized representation of the SMILES batch.
        """
        if not isinstance(smiles_batch, list):
            if "labels" in smiles_batch.data:
                labels = smiles_batch["labels"]
            else:
                labels = None
            smiles_batch = smiles_batch["smiles"]
        else:
            labels = None

        tok = self._tokenize_batch(smiles_batch, **tokenization_kwargs)

        if labels is not None:
            tok["labels"] = labels

        return tok

    @abstractmethod
    def _tokenize_batch(self, smiles_batch: list[str], **tokenization_kwargs) -> dict:
        """
        Internal, tokenizer-specific method to tokenize a batch of SMILES strings.

        Args:
            smiles_batch (list[str]): List of SMILES strings to tokenize.
            **tokenization_kwargs: Additional keyword arguments for the tokenizer.

        Returns:
            dict: Tokenized representation of the SMILES batch.
        """
        raise NotImplementedError

    @torch.no_grad()
    def encode(self, smiles: np.ndarray) -> TensorLike:
        """
        Encodes a list of SMILES strings into embeddings using the encoder-only LLM.

        Args:
            smiles (np.ndarray): An array of SMILES strings to encode.

        Returns:
            TensorLike: A tensor containing the embeddings for the input SMILES strings.
        """
        self.logger.debug(f"{self.__class__.__name__}: Starting to encode {len(smiles)} molecules...")
        start_time = time.time()

        device = get_config().device
        self.to(device)

        embeddings = []

        for i in range(0, len(smiles), self.batch_size):
            batch_smiles = smiles[i:i + self.batch_size]
            inputs = self.tokenize_batch(batch_smiles.tolist(), padding=True, return_tensors="pt")
            input_ids, attention_mask = inputs["input_ids"].to(device), inputs["attention_mask"].to(device)
            if self.use_projections is True:
                batch_embeddings = self.model.get_latent_projections(input_ids, attention_mask)["latent_projections"]
            else:
                batch_embeddings = self.model.get_embeddings(input_ids, attention_mask)["embeddings"]
            embeddings.append(batch_embeddings.detach().cpu())

        embeddings = torch.cat(embeddings, dim=0)

        self.logger.debug(f"{self.__class__.__name__}: Encoding completed in {time.time() - start_time:.1f} seconds.")

        return embeddings

    def fine_tune(self, config: PreTrainingConfig, library: MoleculeLibrary) -> None:
        """
        Fine-tunes the LLM encoder using the provided configuration and molecule library. Performs the general steps of
        preparing a Huggingface dataset, splitting into training and validation sets, and invoking the appropriate
        fine-tuning method based on the specified strategy.

        Args:
            config (PreTrainingConfig): Configuration for fine-tuning.
            library (MoleculeLibrary): Molecule library to use for fine-tuning.
        """
        if isinstance(self.model, PeftModel) or getattr(self, "peft_model", None) is not None:
            raise ValueError("Fine-tuning is not supported when a LoRA adaptor is already applied to the model.")

        tag = (f"{self.model_name}-{library.name}-{config.projection_loss}-{library.embedding_name}-"
               f"{config.projection_loss_weight}-{config.unique_id}")

        config.dataset_size = len(library)

        train_library, val_library = library.train_test_split(test_size=0.1)

        train_dataset_tokenized, train_max_length = tokenize_full_library(
            library=train_library,
            tokenizer=self.tokenizer,
            model_max_length=self.max_length,
            tmp_dir=Path(get_config().tmp_dir) / library.name / "train_tokenized",
            contrastive_learning=config.finetuning_is_contrastive,
            num_examples_per_molecule=config.num_train_epochs,
            include_embeddings=config.finetuning_requires_auxiliary_labels
        )
        self.logger.info(f"Tokenized the training dataset for {tag}.")
        val_dataset_tokenized, val_max_length = tokenize_full_library(
            library=val_library,
            tokenizer=self.tokenizer,
            model_max_length=self.max_length,
            tmp_dir=Path(get_config().tmp_dir) / library.name / "val_tokenized",
            contrastive_learning=config.finetuning_is_contrastive,
            num_examples_per_molecule=config.num_train_epochs,
            include_embeddings=config.finetuning_requires_auxiliary_labels
        )
        self.logger.info(f"Tokenized the validation dataset for {tag}.")
        if config.finetuning_requires_auxiliary_labels:
            train_dataset_tokenized.rename_column("embeddings", "aux_labels")
            val_dataset_tokenized.rename_column("embeddings", "aux_labels")
            config.projection_target_dim = library.embeddings.shape[1]

        self.max_length = max(train_max_length, val_max_length)

        self.logger.info(f"{self.__class__.__name__}: Starting fine-tuning of the LLM encoder on the virtual library "
                         f"'{library.name}' with the following configuration: {repr(config)}.")
        start_time = time.time()
        self._fine_tune(config, train_dataset_tokenized, val_dataset_tokenized, tag)
        self.logger.info(f"{self.__class__.__name__}: Fine-tuned the LLM in {time.time() - start_time:.1f} "
                         f"seconds, and saved the LoRA adaptor as '{tag}'.")

    @abstractmethod
    def _fine_tune(
            self,
            config: PreTrainingConfig,
            train_data: MemoryMappedDataset,
            val_data: MemoryMappedDataset,
            tag: str
    ):
        """
        Implementation of a fine-tuning strategy for the LLM encoder. This method should set up the LoRA adapter,
        create an auxiliary model that combines the encoder with a projection head, and train the model using the
        Trainer API. The training loss should be defined in the fine-tuning configuration.

        Args:
            config (PreTrainingConfig): Configuration for fine-tuning.
            train_data (MemoryMappedDataset): Training dataset.
            val_data (MemoryMappedDataset): Validation dataset.
            tag (str): Tag to use for saving the LoRA adaptor.
        """
        raise NotImplementedError


class LLMWithProjectionHead(torch.nn.Module, ABC):
    """
    Abstract base class for transformer-based models with a projection head on top, designed for downstream tasks
    like regression or embedding refinement. This class is intended to be compatible with the `Trainer` API from the
    Hugging Face Transformers library.

    Subclasses must implement the `_get_last_hidden_state` method to extract the last hidden state from the
    specific model architecture.

    Args:
        model (PreTrainedModel | PeftModel): The pre-trained transformer model, optionally with LoRA adaptors applied.
        projection_dim (int): Size of the output projection (e.g., number of regression targets or embedding dimension).
        pooling_operation (Literal["cls", "mean"]): The pooling_operation strategy to use for obtaining fixed-size
                                                    embedding from the encoder output.
        dropout (float, optional): Dropout rate for regularization. Default is 0.1.
        auxiliary_labels (bool, optional): True if the training should include external labels.
    """
    def __init__(
            self,
            model: PreTrainedModel | PeftModel,
            pooling_operation: Literal["cls", "mean", "max"] = "mean",
            projection_depth: int = 1,
            projection_dim: int = 128,
            output_dim: Optional[int] = None,
            dropout: float = 0.1,
            auxiliary_labels: bool = False,
            **kwargs
    ):
        torch.nn.Module.__init__(self)

        self.model = model
        self.config = model.config
        self.pooling_operation = pooling_operation

        latent_dim = self._get_encoder_hidden_size()

        if output_dim is None:
            output_dim = projection_dim
        if projection_depth == 0:
            projection_dim = latent_dim
        self.output_dim = output_dim

        projection_head = []
        last_dim = latent_dim
        for _ in range(projection_depth):
            projection_head.append(torch.nn.Linear(last_dim, projection_dim))
            last_dim = projection_dim
            projection_head.append(torch.nn.ReLU())
            projection_head.append(torch.nn.Dropout(dropout))
        self.projection_head = torch.nn.Sequential(*projection_head)
        self.final_projection_layer = torch.nn.Linear(projection_dim, output_dim)

        self.external_labels = auxiliary_labels

        self.kwargs = {
            "pooling_operation": pooling_operation,
            "projection_depth": projection_depth,
            "projection_dim": projection_dim,
            "output_dim": output_dim,
            "dropout": dropout,
            "auxiliary_labels": auxiliary_labels
        }

        for key, value in kwargs.items():
            setattr(self, key, value)
            self.kwargs[key] = value

    @property
    def is_lora_model(self) -> bool:
        """
        Indicates whether the underlying model has LoRA adaptors applied.
        """
        return isinstance(self.model, PeftModel)

    @abstractmethod
    def _get_encoder_hidden_size(self) -> int:
        """
        Abstract method to determine the output dimension of the projection head. If the projection dimension is not
        specified during initialization, this method will be called to infer the hidden size of the encoder.

        Returns:
            int: The output dimension of the projection head.
        """
        raise NotImplementedError

    @abstractmethod
    def get_embeddings(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            encoder_labels: Optional[torch.Tensor] = None,
            aux_labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Abstract method to obtain the last hidden state from the encoder given input token IDs and attention mask.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask for the input.
            encoder_labels (torch.Tensor, optional): Labels for the main encoder task (e.g. MLM). Default is None.
            aux_labels (torch.Tensor, optional): Labels for the auxiliary task (e.g. regression). Default is None.

        Returns:
            dict: A dictionary containing at least the following keys:
                - "encoder_logits": The logits from the encoder.
                - "embeddings": The pooled output from the encoder.
                - "encoder_labels": The input labels for the encoder task (can be None).
                - "aux_labels": The input labels for the auxiliary task (can be None).
        """
        raise NotImplementedError

    def get_latent_projections(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            encoder_labels: Optional[torch.Tensor] = None,
            aux_labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Method to obtain the latent space of the projection head given input token IDs and attention mask. This method
        should pass the input through the encoder, apply the specified pooling operation to obtain a fixed-size
        embedding, and then pass the embedding through the projection head.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask for the input.
            encoder_labels (torch.Tensor, optional): Labels for the main encoder task (e.g. MLM). Default is None.
            aux_labels (torch.Tensor, optional): Labels for the auxiliary task (e.g. regression). Default is None.

        Returns:
            dict: A dictionary containing the following keys:
                - "encoder_logits": The logits from the encoder.
                - "embeddings": The pooled output from the encoder.
                - "encoder_labels": The input labels for the encoder task (can be None).
                - "aux_labels": The input labels for the auxiliary task (can be None).
                - "latent_projections": The output of the projection head before the final projection layer.
        """
        outputs = self.get_embeddings(input_ids, attention_mask, encoder_labels, aux_labels)
        outputs["latent_projections"] = self.projection_head(outputs["embeddings"])
        return outputs

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_labels: Optional[torch.Tensor] = None,
            aux_labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Forward pass of the model, as required by the `Trainer` API. This method should pass the input through the
        encoder, apply the specified pooling operation to obtain a fixed-size embedding, and then pass the embedding
        through the projection head.

        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask for the input.
            encoder_labels (torch.Tensor, optional): Labels for the main encoder task (e.g. MLM).
            aux_labels (torch.Tensor, optional): Labels for the auxiliary task (e.g. regression).

        Returns:
            dict: A dictionary containing the following keys:
                - "encoder_logits": The language model logits from the encoder.
                - "encoder_labels": The input labels for the encoder task.
                - "projections": The output of the projection head.
                - "aux_labels": The input labels for possible auxiliary task. Can be None.
        """
        outputs = self.get_latent_projections(input_ids, attention_mask, encoder_labels, aux_labels)
        outputs["projections"] = self.final_projection_layer(outputs["latent_projections"])
        return outputs

    def save_pretrained(self, save_directory: str | Path):
        """
        Saves the model's LoRA adaptors and projection head to the specified directory.

        Args:
            save_directory (str | Path): The directory where the model should be saved.
        """
        if not self.is_lora_model:
            raise ValueError("The model must have LoRA adaptors applied in order to be saved using this method.")

        if isinstance(save_directory, str):
            save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(save_directory)
        torch.save(self.kwargs, save_directory / "projection_head_hyperparameters.pt")
        torch.save(self.projection_head.state_dict(), save_directory / "projection_head.pt")
        torch.save(self.final_projection_layer.state_dict(), save_directory / "projection_layer.pt")

    @classmethod
    def from_pretrained(cls, base_model: PreTrainedModel, load_directory: str | Path) -> "LLMWithProjectionHead":
        """
        Loads the model's LoRA adaptors and projection head from the specified directory.

        Args:
            base_model (PreTrainedModel): The base transformer model to which the LoRA adaptors will be applied.
            load_directory (str): The directory from which the model should be loaded.

        Returns:
            LLMWithProjectionHead: An instance of the model with loaded weights.
        """
        if isinstance(load_directory, str):
            load_directory = Path(load_directory)

        model = PeftModel.from_pretrained(base_model, str(load_directory), trust_remote_code=True, is_trainable=False)
        hyperparameters = torch.load(load_directory / "projection_head_hyperparameters.pt")
        instance = cls(model=model, **hyperparameters)

        projection_head_state_dict = torch.load(load_directory / "projection_head.pt")
        final_projection_layer_state_dict = torch.load(load_directory / "projection_layer.pt")
        instance.projection_head.load_state_dict(projection_head_state_dict)
        instance.final_projection_layer.load_state_dict(final_projection_layer_state_dict)

        return instance

