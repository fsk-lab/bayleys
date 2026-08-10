from typing import Optional, Literal
from abc import ABC
from pathlib import Path
from dataclasses import dataclass
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM
from peft import get_peft_model

from .llm import LLMMoleculeEncoder, LLMWithProjectionHead
from .pretraining import PreTrainingConfig, PairwiseSMILESCollator, MixedObjectiveTrainer

from ..config import get_config
from ..molecule_library import MemoryMappedDataset
from ..utils import pool_transformer_outputs, masked_language_modeling_loss, LoggerCallback


@dataclass
class PairwiseSMILESCollatorForMaskedLM(PairwiseSMILESCollator):
    """
    Specific implementation of a `PairwiseSMILESCollator` for masked language modeling. This collator prepares batches
    of SMILES pairs for training encoder-only LLMs (e.g. BERT / RoBERTa) on a masked language modeling objective. It
    applies random masking to the input token IDs according to the specified `mlm_probability` and prepares the
    corresponding labels for the MLM task.
    """

    mlm_probability: float = 0.15
    vocab_size: Optional[int] = None

    def create_lm_features(self, tokenized: dict[str, list]) -> dict[str, torch.Tensor]:
        """
        Prepares the input IDs for masked language modeling by randomly masking tokens with the probability specified
        in the `mlm_probability` attribute. Out of these masked tokens...
            - 80% are replaced with the tokenizer's mask token ID (e.g. [MASK]),
            - 10% are replaced with random token IDs from the tokenizer's vocabulary,
            - 10% are left unchanged.

        Args:
            tokenized (dict[str, list]): A dictionary containing the tokenized input IDs and attention masks for the
                                         batch. Requires the keys "input_ids" and "special_tokens_mask".

        Returns:
            torch.Tensor: The masked input IDs, where some tokens have been replaced with the mask token ID or random
                          token IDs.
            torch.Tensor: The labels for the masked language modeling task, where only the masked tokens have their
                          original token IDs and the rest are set to `ignore_token_id` (e.g., -

        """
        tokenized = self.pad_batch(tokenized)
        input_ids, special_tokens_mask = tokenized["input_ids"], tokenized["special_tokens_mask"]

        probability_matrix = torch.full(input_ids.shape, self.mlm_probability)
        probability_matrix.masked_fill_(special_tokens_mask.bool(), value=0.0)

        if self.tokenizer.pad_token_id is not None:
            padding_mask = input_ids.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        _labels = input_ids.clone()
        masked_indices = torch.bernoulli(probability_matrix).bool()
        _labels[~masked_indices] = self.ignore_token_id  # Only compute loss on masked tokens

        # 80% of the time, replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, replace masked input tokens with random token
        vocab_size = self.vocab_size if self.vocab_size is not None else len(self.tokenizer)
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_token_ids = torch.randint(vocab_size, input_ids.shape, dtype=torch.long)
        input_ids[indices_random] = random_token_ids[indices_random]

        if self.padding == "max_length":
            labels = torch.full((input_ids.shape[0], self.max_length), self.ignore_token_id, dtype=torch.long)
            labels[:, :input_ids.shape[1]] = _labels
        else:
            labels = _labels

        return {"input_ids": input_ids, "encoder_labels": labels, "attention_mask": tokenized["attention_mask"]}


class EncoderOnlyLLMWithProjectionHead(LLMWithProjectionHead):
    """
    Implementation of a `LLMWithProjectionHead` that uses an encoder-only, RoBERTa-type LLM to generate molecular
    embeddings.
    """
    def _get_encoder_hidden_size(self) -> int:
        """
        Returns the hidden size of the encoder to initialize the projection head for self-supervised fine tuning.
        """
        return self.model.config.hidden_size

    def get_embeddings(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            encoder_labels: Optional[torch.Tensor] = None,
            aux_labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Computes the encoder output (i.e. the language modeling head output) for the given input token IDs and
        attention mask.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask for the input.
            encoder_labels (torch.Tensor, optional): Labels for the main encoder task (e.g. MLM).
            aux_labels (torch.Tensor, optional): Labels for the auxiliary task (e.g. regression).

        Returns:
            dict: A dictionary containing at least the following keys:
                - "encoder_logits": The logits from the encoder.
                - "embeddings": The pooled output from the encoder.
                - "encoder_labels": The input labels for the encoder task (can be None).
                - "aux_labels": The input labels for the auxiliary task (can be None).
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        hidden_state = outputs.hidden_states[-1]
        pooled_output = pool_transformer_outputs(self.pooling_operation, hidden_state, attention_mask)

        return {
            "encoder_logits": outputs["logits"],
            "embeddings": pooled_output,
            "encoder_labels": encoder_labels,
            "aux_labels": aux_labels,
        }


class EncoderOnlyLLMEncoder(LLMMoleculeEncoder, ABC):
    """
    Molecule encoder that uses an encoder-only LLM (e.g. BERT / RoBERTa) to gerenate molecular embeddings. Uses the
    huggingface transformers library.
    """

    model_id: str  # Must be specified in subclasses
    init_kwargs = {}  # Must be specified in subclasses

    def __init__(
            self,
            lora_adaptor: Optional[str] = None,
            use_projections: bool = False,
            pooling: Literal["cls", "mean", "max"] = "mean",
            batch_size: int = 256
    ):
        """
        Initializes the EncoderOnlyLLMEncoder with the specified pooling_operation method.

        Args:
            lora_adaptor: Optional LoRA adaptor to load for the base model.
            use_projections: Whether to use the projection head for encoding. If True, the `encode` method will return
                             the latent space of the projection head instead of the encoder output.
            pooling: Pooling method to aggregate token embeddings into a single fixed-size vector.
                     Options include cls (default) and mean.
            batch_size: Batch size for encoding SMILES strings.
        """
        LLMMoleculeEncoder.__init__(
            self,
            use_projections=use_projections,
            lora_adaptor=lora_adaptor,
            pooling=pooling,
            batch_size=batch_size
        )

        if self.model_id is None:
            self.logger.warning(f"{self.__class__.__name__} initialized without a model_id. The tokenizer and model"
                                f"will not be loaded, and the encoder will not be functional until a model_id is set."
                                f"This is expected behavior if you intend to train a fresh model from scratch.")

            self.tokenizer, self.model = None, None
            return

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        model = AutoModelForMaskedLM.from_pretrained(self.model_id, trust_remote_code=True, **self.init_kwargs)

        if lora_adaptor is not None:
            load_dir = Path(get_config().cache_dir) / "models" / lora_adaptor
            self.model = EncoderOnlyLLMWithProjectionHead.from_pretrained(base_model=model, load_directory=load_dir)
        else:
            self.model = EncoderOnlyLLMWithProjectionHead(
                model=model,
                pooling_operation=pooling,
                projection_depth=0
            )

        self.embedding_dim = self.model.config.hidden_size
        self.model.eval()

    def _tokenize_batch(self, smiles_batch: list[str], **tokenization_kwargs) -> dict:
        """
        Tokenizes a batch of SMILES strings for input into the LLM.
        """
        return self.tokenizer(smiles_batch, truncation=True, max_length=self.max_length, **tokenization_kwargs)

    def _fine_tune(
            self,
            config: PreTrainingConfig,
            train_data: MemoryMappedDataset,
            val_data: MemoryMappedDataset,
            tag: str
    ):
        """
        Internal method to fine-tune the model as specified by the given fine-tuning configuration. This method sets up
        the LoRA adapter, creates an auxiliary model that combines the encoder with a projection head, and trains the
        model using the Trainer API. The training loss is defined in the fine-tuning configuration.

        Args:
            config (PreTrainingConfig): Fine-tuning configuration.
            train_data (MemoryMappedDataset): Training dataset.
            val_data (MemoryMappedDataset): Validation dataset.
            tag (str): Tag to identify the fine-tuned model.
        """
        if self.model.is_lora_model:
            raise ValueError(f"Model is already a LoRA model. Cannot fine-tune further with LoRA.")

        config.lora_target_modules = self.lora_levels[config.lora_target_module_level]
        lora_cfg = config.get_lora_config()
        peft_model = get_peft_model(self.model.model, lora_cfg)
        num_trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)

        aux_model = EncoderOnlyLLMWithProjectionHead(
            model=peft_model,
            pooling_operation=self.pooling,
            projection_depth=config.projection_depth,
            projection_dim=config.projection_dim,
            output_dim=config.projection_target_dim,
            auxiliary_labels=config.finetuning_requires_auxiliary_labels,
        )

        collator = PairwiseSMILESCollatorForMaskedLM(
            tokenizer=self.tokenizer,
            vocab_size=self.model.model.get_input_embeddings().num_embeddings,
            max_length=self.max_length,
            padding="longest",
            mlm_probability=config.mlm_probability
        )

        train_args = config.get_training_args(tag=tag)

        # Train the model using the Trainer API
        trainer = MixedObjectiveTrainer(
            model=aux_model,
            args=train_args,
            train_dataset=train_data,
            eval_dataset=val_data,
            data_collator=collator,
            encoder_loss=masked_language_modeling_loss,
            encoder_loss_kwargs={"ignore_index": -100},
            projection_loss=config.finetuning_projection_loss_function,
            projection_loss_kwargs=config.finetuning_projection_loss_kwargs,
            projection_loss_weight=config.projection_loss_weight,
            callbacks=[LoggerCallback(self.logger)]
        )

        self.logger.debug(f"{self.__class__.__name__}: Starting fine-tuning with {num_trainable_params} trainable "
                          f"parameters...")
        trainer.train()

        aux_model.save_pretrained(Path(get_config().cache_dir) / "models" / tag)
        self.model = aux_model
        self.lora_name = tag
        self.eval(), self.model.eval()


class ChemBERTaZincEncoder(EncoderOnlyLLMEncoder):
    """
    Molecule encoder that uses the ChemBERTa-zinc-base-v1 model to generate molecular embeddings.
    """
    model_id = "seyonec/ChemBERTa-zinc-base-v1"

    lora_levels = {
        0: ["query", "value"],
        1: ["query", "key", "value", "dense"],
        2: ["query", "key", "value", "dense", "intermediate.dense", "output.dense"],
        3: "all-linear"
    }

    @property
    def model_name(self) -> str:
        return f"ChemBERTa-Zinc-{self.pooling}"


class ChemBERTa100MEncoder(EncoderOnlyLLMEncoder):
    """
    Molecule encoder that uses the ChemBERTa-100M-MLM model to generate molecular embeddings.
    """
    model_id = "DeepChem/ChemBERTa-100M-MLM"

    lora_levels = {
        0: ["query", "value"],
        1: ["query", "key", "value", "dense"],
        2: ["query", "key", "value", "dense", "intermediate.dense", "output.dense"],
        3: "all-linear"
    }

    @property
    def model_name(self) -> str:
        return f"ChemBERTa-100M-{self.pooling}"


class MolFormerXLEncoder(EncoderOnlyLLMEncoder):
    """
    Molecule encoder that uses the MoLFormer-XL-both-10pct model to generate molecular embeddings.
    """
    model_id = "ibm/MoLFormer-XL-both-10pct"
    init_kwargs = {"deterministic_eval": True}

    lora_levels = {
        0: ["query", "value"],
        1: ["query", "key", "value", "dense"],
        2: ["query", "key", "value", "dense", "intermediate.dense", "output.dense"],
        3: "all-linear"
    }

    @property
    def model_name(self) -> str:
        return f"MolFormer-XL-{self.pooling}"
