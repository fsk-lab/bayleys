from pathlib import Path
from typing import Literal, Optional
from dataclasses import dataclass
import numpy as np
import torch
from datasets import Dataset
from transformers import T5Tokenizer, T5Config, AutoModelForSeq2SeqLM
from peft import get_peft_model

from .llm import LLMMoleculeEncoder, LLMWithProjectionHead
from .pretraining import PreTrainingConfig, MixedObjectiveTrainer, PairwiseSMILESCollator

from ..config import get_config
from ..molecule_library import MemoryMappedDataset
from ..utils import pool_transformer_outputs, masked_language_modeling_loss, LoggerCallback


@dataclass
class PairwiseSMILESCollatorForSpanCorruption(PairwiseSMILESCollator):
    """
    Specific implementation of the `PairwiseSMILESCollator` for the T5 chemistry model that applies span corruption
    to the input SMILES strings and prepares the input and target sequences for the T5 model.
    """
    noise_density: float = 0.15
    mean_span_length: float = 3.0

    def create_lm_features(self, tokenized: dict[str, list]) -> dict[str, torch.Tensor]:
        """
        Creates the input features for the masked language modeling task by applying span corruption to the input
        token IDs and preparing the corresponding target labels for the T5 model.

        Args:
            tokenized (dict[str, list]): A dictionary containing the tokenized input sequences with keys "input_ids"
                                         and "attention_mask".

        Returns:
            dict[str, torch.Tensor]: A dictionary containing the following keys:
                - "input_ids": The input token IDs for the encoder after applying span corruption.
                - "attention_mask": The attention mask for the encoder input.
                - "labels": The target token IDs for the decoder (i.e. the original input sequence with masked tokens).
        """
        all_input_ids = tokenized["input_ids"]

        corrupted_inputs, decoder_labels = [], []
        for input_ids in all_input_ids:
            corrupted, labels = self._corrupt_single_sequence(input_ids)
            corrupted_inputs.append({"input_ids": corrupted[:self.max_length]})
            decoder_labels.append({"input_ids": labels[:self.max_length]})

        batch_inputs = self.tokenizer.pad(
            corrupted_inputs,
            padding="longest",
            max_length=self.max_length,
            return_tensors="pt"
        )

        batch_labels = self.tokenizer.pad(
            decoder_labels,
            padding="longest",
            max_length=self.max_length,
            return_tensors="pt"
        )["input_ids"]

        batch_labels[batch_labels == self.tokenizer.pad_token_id] = self.ignore_token_id

        return {
            "input_ids": batch_inputs["input_ids"],
            "attention_mask": batch_inputs["attention_mask"],
            "encoder_labels": batch_labels
        }

    def _corrupt_single_sequence(self, input_ids: list[int]) -> tuple[list[int], list[int]]:
        """
        Applies T5-style span corruption to a single input sequence of token IDs and prepares the corresponding target
        labels.

        Example:
            Input sequence: [A, B, C, D, E, F, <EOS>]
            Corrupted input: [A, <extra_id_0>, E, F, <EOS>]
            Decoder labels: [<extra_id_0>, B, C, D, <extra_id_1>, <EOS>]

        Args:
            input_ids (list[int]): The input token IDs for a single sequence.

        Returns:
            tuple[list[int], list[int]]: A tuple containing the corrupted input token IDs and the corresponding target
                                         token IDs for the decoder.
        """
        if torch.is_tensor(input_ids):
            input_ids = input_ids.tolist()

        eos_token_id = self.tokenizer.eos_token_id

        if len(input_ids) > 0 and input_ids[-1] == eos_token_id:
            core_ids = input_ids[:-1]
        else:
            core_ids = input_ids

        spans = self._sample_spans(len(core_ids))

        if len(spans) == 0:
            return core_ids + [eos_token_id], [eos_token_id]

        corrupted_input, decoder_labels = [], []
        current_pos = 0

        for span_idx, (start, end) in enumerate(spans):
            sentinel_token_id = self._get_sentinel_id(span_idx)
            corrupted_input.extend(core_ids[current_pos:start])
            corrupted_input.append(sentinel_token_id)

            decoder_labels.append(sentinel_token_id)
            decoder_labels.extend(core_ids[start:end])

            current_pos = end

        corrupted_input.extend(core_ids[current_pos:])
        decoder_labels.append(self._get_sentinel_id(len(spans)))

        corrupted_input.append(eos_token_id)
        decoder_labels.append(eos_token_id)

        return corrupted_input, decoder_labels

    def _sample_spans(self, num_tokens: int) -> list[tuple[int, int]]:
        """
        Samples random, non-overlapping spans from the input sequence to be masked for the T5-style span corruption.

        Args:
            num_tokens (int): The length of the input sequence.

        Returns:
            list[tuple[int, int]]: A list of tuples where each tuple contains the start and end indices of a span to be
                                    corrupted.
        """
        if num_tokens < 2:
            return []

        # Determine the number of spans
        num_noise_tokens = max(1, int(num_tokens * self.noise_density))
        num_noise_tokens = min(num_noise_tokens, num_tokens - 1)
        num_spans = max(1, int(num_noise_tokens / self.mean_span_length))
        num_spans = min(num_spans, num_tokens - 1)

        # Sample random span lengths that sum up to num_noise_tokens
        span_lengths = []
        remaining_noise_tokens = num_noise_tokens
        for span_idx in range(num_spans):
            spans_left = num_spans - span_idx

            if spans_left == 1:
                span_length = remaining_noise_tokens
            else:
                max_span_length = remaining_noise_tokens - (spans_left - 1)
                span_length = np.random.randint(1, max_span_length + 1)

            span_lengths.append(span_length)
            remaining_noise_tokens -= span_length

        # Place spans without overlap
        spans, occupied = [], set()
        for span_length in span_lengths:
            placed = False
            for _ in range(100):  # Try up to 100 times to place the span
                start = np.random.randint(0, num_tokens - span_length + 1)
                positions = set(range(start, start + span_length))

                if occupied.isdisjoint(positions):
                    spans.append((start, start + span_length))
                    occupied.update(positions)
                    placed = True
                    break

            if not placed:  # simple fallback to minimize computational overhead
                continue

        return sorted(spans, key=lambda x: x[0])

    def _get_sentinel_id(self, span_idx: int) -> int:
        """
        Returns the sentinel token ID for a given span index based on the T5 tokenizer's special tokens.

        Args:
            span_idx (int): The index of the span for which to get the sentinel token ID.

        Returns:
            int: The sentinel token ID corresponding to the given span index.
        """
        sentinel_token = f"<extra_id_{span_idx}>"
        token_id = self.tokenizer.convert_tokens_to_ids(sentinel_token)

        if token_id is None or token_id == self.tokenizer.unk_token_id:
            raise ValueError(f"Sentinel token {sentinel_token} not found in the tokenizer vocabulary.")

        return token_id


class T5ChemWithProjectionHead(LLMWithProjectionHead):
    """
    Implementation of a `LLMWithProjectionHead` that uses a T5 model to generate molecular embeddings.
    """

    def _get_encoder_hidden_size(self) -> int:
        """
        Returns the hidden size of the T5 encoder to initialize the projection head for self-supervised fine tuning.
        """
        return self.config.d_model

    def get_embeddings(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            encoder_labels: Optional[torch.Tensor] = None,
            aux_labels: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Computes the embeddings from the T5 encoder for the given input token IDs and attention mask. The method
        calculates the token embeddings from the encoder and applies the specified pooling operation to obtain a
        fixed-size representation for each input sequence.

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
        if encoder_labels is not None:
            decoder_labels = encoder_labels.clone()
            decoder_labels[decoder_labels == -100] = self.pad_token_id
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=decoder_labels)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                output_hidden_states=True,
                return_dict=True
            )
            hidden_state = outputs.encoder_hidden_states[-1]
        else:
            outputs = self.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            hidden_state = outputs.last_hidden_state

        pooled_output = pool_transformer_outputs(self.pooling_operation, hidden_state, attention_mask.bool())

        return {
            "encoder_logits": outputs.get("logits"),
            "embeddings": pooled_output,
            "encoder_labels": encoder_labels,
            "aux_labels": aux_labels
        }


class T5ChemEncoder(LLMMoleculeEncoder):
    """
    Encoder for obtaining the LLM embeddings of SMILES strings from a pre-trained, chemistry-specific T5 model.
    """

    model_id: str = "GT4SD/multitask-text-and-chemistry-t5-base-augm"

    lora_levels = {
        0: ["q", "v"],
        1: ["q", "k", "v", "o"],
        2: ["q", "k", "v", "o", "wi", "wo"],
        3: "all-linear"
    }

    def __init__(
            self,
            lora_adaptor: str = None,
            use_projections: bool = False,
            pooling: Literal["cls", "mean"] = "mean",
            batch_size: int = 128
    ):
        """
        Initializes the T5ChemEncoder with the specified pooling_operation method.

        Args:
            lora_adaptor: Optional LoRA adaptor to load for the base model.
            use_projections: Whether to use the projection head for encoding. If True, the `encode` method will return
                             the latent space of the projection head instead of the encoder output.
            pooling: Pooling method to aggregate token embeddings into a single fixed-size vector.
                     Options include "cls" and "mean" (default).
            batch_size: Batch size for encoding SMILES strings.
        """
        LLMMoleculeEncoder.__init__(
            self,
            lora_adaptor=lora_adaptor,
            use_projections=use_projections,
            pooling=pooling,
            batch_size=batch_size
        )

        self.tokenizer = T5Tokenizer.from_pretrained(
            self.model_id,
            model_max_length=self.max_length,
            trust_remote_code=True
        )
        model_config = T5Config.from_pretrained(self.model_id)
        model_config.dropout_rate = 0
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id, config=model_config)

        if lora_adaptor is not None:
            load_dir = Path(get_config().cache_dir) / "models" / lora_adaptor
            self.model = T5ChemWithProjectionHead.from_pretrained(base_model=model, load_directory=load_dir)
        else:
            self.model = T5ChemWithProjectionHead(model=model, pooling_operation=pooling, projection_depth=0)

        self.embedding_dim = self.model.config.d_model
        self.model.eval()

    @property
    def model_name(self) -> str:
        return f"T5Chem-{self.pooling}"

    def _tokenize_batch(self, smiles_batch: list[str], **tokenization_kwargs) -> dict:
        """
        Tokenizes a batch of SMILES strings for input into the T5 chemistry model.
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

        aux_model = T5ChemWithProjectionHead(
            model=peft_model,
            pooling_operation=self.pooling,
            projection_dim=config.projection_dim,
            output_dim=config.projection_target_dim,
            projection_depth=config.projection_depth,
            auxiliary_labels=config.finetuning_requires_auxiliary_labels,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        collator = PairwiseSMILESCollatorForSpanCorruption(
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            padding="longest",
            mean_span_length=config.mean_span_length,
            noise_density=config.noise_density,
            ignore_token_id=-100
        )

        train_args = config.get_training_args(tag=tag)

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

        aux_model.save_pretrained(str(Path(get_config().cache_dir) / "models" / tag))
        self.model = aux_model
        self.lora_name = tag
        self.eval(), self.model.eval()
