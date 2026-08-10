from pathlib import Path
from typing import Optional
import torch
import smi_ted
from smi_ted.tokenizer import SmiTedHFTokenizer
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoModel, AutoTokenizer
from peft import get_peft_model

from .llm import LLMMoleculeEncoder, LLMWithProjectionHead
from .pretraining import PreTrainingConfig, MixedObjectiveTrainer
from .llm_roberta import PairwiseSMILESCollatorForMaskedLM

from ..config import get_config
from ..molecule_library import MemoryMappedDataset
from ..utils import masked_language_modeling_loss, LoggerCallback


class SmiTedWithProjectionHead(LLMWithProjectionHead):
    """
    Implementation of a `LLMWithProjectionHead` that uses the SMI-TED model to generate molecular embeddings.
    """

    def _get_encoder_hidden_size(self) -> int:
        """
        Returns the hidden size of the SMI-TED encoder to initialize the projection head for self-supervised fine
        tuning.
        """
        return self.model.config.n_embd

    def get_embeddings(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            encoder_labels: Optional[torch.Tensor] = None,
            aux_labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Computes the embeddings from the SMI-TED encoder for the given input token IDs and attention mask. The method
        calculates the token embeddings from the encoder and applies the autoencoder pooling to obtain a fixed-size
        representation for each input sequence. T

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
        token_embeddings = self.model.smi_ted.encoder(input_ids, attention_mask)
        encoder_logits = self.model.smi_ted.encoder.lang_model(token_embeddings)
        token_embeddings_pooled = self.model.smi_ted.decoder.autoencoder.encoder(
            token_embeddings.view(-1, self.model.smi_ted.max_len * self.model.smi_ted.n_embd)
        )

        return {
            "encoder_logits": encoder_logits,
            "embeddings": token_embeddings_pooled,
            "encoder_labels": encoder_labels,
            "aux_labels": aux_labels
        }


def _patch_smited():
    """
    Patches the `__init__` method of the `MaterialsSmiTedHFModel` class for compatibility with newer `transformer`
    versions.
    """
    from smi_ted.model import MaterialsSmiTedHFModel

    original_init = MaterialsSmiTedHFModel.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)

        if not hasattr(self, "all_tied_weights_keys"):
            try:
                self.post_init()
            except:
                self.all_tied_weights_keys = {}

    MaterialsSmiTedHFModel.__init__ = patched_init


class SmiTedEncoder(LLMMoleculeEncoder):
    """
    Molecule encoder that uses the SMI-TED model to generate molecular embeddings.

    Args:
    lora_adaptor (str): Optional Path to the LoRA adaptor to load for the base model. If None, the base model without
                        any fine-tuning will be used.
    """

    model_id = "bisectgroup/materials-smi-ted-fork"
    max_length = 202

    lora_levels = {
        0: ["query_projection", "value_projection"],
        1: ["query_projection", "value_projection", "key_projection", "out_projection", "lat"],
        2: ["query_projection", "value_projection", "key_projection", "out_projection", "lat", "linear1", "linear2", "autoencoder.encoder.fc1"],
        3: "all-linear"
    }

    def __init__(self, lora_adaptor: str = None, use_projections: bool = False, batch_size: int = 128):
        """
        Initializes the SmiTedEncoder with the specified LoRA adaptor and batch size.

        Args:
            lora_adaptor: Optional LoRA adaptor to load for the base model.
            use_projections: Whether to use the projection head for encoding. If True, the `encode` method will return
                             the latent space of the projection head instead of the encoder output.
            batch_size: Batch size for encoding SMILES strings.
        """
        LLMMoleculeEncoder.__init__(
            self,
            lora_adaptor=lora_adaptor,
            use_projections=use_projections,
            batch_size=batch_size
        )

        self.config = AutoConfig.from_pretrained(self.model_id)
        self.tokenizer = SmiTedHFTokenizer(hf_hub_download(repo_id=self.model_id, filename="vocab.txt"))
        _patch_smited()
        model = AutoModel.from_pretrained(self.model_id)

        if lora_adaptor is not None:
            load_dir = Path(get_config().cache_dir) / "models" / lora_adaptor
            self.model = SmiTedWithProjectionHead.from_pretrained(base_model=model, load_directory=load_dir)
            self.model.model = self.model.model.merge_and_unload()
        else:
            self.model = SmiTedWithProjectionHead(model=model, projection_depth=0)

        self.model.model.smi_ted.tokenizer = self.tokenizer
        self.model.model.smi_ted.set_padding_idx_from_tokenizer()
        self.model.eval()

        self.embedding_dim = self.model.config.n_embd

    @property
    def model_name(self) -> str:
        return "SmiTed"

    def _tokenize_batch(self, smiles_batch: list[str], **tokenization_kwargs) -> dict:
        """
        Tokenizes a batch of SMILES strings using the SMI-TED tokenizer.
        """
        return self.tokenizer(
            smiles_batch,
            truncation=True,
            max_length=self.__class__.max_length,
            # add_special_tokens=True,
            **tokenization_kwargs
        )

    def _fine_tune(
            self, config: PreTrainingConfig,
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
            train_data (MemoryMappedDataset): Training dataset
            val_data (MemoryMappedDataset): Validation dataset
            tag (str): Tag to identify the fine-tuned model.
        """
        if self.model.is_lora_model:
            raise ValueError(f"Model is already a LoRA model. Cannot fine-tune further with LoRA.")

        config.lora_target_modules = self.lora_levels[config.lora_target_module_level]
        lora_cfg = config.get_lora_config()
        peft_model = get_peft_model(self.model.model, lora_cfg)
        num_trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        self.max_length = self.__class__.max_length  # Reset to fixed max length, required for SmiTED model

        aux_model = SmiTedWithProjectionHead(
            model=peft_model,
            projection_dim=config.projection_dim,
            output_dim=config.projection_target_dim,
            projection_depth=config.projection_depth,
            auxiliary_labels=config.finetuning_requires_auxiliary_labels,
        )

        collator = PairwiseSMILESCollatorForMaskedLM(
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            padding="max_length",
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

        aux_model.save_pretrained(str(Path(get_config().cache_dir) / "models" / tag))
        aux_model.model = aux_model.model.merge_and_unload()
        self.model = aux_model
        self.model.model.smi_ted.tokenizer = self.tokenizer
        self.model.model.smi_ted.set_padding_idx_from_tokenizer()
        self.lora_name = tag
        self.model.eval(), self.eval()
