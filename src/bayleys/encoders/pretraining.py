from abc import ABC, abstractmethod
from typing import Literal, Optional, Callable, Any
import os
from pathlib import Path
from dataclasses import dataclass
import torch
from transformers import TrainingArguments, Trainer, PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.trainer import WEIGHTS_NAME, TRAINING_ARGS_NAME, logger
from peft import LoraConfig

from ..utils import AbstractConfig
from ..utils import contrastive_loss
from ..config import get_config


LOSS_FACTORY = {"mse": torch.nn.functional.mse_loss, "contrastive": contrastive_loss}


@dataclass
class PreTrainingConfig(AbstractConfig):
    """
    Coniguration class for pre-training or fine-tuning transformer-based models on molecular libraries. Can be used for
    pre-training from scratch, or for fine-tuning an existing model with a projection head using LoRA.
    """
    # General training hyperparameters
    optimizer: Literal["Adam", "AdamW"] = "AdamW"
    num_train_epochs: int = 5
    train_batch_size: int = 256
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0

    # Specific hyperparameters for finetuning a model with a projection head using LoRA.
    projection_loss: Optional[Literal["mse", "contrastive"]] = "contrastive"
    projection_loss_weight: Optional[float] = 0.0
    projection_dim: int = 128
    projection_depth: int = 1

    lora_r: int = 4
    lora_alpha: int = 8
    lora_dropout: float = 0.05
    lora_target_module_level: int = 0
    lora_bias: Literal["none", "all", "lora_only"] = "none"

    # The following fields are only used for certain strategies, but are included in the main config for simplicity.
    # They should be set by the user when applicable, but are not required for all fine-tuning strategies.
    mlm_probability: float = 0.15
    noise_density: float = 0.15
    mean_span_length: float = 3.0
    contrastive_loss_temperature: float = 0.1

    # The following fields are specific to the model architecture, and should not be set by the user. Instead, they
    # are set and used internally by the fine-tuning pipeline.
    projection_target_dim = None
    lora_target_modules: list = None
    max_seq_length: int = None
    dataset_size: int = None

    def __post_init__(self):
        if self.projection_loss is not None and self.projection_loss not in LOSS_FACTORY:
            raise ValueError(f"Unsupported projection loss type: {self.projection_loss}.")

    def __repr__(self):
        strategy = "supervised" if self.finetuning_requires_auxiliary_labels else "self-supervised"
        return (f"Fine-tuning configuration for LoRA-based {strategy} fine-tuning (Projection loss: "
                f"{self.projection_loss}, Projection loss weight: {self.projection_loss_weight}, Projection head depth:"
                f" {self.projection_depth})")

    @property
    def finetuning_requires_auxiliary_labels(self) -> bool:
        """
        Indicates whether the fine-tuning configuration is for supervised fine-tuning (as opposed to self-supervised
        fine-tuning). This is determined based on whether a projection loss is specified in the configuration.

        Returns:
            bool: True if the configuration is for supervised fine-tuning, False if it is for self-supervised
                  fine-tuning.
        """
        return self.projection_loss == "mse"

    @property
    def finetuning_is_contrastive(self) -> bool:
        """
        Indicates whether the fine-tuning configuration is for contrastive fine-tuning. This is determined based on
        whether the projection loss is set to "contrastive" in the configuration.

        Returns:
            bool: True if the configuration is for contrastive fine-tuning, False otherwise.
        """
        return self.projection_loss == "contrastive" and self.projection_loss_weight > 0.0

    def get_lora_config(self) -> LoraConfig:
        """
        Generates and returns the LoRaConfig object based on the fine-tuning configuration.

        Returns:
            LoraConfig: The LoRA configuration for model fine-tuning.
        """
        if self.lora_target_modules is None:
            raise ValueError("`lora_target_modules` must be specified before calling `get_lora_config()`.")

        return LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.lora_target_modules,
            bias=self.lora_bias,
        )

    @property
    def finetuning_projection_loss_function(self) -> Callable:
        """
        Generates and returns the projection loss function based on the fine-tuning configuration.

        Returns:
            Callable: The projection loss function.
        """
        return LOSS_FACTORY[self.projection_loss]

    @property
    def finetuning_projection_loss_kwargs(self) -> dict[str, any]:
        """
        Generates and returns any additional keyword arguments needed for the projection loss function based on the
        fine-tuning configuration.

        Returns:
            dict[str, any]: A dictionary of additional keyword arguments for the projection loss function.
        """
        if self.projection_loss == "contrastive":
            return {"temperature": self.contrastive_loss_temperature}
        else:
            return {}

    def get_training_args(self, tag: str) -> TrainingArguments:
        """
        Generates and returns the TrainingArguments object based on the configuration.

        Args:
            tag (str): The output directory tag for saving training results.

        Returns:
            TrainingArguments: The training arguments for the Hugging Face Trainer.
        """
        if self.dataset_size is None:
            raise ValueError("Dataset_size must be specified in the pre-training configuration to create a "
                             "TrainingArguments instance.")

        global_config = get_config()

        return TrainingArguments(
            output_dir=str(Path(global_config.cache_dir) / "models" / tag),
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            lr_scheduler_type=self.lr_scheduler_type,
            warmup_ratio=self.warmup_ratio,
            weight_decay=self.weight_decay,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_steps=250,
            label_names=["encoder_labels", "aux_labels"] if self.lora_target_modules is not None else ["labels"],
            gradient_checkpointing=False,
            disable_tqdm=True,
            report_to="none",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            save_total_limit=2,
            greater_is_better=False,
            remove_unused_columns=False,
            dataloader_num_workers=0 if global_config.device == "cpu" else 8,
            dataloader_pin_memory=True,
            bf16=True if global_config.device == "cuda" and torch.cuda.is_bf16_supported() else False,
            use_cpu=True if global_config.device == "cpu" else False,
        )

    @property
    def unique_id(self) -> str:
        """
        Overrides the unique_id property to generate a unique identifier for the fine-tuning configuration that is
        independent of any model- or library-specific information. This ensures that the same fine-tuning configuration
        will have the same unique ID regardless of the specific model or library it is applied to.
        """
        aux_config = PreTrainingConfig(
            projection_loss=self.projection_loss,
            projection_loss_weight=self.projection_loss_weight,
            projection_depth=self.projection_depth,
            projection_dim=self.projection_dim,
            lora_r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            lora_target_module_level=self.lora_target_module_level,
            lora_bias=self.lora_bias,
            optimizer=self.optimizer,
            num_train_epochs=self.num_train_epochs,
            train_batch_size=self.train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            lr_scheduler_type=self.lr_scheduler_type,
            warmup_ratio=self.warmup_ratio,
            weight_decay=self.weight_decay,
            mlm_probability=self.mlm_probability,
        )
        return AbstractConfig.unique_id.fget(aux_config)


@dataclass
class PairwiseSMILESCollator(ABC):
    """
    Abstract base class for collating batches of SMILES strings for fine-tuning transformer-based models. This class
    provides a common interface for collators that can be used for the following types of tasks:
       - self-supervised language modeling (e.g. masked language modeling, span corruption)
       - contrastive modeling (e.g. SimCLR-style contrastive learning with original and randomized SMILES pairs)
       - supervised modeling (e.g. projection-based supervised fine-tuning with auxiliary labels)

    Args:
        tokenizer: The tokenizer to use for tokenizing the SMILES strings.
        max_length: The maximum sequence length for tokenization.
        pairwise: Whether to include both original and randomized SMILES strings in the batch (default: True).
        ignore_token_id: The token ID to use for non-masked tokens in the labels (default: -100)
    """
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast
    max_length: int
    padding: str = "longest"
    ignore_token_id: int = -100

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """
        Collates a batch of examples for masked language modeling by applying random token masking.

        Args:
            examples (list[dict[str, Any]]): A list of examples, where each example is a dictionary containing a
                                            'smiles' key.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing the collated and masked input features.
        """
        inputs = {key: [ex[key] for ex in examples] for key in examples[0].keys()}

        if "input_ids_positive" in inputs and "special_tokens_mask_positive" in inputs:
            inputs["input_ids"] = inputs["input_ids"] + inputs.pop("input_ids_positive")
            inputs["special_tokens_mask"] = inputs["special_tokens_mask"] + inputs.pop("special_tokens_mask_positive")
        if "input_ids_negative" in inputs and "special_tokens_mask_negative" in inputs:
            inputs["input_ids"] = inputs["input_ids"] + inputs.pop("input_ids_negative")
            inputs["special_tokens_mask"] = inputs["special_tokens_mask"] + inputs.pop("special_tokens_mask_negative")

        features = self.create_lm_features(inputs)

        for key in inputs.keys():
            if key not in list(features.keys()) and not key.startswith("special_tokens_mask"):
                if isinstance(inputs[key][0], list):
                    features[key] = torch.tensor(inputs[key], dtype=torch.long)
                else:
                    features[key] = torch.stack(inputs[key], dim=0)

        return features

    def pad_batch(self, features: dict[str, list]) -> dict[str, torch.Tensor]:
        """
        Pads a batch of tokenized features using the tokenizer's pad method.

        Args:
            features (dict[str, list]): A dictionary of tokenized features to pad.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing the padded features as torch tensors.
        """
        padded = self.tokenizer.pad(
            {"input_ids": features["input_ids"], "special_tokens_mask": features["special_tokens_mask"]},
            padding=self.padding,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return padded

    @abstractmethod
    def create_lm_features(self, tokenized: dict[str, list]) -> dict[str, torch.Tensor]:
        raise NotImplementedError("Subclasses of PairwiseSMILESCollator must implement `create_lm_features`.")


class MixedObjectiveTrainer(Trainer):
    """
    Custom Trainer class that computes a mixed objective combining the original loss from an encoder with a
    projection-based loss. This is designed for fine-tuning transformer-based models with a projection head on top,
    as defined in the `LLMWithProjectionHead` class.

    Args:
        args: Positional arguments for the base `Trainer` class.
        projection_loss (Callable): A function that computes the projection loss given the model outputs and labels.
        projection_loss_kwargs (dict): Additional keyword arguments to pass to the projection loss function.
        projection_loss_weight (float): The weight to apply to the projection loss relative to the encoder loss.
        kwargs: Additional keyword arguments for the base `Trainer` class.
    """

    def __init__(
            self,
            *args,
            encoder_loss: Callable,
            encoder_loss_kwargs: dict[str, any],
            projection_loss: Callable,
            projection_loss_kwargs: dict[str, any],
            projection_loss_weight: float = 0.1,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.encoder_loss = encoder_loss
        self.encoder_loss_kwargs = encoder_loss_kwargs
        self.projection_loss = projection_loss
        self.projection_loss_kwargs = projection_loss_kwargs
        self.projection_loss_weight = projection_loss_weight

    def compute_loss(
            self,
            model: torch.nn.Module,
            inputs: dict,
            return_outputs: bool = False,
            num_items_in_batch: int = None
    ) -> tuple[torch.Tensor, dict] | torch.Tensor:
        """
        Computes the mixed loss for the given inputs. Combines the original encoder loss with the projection loss.

        Args:
            model: The model being trained.
            inputs: The input batch for which to compute the loss.
            return_outputs: Whether to return the model outputs along with the loss.
            num_items_in_batch: The number of items in the current batch (not used here).

        Returns:
            If return_outputs is True, returns a tuple of (loss, outputs), where loss is the computed mixed loss and
            outputs is the dictionary of model outputs. If return_outputs is False, returns only the computed mixed
            loss.
        """
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        if self.projection_loss_weight < 1.0:
            encoder_labels = inputs.get("encoder_labels", None)
            if encoder_labels is None:
                raise ValueError("Encoder labels must be provided for a partial LM objective.")
        else:
            encoder_labels = None

        if self.projection_loss_weight > 0.0 and self.projection_loss == LOSS_FACTORY["mse"]:
            aux_labels = inputs.get("aux_labels", None)
            if aux_labels is None:
                raise ValueError("Auxiliary labels must be provided for MSE projection loss.")
        else:
            aux_labels = None

        if self.projection_loss_weight > 0.0 and self.projection_loss == LOSS_FACTORY["contrastive"]:
            if input_ids.shape[0] % 3 != 0:
                raise ValueError("Batch size must be divisible by 3 for contrastive loss (anchor, positive, negative).")
            batch_size = input_ids.shape[0] // 3
        else:
            batch_size = input_ids.shape[0]

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_labels=encoder_labels,
            aux_labels=aux_labels
        )

        if not self.projection_loss_weight < 1.0:
            encoder_loss = torch.tensor(0.0, device=input_ids.device)
        else:
            encoder_loss = self.encoder_loss(
                outputs["encoder_logits"][:batch_size, ...],
                encoder_labels[:batch_size, ...],
                **self.encoder_loss_kwargs
            )

        if not self.projection_loss_weight > 0.0:
            projection_loss = torch.tensor(0.0, device=encoder_loss.device)
        elif self.projection_loss == LOSS_FACTORY["mse"]:
            projections = outputs["projections"][:batch_size, ...]
            projection_loss = self.projection_loss(
                projections.view(-1, projections.shape[-1]),
                aux_labels.view(-1, projections.shape[-1]),
            )
        elif self.projection_loss == LOSS_FACTORY["contrastive"]:
            projections = outputs["projections"][:batch_size, ...]
            positive_projections = outputs["projections"][batch_size: 2 * batch_size, ...]
            negative_projections = outputs["projections"][2 * batch_size:, ...]
            projection_loss = self.projection_loss(
                anchors=projections,
                positives=positive_projections,
                negatives=negative_projections,
                **self.projection_loss_kwargs
            )
        else:
            raise NotImplementedError(f"Projection loss type '{self.projection_loss}' not implemented.")

        mixed_loss = encoder_loss + self.projection_loss_weight * projection_loss

        if return_outputs:
            return mixed_loss, outputs
        else:
            return mixed_loss

    def prediction_step(
            self,
            model: torch.nn.Module,
            inputs: dict,
            prediction_loss_only: bool,
            ignore_keys=None
    ) -> tuple[torch.Tensor, None, None]:
        """
        Custom implementation of the prediction step that computes the loss for evaluation.

        Args:
            model: The model being evaluated.
            inputs: The input batch for which to compute the loss.
            prediction_loss_only: Whether to return only the loss value (not used here).
            ignore_keys: Keys to ignore in the model outputs (not used here).
        """
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            loss = self.compute_loss(model, inputs)

        return loss.detach(), None, None

    def _save(self, output_dir: Optional[str] = None, state_dict: Optional[dict[str, torch.Tensor]] = None):
        """
        Custom save method that saves the model without relying on the `safetensors` saving strategy (i.e. where
        `safetensors.torch.save_file(state_dict, ...)` fails because several `state_dict` entries share the same
        underlying tensor).

        Args:
            output_dir: The directory where the model should be saved. If None, uses the default output directory.
            state_dict: The state dictionary of the model to save. If None, uses the model's current state dictionary.
        """
        # Original Code from Trainer._save()
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        # Custom Code: Save the model state dictionary without using safetensors
        if state_dict is None:
            model_to_save = self.accelerator.unwrap_model(self.model, keep_torch_compile=False)
            state_dict = model_to_save.state_dict()

        torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))

        # Original Code from Trainer._save() for saving tokenizer and processing class
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        elif (
            self.data_collator is not None
            and hasattr(self.data_collator, "tokenizer")
            and self.data_collator.tokenizer is not None
        ):
            logger.info("Saving Trainer.data_collator.tokenizer by default as Trainer.processing_class is `None`")
            self.data_collator.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))


