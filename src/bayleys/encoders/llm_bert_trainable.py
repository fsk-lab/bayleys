from typing import Literal, Optional
from pathlib import Path
from dataclasses import dataclass
from transformers import BertTokenizerFast, BertConfig, BertForMaskedLM

from .pretraining import PreTrainingConfig, MixedObjectiveTrainer
from .llm_roberta import EncoderOnlyLLMEncoder, EncoderOnlyLLMWithProjectionHead, PairwiseSMILESCollatorForMaskedLM

from ..config import get_config
from ..utils import AbstractConfig, masked_language_modeling_loss, LoggerCallback
from ..utils.tokenization import train_smiles_tokenizer, tokenize_full_library
from ..molecule_library import MoleculeLibrary


@dataclass
class BERTArchitecture(AbstractConfig):
    """
    Configuration for the architecture of a BERT-based molecule encoder.
    """
    name: str = "RegularBERT"
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072


class TrainableBERTEncoder(EncoderOnlyLLMEncoder):
    """
    BERT-based molecule encoder that can be pretrained on a molecule library using masked language modeling (MLM).
    """
    model_id = None

    def __init__(
            self,
            architecture: dict | BERTArchitecture,
            model_id: Optional[str] = None,
            pooling: Literal["cls", "mean", "max"] = "mean",
            batch_size: int = 256,
            **kwargs
    ):

        self.model_id = model_id

        EncoderOnlyLLMEncoder.__init__(self, pooling=pooling, batch_size=batch_size)

        if isinstance(architecture, dict):
            self.architecture = BERTArchitecture.from_dict(architecture)
        else:
            self.architecture = architecture

        self.embedding_dim = self.architecture.hidden_size

    @property
    def model_name(self) -> str:
        return f"{self.architecture.name}-{self.pooling}"

    def pretrain(
            self,
            config: PreTrainingConfig,
            library: MoleculeLibrary,
    ):
        """
        Pretrains a BERT-type encoder using masked language modeling (MLM) on the provided molecule library.

          1. Trains a BertWordPieceTokenizer on the SMILES strings in the library.
          2. Splits the dataset into training and validation sets (90% train, 10% validation).
          3. Tokenizes the train and test datasets into a MemoryMappedDataset each.
          4. Trains the BERT model using the HuggingFace Trainer API, using a masked language modeling (MLM) objective.
          5. Saves the trained model and tokenizer to disk.
        """
        config.dataset_size = len(library)

        tag = f"{self.model_name}-{library.name}"
        model_dir = Path(get_config().cache_dir) / "models" / tag

        train_smiles_tokenizer(library, model_dir / "tokenizer", num_views_per_molecule=config.num_train_epochs)
        self.logger.info(f"Successfully trained a BertWordPiece tokenizer for {tag}.")
        tokenizer = BertTokenizerFast(
            vocab_file=str(model_dir / "tokenizer" / "vocab.txt"),
            do_lower_case=False,
            tokenize_chinese_chars=False,
            clean_up_tokenization_spaces=False
        )

        train_library, val_library = library.train_test_split(test_size=0.1)

        train_dataset_tokenized, train_max_length = tokenize_full_library(
            library=train_library,
            tokenizer=tokenizer,
            model_max_length=self.max_length,
            tmp_dir=Path(get_config().tmp_dir) / library.name / "train_tokenized",
            contrastive_learning=config.finetuning_is_contrastive,
            num_examples_per_molecule=config.num_train_epochs
        )
        self.logger.info(f"Tokenized the training dataset for {tag}.")
        val_dataset_tokenized, val_max_length = tokenize_full_library(
            library=val_library,
            tokenizer=tokenizer,
            model_max_length=self.max_length,
            tmp_dir=Path(get_config().tmp_dir) / library.name / "val_tokenized",
            contrastive_learning=config.finetuning_is_contrastive,
            num_examples_per_molecule=config.num_train_epochs
        )
        self.logger.info(f"Tokenized the validation dataset for {tag}.")
        self.max_length = max(train_max_length, val_max_length)

        model_config = BertConfig(
            vocab_size=len(tokenizer),
            hidden_size=self.architecture.hidden_size,
            num_hidden_layers=self.architecture.num_hidden_layers,
            num_attention_heads=self.architecture.num_attention_heads,
            intermediate_size=self.architecture.intermediate_size,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=self.max_length,
            type_vocab_size=1,
            pad_token_id=tokenizer.pad_token_id,
            unk_token_id=tokenizer.unk_token_id,
            cls_token_id=tokenizer.cls_token_id,
            sep_token_id=tokenizer.sep_token_id,
            mask_token_id=tokenizer.mask_token_id,
        )
        model = EncoderOnlyLLMWithProjectionHead(
            model=BertForMaskedLM(config=model_config),
            pooling_operation=self.pooling,
            projection_depth=config.projection_depth,
            projection_dim=config.projection_dim,
            output_dim=config.projection_target_dim,
            auxiliary_labels=False
        )

        collator = PairwiseSMILESCollatorForMaskedLM(
            tokenizer=tokenizer,
            max_length=self.max_length,
            padding="longest",
            mlm_probability=config.mlm_probability
        )

        training_args = config.get_training_args(tag=tag)

        trainer = MixedObjectiveTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset_tokenized,
            eval_dataset=val_dataset_tokenized,
            data_collator=collator,
            encoder_loss=masked_language_modeling_loss,
            encoder_loss_kwargs={"ignore_index": -100},
            projection_loss=config.finetuning_projection_loss_function,
            projection_loss_kwargs=config.finetuning_projection_loss_kwargs,
            projection_loss_weight=config.projection_loss_weight,
            callbacks=[LoggerCallback(self.logger)]
        )

        trainer.train()
        tokenizer.save_pretrained(str(model_dir))
        trainer.save_model(str(model_dir))

        self.tokenizer = tokenizer
        self.model = model


