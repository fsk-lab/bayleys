from dataclasses import dataclass

from .base import MoleculeEncoder
from .fingerprint import MorganFingerprintEncoder
from .mordred import MordredEncoder
from .rdkit_desc import RDKitDescriptorEncoder
from .llm_roberta import ChemBERTaZincEncoder, ChemBERTa100MEncoder, MolFormerXLEncoder
from .llm_bert_trainable import TrainableBERTEncoder
from .llm_t5chem import T5ChemEncoder
from .llm_smi_ted import SmiTedEncoder

from ..utils import AbstractConfig, non_abstract_subclasses


ENCODER_FACTORY = {cls.__name__: cls for cls in non_abstract_subclasses(MoleculeEncoder)}


@dataclass
class MoleculeEncoderConfig(AbstractConfig):

    encoder_type: str
    encoder_kwargs: dict = None

    def __post_init__(self):
        """
        Validates the molecule encoder configuration after initialization by performing the following checks:
          - Ensures that the specified encoder_type exists in the ENCODER_FACTORY.
          - Initializes encoder_kwargs to an empty dictionary if it is None.
        """
        if self.encoder_type not in ENCODER_FACTORY:
            raise ValueError(f"Molecule encoder type '{self.encoder_type}' is not recognized!")

        if self.encoder_kwargs is None:
            self.encoder_kwargs = {}

    def get_encoder(self) -> MoleculeEncoder:
        """
        Instantiates and returns the molecule encoder based on the configuration.

        Returns:
            An instance of the specified molecule encoder.
        """
        encoder_class = ENCODER_FACTORY[self.encoder_type]
        encoder = encoder_class(**self.encoder_kwargs)
        return encoder
