import copy
import torch
from src.model import FOLModel
from src.model.decoder import TransformerDecoder


def create_model_fresh_decoder(config: dict, vocab_size: int, shared_encoder=None) -> FOLModel:
    """Create a FOLModel with a fresh (randomly initialized) decoder.

    Optionally reuses a pre-loaded frozen encoder to save memory and load time.

    Args:
        config: Model config dict with encoder_name, decoder_* params.
        vocab_size: Vocabulary size for the decoder.
        shared_encoder: If provided, reuses this encoder instead of loading from disk.

    Returns:
        FOLModel with fresh decoder weights.
    """
    model = FOLModel(config, vocab_size)

    if shared_encoder is not None:
        model.encoder = shared_encoder

    return model
