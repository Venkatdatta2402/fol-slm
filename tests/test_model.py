import sys
from pathlib import Path

import pytest
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.model.encoder import FrozenT5Encoder
from src.model.decoder import TransformerDecoder
from src.model import FOLModel


@pytest.fixture(scope="module")
def model_config():
    return {
        "encoder_name": "t5-small",  # smaller for fast tests
        "decoder_layers": 2,
        "decoder_heads": 4,
        "decoder_dim": 256,
        "decoder_ff_dim": 512,
        "decoder_dropout": 0.1,
        "max_seq_len": 64,
    }


@pytest.fixture(scope="module")
def vocab_size():
    return 32128  # T5 vocab


@pytest.fixture(scope="module")
def fol_model(model_config, vocab_size):
    return FOLModel(model_config, vocab_size)


class TestFrozenEncoder:
    def test_encoder_params_frozen(self, fol_model):
        for param in fol_model.encoder.parameters():
            assert not param.requires_grad, "Encoder param should be frozen"

    def test_encoder_output_deterministic(self, fol_model):
        input_ids = torch.randint(0, 100, (2, 10))
        mask = torch.ones_like(input_ids)
        out1 = fol_model.encoder(input_ids, mask)
        out2 = fol_model.encoder(input_ids, mask)
        assert torch.allclose(out1, out2), "Frozen encoder should give identical outputs"

    def test_encoder_no_grad_after_backward(self, fol_model):
        for param in fol_model.encoder.parameters():
            assert param.grad is None, "Encoder should never accumulate gradients"


class TestDecoder:
    def test_decoder_output_shape(self, model_config, vocab_size):
        decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=model_config["decoder_dim"],
            n_heads=model_config["decoder_heads"],
            n_layers=model_config["decoder_layers"],
            d_ff=model_config["decoder_ff_dim"],
            dropout=model_config["decoder_dropout"],
            max_seq_len=model_config["max_seq_len"],
            encoder_d_model=512,  # t5-small
        )
        B, T_dec, T_enc = 2, 8, 10
        tgt_ids = torch.randint(0, vocab_size, (B, T_dec))
        encoder_out = torch.randn(B, T_enc, 512)
        enc_mask = torch.ones(B, T_enc)

        logits = decoder(tgt_ids, encoder_out, enc_mask)
        assert logits.shape == (B, T_dec, vocab_size)

    def test_cross_attn_weights_captured(self, model_config, vocab_size):
        decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=model_config["decoder_dim"],
            n_heads=model_config["decoder_heads"],
            n_layers=model_config["decoder_layers"],
            d_ff=model_config["decoder_ff_dim"],
            dropout=model_config["decoder_dropout"],
            max_seq_len=model_config["max_seq_len"],
            encoder_d_model=512,
        )
        B, T_dec, T_enc = 2, 8, 10
        tgt_ids = torch.randint(0, vocab_size, (B, T_dec))
        encoder_out = torch.randn(B, T_enc, 512)

        decoder(tgt_ids, encoder_out)
        assert decoder.cross_attn_weights is not None
        assert len(decoder.cross_attn_weights) == model_config["decoder_layers"]
        # Shape: (B, n_heads, T_dec, T_enc)
        w = decoder.cross_attn_weights[0]
        assert w.shape == (B, model_config["decoder_heads"], T_dec, T_enc)


class TestFOLModel:
    def test_forward_shape(self, fol_model):
        B, S_enc, S_dec = 2, 10, 8
        input_ids = torch.randint(0, 100, (B, S_enc))
        mask = torch.ones(B, S_enc, dtype=torch.long)
        dec_ids = torch.randint(0, 100, (B, S_dec))

        logits = fol_model(input_ids, mask, dec_ids)
        assert logits.shape == (B, S_dec, 32128)

    def test_only_decoder_trainable(self, fol_model):
        trainable = sum(p.numel() for p in fol_model.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in fol_model.parameters() if not p.requires_grad)
        assert trainable > 0, "Decoder params should be trainable"
        assert frozen > 0, "Encoder params should be frozen"
        # Decoder should be smaller than encoder
        assert trainable < frozen

    def test_gradients_flow_through_cross_attention(self, fol_model):
        B, S_enc, S_dec = 2, 10, 8
        input_ids = torch.randint(0, 100, (B, S_enc))
        mask = torch.ones(B, S_enc, dtype=torch.long)
        dec_ids = torch.randint(0, 100, (B, S_dec))

        logits = fol_model(input_ids, mask, dec_ids)
        loss = logits.sum()
        loss.backward()

        # Cross-attention params should have gradients
        for layer in fol_model.decoder.layers:
            for name, param in layer.cross_attn.named_parameters():
                assert param.grad is not None, f"Cross-attn param {name} should have gradient"
                assert param.grad.abs().sum() > 0, f"Cross-attn param {name} gradient should be non-zero"

        # Encoder params should NOT have gradients
        for param in fol_model.encoder.parameters():
            assert param.grad is None
