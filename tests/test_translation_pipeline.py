import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from build_transformer.dataset import BilingualDataset
from build_transformer.model import MultiheadAttiontionBlock, build_transformer


def _test_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(
        WordLevel(
            {
                "[UNK]": 0,
                "[PAD]": 1,
                "[SOS]": 2,
                "[EOS]": 3,
                "hello": 4,
                "你好": 5,
            },
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer


def test_bilingual_dataset_builds_shifted_labels_and_masks() -> None:
    tokenizer = _test_tokenizer()
    dataset = BilingualDataset(
        [{"translation": {"en": "hello", "zh": "你好"}}],
        tokenizer,
        tokenizer,
        "en",
        "zh",
        seq_len=5,
    )

    sample = dataset[0]

    assert sample["encoder_input"].tolist() == [2, 4, 3, 1, 1]
    assert sample["decoder_input"].tolist() == [2, 5, 1, 1, 1]
    assert sample["label"].tolist() == [5, 3, 1, 1, 1]
    assert sample["encoder_mask"].shape == (1, 1, 5)
    assert sample["decoder_mask"].shape == (1, 5, 5)
    assert sample["encoder_mask"].dtype == torch.bool
    assert not sample["decoder_mask"][0, 0, 1]
    assert sample["decoder_mask"][0, 1, 0]


def test_attention_masks_padding_keys_and_preserves_shape() -> None:
    attention = MultiheadAttiontionBlock(d_model=8, h=2, dropout=0.0)
    inputs = torch.randn(2, 4, 8)
    mask = torch.tensor(
        [[[[1, 1, 0, 0]]], [[[1, 1, 1, 0]]]],
        dtype=torch.bool,
    )

    outputs = attention(inputs, inputs, inputs, mask)

    assert outputs.shape == inputs.shape
    assert attention.attention_scores.shape == (2, 2, 4, 4)
    assert torch.all(attention.attention_scores[0, :, :, 2:] == 0)
    assert torch.all(attention.attention_scores[1, :, :, 3] == 0)


def test_encoder_decoder_forward_returns_target_vocabulary_scores() -> None:
    model = build_transformer(
        src_vocab_size=12,
        tgt_vocab_size=14,
        src_seq_len=5,
        tgt_seq_len=5,
        d_model=16,
        N=1,
        h=4,
        dropout=0.0,
        d_ff=32,
    )
    source = torch.randint(1, 12, (2, 5))
    target = torch.randint(1, 14, (2, 5))
    source_mask = (source != 0)[:, None, None, :]
    target_mask = (target != 0)[:, None, None, :] & torch.tril(
        torch.ones(5, 5, dtype=torch.bool)
    )[None, None, :, :]

    encoder_output = model.encode(source, source_mask)
    decoder_output = model.decode(encoder_output, source_mask, target, target_mask)
    scores = model.project(decoder_output)

    assert scores.shape == (2, 5, 14)
    assert len(list(model.parameters())) > 0