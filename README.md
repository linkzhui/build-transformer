# English-Chinese Transformer

A learning implementation of an encoder-decoder Transformer for English-to-Chinese translation, built from basic PyTorch layers and tensor operations. It does not use `nn.Transformer` or `nn.MultiheadAttention`.

## Setup

Use Python 3.10 or newer. The IWSLT 2017 English-Chinese files should be extracted under `data/iwslt2017-en-zh/en-zh/`. With Conda installed, `make setup` creates or updates the local `.conda` environment.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

The training code builds a shared byte-level BPE tokenizer from the training split and uses the IWSLT 2017 TED-talk training and development files. Check the corpus license and TED usage terms before redistributing or using the data commercially.

## Test

```sh
python -m pytest
```

## Train

```sh
make train
```

Run the tests with `make test`.

The default configuration uses CUDA when available, otherwise Apple MPS when available, and falls back to CPU. Training and validation losses are logged to TensorBoard; checkpoints are written under `weights/` after each epoch. Adjust model size, sequence length, batch size, and epoch count in `src/build_transformer/config.py` for your hardware.