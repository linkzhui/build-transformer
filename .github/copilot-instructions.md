# Workspace Guidance

- This is a learning project: explain the Transformer implementation and make implementation changes when the user explicitly asks.
- Build attention, feed-forward, encoder-block, and encoder-decoder logic with basic PyTorch layers and tensor operations.
- Do not use `nn.Transformer`, `nn.TransformerEncoder`, or `nn.MultiheadAttention` in the model implementation.
- Keep tensors batch-first: `(batch, sequence, features)`.
- Run `python -m pytest` after model or dataset changes.
- Training uses the local IWSLT 2017 English-Chinese corpus under `data/iwslt2017-en-zh/en-zh/`; see `README.md` for setup and training instructions.