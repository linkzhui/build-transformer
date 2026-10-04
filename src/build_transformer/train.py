import warnings
from itertools import zip_longest
from pathlib import Path
from xml.etree import ElementTree

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from .config import get_config, get_weights_file_path
from .dataset import BilingualDataset
from .model import build_transformer

def get_or_build_tokenizer(config, ds):
    tokenizer_path = Path(config['tokenizer_file'])
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    if not tokenizer_path.exists():
        tokenizer = Tokenizer(BPE(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()
        trainer = BpeTrainer(
            vocab_size=config['vocab_size'],
            min_frequency=2,
            special_tokens=['[UNK]', '[PAD]', '[SOS]', '[EOS]'],
            initial_alphabet=ByteLevel.alphabet(),
        )
        tokenizer.train_from_iterator(_all_bilingual_sentences(ds, config), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def _all_bilingual_sentences(ds, config):
    for item in ds:
        yield item['translation'][config['lang_src']]
        yield item['translation'][config['lang_tgt']]

def _read_tagged_sentences(path):
    with path.open(encoding='utf-8') as source_file:
        for line in source_file:
            line = line.strip()
            if line.startswith('<seg'):
                segment = ElementTree.fromstring(line)
                text = ''.join(segment.itertext()).strip()
                if text:
                    yield text
            elif line and not line.startswith('<'):
                yield line

def _read_xml_sentences(path):
    root = ElementTree.parse(path).getroot()
    return [
        ''.join(segment.itertext()).strip()
        for segment in root.iter('seg')
    ]

def _make_parallel_data(source_sentences, target_sentences, lang_src, lang_tgt):
    examples = []
    for index, (source, target) in enumerate(
        zip_longest(source_sentences, target_sentences)
    ):
        if source is None or target is None:
            raise ValueError(f'IWSLT source/target line count differs at line {index + 1}')
        examples.append({'translation': {lang_src: source, lang_tgt: target}})
    return examples

def get_ds(config):
    data_dir = Path(config['data_folder'])
    lang_src = config['lang_src']
    lang_tgt = config['lang_tgt']
    pair = f'{lang_src}-{lang_tgt}'

    train_ds_raw = _make_parallel_data(
        _read_tagged_sentences(data_dir / f'train.tags.{pair}.{lang_src}'),
        _read_tagged_sentences(data_dir / f'train.tags.{pair}.{lang_tgt}'),
        lang_src,
        lang_tgt,
    )
    val_ds_raw = _make_parallel_data(
        _read_xml_sentences(data_dir / f'IWSLT17.TED.dev2010.{pair}.{lang_src}.xml'),
        _read_xml_sentences(data_dir / f'IWSLT17.TED.dev2010.{pair}.{lang_tgt}.xml'),
        lang_src,
        lang_tgt,
    )

    # Train one subword vocabulary on both languages, using training data only.
    tokenizer = get_or_build_tokenizer(config, train_ds_raw)

    train_ds = BilingualDataset(train_ds_raw, tokenizer, tokenizer, lang_src, lang_tgt, config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer, tokenizer, lang_src, lang_tgt, config['seq_len'])

    train_dataload = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataload = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False)

    return train_dataload, val_dataload, tokenizer, tokenizer

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(
        vocab_src_len,
        vocab_tgt_len,
        config['seq_len'],
        config['seq_len'],
        d_model=config['d_model'],
        N=config['num_layers'],
        h=config['num_heads'],
        dropout=config['dropout'],
        d_ff=config['d_ff'],
    )
    return model

def train_model(config):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Using device {device}')

    model_folder = Path(config['model_folder'])
    model_folder.mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    writer = SummaryWriter(config["experiment_name"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    start_epoch = 0
    global_step = 0
    if config["preload"]:
        model_filename = get_weights_file_path(config, config["preload"])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename, map_location=device)
        model.load_state_dict(state['model_state_dict'])
        start_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state["global_step"]

    pad_id = tokenizer_tgt.token_to_id('[PAD]')
    if pad_id is None:
        raise ValueError('Target tokenizer is missing [PAD]')
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=0.1)

    for epoch in range(start_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')
        total_train_loss = 0.0
        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(device) # (B, seq_len)
            decoder_input = batch["decoder_input"].to(device) # (B, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch["decoder_mask"].to(device) # (B, 1, seq_len, seq_len)
            label = batch['label'].to(device) # (B, seq_len)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)

            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            writer.add_scalar("train loss", loss.item(), global_step)
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                encoder_input = batch['encoder_input'].to(device)
                decoder_input = batch['decoder_input'].to(device)
                encoder_mask = batch['encoder_mask'].to(device)
                decoder_mask = batch['decoder_mask'].to(device)
                label = batch['label'].to(device)

                encoder_output = model.encode(encoder_input, encoder_mask)
                decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
                logits = model.project(decoder_output)
                total_val_loss += loss_fn(
                    logits.view(-1, tokenizer_tgt.get_vocab_size()),
                    label.view(-1),
                ).item()

        average_train_loss = total_train_loss / max(len(train_dataloader), 1)
        average_val_loss = total_val_loss / max(len(val_dataloader), 1)
        writer.add_scalar('epoch train loss', average_train_loss, epoch)
        writer.add_scalar('validation loss', average_val_loss, epoch)
        writer.flush()
        print(f'epoch {epoch:02d}: train={average_train_loss:.4f} val={average_val_loss:.4f}')

        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step,
            },
            model_filename,
        )

    writer.close()

def main():
    config = get_config()
    train_model(config)

if __name__ == "__main__":
    main()
