from pathlib import Path

def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 350,
        "d_model": 512,
        "num_layers": 6,
        "num_heads": 8,
        "d_ff": 2048,
        "dropout": 0.1,
        "vocab_size": 16000,
        "lang_src": "en",
        "lang_tgt": "zh",
        "data_folder": "data/iwslt2017-en-zh/en-zh",
        "model_folder": "weights",
        "model_basename": "transformer_",
        "preload": None,
        "tokenizer_file": "data/tokenizer_en-zh.json",
        "experiment_name": "runs/transformer-en-zh"
    }

def get_weights_file_path(config, epoch: str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f'{model_basename}{epoch}.pt'
    return str(Path(model_folder) / model_filename)