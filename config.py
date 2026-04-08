from pathlib import Path
def get_config():
    return {
        "batch_size": 4,
        "num_epochs": 10,
        "learning_rate": 1e-4,
        "seq_len": 2200,
        "d_model": 512,
        "d_ff": 2048,
        "h": 4,
        "dropout": 0.1,
        "lang_src": "en",
        "lang_tgt": "hi",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "tokenizer_file": "tokenizers_{0}.json",
        "preload": None,
        "experiment_name": "runs/tmodel"
    }


def get_weights_file_path(config, epoch : str):
    model_folder = config["model_folder"]
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)
