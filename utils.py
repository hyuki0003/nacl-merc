import argparse
import logging
import os
import pickle
import random
import sys

from datetime import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

logging.basicConfig(force=True, level=logging.INFO)

# Per-utterance input feature dimensions of the released feature files.
DATASET_EMBEDDING_DIMS = {
    "iemocap": {"a": 100, "t": 768, "v": 512},
    "iemocap_4": {"a": 100, "t": 768, "v": 512},
    "meld": {"a": 300, "t": 600, "v": 342},
}


def set_seed(seed):
    """Sets random seed everywhere."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # use deterministic algorithms
    print("Seed set", seed)


def load_config(yaml_file_path, dataset, relation_type="eam", optimizer=None, device=None):
    """Load a YAML config block into an argparse.Namespace and attach derived fields.

    Args:
        yaml_file_path: path to the YAML file (e.g. config/iemocap_4.yaml).
        dataset: top-level key inside the YAML file / dataset name.
        relation_type: edge relation set. e: interlocutor, a: intralocutor, m: intermodality.
        optimizer: optional optimizer name override (sgd | rmsprop | adam | adamw).
        device: optional device override (e.g. "cuda:1").
    """
    with open(yaml_file_path, "r") as f:
        config = yaml.safe_load(f)[dataset]

    args = argparse.Namespace(**config)
    args.dataset = dataset
    args.relation_type = relation_type
    if optimizer is not None:
        args.optimizer = optimizer
    if device is not None:
        args.device = device

    # Derived quantities used by the graph encoder.
    args.num_edges = len(args.relation_type)
    args.num_degree = len(args.relation_type)
    if args.relation_type == "eam" and args.modalities == "atv":
        args.num_degree += 1
    args.ffn_embed_dim = args.encoder_embed_dim * args.ffn_embed_scaler
    args.dataset_embedding_dims = DATASET_EMBEDDING_DIMS

    return args


def save_pkl(obj, file):
    with open(file, "wb") as f:
        pickle.dump(obj, f)


def load_pkl(file):
    with open(file, "rb") as f:
        return pickle.load(f)


def make_route(dir_path, file_name=None):
    """Create dir_path (cwd-relative) if missing; back up file_name if it already exists."""
    absolute_path = os.path.join(os.getcwd(), dir_path)

    if not os.path.exists(absolute_path):
        os.makedirs(absolute_path)

    if file_name is None:
        return

    file_path = os.path.join(absolute_path, file_name)

    if os.path.exists(file_path):
        current_datetime = dt.now().strftime("%Y-%m-%d-%H-%M-%S")
        title, extension = os.path.splitext(file_name)
        new_file_name = f"{title}-backup-{current_datetime}-{extension}"
        os.rename(file_path, os.path.join(absolute_path, new_file_name))

    open(file_path, "w").close()


def plot_and_save_loss(train_losses, val_losses, test_losses, filename):
    """Save a train/validation/test loss curve figure to `filename`."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    epochs = list(range(1, len(train_losses) + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.plot(epochs, test_losses, label="Test Loss")

    plt.title("Training vs Validation vs Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename, format="png", bbox_inches="tight", dpi=300)
    plt.close("all")  # close instead of show — avoids blocking on headless servers
    print(f"Plot saved as {filename}")


def get_logger(filepath: str, level=logging.INFO):
    logger = logging.getLogger(__name__)

    if logger.handlers:
        return logger

    logger.setLevel(level)

    fileHandler = logging.FileHandler(filepath)
    streamHandler = logging.StreamHandler(sys.stdout)

    formatter = logging.Formatter(
        fmt="[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s"
    )
    fileHandler.setFormatter(formatter)
    streamHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)

    return logger
