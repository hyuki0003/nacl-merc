"""Re-partition the IEMOCAP data pickle into the paper's train/dev/test split.

Loads ``<data_dir>/<dataset>/data_<dataset>.pkl``, concatenates the existing
'train' + 'dev' + 'test' dialogue lists into one pool of 151 dialogues, and
re-partitions them using the fixed dialogue index lists below (108 train /
12 dev / 31 test). These lists define the split used in the paper and must
not be modified. By default the result is written back to the input pickle
(pass --output to write elsewhere).

Expected pickle schema
----------------------
A dict with keys 'train', 'dev', 'test', each a list of dialogue dicts:

    {
        'vid':      str,                 # dialogue/video id
        'speakers': list[str],           # per-utterance speaker ids
        'labels':   list[int],           # per-utterance emotion labels
        'audio':    np.ndarray (N, 100), # OpenSMILE features
        'visual':   np.ndarray (N, 512), # visual features
        'text':     np.ndarray (N, 768), # SBERT features
        'sentence': list[str],           # utterance transcripts
    }

Labels: iemocap (6-way) {hap: 0, sad: 1, neu: 2, ang: 3, exc: 4, fru: 5};
iemocap_4 (4-way) {hap: 0, sad: 1, neu: 2, ang: 3}.

Usage
-----
    python preprocessing/iemocap_split.py --dataset iemocap
    python preprocessing/iemocap_split.py --dataset iemocap_4 --data_dir data
"""

import argparse
import os
import pickle

import numpy as np

NUM_CLASSES = {"iemocap": 6, "iemocap_4": 4}

# Fixed dialogue indices (into the concatenated train+dev+test pool) that
# define the paper's split. Do not change.
TRAIN_DIALOGUE_IDX = [82, 14, 35, 1, 69, 13, 6, 16, 105, 45, 0, 83, 102, 67, 85, 90, 51, 104, 112, 47, 24, 42, 81, 100, 89, 122, 41, 123, 93, 128, 33, 63, 113, 34, 39, 101, 110, 31, 18, 17, 136, 142, 80, 15, 73, 68, 19, 148, 27, 86, 56, 141, 146, 87, 62, 2, 98, 28, 59, 133, 129, 50, 97, 135, 143, 96, 21, 137, 140, 132, 10, 126, 70, 7, 55, 79, 116, 130, 94, 92, 75, 49, 8, 40, 149, 134, 32, 84, 12, 60, 11, 139, 118, 77, 131, 109, 29, 72, 52, 30, 145, 91, 3, 106, 5, 120, 37, 107]

DEV_DIALOGUE_IDX = [42, 51, 56, 100, 60, 93, 37, 80, 50, 29, 84, 47]

TEST_DIALOGUE_IDX = [144, 78, 150, 53, 22, 108, 124, 74, 4, 95, 138, 20, 115, 38, 46, 57, 25, 111, 58, 23, 44, 117, 64, 147, 88, 26, 119, 121, 125, 9, 43]


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pkl(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def label_counts(dialogues, num_classes):
    """Per-class utterance label counts and their standard deviation."""
    flat = [label for d in dialogues for label in d["labels"]]
    counts = np.bincount(np.array(flat), minlength=num_classes)
    return counts, np.std(counts)


def main(args):
    print("Load dataset... Name: " + args.dataset)

    data_path = os.path.join(args.data_dir, args.dataset, "data_" + args.dataset + ".pkl")
    output_path = args.output if args.output else data_path

    dataset = load_pkl(data_path)
    total_dataset = dataset["train"] + dataset["dev"] + dataset["test"]

    new_dataset = {
        "train": [total_dataset[idx] for idx in TRAIN_DIALOGUE_IDX],
        "dev": [total_dataset[idx] for idx in DEV_DIALOGUE_IDX],
        "test": [total_dataset[idx] for idx in TEST_DIALOGUE_IDX],
    }

    save_pkl(new_dataset, output_path)
    print(f"Saved split to {output_path}")
    print(f"total number of dialogues: "
          f"{len(TRAIN_DIALOGUE_IDX) + len(DEV_DIALOGUE_IDX) + len(TEST_DIALOGUE_IDX)}")

    num_classes = NUM_CLASSES[args.dataset]
    train_counts, train_std = label_counts(new_dataset["train"], num_classes)
    dev_counts, dev_std = label_counts(new_dataset["dev"], num_classes)
    test_counts, test_std = label_counts(new_dataset["test"], num_classes)

    print(f"Train sample count: {train_counts} ({train_std})")
    print(f"Dev sample count: {dev_counts} ({dev_std})")
    print(f"Test sample count: {test_counts} ({test_std})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Re-partition the IEMOCAP data pickle into the paper's fixed split."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="iemocap",
        choices=["iemocap", "iemocap_4"],
        help="Dataset name (6-way iemocap or 4-way iemocap_4).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Root data directory containing <dataset>/data_<dataset>.pkl.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output pickle path. Default: overwrite the input pickle.",
    )

    main(parser.parse_args())
