"""Random search for a balanced MELD train/test dialogue split.

Runs a random search (default 1,000,000 iterations) over the MELD dialogue
indices: each iteration samples a candidate test set (280 dialogues) and a
candidate train set (1,152 dialogues) from the remainder, computes the
per-class utterance label counts of each, and keeps the best candidate found
so far. The best train/test index lists are printed; nothing is written to
disk.

The printed index lists are what goes into elements [7] (trainIds),
[8] (devIds) and [9] (testIds) of the 10-element MELD data pickle
(``data/meld/data_meld.pkl``). This search leaves the dev set empty; the
data loader handles an empty dev index list. Note that the shipped
``data_meld.pkl`` already contains the paper's split, so this script only
needs to be run to regenerate a split from scratch.

Expected pickle schema (10-element list):
    [0] videoIDs, [1] speakers (9-dim one-hot), [2] labels (7-way:
    neu, sur, fea, sad, joy, dis, ang), [3] text (600-d), [4] audio (300-d),
    [5] visual (342-d), [6] sentences, [7] trainIds, [8] devIds, [9] testIds
Elements [0]-[6] are dicts keyed by dialogue id.

Usage
-----
    python preprocessing/meld_split.py --data data/meld/data_meld.pkl --n_iter 1000000
"""

import argparse
import pickle
import random

import numpy as np

NUM_CLASSES = 7
NUM_TO_SAMPLE_TRAIN = 1152
NUM_TO_SAMPLE_TEST = 280

# Classes whose test-set counts are tracked when ranking candidates
# (0: neu, 1: sur, 4: joy, 6: ang). The selection criterion maximizes the
# count of class 0 (neutral) in the sampled test set.
BEST_IDX = [0, 1, 4, 6]


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def remove_indices(source_list, indices_to_remove):
    indices_to_remove_set = set(indices_to_remove)
    return [
        value for i, value in enumerate(source_list)
        if i not in indices_to_remove_set
    ]


def get_samples(dataset, population_indices, num_sample, num_classes):
    """Sample dialogue indices and compute their label distribution.

    Returns (sampled indices, per-class label counts, std of the counts).
    """
    sample_indices = random.sample(population_indices, num_sample)
    sample_labels = [dataset[i] for i in sample_indices if i in dataset.keys()]

    sample_flat = [label for sublist in sample_labels for label in sublist]
    sample_counts = np.bincount(np.array(sample_flat), minlength=num_classes)
    std = np.std(sample_counts)

    return sample_indices, sample_counts, std


def print_config(best_config):
    print("\n--- Final Best Configuration ---")
    if best_config:
        print(f"Best iteration: {best_config['iteration']}")
        print(f"Train indices: {best_config['train_indices']}")
        print(f"Test indices: {best_config['test_indices']}")
        print(f"Train Counts: {best_config['train_counts']} "
              f"(Sum: {np.sum(best_config['train_counts'])}), "
              f"(STD: {best_config['train_std']:.4f})")
        print(f"Test Counts: {best_config['test_counts']} "
              f"(Sum: {np.sum(best_config['test_counts'])}), "
              f"(STD: {best_config['test_std']:.4f})")
    else:
        print("No valid configuration was found.")


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("Load MELD data pickle: " + args.data)
    data = load_pkl(args.data)

    dataset = data[2]  # labels, dict keyed by dialogue id
    total_indices = [i for i in range(len(dataset.keys()))]

    print("\nSearching for a balanced train/test split "
          f"({args.n_iter} iterations)")

    best_config = None
    best_cnt = 0

    for i in range(args.n_iter):
        # 1. Sample a candidate test set
        test_indices, test_counts, test_std = get_samples(
            dataset, total_indices, NUM_TO_SAMPLE_TEST, NUM_CLASSES
        )

        # 2. Remove the test set from the pool
        remaining_after_test = remove_indices(total_indices, test_indices)

        # 3. Sample a candidate train set from the remainder
        train_indices, train_counts, train_std = get_samples(
            dataset, remaining_after_test, NUM_TO_SAMPLE_TRAIN, NUM_CLASSES
        )

        # 4. Keep the candidate if it beats the best one found so far
        best_counts = test_counts[BEST_IDX]
        if best_counts[0] > best_cnt:
            best_cnt = best_counts[0]
            best_config = {
                "iteration": i,
                "train_indices": train_indices,
                "test_indices": test_indices,
                "train_counts": train_counts,
                "test_counts": test_counts,
                "train_std": train_std,
                "test_std": test_std,
            }
            print(f"New Best Found at iteration {i}! "
                  f"Test class-0 count: {best_cnt}")
            print_config(best_config)

    print_config(best_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Random search for a balanced MELD train/test dialogue split."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/meld/data_meld.pkl",
        help="Path to the 10-element MELD data pickle.",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1000000,
        help="Number of random-search iterations.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=69,
        help="Random seed.",
    )

    main(parser.parse_args())
