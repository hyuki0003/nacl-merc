import pickle

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

DEFAULT_FEATURES_PATH = 'data/meld/MELD_features_raw1.pkl'


class MELDDataset(Dataset):
    """MELD utterance-feature dataset restricted to the given dialogue ids.

    The pickle at `path` is a 10-element list; the elements used here are
    [1] speakers (9-dim one-hot), [2] labels (7-way), [3] text (600-d),
    [4] audio (300-d), [5] visual (342-d), each a dict keyed by dialogue id.
    """

    def __init__(self, path, indices=None):
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.videoAudio = {k: data[4][k] for k in indices if k in data[4]}    # audio (300-d)
        self.videoText = {k: data[3][k] for k in indices if k in data[3]}     # text (600-d)
        self.videoVisual = {k: data[5][k] for k in indices if k in data[5]}   # visual (342-d)
        self.videoLabels = {k: data[2][k] for k in indices if k in data[2]}   # labels
        self.videoSpeakers = {k: data[1][k] for k in indices if k in data[1]} # speakers

        self.keys = list(self.videoLabels.keys())
        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return (
            torch.FloatTensor(self.videoText[vid]),
            torch.FloatTensor(self.videoVisual[vid]),
            torch.FloatTensor(self.videoAudio[vid]),
            torch.FloatTensor(self.videoSpeakers[vid]),
            torch.FloatTensor([1] * len(self.videoLabels[vid])),  # mask
            torch.LongTensor(self.videoLabels[vid]),
            vid,
        )

    def __len__(self):
        return self.len


def get_MELD_loaders(batch_size, data_path, num_workers=0, pin_memory=False,
                     features_path=DEFAULT_FEATURES_PATH):
    """Build MELD train/valid/test/all DataLoaders.

    `data_path` provides the split indices (elements [7]/[8]/[9]:
    trainIds/devIds/testIds); `features_path` provides the utterance
    features loaded by each MELDDataset. Returns (train_loader,
    valid_loader, test_loader, all_loader); valid_loader is None when the
    dev index list is empty.
    """
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    train_idx = data[7]
    valid_idx = data[8]
    test_idx = data[9]
    all_idx = list(data[2].keys())  # every dialogue id (from the label dict)

    train_dataset = MELDDataset(features_path, train_idx)
    if len(valid_idx) > 0:
        valid_dataset = MELDDataset(features_path, valid_idx)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_fn,
                                  num_workers=num_workers, pin_memory=pin_memory)
    else:
        valid_loader = None
    test_dataset = MELDDataset(features_path, test_idx)
    all_dataset = MELDDataset(features_path, all_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn,
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn,
                             num_workers=num_workers, pin_memory=pin_memory)
    all_loader = DataLoader(all_dataset, batch_size=batch_size, collate_fn=collate_fn,
                            num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader, all_loader


def collate_fn(batch):
    """Pad a batch of variable-length dialogues into tensors."""
    videoText, videoVisual, videoAudio, videoSpeakers, mask, videoLabels, vid = zip(*batch)

    return {
        "videoAudio": pad_sequence(videoAudio, batch_first=True),
        "videoText": pad_sequence(videoText, batch_first=True),
        "videoVisual": pad_sequence(videoVisual, batch_first=True),
        "videoLabels": pad_sequence(videoLabels, batch_first=True, padding_value=-1),
        "videoSpeakers": pad_sequence(videoSpeakers, batch_first=True),
        "mask": pad_sequence(mask, batch_first=True),
        "vid": vid,
    }
