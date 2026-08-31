# Preprocessing

Expected raw feature files, their schemas, and the split scripts.

## Expected data files

### IEMOCAP

- `data/iemocap/data_iemocap.pkl` — 6-way labels `{hap: 0, sad: 1, neu: 2, ang: 3, exc: 4, fru: 5}`
- `data/iemocap_4/data_iemocap_4.pkl` — 4-way labels `{hap: 0, sad: 1, neu: 2, ang: 3}`

Each is a dict `{'train', 'dev', 'test'}` of dialogue lists. Each dialogue is a dict:

| key | content |
| --- | --- |
| `vid` | dialogue/video id |
| `speakers` | per-utterance speaker ids |
| `labels` | per-utterance emotion labels |
| `audio` | `(N, 100)` OpenSMILE features |
| `visual` | `(N, 512)` visual features |
| `text` | `(N, 768)` SBERT features |
| `sentence` | utterance transcripts |

### MELD

- `data/meld/MELD_features_raw1.pkl` — raw per-utterance features
- `data/meld/data_meld.pkl` — same layout, with the paper's split indices

Both are 10-element lists:

| index | content |
| --- | --- |
| `[0]` | videoIDs |
| `[1]` | speakers (9-dim one-hot) |
| `[2]` | labels (7-way: `neu, sur, fea, sad, joy, dis, ang`) |
| `[3]` | text features (600-d) |
| `[4]` | audio features (300-d) |
| `[5]` | visual features (342-d) |
| `[6]` | sentences |
| `[7]` | trainIds |
| `[8]` | devIds |
| `[9]` | testIds |

Elements `[0]`–`[6]` are dicts keyed by dialogue id.

### Graph datasets

`data/<dataset>/graph_*set.pkl` files are built automatically on the first run
of the main scripts and cached; they do not need to be generated manually.

## Scripts

### `iemocap_split.py`

Re-partitions an IEMOCAP data pickle into the paper's fixed train/dev/test
split (108 / 12 / 31 dialogues, hardcoded index lists) and overwrites the
input pickle (or `--output` if given). Prints per-class label counts per split.

```bash
python preprocessing/iemocap_split.py --dataset iemocap      # 6-way
python preprocessing/iemocap_split.py --dataset iemocap_4    # 4-way
```

### `meld_split.py`

Random search for a balanced MELD train/test dialogue split. Prints the best
train/test index lists (these correspond to elements `[7]`/`[8]`/`[9]` of the
MELD data pickle); writes nothing. The shipped `data_meld.pkl` already
contains the paper's split, so this only needs to be run to regenerate a
split from scratch.

```bash
python preprocessing/meld_split.py --data data/meld/data_meld.pkl --n_iter 1000000
```
