import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2


class TranslationDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ja = self.data[idx]["ja"]
        pt = self.data[idx]["pt"]

        ja = [BOS_ID] + ja + [EOS_ID]
        pt = [BOS_ID] + pt + [EOS_ID]

        return (
            torch.tensor(ja, dtype=torch.long),
            torch.tensor(pt, dtype=torch.long)
        )


def collate_fn(batch):
    ja_batch, pt_batch = zip(*batch)

    ja_batch = pad_sequence(
        ja_batch,
        batch_first=True,
        padding_value=PAD_ID
    )

    pt_batch = pad_sequence(
        pt_batch,
        batch_first=True,
        padding_value=PAD_ID
    )

    return ja_batch, pt_batch
