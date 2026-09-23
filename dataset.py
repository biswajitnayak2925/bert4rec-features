
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import config as cfg


def _left_pad(seq, max_len, pad_value=0):
    seq = seq[-max_len:]
    return [pad_value] * (max_len - len(seq)) + list(seq)


class TrainDataset(Dataset):
    def __init__(self, split, num_items, mask_token, pad_token, max_seq_len=None, mask_prob=None):
        self.uids = list(split.keys())
        self.split = split
        self.num_items = num_items
        self.mask_token = mask_token
        self.pad_token = pad_token
        self.max_seq_len = max_seq_len if max_seq_len is not None else cfg.MAX_SEQ_LEN
        self.mask_prob = mask_prob if mask_prob is not None else cfg.MASK_PROB

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        uid = self.uids[idx]
        seq = _left_pad(self.split[uid]["train"], self.max_seq_len, self.pad_token)
        input_ids = list(seq)
        labels = [-100] * self.max_seq_len
        any_masked = False
        for i in range(self.max_seq_len):
            if input_ids[i] == self.pad_token:
                continue
            if random.random() < self.mask_prob:
                labels[i] = input_ids[i]
                input_ids[i] = self.mask_token
                any_masked = True
        if not any_masked:
            real_positions = [i for i in range(self.max_seq_len) if seq[i] != self.pad_token]
            if real_positions:
                i = random.choice(real_positions)
                labels[i] = input_ids[i]
                input_ids[i] = self.mask_token
        return {"input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long)}


class EvalDataset(Dataset):
    def __init__(self, split, num_items, mask_token, pad_token, stage="val",
                 max_seq_len=None, num_negatives=None, seed=None):
        assert stage in ("val", "test")
        self.uids = list(split.keys())
        self.split = split
        self.num_items = num_items
        self.mask_token = mask_token
        self.pad_token = pad_token
        self.stage = stage
        self.max_seq_len = max_seq_len if max_seq_len is not None else cfg.MAX_SEQ_LEN
        self.num_negatives = num_negatives if num_negatives is not None else cfg.NUM_EVAL_NEGATIVES
        self.rng = np.random.RandomState(cfg.SEEDS[0] if seed is None else seed)
        self.interacted = {}
        for uid, d in split.items():
            self.interacted[uid] = set(d["train"]) | {d["val"], d["test"]}
        self.all_items = np.arange(1, num_items + 1)

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        uid = self.uids[idx]
        d = self.split[uid]
        if self.stage == "val":
            history, target = d["train"], d["val"]
        else:
            history, target = d["train"] + [d["val"]], d["test"]
        seq = history[-(self.max_seq_len - 1):] + [self.mask_token]
        input_ids = _left_pad(seq, self.max_seq_len, self.pad_token)
        excluded = self.interacted[uid]
        available = self.num_items - len(excluded)
        num_negatives = min(self.num_negatives, max(0, available))
        negatives = []
        while len(negatives) < num_negatives:
            cand = self.rng.randint(1, self.num_items + 1)
            if cand not in excluded and cand not in negatives:
                negatives.append(cand)
        return {"input_ids": torch.tensor(input_ids, dtype=torch.long),
                "candidates": torch.tensor([target] + negatives, dtype=torch.long)}
