"""Wav dataset + collate for the Japanese emotional-speech fine-tuning."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, Dataset, Sampler

# Must match the checkpoint's tokens.txt order (and finetune.splits.LABEL_TO_IDX):
# angry, disgusted, fearful, happy, neutral, other, sad, surprised, <unk>.
# The pretrained proj head uses this order, so index 5=OTHER, 6=SAD, 7=SUR.
E2V_LABELS = ["ANG", "DIS", "FEA", "HAP", "NEU", "OTHER", "SAD", "SUR", "UNKNOWN"]
LABEL_TO_IDX = {lab: i for i, lab in enumerate(E2V_LABELS)}


class WavEmotionDataset(Dataset):
    """Reads 16 kHz mono wavs listed in a split manifest, normalizes with F.layer_norm."""

    def __init__(self, manifest: Path, max_seconds: float = 10.0):
        self.rows = []
        with open(manifest, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))
        self.max_samples = int(max_seconds * 16000)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        r = self.rows[idx]
        wav, sr = sf.read(r["clip_path"], dtype="float32")
        assert sr == 16000, f"{r['clip_path']}: expected 16kHz, got {sr}"
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav = F.layer_norm(torch.from_numpy(wav), (wav.shape[-1],)).numpy()
        if len(wav) > self.max_samples:
            wav = wav[: self.max_samples]
        return {"wav": torch.from_numpy(wav).float(),
                "label": LABEL_TO_IDX[r["canonical_emotion"]],
                "utt_id": r["utt_id"]}


def collate(batch: list[dict]) -> dict:
    """Pad wavs to the batch max length; padding_mask marks padded frames."""
    wavs = [b["wav"] for b in batch]
    max_len = max(w.shape[0] for w in wavs)
    feats = wavs[0].new_zeros(len(wavs), max_len)
    padding_mask = torch.zeros(len(wavs), max_len, dtype=torch.bool)
    for i, w in enumerate(wavs):
        feats[i, : w.shape[0]] = w
        padding_mask[i, w.shape[0]:] = True
    return {"feats": feats, "padding_mask": padding_mask,
            "labels": torch.tensor([b["label"] for b in batch]),
            "utt_ids": [b["utt_id"] for b in batch]}


def make_loader(manifest: Path, batch_size: int, train: bool,
                num_workers: int = 4, seed: int = 42) -> torch.utils.data.DataLoader:
    ds = WavEmotionDataset(manifest)
    g = torch.Generator().manual_seed(seed)
    # drop_last only when a full batch exists, otherwise tiny smoke manifests
    # (e.g. --limit 4 with batch_size 32) would yield an empty loader.
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=train,
        collate_fn=collate, num_workers=num_workers,
        generator=g if train else None, drop_last=train and len(ds) >= batch_size)


class BalancedSampler(Sampler):
    """Per-epoch index stream where every dataset gets the same number of rows.

    JTES alone is 81% of the training rows, so a natural pass lets it dominate the
    optimizer steps. Here each dataset contributes ``per_dataset`` rows per epoch:
    small datasets are resampled (with replacement), large ones subsampled, and the
    merged indices are shuffled. ``set_epoch`` makes the draw different each epoch.
    """

    def __init__(self, sizes: list[int], per_dataset: int, seed: int = 42):
        self.sizes = [int(s) for s in sizes]
        self.per_dataset = int(per_dataset)
        self.seed = seed
        self.epoch = 0
        self.offsets = [sum(self.sizes[:i]) for i in range(len(self.sizes))]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.per_dataset * len(self.sizes)

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        parts = []
        for off, n in zip(self.offsets, self.sizes):
            if n == 0:
                continue
            if self.per_dataset <= n:
                idx = torch.randperm(n, generator=g)[: self.per_dataset]
            else:
                idx = torch.randint(0, n, (self.per_dataset,), generator=g)
            parts.append(idx + off)
        all_idx = torch.cat(parts)
        order = torch.randperm(all_idx.numel(), generator=g)
        return iter(all_idx[order].tolist())


def make_balanced_loader(manifests: list[Path], batch_size: int,
                         rows_per_dataset: int | None = None,
                         num_workers: int = 4, seed: int = 42):
    """Concatenate datasets and draw an equal number of rows from each per epoch.

    ``rows_per_dataset`` defaults to total_rows // n_datasets, which keeps the epoch
    length (and therefore the LR schedule) the same as a natural pass while making
    every dataset contribute 1/n of the steps.
    """
    datasets = [WavEmotionDataset(p) for p in manifests]
    sizes = [len(d) for d in datasets]
    total = sum(sizes)
    if rows_per_dataset is None:
        rows_per_dataset = max(1, total // max(len(datasets), 1))
    sampler = BalancedSampler(sizes, rows_per_dataset, seed=seed)
    concat = ConcatDataset(datasets)
    loader = torch.utils.data.DataLoader(
        concat, batch_size=batch_size, sampler=sampler,
        collate_fn=collate, num_workers=num_workers,
        drop_last=len(concat) >= batch_size)
    return loader, sampler, sizes
