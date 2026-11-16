"""Wav dataset + collate for the Japanese emotional-speech fine-tuning."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# Must match finetune.splits.LABEL_TO_IDX (emotion2vec+ 9-class head order).
E2V_LABELS = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD", "SUR", "OTHER", "UNKNOWN"]
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
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=train,
        collate_fn=collate, num_workers=num_workers,
        generator=g if train else None, drop_last=train)
