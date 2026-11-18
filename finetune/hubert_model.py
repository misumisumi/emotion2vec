"""HuggingFace HuBERT backbone + emotion classification head.

`models/kushinada-hubert-base` (imprt/kushinada-hubert-base) は日本語 TV 音声
62,215 時間で事前学習された SSL 特徴量抽出モデルで、分類ヘッドを持たない
(HF `HubertModel`, 12 transformer layers, hidden 768)。

emotion2vec+ (`finetune/model.py`) と同じ枠組みで FT できるように、ここで

  * 分類ヘッド `proj` (LayerNorm -> Dropout -> Linear(768, 9)) を追加し、
  * 上位 n 層 + 最終 layer_norm + `proj` のみを学習対象にする
    (`freeze_hubert_layers`)。

pooling と logits 計算は emotion2vec+ と共通の `model.forward_logits` を使う
(`extract_features` が同じ {"x", "padding_mask"} を返すため)。

入力 wav は `finetune/dataset.py` で `F.layer_norm` 済み。これは公式
Wav2Vec2FeatureExtractor (`do_normalize=true`、発話ごとの zero-mean/unit-var)
とほぼ等価 (eps と biased/unbiased の差のみ) なので、前処理は共通のままでよい。
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import HubertModel


class ClassificationHead(nn.Module):
    """LayerNorm -> Dropout -> Linear; `out_features` mirrors the e2v `proj` head."""

    def __init__(self, hidden_size: int, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size, num_labels)
        self.out_features = num_labels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.dropout(self.norm(x)))


class HubertEmotionClassifier(nn.Module):
    """HuBERT encoder + freshly initialized 9-way head (state_dict: encoder.*/proj.*).

    The `extract_features` / `proj` / `forward_logits` surface is intentionally the
    same as emotion2vec+, so `finetune.train` and `model.forward_logits` work
    unchanged for both backbones.
    """

    backbone = "hubert"

    def __init__(self, model_dir: str, num_labels: int = 9, dropout: float = 0.1):
        super().__init__()
        self.encoder = HubertModel.from_pretrained(str(model_dir), local_files_only=True)
        hidden_size = self.encoder.config.hidden_size
        self.proj = ClassificationHead(hidden_size, num_labels, dropout)
        self.num_labels = num_labels

    def extract_features(self, wavs: torch.Tensor,
                         padding_mask: torch.Tensor | None = None) -> dict:
        """Run the encoder and return pooled-frame features + frame-level padding mask.

        The conv frontend downsamples 16 kHz samples to ~20 ms frames, so the
        sample-level padding mask is converted to a frame-level one here using the
        same kernel/stride arithmetic as the HF feature extractor.
        """
        attention_mask = None if padding_mask is None else (~padding_mask).long()
        out = self.encoder(wavs, attention_mask=attention_mask)
        x = out.last_hidden_state
        if padding_mask is None:
            return {"x": x, "padding_mask": None}
        lengths = (~padding_mask).sum(dim=1)
        feat_lengths = self.encoder._get_feat_extract_output_lengths(lengths)
        frames = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        valid = frames < feat_lengths.clamp(min=1).unsqueeze(1)
        return {"x": x, "padding_mask": ~valid}

    def forward_logits(self, wavs: torch.Tensor,
                       padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        from model import forward_logits  # lazy: avoid a circular import at module load
        return forward_logits(self, wavs, padding_mask)


def freeze_hubert_layers(model: HubertEmotionClassifier, n_train_blocks: int = 3) -> dict:
    """Freeze everything but the top `n_train_blocks` layers, final LN, and `proj`.

    Same "upper few layers only" rule as emotion2vec+ (documented in AGENTS.md):
    conv frontend (feature_extractor), feature_projection, positional conv
    embedding, and `masked_spec_embed` stay frozen. The encoder's final layer_norm
    is trained because the trainable top layers change its input distribution.
    Returns an audit dict with the same keys as `model.freeze_top_layers`.
    """
    layers = model.encoder.encoder.layers
    n_blocks = len(layers)
    n_train_blocks = max(0, min(int(n_train_blocks), n_blocks))
    first_train = n_blocks - n_train_blocks

    trainable, frozen = [], []
    for name, p in model.encoder.named_parameters():
        if name.startswith("encoder.layers."):
            train = int(name.split(".")[2]) >= first_train
        elif name.startswith("encoder.layer_norm."):
            train = True
        else:  # feature_extractor / feature_projection / pos_conv_embed / masked_spec_embed
            train = False
        p.requires_grad_(train)
        (trainable if train else frozen).append(name)

    for name, p in model.proj.named_parameters():
        p.requires_grad_(True)
        trainable.append(f"proj.{name}")

    audit = {
        "backbone": "hubert",
        "hidden_size": model.encoder.config.hidden_size,
        "n_blocks": n_blocks,
        "train_block_indices": list(range(first_train, n_blocks)),
        "final_layer_norm_trainable": True,
        "head": "LayerNorm+Dropout+Linear",
        "n_trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "n_frozen_params": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "trainable_names": trainable,
        "frozen_prefixes": sorted({n.split(".")[0] for n in frozen}),
    }
    return audit


def load_hubert_classifier(model_dir: str, checkpoint: str | None = None,
                           num_labels: int = 9, dropout: float = 0.1,
                           device: str = "cpu") -> HubertEmotionClassifier:
    """Rebuild the model and optionally load a `best_model.pt`/`last_model.pt` state dict."""
    model = HubertEmotionClassifier(model_dir, num_labels=num_labels, dropout=dropout)
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    model.to(device)
    return model
