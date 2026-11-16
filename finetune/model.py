"""emotion2vec+ fine-tuning model wrapper: freeze all but the top blocks.

The checkpoint is the funasr-packaged emotion2vec+ model (model.pt with
d2v_model.* keys + proj). We load it with funasr's AutoModel (which applies
the scope_map d2v_model.->none), then freeze everything except the last N
transformer blocks and the classification head (proj).

Layer choice (user decision 2026-11-16): emotion2vec+ base has depth 8
(blocks.0-7), so the trainable layers are blocks.5,6,7 (the 6th-8th layer)
plus proj. For large (depth 24), the same rule applies: the last 4 blocks
plus proj. The number of trainable blocks is configurable so the rule
"upper few layers only" transfers to any depth.
"""

from __future__ import annotations

import torch

from funasr import AutoModel

# tokens.txt order of emotion2vec+ 9-class head.
E2V_TOKENS = ["angry", "disgusted", "fearful", "happy", "neutral",
              "other", "sad", "surprised", "unknown"]


def load_e2v(model_dir: str, device: str = "cpu"):
    """Load emotion2vec+ through funasr AutoModel (init_param + scope_map come from config.yaml)."""
    am = AutoModel(model=model_dir, device=device, disable_update=True,
                   granularity="forward", extract_embedding=False)
    return am


def freeze_top_layers(model: torch.nn.Module, n_train_blocks: int = 3) -> dict:
    """Freeze all but the last n_train_blocks transformer blocks + proj; return a parameter audit dict.

    Design rationale (official emotion2vec+ fine-tune uses feature_grad_mult=0.0,
    i.e. no gradient through the conv frontend; top-layer-only training is the
    standard wav2vec2/HuBERT fine-tuning practice) is documented in AGENTS.md.
    """
    n_blocks = len(model.blocks)
    first_train = n_blocks - n_train_blocks
    trainable, frozen = [], []
    for name, p in model.named_parameters():
        trainable_idx = name.startswith("blocks.") and int(name.split(".")[1]) >= first_train
        if trainable_idx or name.startswith("proj."):
            p.requires_grad_(True)
            trainable.append(name)
        else:
            p.requires_grad_(False)
            frozen.append(name)
    audit = {
        "n_blocks": n_blocks,
        "train_block_indices": list(range(first_train, n_blocks)),
        "n_trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "n_frozen_params": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "trainable_names": trainable,
        "frozen_prefixes": sorted({n.split(".")[0] for n in frozen}),
    }
    return audit


def forward_logits(model: torch.nn.Module, feats: torch.Tensor,
                   padding_mask: torch.Tensor | None) -> torch.Tensor:
    """Mean-pooled utterance logits over non-padded frames -> (B, 9)."""
    res = model.extract_features(feats, padding_mask=padding_mask)
    x = res["x"]  # (B, T, 768), extra tokens already removed
    pm = res["padding_mask"]
    if pm is None:
        pooled = x.mean(dim=1)
    else:
        keep = (~pm).float().unsqueeze(-1)
        pooled = (x * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)
    return model.proj(pooled)
