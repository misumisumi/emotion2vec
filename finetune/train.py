"""Fine-tune emotion2vec+ (top blocks only) on JVNV + JTES + OGVC.

Usage (smoke test on this machine):
    python finetune/train.py --model-dir <e2v_model_dir> \
        --manifest-root finetune/manifests_ft --datasets jvnv ogvc_acted \
        --max-steps 5 --limit 100 --out runs/smoke

Usage (full run on the training machine):
    python finetune/train.py --config finetune/config/base.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch
from torch import nn, optim

from dataset import make_loader
from model import load_e2v, freeze_top_layers, forward_logits

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("e2v_ft")


def evaluate(model, loader, device, criterion) -> tuple[float, dict]:
    model.eval()
    total, n = 0.0, 0
    correct, per_class = 0, {}
    preds_all, labels_all = [], []
    with torch.no_grad():
        for batch in loader:
            feats = batch["feats"].to(device)
            pm = batch["padding_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = forward_logits(model, feats, pm)
            total += criterion(logits, labels).item() * labels.size(0)
            n += labels.size(0)
            preds = logits.argmax(-1)
            correct += (preds == labels).sum().item()
            preds_all.extend(preds.cpu().tolist())
            labels_all.extend(labels.cpu().tolist())
    wa = correct / max(n, 1)
    per_class_acc = {}
    for c in sorted(set(labels_all)):
        tp = sum(1 for p, l in zip(preds_all, labels_all) if l == c and p == c)
        n_c = sum(1 for l in labels_all if l == c)
        per_class_acc[c] = tp / max(n_c, 1)
    ua = sum(per_class_acc.values()) / max(len(per_class_acc), 1)
    return total / max(n, 1), {"wa": wa, "ua": ua, "n": n, "per_class": per_class_acc}


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(prog="finetune.train")
    ap.add_argument("--config", type=Path, default=None, help="YAML config (overrides defaults)")
    ap.add_argument("--model-dir", required=True, help="emotion2vec+ model dir (with model.pt + config.yaml)")
    ap.add_argument("--manifest-root", type=Path, default=here / "manifests_ft")
    ap.add_argument("--datasets", nargs="*", default=["jvnv", "jtes", "ogvc_acted", "ogvc_natural"])
    ap.add_argument("--train-blocks", type=int, default=3,
                    help="number of top transformer blocks to train (base: 3 -> blocks.5-7)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=None, help="cap optimizer steps (smoke test)")
    ap.add_argument("--limit", type=int, default=None, help="cap rows per dataset (smoke test)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--max-seconds", type=float, default=10.0, help="truncate wavs longer than this")
    ap.add_argument("--eval-every", type=int, default=10, help="optimizer steps between evals")
    ap.add_argument("--early-stop-patience", type=int, default=5,
                    help="stop when eval_loss has not improved for this many evals; 0 disables")
    ap.add_argument("--early-stop-min-delta", type=float, default=1e-4,
                    help="minimum eval_loss improvement counted as improvement")
    ap.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True,
                    help="write TensorBoard event files under <out>/tensorboard")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=here / "runs" / "run")
    args = ap.parse_args(argv)

    if args.config is not None:
        try:
            import yaml
            cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
            for k, v in cfg.items():
                key = k.replace("-", "_")
                if hasattr(args, key) and v is not None:
                    setattr(args, key, v)
        except ImportError:
            logger.warning("PyYAML not installed; ignoring --config")

    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    am = load_e2v(args.model_dir, device=args.device)
    model = am.model
    model.to(args.device)
    audit = freeze_top_layers(model, n_train_blocks=args.train_blocks)
    logger.info("param audit: n_blocks=%s train_block_indices=%s trainable=%s frozen=%s",
                audit["n_blocks"], audit["train_block_indices"],
                audit["n_trainable_params"], audit["n_frozen_params"])
    logger.info("trainable prefixes: %s",
                sorted({n.split(".")[0] + "." + n.split(".")[1] for n in audit["trainable_names"]}))
    (args.out / "param_audit.json").write_text(
        json.dumps({k: v for k, v in audit.items() if k != "trainable_names"}, indent=2),
        encoding="utf-8")

    train_rows, eval_rows = [], []
    for ds in args.datasets:
        tr = args.manifest_root / ds / "train.jsonl"
        ev = args.manifest_root / ds / "eval.jsonl"
        if not tr.is_file():
            logger.warning("skip missing %s", tr)
            continue
        if args.limit:
            with open(tr, encoding="utf-8") as fh:
                head = [next(fh) for _ in range(args.limit)]
            tmp = args.out / f"_smoke_train_{ds}.jsonl"
            tmp.write_text("".join(head), encoding="utf-8")
            tr = tmp
            with open(ev, encoding="utf-8") as fh:
                head = [next(fh) for _ in range(args.limit)]
            tmp = args.out / f"_smoke_eval_{ds}.jsonl"
            tmp.write_text("".join(head), encoding="utf-8")
            ev = tmp
        train_rows.append(tr)
        eval_rows.append(ev)

    train_loaders = [make_loader(p, args.batch_size, train=True,
                                  num_workers=args.num_workers, seed=args.seed)
                     for p in train_rows]
    eval_loaders = [make_loader(p, args.batch_size, train=False,
                                 num_workers=args.num_workers)
                     for p in eval_rows]

    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return (step + 1) / max(args.warmup_steps, 1)
        return max(0.0, 1.0 - (step - args.warmup_steps) / max(1, args.epochs * 1000))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    writer = None
    if args.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=str(args.out / "tensorboard"))
        except ImportError:
            logger.warning("tensorboard not installed; event logging disabled")

    steps = 0
    best_eval = float("inf")
    evals_since_best = 0
    stopped_early = False
    log_path = args.out / "train.log"
    log_lines: list[str] = []

    def log(msg: str) -> None:
        logger.info(msg)
        log_lines.append(msg)
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    log(f"config: datasets={args.datasets} train_blocks={args.train_blocks} "
        f"batch_size={args.batch_size} lr={args.lr} epochs={args.epochs} "
        f"max_steps={args.max_steps} limit={args.limit} device={args.device}")

    model.train()
    stop = False
    for epoch in range(args.epochs):
        if stop:
            break
        for loader in train_loaders:
            for batch in loader:
                feats = batch["feats"].to(args.device)
                pm = batch["padding_mask"].to(args.device)
                labels = batch["labels"].to(args.device)
                logits = forward_logits(model, feats, pm)
                loss = criterion(logits, labels)
                if not torch.isfinite(loss):
                    log(f"step {steps}: NON-FINITE LOSS {loss.item()}")
                    return 1
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 5.0)
                optimizer.step()
                scheduler.step()
                steps += 1
                log(f"epoch {epoch} step {steps} loss {loss.item():.6f} lr {scheduler.get_last_lr()[0]:.2e}")
                if writer is not None:
                    writer.add_scalar("train/loss", loss.item(), steps)
                    writer.add_scalar("train/lr", scheduler.get_last_lr()[0], steps)
                if steps % args.eval_every == 0:
                    model.eval()
                    ev_loss = ev_metrics = None
                    ev_loss_total, ev_n = 0.0, 0
                    correct = 0
                    with torch.no_grad():
                        for el in eval_loaders:
                            for eb in el:
                                ef = eb["feats"].to(args.device)
                                epm = eb["padding_mask"].to(args.device)
                                elabels = eb["labels"].to(args.device)
                                elogits = forward_logits(model, ef, epm)
                                ev_loss_total += criterion(elogits, elabels).item() * elabels.size(0)
                                ev_n += elabels.size(0)
                                correct += (elogits.argmax(-1) == elabels).sum().item()
                    ev_loss = ev_loss_total / max(ev_n, 1)
                    ev_wa = correct / max(ev_n, 1)
                    log(f"epoch {epoch} step {steps} eval_loss {ev_loss:.6f} eval_wa {ev_wa:.4f}")
                    if writer is not None:
                        writer.add_scalar("eval/loss", ev_loss, steps)
                        writer.add_scalar("eval/wa", ev_wa, steps)
                    if ev_loss < best_eval - args.early_stop_min_delta:
                        best_eval = ev_loss
                        evals_since_best = 0
                        ckpt_path = args.out / "best_model.pt"
                        torch.save(model.state_dict(), ckpt_path)
                        log(f"saved best model to {ckpt_path} (eval_loss={ev_loss:.6f})")
                    else:
                        evals_since_best += 1
                        if 0 < args.early_stop_patience <= evals_since_best:
                            log(f"early stopping at step {steps}: no eval_loss improvement "
                                f"for {evals_since_best} evals (best={best_eval:.6f})")
                            stopped_early = True
                            model.train()
                            stop = True
                            break
                    model.train()
                if args.max_steps is not None and steps >= args.max_steps:
                    stop = True
                    break

    if best_eval == float("inf"):
        ckpt_path = args.out / "best_model.pt"
        torch.save(model.state_dict(), ckpt_path)
        log(f"saved final model to {ckpt_path}")

    meta = {
        "model_dir": str(Path(args.model_dir).resolve()),
        "datasets": args.datasets,
        "train_blocks": args.train_blocks,
        "train_block_indices": audit["train_block_indices"],
        "n_blocks": audit["n_blocks"],
        "n_trainable_params": audit["n_trainable_params"],
        "n_frozen_params": audit["n_frozen_params"],
        "steps": steps,
        "best_eval_loss": best_eval if best_eval != float("inf") else None,
        "early_stopped": stopped_early,
        "evals_since_best": evals_since_best,
        "seed": args.seed,
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "torch_version": torch.__version__,
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                        encoding="utf-8")
    log(f"done: steps={steps} best_eval_loss={meta['best_eval_loss']} early_stopped={stopped_early}")
    if writer is not None:
        writer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
