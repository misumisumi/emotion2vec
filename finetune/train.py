"""Fine-tune a speech-SSL backbone (top blocks only) on JVNV + JTES + OGVC.

Backbones: emotion2vec+ (funasr, pretrained 9-class proj head) and HuggingFace
HuBERT (e.g. kushinada-hubert-base, freshly added classification head); pick one
with --arch / the `arch` key of the YAML config.

Usage (smoke test on this machine):
    python finetune/train.py --model-dir <e2v_model_dir> \
        --manifest-root finetune/manifests_ft --datasets jvnv ogvc_acted \
        --max-steps 5 --limit 100 --out outputs/smoke

Usage (full run on the training machine):
    python finetune/train.py --config finetune/config/base.yaml
    python finetune/train.py --config finetune/config/hubert.yaml

Outputs default to outputs/<YYYYMMDD_HHMMSS>/ (see resolve_output_dir).
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
import time
from pathlib import Path

import torch
from torch import nn, optim

from dataset import make_loader, make_balanced_loader, E2V_LABELS
from model import build_model, forward_logits

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("e2v_ft")


def read_head(path: Path, n: int) -> list[str]:
    """First n lines of a manifest, tolerating n > number of lines (--limit smoke runs)."""
    with open(path, encoding="utf-8") as fh:
        return list(itertools.islice(fh, n))


def resolve_output_dir(repo_root: Path) -> Path:
    """Default output dir: outputs/<YYYYMMDD_HHMMSS> (with a suffix if it already exists)."""
    root = repo_root / "outputs"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    candidate = root / stamp
    if not candidate.exists() or not any(candidate.iterdir()):
        return candidate
    for i in range(2, 100):
        candidate = root / f"{stamp}-{i}"
        if not candidate.exists() or not any(candidate.iterdir()):
            return candidate
    return root / f"{stamp}-{time.time_ns()}"


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(prog="finetune.train")
    ap.add_argument("--config", type=Path, default=None,
                    help="YAML config; keys also given on the CLI keep their CLI value")
    ap.add_argument("--arch", default="auto",
                    choices=["auto", "emotion2vec", "hubert"],
                    help="backbone: emotion2vec (funasr, pretrained 9-class head), "
                         "hubert (HuggingFace HuBERT + freshly initialized head), or "
                         "auto = infer from --model-dir (model.pt -> emotion2vec, "
                         "config.json -> hubert)")
    ap.add_argument("--model-dir", default=None,
                    help="backbone dir: emotion2vec+ (model.pt + config.yaml) or a "
                         "HuggingFace HuBERT dir (config.json + pytorch_model.bin); "
                         "required via CLI or config")
    ap.add_argument("--manifest-root", type=Path, default=here / "manifests_ft")
    ap.add_argument("--datasets", nargs="*",
                    default=["jvnv", "jtes", "ogvc_acted", "ogvc_natural", "hcudb1"])
    ap.add_argument("--train-blocks", type=int, default=3,
                    help="number of top transformer blocks to train (emotion2vec base: "
                         "3 -> blocks.5-7; HuBERT base: 3 -> encoder.layers.9-11)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=None, help="cap optimizer steps (smoke test)")
    ap.add_argument("--limit", type=int, default=None, help="cap rows per dataset (smoke test)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--head-lr", type=float, default=None,
                    help="separate LR for the classification head (default: same as --lr)")
    ap.add_argument("--head-dropout", type=float, default=0.1,
                    help="dropout in the HuBERT classification head (emotion2vec uses "
                         "the checkpoint's own proj head; ignored there)")
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--max-seconds", type=float, default=10.0, help="truncate wavs longer than this")
    ap.add_argument("--eval-every", type=int, default=0,
                    help="0 = evaluate once at the end of every epoch; N>0 = every N optimizer steps")
    ap.add_argument("--eval-metric", default="macro_loss",
                    choices=["macro_loss", "macro_wa", "pooled_loss", "pooled_wa"],
                    help="metric used for best-checkpoint selection and early stopping "
                         "(macro_* averages the datasets equally, pooled_* averages the rows)")
    ap.add_argument("--early-stop-patience", type=int, default=0,
                    help="stop after this many evals without improvement; 0 disables "
                         "(0 runs all --epochs and keeps the best checkpoint)")
    ap.add_argument("--early-stop-min-delta", type=float, default=1e-3,
                    help="minimum metric improvement counted as improvement")
    ap.add_argument("--dataset-sampling", default="balance",
                    choices=["balance", "natural"],
                    help="balance = each dataset contributes 1/N of the epoch steps "
                         "(JTES alone is 81%% of the rows); natural = one pass per dataset")
    ap.add_argument("--rows-per-dataset", type=int, default=None,
                    help="rows each dataset contributes per epoch when sampling=balance "
                         "(default: total rows / n_datasets, keeps the epoch length)")
    ap.add_argument("--save-last", action=argparse.BooleanOptionalAction, default=True,
                    help="also overwrite last_model.pt at every epoch end")
    ap.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True,
                    help="write TensorBoard event files under <out>/tensorboard")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None,
                    help="output directory (default: outputs/<YYYYMMDD_HHMMSS>)")
    args = ap.parse_args(argv)

    # Track which flags were actually typed so the YAML only fills the gaps.
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    option_actions = getattr(ap, "_option_string_actions", {})
    cli_dests = {option_actions[t.split("=", 1)[0]].dest
                 for t in raw_argv if t.split("=", 1)[0] in option_actions}

    if args.config is not None:
        try:
            import yaml
            cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
            for k, v in cfg.items():
                key = k.replace("-", "_")
                if hasattr(args, key) and v is not None and key not in cli_dests:
                    setattr(args, key, v)
        except ImportError:
            logger.warning("PyYAML not installed; ignoring --config")

    # Values coming from the YAML are plain str/int; restore the types argparse
    # would have produced so Path methods (.mkdir/.is_file) keep working.
    for key in ("config", "manifest_root", "out"):
        val = getattr(args, key)
        if val is not None:
            setattr(args, key, Path(val))
    for key in ("train_blocks", "epochs", "max_steps", "limit", "batch_size",
                "warmup_steps", "eval_every", "early_stop_patience", "num_workers", "seed"):
        val = getattr(args, key)
        if val is not None:
            setattr(args, key, int(val))
    for key in ("lr", "head_lr", "head_dropout", "weight_decay", "max_seconds",
                "early_stop_min_delta"):
        val = getattr(args, key)
        if val is not None:
            setattr(args, key, float(val))

    if not args.model_dir:
        ap.error("--model-dir is required (pass it on the CLI or set model_dir in --config)")
    if args.arch == "auto":
        model_path = Path(args.model_dir)
        if (model_path / "model.pt").is_file():
            args.arch = "emotion2vec"
        elif (model_path / "config.json").is_file():
            args.arch = "hubert"
        else:
            ap.error(f"cannot infer --arch from {model_path}: expected model.pt "
                     "(emotion2vec) or config.json (HuBERT); pass --arch explicitly")
        logger.info("arch auto-detected from model dir: %s", args.arch)
    if args.arch not in ("emotion2vec", "hubert"):
        ap.error(f"unknown arch {args.arch!r} (expected 'emotion2vec', 'hubert' or 'auto')")

    if args.out is None:
        args.out = resolve_output_dir(here.parent)
    logger.info("output dir: %s", args.out)

    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    model, audit = build_model(
        args.arch, args.model_dir, n_train_blocks=args.train_blocks,
        num_labels=len(E2V_LABELS), head_dropout=args.head_dropout,
        device=args.device)
    logger.info("arch=%s model_dir=%s", args.arch, args.model_dir)
    logger.info("param audit: n_blocks=%s train_block_indices=%s trainable=%s frozen=%s",
                audit["n_blocks"], audit["train_block_indices"],
                audit["n_trainable_params"], audit["n_frozen_params"])
    logger.info("trainable prefixes: %s",
                sorted({n.split(".")[0] + "." + n.split(".")[1] for n in audit["trainable_names"]}))
    (args.out / "param_audit.json").write_text(
        json.dumps({k: v for k, v in audit.items() if k != "trainable_names"}, indent=2),
        encoding="utf-8")

    train_specs: list[tuple[str, Path]] = []
    eval_specs: list[tuple[str, Path]] = []
    for ds in args.datasets:
        tr = args.manifest_root / ds / "train.jsonl"
        ev = args.manifest_root / ds / "eval.jsonl"
        if not tr.is_file() and not ev.is_file():
            logger.warning("skip %s: neither %s nor %s exists", ds, tr, ev)
            continue
        if tr.is_file():
            if args.limit:
                tmp = args.out / f"_smoke_train_{ds}.jsonl"
                tmp.write_text("".join(read_head(tr, args.limit)), encoding="utf-8")
                tr = tmp
            train_specs.append((ds, tr))
        else:
            # Evaluation-only corpus (e.g. hcudb1): all rows go to eval, the
            # dataset never contributes a training batch.
            logger.info("%s is evaluation-only (missing %s): excluded from training", ds, tr)
        if ev.is_file():
            if args.limit:
                tmp = args.out / f"_smoke_eval_{ds}.jsonl"
                tmp.write_text("".join(read_head(ev, args.limit)), encoding="utf-8")
                ev = tmp
            eval_specs.append((ds, ev))
        else:
            logger.warning("skip missing %s", ev)
    logger.info("train datasets: %s | eval datasets: %s",
                [n for n, _ in train_specs] or "none", [n for n, _ in eval_specs] or "none")
    if not train_specs:
        logger.error("no trainable dataset under %s (eval-only corpora cannot train)",
                     args.manifest_root)
        return 2

    balanced_sampler = None
    if args.dataset_sampling == "balance":
        train_loader, balanced_sampler, train_sizes = make_balanced_loader(
            [p for _, p in train_specs], args.batch_size,
            rows_per_dataset=args.rows_per_dataset,
            num_workers=args.num_workers, seed=args.seed)
        train_loaders: list[tuple[str, torch.utils.data.DataLoader]] = [("all", train_loader)]
        steps_per_epoch = len(train_loader)
        train_rows_per_dataset = {name: balanced_sampler.per_dataset for name, _ in train_specs}
        available_rows_per_dataset = dict(zip((n for n, _ in train_specs), train_sizes))
    else:
        train_loaders = [(name, make_loader(p, args.batch_size, train=True,
                                            num_workers=args.num_workers, seed=args.seed))
                         for name, p in train_specs]
        steps_per_epoch = sum(len(loader) for _, loader in train_loaders)
        train_rows_per_dataset = {name: len(loader.dataset)
                                  for name, loader in train_loaders}
        available_rows_per_dataset = dict(train_rows_per_dataset)
    eval_loaders = [(name, make_loader(p, args.batch_size, train=False,
                                       num_workers=args.num_workers))
                    for name, p in eval_specs]
    if steps_per_epoch == 0:
        logger.error("no training batches: check --limit/--datasets (batch_size=%s)", args.batch_size)
        return 2
    # Linear LR decay over the real number of steps (was a hard-coded epochs*1000).
    planned_steps = steps_per_epoch * max(args.epochs, 1)
    if args.max_steps is not None:
        planned_steps = min(planned_steps, args.max_steps)
    decay_steps = max(planned_steps - args.warmup_steps, 1)

    n_out = getattr(model.proj, "out_features", None)
    if n_out is not None and n_out != len(E2V_LABELS):
        logger.error("proj head has %s outputs but %s labels are mapped", n_out, len(E2V_LABELS))
        return 2

    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and n.startswith("proj.")]
    backbone_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and not n.startswith("proj.")]
    if args.head_lr is not None and head_params:
        # A freshly initialized HuBERT head usually needs a larger LR than the
        # pretrained top blocks; emotion2vec's proj head is pretrained, so the
        # default (single LR) stays unchanged for it.
        param_groups = [{"params": backbone_params, "lr": args.lr},
                        {"params": head_params, "lr": args.head_lr}]
    else:
        param_groups = [{"params": params, "lr": args.lr}]
    optimizer = optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)

    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return (step + 1) / max(args.warmup_steps, 1)
        return max(0.0, 1.0 - (step - args.warmup_steps) / decay_steps)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    writer = None
    if args.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=str(args.out / "tensorboard"))
        except ImportError:
            logger.warning("tensorboard not installed; event logging disabled")

    steps = 0
    best_metric: float | None = None
    best_step: int | None = None
    best_epoch: int | None = None
    evals_since_best = 0
    stopped_early = False
    stop = False
    epochs_completed = 0
    eval_history: list[dict] = []
    log_path = args.out / "train.log"
    log_lines: list[str] = []

    def log(msg: str) -> None:
        logger.info(msg)
        log_lines.append(msg)
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    def run_eval() -> tuple[dict, dict]:
        """Eval over all datasets; return (aggregate metrics, per-dataset breakdown).

        macro_* average the datasets with equal weight (JTES has 44% of the eval rows
        and 81% of the train rows, so pooled metrics are dominated by it);
        pooled_* average the rows.
        """
        model.eval()
        pooled_loss, pooled_n, pooled_correct = 0.0, 0, 0
        per_dataset: dict[str, dict] = {}
        with torch.no_grad():
            for name, loader in eval_loaders:
                ds_loss, ds_n, ds_correct = 0.0, 0, 0
                for eb in loader:
                    ef = eb["feats"].to(args.device)
                    epm = eb["padding_mask"].to(args.device)
                    elabels = eb["labels"].to(args.device)
                    elogits = forward_logits(model, ef, epm)
                    ds_loss += criterion(elogits, elabels).item() * elabels.size(0)
                    ds_n += elabels.size(0)
                    ds_correct += (elogits.argmax(-1) == elabels).sum().item()
                per_dataset[name] = {"loss": ds_loss / max(ds_n, 1),
                                     "wa": ds_correct / max(ds_n, 1),
                                     "n": ds_n}
                pooled_loss += ds_loss
                pooled_n += ds_n
                pooled_correct += ds_correct
        model.train()
        n_ds = max(len(per_dataset), 1)
        metrics = {
            "pooled_loss": pooled_loss / max(pooled_n, 1),
            "pooled_wa": pooled_correct / max(pooled_n, 1),
            "macro_loss": sum(m["loss"] for m in per_dataset.values()) / n_ds,
            "macro_wa": sum(m["wa"] for m in per_dataset.values()) / n_ds,
        }
        return metrics, per_dataset

    def handle_eval(epoch: int) -> bool:
        """Log/save on an eval boundary; return True when early stopping fires."""
        nonlocal best_metric, best_step, best_epoch, evals_since_best, stopped_early
        metrics, per_dataset = run_eval()
        current = metrics[args.eval_metric]
        detail = " ".join(f"{n}:{m['loss']:.4f}/{m['wa']:.4f}"
                          for n, m in per_dataset.items())
        log(f"epoch {epoch} step {steps} eval_{args.eval_metric} {current:.6f} "
            f"macro_wa {metrics['macro_wa']:.4f} pooled_loss {metrics['pooled_loss']:.6f} "
            f"pooled_wa {metrics['pooled_wa']:.4f} | {detail}")
        eval_history.append({"epoch": epoch, "step": steps, **metrics,
                             "per_dataset": per_dataset})
        if writer is not None:
            for key, value in metrics.items():
                writer.add_scalar(f"eval/{key}", value, steps)
            for name, m in per_dataset.items():
                writer.add_scalar(f"eval/{name}/loss", m["loss"], steps)
                writer.add_scalar(f"eval/{name}/wa", m["wa"], steps)

        # lower is better for *_loss, higher for *_wa
        sign = -1.0 if args.eval_metric.endswith("wa") else 1.0
        if best_metric is None or sign * current < sign * best_metric - args.early_stop_min_delta:
            best_metric, best_step, best_epoch = current, steps, epoch
            evals_since_best = 0
            ckpt_path = args.out / "best_model.pt"
            torch.save(model.state_dict(), ckpt_path)
            (args.out / "best_meta.json").write_text(json.dumps(
                {"metric": args.eval_metric, "value": current, "epoch": epoch,
                 "step": steps, "metrics": metrics, "per_dataset": per_dataset},
                indent=2, ensure_ascii=False), encoding="utf-8")
            log(f"saved best model to {ckpt_path} "
                f"({args.eval_metric}={current:.6f}, epoch {epoch}, step {steps})")
            return False
        if steps <= args.warmup_steps:
            log(f"step {steps} is inside warmup ({args.warmup_steps}); "
                f"early-stop patience not counted")
            return False
        evals_since_best += 1
        if 0 < args.early_stop_patience <= evals_since_best:
            log(f"early stopping at step {steps}: {args.eval_metric} did not improve for "
                f"{evals_since_best} evals (best={best_metric:.6f} at epoch {best_epoch}, "
                f"step {best_step})")
            stopped_early = True
            return True
        return False

    log(f"config: arch={args.arch} model_dir={args.model_dir} "
        f"datasets={args.datasets} train_blocks={args.train_blocks} "
        f"batch_size={args.batch_size} lr={args.lr} head_lr={args.head_lr} "
        f"epochs={args.epochs} "
        f"steps_per_epoch={steps_per_epoch} planned_steps={planned_steps} "
        f"sampling={args.dataset_sampling} rows_per_dataset={train_rows_per_dataset} "
        f"eval_every={args.eval_every} eval_metric={args.eval_metric} "
        f"early_stop_patience={args.early_stop_patience} "
        f"early_stop_min_delta={args.early_stop_min_delta} "
        f"max_steps={args.max_steps} limit={args.limit} device={args.device}")
    log("train rows available per dataset: "
        + ", ".join(f"{n}={available_rows_per_dataset[n]}" for n, _ in train_specs))
    label_note = ("must match the checkpoint tokens.txt order"
                  if args.arch == "emotion2vec"
                  else "freshly initialized head; order fixed by finetune/dataset.py")
    log(f"head index -> label ({label_note}): "
        + ", ".join(f"{i}={lab}" for i, lab in enumerate(E2V_LABELS)))

    model.train()
    for epoch in range(1, args.epochs + 1):
        if stop:
            break
        if balanced_sampler is not None:
            balanced_sampler.set_epoch(epoch - 1)  # keep the 0-based sampling sequence
        for name, loader in train_loaders:
            if stop:
                break
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
                lrs = scheduler.get_last_lr()
                lr_msg = f"{lrs[0]:.2e}"
                if len(lrs) > 1:
                    lr_msg += f"/head {lrs[-1]:.2e}"
                log(f"epoch {epoch} step {steps} loss {loss.item():.6f} lr {lr_msg}")
                if writer is not None:
                    writer.add_scalar("train/loss", loss.item(), steps)
                    writer.add_scalar("train/lr", scheduler.get_last_lr()[0], steps)
                    if len(lrs) > 1:
                        writer.add_scalar("train/lr_head", lrs[-1], steps)
                if args.eval_every > 0 and steps % args.eval_every == 0:
                    if handle_eval(epoch):
                        stop = True
                        break
                if args.max_steps is not None and steps >= args.max_steps:
                    stop = True
                    break
        if stop:
            break  # stopped mid-epoch: skip the epoch-end eval
        if args.eval_every == 0 and handle_eval(epoch):
            stop = True
            break
        epochs_completed = epoch
        if args.save_last:
            torch.save(model.state_dict(), args.out / "last_model.pt")

    if best_metric is None:
        ckpt_path = args.out / "best_model.pt"
        torch.save(model.state_dict(), ckpt_path)
        log(f"no eval ran; saved final model to {ckpt_path}")

    meta = {
        "arch": args.arch,
        "model_dir": str(Path(args.model_dir).resolve()),
        "datasets": args.datasets,
        "train_blocks": args.train_blocks,
        "train_block_indices": audit["train_block_indices"],
        "n_blocks": audit["n_blocks"],
        "n_trainable_params": audit["n_trainable_params"],
        "n_frozen_params": audit["n_frozen_params"],
        "backbone_audit": {k: v for k, v in audit.items() if k != "trainable_names"},
        "head_lr": args.head_lr,
        "head_dropout": args.head_dropout,
        "steps": steps,
        "steps_per_epoch": steps_per_epoch,
        "planned_steps": planned_steps,
        "epochs_completed": epochs_completed,
        "dataset_sampling": args.dataset_sampling,
        "train_rows_available": available_rows_per_dataset,
        "train_rows_per_epoch": train_rows_per_dataset,
        "eval_every": args.eval_every,
        "eval_metric": args.eval_metric,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_min_delta": args.early_stop_min_delta,
        "warmup_steps": args.warmup_steps,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "best_step": best_step,
        "eval_history": eval_history,
        "label_map": {str(i): lab for i, lab in enumerate(E2V_LABELS)},
        "early_stopped": stopped_early,
        "evals_since_best": evals_since_best,
        "seed": args.seed,
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "torch_version": torch.__version__,
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                        encoding="utf-8")
    log(f"done: steps={steps} best_{args.eval_metric}={best_metric} "
        f"(epoch {best_epoch}, step {best_step}) early_stopped={stopped_early}")
    if writer is not None:
        writer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
