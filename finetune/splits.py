"""Train/eval/test split builder for the Japanese emotional-speech fine-tuning.

Reads the manifests produced by finetune/build_clips.py (JVNV / JTES /
OGVC-Acted / OGVC-Natural; clips materialized under <repo>/data/clips),
OGVC-Natural), filters rows to the emotion2vec+ 9-class label space, splits
each dataset into train/eval/test, and records every exclusion (which
speaker or utterance was dropped and why) so the split is fully auditable.

Usage:
    python finetune/splits.py [--manifest-root DIR] [--out-root DIR] [--seed 42]

Outputs per dataset under <out-root>/<dataset>/:
    train.jsonl, eval.jsonl, test.jsonl
    exclusions.md  (human-readable record of every dropped row / assigned speaker)
    exclusions.json (machine-readable, same content)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

# emotion2vec+ 9-class label space in the exact order of the checkpoint's
# tokens.txt (angry, disgusted, fearful, happy, neutral, other, sad, surprised,
# <unk>). The pretrained proj head was trained with this order, so index 5 is
# OTHER, 6 is SAD and 7 is SUR: do not reorder.
# "other" and "unknown" exist in the model head but do not occur in the
# training corpora (OGVC OTH/ACC/ANT are excluded as non-emotion labels).
E2V_LABELS = ["ANG", "DIS", "FEA", "HAP", "NEU", "OTHER", "SAD", "SUR", "UNKNOWN"]
LABEL_TO_IDX = {lab: i for i, lab in enumerate(E2V_LABELS)}

# JTES speaker assignment fixed by the user (2026-11-16):
# eval = f29/f10/m10/m20, test = f35/f16/m22/m14, the remaining 92 speakers train.
JTES_EVAL_SPEAKERS = ["f10", "f29", "m10", "m20"]
JTES_TEST_SPEAKERS = ["f16", "f35", "m14", "m22"]

# Speaker-based splits for the 13-speaker OGVC-Natural are derived from the
# data with a fixed seed (see split_ogvc_natural); utterance-based splits for
# the 4-speaker datasets stratify by (speaker, emotion) so every stratum
# appears in every split.
UTTERANCE_SPLIT = {"train": 0.7, "eval": 0.15, "test": 0.15}


def load_manifest(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def majority_label(raters: list[str]) -> tuple[str | None, int]:
    """Return (majority label, vote count) for 3 raters; (None, best) when no majority."""
    top = Counter(raters).most_common(1)[0]
    return top[0], top[1]


def filter_rows(rows: list[dict], dataset: str, exclusions: list[dict]) -> list[dict]:
    """Keep rows with a canonical emotion inside the e2v 9-class space.

    Exclusions record utt_id, speaker, reason (and raters for OGVC-Natural
    rows whose majority vote did not decide a label).
    """
    kept: list[dict] = []
    for r in rows:
        canon = r.get("canonical_emotion")
        if r.get("excluded") or canon is None:
            reason = "excluded_emotion" if r.get("excluded") else "no_canonical_emotion"
            exclusions.append(_excl(r, reason, dataset))
            continue
        if canon not in LABEL_TO_IDX or canon in ("OTHER", "UNKNOWN"):
            exclusions.append(_excl(r, f"outside_e2v_space:{canon}", dataset))
            continue
        if dataset == "ogvc_natural":
            raters = r.get("raters") or []
            if len(raters) >= 2:
                lab, n = majority_label(raters)
                if n < 2:
                    exclusions.append(_excl(r, "no_majority_vote", dataset))
                    continue
        kept.append(r)
    return kept


def _excl(r: dict, reason: str, dataset: str) -> dict:
    e = {
        "dataset": dataset,
        "utt_id": r.get("utt_id"),
        "speaker": r.get("speaker"),
        "native_emotion": r.get("native_emotion"),
        "reason": reason,
    }
    if r.get("raters"):
        e["raters"] = r["raters"]
    return e


def split_utterance_stratified(rows: list[dict], seed: int) -> dict[str, list[dict]]:
    """Split by utterance, stratified by (speaker, emotion), fixed seed."""
    rng = random.Random(seed)
    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        strata[(r["speaker"], r["canonical_emotion"])].append(r)
    out: dict[str, list[dict]] = {"train": [], "eval": [], "test": []}
    for key in sorted(strata):
        members = sorted(strata[key], key=lambda r: r["utt_id"])
        rng.shuffle(members)
        n = len(members)
        n_eval = round(n * UTTERANCE_SPLIT["eval"])
        n_test = round(n * UTTERANCE_SPLIT["test"])
        n_train = n - n_eval - n_test
        out["train"].extend(members[:n_train])
        out["eval"].extend(members[n_train:n_train + n_eval])
        out["test"].extend(members[n_train + n_eval:])
    for split in out.values():
        split.sort(key=lambda r: r["utt_id"])
    return out


def split_jtes(rows: list[dict]) -> dict[str, list[dict]]:
    """Speaker-based split fixed by the user: eval/test speakers are excluded from training."""
    out: dict[str, list[dict]] = {"train": [], "eval": [], "test": []}
    for r in rows:
        if r["speaker"] in JTES_EVAL_SPEAKERS:
            out["eval"].append(r)
        elif r["speaker"] in JTES_TEST_SPEAKERS:
            out["test"].append(r)
        else:
            out["train"].append(r)
    for split in out.values():
        split.sort(key=lambda r: r["utt_id"])
    return out


def split_ogvc_natural_speakers(rows: list[dict], seed: int) -> dict[str, list[dict]]:
    """Speaker-based split over the 13 OGVC-Natural speakers.

    Speakers are assigned greedily (largest first) to keep the utterance
    ratio close to 70/15/15 while every split gets >=1 speaker.
    """
    by_spk: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_spk[r["speaker"]].append(r)
    speakers = sorted(by_spk, key=lambda s: (-len(by_spk[s]), s))
    total = len(rows)
    target = {"train": 0.7 * total, "eval": 0.15 * total, "test": 0.15 * total}
    counts = {"train": 0, "eval": 0, "test": 0}
    assign: dict[str, str] = {}
    # Deterministic tie-break: seed only shuffles the order of same-size speakers.
    rng = random.Random(seed)
    rng.shuffle(speakers := sorted(by_spk, key=lambda s: (-len(by_spk[s]), s)))
    for spk in speakers:
        n = len(by_spk[spk])
        cand = min(("train", "eval", "test"), key=lambda s: counts[s] / max(target[s], 1e-9))
        assign[spk] = cand
        counts[cand] += n
    out: dict[str, list[dict]] = {"train": [], "eval": [], "test": []}
    for r in rows:
        out[assign[r["speaker"]]].append(r)
    for split in out.values():
        split.sort(key=lambda r: r["utt_id"])
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_exclusions(out_dir: Path, dataset: str, exclusions: list[dict],
                    split_info: dict, split_rows: dict[str, list[dict]]) -> None:
    """Human- and machine-readable record: dropped rows + speaker/utterance assignment."""
    rec = {
        "dataset": dataset,
        "n_source_rows": split_info["n_source"],
        "n_excluded": len(exclusions),
        "exclusions": exclusions,
        "split_method": split_info["method"],
        "split_assignment": split_info["assignment"],
        "split_counts": {k: len(v) for k, v in split_rows.items()},
        "label_counts": {
            k: dict(sorted(Counter(r["canonical_emotion"] for r in v).items()))
            for k, v in split_rows.items()
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "exclusions.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        f"# {dataset} — split & exclusion record",
        "",
        f"- source rows: {split_info['n_source']}",
        f"- excluded rows: {len(exclusions)}",
        f"- split method: {split_info['method']}",
    ]
    lines += ["- split counts: " + ", ".join(f"{k}={len(v)}" for k, v in split_rows.items())]
    if split_info["assignment"]:
        lines.append("")
        lines.append("## Speaker assignment (which speakers were excluded from training)")
        for split, speakers in sorted(split_info["assignment"].items()):
            lines.append(f"- {split}: {', '.join(speakers)}")
    if split_info.get("dropped_utterances"):
        lines.append("")
        lines.append("## Dropped utterances")
        for d in split_info["dropped_utterances"]:
            lines.append(f"- {d['utt_id']} (speaker={d['speaker']}, emotion={d.get('native_emotion')}, reason={d['reason']})")
    if exclusions:
        reasons = Counter(e["reason"] for e in exclusions)
        lines.append("")
        lines.append("## Exclusion reasons")
        for reason, n in sorted(reasons.items()):
            lines.append(f"- {reason}: {n}")
        lines.append("")
        lines.append("All excluded rows (first 50 shown in full list below):")
        for e in exclusions[:50]:
            raters = f", raters={e['raters']}" if "raters" in e else ""
            lines.append(f"- {e['utt_id']} (speaker={e['speaker']}, native={e['native_emotion']}, reason={e['reason']}{raters})")
        if len(exclusions) > 50:
            lines.append(f"- ... and {len(exclusions) - 50} more (see exclusions.json)")
    (out_dir / "exclusions.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_dataset(name: str, manifest: Path, out_root: Path, seed: int) -> dict:
    rows = load_manifest(manifest)
    exclusions: list[dict] = []
    kept = filter_rows(rows, name, exclusions)

    if name == "jtes":
        splits = split_jtes(kept)
        method = "speaker-based (user-fixed: eval=f10,f29,m10,m20; test=f16,f35,m14,m22)"
        assignment = {
            "eval": sorted(JTES_EVAL_SPEAKERS),
            "test": sorted(JTES_TEST_SPEAKERS),
            "train": sorted({r["speaker"] for r in splits["train"]}),
        }
    elif name == "ogvc_natural":
        splits = split_ogvc_natural_speakers(kept, seed)
        method = "speaker-based (greedy ratio fit, seed={})".format(seed)
        assignment = {
            s: sorted({r["speaker"] for r in splits[s]}) for s in ("train", "eval", "test")
        }
    else:  # jvnv, ogvc_acted: only 4 speakers -> utterance-based stratified split
        splits = split_utterance_stratified(kept, seed)
        method = f"utterance-based, stratified by (speaker, emotion), seed={seed}"
        assignment = {}

    for split, srows in splits.items():
        write_jsonl(out_root / name / f"{split}.jsonl", srows)
    write_exclusions(out_root / name,
                     name,
                     exclusions,
                     {"n_source": len(rows), "method": method, "assignment": assignment},
                     splits)
    return {"dataset": name, "source": len(rows), "kept": len(kept),
            "excluded": len(exclusions),
            **{s: len(splits[s]) for s in ("train", "eval", "test")}}


DATASETS = ["jvnv", "jtes", "ogvc_acted", "ogvc_natural"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="finetune.splits")
    here = Path(__file__).resolve().parent.parent
    ap.add_argument("--manifest-root", type=Path,
                    default=here / "finetune" / "manifests_src",
                    help="directory containing {jvnv,jtes,ogvc_acted,ogvc_natural}.jsonl "
                         "(built by finetune/build_clips.py; clips live in this repo)")
    ap.add_argument("--out-root", type=Path, default=here / "finetune" / "manifests_ft")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    args = ap.parse_args(argv)

    for name in args.datasets:
        manifest = args.manifest_root / f"{name}.jsonl"
        if not manifest.is_file():
            print(f"manifest not found: {manifest}", file=sys.stderr)
            return 2
        info = build_dataset(name, manifest, args.out_root, args.seed)
        print(f"{name}: source={info['source']} kept={info['kept']} excluded={info['excluded']} "
              f"train={info['train']} eval={info['eval']} test={info['test']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
