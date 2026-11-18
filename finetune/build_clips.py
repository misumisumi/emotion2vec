"""Materialize 16 kHz/16-bit/mono clips inside THIS repository (emotion2vec/data/clips).

Reads the SER-model-comparison source manifests (they only supply corpus
paths and labels), converts every utterance from its original corpus file
with ffmpeg, and writes new manifests whose clip_path points at the fresh
clips under <repo>/data/clips/<dataset>/. The SER clips are never read.

JVNV's manifest source_path points at an unzipped tree that no longer
exists on this machine; the original jvnv_ver1.zip archive contains the
same wav files, so JVNV rows are materialized straight from the zip
(ffmpeg reads zip members via the "zip:" protocol is not portable, so we
extract members to a temp dir first).

HCUDB1 has no SER-model-comparison manifest: its rows are built directly
from the corpus (HCUDB1/wav/<speaker>/*.wav) and the evaluator summary
(HCUDB1/eval/collected_result(all).csv, CP932), so ``--manifest-root`` is
not needed for it.

Usage:
    python finetune/build_clips.py \
        --manifest-root ../SER-model-comparison/manifests \
        --jvnv-zip /localtmp/sumiharu.kobayashi.r4/datasets/jvnv_ver1.zip \
        [--datasets jvnv jtes ogvc_acted ogvc_natural hcudb1] [--workers 16]
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT_ROOT = REPO / "finetune" / "manifests_src"
DEFAULT_CLIP_ROOT = REPO / "data" / "clips"
DEFAULT_HCUDB_ROOT = Path("/disk/fs1/Corpus/HCUDB/01_org/HCUDB1")

# HCUDB1's 11 acted emotions mapped onto the emotion2vec+ label space.
# リラックス・気楽 and 眠い・疲れた are deliberately NOT mapped: both sit on a
# boundary (HAP/NEU and NEU/SAD) and the user chose to exclude them from the
# evaluation set rather than force an arbitrary label (2026-11-18).
HCUDB_ACTED_TO_E2V = {
    "狂喜・楽しい": "HAP",
    "驚き": "SUR",
    "怒り": "ANG",
    "恐れ": "FEA",
    "余裕・嬉しい": "HAP",
    "冷静": "NEU",
    "嫌い": "DIS",
    "軽蔑": "DIS",
    "憂鬱・悲しい": "SAD",
}
HCUDB_AMBIGUOUS_ACTED = ("リラックス・気楽", "眠い・疲れた")
# Column order of the 11 perceived-emotion counts in collected_result(all).csv.
HCUDB_PERCEIVED_JP = [
    "狂喜・楽しい", "驚き", "怒り", "恐れ", "余裕・嬉しい", "冷静",
    "嫌い", "軽蔑", "リラックス・気楽", "眠い・疲れた", "憂鬱・悲しい",
]


def _ffmpeg_clip(src: Path, dst: Path, start: float | None, end: float | None) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return
    cmd = ["ffmpeg", "-y", "-v", "error"]
    if start is not None and end is not None:
        cmd += ["-ss", str(start), "-to", str(end)]
    cmd += ["-i", str(src), "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", str(dst)]
    subprocess.run(cmd, check=True, capture_output=True)


def _extract_jvnv_member(zf: zipfile.ZipFile, source_path: str, tmpdir: Path) -> Path:
    # manifest source: .../jvnv_v1/F1/happy/free/F1_happy_free_01.wav
    marker = "jvnv_v1/"
    idx = source_path.index(marker) + len(marker)
    member = "jvnv_v1/" + source_path[idx:]
    out = tmpdir / Path(member).name
    if not out.exists():
        with zf.open(member) as src, open(out, "wb") as dst:
            shutil.copyfileobj(src, dst)
    return out


def _opt_int(value: str) -> int | None:
    """Parse an aggregate CSV cell; the corpus has a few '#N/A' cells."""
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def hcudb1_source_rows(hcudb_root: Path) -> list[dict]:
    """Build HCUDB1 manifest rows from the corpus wavs + evaluator summary.

    Gold label = the acted emotion (演技感情) the speaker was instructed to
    perform, mapped to the emotion2vec+ 9-class space. The 16 evaluators'
    perceived-emotion counts are kept in ``metadata`` so the label can be
    switched to a perceived majority later without re-reading the corpus.
    """
    csv_path = hcudb_root / "eval" / "collected_result(all).csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"HCUDB1 evaluator summary not found: {csv_path}")
    rows: list[dict] = []
    with open(csv_path, encoding="cp932", newline="") as fh:
        for rec in csv.DictReader(fh):
            fname = (rec.get("音声ファイル名") or "").strip()
            if not fname:
                continue
            speaker = rec["話者ID"].strip()
            acted = rec["演技感情"].strip()
            excluded = acted in HCUDB_AMBIGUOUS_ACTED
            votes = {k: int(rec[k]) for k in HCUDB_PERCEIVED_JP}
            top = max(votes.values())
            row = {
                "dataset": "hcudb1",
                "utt_id": Path(fname).stem,
                "speaker": speaker,
                "native_emotion": acted,
                "canonical_emotion": None if excluded else HCUDB_ACTED_TO_E2V[acted],
                "excluded": excluded,
                "text": rec["セリフ"].strip(),
                "take": int(rec["テイク"]),
                "source_path": str(hcudb_root / "wav" / speaker / fname),
                "metadata": {
                    "sex": rec["性別"].strip(),
                    "acted_emotion": acted,
                    "valence_sum": _opt_int(rec["快不快"]),
                    "arousal_sum": _opt_int(rec["活性非活性"]),
                    "perceived_votes": votes,
                    "perceived_plurality": [k for k, v in votes.items() if v == top],
                },
            }
            if excluded:
                row["exclusion_reason"] = f"ambiguous_mapping:{acted}"
            rows.append(row)
    rows.sort(key=lambda r: r["utt_id"])
    return rows


def build_dataset(name: str, manifest: Path, clip_root: Path, out_manifest: Path,
                  jvnv_zip: Path | None, workers: int,
                  hcudb_root: Path | None = None) -> dict:
    if name == "hcudb1":
        rows = hcudb1_source_rows(hcudb_root or DEFAULT_HCUDB_ROOT)
    else:
        rows = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l]
    zf = None
    tmpdir = None
    if name == "jvnv":
        if jvnv_zip is None or not jvnv_zip.is_file():
            raise FileNotFoundError(f"--jvnv-zip required for jvnv (not found: {jvnv_zip})")
        zf = zipfile.ZipFile(jvnv_zip)
        tmpdir = Path(tempfile.mkdtemp(prefix="jvnv_extract_"))

    def work(row: dict) -> None:
        dst = clip_root / name / f"{row['utt_id']}.wav"
        if name == "jvnv":
            src = _extract_jvnv_member(zf, row["source_path"], tmpdir)
            _ffmpeg_clip(src, dst, None, None)
        else:
            _ffmpeg_clip(Path(row["source_path"]), dst, row.get("start"), row.get("end"))
        row["clip_path"] = str(dst)

    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(work, rows))
    finally:
        if zf is not None:
            zf.close()
        if tmpdir is not None:
            shutil.rmtree(tmpdir, ignore_errors=True)

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(out_manifest, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    missing = [r for r in rows if not Path(r["clip_path"]).is_file()]
    return {"dataset": name, "rows": len(rows), "missing": len(missing),
            "excluded": sum(1 for r in rows if r.get("excluded"))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="finetune.build_clips")
    here = Path(__file__).resolve().parent.parent
    ap.add_argument("--manifest-root", type=Path,
                    default=here.parent / "SER-model-comparison" / "manifests")
    ap.add_argument("--clip-root", type=Path, default=REPO / "data" / "clips")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--jvnv-zip", type=Path,
                    default=Path("/localtmp/sumiharu.kobayashi.r4/datasets/jvnv_ver1.zip"))
    ap.add_argument("--hcudb-root", type=Path, default=DEFAULT_HCUDB_ROOT,
                    help="HCUDB1 corpus root (wav/ + eval/); used for --datasets hcudb1")
    ap.add_argument("--datasets", nargs="*",
                    default=["jvnv", "jtes", "ogvc_acted", "ogvc_natural", "hcudb1"])
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args(argv)

    for name in args.datasets:
        manifest = args.manifest_root / f"{name}.jsonl"
        if name != "hcudb1" and not manifest.is_file():
            print(f"manifest not found: {manifest}", file=sys.stderr)
            return 2
        info = build_dataset(name, manifest, args.clip_root,
                             args.out_root / f"{name}.jsonl", args.jvnv_zip, args.workers,
                             args.hcudb_root)
        print(f"{name}: rows={info['rows']} excluded={info['excluded']} "
              f"missing={info['missing']} -> {args.out_root / (name + '.jsonl')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
