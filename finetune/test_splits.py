import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from splits import (E2V_LABELS, JTES_EVAL_SPEAKERS, JTES_TEST_SPEAKERS,
                    LABEL_TO_IDX, filter_rows, majority_label, split_jtes,
                    split_utterance_stratified)


def test_majority_label_decides():
    assert majority_label(["NEU", "NEU", "HAP"]) == ("NEU", 2)
    assert majority_label(["NEU", "HAP", "SAD"]) == ("NEU", 1)


def test_filter_rows_drops_non_emotion_and_no_majority():
    rows = [
        {"utt_id": "a", "speaker": "s", "native_emotion": "ACC",
         "canonical_emotion": None, "excluded": True},
        {"utt_id": "b", "speaker": "s", "native_emotion": "OTH",
         "canonical_emotion": None, "excluded": True},
        {"utt_id": "c", "speaker": "s", "native_emotion": "JOY",
         "canonical_emotion": "HAP", "excluded": False},
        {"utt_id": "d", "speaker": "s", "native_emotion": "NEU",
         "canonical_emotion": "NEU", "excluded": False,
         "raters": ["NEU", "HAP", "SAD"]},
    ]
    excl = []
    kept = filter_rows(rows, "ogvc_natural", excl)
    assert [r["utt_id"] for r in kept] == ["c"]
    reasons = sorted(e["reason"] for e in excl)
    assert reasons == ["excluded_emotion", "excluded_emotion", "no_majority_vote"]
    assert all("raters" in e for e in excl if e["reason"] == "no_majority_vote")


def test_label_space_is_e2v_9class():
    # The pretrained proj head follows tokens.txt: index 5=OTHER, 6=SAD, 7=SUR.
    assert E2V_LABELS == ["ANG", "DIS", "FEA", "HAP", "NEU", "OTHER", "SAD", "SUR", "UNKNOWN"]
    assert len(LABEL_TO_IDX) == 9


def test_split_jtes_user_fixed_speakers():
    rows = []
    for spk in ["f01", "f10", "f29", "m10", "m20", "f16", "f35", "m14", "m22"]:
        for i in range(3):
            rows.append({"utt_id": f"{spk}_x{i}", "speaker": spk,
                         "canonical_emotion": "NEU"})
    out = split_jtes(rows)
    assert {r["speaker"] for r in out["eval"]} == set(JTES_EVAL_SPEAKERS)
    assert {r["speaker"] for r in out["test"]} == set(JTES_TEST_SPEAKERS)
    assert {r["speaker"] for r in out["train"]} == {"f01"}
    assert len(out["train"]) == 3


def test_stratified_split_is_complete_and_disjoint():
    rows = []
    for spk in ["F1", "M1"]:
        for emo in ["HAP", "ANG"]:
            for i in range(20):
                rows.append({"utt_id": f"{spk}_{emo}_{i:02d}", "speaker": spk,
                             "canonical_emotion": emo})
    out = split_utterance_stratified(rows, seed=42)
    ids = [r["utt_id"] for s in out.values() for r in s]
    assert len(ids) == len(set(ids)) == 80
    for split, members in out.items():
        for emo in ("HAP", "ANG"):
            assert any(r["canonical_emotion"] == emo for r in members)


def test_filter_rows_honours_exclusion_reason():
    rows = [
        {"utt_id": "a", "speaker": "FA", "native_emotion": "リラックス・気楽",
         "canonical_emotion": None, "excluded": True,
         "exclusion_reason": "ambiguous_mapping:リラックス・気楽"},
        {"utt_id": "b", "speaker": "FA", "native_emotion": "冷静",
         "canonical_emotion": "NEU", "excluded": False},
    ]
    excl = []
    kept = filter_rows(rows, "hcudb1", excl)
    assert [r["utt_id"] for r in kept] == ["b"]
    assert [e["reason"] for e in excl] == ["ambiguous_mapping:リラックス・気楽"]


def test_hcudb1_mapping_covers_all_but_ambiguous_emotions():
    from build_clips import (HCUDB_ACTED_TO_E2V, HCUDB_AMBIGUOUS_ACTED,
                             HCUDB_PERCEIVED_JP)
    acted = set(HCUDB_ACTED_TO_E2V)
    ambiguous = set(HCUDB_AMBIGUOUS_ACTED)
    assert acted | ambiguous == set(HCUDB_PERCEIVED_JP)
    assert not acted & ambiguous
    assert ambiguous == {"リラックス・気楽", "眠い・疲れた"}
    mapped = set(HCUDB_ACTED_TO_E2V.values())
    assert mapped <= set(E2V_LABELS)
    assert "OTHER" not in mapped and "UNKNOWN" not in mapped
    assert HCUDB_ACTED_TO_E2V["狂喜・楽しい"] == "HAP"
    assert HCUDB_ACTED_TO_E2V["余裕・嬉しい"] == "HAP"
    assert HCUDB_ACTED_TO_E2V["嫌い"] == "DIS"
    assert HCUDB_ACTED_TO_E2V["軽蔑"] == "DIS"
    assert HCUDB_ACTED_TO_E2V["憂鬱・悲しい"] == "SAD"


def test_eval_only_dataset_writes_no_train_manifest(tmp_path):
    from splits import build_dataset
    rows = []
    # every selected speaker plus one speaker that must be dropped
    for spk in ("FA", "FB", "MC", "ME", "FG"):
        for i in range(2):
            rows.append({"dataset": "hcudb1", "utt_id": f"{spk}-01-01-{i}",
                         "speaker": spk, "native_emotion": "冷静",
                         "canonical_emotion": "NEU", "excluded": False,
                         "clip_path": f"/x/{spk}-01-01-{i}.wav"})
    manifest = tmp_path / "hcudb1.jsonl"
    manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                        encoding="utf-8")
    out = tmp_path / "out"
    info = build_dataset("hcudb1", manifest, out, seed=42)
    assert (info["train"], info["eval"], info["test"]) == (0, 8, 0)
    assert (out / "hcudb1" / "eval.jsonl").is_file()
    assert not (out / "hcudb1" / "train.jsonl").exists()
    assert not (out / "hcudb1" / "test.jsonl").exists()
    written = [json.loads(l) for l in
               (out / "hcudb1" / "eval.jsonl").read_text(encoding="utf-8").splitlines() if l]
    assert {r["speaker"] for r in written} == {"FA", "FB", "MC", "ME"}
    excl = json.loads((out / "hcudb1" / "exclusions.json").read_text(encoding="utf-8"))
    assert excl["n_excluded"] == 2
    assert {e["reason"] for e in excl["exclusions"]} == {"speaker_not_selected"}
