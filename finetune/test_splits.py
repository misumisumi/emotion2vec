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
    assert E2V_LABELS == ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD", "SUR", "OTHER", "UNKNOWN"]
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
