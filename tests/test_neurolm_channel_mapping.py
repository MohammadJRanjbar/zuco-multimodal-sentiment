import os

import numpy as np
import pytest

from src.neurolm import channel_mapping as cm

NEUROLM_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "NeuroLM")


def test_vocabulary_matches_neurolm_repository():
    assert len(cm.NEUROLM_CHANNEL_VOCAB) == 139
    assert cm.NEUROLM_CHANNEL_VOCAB.index("pad") == 136
    if not os.path.exists(os.path.join(NEUROLM_DIR, "dataset.py")):
        pytest.skip("NeuroLM repository not cloned next to this repository")
    assert cm.verify_vocabulary(NEUROLM_DIR)


def test_zuco_chanlocs_are_the_hydrocel_template():
    labels, rows = cm.load_zuco_chanlocs()
    assert len(labels) == 105 and labels[-1] == "Cz" and len(set(labels)) == 105
    sfp = cm.read_sfp()
    for row in rows:
        x, y, z = sfp[row["labels"]]
        np.testing.assert_allclose([float(row["X"]), float(row["Y"]), float(row["Z"])], [y, -x, z], atol=1e-6)


def test_alignment_places_egi_vertex_at_cz():
    egi, standard, info = cm.aligned_unit_positions()
    assert info["cz_offset_after_pitch_correction_deg"] < 1.0
    for vectors in (egi, standard):
        norms = np.linalg.norm(np.array(list(vectors.values())), axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-9)


def test_mapping_is_one_to_one_and_within_cap():
    labels, _ = cm.load_zuco_chanlocs()
    rows = cm.build_channel_mapping(labels, max_angle_deg=6.0)
    retained = [r for r in rows if r["status"] in {"exact", "approximate"}]
    targets = [r["target"] for r in retained]
    assert len(targets) == len(set(targets))
    assert all(cm.NEUROLM_CHANNEL_VOCAB[r["vocab_index"]] == r["target"] for r in retained)
    assert all(r["angle_deg"] <= 6.0 for r in retained)
    assert {r["status"] for r in rows} <= set(cm.STATUS_ORDER)
    by_label = {r["zuco_label"]: r for r in rows}
    assert by_label["Cz"]["status"] == "exact" and by_label["Cz"]["target"] == "CZ"
    # Electrodes that sit within one degree of a 10-20 site must be found.
    assert by_label["E70"]["target"] == "O1"
    assert by_label["E83"]["target"] == "O2"
    assert by_label["E124"]["target"] == "F4"
    summary = cm.summarize_mapping(rows)
    assert summary["n_zuco_channels"] == 105
    assert sum(summary["counts"].values()) == 105


def test_mapping_respects_reference_mode_and_exclusions():
    labels, _ = cm.load_zuco_chanlocs()
    rows = {r["zuco_label"]: r for r in cm.build_channel_mapping(labels, reference_mode="none")}
    assert rows["Cz"]["status"] == "excluded"
    rows = {r["zuco_label"]: r for r in cm.build_channel_mapping(labels, excluded_targets={"O1": "untrained"})}
    assert rows["E70"]["target"] != "O1"
    assert "O1" not in {r["target"] for r in rows.values()}


def test_ten_twenty_view_uses_only_ten_twenty_names():
    labels, _ = cm.load_zuco_chanlocs()
    rows = cm.build_channel_mapping(labels, target_set="10-20", max_angle_deg=8.6)
    retained = {r["target"] for r in rows if r["status"] in {"exact", "approximate"}}
    assert retained <= set(cm.TEN_TWENTY_19)


def test_embedding_evidence_flags_untouched_rows():
    rng = np.random.default_rng(0)
    weight = rng.standard_normal((256, 64)) * 0.5
    shared = rng.standard_normal(64)
    trained = [i for i in range(139) if i not in (5, 6)]
    weight[trained] = 3.0 * shared + 0.3 * rng.standard_normal((len(trained), 64))
    rows, reference = cm.embedding_row_evidence(weight)
    flagged = {r["vocab_index"] for r in rows if r["likely_untrained"]}
    assert flagged == {5, 6}
    assert reference["unused_rows"] == 256 - 139


def test_montage_consistency_detects_label_order():
    rng = np.random.default_rng(1)
    points = rng.standard_normal((30, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    angle = np.arccos(np.clip(points @ points.T, -1, 1))
    corr = np.exp(-angle / 0.4)
    good = cm.montage_consistency(corr, points, n_permutations=200)
    assert good["spearman_rho"] < -0.9 and good["p_value_one_sided"] < 0.01
    shuffled = cm.montage_consistency(corr, points[rng.permutation(30)], n_permutations=200)
    assert shuffled["p_value_one_sided"] > 0.01
