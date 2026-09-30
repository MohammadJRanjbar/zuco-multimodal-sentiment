"""Map ZuCo's EGI HydroCel electrodes onto NeuroLM's standard channel names.

NeuroLM (and LaBraM) identify every EEG token by the index of its electrode
name in ``standard_1020`` (``NeuroLM/dataset.py``); that index selects a learned
spatial embedding. ZuCo was recorded with a 128-channel EGI HydroCel Geodesic
Sensor Net, whose electrodes are named ``E1``...``E128`` plus the vertex
reference ``Cz``. Those names are not in NeuroLM's vocabulary, so every ZuCo
electrode has to be placed at a 10-10 site, or left out.

The mapping is geometric and deterministic; nothing is hand-assigned:

1. Electrode labels and their order come from the dataset authors' EEGLAB
   ``chanlocs`` (``montages/zuco_chanlocs.csv``).
2. Template coordinates come from MNE-Python's ``GSN-HydroCel-129.sfp`` and
   ``standard_1005.elc`` (vendored in ``montages/``, BSD-3-Clause).
3. Both templates are put in a head frame built from their own fiducials
   (EGI ``FidNz/FidT9/FidT10``; 10-05 ``Nz/LPA/RPA``, the same pairing MNE
   uses), projected onto their best-fitting spheres, and the residual pitch is
   removed with the one landmark that is identical by construction: the EGI
   vertex reference is placed at 10-20 ``Cz``.
4. An electrode is *approximate* only when it and a 10-10 site are each
   other's nearest neighbour and lie within ``max_angle_deg``. Everything else
   is reported as *missing*; nothing is silently reassigned.
"""

import csv
import hashlib
import json
import os

import numpy as np


MONTAGE_DIR = os.path.join(os.path.dirname(__file__), "montages")
ELC_PATH = os.path.join(MONTAGE_DIR, "standard_1005.elc")
SFP_PATH = os.path.join(MONTAGE_DIR, "GSN-HydroCel-129.sfp")
ZUCO_CHANLOCS_PATH = os.path.join(MONTAGE_DIR, "zuco_chanlocs.csv")

# Verbatim copy of ``standard_1020`` from NeuroLM/dataset.py (commit 0cda987).
# The list position is the embedding row NeuroLM uses for that electrode.
# ``verify_vocabulary`` checks this copy against a cloned NeuroLM repository.
NEUROLM_CHANNEL_VOCAB = [
    "FP1", "FPZ", "FP2",
    "AF9", "AF7", "AF5", "AF3", "AF1", "AFZ", "AF2", "AF4", "AF6", "AF8", "AF10",
    "F9", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8", "F10",
    "FT9", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "FT10",
    "T9", "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8", "T10",
    "TP9", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "TP10",
    "P9", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "P10",
    "PO9", "PO7", "PO5", "PO3", "PO1", "POZ", "PO2", "PO4", "PO6", "PO8", "PO10",
    "O1", "OZ", "O2", "O9", "CB1", "CB2",
    "IZ", "O10", "T3", "T5", "T4", "T6", "M1", "M2", "A1", "A2",
    "CFC1", "CFC2", "CFC3", "CFC4", "CFC5", "CFC6", "CFC7", "CFC8",
    "CCP1", "CCP2", "CCP3", "CCP4", "CCP5", "CCP6", "CCP7", "CCP8",
    "T1", "T2", "FTT9h", "TTP7h", "TPP9h", "FTT10h", "TPP8h", "TPP10h",
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "pad", "I1", "I2",
]
PAD_CHANNEL = "pad"

# Vocabulary names that are never mapping targets, with the reason.
NON_TARGET_REASONS = {
    **{name: "old 10-20 alias of T7/T8/P7/P8" for name in ["T3", "T4", "T5", "T6"]},
    **{name: "ear/mastoid site, not a scalp 10-10 site" for name in ["A1", "A2", "M1", "M2"]},
    **{name: "10-05 extension, not 10-10" for name in [
        "CCP1", "CCP2", "CCP3", "CCP4", "CCP5", "CCP6", "CCP7", "CCP8",
        "CFC1", "CFC2", "CFC3", "CFC4", "CFC5", "CFC6", "CFC7", "CFC8",
        "FTT9h", "TTP7h", "TPP9h", "FTT10h", "TPP8h", "TPP10h", "I1", "I2",
    ]},
    **{name: "no standard_1005 coordinates" for name in ["O9", "O10", "CB1", "CB2", "T1", "T2"]},
}

# The classical 19-electrode 10-20 montage, using NeuroLM's names.
TEN_TWENTY_19 = [
    "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T7", "C3", "CZ",
    "C4", "T8", "P7", "P3", "PZ", "P4", "P8", "O1", "O2",
]

STATUS_ORDER = ["exact", "approximate", "missing", "excluded"]
DEFAULT_MAX_ANGLE_DEG = 6.0


# ----------------------------------------------------------------------------
# Template coordinates
# ----------------------------------------------------------------------------


def read_elc(path=ELC_PATH):
    """Read an ASA ``.elc`` file into ``{label: xyz}`` (mm)."""
    lines = open(path).read().splitlines()
    start = lines.index("Positions")
    stop = lines.index("Labels")
    positions = [[float(value) for value in line.split()] for line in lines[start + 1:stop]]
    labels = " ".join(lines[stop + 1:]).split()
    if len(labels) != len(positions):
        raise ValueError(f"{path}: {len(labels)} labels for {len(positions)} positions")
    return {label: np.asarray(xyz, dtype=float) for label, xyz in zip(labels, positions)}


def read_sfp(path=SFP_PATH):
    """Read an EGI ``.sfp`` file into ``{label: xyz}`` (cm)."""
    positions = {}
    for line in open(path):
        parts = line.split()
        if len(parts) == 4:
            positions[parts[0]] = np.asarray([float(value) for value in parts[1:]])
    return positions


def _head_frame(nasion, lpa, rpa):
    """Return a function mapping points into the MNE/Neuromag head frame."""
    right = (rpa - lpa) / np.linalg.norm(rpa - lpa)
    origin = lpa + np.dot(nasion - lpa, right) * right
    anterior = (nasion - origin) / np.linalg.norm(nasion - origin)
    superior = np.cross(right, anterior)
    rotation = np.vstack([right, anterior, superior])
    return lambda points: (rotation @ (np.asarray(points, dtype=float) - origin).T).T


def _fit_sphere(points):
    """Least-squares sphere through ``points``; returns ``(center, radius)``."""
    points = np.asarray(points, dtype=float)
    design = np.hstack([2.0 * points, np.ones((len(points), 1))])
    target = (points ** 2).sum(axis=1)
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    center = solution[:3]
    return center, float(np.sqrt(solution[3] + center @ center))


def _rotation_x(angle):
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, cos, -sin], [0.0, sin, cos]])


def _unit(vector):
    return vector / np.linalg.norm(vector)


def aligned_unit_positions():
    """Unit-sphere positions of EGI and 10-05 electrodes in one head frame.

    Returns ``(egi, standard, info)`` where ``egi`` and ``standard`` map labels to
    unit vectors (x right, y anterior, z superior) and ``info`` records the
    alignment diagnostics written into the report.
    """
    elc = read_elc()
    sfp = read_sfp()
    to_std = _head_frame(elc["Nz"], elc["LPA"], elc["RPA"])
    to_egi = _head_frame(sfp["FidNz"], sfp["FidT9"], sfp["FidT10"])
    std_xyz = {k: to_std(v) for k, v in elc.items() if k not in {"Nz", "LPA", "RPA"}}
    egi_xyz = {k: to_egi(v) for k, v in sfp.items() if not k.startswith("Fid")}

    # Fit spheres to the upper head only; face/neck sites distort the fit.
    std_center, std_radius = _fit_sphere([v for v in std_xyz.values() if v[2] > 0])
    egi_center, egi_radius = _fit_sphere([v for v in egi_xyz.values() if v[2] > 0])
    standard = {k: _unit(v - std_center) for k, v in std_xyz.items()}
    egi = {k: _unit(v - egi_center) for k, v in egi_xyz.items()}

    before = angular_distance_deg(egi["Cz"], standard["Cz"])
    pitch = np.arctan2(standard["Cz"][1], standard["Cz"][2]) - np.arctan2(
        egi["Cz"][1], egi["Cz"][2]
    )
    rotation = _rotation_x(-pitch)
    egi = {k: _unit(rotation @ v) for k, v in egi.items()}
    after = angular_distance_deg(egi["Cz"], standard["Cz"])
    info = {
        "egi_template": os.path.basename(SFP_PATH),
        "standard_template": os.path.basename(ELC_PATH),
        "egi_fiducials": ["FidNz", "FidT9", "FidT10"],
        "standard_fiducials": ["Nz", "LPA", "RPA"],
        "standard_sphere_radius_mm": std_radius,
        "egi_sphere_radius_cm": egi_radius,
        "cz_offset_before_pitch_correction_deg": before,
        "pitch_correction_deg": float(np.degrees(pitch)),
        "cz_offset_after_pitch_correction_deg": after,
    }
    return egi, standard, info


def angular_distance_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def candidate_targets(target_set="10-10"):
    """NeuroLM names eligible as mapping targets, in vocabulary order."""
    standard_names = {name.upper() for name in read_elc()}
    if target_set == "10-20":
        return list(TEN_TWENTY_19)
    if target_set != "10-10":
        raise ValueError("target_set must be '10-10' or '10-20'")
    names = []
    for name in NEUROLM_CHANNEL_VOCAB:
        if name in NON_TARGET_REASONS or name == PAD_CHANNEL or "-" in name:
            continue
        if name.upper() in standard_names:
            names.append(name)
    return names


def neighbour_spacing_deg(target_set="10-10"):
    """Median angle between each 10-10 site and its nearest 10-10 neighbour."""
    _, standard, _ = aligned_unit_positions()
    by_upper = {k.upper(): v for k, v in standard.items()}
    points = np.array([by_upper[name] for name in candidate_targets(target_set)])
    angles = np.degrees(np.arccos(np.clip(points @ points.T, -1.0, 1.0)))
    np.fill_diagonal(angles, np.inf)
    return float(np.median(angles.min(axis=1)))


# ----------------------------------------------------------------------------
# ZuCo channel labels
# ----------------------------------------------------------------------------


def load_zuco_chanlocs(path=ZUCO_CHANLOCS_PATH):
    """ZuCo electrode labels in data order, from the authors' EEGLAB chanlocs."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} is missing; build it with scripts/check_channel_mapping.py "
            "--chanlocs-mat from the ZuCo authors' EEGLAB file"
        )
    with open(path) as handle:
        rows = list(csv.DictReader(handle))
    labels = [row["labels"].strip() for row in rows]
    if len(set(labels)) != len(labels):
        raise ValueError("duplicate labels in ZuCo chanlocs")
    return labels, rows


# ----------------------------------------------------------------------------
# Mapping
# ----------------------------------------------------------------------------


def build_channel_mapping(
    zuco_labels,
    *,
    reference_label="Cz",
    reference_mode="average",
    max_angle_deg=DEFAULT_MAX_ANGLE_DEG,
    target_set="10-10",
    excluded_targets=None,
    excluded_channels=None,
):
    """Assign each ZuCo electrode to at most one NeuroLM channel name.

    ``reference_mode='average'`` re-references every trial to the average of
    all electrodes, which restores a real signal on the recording reference
    (Cz); with ``'none'`` the reference stays flat and is excluded.
    ``excluded_targets`` maps NeuroLM names to reasons (for example, spatial
    embeddings that the checkpoint evidence marks as untrained).
    """
    excluded_targets = dict(excluded_targets or {})
    excluded_channels = dict(excluded_channels or {})
    egi, standard, alignment = aligned_unit_positions()
    std_upper = {k.upper(): v for k, v in standard.items()}
    targets = [t for t in candidate_targets(target_set) if t not in excluded_targets]
    vocab_index = {name: i for i, name in enumerate(NEUROLM_CHANNEL_VOCAB)}
    std_radius = alignment["standard_sphere_radius_mm"]

    rows = []
    for index, label in enumerate(zuco_labels):
        rows.append({
            "zuco_index": index,
            "zuco_label": label,
            "target": "",
            "vocab_index": -1,
            "angle_deg": np.nan,
            "distance_mm": np.nan,
            "status": None,
            "reason": "",
            "nearest_target": "",
            "nearest_angle_deg": np.nan,
        })

    claimed = {}
    for row in rows:
        label = row["zuco_label"]
        if label in excluded_channels:
            row.update(status="excluded", reason=excluded_channels[label])
            continue
        if label == reference_label and reference_mode == "none":
            row.update(status="excluded", reason="flat recording reference (no re-referencing)")
            continue
        upper = label.upper()
        if upper in vocab_index and upper not in NON_TARGET_REASONS and upper != PAD_CHANNEL:
            if upper in excluded_targets:
                row.update(status="excluded", reason=f"target {upper}: {excluded_targets[upper]}")
                continue
            row.update(
                status="exact",
                target=upper,
                vocab_index=vocab_index[upper],
                angle_deg=0.0,
                distance_mm=0.0,
                nearest_target=upper,
                nearest_angle_deg=0.0,
                reason="identical electrode name",
            )
            claimed[upper] = row["zuco_index"]

    open_targets = [t for t in targets if t not in claimed]
    candidates = [row for row in rows if row["status"] is None]
    positioned = [row for row in candidates if row["zuco_label"] in egi]
    for row in candidates:
        if row["zuco_label"] not in egi:
            row.update(status="missing", reason="label not in the GSN-HydroCel-129 template")

    if positioned and open_targets:
        egi_points = np.array([egi[row["zuco_label"]] for row in positioned])
        target_points = np.array([std_upper[t.upper()] for t in open_targets])
        angles = np.degrees(np.arccos(np.clip(egi_points @ target_points.T, -1.0, 1.0)))
        best_target = angles.argmin(axis=1)
        best_electrode = angles.argmin(axis=0)
        for i, row in enumerate(positioned):
            j = int(best_target[i])
            angle = float(angles[i, j])
            row["nearest_target"] = open_targets[j]
            row["nearest_angle_deg"] = angle
            mutual = int(best_electrode[j]) == i
            if mutual and angle <= max_angle_deg:
                row.update(
                    status="approximate",
                    target=open_targets[j],
                    vocab_index=vocab_index[open_targets[j]],
                    angle_deg=angle,
                    distance_mm=float(np.radians(angle) * std_radius),
                    reason="mutual nearest 10-10 site",
                )
            elif mutual:
                row.update(
                    status="missing",
                    reason=f"nearest site {open_targets[j]} is {angle:.1f} deg away (> {max_angle_deg:g})",
                )
            else:
                rival = positioned[int(best_electrode[j])]["zuco_label"]
                row.update(
                    status="missing",
                    reason=f"nearest site {open_targets[j]} is closer to {rival}",
                )
    for row in rows:
        if row["status"] is None:
            row.update(status="missing", reason="no open 10-10 target")
    return rows


def summarize_mapping(rows):
    counts = {status: sum(row["status"] == status for row in rows) for status in STATUS_ORDER}
    retained = counts["exact"] + counts["approximate"]
    total = len(rows)
    angles = [row["angle_deg"] for row in rows if row["status"] == "approximate"]
    return {
        "n_zuco_channels": total,
        "counts": counts,
        "n_retained": retained,
        "n_not_retained": total - retained,
        "percent_retained": 100.0 * retained / total if total else 0.0,
        "approximate_angle_deg": {
            "median": float(np.median(angles)) if angles else None,
            "max": float(np.max(angles)) if angles else None,
        },
    }


def retained_channels(rows):
    """Retained electrodes in ZuCo data order: ``[(index, label, target, vocab)]``."""
    return [
        (row["zuco_index"], row["zuco_label"], row["target"], row["vocab_index"])
        for row in rows
        if row["status"] in {"exact", "approximate"}
    ]


def mapping_fingerprint(rows):
    payload = [(index, label, target) for index, label, target, _ in retained_channels(rows)]
    return hashlib.sha256(json.dumps(payload).encode("utf-8")).hexdigest()[:16]


def verify_vocabulary(neurolm_dir):
    """Assert the vendored vocabulary equals ``standard_1020`` in NeuroLM."""
    import ast

    source = open(os.path.join(neurolm_dir, "dataset.py")).read()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "standard_1020":
            upstream = ast.literal_eval(node.value)
            if upstream != NEUROLM_CHANNEL_VOCAB:
                raise AssertionError("NeuroLM standard_1020 differs from the vendored vocabulary")
            return True
    raise AssertionError("standard_1020 not found in NeuroLM/dataset.py")


# ----------------------------------------------------------------------------
# Evidence checks
# ----------------------------------------------------------------------------


def embedding_row_evidence(weight, vocab=NEUROLM_CHANNEL_VOCAB, z_limit=4.0):
    """Compare spatial-embedding rows with rows that are never indexed.

    NeuroLM's spatial embeddings have 256 rows but only ``len(vocab)`` names, so
    rows ``len(vocab):`` keep their initial random values (only decoupled
    weight decay touches them). A named row whose norm and similarity to other
    named rows both fall inside the unused-row distribution is reported as
    ``indistinguishable_from_unused``. This is descriptive: small learning rates
    move rows little relative to an N(0, 1) initialisation, so trained rows can
    be indistinguishable too. It must not be used on its own to drop channels.
    """
    weight = np.asarray(weight, dtype=np.float64)
    n_named = len(vocab)
    if weight.shape[0] <= n_named + 8:
        raise ValueError("not enough unused embedding rows to form a reference")
    norms = np.linalg.norm(weight, axis=1)
    unit = weight / norms[:, None]
    reference = np.arange(n_named, weight.shape[0])
    named = np.arange(n_named)
    ref_mean, ref_std = norms[reference].mean(), norms[reference].std(ddof=1)
    cos = unit @ unit[named].T
    cos[named, named] = 0.0
    max_cos = np.abs(cos).max(axis=1)
    ref_cos_limit = float(np.quantile(max_cos[reference], 0.99))
    rows = []
    for index in named:
        z = float((norms[index] - ref_mean) / ref_std) if ref_std > 0 else float("inf")
        rows.append({
            "vocab_index": int(index),
            "name": vocab[index],
            "norm": float(norms[index]),
            "norm_z_vs_unused": z,
            "max_abs_cos_to_named": float(max_cos[index]),
            "indistinguishable_from_unused": bool(abs(z) <= z_limit and max_cos[index] <= ref_cos_limit),
        })
    return rows, {
        "unused_rows": int(len(reference)),
        "unused_norm_mean": float(ref_mean),
        "unused_norm_std": float(ref_std),
        "unused_max_abs_cos_q99": ref_cos_limit,
        "z_limit": z_limit,
    }


def montage_consistency(corr, unit_positions, n_permutations=1000, seed=0):
    """Test that signal correlation falls with the claimed electrode distance.

    ``corr`` is a channel x channel correlation matrix estimated from raw EEG;
    ``unit_positions`` the claimed positions in the same order. Volume
    conduction makes neighbouring electrodes strongly correlated, so a correct
    label order gives a strongly negative Spearman rho between |corr| and
    angular distance. Permuting positions gives the null distribution.
    """
    from scipy.stats import rankdata

    corr = np.asarray(corr, dtype=float)
    positions = np.asarray(unit_positions, dtype=float)
    n = len(positions)
    upper = np.triu_indices(n, k=1)
    distance = np.degrees(np.arccos(np.clip(positions @ positions.T, -1.0, 1.0)))
    signal_rank = rankdata(np.abs(corr[upper]))

    def rho(dist):
        dist_rank = rankdata(dist[upper])
        return float(np.corrcoef(signal_rank, dist_rank)[0, 1])

    observed = rho(distance)
    rng = np.random.default_rng(seed)
    null = np.array([
        rho(distance[np.ix_(order, order)])
        for order in (rng.permutation(n) for _ in range(n_permutations))
    ])
    np.fill_diagonal(distance, np.inf)
    abs_corr = np.abs(corr.copy())
    np.fill_diagonal(abs_corr, -np.inf)
    nearest3 = np.argsort(distance, axis=1)[:, :3]
    partner = abs_corr.argmax(axis=1)
    hit_rate = float(np.mean([partner[i] in nearest3[i] for i in range(n)]))
    return {
        "spearman_rho": observed,
        "null_mean": float(null.mean()),
        "null_std": float(null.std()),
        "p_value_one_sided": float((1 + np.sum(null <= observed)) / (1 + n_permutations)),
        "most_correlated_partner_within_3_nearest": hit_rate,
        "n_channels": n,
        "n_permutations": n_permutations,
    }


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------


def _fmt(value, digits=1):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return ""
    return f"{value:.{digits}f}"


def mapping_table_markdown(rows):
    lines = [
        "| # | ZuCo channel | NeuroLM channel | status | angle (deg) | ~distance (mm) | note |",
        "|---:|---|---|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['zuco_index']} | {row['zuco_label']} | {row['target'] or '—'} | "
            f"{row['status']} | {_fmt(row['angle_deg'])} | {_fmt(row['distance_mm'])} | "
            f"{row['reason']} |"
        )
    return "\n".join(lines)


def save_mapping(rows, summary, path_prefix):
    """Write ``<prefix>.csv`` and ``<prefix>.json`` for downstream scripts."""
    os.makedirs(os.path.dirname(path_prefix) or ".", exist_ok=True)
    fields = list(rows[0].keys())
    with open(path_prefix + ".csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    serializable = [
        {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in row.items()}
        for row in rows
    ]
    with open(path_prefix + ".json", "w") as handle:
        json.dump({"summary": summary, "rows": serializable}, handle, indent=2)
