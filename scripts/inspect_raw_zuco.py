"""Inspect raw ZuCo Task 1 EEG (``rawData``) without assuming shapes.

Reports, from the data: array shapes and orientation, channel count, subject
IDs, valid/missing/corrupt trials, sentence durations, NaN and flat-channel
patterns, amplitude (unit evidence), whether the data are already
average-referenced, and the 50 Hz line-frequency signature that confirms the
500 Hz sampling rate stored in the authors' EEGLAB file.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import signal  # noqa: E402

from src.labels import label_lookup  # noqa: E402
from src.neurolm.config import load_config, save_json  # noqa: E402
from src.neurolm.dataset import iter_subject_trials, subject_files, subject_from_path  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/neurolm_probe.yaml")
    parser.add_argument("--mat-dir", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--report", default="reports/raw_zuco_inspection.md")
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--max-sentences", type=int, default=None)
    parser.add_argument("--psd-trials-per-subject", type=int, default=20)
    return parser.parse_args()


def average_reference_ratio(raw, reference_index):
    others = np.delete(raw, reference_index, axis=0)
    finite = np.isfinite(others).all(axis=0)
    if finite.sum() < 10:
        return np.nan
    stds = others[:, finite].std(axis=1)
    return float(others[:, finite].mean(axis=0).std() / max(np.median(stds), 1e-12))


def line_signature(raw, fs, line=50.0):
    """Power at ``line`` Hz relative to 44-48 and 52-56 Hz (log10 ratio)."""
    data = np.nan_to_num(raw[:-1])
    if data.shape[1] < fs:
        return None, None
    freqs, psd = signal.welch(data, fs=fs, nperseg=int(fs), axis=1)
    mean_psd = psd.mean(axis=0)
    at = mean_psd[np.argmin(np.abs(freqs - line))]
    flank = mean_psd[((freqs >= 44) & (freqs <= 48)) | ((freqs >= 52) & (freqs <= 56))].mean()
    return float(np.log10(at / flank)), (freqs, mean_psd)


def describe(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {}
    q = np.quantile(values, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    return {"n": int(len(values)), "mean": float(values.mean()), "min": float(q[0]), "q05": float(q[1]),
            "q25": float(q[2]), "median": float(q[3]), "q75": float(q[4]), "q95": float(q[5]), "max": float(q[6])}


def main():
    args = parse_args()
    config = load_config(args.config)
    fs = float(config["data"]["source_sfreq"])
    reference_index = int(config["preprocess"]["reference_channel_index"])
    lookup = label_lookup(args.labels_csv)
    os.makedirs(args.out_dir, exist_ok=True)
    records, ratios, line_ratios, psd_sum, psd_n = [], [], [], None, 0
    files = subject_files(args.mat_dir, args.subjects)
    for path in files:
        subject = subject_from_path(path)
        psd_used = 0
        for trial, record in iter_subject_trials(path, lookup=lookup, sfreq=fs, limit=args.max_sentences):
            if trial is not None:
                raw = trial.raw.astype(np.float64)
                record["average_reference_ratio"] = average_reference_ratio(raw, reference_index)
                ratios.append(record["average_reference_ratio"])
                if psd_used < args.psd_trials_per_subject:
                    value, spectrum = line_signature(raw, fs)
                    if value is not None:
                        line_ratios.append(value)
                        psd_sum = spectrum[1] if psd_sum is None else psd_sum + spectrum[1]
                        freqs = spectrum[0]
                        psd_n += 1
                        psd_used += 1
            records.append(record)
        ok = sum(r["status"] == "ok" for r in records if r["subject_id"] == subject)
        print(f"{subject}: {ok} usable trials")

    table = pd.DataFrame(records)
    table.to_csv(os.path.join(args.out_dir, "raw_zuco_trials.csv"), index=False)
    ok = table[table["status"] == "ok"]
    peak = None
    if psd_sum is not None:
        mean_psd = psd_sum / psd_n
        band = (freqs >= 30) & (freqs <= 70)
        deviations = np.log10(mean_psd[band]) - np.convolve(np.log10(mean_psd[band]), np.ones(7) / 7, mode="same")
        peak = float(freqs[band][3:-3][np.argmax(np.abs(deviations[3:-3]))])
    flat_counts = {}
    for value in ok.get("flat_channels", pd.Series(dtype=str)).fillna(""):
        for channel in filter(None, str(value).split(",")):
            flat_counts[int(channel)] = flat_counts.get(int(channel), 0) + 1
    summary = {
        "mat_dir": args.mat_dir,
        "subjects": sorted(table["subject_id"].unique().tolist()),
        "n_subject_files": len(files),
        "status_counts": table["status"].value_counts().to_dict(),
        "valid_trials": int(len(ok)),
        "valid_trials_per_subject": ok.groupby("subject_id").size().to_dict(),
        "unique_sentences_with_eeg": int(ok["sentence_id"].nunique()),
        "raw_shapes_stored": table["raw_shape"].value_counts().head(5).to_dict(),
        "n_channels_values": ok["n_channels"].value_counts().to_dict(),
        "sampling_rate_hz": fs,
        "sampling_rate_source": "EEG.srate = 500 in the authors' EEGLAB file (gip_ZAB_SR5_EEG.mat)",
        "line_noise_log10_ratio_at_50hz": describe(line_ratios),
        "largest_spectral_deviation_30_70hz": peak,
        "duration_s": describe(ok["duration_s"]),
        "duration_s_values": ok["duration_s"].round(4).tolist(),
        "n_samples": describe(ok["n_samples"]),
        "nan_fraction": describe(ok["nan_fraction"]),
        "trials_with_any_nan": int((ok["nan_fraction"] > 0).sum()),
        "flat_channel_counts": {str(k): v for k, v in sorted(flat_counts.items())},
        "reference_channel_flat_fraction": float(ok["reference_channel_flat"].mean()) if len(ok) else None,
        "median_channel_std_raw_units": describe(ok["median_channel_std"]),
        "average_reference_ratio": describe(ratios),
    }
    save_json(summary, os.path.join(args.out_dir, "raw_zuco_summary.json"))
    write_report(args.report, summary, table)
    print("summary ->", os.path.join(args.out_dir, "raw_zuco_summary.json"))


def write_report(path, summary, table):
    d = summary["duration_s"]
    ratio = summary["average_reference_ratio"]
    lines = [
        "# Raw ZuCo Task 1 EEG inspection", "",
        "Generated by `scripts/inspect_raw_zuco.py` from `results*_SR.mat` (`sentenceData.rawData`).", "",
        "| item | value |", "|---|---|",
        f"| subject files | {summary['n_subject_files']} |",
        f"| subjects | {', '.join(summary['subjects'])} |",
        f"| stored rawData shapes (top) | {summary['raw_shapes_stored']} |",
        f"| channels per trial | {summary['n_channels_values']} |",
        f"| sampling rate | {summary['sampling_rate_hz']:g} Hz ({summary['sampling_rate_source']}) |",
        f"| 50 Hz line signature (log10 power ratio, median) | {summary['line_noise_log10_ratio_at_50hz'].get('median')} |",
        f"| largest 30–70 Hz spectral deviation at | {summary['largest_spectral_deviation_30_70hz']} Hz |",
        f"| valid trials | {summary['valid_trials']} |",
        f"| sentences with ≥1 valid trial | {summary['unique_sentences_with_eeg']} |",
        f"| status counts | {summary['status_counts']} |",
        f"| duration (s) min / median / mean / q95 / max | {d.get('min', float('nan')):.2f} / {d.get('median', float('nan')):.2f} / "
        f"{d.get('mean', float('nan')):.2f} / {d.get('q95', float('nan')):.2f} / {d.get('max', float('nan')):.2f} |",
        f"| trials with any NaN | {summary['trials_with_any_nan']} |",
        f"| flat channels (index: trials) | {summary['flat_channel_counts']} |",
        f"| reference channel (104) flat in | {summary['reference_channel_flat_fraction']} of trials |",
        f"| median channel std (raw units) | {summary['median_channel_std_raw_units'].get('median')} |",
        f"| average-reference ratio, median | {ratio.get('median')} (≈0 means already average-referenced) |",
        "", "## Valid trials per subject", "",
        "| subject | valid trials |", "|---|---:|",
    ]
    for subject, count in summary["valid_trials_per_subject"].items():
        lines.append(f"| {subject} | {count} |")
    problems = table[table["status"] != "ok"]
    lines += ["", "## Missing or unusable trials", "",
              f"{len(problems)} rows (listed in `raw_zuco_trials.csv`).", ""]
    if len(problems):
        lines.append(problems.groupby(["subject_id", "status"]).size().to_string())
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
