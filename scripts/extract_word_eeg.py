"""Extract word-level, fixation-locked EEG (ZuCo TRT band power) per reader x sentence.

Resumable: subjects already saved in --out-dir are skipped. --task NR or TSR reads the other ZuCo 1.0
reading tasks; their sentences have no sentiment label (label 99, text-derived sentence ids).
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)  # show progress in Colab before any crash

import pandas as pd  # noqa: E402

from src.fusion.word_eeg import extract_subject, save_subject, write_manifest  # noqa: E402
from src.labels import label_lookup, match_sentence, unlabelled_match  # noqa: E402
from src.neurolm.dataset import subject_files, subject_from_path  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mat-dir", required=True)
    parser.add_argument("--labels-csv", default=None, help="required for --task SR")
    parser.add_argument("--task", default="SR", choices=["SR", "NR", "TSR"])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--measure", default="TRT", choices=["TRT", "FFD", "GD", "GPT", "SFD"])
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.labels_csv:
        lookup, match = label_lookup(args.labels_csv), match_sentence
    elif args.task == "SR":
        raise SystemExit("--labels-csv is required for --task SR")
    else:
        lookup, match = None, unlabelled_match
    os.makedirs(args.out_dir, exist_ok=True)
    records_path = os.path.join(args.out_dir, "word_eeg_trials.csv")
    old = pd.read_csv(records_path).to_dict("records") if os.path.exists(records_path) else []
    records = []
    for path in subject_files(args.mat_dir, args.subjects, args.task):
        subject = subject_from_path(path)
        if os.path.exists(os.path.join(args.out_dir, f"{subject}.npz")) and not args.overwrite:
            records += [r for r in old if r["subject_id"] == subject]
            print(f"skip {subject}: already extracted")
            continue
        started = time.time()
        trials, subject_records = extract_subject(path, lookup, match, args.measure)
        save_subject(args.out_dir, subject, trials)
        records += subject_records
        words = sum(len(t["words"]) for t in trials)
        with_eeg = sum(int((~pd.isna(t["features"][:, 0, 0])).sum()) for t in trials)
        print(f"{subject}: {len(trials)} trials, {words} words, {with_eeg} with EEG ({time.time() - started:.0f}s)")
    summary = write_manifest(args.out_dir, records, args.measure)
    print(summary)


if __name__ == "__main__":
    main()
