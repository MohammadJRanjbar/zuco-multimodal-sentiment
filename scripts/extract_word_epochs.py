"""Extract a raw 1-s EEG epoch around each word's first fixation (ZuCo SR), for pretrained EEG models.

For every labelled sentence of every reader whose file has per-fixation EEG (``word.rawEEG``; 9 of the
12 ZuCo readers), fixation onsets are located in the sentence EEG as for the fixation-related potentials
(scripts/extract_frp.py), the sentence is preprocessed as for NeuroLM (0.1-75 Hz, 50 Hz notch, average
reference, 200 Hz) and -200..+800 ms around each word's first fixation is kept. See src/wordinfo/epochs.py
for the saved layout. Resumable per subject; readers without per-fixation EEG are listed as skipped in
word_epochs.json.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)

import numpy as np  # noqa: E402

from src.followup import frp  # noqa: E402
from src.labels import label_lookup, match_sentence  # noqa: E402
from src.neurolm.dataset import sample_id, subject_files, subject_from_path  # noqa: E402
from src.neurolm.preprocess import PreprocessConfig  # noqa: E402
from src.progress import progress  # noqa: E402
from src.wordinfo.epochs import TMAX_S, TMIN_S, save_subject_epochs, word_epochs  # noqa: E402


def process_subject(path, lookup, cfg):
    """Trials with epochs, or ``None`` when the file has no per-fixation EEG."""
    subject = subject_from_path(path)
    trials, seen, located, fixated = [], set(), 0, 0
    for _, text, parse in progress(frp.iter_frp_sentences(path), desc=subject, unit="sentence"):
        sentence_id, label = match_sentence(text, lookup)
        if sentence_id is None or sentence_id in seen:
            continue
        seen.add(sentence_id)
        parsed = parse()
        if parsed is not None and "missing_field" in parsed:
            return None, {}
        if parsed is None or parsed.get("raw") is None:
            continue
        onsets, _, _ = frp.locate_sentence(parsed["raw"], parsed["segments"])
        epochs, index = word_epochs(parsed["raw"], onsets, cfg)
        fixated += int((np.asarray(parsed["fixations"]) > 0).sum())
        located += len(index)
        trials.append({"sample_id": sample_id(subject, sentence_id), "sentence_id": int(sentence_id),
                       "label": int(label), "words": parsed["words"], "fixations": parsed["fixations"],
                       "epochs": epochs, "epoch_index": index,
                       **{f"{n.lower()}_ms": parsed["times"][n] for n in frp.READING_MEASURES}})
    return trials, {"sentences": len(trials), "words_fixated": fixated, "epochs": located,
                    "epochs_per_fixated_word": located / max(fixated, 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mat-dir", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for flag, value in (("--mat-dir", args.mat_dir), ("--labels-csv", args.labels_csv)):
        if not value or not os.path.exists(value):
            raise SystemExit(f"{flag} is empty or missing ({value!r}); run the notebook's path cells again.")
    lookup = label_lookup(args.labels_csv)
    cfg = PreprocessConfig()
    os.makedirs(args.out_dir, exist_ok=True)
    report_path = os.path.join(args.out_dir, "word_epochs.json")
    report = json.load(open(report_path)) if os.path.exists(report_path) else {"subjects": {}, "skipped": {}}
    report.update({"tmin_s": TMIN_S, "tmax_s": TMAX_S, "sfreq": cfg.target_sfreq, "preprocessing": cfg.to_dict()})
    started = time.time()
    for path in subject_files(args.mat_dir, args.subjects):
        subject = subject_from_path(path)
        if not args.overwrite and (os.path.exists(os.path.join(args.out_dir, f"{subject}.npz"))
                                   or subject in report["skipped"]):
            print(f"skip {subject}: done earlier")
            continue
        trials, summary = process_subject(path, lookup, cfg)
        if trials is None:
            print(f"skip {subject}: no per-fixation EEG in this file")
            report["skipped"][subject] = "no per-fixation EEG (word.rawEEG)"
        else:
            save_subject_epochs(args.out_dir, subject, trials)
            report["subjects"][subject] = summary
            print(f"{subject}: {summary}")
        json.dump(report, open(report_path, "w"), indent=1)
    total = sum(s["epochs"] for s in report["subjects"].values())
    print(f"done in {(time.time() - started) / 60:.1f} min: {len(report['subjects'])} readers, {total:,} word epochs; "
          f"skipped {', '.join(report['skipped']) or 'none'}")


if __name__ == "__main__":
    main()
