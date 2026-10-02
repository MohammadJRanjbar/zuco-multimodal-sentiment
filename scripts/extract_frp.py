"""Extract fixation-related potentials (FRPs) per reader x sentence from raw ZuCo files.

Output (--out-dir) uses the word-EEG cache format, so every word-EEG analysis
(analyze_eeg_signal.py, scan_eeg_encoding.py, ...) runs on it unchanged:
features [n_words, 8 windows, 105 channels] = mean amplitude (microvolts) in
0-100, ..., 600-700 ms and the N400 window 300-500 ms after the word's first
fixation. Also writes per-word timing (timing/<subject>.npz), per-subject grand
averages (grand_average/<subject>.npz) and frp_extraction.json with how many
fixations were located in the sentence EEG. Resumable per subject.

--probe N processes only N labelled sentences of the first subject and prints
the data layout and match statistics, without writing the cache.
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
from src.fusion.word_eeg import save_subject  # noqa: E402
from src.labels import label_lookup, match_sentence  # noqa: E402
from src.neurolm.dataset import sample_id, subject_files, subject_from_path  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mat-dir", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--probe", type=int, default=0, help="only inspect N sentences of the first subject")
    parser.add_argument("--min-match-rate", type=float, default=0.5,
                        help="stop after the first subject if fewer fixations than this are located")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def process_subject(path, lookup, limit=None):
    subject = subject_from_path(path)
    trials, timing, counts, seen = [], [], {}, set()
    sums, n_epochs, layout = None, 0, None
    ffd_pairs = []
    from src.progress import progress

    for position, text, parse in progress(frp.iter_frp_sentences(path), desc=subject, unit="sentence"):
        sentence_id, label = match_sentence(text, lookup)
        if sentence_id is None or sentence_id in seen:
            continue
        seen.add(sentence_id)
        parsed = parse()
        if parsed is None:
            counts["no_data"] = counts.get("no_data", 0) + 1
            continue
        if "missing_field" in parsed:
            raise SystemExit(f"{subject}: word struct has no '{parsed['missing_field']}' field; "
                             f"available fields: {parsed['word_fields']}")
        raw = parsed["raw"]
        if raw is None:
            counts["raw_unreadable"] = counts.get("raw_unreadable", 0) + 1
            continue
        if layout is None:
            n_segments = [len(s) for s in parsed["segments"]]
            layout = {"raw_shape": list(raw.shape), "n_words": len(parsed["words"]),
                      "segments_per_word": n_segments,
                      "segment_shapes": [list(s.shape) for word in parsed["segments"] for s in word][:5],
                      "fixations_field": parsed["fixations"].tolist()}
        onsets, durations, method_counts = frp.locate_sentence(raw, parsed["segments"])
        for key, value in method_counts.items():
            counts[key] = counts.get(key, 0) + value
        x = frp.preprocess_sentence(raw)
        features, epochs, word_timing = frp.epoch_words(x, onsets, durations)
        if len(epochs):
            sums = epochs.sum(axis=0, dtype=np.float64) if sums is None else sums + epochs.sum(axis=0)
            n_epochs += len(epochs)
        ffd = parsed["times"]["FFD"]
        ok = np.isfinite(ffd) & np.isfinite(word_timing["first_fix_ms"])
        ffd_pairs += list(zip(ffd[ok].tolist(), word_timing["first_fix_ms"][ok].tolist()))
        trials.append({"sample_id": sample_id(subject, sentence_id), "subject_id": subject,
                       "sentence_id": int(sentence_id), "label": int(label), "words": parsed["words"],
                       "features": features, "fixations": parsed["fixations"],
                       **{f"{name.lower()}_ms": parsed["times"][name] for name in frp.READING_MEASURES}})
        timing.append(word_timing)
        if limit and len(trials) >= limit:
            break
    located = counts.get("exact", 0) + counts.get("offset", 0)
    attempted = sum(v for k, v in counts.items() if k not in ("no_data", "raw_unreadable"))
    ffd = np.array(ffd_pairs)
    summary = {"subject": subject, "sentences": len(trials), "fixation_counts": counts,
               "match_rate": located / attempted if attempted else 0.0,
               "words_with_epoch": int(sum(np.isfinite(t["features"][:, 0, 0]).sum() for t in trials)),
               "words_fixated": int(sum((t["fixations"] > 0).sum() for t in trials)),
               "ffd_vs_segment_r": float(np.corrcoef(ffd[:, 0], ffd[:, 1])[0, 1]) if len(ffd) > 2 else None,
               "ffd_minus_segment_ms_median": float(np.median(ffd[:, 0] - ffd[:, 1])) if len(ffd) else None,
               "layout_first_sentence": layout}
    return trials, timing, sums, n_epochs, summary


def save_timing(out_dir, subject, trials, timing):
    offsets = np.cumsum([0] + [len(t["words"]) for t in trials])
    arrays = {"offsets": offsets.astype(np.int64)}
    for key in ("onset_ms", "first_fix_ms", "next_fix_ms"):
        arrays[key] = (np.concatenate([t[key] for t in timing]) if timing else np.zeros(0)).astype(np.float32)
    os.makedirs(os.path.join(out_dir, "timing"), exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "timing", f"{subject}.npz"), **arrays)


def main():
    args = parse_args()
    lookup = label_lookup(args.labels_csv)
    paths = subject_files(args.mat_dir, args.subjects)
    if not paths:
        raise SystemExit(f"no results*_SR.mat files in {args.mat_dir}")
    if args.probe:
        trials, _, sums, n_epochs, summary = process_subject(paths[0], lookup, limit=args.probe)
        print(json.dumps(summary, indent=1))
        print(f"{n_epochs} epochs from {len(trials)} sentences")
        return
    os.makedirs(os.path.join(args.out_dir, "grand_average"), exist_ok=True)
    report_path = os.path.join(args.out_dir, "frp_extraction.json")
    report = json.load(open(report_path)) if os.path.exists(report_path) else {"subjects": {}}
    started = time.time()
    for index, path in enumerate(paths):
        subject = subject_from_path(path)
        if os.path.exists(os.path.join(args.out_dir, f"{subject}.npz")) and not args.overwrite:
            print(f"skip {subject}: already extracted")
            continue
        trials, timing, sums, n_epochs, summary = process_subject(path, lookup)
        print(f"{subject}: {summary['sentences']} sentences, match rate {summary['match_rate']:.3f}, "
              f"{summary['words_with_epoch']} words with an epoch, FFD vs segment r = {summary['ffd_vs_segment_r']}")
        if index == 0 and summary["match_rate"] < args.min_match_rate:
            json.dump(summary, open(os.path.join(args.out_dir, "frp_probe_failed.json"), "w"), indent=1)
            raise SystemExit(f"only {summary['match_rate']:.1%} of fixations were located in rawData; "
                             f"see frp_probe_failed.json (counts: {summary['fixation_counts']})")
        save_subject(args.out_dir, subject, trials)
        save_timing(args.out_dir, subject, trials, timing)
        if sums is not None:
            np.savez_compressed(os.path.join(args.out_dir, "grand_average", f"{subject}.npz"),
                                sums=sums.astype(np.float32), n=n_epochs)
        report["subjects"][subject] = summary
        report.update({"windows_ms": frp.WINDOWS_MS, "window_names": frp.WINDOW_NAMES, "sfreq": frp.SFREQ,
                       "pre_ms": frp.PRE_MS, "post_ms": frp.POST_MS})
        json.dump(report, open(report_path, "w"), indent=1)
    print(f"done in {(time.time() - started) / 60:.1f} min; report: {report_path}")


if __name__ == "__main__":
    main()
