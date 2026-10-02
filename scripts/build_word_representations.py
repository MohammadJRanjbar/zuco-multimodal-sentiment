"""Build per-word EEG representations from raw word epochs, each as a word-EEG cache under --out-root/<name>.

Representations (see src/wordinfo/representations.py): raw, cbramod, neurolm, eye_tracking, and, for
comparison on exactly the same readers and words, the existing trt (band power over total reading time)
and frp (fixation-related window means) caches restricted to the words that have an epoch. Resumable
per representation and subject.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)

import numpy as np  # noqa: E402

from src.fusion.word_eeg import save_subject  # noqa: E402
from src.progress import progress  # noqa: E402
from src.wordinfo import representations as reps  # noqa: E402
from src.wordinfo.epochs import load_subject_epochs, subject_epoch_files  # noqa: E402

MODEL_REPS = ("raw", "cbramod", "neurolm", "eye_tracking")
CACHE_REPS = ("trt", "frp")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs-dir", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--representations", nargs="+", default=["raw", "cbramod", "eye_tracking", "trt", "frp"],
                        choices=MODEL_REPS + CACHE_REPS)
    parser.add_argument("--trt-dir", help="word-EEG cache with TRT band power (for 'trt')")
    parser.add_argument("--frp-dir", help="word-EEG cache with fixation-related window means (for 'frp')")
    parser.add_argument("--neurolm-dir", help="NeuroLM clone (for 'neurolm')")
    parser.add_argument("--neurolm-checkpoint", help="NeuroLM-B.pt (for 'neurolm')")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def model_trials(trials, n_words, epoch_word, values):
    """Trial dicts whose features are ``values`` at the words with an epoch and NaN elsewhere."""
    features = np.full((n_words, values.shape[1]), np.nan, np.float32)
    features[epoch_word] = values
    return [{**t, "features": features[t["word_offset"]:t["word_offset"] + len(t["words"])]} for t in trials]


def cache_trials(cache_path, trials, n_words, epoch_word):
    """The cache's trials of this subject, with NaN features for words that have no epoch."""
    has_epoch = np.zeros(n_words, bool)
    has_epoch[epoch_word] = True
    by_sentence = {t["sentence_id"]: t for t in trials}
    with np.load(cache_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    out, mismatched = [], 0
    offsets = arrays["offsets"]
    for i, sid in enumerate(arrays["sentence_id"]):
        epoch_trial = by_sentence.get(int(sid))
        start, stop = int(offsets[i]), int(offsets[i + 1])
        if epoch_trial is None or len(epoch_trial["words"]) != stop - start:
            mismatched += epoch_trial is not None
            continue
        mask = has_epoch[epoch_trial["word_offset"]:epoch_trial["word_offset"] + stop - start]
        features = arrays["features"][start:stop].astype(np.float32).copy()
        features[~mask] = np.nan
        out.append({**epoch_trial, "features": features})
    return out, mismatched


def main():
    args = parse_args()
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    files = subject_epoch_files(args.epochs_dir)
    if not files:
        raise SystemExit(f"no word epochs in {args.epochs_dir}; run scripts/extract_word_epochs.py first")
    extractors = {}
    summary_path = os.path.join(args.out_root, "representations.json")
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
    started = time.time()
    for name in args.representations:
        out_dir = os.path.join(args.out_root, name)
        todo = [p for p in files if args.overwrite or not os.path.exists(os.path.join(out_dir, os.path.basename(p)))]
        if not todo:
            print(f"{name}: done earlier")
            continue
        if name == "cbramod":
            extractors[name] = reps.CBraModFeatures(device=device, batch_size=args.batch_size)
        elif name == "neurolm":
            if not (args.neurolm_dir and args.neurolm_checkpoint):
                raise SystemExit("'neurolm' needs --neurolm-dir and --neurolm-checkpoint")
            extractors[name] = reps.NeuroLMFeatures(args.neurolm_dir, args.neurolm_checkpoint, device=device)
        elif name in CACHE_REPS:
            source = args.trt_dir if name == "trt" else args.frp_dir
            if not source or not os.path.isdir(source):
                raise SystemExit(f"'{name}' needs --{name}-dir (got {source!r})")
        info = {"subjects": {}}
        for path in progress(todo, desc=name, unit="subject"):
            subject = os.path.basename(path)[:-4]
            arrays, trials = load_subject_epochs(path)
            n_words, epoch_word = len(arrays["words"]), arrays["epoch_word"]
            if name in CACHE_REPS:
                cache_path = os.path.join(source, f"{subject}.npz")
                if not os.path.exists(cache_path):
                    print(f"  {name}: no {subject} in {source}; skipped")
                    continue
                out_trials, mismatched = cache_trials(cache_path, trials, n_words, epoch_word)
                info["subjects"][subject] = {"trials": len(out_trials), "word_count_mismatches": mismatched}
            else:
                if name == "raw":
                    values = reps.raw_timecourse(arrays["epochs"])
                elif name == "eye_tracking":
                    values = reps.eye_tracking(arrays, epoch_word)
                else:
                    values = extractors[name](arrays["epochs"])
                out_trials = model_trials(trials, n_words, epoch_word, values)
                info["subjects"][subject] = {"trials": len(out_trials), "words_with_epoch": int(len(epoch_word)),
                                             "dim": int(values.shape[1])}
            save_subject(out_dir, subject, out_trials)
        summary[name] = {**summary.get(name, {}), **info}
        os.makedirs(args.out_root, exist_ok=True)
        json.dump(summary, open(summary_path, "w"), indent=1)
        print(f"{name}: {info['subjects']}")
    print(f"done in {(time.time() - started) / 60:.1f} min; summary {summary_path}")


if __name__ == "__main__":
    main()
