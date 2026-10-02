"""Download ZuCo 1.0 NR or TSR subject files from OSF one at a time and extract word-level EEG.

The result is a word-EEG cache in the same format as the SR one (TRT band power per word), with
unlabelled sentences (label 99, text-derived sentence ids). EEG-to-text uses it as extra English
training data. Each .mat file (0.4-1.4 GB) is deleted after extraction, so the disk never holds more
than one; finished subjects are skipped, so the script can simply be re-run after a disconnect.
OSF often answers 502/403 for a while (rate limiting): each file is retried with growing waits, a file
that still fails is skipped, and skipped files are tried again at the end.

Source: ZuCo 1.0, https://osf.io/q3zws/ ("task2 - NR" and "task3 - TSR", "Matlab files").
"""

import argparse
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)

import pandas as pd  # noqa: E402

from src.fusion.word_eeg import extract_subject, save_subject, write_manifest  # noqa: E402
from src.labels import unlabelled_match  # noqa: E402
from src.progress import progress  # noqa: E402

OSF_FILES = {
    "NR": {"ZAB": "uzve3", "ZDM": "8xzbc", "ZDN": "pc9hk", "ZGW": "4b7dm", "ZJM": "5p4yd", "ZJN": "jn7x9",
           "ZJS": "qwzr3", "ZKB": "pfdh5", "ZKH": "9s6az", "ZKW": "m7afd", "ZMG": "cnakr", "ZPH": "gsf56"},
    "TSR": {"ZAB": "58mfb", "ZDM": "a5tdx", "ZDN": "u5mtw", "ZGW": "e7agv", "ZJM": "83vjz", "ZJN": "cukde",
            "ZJS": "np5jf", "ZKB": "rp5vc", "ZKH": "p2s9g", "ZKW": "q5kp7", "ZMG": "dywc2", "ZPH": "963s7"},
}


def download(url, path, attempts=6):
    request = urllib.request.Request(url, headers={"User-Agent": "zuco-multimodal-sentiment research download"})
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=120) as response, open(path + ".part", "wb") as out:
                total = int(response.headers.get("Content-Length") or 0) or None
                bar = progress(total=total, unit="B", unit_scale=True, desc=os.path.basename(path))
                while chunk := response.read(1 << 22):
                    out.write(chunk)
                    bar.update(len(chunk))
                bar.close()
            if total and os.path.getsize(path + ".part") != total:
                raise IOError(f"incomplete download ({os.path.getsize(path + '.part'):,} of {total:,} bytes)")
            os.replace(path + ".part", path)
            return
        except Exception as error:  # network errors come in many types
            if attempt == attempts:
                raise
            wait = min(30 * 2 ** (attempt - 1), 480)
            print(f"  download attempt {attempt}/{attempts} failed: {error}; retrying in {wait} s")
            time.sleep(wait)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, choices=sorted(OSF_FILES))
    parser.add_argument("--out-dir", required=True, help="word-EEG cache for this task (e.g. on Drive)")
    parser.add_argument("--tmp-dir", default="/content/zuco_download", help="local disk for one .mat file")
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--measure", default="TRT", choices=["TRT", "FFD", "GD", "GPT", "SFD"])
    parser.add_argument("--rounds", type=int, default=3, help="passes over the subjects whose download failed")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.tmp_dir, exist_ok=True)
    records_path = os.path.join(args.out_dir, "word_eeg_trials.csv")
    records = pd.read_csv(records_path).to_dict("records") if os.path.exists(records_path) else []
    pending = [s for s in (args.subjects or sorted(OSF_FILES[args.task]))
               if not os.path.exists(os.path.join(args.out_dir, f"{s}.npz"))]
    print(f"{args.task}: {len(pending)} subjects to fetch: {' '.join(pending) or 'none'}")
    for round_ in range(1, args.rounds + 1):
        if not pending:
            break
        if round_ > 1:
            print(f"round {round_}: retrying {' '.join(pending)} after a 5 min pause")
            time.sleep(300)
        failed = []
        for subject in pending:
            started = time.time()
            path = os.path.join(args.tmp_dir, f"results{subject}_{args.task}.mat")
            try:
                if not os.path.exists(path):
                    download(f"https://osf.io/download/{OSF_FILES[args.task][subject]}/", path)
            except Exception as error:
                print(f"{subject}: download failed ({error}); will retry later")
                failed.append(subject)
                continue
            trials, subject_records = extract_subject(path, None, unlabelled_match, args.measure)
            save_subject(args.out_dir, subject, trials)
            records = [r for r in records if r["subject_id"] != subject] + subject_records
            write_manifest(args.out_dir, records, args.measure)
            os.remove(path)
            words = sum(len(t["words"]) for t in trials)
            print(f"{subject}: {len(trials)} sentences, {words} words ({(time.time() - started) / 60:.1f} min)")
        pending = failed
    if records:
        print(write_manifest(args.out_dir, records, args.measure))
    if pending:
        raise SystemExit(f"{args.task}: still missing {' '.join(pending)} (OSF refused the download); "
                         "run the cell again later, finished subjects are kept")


if __name__ == "__main__":
    main()
