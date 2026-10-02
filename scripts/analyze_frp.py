"""Fixation-related potentials: timing check, N400 regression, sentiment terms.

1. Grand-average FRP per electrode group. Correct onsets give an occipital
   positive peak (lambda/P1) about 80-130 ms after fixation onset; its
   presence and latency check the onset alignment.
2. N400 regression at the trial level (reader x word), with reader fixed
   effects. The dependent variable is the mean centro-parietal amplitude
   300-500 ms after fixation onset. Predictors (standardized):
   - surprisal (GPT-2), the positive control: higher surprisal should give a
     more negative N400;
   - Zipf frequency, length, relative position, content word;
   - log first-fixation duration and log time to the next fixation (overlap
     with the next fixation's response);
   - word valence and |valence| for VADER-lexicon words (0 otherwise, with an
     in-lexicon indicator).
   CIs come from a bootstrap over sentences.
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.diagnostics.targets import build_word_table, lm_surprisal, valence_lexicon  # noqa: E402
from src.followup import frp  # noqa: E402
from src.fusion.word_eeg import load_word_eeg  # noqa: E402

PREDICTORS = ["surprisal", "zipf", "length", "relative_position", "is_content", "log_first_fix", "log_next_fix",
              "in_lexicon", "valence", "abs_valence"]
SENTIMENT_TERMS = ["valence", "abs_valence"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frp-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--surprisal-model", default="gpt2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def attach_timing(trials, frp_dir):
    by_subject = {}
    for trial in trials:
        by_subject.setdefault(trial["subject_id"], []).append(trial)
    for subject, subject_trials in by_subject.items():
        path = os.path.join(frp_dir, "timing", f"{subject}.npz")
        timing = dict(np.load(path)) if os.path.exists(path) else None
        start = 0
        for trial in subject_trials:
            n = len(trial["words"])
            for key in ("onset_ms", "first_fix_ms", "next_fix_ms"):
                trial[key] = timing[key][start:start + n] if timing is not None else np.full(n, np.nan)
            start += n
    return trials


def grand_average(frp_dir, labels):
    files = sorted(glob.glob(os.path.join(frp_dir, "grand_average", "*.npz")))
    if not files:
        return None
    per_subject = []
    for path in files:
        data = np.load(path)
        if int(data["n"]) > 0:
            per_subject.append(data["sums"] / float(data["n"]))
    mean = np.mean(per_subject, axis=0)  # [channels, samples]
    times = (np.arange(mean.shape[1]) - frp.PRE_MS * frp.SFREQ / 1000) * 1000 / frp.SFREQ
    groups = {"occipital": frp.channel_indices(frp.OCCIPITAL, labels),
              "centro-parietal": frp.channel_indices(frp.CENTRO_PARIETAL, labels)}
    curves = {name: mean[idx].mean(axis=0) for name, idx in groups.items()}
    subject_curves = {name: np.array([s[idx].mean(axis=0) for s in per_subject]) for name, idx in groups.items()}
    window = (times >= 50) & (times <= 200)
    peak = int(np.argmax(np.where(window, curves["occipital"], -np.inf)))
    n400 = (times >= 300) & (times < 500)
    return {"times": times, "curves": curves, "subject_curves": subject_curves, "n_subjects": len(per_subject),
            "lambda_peak_ms": float(times[peak]), "lambda_peak_uv": float(curves["occipital"][peak]),
            "occipital_baseline_sd": float(curves["occipital"][times < 0].std()),
            "n400_cp_mean_uv": float(curves["centro-parietal"][n400].mean())}


def plot_grand_average(ga, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharex=True)
    for ax, name in zip(axes, ("occipital", "centro-parietal")):
        for curve in ga["subject_curves"][name]:
            ax.plot(ga["times"], curve, color="0.8", lw=0.6)
        ax.plot(ga["times"], ga["curves"][name], color="C0", lw=2, label=f"mean of {ga['n_subjects']} readers")
        ax.axvline(0, color="k", lw=0.8)
        ax.axhline(0, color="k", lw=0.5)
        if name == "centro-parietal":
            ax.axvspan(300, 500, color="C1", alpha=0.12, label="N400 window")
        ax.set_title(f"{name} electrodes")
        ax.set_xlabel("ms from fixation onset")
        ax.set_ylabel("microvolts")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def trial_table(trials, words, labels):
    roi = frp.channel_indices(frp.CENTRO_PARIETAL, labels)
    occipital = frp.channel_indices(frp.OCCIPITAL, labels)
    rows = []
    for trial in trials:
        n400 = np.asarray(trial["features"])[:, frp.N400_INDEX][:, roi].mean(axis=1)
        p1 = np.asarray(trial["features"])[:, 1][:, occipital].mean(axis=1)
        for w in range(len(trial["words"])):
            if np.isfinite(n400[w]):
                rows.append({"reader": trial["subject_id"], "sentence_id": trial["sentence_id"], "word_index": w,
                             "n400": float(n400[w]), "p1_100_200": float(p1[w]),
                             "first_fix_ms": float(trial["first_fix_ms"][w]),
                             "next_fix_ms": float(trial["next_fix_ms"][w])})
    table = pd.DataFrame(rows).merge(words, on=["sentence_id", "word_index"], how="left")
    table["log_first_fix"] = np.log(table["first_fix_ms"].clip(lower=1))
    table["log_next_fix"] = np.log(table["next_fix_ms"].clip(lower=1))
    for column in ("valence", "abs_valence"):
        table[column] = table[column].fillna(0.0)
    return table


def regression(table, outcome, predictors, n_boot, seed):
    """OLS with reader fixed effects (within-reader demeaning); sentence-bootstrap CIs."""
    data = table.dropna(subset=[outcome] + predictors).copy()
    constant = [p for p in predictors if data[p].nunique() < 2]
    predictors = [p for p in predictors if p not in constant]
    X = data[predictors].to_numpy(dtype=np.float64)
    X = (X - X.mean(axis=0)) / np.where(X.std(axis=0) > 0, X.std(axis=0), 1.0)
    y = np.array(data[outcome], dtype=np.float64)
    readers = pd.factorize(data["reader"])[0]
    for r in np.unique(readers):
        rows = readers == r
        X[rows] -= X[rows].mean(axis=0)
        y[rows] -= y[rows].mean()
    codes, sentences = pd.factorize(data["sentence_id"])
    p = X.shape[1]
    xtx = np.zeros((len(sentences), p, p))
    xty = np.zeros((len(sentences), p))
    np.add.at(xtx, codes, X[:, :, None] * X[:, None, :])
    np.add.at(xty, codes, X * y[:, None])
    beta = np.linalg.solve(xtx.sum(0), xty.sum(0))
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(sentences), np.full(len(sentences), 1 / len(sentences)), size=n_boot)
    boots = np.array([np.linalg.solve(np.tensordot(w, xtx, 1), w @ xty) for w in weights])
    rows = []
    for i, name in enumerate(predictors):
        low, high = np.percentile(boots[:, i], [2.5, 97.5])
        rows.append({"predictor": name, "beta_uv_per_sd": float(beta[i]), "ci_low": float(low), "ci_high": float(high),
                     "excludes_zero": bool(low > 0 or high < 0)})
    for name in constant:
        rows.append({"predictor": name, "beta_uv_per_sd": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                     "excludes_zero": False})
    return pd.DataFrame(rows), int(len(data)), int(len(sentences))


def write_report(path, ga, coefficients, n_rows, n_sentences, extraction):
    lines = ["# Fixation-related potentials (ZuCo)", ""]
    if extraction:
        rates = [s["match_rate"] for s in extraction["subjects"].values()]
        words = sum(s["words_with_epoch"] for s in extraction["subjects"].values())
        corr = [s["ffd_vs_segment_r"] for s in extraction["subjects"].values() if s["ffd_vs_segment_r"] is not None]
        lines += [f"Fixations located in the sentence EEG: {np.mean(rates):.1%} on average "
                  f"(range {min(rates):.1%}-{max(rates):.1%}) over {len(rates)} readers; {words:,} word epochs. "
                  f"Segment length vs ZuCo first-fixation duration: r = {np.mean(corr):.3f} (should be close to 1).", ""]
    if ga:
        ok = 60 <= ga["lambda_peak_ms"] <= 160 and ga["lambda_peak_uv"] > 3 * ga["occipital_baseline_sd"]
        lines += ["## Timing check", "",
                  f"Occipital peak at {ga['lambda_peak_ms']:.0f} ms ({ga['lambda_peak_uv']:.2f} microvolts; "
                  f"baseline SD {ga['occipital_baseline_sd']:.2f}). "
                  + ("**Pass:** a clear lambda/P1 response at the expected latency, so onsets are aligned."
                     if ok else "**Check:** no clear lambda/P1 response at 60-160 ms; onsets may be misaligned."),
                  f"Centro-parietal mean amplitude 300-500 ms: {ga['n400_cp_mean_uv']:.2f} microvolts. "
                  "Plot: `plots/frp_grand_average.png`.", ""]
    lines += ["## N400 regression (trial level, reader fixed effects)", "",
              f"{n_rows:,} reader-word observations from {n_sentences} sentences. Coefficients are microvolts per "
              "standard deviation of the predictor; 95% CIs from a sentence bootstrap.", "",
              "| predictor | beta | 95% CI |", "|---|---:|---|"]
    for _, row in coefficients.iterrows():
        mark = " *" if row["excludes_zero"] else ""
        lines.append(f"| {row['predictor']} | {row['beta_uv_per_sd']:.3f}{mark} | "
                     f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}] |")
    surprisal = coefficients.set_index("predictor").loc["surprisal"]
    if not np.isfinite(surprisal["beta_uv_per_sd"]):
        surprisal = surprisal.copy()
        surprisal["excludes_zero"] = False
    sentiment = coefficients[coefficients["predictor"].isin(SENTIMENT_TERMS)]
    lines += ["", "\\* CI excludes 0.", "",
              "* **Positive control (surprisal):** "
              + ("more surprising words give a more negative N400, as expected; the pipeline detects a known "
                 "semantic effect." if surprisal["excludes_zero"] and surprisal["beta_uv_per_sd"] < 0 else
                 "no expected negative surprisal effect; treat the sentiment result below with caution."),
              "* **Sentiment terms (valence, |valence|):** "
              + ("at least one CI excludes 0 after the lexical and timing controls."
                 if sentiment["excludes_zero"].any() else
                 "no effect beyond the lexical and timing controls.")]
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    os.makedirs(os.path.join(args.out_dir, "plots"), exist_ok=True)
    labels = frp.zuco_channel_labels()
    trials = attach_timing(load_word_eeg(args.frp_dir), args.frp_dir)
    ga = grand_average(args.frp_dir, labels)
    if ga:
        plot_grand_average(ga, os.path.join(args.out_dir, "plots", "frp_grand_average.png"))
    sentences = {}
    for trial in trials:
        sentences.setdefault(trial["sentence_id"], trial["words"])
    ids = sorted(sentences)
    cache = os.path.join(args.out_dir, f"surprisal_{args.surprisal_model.replace('/', '_')}.json")
    if os.path.exists(cache):
        surprisal = {int(k): np.array(v) for k, v in json.load(open(cache)).items()}
    else:
        values = lm_surprisal([sentences[i] for i in ids], args.surprisal_model, args.device)
        surprisal = dict(zip(ids, values))
        json.dump({str(k): v.tolist() for k, v in surprisal.items()}, open(cache, "w"))
    words = build_word_table(trials, lexicon=valence_lexicon(), surprisal=surprisal)
    table = trial_table(trials, words, labels)
    coefficients, n_rows, n_sentences = regression(table, "n400", PREDICTORS, args.n_boot, args.seed)
    coefficients.to_csv(os.path.join(args.out_dir, "n400_regression.csv"), index=False)
    extraction_path = os.path.join(args.frp_dir, "frp_extraction.json")
    extraction = json.load(open(extraction_path)) if os.path.exists(extraction_path) else None
    summary = {"n_rows": n_rows, "n_sentences": n_sentences, "coefficients": coefficients.to_dict("records")}
    if ga:
        summary["grand_average"] = {k: v for k, v in ga.items() if k not in ("times", "curves", "subject_curves")}
        np.savez_compressed(os.path.join(args.out_dir, "frp_grand_average_curves.npz"), times=ga["times"],
                            **{k.replace("-", "_"): v for k, v in ga["curves"].items()})
    json.dump(summary, open(os.path.join(args.out_dir, "frp_report.json"), "w"), indent=1)
    report = os.path.join(args.out_dir, "frp_report.md")
    write_report(report, ga, coefficients, n_rows, n_sentences, extraction)
    print(open(report).read())


if __name__ == "__main__":
    main()
