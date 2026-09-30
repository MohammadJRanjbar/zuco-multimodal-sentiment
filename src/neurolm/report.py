"""Predeclared decision rules and the markdown results report.

The rules are fixed in ``configs/neurolm_probe.yaml`` (``decision``) before any
result is seen:

* a protocol is **above chance** when, for the primary feature and classifier,
  the sentence-level label-permutation p-value is below ``alpha``, the
  seed-averaged macro-F1 exceeds the permutation-null mean by at least
  ``min_margin_macro_f1``, and at least ``min_seed_agreement`` seeds lie above
  the null mean;
* NeuroLM **beats handcrafted** when the paired sentence-cluster interval of
  the macro-F1 difference (same classifier, same trials) excludes zero and the
  paired permutation p-value is below ``alpha``;
* fusion is only considered after fine-tuning, and only if the frozen joint
  result clears ``fusion_margin_macro_f1``.
"""

import numpy as np

PROTOCOL_NAMES = {
    "joint": "unseen subject + unseen sentence (primary)",
    "text": "unseen sentence, seen subjects",
    "subject": "unseen subject, seen sentences",
}


def _fmt(x, digits=4):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{digits}f}"


def above_chance(sanity_block, decision):
    observed = sanity_block["observed"]["macro_f1"]
    null = sanity_block["label_permutation_sentence"]
    seeds_above = int(np.sum(np.asarray(observed["values"]) > null["null_mean"]))
    margin = observed["mean"] - null["null_mean"]
    passed = (
        null["p_value"] < decision["alpha"]
        and margin >= decision["min_margin_macro_f1"]
        and seeds_above >= min(decision["min_seed_agreement"], len(observed["values"]))
    )
    return {
        "passed": bool(passed),
        "observed": observed["mean"],
        "observed_std": observed["std"],
        "null_mean": null["null_mean"],
        "null_q95": null["null_q95"],
        "p_value": null["p_value"],
        "margin": margin,
        "seeds_above_null": seeds_above,
        "n_seeds": len(observed["values"]),
        "n_permutations": null["n_permutations"],
    }


def beats_handcrafted(paired_rows, protocol, primary_feature, decision, classifier="logreg"):
    for row in paired_rows:
        if (row["protocol"] == protocol and row["candidate"] == f"{primary_feature}__{classifier}"
                and row["baseline"] == f"handcrafted__{classifier}" and row["cluster"] == "sentence_id"):
            low = row["ci95"][0]
            return {
                "passed": bool(low > 0 and row["p_value_two_sided"] < decision["alpha"]),
                "delta": row["delta_macro_f1"],
                "ci95": row["ci95"],
                "p_value": row["p_value_two_sided"],
            }
    return None


def decide(probe, sanity, config):
    decision = config["decision"]
    primary = config["probe"]["primary_feature"]
    verdicts = {p: above_chance(block, decision) for p, block in sanity["protocols"].items()}
    joint = verdicts.get("joint")
    hand = {p: beats_handcrafted(probe["paired"], p, primary, decision) for p in verdicts}
    answers = {}
    answers["Q1"] = (joint["passed"] if joint else None,
                     "frozen EEG above chance under the primary joint protocol" if joint else "joint protocol not run")
    answers["Q2"] = (hand.get("joint", {}) or {}).get("passed") if hand.get("joint") else None
    answers["Q3"] = verdicts["subject"]["passed"] if "subject" in verdicts else None
    answers["Q4"] = verdicts["text"]["passed"] if "text" in verdicts else None
    answers["Q5"] = joint["passed"] if joint else None

    subject_id = sanity.get("subject_id", {})
    identity = subject_id.get(primary, {}).get("accuracy", {}).get("mean")
    chance = subject_id.get("chance_accuracy")
    strong_identity = identity is not None and chance is not None and identity > 3 * chance
    shuffled = sanity["protocols"].get("joint", {}).get("shuffled_association", {}).get("macro_f1", {}).get("mean")
    duration = sanity["protocols"].get("joint", {}).get("duration_only", {}).get("macro_f1", {}).get("mean")
    answers["Q6"] = {
        "subject_accuracy": identity,
        "subject_chance": chance,
        "strong_subject_identity": strong_identity,
        "sentiment_above_chance": joint["passed"] if joint else None,
        "shuffled_association_macro_f1": shuffled,
        "duration_only_macro_f1": duration,
    }
    fusion_ok = bool(
        joint and joint["passed"] and joint["margin"] >= decision["fusion_margin_macro_f1"]
        and (shuffled is None or joint["observed"] - shuffled >= decision["min_margin_macro_f1"])
        and (duration is None or joint["observed"] - duration >= decision["min_margin_macro_f1"])
    )
    answers["Q7"] = fusion_ok
    if not joint or not joint["passed"]:
        recommendation = "stop"
    else:
        recommendation = "fine-tune encoder"
    return {"verdicts": verdicts, "handcrafted": hand, "answers": answers, "recommendation": recommendation}


def _ci(values):
    if not values:
        return "—"
    return f"[{values[0]:.3f}, {values[1]:.3f}]"


def _yes_no(value):
    return {True: "**Yes**", False: "**No**", None: "not evaluated"}[value]


def build_markdown(probe, sanity, config, mapping=None, inspection=None, manifest=None):
    result = decide(probe, sanity, config)
    answers = result["answers"]
    primary = config["probe"]["primary_feature"]
    lines = ["# Frozen NeuroLM/LaBraM EEG-only sentiment probe — results", ""]
    if manifest:
        ckpt = manifest.get("checkpoint", {})
        lines += [
            f"* Checkpoint: `{ckpt.get('name')}` from `{ckpt.get('repo')}` @ `{ckpt.get('revision')}` "
            f"(sha256 `{ckpt.get('sha256')}`)",
            f"* Primary view: `{manifest.get('view')}` with {len(manifest.get('channel_mapping', []))} channels; "
            f"primary feature `{primary}`, classifier `{config['probe']['primary_classifier']}`",
            f"* Trials: {manifest.get('trial_alignment')}",
            f"* Git: `{(manifest.get('git') or {}).get('commit')}` (dirty={(manifest.get('git') or {}).get('dirty')})",
            f"* Seeds: {manifest.get('seeds')}",
            "",
        ]
    if inspection:
        d = inspection.get("duration_s", {})
        lines += [f"Raw ZuCo: {inspection.get('valid_trials')} valid subject × sentence trials from "
                  f"{inspection.get('n_subject_files')} subjects; duration median {_fmt(d.get('median'), 2)} s "
                  f"(max {_fmt(d.get('max'), 2)} s).", ""]
    labels = sanity["label_distribution"]["trials"]
    lines += [
        "## Chance levels", "",
        f"* Majority class accuracy {_fmt(labels['majority_accuracy'])}, majority macro-F1 {_fmt(labels['majority_macro_f1'])}",
        f"* Uniform random: accuracy {_fmt(labels['uniform_random_expected_accuracy'])}, macro-F1 "
        f"{_fmt(labels['uniform_random_expected_macro_f1'])}; prior-matched random macro-F1 "
        f"{_fmt(labels['prior_matched_random_expected_macro_f1'])}; balanced-accuracy chance "
        f"{_fmt(labels['balanced_accuracy_chance'])}",
        "",
        "## Main results (macro-F1, mean ± std over seeds; 95% sentence-cluster CI)", "",
        "| protocol | feature | view | classifier | macro-F1 | accuracy | balanced acc. | CI (sentences) | CI (subjects) |",
        "|---|---|---|---|---|---:|---:|---|---|",
    ]
    for row in probe["rows"]:
        lines.append(
            f"| {row['protocol']} | {row['feature']} | {row['view']} | {row['classifier']} | "
            f"{_fmt(row['macro_f1_mean'])} ± {_fmt(row['macro_f1_std'])} | {_fmt(row['accuracy_mean'])} | "
            f"{_fmt(row['balanced_accuracy_mean'])} | {_ci(row.get('macro_f1_ci95_sent'))} | "
            f"{_ci(row.get('macro_f1_ci95_subj'))} |"
        )
    lines += ["", "### Secondary: sentence-level aggregation over test subjects", "",
              "Held-out probabilities of all test subjects are averaged per sentence (no extra training).", "",
              "| protocol | feature | view | classifier | sentence-level macro-F1 |", "|---|---|---|---|---|"]
    for row in probe["rows"]:
        if "sentence_aggregated_macro_f1_mean" in row:
            lines.append(f"| {row['protocol']} | {row['feature']} | {row['view']} | {row['classifier']} | "
                         f"{_fmt(row['sentence_aggregated_macro_f1_mean'])} ± {_fmt(row['sentence_aggregated_macro_f1_std'])} |")
    lines += ["", "### Per-class results of the primary feature (means over seeds)", "",
              "| protocol | classifier | class | precision | recall | F1 |", "|---|---|---|---:|---:|---:|"]
    for row in probe["rows"]:
        if row["feature"] != primary or row["view"] != config["probe"]["primary_view"]:
            continue
        for name in ("negative", "neutral", "positive"):
            lines.append(f"| {row['protocol']} | {row['classifier']} | {name} | {_fmt(row.get(f'precision_{name}'))} | "
                         f"{_fmt(row.get(f'recall_{name}'))} | {_fmt(row.get(f'f1_{name}'))} |")
    lines += ["", "Confusion matrices (rows = true negative/neutral/positive, columns = predicted; summed over seeds):", ""]
    for row in probe["rows"]:
        if row["feature"] in (primary, "handcrafted") and row["classifier"] == config["probe"]["primary_classifier"]:
            lines.append(f"* {row['protocol']} · {row['feature']} · {row['view']}: `{row.get('confusion_matrix_sum_over_seeds')}`")
    lines += ["", "## Above-chance tests (primary feature, sentence-level label permutation)", "",
              "| protocol | observed | null mean | null q95 | p | margin | seeds > null | verdict |",
              "|---|---:|---:|---:|---:|---:|---:|---|"]
    for protocol, v in result["verdicts"].items():
        lines.append(
            f"| {PROTOCOL_NAMES.get(protocol, protocol)} | {_fmt(v['observed'])} | {_fmt(v['null_mean'])} | "
            f"{_fmt(v['null_q95'])} | {_fmt(v['p_value'])} | {_fmt(v['margin'])} | {v['seeds_above_null']}/"
            f"{v['n_seeds']} | {'above chance' if v['passed'] else 'not above chance'} |"
        )
    lines += ["", "## Sanity checks", "",
              "| protocol | shuffled association | random Gaussian | duration only | trial-level null mean |",
              "|---|---:|---:|---:|---:|"]
    for protocol, block in sanity["protocols"].items():
        trial = block.get("label_permutation_trial", {})
        lines.append(
            f"| {protocol} | {_fmt(block['shuffled_association']['macro_f1']['mean'])} | "
            f"{_fmt(block['random_gaussian']['macro_f1']['mean'])} | {_fmt(block['duration_only']['macro_f1']['mean'])} | "
            f"{_fmt(trial.get('null_mean'))} |"
        )
    sid = sanity.get("subject_id", {})
    lines += ["", f"Subject-ID decoding ({sid.get('n_subjects')} subjects, chance accuracy "
                  f"{_fmt(sid.get('chance_accuracy'))}, unseen-sentence split):", ""]
    for name, block in sid.items():
        if isinstance(block, dict) and "accuracy" in block:
            lines.append(f"* `{name}`: accuracy {_fmt(block['accuracy']['mean'])}, macro-F1 {_fmt(block['macro_f1']['mean'])}")
    collapsed = {k: v["warnings"] for k, v in sanity.get("collapse", {}).items() if v["warnings"]}
    lines += ["", "Embedding collapse: " + ("none detected" if not collapsed else str(collapsed)), ""]
    if probe.get("paired"):
        lines += ["## NeuroLM vs handcrafted (paired, same trials and seeds)", "",
                  "| protocol | candidate | baseline | cluster | Δ macro-F1 | 95% CI | p |", "|---|---|---|---|---:|---|---:|"]
        for row in probe["paired"]:
            lines.append(f"| {row['protocol']} | {row['candidate']} | {row['baseline']} | {row['cluster']} | "
                         f"{_fmt(row['delta_macro_f1'])} | [{_fmt(row['ci95'][0])}, {_fmt(row['ci95'][1])}] | "
                         f"{_fmt(row['p_value_two_sided'])} |")
        lines.append("")
    q6 = answers["Q6"]
    lines += [
        "## Decision questions", "",
        f"**Q1. Does frozen NeuroLM/LaBraM EEG perform above chance on sentiment?** {_yes_no(answers['Q1'][0])} "
        "(primary joint protocol, predeclared rule).",
        "",
        f"**Q2. Does it outperform the handcrafted EEG representation?** {_yes_no(answers['Q2'])}"
        + (f" (joint, logreg: Δ = {_fmt(result['handcrafted']['joint']['delta'])}, CI "
           f"{[round(x, 4) for x in result['handcrafted']['joint']['ci95']]}, p = {_fmt(result['handcrafted']['joint']['p_value'])})"
           if result["handcrafted"].get("joint") else ""),
        "",
        f"**Q3. Does performance survive unseen-subject evaluation?** {_yes_no(answers['Q3'])}",
        "",
        f"**Q4. Does performance survive unseen-sentence evaluation?** {_yes_no(answers['Q4'])}",
        "",
        f"**Q5. Does performance survive unseen subject + unseen text?** {_yes_no(answers['Q5'])}",
        "",
        "**Q6. Is sentiment performance genuine, or mostly subject identity?** "
        f"Subject-ID accuracy {_fmt(q6['subject_accuracy'])} vs chance {_fmt(q6['subject_chance'])}"
        f" ({'strong' if q6['strong_subject_identity'] else 'weak'} subject identity); sentiment above chance: "
        f"{_yes_no(q6['sentiment_above_chance'])}; shuffled-association macro-F1 "
        f"{_fmt(q6['shuffled_association_macro_f1'])}; duration-only macro-F1 {_fmt(q6['duration_only_macro_f1'])}."
        + (" The representation carries substantial subject-specific information while sentiment is not "
           "decodable above chance." if q6["strong_subject_identity"] and not q6["sentiment_above_chance"] else ""),
        "",
        f"**Q7. Is the signal strong enough to justify EEG+text fusion?** {_yes_no(answers['Q7'])}",
        "",
        f"## Recommendation: **{result['recommendation']}**",
        "",
    ]
    if result["recommendation"] == "stop":
        lines.append(
            "Under this protocol the frozen pretrained EEG representation does not provide reliable sentiment "
            "information. Do not proceed to multimodal LoRA or fusion with this representation."
        )
    else:
        lines.append(
            "The frozen representation is above chance under the primary protocol. The next step is lightweight "
            "fine-tuning of the EEG encoder with the same splits and controls; fusion only if the fine-tuned "
            "EEG-only model becomes meaningfully predictive."
        )
    lines += ["", "*Generated by `scripts/build_neurolm_report.py`; verdicts follow the predeclared rules in "
                  "`configs/neurolm_probe.yaml` and must be read together with the tables above.*"]
    return "\n".join(lines) + "\n", result
