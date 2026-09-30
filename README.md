# ZuCo multimodal sentiment classification

Can sentence-level EEG features improve a strong text sentiment classifier?
This project combines LaBSE sentence representations with the classical EEG
features extracted from ZuCo Task 1. It keeps the text-only, EEG-only, fusion,
and control experiments in one evaluation pipeline so their scores are directly
comparable.

The main model uses LaBSE's `[CLS]` representation and a compact EEG set encoder.
The reusable cache contains 2,520 classical features per subject sentence. The
flat Cz reference channel is excluded during loading, leaving 104 electrodes x
24 feature families = 2,496 model inputs. A shared MLP encodes the electrodes,
attention pools them into one subject representation, and a masked mean combines
the available subjects for the sentence. The text and EEG representations are
then fused for ternary sentiment classification.

## Experiment setups

| setup | text branch | EEG branch | purpose |
| --- | --- | --- | --- |
| `text_frozen` | frozen LaBSE + CLS | no | frozen text baseline |
| `text_finetune` | fine-tuned LaBSE + CLS | no | primary text baseline |
| `eeg_only` | no | aligned EEG | EEG-only baseline |
| `concat_finetune` | fine-tuned | aligned EEG | late hidden-state concatenation |
| `gated_finetune` | fine-tuned | aligned EEG | gated residual fusion |
| `gated_shuffled_finetune` | fine-tuned | shuffled inside each split | alignment control |
| `gated_noise_finetune` | fine-tuned | matched random noise | regularization control |
| `gated_zero_finetune` | fine-tuned | contribution forced to zero | architecture/optimization control |

Frozen variants of both fusion models are also available from the command line.
The gated model starts with a small EEG contribution, so it behaves approximately
like the text model until the validation data supports using the second modality.
The aligned, shuffled, noise, and zero gated variants reset to the same task-head
initialization for each seed and fold. Their saved initialization fingerprints
are checked during reporting.

## Evaluation protocol

The 400 unique sentences are the units of evaluation. The outer stratified folds
hold out complete sentences; every subject's EEG recording of a sentence follows
that sentence into the same train, validation, or test partition. The test fold
is evaluated once, after the best epoch has been selected by validation macro-F1.

All EEG preprocessing is fitted inside each fold using training sentences only:

1. median imputation;
2. per-feature standardization;
3. transformation of validation and test EEG with the stored training values.

The shuffled-EEG control permutes recordings independently inside train,
validation, and test. It destroys text-EEG alignment without moving any EEG row
across a split boundary.

Accuracy, macro-F1, weighted F1, per-class F1, confusion matrices, fold histories,
and out-of-fold predictions are saved. Fusion models are compared with the
matching text baseline using a paired sentence-level bootstrap interval.

Five folds make every sentence a held-out test example exactly once within one
seed. Three seeds repeat the complete five-fold procedure with different split
and initialization randomness. Bootstrap resampling is a separate reporting
step: it repeatedly resamples the 400 paired sentence predictions with
replacement and recalculates the candidate-minus-baseline macro-F1 difference.
The current report averages the three seed-specific differences and interval
endpoints. It measures sentence-sampling uncertainty for these trained runs; it
is not a formal interval over arbitrary seeds, unseen readers, or a new dataset.

For gated runs, each held-out sentence also stores:

- the text embedding norm;
- the raw EEG embedding norm;
- the candidate and effective gated EEG-contribution norms;
- logits with and without the EEG contribution.

Every fold stores the full final gate vector, diagnostic summaries, the explicit
initialization seed, and a task-module initialization fingerprint. Reporting
also creates `diagnostics.*`, `control_comparisons.*`, and
`diagnostic_metadata.json`.

## Classical feature cache

`extract_features.py` reads the original `results*_SR.mat` files and writes one
compressed `.npz` per subject. Each EEG row contains:

- 16 raw-signal statistics for each of 105 electrodes;
- ZuCo's eight sentence-level band means (`t1`, `t2`, `a1`, `a2`, `b1`, `b2`,
  `g1`, `g2`) for each electrode.

The default line-length feature is mean absolute successive difference rather
than cumulative path length. This removes its direct dependence on the number of
time samples in a sentence. Pass `--line-length sum` to reproduce the earlier
cumulative definition as a sensitivity analysis.

The raw files, sentence labels, and reusable artifacts belong on Drive rather
than in the repository. The Colab notebook reads the fixed labels CSV from the
main Thesis data folder and the subject files from its `zuco_og_raw` subfolder.
Extracted EEG features are stored under the project's `CachedArtifacts` folder,
so extraction is performed once and reused.

## Colab

Open `notebooks/zuco_multimodal_colab.ipynb` and run all four sections from top
to bottom. A CPU runtime is sufficient. The current notebook mounts Drive,
updates the repository, verifies the saved v2 results and feature cache, runs
the no-training Step 1 analysis, and displays its findings and plots. It does
not install packages, load LaBSE, or repeat completed training.

The earlier 15-section v1/v2 training runner is retained at
`notebooks/archive/zuco_v1_v2_training_colab.ipynb` for reproducibility. It is
not part of the current workflow and should only be used if the completed model
runs genuinely need to be reproduced from the beginning.

## Command-line use

Install the dependencies:

```bash
pip install -r requirements.txt
```

Build the reusable feature cache:

```bash
python extract_features.py \
  --mat-dir /path/to/zuco_og_raw \
  --labels-csv /path/to/Thesis/Data/zuco_sentiment_labels_task1_fixed.csv \
  --out-dir /path/to/drive/zuco_classical_features/classical_v2_normalized
```

Validate the cache:

```bash
python inspect_data.py \
  --labels-csv /path/to/Thesis/Data/zuco_sentiment_labels_task1_fixed.csv \
  --features-dir /path/to/drive/zuco_classical_features/classical_v2_normalized
```

Run the default three-seed suite:

```bash
python run.py \
  --labels-csv /path/to/Thesis/Data/zuco_sentiment_labels_task1_fixed.csv \
  --features-dir /path/to/drive/zuco_classical_features/classical_v2_normalized \
  --results-base /path/to/drive/zuco_multimodal_results \
  --run-tag v1_full
```

A narrower experiment is selected with `--setups` and `--seeds`:

```bash
python run.py \
  --labels-csv /path/to/Thesis/Data/zuco_sentiment_labels_task1_fixed.csv \
  --features-dir /path/to/features \
  --results-base /path/to/results \
  --run-tag v1_gated \
  --setups text_finetune gated_finetune gated_shuffled_finetune \
  --seeds 42 52 62
```

## Drive result layout

Reusable artifacts and run outputs are kept separate:

```text
MyDrive/Thesis/
  CachedArtifacts/zuco_multimodal_sentiment/
    eeg_features/classical_v2_normalized_line_length/
    models/LaBSE/
  Results/zuco_multimodal_sentiment/
```

Every setup and seed is saved immediately and independently:

```text
<results-base>/
  v1_full/
    run_manifest.json
    text_finetune/
      seed_42.json
      seed_52.json
      seed_62.json
    gated_finetune/
      seed_42.json
      ...
    tables/
      summary.csv
      summary.json
      summary.md
      control_comparisons.csv
      diagnostics.csv
      diagnostic_metadata.json
    plots/
      scores.png
      confusions.png
```

The JSON files contain the complete fold histories and out-of-fold predictions,
so tables and plots can be regenerated without retraining:

```bash
python plot_results.py --run-dir /path/to/results/v1_full
```

## Step 1 closeout analysis

After the matched aligned, shuffled, noise, and zero experiments are complete,
`analyze_results.py` tests whether aligned EEG helps any predefined text-hard
subset. It reads the saved predictions and logits only; it does not load LaBSE
or retrain a model. The analysis includes:

- low-confidence and no-EEG-error subsets;
- class, sentence-length, confidence-quartile, and reader-coverage diagnostics;
- favorable and unfavorable prediction flips caused by each modality;
- aligned-versus-control accuracy and macro-F1 differences; and
- sentence-cluster bootstrap intervals that retain all seed predictions for a
  resampled sentence.

The confidence score is the softmax confidence of `logits_without_eeg` inside
the matched gated model. It is not a confidence score from the separately
trained text-only model, whose logits were not saved.

Run the minimal `notebooks/zuco_multimodal_colab.ipynb` notebook after the
controlled suite. Its derived outputs are written to:

```text
Results/zuco_multimodal_sentiment/v3_closeout_analysis/
  analysis_manifest.json
  tables/
    findings.md
    decision.json
    sentence_diagnostics.csv
    paired_predictions.csv
    prediction_flips.csv
    modality_contribution_subsets.csv
    subset_control_comparisons.csv
  plots/
    text_hard_subset_deltas.png
    prediction_flips.png
```

This phase has a predeclared exploratory stop screen. A text-hard subset must
show at least `0.015` aligned-minus-control accuracy, a positive sentence-cluster
interval, and a positive direction in at least two seeds against every control.
Failure closes the current pooled classical-feature fusion pipeline; it does not
claim that EEG intrinsically contains no sentiment information.

## Frozen NeuroLM/LaBraM EEG-only probe

A separate path (branch `feature/neurolm-eeg-probe`) asks whether pretrained
EEG representations carry sentiment information *before* any fusion is tried:
raw ZuCo `rawData` → trial-local preprocessing matched to the released
NeuroLM code → ZuCo→10-10 channel mapping → frozen NeuroLM-B (VQ encoder and
GPT hidden states) → one pooled vector per subject × sentence trial → simple
probes. Subjects are never averaged in the primary analysis.

* Channel mapping: labels from the ZuCo authors' EEGLAB `chanlocs`, geometric
  mutual-nearest matching to 10-10 sites (≤ 6°, ≈1 cm); see
  `reports/channel_mapping.md`.
* Three protocols with hard disjointness assertions: unseen sentences,
  unseen subjects, and unseen subjects + unseen sentences (primary).
* Baselines on identical trials: handcrafted features (logistic regression,
  linear SVM) vs frozen NeuroLM features (logistic regression, linear SVM,
  MLP, linear layer); 5 seeds; sentence- and subject-cluster bootstrap; paired
  permutation tests.
* Controls: sentence- and trial-level label permutation nulls, shuffled
  sentence association, subject-ID decoding, Gaussian features,
  label-distribution baselines, collapse diagnostics, duration-only baseline.
* Embeddings are cached under a fingerprint of checkpoint, channels,
  preprocessing, length handling, and pooling.

Run `notebooks/neurolm_probe_colab.ipynb` top to bottom (GPU recommended for
extraction):
[open in Colab](https://colab.research.google.com/github/MohammadJRanjbar/zuco-multimodal-sentiment/blob/feature/neurolm-eeg-probe/notebooks/neurolm_probe_colab.ipynb).
It clones this branch from `MohammadJRanjbar/zuco-multimodal-sentiment`.
Command-line equivalents:

```bash
pip install -r requirements-neurolm.txt
python scripts/inspect_raw_zuco.py --mat-dir RAW --labels-csv LABELS --out-dir RUN/inspection
python scripts/check_channel_mapping.py --out-dir RUN/mapping --neurolm-dir ../NeuroLM \
    --checkpoint NeuroLM-B.pt --inspection-summary RUN/inspection/raw_zuco_summary.json --mat-dir RAW
python scripts/extract_neurolm_features.py --mat-dir RAW --labels-csv LABELS --neurolm-dir ../NeuroLM \
    --checkpoint NeuroLM-B.pt --cache-dir CACHE --mapping-dir RUN/mapping [--smoke]
python scripts/run_neurolm_probe.py --cache-dir CACHE --handcrafted-dir FEATURES --results-dir RESULTS
python scripts/run_neurolm_sanity.py --cache-dir CACHE --handcrafted-dir FEATURES --results-dir RESULTS
python scripts/build_neurolm_report.py --run-dir RESULTS/probe_v1
```

Protocol-specific configs: `configs/neurolm_probe_{joint,text,subject}_holdout.yaml`.
Tests: `python -m pytest`.

## Repository layout

```text
extract_features.py       original .mat files -> reusable subject caches
inspect_data.py           cache alignment and data summary
run.py                    resumable cross-validation suite
plot_results.py           rebuild tables and plots from saved JSON
analyze_results.py        analyze saved controlled predictions without training
src/features.py           classical EEG statistics
src/zuco_io.py            MATLAB/HDF5 reader
src/data.py               sentence grouping, fold preprocessing, controls
src/model.py              LaBSE, EEG set encoder, and fusion heads
src/engine.py             fold training and best-epoch evaluation
src/experiment.py         setup/seed orchestration
src/reporting.py          summaries, bootstrap deltas, and plots
src/closeout_analysis.py  text-hard subsets, prediction flips, and stop screen
notebooks/                minimal Colab runner
tests/                    feature, split, preprocessing, and model checks
PROJECT_LOG.md            concise experiment and implementation decisions
src/neurolm/              frozen NeuroLM probe: mapping, preprocessing, encoder,
                          pooling, cache, splits, evaluation, sanity, report
scripts/                  NeuroLM probe entry points (inspect, map, extract,
                          probe, sanity, report)
configs/                  NeuroLM probe configs (base + one per protocol)
reports/                  channel mapping and probe results
```
