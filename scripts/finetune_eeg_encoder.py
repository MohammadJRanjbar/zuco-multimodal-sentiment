"""Fine-tune pretrained CBraMod to map a word's raw EEG epoch to that word's embedding (contrastive).

The best available chance for word identity in this EEG: the whole pretrained model is trained, on
single-reader epochs, to pick the read word's LaBSE input embedding among the word types in the batch
(InfoNCE, learned temperature). Five folds split by sentence give out-of-fold predictions for every
epoch; a control run is identical except that the words are shuffled across training epochs (it can
learn the word-frequency distribution but not which EEG goes with which word).

Writes two word-EEG caches under --out-root, <name> and <name>_shuffled, whose per-word features are
the predicted embeddings (so scripts/decode_eeg_to_text.py and scripts/word_information.py test them
like any representation), plus <name>_training.json with per-fold validation retrieval and the
single-trial retrieval of held-out words (real vs shuffled).
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)

import numpy as np  # noqa: E402

from src.brainshaping.data import grouped_splits  # noqa: E402
from src.brainshaping.encoding import static_word_vectors, strip_punctuation  # noqa: E402
from src.fusion.word_eeg import save_subject  # noqa: E402
from src.progress import progress  # noqa: E402
from src.wordinfo.epochs import load_subject_epochs, subject_epoch_files  # noqa: E402
from src.wordinfo.representations import region_pool, scalp_regions  # noqa: E402

TARGET_MODELS = {"labse": "sentence-transformers/LaBSE"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs-dir", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--name", default="cbramod_ft")
    parser.add_argument("--target-model", default="labse", choices=sorted(TARGET_MODELS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr-backbone", type=float, default=5e-5)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--controls", nargs="*", default=["shuffled"], choices=["shuffled"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_all(epochs_dir):
    """All subjects' epochs with per-epoch word, sentence and subject, plus per-subject trials for writing."""
    epochs, words, sentences, subjects, per_subject = [], [], [], [], {}
    for path in subject_epoch_files(epochs_dir):
        subject = os.path.basename(path)[:-4]
        arrays, trials = load_subject_epochs(path)
        word_trial = np.repeat(np.arange(len(trials)), [len(t["words"]) for t in trials])
        index = arrays["epoch_word"]
        epochs.append(arrays["epochs"])
        words += [str(w) for w in arrays["words"][index]]
        sentences.append(np.array([trials[k]["sentence_id"] for k in word_trial[index]], dtype=np.int64))
        subjects += [subject] * len(index)
        per_subject[subject] = (arrays, trials, len(words) - len(index))
    return np.concatenate(epochs), np.array(words, dtype=object), np.concatenate(sentences), np.array(subjects), \
        per_subject


class Head:
    """CBraMod backbone + region pooling + MLP to the target embedding size."""

    @staticmethod
    def build(target_dim, device):
        import torch
        import torch.nn as nn
        from braindecode.models import CBraMod

        backbone = CBraMod.from_pretrained("braindecode/cbramod-pretrained", return_encoder_output=True)
        regions = scalp_regions()

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = backbone
                self.regions = [torch.as_tensor(r) for r in regions]
                width = len(regions) * 200
                self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 512), nn.GELU(), nn.Dropout(0.1),
                                          nn.Linear(512, target_dim))
                self.log_scale = nn.Parameter(torch.tensor(float(np.log(1 / 0.07))))

            def forward(self, x):
                feats = self.backbone(x, return_features=True)["features"].mean(dim=2)  # [B, C, 200]
                pooled = torch.cat([feats[:, r.to(feats.device)].mean(dim=1) for r in self.regions], dim=1)
                return self.head(pooled)

        return Model().to(device)


def train_fold(X, type_of, bank, train_rows, val_rows, args, device, seed):
    """Train on ``train_rows`` (early stopping on validation retrieval loss); return the model."""
    import torch
    import torch.nn.functional as F

    from src.eeg2text.train import autocast, precision_for

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Head.build(bank.shape[1], device)
    groups = [{"params": model.backbone.parameters(), "lr": args.lr_backbone},
              {"params": list(model.head.parameters()) + [model.log_scale], "lr": args.lr_head}]
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    dtype, _ = precision_for(torch.device(device))
    bank_t = F.normalize(torch.as_tensor(bank, dtype=torch.float32, device=device), dim=1)
    val_types = np.unique(type_of[val_rows])
    best, best_state, waited, history = np.inf, None, 0, []

    def batch_x(rows):
        return torch.as_tensor(X[rows].astype(np.float32) / 100.0, device=device)

    for epoch in range(args.max_epochs):
        model.train()
        order = rng.permutation(train_rows)
        losses = []
        for start in range(0, len(order), args.batch_size):
            rows = order[start:start + args.batch_size]
            types, inverse = np.unique(type_of[rows], return_inverse=True)
            with autocast(torch.device(device), dtype):
                pred = F.normalize(model(batch_x(rows)).float(), dim=1)
            logits = pred @ bank_t[types].T * model.log_scale.exp().clamp(max=100)
            loss = F.cross_entropy(logits, torch.as_tensor(inverse, device=device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        val_loss = retrieval_loss(model, X, type_of, bank_t, val_rows, val_types, args.batch_size, device, dtype)
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_loss": val_loss})
        if val_loss < best - 1e-4:
            best, waited = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= args.patience:
                break
    model.load_state_dict(best_state)
    return model, history


def predict(model, X, rows, batch_size, device):
    import torch

    from src.eeg2text.train import autocast, precision_for

    dtype, _ = precision_for(torch.device(device))
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            x = torch.as_tensor(X[rows[start:start + batch_size]].astype(np.float32) / 100.0, device=device)
            with autocast(torch.device(device), dtype):
                out.append(model(x).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, model.head[-1].out_features), np.float32)


def retrieval_loss(model, X, type_of, bank_t, rows, candidate_types, batch_size, device, dtype):
    """Cross-entropy of picking each row's word among ``candidate_types`` (lower is better)."""
    import torch
    import torch.nn.functional as F

    pred = F.normalize(torch.as_tensor(predict(model, X, rows, batch_size, device), device=device), dim=1)
    position = {t: i for i, t in enumerate(candidate_types)}
    target = torch.as_tensor([position[t] for t in type_of[rows]], device=device)
    logits = pred @ bank_t[candidate_types].T * model.log_scale.exp().clamp(max=100)
    return float(F.cross_entropy(logits, target))


def retrieval_ranks(pred, type_of, bank, rows, candidate_types):
    """1-based rank of each row's own word among ``candidate_types`` (cosine)."""
    p = pred / np.maximum(np.linalg.norm(pred, axis=1, keepdims=True), 1e-12)
    b = bank[candidate_types] / np.maximum(np.linalg.norm(bank[candidate_types], axis=1, keepdims=True), 1e-12)
    sims = p @ b.T
    position = {t: i for i, t in enumerate(candidate_types)}
    own = sims[np.arange(len(rows)), [position[t] for t in type_of[rows]]]
    return 1 + (sims > own[:, None]).sum(axis=1)


def main():
    args = parse_args()
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    started = time.time()
    X, words, sentences, subjects, per_subject = load_all(args.epochs_dir)
    bare = np.array([strip_punctuation(w).lower() or w for w in words], dtype=object)
    types, type_of = np.unique(bare, return_inverse=True)
    vectors, found = static_word_vectors([list(types)], np.zeros(len(types), int), np.arange(len(types)),
                                         TARGET_MODELS[args.target_model])
    vectors = np.asarray(vectors, dtype=np.float64)
    bank = ((vectors - vectors[found].mean(axis=0)) / np.maximum(vectors[found].std(axis=0), 1e-8)).astype(np.float32)
    usable = np.asarray(found)[type_of]
    print(f"{len(X):,} word epochs, {len(types):,} word types ({int(usable.sum()):,} epochs with a target), "
          f"{len(np.unique(sentences))} sentences, {len(per_subject)} readers; device {device}")

    runs = {args.name: None, **{f"{args.name}_{c}": c for c in args.controls}}
    report = {"target": TARGET_MODELS[args.target_model], "folds": args.folds, "runs": {}}
    for run, control in runs.items():
        out_of_fold = np.zeros((len(X), bank.shape[1]), np.float32)
        ranks, chance, histories = [], [], []
        folds = grouped_splits(sentences, args.folds, args.seed)
        for f, (train_all, test) in enumerate(progress(folds, desc=run, unit="fold")):
            train_all = train_all[usable[train_all]]
            inner = grouped_splits(sentences[train_all], 10, args.seed + f)[0]
            train_rows, val_rows = train_all[inner[0]], train_all[inner[1]]
            fold_types = type_of.copy()
            if control == "shuffled":  # words reassigned at random among training epochs
                rng = np.random.default_rng(args.seed + 100 + f)
                fold_types[train_rows] = type_of[train_rows][rng.permutation(len(train_rows))]
            model, history = train_fold(X, fold_types, bank, train_rows, val_rows, args, device, args.seed + f)
            out_of_fold[test] = predict(model, X, test, args.batch_size, device)
            scored = test[usable[test]]
            candidates = np.unique(type_of[scored])
            ranks.append(retrieval_ranks(out_of_fold[scored], type_of, bank, scored, candidates))
            chance.append(np.full(len(scored), np.mean(1.0 / np.arange(1, len(candidates) + 1))))
            histories.append(history)
            print(f"  {run} fold {f + 1}: best val loss {min(h['val_loss'] for h in history):.3f} after "
                  f"{len(history)} epochs; held-out top-1 {np.mean(ranks[-1] == 1):.4f}")
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
        r = np.concatenate(ranks)
        report["runs"][run] = {"top1": float(np.mean(r == 1)), "top10": float(np.mean(r <= 10)),
                               "mrr": float(np.mean(1.0 / r)), "chance_mrr": float(np.concatenate(chance).mean()),
                               "history": histories}
        for subject, (arrays, trials, start) in per_subject.items():
            n = len(arrays["epoch_word"])
            values = out_of_fold[start:start + n]
            features = np.full((len(arrays["words"]), values.shape[1]), np.nan, np.float32)
            features[arrays["epoch_word"]] = values
            save_subject(os.path.join(args.out_root, run), subject,
                         [{**t, "features": features[t["word_offset"]:t["word_offset"] + len(t["words"])]}
                          for t in trials])
    report["runtime_min"] = (time.time() - started) / 60
    os.makedirs(args.out_root, exist_ok=True)
    json.dump(report, open(os.path.join(args.out_root, f"{args.name}_training.json"), "w"), indent=1)
    lines = [f"# Fine-tuned CBraMod: EEG epoch -> {report['target']} word embedding", "",
             "Held-out sentences (5 folds), single-reader epochs; rank of the read word among the word types of the "
             "held-out fold.", "", "| run | top-1 | top-10 | mean reciprocal rank | chance MRR |", "|---|---:|---:|---:|---:|"]
    for run, res in report["runs"].items():
        lines.append(f"| {run} | {res['top1']:.4f} | {res['top10']:.4f} | {res['mrr']:.4f} | {res['chance_mrr']:.4f} |")
    lines += ["", "The real run shows word information only if it beats the shuffled-words control (which can learn "
              "word frequency but not the EEG-word pairing)."]
    open(os.path.join(args.out_root, f"{args.name}_training.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
