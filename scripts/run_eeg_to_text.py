"""EEG-to-text generation in English (ZuCo) and Persian (TeCo), single-language and joint training.

For every combination of
  training setting  en | fa | joint (both languages, shared model, per-language input layers)
  EEG encoder       continuous | vq (learned discrete EEG tokens, codebook shared across languages)
  input condition   eeg | shuffled_eeg | noise | word_vectors, mbart_vectors (positive controls)
an mBART-50 model (LoRA) is trained to write each sentence from one reader's word-level EEG and evaluated
on unseen sentences with teacher forcing and free-running generation, per reader and on the reader
average of each test sentence (condition names ending in _avg). The EEG model is also tested with noise
and shuffled EEG at test time. Training can be augmented (word-span crops, averaging readers of a
sentence, input noise and dropout; identical for every input condition) and, for English, extended with
the ZuCo NR/TSR recordings (scripts/fetch_zuco_task.py). Finished runs are reused. Writes, under
--results-dir/--run-tag: one folder per run (metrics.json, generations.csv, weights.pt), sentences.csv,
and the comparison report eeg_to_text_report.md (positive-control check and sentiment of the generated
text included).
"""

import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.eeg2text import data as d  # noqa: E402
from src.eeg2text.augment import Augment, average_by_sentence  # noqa: E402
from src.eeg2text.metrics import corpus_bleu  # noqa: E402
from src.eeg2text.model import EEGToText, MBartCodec, code_usage, word_token_vectors  # noqa: E402
from src.eeg2text.report import summarize  # noqa: E402
from src.eeg2text.train import Settings, evaluate, train  # noqa: E402

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zuco-word-eeg-dir")
    parser.add_argument("--zuco-extra-dirs", nargs="*", default=[],
                        help="word-EEG caches of other ZuCo tasks (NR, TSR), used as extra English training data")
    parser.add_argument("--teco-trt-dir")
    parser.add_argument("--teco-labels-csv")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--run-tag", default="eeg_to_text_v2")
    parser.add_argument("--settings", nargs="+", default=["en", "fa", "joint"], choices=["en", "fa", "joint"])
    parser.add_argument("--encoders", nargs="+", default=["continuous"], choices=["continuous", "vq"])
    parser.add_argument("--inputs", nargs="+", default=list(d.INPUTS), choices=list(d.INPUTS))
    parser.add_argument("--model", default="facebook/mbart-large-50")
    parser.add_argument("--static-model", default="sentence-transformers/LaBSE",
                        help="word-identity vectors for the word_vectors positive control")
    parser.add_argument("--sentiment-model", default="sentence-transformers/LaBSE",
                        help="sentence embeddings for the sentiment classifier ('none' to skip)")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--min-epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--augment-crop", type=float, default=0.3, help="probability of a word-span crop")
    parser.add_argument("--augment-mix", type=float, default=0.3,
                        help="probability of averaging with other readers of the same sentence")
    parser.add_argument("--augment-noise", type=float, default=0.1, help="std of Gaussian input noise")
    parser.add_argument("--augment-dropout", type=float, default=0.1, help="share of input features zeroed")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-lora", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--full-finetune", action="store_true")
    parser.add_argument("--source-layout", default="mbart", choices=["mbart", "plain"],
                        help="mbart: [language code] words [</s>] as in mBART-50 training; plain: words only")
    parser.add_argument("--vq-codes", type=int, default=512)
    parser.add_argument("--vq-dim", type=int, default=64)
    parser.add_argument("--fold", type=int, default=None, help="rotate the test fifth (0-4); default: one seeded split")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--no-test-swaps", action="store_true", help="skip feeding the EEG model noise/shuffled EEG")
    parser.add_argument("--no-reader-average", action="store_true", help="skip the reader-averaged test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def load_seq2seq(name):
    from transformers import AutoModelForSeq2SeqLM

    return AutoModelForSeq2SeqLM.from_pretrained(name), MBartCodec(name)


def langs_of(setting):
    return ["en", "fa"] if setting == "joint" else [setting]


def items_for(corpus, condition, labels, seed):
    inputs = d.build_inputs(corpus, condition, seed)
    out = {"train": [], "val": [], "test": []}
    for i, ((x, fixated), part) in enumerate(zip(inputs, corpus.part)):
        out[part].append({"x": x, "fixated": fixated, "labels": labels[i], "trial": f"{corpus.lang}:{i}",
                          "words": corpus.words[i], "sentence": int(corpus.sentence_id[i])})
    return out


def attach_mbart_vectors(corpora, model_name):
    lm, codec = load_seq2seq(model_name)
    for corpus in corpora.values():
        words = d.unique_words(corpus)
        d.attach_vectors(corpus, "mbart_vectors", words, *word_token_vectors(lm, codec, words))
    del lm


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    root = os.path.join(args.results_dir, args.run_tag)
    os.makedirs(root, exist_ok=True)
    needed = sorted({lang for s in args.settings for lang in langs_of(s)})
    corpora = {}
    if "en" in needed:
        corpora["en"] = d.load_zuco(args.zuco_word_eeg_dir)
    if "fa" in needed:
        corpora["fa"] = d.load_teco(args.teco_trt_dir, args.teco_labels_csv)
    for lang, corpus in corpora.items():
        d.assign_parts(corpus, seed=args.seed, fold=args.fold)
        if lang == "en":
            for path in args.zuco_extra_dirs:
                session = os.path.basename(os.path.normpath(path))
                dropped = d.add_training_data(corpus, d.load_zuco(path, session=session))
                print(f"en: added {session} as training data ({dropped} trials of held-out sentences dropped)")
        d.normalize(corpus)
        if "word_vectors" in args.inputs:
            d.attach_static_vectors(corpus, args.static_model)
        counts = pd.Series(corpus.part).value_counts().to_dict()
        sessions = pd.Series(corpus.session).value_counts().to_dict()
        print(f"{lang}: {len(corpus.words)} trials, {len(np.unique(corpus.sentence_id))} sentences, "
              f"{corpus.dim} EEG features per word; trials per part {counts}; per recording {sessions}")
    if "mbart_vectors" in args.inputs:
        attach_mbart_vectors(corpora, args.model)
    pd.concat([pd.DataFrame({"lang": lang, "sentence_id": c.sentence_id, "text": c.texts, "label": c.label,
                             "part": c.part}).drop_duplicates("sentence_id") for lang, c in corpora.items()]
              ).to_csv(os.path.join(root, "sentences.csv"), index=False)

    augment = Augment(crop=args.augment_crop, mix=args.augment_mix, noise=args.augment_noise,
                      dropout=args.augment_dropout)
    settings = Settings(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, lr_lora=args.lr_lora,
                        patience=args.patience, min_epochs=args.min_epochs, seed=args.seed,
                        max_new_tokens=args.max_new_tokens, num_beams=args.num_beams, augment=augment)
    key_base = {"model": args.model, "settings": settings.to_dict(), "lora_r": args.lora_r, "fold": args.fold,
                "full_finetune": args.full_finetune, "vq": [args.vq_codes, args.vq_dim],
                "source_layout": args.source_layout,
                "extra": sorted(os.path.basename(os.path.normpath(p)) for p in args.zuco_extra_dirs),
                "reader_average": not args.no_reader_average}
    for encoder in args.encoders:
        for setting in args.settings:
            for condition in args.inputs:
                name = f"{setting}_{encoder}_{condition}"
                out_dir = os.path.join(root, name)
                key = hashlib.sha256(json.dumps({**key_base, "name": name}, sort_keys=True).encode()).hexdigest()[:16]
                metrics_path = os.path.join(out_dir, "metrics.json")
                if os.path.exists(metrics_path) and json.load(open(metrics_path)).get("key") == key:
                    print(f"{name}: reusing finished run")
                    continue
                started = time.time()
                print(f"=== {name} ===")
                torch.manual_seed(args.seed)
                lm, codec = load_seq2seq(args.model)
                langs = langs_of(setting)
                labels = {lang: codec.encode(corpora[lang].texts, lang) for lang in langs}
                data = {lang: items_for(corpora[lang], condition, labels[lang], args.seed) for lang in langs}
                dims = {lang: data[lang]["train"][0]["x"].shape[1] for lang in langs}
                vq = {"codes": args.vq_codes, "dim": args.vq_dim} if encoder == "vq" else None
                model = EEGToText(lm, codec, dims, lora={"r": args.lora_r, "alpha": 2 * args.lora_r, "dropout": 0.05,
                                                         "targets": LORA_TARGETS},
                                  vq=vq, full_finetune=args.full_finetune, source_layout=args.source_layout).to(device)
                info = train(model, {lang: data[lang]["train"] for lang in langs},
                             {lang: data[lang]["val"] for lang in langs}, settings, device, log=print)
                records, codes_by_lang = [], {}
                for lang in langs:
                    tested = [condition] + ([] if args.no_test_swaps or condition != "eeg" else ["noise", "shuffled_eeg"])
                    for test_condition in tested:
                        items = (data[lang]["test"] if test_condition == condition else
                                 items_for(corpora[lang], test_condition, labels[lang], args.seed + 1)["test"])
                        test_sets = [(test_condition, items)]
                        if not args.no_reader_average:
                            test_sets.append((f"{test_condition}_avg", average_by_sentence(items)))
                        corpus = corpora[lang]
                        for tested_name, test_items in test_sets:
                            result, codes = evaluate(model, test_items, lang, settings, device,
                                                     desc=f"{name} {lang} test ({tested_name})")
                            for r in result:
                                i = int(r["trial"].split(":")[1])
                                averaged = r["trial"].endswith(":avg")
                                records.append({**r, "lang": lang, "condition": tested_name,
                                                "sentence_id": int(corpus.sentence_id[i]),
                                                "reader": "average" if averaged else corpus.reader[i],
                                                "label": int(corpus.label[i]), "gold": corpus.texts[i]})
                            if codes is not None and tested_name == condition:
                                codes_by_lang[lang] = codes
                generations = pd.DataFrame(records)
                os.makedirs(out_dir, exist_ok=True)
                generations.to_csv(os.path.join(out_dir, "generations.csv"), index=False)
                torch.save(model.trainable_state(), os.path.join(out_dir, "weights.pt"))
                summary = {}
                for (lang, test_condition), sub in generations.groupby(["lang", "condition"]):
                    summary[f"{lang}/{test_condition}"] = {
                        "tf_accuracy": float(sub["tf_correct"].sum() / max(sub["tf_scored"].sum(), 1)),
                        "tf_bleu4": corpus_bleu(sub["tf_text"], sub["gold"]),
                        "free_bleu4": corpus_bleu(sub["free_text"], sub["gold"])}
                    print(f"  {lang} tested on {test_condition}: {summary[f'{lang}/{test_condition}']}")
                json.dump({"key": key, "setting": setting, "encoder": encoder, "input": condition, "langs": langs,
                           "train": info, "test": summary, "runtime_min": (time.time() - started) / 60,
                           "vq_code_usage": code_usage(codes_by_lang, args.vq_codes) if codes_by_lang else None,
                           "examples": generations[generations["condition"] == condition].head(8)[
                               ["lang", "gold", "tf_text", "free_text"]].to_dict("records")},
                          open(metrics_path, "w"), indent=1, ensure_ascii=False, default=float)
                del model, lm
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    sentiment = None if args.sentiment_model == "none" else args.sentiment_model
    summarize(root, sentiment, str(device))
    print(open(os.path.join(root, "eeg_to_text_report.md")).read())


if __name__ == "__main__":
    main()
