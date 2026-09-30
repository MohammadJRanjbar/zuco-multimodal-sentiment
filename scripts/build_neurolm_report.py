"""Assemble reports/neurolm_probe_results.md from saved probe and sanity outputs."""

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.neurolm.config import load_config, save_json  # noqa: E402
from src.neurolm.report import build_markdown  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/neurolm_probe.yaml")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--inspection-summary", default=None)
    parser.add_argument("--mapping-checks", default=None)
    parser.add_argument("--report", default="reports/neurolm_probe_results.md")
    args = parser.parse_args()

    config = load_config(args.config)
    probe = json.load(open(os.path.join(args.run_dir, "tables", "probe_summary.json")))
    sanity = json.load(open(os.path.join(args.run_dir, "sanity", "sanity_summary.json")))
    manifest = json.load(open(os.path.join(args.run_dir, "manifest.json")))
    inspection = json.load(open(args.inspection_summary)) if args.inspection_summary else None
    mapping = json.load(open(args.mapping_checks)) if args.mapping_checks else None
    markdown, decision = build_markdown(probe, sanity, config, mapping, inspection, manifest)
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w") as handle:
        handle.write(markdown)
    shutil.copy(args.report, os.path.join(args.run_dir, "neurolm_probe_results.md"))
    save_json(decision, os.path.join(args.run_dir, "decision.json"))
    print(markdown)


if __name__ == "__main__":
    main()
