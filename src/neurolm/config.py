"""YAML configs with single inheritance, plus run metadata helpers."""

import copy
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def deep_merge(base, override):
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path, overrides=None):
    """Load a YAML config; a top-level ``base:`` names a parent file."""
    with open(path) as handle:
        config = yaml.safe_load(handle) or {}
    parent = config.pop("base", None)
    if parent:
        config = deep_merge(load_config(os.path.join(os.path.dirname(path), parent)), config)
    config = deep_merge(config, overrides or {})
    config.setdefault("config_path", os.path.abspath(path))
    return config


def set_path_overrides(config, **paths):
    for key, value in paths.items():
        if value is not None:
            config.setdefault("paths", {})[key] = value
    return config


def git_state(root=REPO_ROOT):
    def run(*args):
        try:
            return subprocess.check_output(["git", "-C", root, *args], stderr=subprocess.DEVNULL).decode().strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def environment():
    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in ("numpy", "scipy", "sklearn", "torch", "pandas", "h5py"):
        try:
            module = __import__(name)
            info[name] = getattr(module, "__version__", "?")
        except ImportError:
            info[name] = None
    try:
        import torch

        info["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001 - environment reporting must never fail a run
        info["cuda_device"] = None
    return info


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


def save_json(value, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(value, handle, indent=2, default=_json_default)
    os.replace(temporary, path)


def run_manifest(config, extra=None):
    return {
        "experiment": config.get("experiment"),
        "config": config,
        "git": git_state(),
        "environment": environment(),
        "created_unix": time.time(),
        **(extra or {}),
    }
