"""Save a run's small result files to GitHub and list its large files.

Copies text-like results (.md, .json, .csv, .png, .txt, .yaml up to --max-mb)
from one or more result folders (on Drive) into ``saved_results/<name>/`` in the
repository. Large or binary artifacts (model weights, caches) stay on Drive and
are listed in ``saved_results/<name>/LARGE_FILES.md`` with size and SHA-256 so
they can be found and verified later.

--commit commits locally; --push also pushes, authenticating with the
GITHUB_TOKEN environment variable (in Colab: a secret named GITHUB_TOKEN).
--hf-repo additionally uploads the large files (weights, caches) to a private
Hugging Face model repository, authenticating with HF_TOKEN. Tokens are never
printed.
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import time

TEXT_EXTENSIONS = {".md", ".json", ".csv", ".png", ".txt", ".yaml", ".yml"}
DEFAULT_REMOTE = "https://github.com/MohammadJRanjbar/zuco-multimodal-sentiment.git"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", nargs="+", required=True, help="result folder(s) to save")
    parser.add_argument("--name", required=True, help="folder name under saved_results/")
    parser.add_argument("--max-mb", type=float, default=5.0, help="largest file copied into git")
    parser.add_argument("--repo", default=REPO_ROOT)
    parser.add_argument("--commit", action="store_true")
    parser.add_argument("--push", action="store_true", help="commit and push (needs GITHUB_TOKEN)")
    parser.add_argument("--remote-url", default=DEFAULT_REMOTE)
    parser.add_argument("--branch", default=None, help="defaults to the current branch")
    parser.add_argument("--author-name", default=os.environ.get("GIT_AUTHOR_NAME", "colab"))
    parser.add_argument("--author-email", default=os.environ.get("GIT_AUTHOR_EMAIL", "colab@users.noreply.github.com"))
    parser.add_argument("--hf-repo", default=None,
                        help="also upload large files to this private HF model repo, e.g. user/zuco-eeg-weights")
    return parser.parse_args()


def sha256(path, chunk=1 << 24):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def collect(sources, destination, max_bytes):
    copied, large = [], []
    for source in sources:
        source = os.path.abspath(source)
        base = os.path.basename(source.rstrip("/"))
        for folder, _, files in os.walk(source):
            for name in sorted(files):
                path = os.path.join(folder, name)
                relative = os.path.join(base, os.path.relpath(path, source))
                size = os.path.getsize(path)
                if os.path.splitext(name)[1].lower() in TEXT_EXTENSIONS and size <= max_bytes:
                    target = os.path.join(destination, relative)
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    shutil.copy2(path, target)
                    copied.append((relative, size))
                else:
                    large.append((path, size, sha256(path)))
    return copied, large


def upload_to_hub(large, sources, name, repo_id):
    """Upload large files to a private HF model repo in one commit; returns repo paths."""
    from huggingface_hub import CommitOperationAdd, HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set; add it as a Colab secret (a Hugging Face write token)")
    api = HfApi(token=token)
    api.create_repo(repo_id, private=True, exist_ok=True, repo_type="model")
    bases = {os.path.abspath(s): os.path.basename(os.path.abspath(s).rstrip("/")) for s in sources}
    operations, locations = [], {}
    for path, _, _ in large:
        source = max((s for s in bases if path.startswith(s + os.sep)), key=len)
        in_repo = "/".join([name, bases[source], os.path.relpath(path, source)])
        operations.append(CommitOperationAdd(path_in_repo=in_repo, path_or_fileobj=path))
        locations[path] = f"{repo_id}/{in_repo}"
    if operations:
        api.create_commit(repo_id=repo_id, operations=operations, repo_type="model",
                          commit_message=f"Upload large files: {name}")
    print(f"uploaded {len(operations)} large files to https://huggingface.co/{repo_id} (private)")
    return locations


def write_large_files(destination, large, hub=None):
    hub = hub or {}
    lines = ["# Large files kept on Drive", "",
             "These are not in git (too large or binary). Paths are as seen from Colab."
             + (" Copies are in the private Hugging Face repo shown." if hub else ""), "",
             "| path | size (MB) | sha256 | Hugging Face copy |", "|---|---:|---|---|"]
    for path, size, digest in large:
        lines.append(f"| `{path}` | {size / 1e6:.1f} | `{digest}` | {hub.get(path, '—')} |")
    if not large:
        lines.append("| — | — | — | — |")
    with open(os.path.join(destination, "LARGE_FILES.md"), "w") as handle:
        handle.write("\n".join(lines) + "\n")


def git(repo, *args, secret=None, check=True):
    result = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)
    output = (result.stdout + result.stderr).strip()
    if secret:
        output = output.replace(secret, "***")
    if output:
        print(output)
    if check and result.returncode != 0:
        raise SystemExit(f"git {args[0]} failed")
    return result


def main():
    args = parse_args()
    destination = os.path.join(args.repo, "saved_results", args.name)
    os.makedirs(destination, exist_ok=True)
    copied, large = collect(args.source, destination, args.max_mb * 1e6)
    hub = upload_to_hub(large, args.source, args.name, args.hf_repo) if args.hf_repo else None
    write_large_files(destination, large, hub)
    total = sum(size for _, size in copied)
    print(f"copied {len(copied)} files ({total / 1e6:.1f} MB) to {destination}; "
          f"{len(large)} large files listed in LARGE_FILES.md")
    if not (args.commit or args.push):
        return
    identity = ["-c", f"user.name={args.author_name}", "-c", f"user.email={args.author_email}"]
    git(args.repo, "add", os.path.relpath(destination, args.repo))
    staged = git(args.repo, "diff", "--cached", "--quiet", check=False)
    if staged.returncode == 0:
        print("nothing new to commit")
    else:
        git(args.repo, *identity, "commit", "-m", f"Save results: {args.name} ({time.strftime('%Y-%m-%d %H:%M')})")
    if not args.push:
        return
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise SystemExit("GITHUB_TOKEN is not set; add it as a Colab secret (see the notebook cell)")
    branch = args.branch or git(args.repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    url = args.remote_url.replace("https://", f"https://x-access-token:{token}@")
    git(args.repo, *identity, "pull", "--rebase", url, branch, secret=token)
    git(args.repo, "push", url, f"HEAD:{branch}", secret=token)
    print(f"pushed saved_results/{args.name} to {args.remote_url} ({branch})")


if __name__ == "__main__":
    sys.exit(main())
