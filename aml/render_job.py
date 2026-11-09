"""Render an Azure ML job YAML whose command carries aml/src as an embedded tarball.

The subscription's storage policy disables public network access on every storage account, so
`az ml job create` cannot upload a local `code:` snapshot from outside the workspace's managed VNet.
The helper scripts are small, so they travel inside the job definition instead (base64 tar.gz) and
are unpacked on each node before anything runs.

AML caps the total size of a job's docker arguments (environment plus command); a payload past roughly
64K base64 characters fails before the container starts (ArgumentTooLong, visible only in the run's
RunHistory details). The demo job therefore carries no payload: it downloads the same tarball from the
frontend (GET /api/link/src with its job token) and checks it against AXK2_SRC_SHA256.

Templates with __AXK2_SRC_GIT__ carry no payload either: they download aml/src from this public repository
at the pinned commit (raw.githubusercontent.com, the host the overlay already fetches from) and check every
file against __AXK2_SRC_MANIFEST__ (name:sha256 of the git blob). Rendering refuses unless aml/src is
committed and that commit is on the remote. The manifest also lists the benchmark-only files of aml/src/bench
(bench/<name>) for templates whose download command creates the bench/ directory (os.makedirs); older
templates get the top-level files only. The embedded payload and the demo's archive never carry aml/src/bench.

usage: python aml/render_job.py aml/jobs/<template>.yml [more templates...]
writes: aml/jobs/.rendered/<template>.yml
The demo template also needs AXK2_LINK_URL and AXK2_LINK_TOKEN in the environment; the demo
frontend's supervisor calls render() with a fresh token for every submission instead.
"""
import base64
import gzip
import hashlib
import io
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
PLACEHOLDER = "__AXK2_SRC_B64__"
SRC_URL = "__AXK2_SRC_URL__"
SRC_GIT = "__AXK2_SRC_GIT__"
LINK = ("AXK2_LINK_URL", "AXK2_LINK_TOKEN")
SAFE = re.compile(r"[A-Za-z0-9._~:/?=&%+,@-]+")  # URL/token characters; no quotes, $ or braces (AML expressions)


def src_files():
    return [path for path in sorted((HERE / "src").iterdir()) if path.is_file() and path.suffix in {".py", ".sh", ".json"}]


def bench_files():
    """Benchmark-protocol files (aml/src/bench): fetched by git-pinned benchmark jobs, never in the payload."""
    folder = HERE / "src" / "bench"
    return [path for path in sorted(folder.iterdir()) if path.is_file() and path.suffix in {".py", ".sh", ".json"}] \
        if folder.is_dir() else []


def payload():
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as tar:
            for path in src_files():
                data = path.read_bytes().replace(b"\r\n", b"\n")
                info = tarfile.TarInfo(path.name)
                info.size, info.mtime = len(data), 0
                info.mode = 0o755 if path.suffix == ".sh" else 0o644
                tar.addfile(info, io.BytesIO(data))
    return base64.b64encode(buffer.getvalue()).decode()


def git_pin():
    """(owner/repo@commit, name:sha256,...) of aml/src at HEAD, which must be committed and pushed."""
    def git(*args):
        return subprocess.run(["git", "-C", str(HERE.parent), *args], capture_output=True, check=True).stdout
    commit = git("rev-parse", "HEAD").decode().strip()
    if git("status", "--porcelain", "--", "aml/src").strip():
        raise ValueError("aml/src has uncommitted changes")
    if not git("branch", "-r", "--contains", commit).strip():
        raise ValueError(f"commit {commit} is not on the remote; push it first")
    found = re.search(r"github\.com[:/]([\w.-]+/[\w.-]+?)(?:\.git)?/?$", git("remote", "get-url", "origin").decode().strip())
    if not found:
        raise ValueError("origin is not a GitHub repository")
    names = [path.name for path in src_files()] + [f"bench/{path.name}" for path in bench_files()]
    manifest = ",".join(f"{name}:{hashlib.sha256(git('show', f'{commit}:aml/src/{name}')).hexdigest()}" for name in names)
    return f"{found.group(1)}@{commit}", manifest


def manifest_for(template, manifest):
    """The manifest a template can download: bench/<name> entries only if its command creates directories."""
    if "os.makedirs" in template:
        return manifest
    return ",".join(item for item in manifest.split(",") if "/" not in item.split(":", 1)[0])


def digest(encoded):
    return hashlib.sha256(base64.b64decode(encoded)).hexdigest()


def src_url(link_url):
    """wss://host/ws/link -> https://host/api/link/src (where a demo job downloads its scripts)."""
    parts = urlsplit(link_url)
    prefix = parts.path.removesuffix("/ws/link").rstrip("/")
    return f"{'https' if parts.scheme == 'wss' else 'http'}://{parts.netloc}{prefix}/api/link/src" if parts.netloc else ""


def render(text, encoded, values=None):
    """Fills the payload (or its digest) and __NAME__ for every NAME in values; refuses to leave any behind."""
    if PLACEHOLDER not in text and SRC_URL not in text and SRC_GIT not in text:
        raise ValueError(f"the template has none of {PLACEHOLDER}, {SRC_URL} or {SRC_GIT}")
    text = text.replace(PLACEHOLDER, encoded).replace("__AXK2_SRC_SHA256__", digest(encoded))
    values = dict(values or {})
    if SRC_URL in text and "AXK2_SRC_URL" not in values and values.get("AXK2_LINK_URL"):
        values["AXK2_SRC_URL"] = src_url(values["AXK2_LINK_URL"])
    for name, value in values.items():
        if not SAFE.fullmatch(value):
            raise ValueError(f"{name} has characters a job template cannot carry")
        text = text.replace(f"__{name}__", value)
    left = sorted(set(re.findall(r"__AXK2_[A-Z0-9_]+__", text)))
    if left:
        raise ValueError(f"unfilled placeholders: {', '.join(left)}")
    return text


def main(templates):
    encoded = payload()
    values = {name: os.environ[name] for name in LINK if os.environ.get(name)}
    target_dir = HERE / "jobs" / ".rendered"
    target_dir.mkdir(exist_ok=True)
    for template in map(Path, templates):
        try:
            source = template.read_text(encoding="utf-8")
            pinned = {}
            if SRC_GIT in source:
                repo, manifest = git_pin()
                pinned = {"AXK2_SRC_GIT": repo, "AXK2_SRC_MANIFEST": manifest_for(source, manifest)}
            text = render(source, encoded, {**values, **pinned})
        except (ValueError, subprocess.CalledProcessError) as error:
            raise SystemExit(f"{template}: {error}")
        target = target_dir / template.name
        target.write_text(text, encoding="utf-8", newline="\n")
        carried = len(encoded) if PLACEHOLDER in source else 0
        pin = f", scripts from {pinned['AXK2_SRC_GIT']}" if pinned else ""
        print(f"{target} ({carried} base64 chars embedded, {len(text)} chars{pin})")


if __name__ == "__main__":
    main(sys.argv[1:])
