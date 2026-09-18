"""Provenance, content-addressed reuse and atomic artifact publication."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .contracts import CONTRACT_VERSION, STATUSES, stable_hash


def _run(cmd) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def environment_fingerprint(repo: Path) -> Dict[str, str]:
    freeze = _run([sys.executable, "-m", "pip", "freeze"])
    return {"python": platform.python_version(), "platform": platform.platform(),
            "source_sha": _run(["git", "-C", str(repo), "rev-parse", "HEAD"]),
            "source_dirty": bool(_run(["git", "-C", str(repo), "status", "--porcelain"])),
            "dependency_fingerprint": hashlib.sha256(freeze.encode()).hexdigest()[:16],
            "contract_version": CONTRACT_VERSION}


def artifact_dir(root: Path, dataset: str, protocol: str, variant: str, master_seed: int,
                 config_hash: str) -> Path:
    return Path(root) / dataset / protocol / variant / f"seed{master_seed}" / config_hash[:16]


def write_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with open(tmp, "w") as fh:
        json.dump(payload, fh, default=float)
    os.replace(tmp, path)


def claim(path: Path, owner: str) -> bool:
    """Exclusive ownership of an output directory; two workers never write one result."""
    path.mkdir(parents=True, exist_ok=True)
    lock = path / "OWNER"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{owner} pid={os.getpid()} t={time.time()}".encode()); os.close(fd)
        return True
    except FileExistsError:
        return lock.read_text().startswith(owner)


def record_status(root: Path, entry: Dict[str, Any]) -> None:
    assert entry["status"] in STATUSES, entry["status"]
    ledger = Path(root) / "status_ledger.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger, "a") as fh:
        fh.write(json.dumps({**entry, "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                            default=str) + "\n")
