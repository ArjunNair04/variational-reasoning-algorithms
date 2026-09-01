"""Atomic cell-completion receipts for YAML-driven LM experiments."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import socket
from pathlib import Path
from typing import Any, Iterable


RECEIPT_SCHEMA_VERSION = 1


class ResultContractError(RuntimeError):
    """A required result artifact is absent, corrupt, or mismatched."""


def acquire_cell_lock(
    result_root: str | os.PathLike[str],
    *,
    task: str,
    tag: str,
    method: str,
    seed: int,
):
    """Acquire a process-scoped nonblocking lock for one result identity."""
    try:
        import fcntl
    except ImportError as exc:  # pragma: no cover - Linux/macOS contract
        raise ResultContractError("cell locking requires fcntl") from exc
    path = Path(result_root) / (
        f".running_{task}__{tag}__{method}_s{int(seed)}.lock"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        stream.seek(0)
        owner = stream.read().strip() or "unknown owner"
        stream.close()
        raise ResultContractError(f"cell is already running ({owner}): {path}") from exc
    stream.seek(0)
    stream.truncate()
    stream.write(f"pid={os.getpid()} host={socket.gethostname()}\n")
    stream.flush()
    os.fsync(stream.fileno())
    return stream


def release_cell_lock(stream) -> None:
    try:
        import fcntl

        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    finally:
        stream.close()


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode("utf-8")


def cell_fingerprint(identity: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(identity)).hexdigest()


def receipt_path(
    out: str | Path,
    task: str,
    tag: str,
    method: str,
    seed: int,
) -> Path:
    return Path(out) / f"complete_{task}__{tag}__{method}_s{seed}.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(path: str | Path, root: str | Path) -> dict[str, Any]:
    path = Path(path)
    root = Path(root)
    if not path.is_file() or path.is_symlink():
        raise ResultContractError(f"required artifact is not a regular file: {path}")
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ResultContractError(f"artifact escapes result root: {path}") from exc
    return {
        "path": relative.as_posix(),
        "size": path.stat().st_size,
        "sha256": _sha256(path),
    }


def adapter_artifacts(path: str | Path) -> list[Path]:
    root = Path(path)
    if not root.is_dir() or root.is_symlink():
        raise ResultContractError(f"required adapter directory is missing: {root}")
    weights = [
        candidate
        for candidate in (
            root / "adapter_model.safetensors",
            root / "adapter_model.bin",
        )
        if candidate.is_file() and not candidate.is_symlink()
    ]
    if not weights:
        raise ResultContractError(f"adapter weights are missing: {root}")
    config = root / "adapter_config.json"
    if not config.is_file() or config.is_symlink():
        raise ResultContractError(f"adapter config is missing: {config}")
    return [config, *weights]


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{secrets.token_hex(6)}.tmp"
    )
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def write_completion_receipt(
    path: str | Path,
    *,
    identity: dict[str, Any],
    artifact_paths: Iterable[str | Path],
    result_root: str | Path,
) -> dict[str, Any]:
    result_root = Path(result_root)
    artifacts = [
        artifact_record(artifact_path, result_root)
        for artifact_path in sorted({Path(value) for value in artifact_paths})
    ]
    payload = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "complete",
        "cell_fingerprint": cell_fingerprint(identity),
        "identity": identity,
        "artifacts": artifacts,
    }
    atomic_write_json(path, payload)
    return payload


def validate_completion_receipt(
    path: str | Path,
    *,
    expected_fingerprint: str | None = None,
    result_root: str | Path | None = None,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ResultContractError(f"invalid completion receipt {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ResultContractError(f"completion receipt is not an object: {path}")
    if payload.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise ResultContractError(f"unsupported completion receipt schema: {path}")
    if payload.get("status") != "complete":
        raise ResultContractError(f"completion receipt is not complete: {path}")
    identity = payload.get("identity")
    if not isinstance(identity, dict):
        raise ResultContractError(f"completion receipt has no identity: {path}")
    fingerprint = cell_fingerprint(identity)
    if payload.get("cell_fingerprint") != fingerprint:
        raise ResultContractError(f"completion receipt fingerprint is corrupt: {path}")
    if expected_fingerprint is not None and fingerprint != expected_fingerprint:
        raise ResultContractError(f"completion receipt belongs to another cell: {path}")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ResultContractError(f"completion receipt has no artifacts: {path}")
    root = Path(result_root) if result_root is not None else path.parent
    for index, record in enumerate(artifacts):
        if not isinstance(record, dict):
            raise ResultContractError(f"invalid artifact record {index}: {path}")
        relative = Path(str(record.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ResultContractError(f"unsafe artifact path in receipt: {relative}")
        artifact = root / relative
        if not artifact.is_file() or artifact.is_symlink():
            raise ResultContractError(f"receipt artifact is missing: {artifact}")
        if artifact.stat().st_size != record.get("size"):
            raise ResultContractError(f"receipt artifact size changed: {artifact}")
        if verify_hashes and _sha256(artifact) != record.get("sha256"):
            raise ResultContractError(f"receipt artifact hash changed: {artifact}")
    return payload


def validate_receipt_identity(
    payload: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    identity = payload["identity"]
    mismatches = {
        key: (identity.get(key), value)
        for key, value in expected.items()
        if identity.get(key) != value
    }
    if mismatches:
        raise ResultContractError(
            f"completion receipt identity mismatch: {mismatches}"
        )
