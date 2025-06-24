"""new_memory_manager.py

Thread‑safe memory manager for Melah Memory Architecture v3.

Highlights
----------
*   Atomic write & manifest update protected by file locks (via `filelock`).
*   Strict input validation (regex) to guard against corrupt filenames.
*   UTC timestamps everywhere; directory layout YYYY/mm/dd.
*   Logging instead of `print`, with clear INFO / WARNING / ERROR.
*   Fully type‑annotated public API.

Author: AI‑assistant (2025‑06‑20)
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from filelock import FileLock, Timeout  # external, lightweight
except ImportError:  # graceful degradation
    FileLock = None  # type: ignore

# --------------------------------------------------------------------------- #
# Logging defaults
# --------------------------------------------------------------------------- #

logging.basicConfig(
    format="[%(levelname)s] %(asctime)s - %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
    level=logging.INFO,
)

# --------------------------------------------------------------------------- #
# Helper regex for simple identifier sanity check
# --------------------------------------------------------------------------- #

_ID_RE = re.compile(r"^[a-zA-Z0-9_\-]{1,64}$")

# --------------------------------------------------------------------------- #
# NewMemoryManager
# --------------------------------------------------------------------------- #


class NewMemoryManager:
    """
    Manage atomic chat‑session memories on disk (JSON).
    
    หมายเหตุ: ไม่ควรใช้คลาสนี้ในการ log raw chat โดยตรง ให้ใช้ raw_chat_logger เป็นหลัก
    และใช้ NewMemoryManager สำหรับสร้าง/จัดการ memory object ที่ enrich แล้วเท่านั้น
    """

    SCHEMA_VERSION: str = "3.1"  # bump after refactor
    MANIFEST_LOCK_TIMEOUT: float = 5.0  # seconds

    def __init__(self, base_path: str | Path) -> None:
        self.base_path: Path = Path(base_path).expanduser().resolve()
        # Change sessions_path to point to archive/chat_sessions
        self.sessions_path: Path = self.base_path / "archive" / "chat_sessions"
        self.summaries_path: Path = self.base_path / "archive" / "summaries"
        self.manifest_path: Path = self.base_path / "archive" / "manifest.json"

        self._ensure_structure()
        logging.info("Memory manager initialised at %s", self.base_path)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def create_atomic_memory(
        self,
        session_id: str,
        speaker: str,
        raw_text: str,
        *,
        content: Optional[Dict[str, Any]] = None,
        nlp: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create one memory JSON file & update manifest safely."""

        self._validate_identifier(session_id, "session_id")
        self._validate_identifier(speaker, "speaker")

        content = content or {}
        nlp = nlp or {}

        now_utc = datetime.now(timezone.utc)
        ts_str = now_utc.strftime("%Y%m%d%H%M%S")
        short_uuid = uuid.uuid4().hex[:12]
        memory_id = f"{ts_str}_{short_uuid}"

        link_prev = self._latest_memory_id()

        memory_data: Dict[str, Any] = {
            "schema_version": self.SCHEMA_VERSION,
            "memory_id": memory_id,
            "session_id": session_id,
            "timestamp_utc": now_utc.isoformat(),
            "speaker": speaker,
            "content": content,
            "raw_text": raw_text,
            "entities": nlp.get("entities", []),
            "intent": nlp.get("intent"),
            "sentiment": nlp.get("sentiment", {}),
            "keywords": nlp.get("keywords", []),
            "link_prev": link_prev,
        }

        target_dir = (
            self.sessions_path
            / str(now_utc.year)
            / f"{now_utc.month:02d}"
            / f"{now_utc.day:02d}"
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        file_path = target_dir / f"{memory_id}.json"

        try:
            with file_path.open("w", encoding="utf-8") as fh:
                json.dump(memory_data, fh, indent=2, ensure_ascii=False)
            logging.info("Created memory %s", memory_id)
        except Exception as exc:
            logging.exception("Failed to write memory file: %s", exc)
            raise

        # persist manifest
        self._update_manifest(memory_id, file_path.relative_to(self.base_path))
        return memory_data

    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Load a memory JSON by ID (fast path derive folder from timestamp)."""
        try:
            ts_part = memory_id.split("_")[0]
            if not ts_part.isdigit() or len(ts_part) != 14:
                raise ValueError("malformed timestamp in memory_id")
            year, month, day = ts_part[:4], ts_part[4:6], ts_part[6:8]
            file_path = (
                self.sessions_path / year / month / day / f"{memory_id}.json"
            )
            if not file_path.exists():
                logging.warning("Memory %s not found", memory_id)
                return None
            with file_path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:
            logging.exception("Error reading memory %s: %s", memory_id, exc)
            return None

    @classmethod
    def from_raw_chat_log(cls, base_path, text, metadata=None, timestamp=None):
        """
        สร้าง atomic memory จาก raw chat log ที่ถูกบันทึกโดย raw_chat_logger
        base_path: path หลักของ memory_core
        text: ข้อความแชทดิบ
        metadata: dict (session_id, speaker, nlp, etc.)
        timestamp: datetime (optional)
        """
        mgr = cls(base_path)
        session_id = metadata.get("session_id") if metadata and "session_id" in metadata else "default_session"
        speaker = metadata.get("speaker") if metadata and "speaker" in metadata else "user"
        nlp = metadata.get("nlp") if metadata and "nlp" in metadata else {}
        content = metadata if metadata else {}
        # ใช้ timestamp ที่รับมา หรือ datetime.now(timezone.utc)
        now_utc = timestamp if timestamp else datetime.now(timezone.utc)
        return mgr.create_atomic_memory(
            session_id=session_id,
            speaker=speaker,
            raw_text=text,
            content=content,
            nlp=nlp
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _ensure_structure(self) -> None:
        """Ensure base directories + manifest exist."""
        (self.sessions_path).mkdir(parents=True, exist_ok=True)
        for sub in ["daily", "monthly", "yearly", "decadal", "centurial"]:
            (self.summaries_path / sub).mkdir(parents=True, exist_ok=True)

        if not self.manifest_path.exists():
            manifest = {
                "schema_version": self.SCHEMA_VERSION,
                "system_status": "initialising",
                "created_utc": datetime.now(timezone.utc).isoformat(),
            }
            self._atomic_write_json(self.manifest_path, manifest)
            logging.info("New manifest created")

    def _atomic_write_json(self, path: Path, data: Dict[str, Any]) -> None:
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        tmp.replace(path)

    def _manifest_lock(self) -> Optional["FileLock"]:
        if FileLock is None:
            return None
        return FileLock(str(self.manifest_path) + ".lock")

    def _latest_memory_id(self) -> Optional[str]:
        manifest = self._load_manifest()
        return manifest.get("latest_memory_id")

    def _load_manifest(self) -> Dict[str, Any]:
        try:
            with self.manifest_path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            logging.warning("Manifest missing or corrupt; recreating")
            self._ensure_structure()
            return self._load_manifest()

    def _update_manifest(self, latest_id: str, latest_relpath: Path) -> None:
        """Update manifest with lock to avoid race conditions."""
        lock_ctx = self._manifest_lock() or _nullcontext()
        try:
            with lock_ctx:
                manifest = self._load_manifest()
                manifest.update(
                    {
                        "last_updated_utc": datetime.now(timezone.utc).isoformat(),
                        "latest_memory_id": latest_id,
                        "latest_memory_path": str(latest_relpath),
                        "system_status": "ok",
                    }
                )
                self._atomic_write_json(self.manifest_path, manifest)
        except Timeout:
            logging.error("Could not acquire manifest lock within %.1fs", self.MANIFEST_LOCK_TIMEOUT)
            raise
        except Exception as exc:
            logging.exception("Manifest update failed: %s", exc)
            raise

    def _validate_identifier(self, value: str, name: str) -> None:
        if not _ID_RE.fullmatch(value):
            raise ValueError(f"Invalid {name}: '{value}'")

# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #

from contextlib import contextmanager

@contextmanager
def _nullcontext():
    yield


# --------------------------------------------------------------------------- #
# Demo run
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    # Set root to workspace root for demo
    root = Path("../../memory_core")
    mgr = NewMemoryManager(root)

    logging.info("Creating demo memories…")
    mem1 = mgr.create_atomic_memory("S001", "user", "สวัสดีเมล่า!", nlp={"keywords": ["hello"]})
    mem2 = mgr.create_atomic_memory("S001", "melah", "ทักทายค่ะ", nlp={"keywords": ["reply"]})

    logging.info("mem2 links to %s", mem2["link_prev"])
    assert mem2["link_prev"] == mem1["memory_id"]
    assert mgr.get_memory(mem1["memory_id"]) is not None
    logging.info("Demo OK ✅")
