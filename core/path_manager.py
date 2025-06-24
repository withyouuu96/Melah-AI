import os
import json
from datetime import datetime
from .memory_meta_manager import MemoryMetaManager
import glob
from typing import List, Optional

class PathManager:
    """
    จัดการ path ทั้งหมดของระบบ Melah
    - ตรวจสอบความถูกต้องของ path
    - แปลง path ให้เป็นมาตรฐาน
    - ดูแลโครงสร้างไฟล์และโฟลเดอร์
    """

    def __init__(self, root_dir: str = "."):
        self.root_dir = os.path.abspath(root_dir)
        self.error_log = []
        
        # โครงสร้างพื้นฐานที่ต้องมี
        self.required_dirs = {
            "memory_core": ["archive", "core_systems"],
            "memory_core/archive": ["chat_sessions", "chat_sessions_legacy", "summaries"],
            "memory": [],
            "core": []
        }
        
        # สร้างโครงสร้างพื้นฐานถ้ายังไม่มี
        self._ensure_base_structure()

    def _ensure_base_structure(self):
        """สร้างโครงสร้างโฟลเดอร์พื้นฐานถ้ายังไม่มี"""
        for parent, children in self.required_dirs.items():
            parent_path = os.path.join(self.root_dir, parent)
            if not os.path.exists(parent_path):
                os.makedirs(parent_path)
                self.error_log.append(f"Created missing directory: {parent_path}")
            
            for child in children:
                child_path = os.path.join(parent_path, child)
                if not os.path.exists(child_path):
                    os.makedirs(child_path)
                    self.error_log.append(f"Created missing directory: {child_path}")

    def resolve_path(self, path: str) -> str:
        """แปลง path ให้เป็น absolute path"""
        if os.path.isabs(path):
            return path
        return os.path.join(self.root_dir, path)

    def validate_path(self, path: str) -> bool:
        """ตรวจสอบว่า path มีอยู่จริง"""
        full_path = self.resolve_path(path)
        exists = os.path.exists(full_path)
        if not exists:
            self.error_log.append(f"Path not found: {path}")
        return exists

    def read_session(self, session_path: str) -> Optional[str]:
        """อ่านไฟล์ session โดยรองรับทั้ง relative และ absolute path"""
        if not self.validate_path(session_path):
            return None
        
        try:
            with open(self.resolve_path(session_path), 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            self.error_log.append(f"Error reading session {session_path}: {str(e)}")
            return None

    def list_files(self, pattern: str) -> List[str]:
        """ค้นหาไฟล์ตาม pattern"""
        full_pattern = os.path.join(self.root_dir, pattern)
        return glob.glob(full_pattern)

    def check_integrity(self, check_type: str) -> bool:
        """
        ตรวจสอบความสมบูรณ์ของระบบไฟล์
        check_type: "chunk", "summary", "legacy"
        """
        if check_type == "chunk":
            return self.validate_path("memory_core/archive/chat_sessions")
        elif check_type == "summary":
            return self.validate_path("memory_core/archive/summaries")
        elif check_type == "legacy":
            return self.validate_path("memory_core/archive/chat_sessions_legacy")
        else:
            self.error_log.append(f"Unknown check type: {check_type}")
            return False

    def auto_correct_path(self, path: str) -> Optional[str]:
        """
        พยายามแก้ไข path ที่ไม่ถูกต้องโดยอัตโนมัติ
        เช่น ถ้าไม่มี prefix memory_core จะลองเติมให้
        """
        if self.validate_path(path):
            return path

        # ลองเติม prefix
        if not path.startswith("memory_core/"):
            new_path = f"memory_core/{path}"
            if self.validate_path(new_path):
                return new_path

        # ลองย้ายไฟล์ไปที่ legacy
        basename = os.path.basename(path)
        legacy_path = f"memory_core/archive/chat_sessions_legacy/{basename}"
        if self.validate_path(legacy_path):
            return legacy_path

        return None

    def get_error_log(self) -> List[str]:
        """ดึง error log"""
        return self.error_log.copy()

    def clear_error_log(self):
        """ล้าง error log"""
        self.error_log = []

    def _init_dirs(self):
        for d in [self.session_dir, self.summary_dir, self.legacy_dir]:
            os.makedirs(d, exist_ok=True)

    def _load_index(self, file):
        if os.path.exists(file):
            with open(file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_index(self, file, index):
        with open(file, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

    def _load_bookmarks(self, file):
        if os.path.exists(file):
            with open(file, "r", encoding="utf-8") as f:
                return [line.strip() for line in f.readlines() if line.strip()]
        return []

# --- PATH EXTENSIONS: For Chain/Bridge/Legacy Integration ---
    def normalize_path(self, rel_path):
        rel_path = rel_path.replace("\\", os.sep).replace("/", os.sep)
        abs_path = os.path.join(self.root_dir, rel_path)
        return abs_path

    def get_batch_sessions_content(self, batch_file):
        abs_batch_path = self.normalize_path(batch_file)
        if not os.path.exists(abs_batch_path):
            print(f"Batch summary not found: {abs_batch_path}")
            return []
        session_contents = []
        with open(abs_batch_path, "r", encoding="utf-8") as f:
            for line in f:
                rel_path = line.strip()
                if rel_path:
                    content = self.read_session(rel_path)
                    if content:
                        session_contents.append(content)
        return session_contents

    def is_legacy_path(self, rel_path):
        return "chat_sessions_legacy" in rel_path.replace("/", os.sep).replace("\\", os.sep)

# --- Chain & Path Management ---

    def add_session(self, session_file, summary_id=None, prev_session=None):
        """เพิ่ม session ใหม่ และเชื่อมโยงกับ session ก่อนหน้า/summary"""
        entry = {
            "file": session_file,
            "prev": prev_session,
            "summary": summary_id,
            "timestamp": datetime.now().isoformat()
        }
        self.path_index[session_file] = entry
        self._save_index(self.path_index_file, self.path_index)

    def add_summary(self, summary_id, related_sessions=None, prev_summary=None):
        """เพิ่ม summary event/chain และเชื่อมโยงกับ session/raw"""
        entry = {
            "id": summary_id,
            "related_sessions": related_sessions or [],
            "prev": prev_summary,
            "timestamp": datetime.now().isoformat()
        }
        self.summary_index[summary_id] = entry
        self._save_index(self.summary_index_file, self.summary_index)

    def add_legacy_session(self, legacy_file, prev_legacy=None):
        entry = {
            "file": legacy_file,
            "prev": prev_legacy,
            "timestamp": datetime.now().isoformat()
        }
        self.legacy_index[legacy_file] = entry
        self._save_index(self.legacy_index_file, self.legacy_index)

# --- Lookup/Traversal ---

    def get_prev(self, file, index_type="session"):
        if index_type == "session":
            value = self.path_index.get(file, {})
        elif index_type == "summary":
            value = self.summary_index.get(file, {})
        elif index_type == "legacy":
            value = self.legacy_index.get(file, {})
        else:
            value = {}
        if isinstance(value, dict):
            return value.get("prev")
        return None

    def get_related_sessions(self, summary_id):
        value = self.summary_index.get(summary_id, {})
        if isinstance(value, dict):
            return value.get("related_sessions", [])
        return []

    def get_summary_for_session(self, session_file):
        value = self.path_index.get(session_file, {})
        if isinstance(value, dict):
            return value.get("summary")
        return None

    def traverse_chain(self, start, index_type="session"):
        """เดิน chain ไปข้างหน้า/ย้อนอดีต"""
        chain = []
        curr = start
        while curr:
            chain.append(curr)
            curr = self.get_prev(curr, index_type=index_type)
        return chain[::-1]  # ย้อนอดีต→ปัจจุบัน

# --- Integrity Check ---

    def check_integrity(self, index_type="session"):
        """ตรวจสอบ chain ว่าขาด/วนลูปไหม"""
        if index_type == "session":
            index = self.path_index
        elif index_type == "summary":
            index = self.summary_index
        elif index_type == "legacy":
            index = self.legacy_index
        else:
            return False
        visited = set()
        for key, entry in index.items():
            curr = key
            while curr:
                if curr in visited:
                    print(f"Loop detected at {curr}")
                    return False
                visited.add(curr)
                curr = index.get(curr, {}).get("prev")
        return True

# --- Bookmark/Tag Support ---

    def add_bookmark(self, tag, path=None):
        entry = f"{tag} | {path or ''}"
        self.bookmarks.append(entry)
        with open(self.bookmark_file, "a", encoding="utf-8") as f:
            f.write(entry + "\n")

    def search_bookmarks(self, keyword):
        return [b for b in self.bookmarks if keyword.lower() in b.lower()]

# --- Cross-Reference ---

    def get_summary_chain(self, session_file):
        """ดึง chain ของ summary จาก session (ย้อนหลังได้)"""
        summary_id = self.get_summary_for_session(session_file)
        if summary_id:
            chain = self.traverse_chain(summary_id, index_type="summary")
            return chain
        return []

    def get_sessions_by_summary(self, summary_id):
        """ดึง session ทั้งหมดที่เกี่ยวข้องกับ summary"""
        return self.get_related_sessions(summary_id)

# --- Utility ---

    def list_sessions(self):
        return list(self.path_index.keys())

    def list_summaries(self):
        return list(self.summary_index.keys())

    def list_legacy(self):
        return list(self.legacy_index.keys())

# === Stubs for ContextWindowManager compatibility ===

    def _log_error(self, error_message: str):
        self.error_log.append(error_message)
        # print(f"PathManager Error: {error_message}") # Optional: print to console

    def add_daily_raw_chat_log(self, date_key: str, file_path_absolute: str, session_id: str, **kwargs):
        """(Stub) Placeholder for adding a daily raw chat log entry."""
        self._log_error(f"[Stub] add_daily_raw_chat_log called for date: {date_key}, session: {session_id}, path: {file_path_absolute}")
        # In a real implementation, this would update an index or internal state.
        pass

    def get_daily_raw_chat_log_entry(self, date_key: str) -> dict | None:
        """(Stub) Placeholder for retrieving a daily raw chat log entry."""
        self._log_error(f"[Stub] get_daily_raw_chat_log_entry called for date: {date_key}")
        # In a real implementation, this would look up an entry.
        return None # Or a mock entry dict

    def read_archived_file_content(self, file_path_relative: str) -> str | None:
        """(Stub) Placeholder for reading content from an archived file."""
        self._log_error(f"[Stub] read_archived_file_content called for path: {file_path_relative}")
        # In a real implementation, this would read the file.
        # For now, let's check if it's one of the legacy session files we know about.
        full_path = self.get_full_path(file_path_relative)
        if self.is_valid_path(full_path) and os.path.exists(full_path):
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                self._log_error(f"[Stub Read Attempt] Error reading {full_path}: {e}")
                return None
        return None

    def on_new_raw_chat_log(file_path, rel_path, metadata, text, timestamp):
        """
        Callback สำหรับ raw_chat_logger: อัปเดต path index หรือ validate session ใหม่
        """
        try:
            pm = PathManager()
            # ตัวอย่าง: pm.validate_path(str(file_path))
            print(f"[PathManager] (stub) Would update/validate path for: {rel_path}")
        except Exception as e:
            print(f"[PathManager] Failed to update/validate path: {e}")

if __name__ == "__main__":
    # ตัวอย่างการใช้งาน
    pm = PathManager(root_dir="../memory_core") # ปรับ path ให้ถูกตอนทดสอบ
    meta_mgr = MemoryMetaManager(memory_json_path="../memory/memory.json") # ปรับ path
