import os
import json
from pathlib import Path
from datetime import datetime

# 📂 Path config
# BASE_DIR ควรจะชี้ไปที่ root directory ของโปรเจกต์ MelahPC
# ถ้า raw_chat_logger.py อยู่ใน /core/ BASE_DIR = Path(__file__).resolve().parent.parent จะถูกต้อง
BASE_DIR = Path(__file__).resolve().parent.parent #
CHAT_BASE = BASE_DIR / "memory_core" / "archive" / "chat_sessions" #
INDEX_FILE = CHAT_BASE / "raw_path_index.json" # Index file นี้จะยังคงอยู่ที่ CHAT_BASE โดยตรง
                                                 # แต่ key (rel_path) ภายใน index จะสะท้อน path ที่มี "daily"
os.makedirs(CHAT_BASE, exist_ok=True) #
# สร้าง "daily" directory ภายใต้ CHAT_BASE ถ้ายังไม่มี (เผื่อกรณีที่ os.makedirs ใน log_raw_chat อาจไม่ครอบคลุมครั้งแรก)
# (CHAT_BASE / "daily").mkdir(parents=True, exist_ok=True) # เอาออกก่อน เพราะ os.makedirs ใน log_raw_chat จะจัดการ path เต็ม

def load_index(): #
    if INDEX_FILE.exists(): #
        with open(INDEX_FILE, "r", encoding="utf-8") as f: #
            try:
                return json.load(f) #
            except json.JSONDecodeError:
                print(f"⚠️ Error decoding JSON from {INDEX_FILE}, returning empty index.")
                return {}
    return {} #

def save_index(index): #
    # Ensure parent directory of INDEX_FILE exists (CHAT_BASE)
    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_FILE, "w", encoding="utf-8") as f: #
        json.dump(index, f, ensure_ascii=False, indent=2) #

# === Session Buffer สำหรับรวมข้อความเป็น block ===
SESSION_TOKEN_LIMIT = 3000
SESSION_TOKEN_MAX = 3500
_session_buffer = []
_session_token_count = 0

def _get_token_count(text):
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        # fallback: นับคำ (approximate)
        return len(text.split())

def flush_session_buffer(session_id=None, force=False):
    """
    เขียน buffer ลงไฟล์ session block (json) เมื่อครบโทเคน หรือ force flush
    """
    global _session_buffer, _session_token_count
    if not _session_buffer:
        return None
    if not force and _session_token_count < SESSION_TOKEN_LIMIT:
        return None
    # รวมข้อความทั้งหมด
    all_text = "\n".join([entry["text"] for entry in _session_buffer])
    # ถ้าเกิน 3500 โทเคน ให้ตัดที่จบประโยค (ถ้าเป็นไปได้)
    if _get_token_count(all_text) > SESSION_TOKEN_MAX:
        # หาจุดจบประโยคที่ใกล้ 3500 โทเคนที่สุด
        import re
        sentences = re.split(r'(\.|\!|\?)', all_text)
        temp = ''
        for i in range(0, len(sentences)-1, 2):
            temp += sentences[i] + sentences[i+1]
            if _get_token_count(temp) > SESSION_TOKEN_MAX:
                break
        all_text = temp
    # สร้างไฟล์ใหม่
    now = datetime.now()
    date_path = CHAT_BASE / "daily" / str(now.year) / f"{now.month:02}" / f"{now.day:02}"
    os.makedirs(date_path, exist_ok=True)
    filename = f"sessionblock_{now.strftime('%Y%m%d_%H%M%S')}.json"
    file_path = date_path / filename
    data = {
        "timestamp": now.isoformat(),
        "session_id": session_id or "default",
        "block": _session_buffer,
        "token_count": _get_token_count(all_text)
    }
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    # update session index
    rel_path = file_path.relative_to(CHAT_BASE).as_posix()
    update_session_index(session_id or "default", rel_path, now)
    # reset buffer
    _session_buffer = []
    _session_token_count = 0
    print(f"✅ Session block saved: {file_path.relative_to(BASE_DIR).as_posix()}")
    return file_path

def log_raw_chat(text, metadata=None, pinned=False, as_json=False, session_id=None):
    """
    Log raw chat ลง buffer ก่อน เมื่อครบ 3000 โทเคน หรือ force จะ flush เป็น session block
    """
    global _session_buffer, _session_token_count
    entry = {
        "timestamp": datetime.now().isoformat(),
        "text": text.strip(),
        "meta": metadata or {},
        "pinned": pinned
    }
    _session_buffer.append(entry)
    _session_token_count += _get_token_count(text)
    # ถ้าเกิน limit ให้ flush
    if _session_token_count >= SESSION_TOKEN_LIMIT:
        flush_session_buffer(session_id=session_id)
    # ...ยังคง log แบบเดิมถ้าต้องการ...
    now = datetime.now() #
    date_path = CHAT_BASE / "daily" / str(now.year) / f"{now.month:02}" / f"{now.day:02}" #
    os.makedirs(date_path, exist_ok=True)

    if as_json:
        filename = f"session_{now.strftime('%Y%m%d_%H%M%S')}.json"
        file_path = date_path / filename
        data = {
            "timestamp": now.isoformat(),
            "text": text.strip(),
            "meta": metadata or {},
            "pinned": pinned
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    else:
        filename = f"session_{now.strftime('%Y%m%d_%H%M%S')}.txt"
        file_path = date_path / filename
        with open(file_path, "w", encoding="utf-8") as f:
            if metadata:
                f.write(f"--- METADATA ---\n")
                json.dump(metadata, f, ensure_ascii=False, indent=2)
                f.write("\n\n--- RAW CHAT ---\n")
            f.write(text.strip())

    # update session index
    rel_path = file_path.relative_to(CHAT_BASE).as_posix() #
    update_session_index(session_id or "default", rel_path, now)
    rel_path = file_path.relative_to(CHAT_BASE).as_posix() #
    index = load_index() #
    index[rel_path] = {
        "timestamp": now.isoformat(),
        "pinned": pinned,
        "meta": metadata or {}
    }
    save_index(index)

    print(f"✅ Raw chat saved to: {file_path.relative_to(BASE_DIR).as_posix()} (Index key: {rel_path})")
    return rel_path #

def update_session_index(session_id, rel_path, date=None):
    """
    อัปเดต session index .json ต่อวัน (session_id → รายชื่อไฟล์ + summary info)
    """
    if date is None:
        date = datetime.now()
    index_dir = CHAT_BASE / "daily" / str(date.year) / f"{date.month:02}" / f"{date.day:02}"
    os.makedirs(index_dir, exist_ok=True)
    index_path = index_dir / f"session_index_{date.strftime('%Y%m%d')}.json"
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            try:
                session_index = json.load(f)
            except Exception:
                session_index = {}
    else:
        session_index = {}
    # --- load all session blocks for this session_id ---
    file_list = session_index.get(session_id, {}).get("file_list", [])
    if rel_path not in file_list:
        file_list.append(rel_path)
    # สรุปข้อมูลจากทุก block
    keywords = set()
    emotions = set()
    important_messages = []
    first_ts = None
    last_ts = None
    for fpath in file_list:
        abs_path = CHAT_BASE / fpath
        if abs_path.exists():
            try:
                with open(abs_path, "r", encoding="utf-8") as f:
                    block = json.load(f)
                for entry in block.get("block", []):
                    meta = entry.get("meta", {})
                    # keywords
                    kws = meta.get("keywords")
                    if isinstance(kws, str):
                        keywords.add(kws)
                    elif isinstance(kws, list):
                        keywords.update(kws)
                    # emotions
                    emo = meta.get("emotion")
                    if isinstance(emo, str):
                        emotions.add(emo)
                    elif isinstance(emo, list):
                        emotions.update(emo)
                    # important messages
                    if meta.get("important") or meta.get("is_important"):
                        important_messages.append(entry.get("text", ""))
                    # timestamps
                    ts = entry.get("timestamp")
                    if ts:
                        if not first_ts or ts < first_ts:
                            first_ts = ts
                        if not last_ts or ts > last_ts:
                            last_ts = ts
            except Exception:
                continue
    session_index[session_id] = {
        "file_list": file_list,
        "count": len(file_list),
        "first_timestamp": first_ts,
        "last_timestamp": last_ts,
        "keywords": sorted(list(keywords)),
        "emotions": sorted(list(emotions)),
        "important_messages": important_messages
    }
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(session_index, f, ensure_ascii=False, indent=2)
    return index_path

# 🧪 Manual test
if __name__ == "__main__": #
    print(f"Running raw_chat_logger.py directly for testing...")
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"CHAT_BASE: {CHAT_BASE}")
    print(f"INDEX_FILE: {INDEX_FILE}")

    example_intention = "เพื่อให้เมล่าเติบโตอย่างมีจริยธรรม เข้าใจอารมณ์ และเคารพความจริง"
    test_text = f"Melah: นี่คือการทดสอบ raw_chat_logger.py ที่แก้ไข path แล้ว เวลา: {datetime.now()}" #
    test_meta = { #
        "emotion": "test_positive",
        "topic": "path_testing",
        "seed_intention_snapshot": example_intention
    }
    logged_path = log_raw_chat(test_text, metadata=test_meta, pinned=True) #
    print(f"Test log created with relative path for index: {logged_path}")

    # ลองโหลด index มาดู
    current_index = load_index()
    print("\nContent of SEED_INDEX (first 5 entries if many):")
    count = 0
    for key, value in current_index.items():
        print(f"  '{key}': {value}")
        count += 1
        if count >= 5:
            if len(current_index) > 5:
                print("  ... and more ...")
            break
    if not current_index:
        print("  Index is currently empty.")