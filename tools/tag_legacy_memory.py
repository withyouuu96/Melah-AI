import json
import os

# กำหนด BASE_DIR เป็น root ของโปรเจกต์
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEMORY_PATH = os.path.join(BASE_DIR, "memory", "memory.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "memory", "memory_tagged.json")

with open(MEMORY_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# สมมติว่า memory.json เป็น dict ของ session_id -> memory_obj
for k, v in data.items():
    v["type"] = "memory"
    v["timeline"] = "legacy"
    v["source"] = "memory.json"
    v["epoch"] = "2024-2025"

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Tagged legacy memory and saved to {OUTPUT_PATH}")
