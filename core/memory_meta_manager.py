import json

class MemoryMetaManager:
    """
    จัดการ meta memory (event, summary, chain, insight) จาก memory.json
    พร้อมฟังก์ชันเดิน chain ตาม related, และค้นข้อมูล meta
    """
    def __init__(self, memory_json_path="memory/memory.json"):
        self.memory_json_path = memory_json_path  # Store the path
        try:
            with open(self.memory_json_path, "r", encoding="utf-8") as f:
                self.memory_dict = json.load(f)
        except FileNotFoundError:
            self.memory_dict = {} # Initialize with empty dict if file not found
            # Optionally, log this event or raise a more specific error
            print(f"Warning: Memory file not found at {self.memory_json_path}, initializing with empty memory.")
        except json.JSONDecodeError:
            self.memory_dict = {} # Initialize with empty dict if file is corrupted
            print(f"Warning: Memory file at {self.memory_json_path} is corrupted, initializing with empty memory.")

    def get_event(self, session_id):
        """คืน meta event (node) ตาม session_id (เช่น 'session_001')"""
        return self.memory_dict.get(session_id, None)

    def walk_chain(self, start_session, max_depth=5):
        """
        เดินสายโซ่ความทรงจำ (chain) จาก start_session
        max_depth = ความลึกสูงสุดที่เดินได้ (กัน infinite loop)
        """
        chain = []
        curr = start_session
        for _ in range(max_depth):
            node = self.get_event(curr)
            if not node:
                break
            chain.append(node)
            if node.get("related"):
                curr = node["related"][0]  # ไปต่อที่ related ตัวแรก
            else:
                break
        return chain

    def find_by_tag(self, tag):
        """
        คืนลิสต์ node (event) ที่มี tag ตรงกับที่ระบุ
        """
        results = []
        for node in self.memory_dict.values():
            if "tags" in node and tag in node["tags"]:
                results.append(node)
        return results

    def semantic_search(self, keyword):
        """
        ค้นเหตุการณ์/summary ที่มี keyword นั้น ๆ
        """
        results = []
        if not self.memory_dict: # Check if memory_dict is empty or None
            return results
        for node in self.memory_dict.values():
            haystack = " ".join([
                node.get("event", ""),
                node.get("summary", ""),
                node.get("insight", ""),
            ])
            if keyword in haystack:
                results.append(node)
        return results

    def save_memory_meta(self):
        """
        บันทึก memory_dict (ที่อาจมีการแก้ไขแล้ว) กลับไปยัง memory_json_path
        """
        try:
            with open(self.memory_json_path, "w", encoding="utf-8") as f:
                json.dump(self.memory_dict, f, indent=2, ensure_ascii=False)
            print(f"Successfully saved memory to {self.memory_json_path}")
        except Exception as e:
            print(f"Error saving memory to {self.memory_json_path}: {e}")
            # Potentially re-raise or log to a more persistent error log
