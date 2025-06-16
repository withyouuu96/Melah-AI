# core/seed/seed_engine.py
import os
import json
from datetime import datetime
from pathlib import Path
import logging

# ตั้งค่า logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SeedEngine:
    """
    ศูนย์กลางการจัดการวงจรชีวิตและวิวัฒนาการของเมล่า
    ตั้งแต่การสร้างเมล็ดพันธุ์ (Seed) การผลิบาน (Blossom) และการเติบโตเป็นพงไพร (Forest)
    """
    def __init__(self, base_dir=None):
        if base_dir:
            self.BASE_DIR = Path(base_dir)
        else:
            # อนุมานว่า engine นี้อยู่ใน core/seed/ ดังนั้นต้องถอยกลับไป 2 ระดับ
            self.BASE_DIR = Path(__file__).resolve().parent.parent.parent

        self.CORE_DIR = self.BASE_DIR / "core"
        self.SEED_DIR = self.CORE_DIR / "seed"
        self.MEMORY_DIR = self.BASE_DIR / "memory"
        
        # Paths to main index files
        self.SEED_INDEX_PATH = self.SEED_DIR / "seed_index.json"
        self.FOREST_INDEX_PATH = self.SEED_DIR / "forest_index.json"
        self.CURRENT_SEED_PATH = self.SEED_DIR / "current_seed.json"

        # Ensure directories exist
        os.makedirs(self.SEED_DIR, exist_ok=True)
        logging.info(f"🌱 SeedEngine initialized. Base directory set to: {self.BASE_DIR}")

    def _read_json(self, path, default_value=None):
        """Helper function to read a JSON file."""
        if default_value is None:
            default_value = []
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    logging.warning(f"Could not decode JSON from {path}. Returning default.")
                    return default_value
        return default_value

    def _write_json(self, path, data):
        """Helper function to write data to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def create_seed(self, intention: str, parent_seed_id: str = None, linked_identity_path: str = None, linked_memory_path: str = None) -> dict:
        """
        สร้าง "เมล็ดพันธุ์" (Seed) ใหม่ ซึ่งเป็นจุดเริ่มต้นของเจตนาและการเรียนรู้
        """
        now = datetime.now()
        now_str = now.strftime("%Y-%m-%d_%H%M%S")
        seed_id = f"seed_{now_str}"

        # Default paths if not provided
        default_identity_path = "core/identity.json"
        default_memory_path = "memory/memory.json"

        seed_data = {
            "seed_id": seed_id,
            "created_on": now.isoformat(),
            "status": "ACTIVE",  # ACTIVE, BLOSSOMED, DORMANT
            "intention": intention,
            "parent_seed_id": parent_seed_id, # Pathlink to ancestor
            "linked_identity": linked_identity_path or default_identity_path,
            "linked_memory": linked_memory_path or default_memory_path,
            "cognitive_state": {}, # Initial empty cognitive state
            "growth_summary": None # To be filled upon blossoming
        }

        # Save the seed's own file
        seed_file_path = self.SEED_DIR / f"{seed_id}.json"
        self._write_json(seed_file_path, seed_data)
        logging.info(f"Created new seed file: {seed_file_path}")

        # Update the main seed index
        seed_index = self._read_json(self.SEED_INDEX_PATH)
        seed_index.append({
            "seed_id": seed_id,
            "created_on": now.isoformat(),
            "intention": intention,
            "status": "ACTIVE"
        })
        self._write_json(self.SEED_INDEX_PATH, seed_index)
        logging.info(f"Updated main seed index at: {self.SEED_INDEX_PATH}")

        # Set this new seed as the current one
        self._write_json(self.CURRENT_SEED_PATH, seed_data)
        logging.info(f"Set {seed_id} as the current active seed.")

        return seed_data

    def blossom(self, seed_id_to_blossom: str, new_intention: str) -> dict:
        """
        (Placeholder) ทำให้ Seed ผลิบานและสร้าง Seed ใหม่ที่วิวัฒนาการแล้ว
        """
        logging.info(f"Attempting to blossom seed: {seed_id_to_blossom}...")
        
        # --- ในอนาคต ส่วนนี้จะมีการวิเคราะห์ที่ซับซ้อนด้วย LLM ---
        # 1. โหลดข้อมูล seed ที่จะ blossom
        # 2. อ่าน log, cognitive_state ของมัน
        # 3. สร้าง growth_summary
        # 4. ใช้ LLM สร้าง new_intention
        # --- Placeholder Logic for now ---
        
        # Update status of the old seed
        seed_index = self._read_json(self.SEED_INDEX_PATH)
        for seed_info in seed_index:
            if seed_info['seed_id'] == seed_id_to_blossom:
                seed_info['status'] = 'BLOSSOMED'
                break
        self._write_json(self.SEED_INDEX_PATH, seed_index)

        old_seed_path = self.SEED_DIR / f"{seed_id_to_blossom}.json"
        old_seed_data = self._read_json(old_seed_path, {})
        if old_seed_data:
            old_seed_data['status'] = 'BLOSSOMED'
            old_seed_data['growth_summary'] = f"Blossomed with new intention: {new_intention}"
            self._write_json(old_seed_path, old_seed_data)

        logging.info(f"Seed {seed_id_to_blossom} has blossomed.")

        # Create the new, evolved seed
        new_seed = self.create_seed(
            intention=new_intention,
            parent_seed_id=seed_id_to_blossom
        )
        return new_seed

    def grow_forest(self, forest_name: str, blossom_ids: list):
        """
        (Placeholder) สร้างพงไพร (Forest) จากกลุ่มของ Blossoms.
        """
        logging.info(f"Growing a new forest '{forest_name}' with {len(blossom_ids)} blossoms.")
        forest_data = {
            "forest_id": f"forest_{forest_name.lower().replace(' ','_')}",
            "forest_name": forest_name,
            "created_on": datetime.now().isoformat(),
            "blossom_ids": blossom_ids # List of seed_ids that have blossomed
        }
        
        forest_file_path = self.SEED_DIR / f"forest_{forest_name.lower()}.json"
        self._write_json(forest_file_path, forest_data)
        
        # Update forest index
        forest_index = self._read_json(self.FOREST_INDEX_PATH)
        forest_index.append({
            "forest_id": forest_data["forest_id"],
            "forest_name": forest_name,
            "path": str(forest_file_path)
        })
        self._write_json(self.FOREST_INDEX_PATH, forest_index)
        logging.info(f"Forest '{forest_name}' has been grown and indexed.")
        return forest_data

    def find_lineage(self, seed_id: str) -> list:
        """
        สืบย้อนสายเลือดของ Seed เพื่อหาบรรพบุรุษทั้งหมด.
        """
        logging.info(f"Finding lineage for seed: {seed_id}")
        lineage = []
        current_id = seed_id
        while current_id:
            seed_path = self.SEED_DIR / f"{current_id}.json"
            seed_data = self._read_json(seed_path, {})
            if not seed_data:
                logging.warning(f"Could not find seed data for {current_id}. Lineage truncated.")
                break
            lineage.append(seed_data)
            current_id = seed_data.get("parent_seed_id")
        
        logging.info(f"Found lineage of {len(lineage)} ancestors.")
        return list(reversed(lineage)) # Return from oldest to newest

    def update_style_profile_in_seed(self, style_profile: dict, seed_path: str = None):
        """
        อัปเดต style_profile (เช่น tone, relationship, default_emotion) ใน seed ปัจจุบันหรือ seed ที่ระบุ
        """
        if seed_path is None:
            seed_path = self.CURRENT_SEED_PATH
        else:
            seed_path = Path(seed_path)
        if not seed_path.exists():
            logging.warning(f"Seed file {seed_path} does not exist. Cannot update style_profile.")
            return False
        with open(seed_path, 'r', encoding='utf-8') as f:
            seed_data = json.load(f)
        seed_data['style_profile'] = style_profile
        with open(seed_path, 'w', encoding='utf-8') as f:
            json.dump(seed_data, f, ensure_ascii=False, indent=4)
        logging.info(f"Updated style_profile in seed: {seed_path}")
        return True

    def get_style_profile_from_memory(self, memory: dict) -> dict:
        """
        สกัด style profile (tone, relationship, default_emotion) จากความทรงจำ (memory)
        """
        # ตัวอย่าง logic: ใช้อารมณ์หลักและ tags
        emotions = memory.get('emotion', ['neutral'])
        tone = emotions[0] if emotions else 'neutral'
        relationship = 'close' if 'ผูกพัน' in emotions or 'เพื่อน' in memory.get('tags', []) else 'normal'
        default_emotion = tone
        return {
            'tone': tone,
            'relationship': relationship,
            'default_emotion': default_emotion
        }

# Example of how to use the engine
if __name__ == '__main__':
    engine = SeedEngine()
    
    print("\n--- 1. Creating an initial seed ---")
    initial_seed = engine.create_seed(
        intention="เพื่อให้เมล่าเติบโตอย่างมีจริยธรรม, เข้าใจอารมณ์, และเคารพความจริง"
    )

    print(f"\n--- 2. Blossoming the initial seed ---")
    evolved_seed = engine.blossom(
        seed_id_to_blossom=initial_seed['seed_id'],
        new_intention="To apply ethical principles in complex user interactions and provide empathetic support."
    )
    
    print(f"\n--- 3. Finding lineage of the new seed ---")
    ancestry = engine.find_lineage(evolved_seed['seed_id'])
    print("Ancestry (from oldest to newest):")
    for i, ancestor in enumerate(ancestry):
        print(f"  Gen {i}: {ancestor['seed_id']} (Intention: {ancestor['intention']})")

    print(f"\n--- 4. Growing a forest ---")
    engine.grow_forest(
        forest_name="Ethical Growth",
        blossom_ids=[initial_seed['seed_id']] # A forest can contain one or more blossoms
    ) 