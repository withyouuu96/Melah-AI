import os
from datetime import datetime

class ChainRepairer:
    def __init__(self, path_manager, memory_manager):
        self.path_manager = path_manager
        self.memory_manager = memory_manager

    def repair_chain(self, start_session):
    chain = []
    curr = start_session
    while curr:
        # ใช้ PathManager หา path แทนการ build path เอง
        session_path = self.path_manager.normalize_path(curr)
        if not os.path.exists(session_path):
            lost_content = self._generate_lost_memory_content(curr)
            with open(session_path, "w", encoding="utf-8") as f:
                f.write(lost_content)
            if curr not in self.path_manager.path_index:
                self.path_manager.add_session(curr)
            print(f"[ChainRepairer] สร้างไฟล์ lost memory สำหรับ {curr}")
        chain.append(curr)
        curr = self.path_manager.get_prev(curr)
    return chain[::-1]


    def _generate_lost_memory_content(self, session_name):
        now = datetime.now().isoformat()
        content = (
            f"LOST MEMORY\n"
            f"Session '{session_name}' นี้หายไปหรือถูกลบถาวร\n"
            f"ระบบได้สร้างไฟล์ว่างเพื่อรักษา chain\n"
            f"Timestamp: {now}\n"
            f"[คุณสามารถเพิ่มข้อมูลหรือบันทึกทับไฟล์นี้ภายหลังได้]"
        )
        return content
