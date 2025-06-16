from .memory_meta_manager import MemoryMetaManager

class MemoryPathBridge:
    """
    ตัวกลางเชื่อมโยงระบบความจำทุกยุคของ MelahPC
    ดึงเนื้อหาความจำผ่าน meta chain (memory.json) เพื่อความยืดหยุ่น
    """

    def __init__(self, path_manager, meta_manager=None):
        self.path_manager = path_manager
        # รับ meta_manager จากภายนอก ถ้าไม่มีให้สร้างเอง (กรณี stand-alone)
        if meta_manager is None:
            self.meta_manager = MemoryMetaManager("memory/memory.json")
        else:
            self.meta_manager = meta_manager

    def get_session_chain(self, start_session, max_depth=5):
        """
        เดิน meta chain จาก memory.json โดยใช้ meta_manager
        คืนลิสต์ path ของ session ทั้งหมด (ลำดับจากโซ่ meta)
        """
        chain_nodes = self.meta_manager.walk_chain(start_session, max_depth=max_depth)
        session_chain = [node["path"] for node in chain_nodes if "path" in node]
        return session_chain

    def get_sessions_content(self, session_chain):
        """
        รับลิสต์ path session/summary (จาก meta chain)
        คืน list เนื้อหา (text) ของแต่ละ session/summary
        """
        contents = []
        for session_file in session_chain:
            content = self.path_manager.read_session(session_file)
            if content is not None:
                contents.append(content)
        return contents

    def split_full_sentences(self, text, max_tokens=2048):
        """
        ตัดข้อความเป็นก้อน ๆ (chunk) ไม่เกิน max_tokens เพื่อส่งเข้า LLM
        """
        import tiktoken  # ต้องติดตั้ง tiktoken หรือเปลี่ยนวิธีนับ token ตามระบบ
        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        sentences = text.split('\n')
        chunks, curr_chunk = [], []
        curr_tokens = 0

        for sent in sentences:
            tokens = len(enc.encode(sent))
            if curr_tokens + tokens > max_tokens:
                chunks.append('\n'.join(curr_chunk))
                curr_chunk, curr_tokens = [], 0
            curr_chunk.append(sent)
            curr_tokens += tokens
        if curr_chunk:
            chunks.append('\n'.join(curr_chunk))
        return chunks

    def get_chain_chunks(self, start_session, max_depth=5, max_tokens=2048):
        """
        ดึง meta chain, รวมเนื้อหา, แล้วแบ่ง chunk เพื่อป้อนเข้า LLM (ไม่เกิน max_tokens ต่อก้อน)
        """
        session_chain = self.get_session_chain(start_session, max_depth=max_depth)
        contents = self.get_sessions_content(session_chain)
        full_text = "\n".join(contents)
        return self.split_full_sentences(full_text, max_tokens=max_tokens)
