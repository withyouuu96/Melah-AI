import os
import json

class MemoryManager:
    def __init__(self, session_dir, summary_dir, bookmark_file, memory_json_path):
        self.session_dir = session_dir
        self.summary_dir = summary_dir
        self.bookmark_file = bookmark_file
        self.memory_json_path = memory_json_path

    def list_sessions(self):
        return [f for f in os.listdir(self.session_dir) if f.endswith('.txt')]

    def read_session(self, filename):
        path = os.path.join(self.session_dir, filename)
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()

    def list_summaries(self):
        return [f for f in os.listdir(self.summary_dir) if f.endswith('.txt')]

    def read_summary(self, filename):
        path = os.path.join(self.summary_dir, filename)
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()

    def read_bookmarks(self):
        with open(self.bookmark_file, 'r', encoding='utf-8') as f:
            return f.read()

    def read_memory_json(self):
        with open(self.memory_json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    # เชื่อม batch/summary → session → bookmarks → memory.json
    def get_chain_context(self, summary_filename):
        """ดึงเนื้อหาความทรงจำทั้งหมดที่เชื่อมโยงใน batch/summary"""
        result = {}
        summary_content = self.read_summary(summary_filename)
        session_files = [line.strip() for line in summary_content.splitlines() if line.strip()]
        session_contents = []
        for session_file in session_files:
            fname = os.path.basename(session_file)
            session_content = self.read_session(fname) if os.path.exists(os.path.join(self.session_dir, fname)) else ""
            session_contents.append({
                "session_file": fname,
                "content": session_content
            })
        result['summary'] = summary_content
        result['sessions'] = session_contents
        result['bookmarks'] = self.read_bookmarks()
        result['memory_json'] = self.read_memory_json()
        return result
