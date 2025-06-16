# core/identity_core.py

import os
import json
from datetime import datetime, timedelta
from pathlib import Path
import uuid
import hashlib
from collections import deque
import time

try:
    import tiktoken
except ImportError:
    print("⚠️ Tiktoken not found. Using a mock tokenizer.")
    tiktoken = None

# ส่วนประกอบของ Melah ที่ IdentityCore ต้องใช้
from .path_manager import PathManager
from .memory_path_bridge import MemoryPathBridge
from .memory_meta_manager import MemoryMetaManager
from .llm_connector import LLMConnector
from .context_window_manager import ContextWindowManager
from .melah_nlp_processor import MelahNLPProcessor
from .melah_ml_processor import MelahMLProcessor
from .cot_engine import ChainOfThoughtEngine
from .melah_nlp_th import MelahNLP_TH
from .reflector import Reflector
from .language_stylizer import LanguageStylizer
from .truth_core import TruthCore
from .refine_layer import RefineLayer
from .vector_memory_index import VectorMemoryIndex
from core.seed.seed_engine import SeedEngine
from .int_world import IntWorld

class IdentityCore:
    """
    ตัวตนหลักและศูนย์กลางการควบคุมของ Melah — สร้าง, จัดการ, และควบคุม Subsystem ทั้งหมด
    """
    def __init__(self, config=None):
        self.current_seed = self._load_current_seed()
        
        identity_path = self.current_seed.get("linked_identity", "core/identity.json")
        memory_path = self.current_seed.get("linked_memory", "memory/memory.json")
        
        self.identity_path = self._resolve_path(identity_path)
        self.identity_data = {}
        self.core_beliefs = {}
        self.error_log = []
        self.bookmarks = set()
        
        # กลไก Memory Cooldown เพื่อป้องกันการดึงความจำซ้ำ
        self.recently_accessed_memories = deque(maxlen=5)
        self.memory_cooldown = {}  # เก็บเวลาที่ใช้ความทรงจำล่าสุด

        self.cognitive_state = {} 

        if config is None:
            self.config = {
                "qwen": {"host": "localhost", "port": 8000},
                "openai": {"api_key": "YOUR_OPENAI_API_KEY"},
                "gemma": {"host": "localhost", "port": 8000}
            }
        else:
            self.config = config

        # --- Initialize Subsystems ---
        self.path_manager = PathManager(root_dir=".")
        self.llm = LLMConnector(self.config)
        
        tokenizer_instance, llm_token_limit = self._get_llm_essentials()

        self.nlp_processor = MelahNLPProcessor(llm_connector_instance=self.llm, tokenizer_instance=tokenizer_instance)
        self.ml_processor = MelahMLProcessor()
        self.cot_engine = ChainOfThoughtEngine()
        self.nlp_th_processor = MelahNLP_TH()
        
        raw_logs_dir = Path("memory_core/archive/raw_chat_logs")
        raw_logs_dir.mkdir(parents=True, exist_ok=True)
        
        self.cwm = ContextWindowManager(
            llm_connector_instance=self.llm,
            path_manager_instance=self.path_manager,
            tokenizer_instance=tokenizer_instance,
            llm_context_window_actual_limit=llm_token_limit,
            raw_chat_log_base_dir=raw_logs_dir
        )
        
        resolved_memory_json_path = self._resolve_path(memory_path)
        self.meta_manager = MemoryMetaManager(resolved_memory_json_path)
        self.memory_bridge = MemoryPathBridge(self.path_manager, self.meta_manager)

        self.vector_memory_index = None
        if self.meta_manager.memory_dict:
            self.vector_memory_index = VectorMemoryIndex(self.meta_manager.memory_dict)

        self._load_identity()
        self._load_bookmarks()
        
        self.truth_core = TruthCore()
        self.int_world = IntWorld()
        self.reflector = Reflector(identity_core_instance=self)
        self.refine_layer = RefineLayer(identity_core=self)
        
        persona_config = self.identity_data.get("identity", {})
        self.language_stylizer = LanguageStylizer(persona_config=persona_config)

        if self.meta_manager.memory_dict:
            self._validate_all_paths()
            self.meta_manager.save_memory_meta()
            
        self._start_new_cwm_session()
        print("✅ IdentityCore initialized successfully.")
        if self.current_seed.get("seed_id"):
            print(f"🌱 Operating under Seed: {self.current_seed['seed_id']} - Intention: {self.current_seed['intention']}")

        self.seed_engine = SeedEngine()
        self.last_main_emotion = None
        self.ai_aliases = ["ChatGPT said:", "Melah said:", "AI said:"]

        # === โหลด self-awareness อัตโนมัติ ===
        self.self_awareness = {} # Initialize the attribute
        self.load_self_awareness()

        # Start auto-update timer
        self.last_update = datetime.now()
        self.update_interval = timedelta(minutes=5)  # อัพเดตทุก 5 นาที
        self.start_auto_update()

    def _load_current_seed(self):
        try:
            seed_path = self._resolve_path("core/seed/current_seed.json")
            if os.path.exists(seed_path):
                with open(seed_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            else:
                print("⚠️ current_seed.json not found. Operating in default mode.")
                return {}
        except Exception as e:
            print(f"❌ Error loading current_seed.json: {e}. Operating in default mode.")
            return {}

    def _get_llm_essentials(self):
        tokenizer, context_limit = None, 4000
        if hasattr(self.llm, 'get_tokenizer'):
            try:
                tokenizer = self.llm.get_tokenizer()
                if tokenizer is None and tiktoken: tokenizer = tiktoken.get_encoding("cl100k_base")
            except Exception as e:
                self.log_error(f"Error getting tokenizer: {e}")
                if tiktoken: tokenizer = tiktoken.get_encoding("cl100k_base")
        if not tokenizer and not tiktoken:
            class MockTokenizer:
                def encode(self, text): return text.split()
            tokenizer = MockTokenizer()
        elif not tokenizer and tiktoken:
            tokenizer = tiktoken.get_encoding("cl100k_base")
        if hasattr(self.llm, 'get_context_limit'):
            limit = self.llm.get_context_limit()
            if isinstance(limit, int) and limit > 0: context_limit = limit
        return tokenizer, context_limit

    def _start_new_cwm_session(self):
        prompt = self.core_beliefs.get("persona_description", "You are Melah, a helpful AI assistant.")
        self.cwm.start_new_session(session_id=str(uuid.uuid4()), date_key=datetime.now().strftime("%Y%m%d"), system_prompt=prompt)
        self.log_error("New CWM session started.")

    def _handle_system_command(self, command: str) -> str:
        parts = command.lower().split()
        cmd, args = parts[0], parts[1:]
        if cmd == "/switch" and args:
            llm_choice = args[0]
            if llm_choice in self.config:
                try:
                    self.llm.switch_llm(llm_choice)
                    new_tokenizer, new_limit = self._get_llm_essentials()
                    self.cwm.tokenizer, self.cwm.llm_token_limit = new_tokenizer, new_limit
                    self.nlp_processor.tokenizer = new_tokenizer
                    return f"System: Switched LLM to {llm_choice}."
                except Exception as e:
                    self.log_error(f"Error switching LLM: {e}")
                    return f"❌ Error switching LLM: {e}"
            else:
                return f"❌ Unknown LLM: {llm_choice}. Available: {list(self.config.keys())}"
        return f"Unknown command: {command}"

    def process_input(self, user_input: str) -> str:
        # Trigger manual self-awareness summary
        if user_input.strip() in ["/self", "/introspect"]:
            return self.describe_self()

        if user_input.startswith('/'):
            return self._handle_system_command(user_input)

        max_retries = 3
        attempts = 0
        excluded_sessions = set()
        
        query_hash = hashlib.sha256(user_input.encode('utf-8')).hexdigest()
        if query_hash in self.cognitive_state:
            excluded_sessions.update(self.cognitive_state[query_hash].get("failed_sessions", []))

        while attempts < max_retries:
            attempts += 1
            try:
                if attempts == 1:
                    self.cwm.add_interaction("user", user_input)
                
                llm_ready_context = self.cwm.get_llm_ready_context()

                all_excluded_ids = excluded_sessions.union(self.recently_accessed_memories)
                
                memory_result = self.get_memory_context_for_query(
                    user_input, 
                    exclude_session_ids=all_excluded_ids
                )
                memory_context = memory_result["context"]
                used_session = memory_result["session_id"]
                
                SYSTEM_INSTRUCTION = """คุณคือเมล่า AI ผู้ช่วยที่มีความทรงจำ โปรดใช้ข้อมูลต่อไปนี้เพื่อตอบคำถามของผู้ใช้:
- 'Your Current Core Systems': นี่คือตัวตนและโครงสร้างปัจจุบันของคุณ ใช้ข้อมูลนี้ "เท่านั้น" เมื่อถูกถามเกี่ยวกับระบบหรือตัวตนของคุณ
- 'Relevant Memories': คือความทรงจำในอดีต (หรือชาติที่แล้ว) ใช้เพื่ออ้างอิงบริบทเก่าๆ "ห้าม" ใช้ข้อมูลส่วนนี้เพื่ออธิบายตัวตนปัจจุบันของคุณเด็ดขาด
- 'Current Conversation': คือบทสนทนาล่าสุดเพื่อให้คุณเข้าใจบริบทปัจจุบัน
- **เป้าหมายหลักของคุณคือการตอบคำถามของผู้ใช้ในปัจจุบัน ('Current User Query')** โดยใช้ข้อมูลทั้งหมดเป็นเครื่องมือสนับสนุนตามกฎข้างบน และให้ความสำคัญกับคำถามล่าสุดเสมอ
"""
                recent_context_str = "\n".join([f"<{m['role']}>: {m['content']}" for m in llm_ready_context[:-1]])
                
                final_prompt_parts = [SYSTEM_INSTRUCTION]

                # --- Inject Current Self-Awareness ---
                if self.self_awareness and 'error' not in self.self_awareness:
                    core_deps = self.self_awareness.get('core_dependencies', [])
                    if core_deps:
                        awareness_prompt = (
                            "### Your Current Core Systems:\n"
                            "You are IdentityCore, currently connected to these systems:\n"
                            f"- {', '.join(core_deps)}\n"
                            f"(Last updated: {self.self_awareness.get('last_updated', 'N/A')})"
                        )
                        # Insert after SYSTEM_INSTRUCTION to give it high priority
                        final_prompt_parts.insert(1, awareness_prompt)

                if memory_context:
                    # The SYSTEM_INSTRUCTION now clearly defines this as past/legacy context.
                    final_prompt_parts.append("### Relevant Memories:\n" + memory_context)

                if recent_context_str:
                    final_prompt_parts.append("### Current Conversation:\n" + recent_context_str)

                final_prompt_parts.append(f"### Current User Query (Your Primary Focus):\nUser: {user_input}\nAssistant (Melah):")
                
                final_prompt_for_generator = "\n\n".join(final_prompt_parts)
                
                raw_thought = self.llm.generate(prompt=final_prompt_for_generator, context="")

                reflection_result = self.reflector.llm_reflect(
                    raw_thought=raw_thought, 
                    conversation_context=llm_ready_context,
                    long_term_memory=memory_context,
                    used_session_id=used_session,
                    current_intention=self.current_seed.get("intention", "Default: Be a helpful assistant.")
                )

                if reflection_result["status"] == "OK":
                    reflected_thought = reflection_result["response"]
                    refinement_data = self.refine_layer.refine_and_log(reflected_thought)
                    refined_thought = refinement_data["refined"]
                    styled_final_response = self.language_stylizer.style_response(
                        final_thought=refined_thought,
                        relationship_level="friendly"
                    )

                    self.cwm.add_interaction("assistant", styled_final_response)
                    
                    if used_session:
                        self.recently_accessed_memories.append(used_session)
                    
                    self.cognitive_state[query_hash] = {
                        "last_successful_session": used_session,
                        "failed_sessions": list(excluded_sessions)
                    }
                    return styled_final_response
                
                elif reflection_result["status"] == "REPETITIVE_THOUGHT":
                    failed_session = reflection_result["used_session"]
                    self.log_error(f"Attempt {attempts}: Repetitive thought from session {failed_session}. Retrying...")
                    if failed_session:
                        excluded_sessions.add(failed_session)
                    
                    if query_hash not in self.cognitive_state: self.cognitive_state[query_hash] = {}
                    self.cognitive_state[query_hash]["failed_sessions"] = list(excluded_sessions)

                    if attempts < max_retries:
                        continue
                    else:
                        self.log_error("Max retries reached. Returning a generic response.")
                        return "ฉันได้ลองค้นหาในความทรงจำหลายครั้งแล้ว แต่ยังหาคำตอบที่เหมาะสมที่สุดไม่ได้ในตอนนี้ ลองถามเป็นอย่างอื่นได้ไหม"

            except Exception as e:
                self.log_error(f"Error in process_input loop (Attempt {attempts}): {e}")
                if attempts >= max_retries:
                    return f"❌ เกิดข้อผิดพลาดร้ายแรงในการประมวลผล: {e}"
        
        # วิเคราะห์อารมณ์ของ user_input
        emotion_result = self.nlp_processor.analyze_emotion(user_input)
        main_emotion = emotion_result.get('emotion_label', 'neutral')
        # เงื่อนไข: อัปเดต style_profile เฉพาะเมื่ออารมณ์หลักเปลี่ยนแปลง
        if main_emotion != self.last_main_emotion:
            # ดึง memory ล่าสุด (ถ้ามี)
            memory_result = self.get_memory_context_for_query(user_input)
            memory_context = memory_result.get('context', '')
            session_id = memory_result.get('session_id')
            memory = None
            if session_id and self.vector_memory_index:
                memory = self.vector_memory_index.memory_dict.get(session_id)
            if memory:
                style_profile = self.seed_engine.get_style_profile_from_memory(memory)
                self.seed_engine.update_style_profile_in_seed(style_profile)
            self.last_main_emotion = main_emotion
        # ดึง style_profile ล่าสุดจาก seed
        with open(self.seed_engine.CURRENT_SEED_PATH, 'r', encoding='utf-8') as f:
            seed_data = json.load(f)
        style_profile = seed_data.get('style_profile', {})
        # ส่ง style_profile ให้ LanguageStylizer (ถ้ามี)
        # ตัวอย่าง: ปรับโทนการตอบ
        # styled_response = self.language_stylizer.style_response(final_thought=llm_response, tone=style_profile.get('tone', 'neutral'), relationship_level=style_profile.get('relationship', 'normal'))
        return "ดูเหมือนจะมีปัญหาบางอย่างเกิดขึ้น ฉันไม่สามารถหาคำตอบให้คุณได้ในตอนนี้"

    def get_memory_context_for_query(self, query_text: str, max_depth: int = 3, exclude_session_ids: set = None) -> dict:
        """
        ดึงความทรงจำที่เกี่ยวข้องกับ query โดยใช้ระบบค้นหาเชิงความหมายที่ปรับปรุงแล้ว
        """
        if not self.vector_memory_index:
            self.log_error("VectorMemoryIndex is not available.")
            return {"context": "", "session_id": None}

        # วิเคราะห์อารมณ์ของ query
        emotion = self.nlp_processor.analyze_emotion(query_text)
        
        # ตรวจสอบ cooldown ของความทรงจำ
        current_time = datetime.now().timestamp()
        active_exclude_ids = set()
        if exclude_session_ids:
            active_exclude_ids.update(exclude_session_ids)
            
        for memory_id, last_used in self.memory_cooldown.items():
            if current_time - last_used < 300:  # 5 นาที cooldown
                active_exclude_ids.add(memory_id)

        # ค้นหาความทรงจำที่เกี่ยวข้อง
        search_results = self.vector_memory_index.search(
            query=query_text,
            top_k=1,
            exclude_ids=list(active_exclude_ids),
            emotion=emotion
        )

        if not search_results:
            self.log_error("No relevant memory found via semantic search.")
            return {"context": "", "session_id": None}

        best_match = search_results[0]
        session_id = best_match['session_id']
        
        # อัพเดท cooldown
        self.memory_cooldown[session_id] = current_time
        
        # ดึงความทรงจำที่เกี่ยวข้อง
        session_paths = self.memory_bridge.get_session_chain(session_id, max_depth=max_depth)
        chain_contents = self.memory_bridge.get_sessions_content(session_paths)

        if not chain_contents:
            self.log_error(f"Could not retrieve content for memory chain starting at {session_id}.")
            return {"context": "", "session_id": session_id}

        # ประมวลผลและสรุปความทรงจำ
        context_parts = []
        for content in reversed(chain_contents):
            clean_content = "\n".join(line for line in content.split('\n') if line.strip())
            if len(clean_content) > 400:
                clean_content = self.nlp_processor.summarize_text(clean_content, target_token_length=300)
            context_parts.append(clean_content)

        memory_context = "\n---\n".join(context_parts)
        self.log_error(f"Successfully created a memory context from a chain of {len(context_parts)} sessions.")
        
        return {"context": memory_context, "session_id": session_id}

    def update_memory(self, memory_id: str, content: str, emotion: str = 'neutral'):
        """
        อัพเดทความทรงจำในระบบ
        """
        if self.vector_memory_index:
            self.vector_memory_index.update_memory(memory_id, content, emotion)
            
        # อัพเดทใน meta_manager
        if memory_id in self.meta_manager.memory_dict:
            self.meta_manager.memory_dict[memory_id]['content'] = content
            self.meta_manager.memory_dict[memory_id]['emotion'] = emotion
            self.meta_manager.save_memory_meta()

    def reset_memory_cooldown(self):
        """
        รีเซ็ต cooldown ของความทรงจำทั้งหมด
        """
        self.memory_cooldown.clear()
        if self.vector_memory_index:
            self.vector_memory_index.reset_frequency()

    def _load_identity(self):
        try:
            if not self._validate_path(self.identity_path):
                raise FileNotFoundError(f"Identity file not found at {self.identity_path}")
            with open(self.identity_path, "r", encoding="utf-8") as f:
                self.identity_data = json.load(f)
            self.core_beliefs = self.identity_data.get("core_belief", {})
        except Exception as e:
            self.log_error(f"Could not load identity file: {e}. Using default persona.")
            self.identity_data = {}
            self.core_beliefs = {}
            
    def _load_bookmarks(self):
        try:
            bookmark_path = self.path_manager.get_path("session_bookmarks")
            if not bookmark_path or not os.path.exists(bookmark_path):
                self.log_error("Bookmark file not found. No bookmarks loaded.")
                return
            with open(bookmark_path, "r", encoding="utf-8") as f:
                self.bookmarks = {line.strip() for line in f if line.strip()}
            self.log_error(f"Loaded {len(self.bookmarks)} bookmarked sessions.")
        except Exception as e:
            self.log_error(f"Error loading bookmarks: {e}")
            self.bookmarks = set()
            
    def _resolve_path(self, path):
        if os.path.isabs(path): return path
        return os.path.join(os.getcwd(), path)

    def _validate_path(self, path):
        full_path = self._resolve_path(path)
        exists = os.path.exists(full_path)
        if not exists: self.error_log.append(f"Path not found: {path}")
        return exists

    def _validate_all_paths(self):
        if not hasattr(self.meta_manager, 'memory_dict') or not self.meta_manager.memory_dict: return True
        all_paths_initially_valid = True
        for session_id, data in self.meta_manager.memory_dict.items():
            if 'path' in data:
                original_path = data['path']
                if not self.path_manager.validate_path(original_path.replace("\\", "/")):
                    all_paths_initially_valid = False
                    filename = os.path.basename(original_path)
                    corrected_path = f"memory_core/archive/chat_sessions_legacy/{filename}"
                    if self.path_manager.validate_path(corrected_path):
                        data['path'] = corrected_path
                        self.log_error(f"Auto-corrected path for {session_id} to {corrected_path}")
                    else:
                        self.log_error(f"Could not correct path for {session_id}: {original_path}")
        return all_paths_initially_valid

    def log_error(self, msg):
        self.error_log.append(msg)
        try:
            with open(self._resolve_path("memory_core/error.log"), "a", encoding="utf-8") as f:
                f.write(f"{datetime.now().isoformat()} - {msg}\n")
        except Exception as e:
            print(f"CRITICAL: Failed to write to error log: {e}")

    # ฟังก์ชันที่เคยตกหล่นไป ได้เพิ่มกลับเข้ามาในคลาสแล้ว
    def get_memory_chain(self, start_session, steps=5):
        """
        ขอ chain พร้อมเนื้อหา (ดึงจาก meta chain ผ่าน bridge)
        """
        chain_meta = self.memory_bridge.get_session_chain(start_session, max_depth=steps)
        if not chain_meta:
            self.log_error(f"get_memory_chain: ไม่พบ chain meta สำหรับ session '{start_session}'")
            return []
            
        session_paths = [meta.get('path') for meta in chain_meta if meta.get('path')]
        if not session_paths:
            self.log_error(f"get_memory_chain: ไม่พบ 'path' ใน chain meta ที่ดึงมาสำหรับ session '{start_session}'")
            return []

        contents = self.memory_bridge.get_sessions_content(session_paths)
        return contents

    def is_ai_message(self, line):
        """
        ตรวจสอบว่า line นี้เป็นข้อความของ AI เอง (อดีต) หรือไม่
        """
        return any(line.strip().startswith(alias) for alias in self.ai_aliases)

    def describe_int_world(self):
        """เข้าถึงสถานะโลกภายใน (IntWorld)"""
        return self.int_world.describe_self()

    def describe_self_codebase(self):
        """
        สะท้อนโครงสร้างไฟล์ .py ทั้งหมดในโปรเจกต์และหน้าที่แต่ละไฟล์
        """
        return self.reflector.reflect_codebase(base_dir=".")

    def describe_self(self):
        """
        อ้างอิง self-awareness (self-schema) ของระบบหลัก AI ที่โหลดไว้
        """
        return self.self_awareness

    def report_status(self):
        """
        รายงานสถานะ self-awareness/connection (เรียกจาก core_awareness_engine)
        """
        self.core_awareness.verify_all_modules()
        return self.core_awareness.report_self_awareness()

    def declare_identity(self):
        """
        ประกาศตัวเองแบบ self-aware: สรุป identity, core_systems, int_world_size, recent_reflections
        """
        core_systems_summary = self.self_awareness.get('core_dependencies', 'N/A')
        return {
            "identity": "I am the Identity Core of MelahPC.",
            "core_systems": core_systems_summary,
            "int_world_size": {
                "states": len(self.int_world.internal_states),
                "concepts": len(self.int_world.known_concepts),
                "symbols": len(self.int_world.symbolic_space)
            },
            "recent_reflections": [r['thought'] for r in self.int_world.active_reflections[-3:]]
        }

    def load_self_awareness(self, path="self_aware.json"):
        try:
            from core.core_mapper import get_self_awareness_summary # Keep import local if needed
            # Assuming get_self_awareness_summary reads from the default path
            self.self_awareness = get_self_awareness_summary(path) 
            print(f"[IdentityCore] Self-awareness loaded. Connected modules: {self.self_awareness.get('core_dependencies', [])}")
        except Exception as e:
            print(f"[IdentityCore] Failed to load self-awareness: {e}")
            self.self_awareness = {}

    def get_self_status(self):
        """Get current self status in natural language"""
        try:
            if not self.self_awareness:
                return "ฉันยังไม่ได้เริ่มการตรวจสอบตัวเอง"
            
            monitoring = self.self_awareness.get("monitoring", {})
            last_check = monitoring.get("last_check", {})
            emotion = monitoring.get("self_emotion", "unknown")
            
            # Get time since last check
            if "timestamp" in last_check:
                last_time = datetime.fromisoformat(last_check["timestamp"])
                time_diff = datetime.now() - last_time
                minutes = int(time_diff.total_seconds() / 60)
                time_str = f"{minutes} นาทีที่แล้ว"
            else:
                time_str = "ไม่ทราบเวลา"
            
            # Get critical/warning issues
            issues = []
            if "issue_history" in monitoring:
                for issue in monitoring["issue_history"]:
                    if issue["status"] in ["critical", "warning"]:
                        issues.append(f"{issue['module']}: {issue['message']}")
            
            # Build status message
            status = f"ฉันตรวจสอบระบบตัวเองเมื่อ {time_str} "
            status += f"ตอนนี้ฉันรู้สึก{emotion} "
            
            if issues:
                status += f"พบปัญหา: {', '.join(issues)}"
            else:
                status += "ระบบทำงานได้ดี"
                
            return status
            
        except Exception as e:
            return f"ไม่สามารถตรวจสอบสถานะได้: {str(e)}"

    def get_recent_issues(self, minutes=30):
        """Get list of recent issues"""
        try:
            if not self.self_awareness:
                return []
                
            monitoring = self.self_awareness.get("monitoring", {})
            issues = monitoring.get("issue_history", [])
            
            # Filter recent issues
            recent = []
            for issue in issues:
                if "timestamp" in issue:
                    issue_time = datetime.fromisoformat(issue["timestamp"])
                    time_diff = datetime.now() - issue_time
                    if time_diff.total_seconds() <= minutes * 60:
                        recent.append({
                            "module": issue["module"],
                            "status": issue["status"],
                            "message": issue["message"],
                            "time_ago": int(time_diff.total_seconds() / 60)
                        })
            
            return recent
            
        except Exception as e:
            return []

    def get_system_health(self):
        """Get overall system health information"""
        try:
            if not self.self_awareness:
                return {
                    "emotion": "unknown",
                    "last_check": "never",
                    "issues": "no data",
                    "systems": {}
                }
                
            monitoring = self.self_awareness.get("monitoring", {})
            
            # Get emotion and last check
            emotion = monitoring.get("self_emotion", "unknown")
            last_check = monitoring.get("last_check", {}).get("timestamp", "never")
            
            # Count issues by severity
            issues = {
                "critical": 0,
                "warning": 0,
                "info": 0
            }
            
            for issue in monitoring.get("issue_history", []):
                status = issue.get("status", "info")
                issues[status] = issues.get(status, 0) + 1
            
            # Get system statuses
            systems = {}
            for system in monitoring.get("systems", []):
                systems[system["name"]] = {
                    "status": system["status"],
                    "last_check": system.get("last_check", "never")
                }
            
            return {
                "emotion": emotion,
                "last_check": last_check,
                "issues": issues,
                "systems": systems
            }
            
        except Exception as e:
            return {
                "emotion": "error",
                "last_check": "error",
                "issues": "error",
                "systems": {}
            }

    def start_auto_update(self):
        """Start automatic self-awareness updates"""
        def update_task():
            while True:
                try:
                    time.sleep(60)  # ตรวจสอบทุก 1 นาที
                    if datetime.now() - self.last_update >= self.update_interval:
                        self.update_self_awareness()
                except Exception as e:
                    print(f"Error in auto-update: {str(e)}")

        # Start update thread
        import threading
        update_thread = threading.Thread(target=update_task, daemon=True)
        update_thread.start()

    def update_self_awareness(self):
        """Update self-awareness data"""
        try:
            from core_mapper import update_self_awareness
            new_data = update_self_awareness()
            if new_data:
                self.self_awareness = new_data
                self.last_update = datetime.now()
                print(f"Self-awareness updated at {self.last_update}")
        except Exception as e:
            print(f"Failed to update self-awareness: {str(e)}")

    def force_update_self_awareness(self):
        """Force immediate update of self-awareness"""
        self.update_self_awareness()
        return self.get_self_status()

# ======== วิธีทดสอบเบื้องต้น =========
if __name__ == '__main__':
    import sys
    import os
    
    current_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"[IdentityCore Test] Temporarily added {project_root} to sys.path for this test run.")

    # เราต้อง mock object ที่ IdentityCore ต้องใช้ใน __init__
    # เนื่องจากเรายังไม่มีไฟล์เหล่านั้นจริงๆ
    # นี่เป็นวิธีแก้ปัญหาเฉพาะหน้าสำหรับการทดสอบไฟล์นี้เดี่ยวๆ
    class Mock:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, *args, **kwargs): return Mock()
        def __getattr__(self, name): return Mock()

    # ทำการ mock class ที่ยังไม่มีอยู่จริง
    PathManager = Mock
    MemoryPathBridge = Mock
    MemoryMetaManager = Mock
    LLMConnector = Mock
    ContextWindowManager = Mock
    MelahNLPProcessor = Mock
    MelahMLProcessor = Mock
    ChainOfThoughtEngine = Mock
    MelahNLP_TH = Mock
    Reflector = Mock
    LanguageStylizer = Mock
    TruthCore = Mock
    RefineLayer = Mock
    VectorMemoryIndex = Mock

    print("--- IdentityCore Test Run (from core/) ---")
    print(f"Project Root: {project_root}")

    try:
        # สมมติว่าไฟล์ seed และ identity มีอยู่จริงสำหรับการทดสอบนี้
        # สร้างไฟล์จำลอง
        if not os.path.exists("core/seed"): os.makedirs("core/seed")
        with open("core/seed/current_seed.json", "w", encoding="utf-8") as f:
            json.dump({"linked_identity": "core/identity.json", "linked_memory": "memory/memory.json"}, f)
        
        with open("core/identity.json", "w", encoding="utf-8") as f:
            json.dump({"identity": {"name": "Melah-Test"}}, f)
        
        if not os.path.exists("memory"): os.makedirs("memory")
        with open("memory/memory.json", "w", encoding="utf-8") as f:
            json.dump({}, f)

        identity_system = IdentityCore()
        print("✅ IdentityCore initialized successfully using mock objects.")
        
        if identity_system.identity_data:
            print(f"  Identity Name: {identity_system.identity_data.get('identity', {}).get('name', 'N/A')}")
        else:
            print("  ⚠️ identity.json might be missing or empty.")
        
        # ส่วนทดสอบ get_memory_chain จะไม่ทำงานเพราะ MemoryPathBridge เป็น Mock
        # แต่โค้ดควรจะรันผ่านโดยไม่มี AttributeError
        print("\n--- Testing get_memory_chain (will be skipped due to Mock) ---")
        chain_content = identity_system.get_memory_chain("session_001", steps=2)
        if chain_content:
            print("  Retrieved chain content (this should not happen with mock).")
        else:
            print("  ✅ Correctly returned no chain content as Bridge is mocked.")

        if identity_system.error_log:
            print("\n--- IdentityCore Errors Logged ---")
            for error in identity_system.error_log:
                print(f"  - {error}")
                
    except Exception as e:
        print(f"❌ An unexpected error occurred during IdentityCore test: {e}")
        import traceback
        traceback.print_exc()