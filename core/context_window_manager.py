# core/context_window_manager.py
from collections import deque
import json
from datetime import datetime
from pathlib import Path
from .raw_chat_logger import log_raw_chat  # เพิ่ม import log_raw_chat

# --- Import ที่จำเป็น (คุณจะต้อง uncomment และตรวจสอบ path เมื่อนำไปรวมกับ project จริง) ---
# from llm_connector import LLMConnector # สมมติว่า LLMConnector อยู่ใน core/
# from path_manager import PathManager   # สมมติว่า PathManager อยู่ใน core/
# import tiktoken # หรือ tokenizer อื่นๆ ที่ LLM คุณใช้

class ContextWindowManager:
    def __init__(self,
                 llm_connector_instance,    # LLMConnector instance ที่ config แล้ว
                 path_manager_instance,       # PathManager instance ที่ config แล้ว
                 tokenizer_instance,          # Tokenizer ที่ LLM ใช้งานจริง (เช่น tiktoken.Encoding)
                 llm_context_window_actual_limit: int,
                 raw_chat_log_base_dir: Path # เช่น .../memory_core/memory_archive/raw_chat_logs/
                ):

        self.llm_connector = llm_connector_instance
        self.path_manager = path_manager_instance
        self.tokenizer = tokenizer_instance
        self.llm_token_limit = llm_context_window_actual_limit

        # conversation_history_buffer เก็บ message dicts: {"role": ..., "content": ...}
        self.conversation_history_buffer = deque()

        self.current_session_id: str | None = None
        self.current_session_date_key: str | None = None # YYYYMMDD
        self.raw_chat_log_base_dir = raw_chat_log_base_dir.resolve() # ทำให้เป็น absolute
        self.current_raw_chat_log_file: Path | None = None # Path ไปยังไฟล์ chat_YYYYMMDD.json ของ session ปัจจุบัน

        # ค่าประมาณสำหรับการจัดการ buffer ภายใน (ยังไม่ถูกใช้ในการย่อ buffer อัตโนมัติในตอนนี้)
        self.simulated_token_byte_ratio = 10 # 1 token จำลอง = 10 bytes
        # self.max_buffer_simulated_tokens_soft_limit = (90 * 1024 * 1024) // self.simulated_token_byte_ratio # 90MB

        print(f"ℹ️ ContextWindowManager initialized. LLM Token Limit: {self.llm_token_limit} tokens.")
        print(f"ℹ️ Raw chat logs base directory: {self.raw_chat_log_base_dir}")

    def start_new_session(self, session_id: str, date_key: str, system_prompt: str = None):
        """
        เริ่มต้น session ใหม่, ล้าง buffer เก่า, ตั้งค่า session ID/date key,
        และกำหนดไฟล์ log ประจำวันสำหรับ session นี้ รวมถึงลงทะเบียน log file กับ PathManager ถ้ายังไม่มี
        """
        self.conversation_history_buffer.clear()
        self.current_session_id = session_id
        self.current_session_date_key = date_key # YYYYMMDD

        year = date_key[:4]
        month = date_key[4:6]
        day = date_key[6:]
        session_log_dir = self.raw_chat_log_base_dir / year / month / day
        session_log_dir.mkdir(parents=True, exist_ok=True)
        self.current_raw_chat_log_file = session_log_dir / f"chat_{date_key}.json"

        print(f"🚀 CWM: New session '{self.current_session_id}' started for date '{self.current_session_date_key}'.")
        print(f"📝 CWM: Logging raw chat for this session to: {self.current_raw_chat_log_file}")

        # ลงทะเบียนไฟล์ log กับ PathManager ถ้ายังไม่เคยลงทะเบียนสำหรับวันนั้น
        # PathManager.add_daily_raw_chat_log จะจัดการเรื่องการสร้าง entry ใหม่หรือไม่อย่างไร
        self.path_manager.add_daily_raw_chat_log(
            date_key=self.current_session_date_key,
            file_path_absolute=self.current_raw_chat_log_file,
            session_id=self.current_session_id
            # number_of_messages อาจจะ update ทีหลัง
        )

        if system_prompt:
            # เพิ่ม system prompt เข้า buffer และ log โดยไม่ให้ _log_interaction_to_file เรียกซ้ำ
            self.add_interaction(role="system", message=system_prompt, _is_internal_call=True)


    def _log_interaction_to_file(self, interaction_entry_for_log: dict):
        """
        (Private) บันทึก interaction ลงในไฟล์ self.current_raw_chat_log_file (chat_YYYYMMDD.json)
        ไฟล์ log จะเป็น list ของ interaction entries
        """
        if not self.current_raw_chat_log_file:
            print("⚠️ CWM: Cannot log interaction - Current raw chat log file not set (session not started?).")
            return

        log_data = []
        if self.current_raw_chat_log_file.exists():
            try:
                with open(self.current_raw_chat_log_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    if content:
                        log_data = json.loads(content)
                        if not isinstance(log_data, list):
                            print(f"⚠️ CWM: Log file {self.current_raw_chat_log_file.name} content is not a list. Initializing new log.")
                            log_data = []
            except json.JSONDecodeError:
                print(f"⚠️ CWM: Error decoding JSON from {self.current_raw_chat_log_file.name}. Initializing new log.")
                log_data = []
            except Exception as e:
                print(f"❌ CWM: Error reading log file {self.current_raw_chat_log_file.name}: {e}. Initializing new log.")
                log_data = []

        log_data.append(interaction_entry_for_log)

        try:
            with open(self.current_raw_chat_log_file, "w", encoding="utf-8") as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"❌ CWM: Error writing to log file {self.current_raw_chat_log_file.name}: {e}")

    def add_interaction(self, role: str, message: str, timestamp: str = None, _is_internal_call: bool = False):
        """
        เพิ่มข้อความใหม่เข้า conversation_history_buffer และบันทึกลง raw chat log.
        _is_internal_call: parameter ภายในเพื่อป้องกันการ log ซ้ำซ้อนจาก system prompt ตอนเริ่ม session
        """
        if not self.current_session_id:
            print("⚠️ CWM: Cannot add interaction - Session not started. Call start_new_session() first.")
            return

        ts = timestamp if timestamp else datetime.now().isoformat()
        interaction_entry_for_buffer = {"role": role, "content": message}
        self.conversation_history_buffer.append(interaction_entry_for_buffer)

        if not _is_internal_call: # ถ้าไม่ได้ถูกเรียกจากภายใน (เช่น system prompt ตอน start_new_session) ให้ log
            interaction_entry_for_log = {
                "session_id": self.current_session_id,
                "timestamp": ts,
                "role": role,
                "content": message
            }
            self._log_interaction_to_file(interaction_entry_for_log)
            # --- เพิ่มการบันทึกแชทดิบแบบ real-time ---
            meta = {
                "session_id": self.current_session_id,
                "role": role
            }
            log_raw_chat(message, metadata=meta, pinned=False, as_json=True)

    def _count_tokens(self, messages: list[dict]) -> int:
        """ (Private) นับจำนวน token จริงของ list of messages โดยใช้ self.tokenizer """
        if not self.tokenizer:
            print("⚠️ CWM: Tokenizer not available, cannot count actual tokens. Returning 0.")
            return 0 # หรือจะ raise error

        # การนับ token จริงอาจจะซับซ้อนกว่านี้ขึ้นอยู่กับ format ที่ LLM API ต้องการ
        # ตัวอย่างนี้จะนับ token จาก content ของแต่ละ message และเผื่อ token สำหรับ role
        # สำหรับ OpenAI (tiktoken) การนับ token สำหรับ chat completion มี format เฉพาะ
        # ดูเพิ่มเติม: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        
        num_tokens = 0
        # ตัวอย่างง่ายๆ สำหรับการนับ (อาจจะต้องปรับให้ตรงกับ LLM ที่ใช้จริง)
        for message in messages:
            num_tokens += len(self.tokenizer.encode(message.get("content", "")))
            num_tokens += len(self.tokenizer.encode(message.get("role", ""))) # ประมาณการคร่าวๆ
            # OpenAI อาจจะมี token พิเศษสำหรับแบ่ง message ด้วย
        return num_tokens


    def get_llm_ready_context(self, current_user_input_content: str = None) -> list[dict]:
        """
        เตรียม list ของ message dicts ที่จะส่งให้ LLM,
        โดยพยายามรักษา original content และจัดการเมื่อยาวเกิน token limit
        ด้วยการ "ตัดทอน" (truncate) ส่วนที่เก่าที่สุดของ context ที่จะส่งออกไปก่อน
        """
        if not self.tokenizer:
             raise ValueError("CWM: Tokenizer instance is required for get_llm_ready_context.")

        # 1. สร้าง Context ที่จะพิจารณาส่ง (Working Context)
        working_context_messages = list(self.conversation_history_buffer)
        if current_user_input_content: # input ใหม่ของผู้ใช้
            working_context_messages.append({"role": "user", "content": current_user_input_content})

        # 2. นับ Token จริง และจัดการถ้าเกิน Limit
        current_total_tokens = self._count_tokens(working_context_messages)

        # 3. ถ้าเกิน Limit, ทำการตัดทอนจากส่วนที่เก่าที่สุด (หัว deque/list)
        #    จนกว่าจะอยู่ใน limit หรือเหลือแค่ message ล่าสุด (ถ้า message เดียวก็ยังยาวเกิน)
        while current_total_tokens > self.llm_token_limit and len(working_context_messages) > 1:
            # ลบ message ที่เก่าที่สุด (อันแรกใน list) ออก
            removed_message = working_context_messages.pop(0) # ถ้าเป็น deque ใช้ popleft()
            print(f"ℹ️ CWM: Truncating oldest message to fit token limit: [{removed_message['role']}] {removed_message['content'][:30]}...")
            current_total_tokens = self._count_tokens(working_context_messages) # นับ token ใหม่

        # 4. กรณีที่แม้แต่ message เดียว (current_user_input) ก็ยังยาวเกินไป
        if not working_context_messages and current_user_input_content: # ถ้า buffer ว่างเปล่าแต่มี input ใหม่
             print(f"⚠️ CWM: Current user input might be too long for the LLM context window!")
             # อาจจะต้องตัด current_user_input หรือแจ้ง error
             # ตอนนี้จะยังคงส่งไป (LLM API บางตัวอาจจะตัดให้เอง หรือคืน error)
             # หรือเราจะ implement logic การตัด message สุดท้ายที่นี่ก็ได้
             # เพื่อความปลอดภัย อาจจะคืน list ที่มี message เดียวที่ถูกตัดแล้ว
             single_message_tokens = self.tokenizer.encode(current_user_input_content)
             if len(single_message_tokens) > self.llm_token_limit:
                 truncated_tokens = single_message_tokens[:self.llm_token_limit - 50] # -50 เผื่อไว้
                 truncated_content = self.tokenizer.decode(truncated_tokens)
                 print(f"⚠️ CWM: Truncating excessively long single user input.")
                 return [{"role": "user", "content": truncated_content}]
             else:
                 return [{"role": "user", "content": current_user_input_content}]

        elif current_total_tokens > self.llm_token_limit and len(working_context_messages) == 1:
            # กรณีเหลือ message เดียวใน working_context แต่ก็ยังเกิน (น่าจะเป็น current_user_input ที่เพิ่ง add)
            print(f"⚠️ CWM: The final single message is still too long for the LLM context window!")
            # ทำการตัด message สุดท้ายนี้
            last_message = working_context_messages[0]
            tokens = self.tokenizer.encode(last_message["content"])
            if len(tokens) > self.llm_token_limit:
                truncated_tokens = tokens[:self.llm_token_limit - 50] # -50 เผื่อ role และอื่นๆ
                truncated_content = self.tokenizer.decode(truncated_tokens)
                print(f"⚠️ CWM: Truncating excessively long single message in buffer.")
                working_context_messages = [{"role": last_message["role"], "content": truncated_content}]

        print(f"ℹ️ CWM: Final context for LLM has approx. {current_total_tokens} tokens (Limit: {self.llm_token_limit}).")
        return working_context_messages

    def retrieve_archived_context(self, query_or_reference: str, target_date_key: str = None) -> list[dict] | None:
        """
        (ตัวอย่าง) ดึงข้อมูลเก่าจาก `chat_YYYYMMDD.json` ผ่าน PathManager
        และอาจจะสรุปก่อนส่งกลับเป็น list of message dicts (สำหรับใส่ใน context)
        """
        print(f"🔍 CWM: Attempting to retrieve archived context for query: '{query_or_reference}' for date: {target_date_key or 'any'}")
        
        if not target_date_key:
            target_date_key = self.current_session_date_key # ลองใช้วันของ session ปัจจุบันก่อน
            if not target_date_key:
                print("⚠️ CWM: Cannot retrieve archived context without a target_date_key or active session date.")
                return None

        # PathManager ควรมี method get_daily_raw_chat_log_entry หรือคล้ายกัน
        raw_chat_log_entry = self.path_manager.get_daily_raw_chat_log_entry(target_date_key)

        if raw_chat_log_entry and raw_chat_log_entry.get("file_path_relative"):
            archived_chat_content_str = self.path_manager.read_archived_file_content(raw_chat_log_entry["file_path_relative"])
            if archived_chat_content_str:
                try:
                    archived_log_data = json.loads(archived_chat_content_str) # คาดหวัง list ของ interaction dicts
                    if not isinstance(archived_log_data, list):
                        print(f"⚠️ CWM: Archived chat for {target_date_key} is not a list of interactions.")
                        return None

                    # --- ส่วนนี้คือการค้นหาและสรุปข้อมูลที่ดึงมา (ตัวอย่างง่ายๆ) ---
                    relevant_messages_from_archive = []
                    if query_or_reference:
                        for msg_entry in reversed(archived_log_data): # ค้นจากท้ายมา
                            if isinstance(msg_entry, dict) and query_or_reference.lower() in msg_entry.get("content","").lower():
                                # ดึงเฉพาะ role และ content เพื่อให้ format เหมือน buffer
                                relevant_messages_from_archive.append({"role": msg_entry.get("role"), "content": msg_entry.get("content")})
                                if len(relevant_messages_from_archive) >= 3: # เอามา 3 messages ที่เกี่ยวข้อง
                                    break
                        relevant_messages_from_archive.reverse()

                    if relevant_messages_from_archive:
                        print(f"ℹ️ CWM: Retrieved {len(relevant_messages_from_archive)} relevant messages from archive for '{query_or_reference}'.")
                        # สร้างเป็น system message เพื่อแจ้ง LLM ว่านี่คือข้อมูลจากอดีต
                        # อาจจะต้องมีการสรุปข้อความเหล่านี้อีกทีถ้ามันยาวเกินไป
                        context_from_archive_str = f"System: Regarding your query about '{query_or_reference}', here is some relevant past context from {target_date_key}:\n"
                        for msg in relevant_messages_from_archive:
                            context_from_archive_str += f"{msg['role']}: {msg['content']}\n"
                        
                        # ต้องระวังไม่ให้ system message นี้ยาวเกินไป
                        # เราจะคืนเป็น list ที่มี message เดียวที่เป็น system message
                        # ส่วนที่เรียก get_llm_ready_context จะต้องนำไปรวมกับ context ปัจจุบันและจัดการ token limit อีกที
                        return [{"role": "system", "content": context_from_archive_str.strip()}]
                    else:
                        print(f"ℹ️ CWM: No specific messages matching '{query_or_reference}' found in archive for {target_date_key}.")
                        return [{"role": "system", "content": f"System: No specific details matching '{query_or_reference}' found in records for {target_date_key}."}]


                except json.JSONDecodeError:
                    print(f"⚠️ CWM: Error decoding JSON from archived chat for {target_date_key}.")
                except Exception as e:
                    print(f"❌ CWM: Error processing archived chat for {target_date_key}: {e}")
        else:
            print(f"ℹ️ CWM: No archived chat log entry found in PathManager for date {target_date_key}.")
        return None


    def _compact_buffer_by_summarizing_oldest(self): # ยังไม่ implement ในขั้นนี้
        """
        (Optional - Future Enhancement)
        ลดขนาด conversation_history_buffer เมื่อมันใหญ่เกินไป (ตาม simulated tokens)
        โดยการสรุปส่วนที่เก่าที่สุด และแทนที่ด้วยบทสรุปนั้น
        """
        print("⚠️ CWM: _compact_buffer_by_summarizing_oldest() is a future enhancement and not yet implemented.")
        # Logic:
        # 1. ตรวจสอบขนาด buffer (ตาม "token จำลอง" หรือจำนวน messages)
        # 2. ถ้าใหญ่ไป เลือก N turns แรก (เก่าสุด) จาก self.conversation_history_buffer
        # 3. สร้าง prompt สรุป N turns นั้น
        # 4. เรียก self.llm_connector.generate(prompt_for_summary)
        # 5. สร้าง system message ใหม่ที่เป็นบทสรุป {"role": "system", "content": "Summary of earlier conversation: ..."}
        # 6. ลบ N turns เดิมออกจากหัว deque ของ self.conversation_history_buffer
        # 7. เพิ่ม system message (บทสรุป) เข้าไปที่หัว deque แทน
        pass

    def on_new_raw_chat_log(file_path, rel_path, metadata, text, timestamp):
        """
        Callback สำหรับ raw_chat_logger: อัปเดต context buffer จาก raw chat ใหม่
        """
        try:
            # สมมุติว่ามี ContextWindowManager instance ที่ config ไว้แล้ว (หรือสร้างใหม่ถ้าจำเป็น)
            # ต้องปรับตามการใช้งานจริงใน orchestrator/main
            print(f"[ContextWindowManager] (stub) Would update context buffer from: {rel_path}")
            # ตัวอย่าง: context_manager.update_context_buffer_from_log(file_path, rel_path, metadata, text, timestamp)
        except Exception as e:
            print(f"[ContextWindowManager] Failed to update context buffer: {e}")