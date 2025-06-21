# core/vector_memory_index.py

import logging
import os
from typing import List, Dict, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer, util
import faiss
from collections import defaultdict
import pickle # เพิ่มเข้ามาเพื่อบันทึกและโหลด object
from datetime import datetime
import re
import json # Added for export/import

logger = logging.getLogger(__name__)

class VectorMemoryIndex:
    """ระบบจัดการความทรงจำเชิงความหมาย
    
    Role: Semantic Memory Management System
    
    Responsibilities:
    - สร้างและจัดการดัชนีความทรงจำเชิงความหมาย
    - ค้นหาความทรงจำที่เกี่ยวข้อง
    - จัดการความสัมพันธ์ระหว่างความทรงจำ
    - รักษาคุณภาพและความทันสมัยของดัชนี
    """
    def __init__(self, memory_dict: Dict, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2', index_path: Optional[str] = None, auto_save: bool = True):
        """
        Args:
            memory_dict (Dict): Dictionary ของความทรงจำจาก MemoryMetaManager
            model_name (str): ชื่อของ pre-trained model จาก sentence-transformers
            index_path (Optional[str]): Path ไปยัง folder ที่จะบันทึก/โหลด index
            auto_save (bool): บันทึกอัตโนมัติหลังการเพิ่ม/อัปเดต memory
        """
        self.memory_dict = memory_dict
        self.model = None
        self.index = None
        self.embeddings = None
        self.memory_ids_map = []
        self.memory_frequency = defaultdict(int)
        self.emotion_weights = {
            'happy': 1.2, 'sad': 1.2, 'angry': 1.2,
            'fear': 1.2, 'surprise': 1.2, 'neutral': 1.0
        }
        self.active_memory_ids = set()
        self.max_active_memories = 1000
        self.auto_save = auto_save
        self.conversation_buffer = []
        self.buffer_limit = 4
        self.memory_counter = 0
        
        # กำหนด index_path เป็น "vector_index" ถ้าไม่มีการระบุ
        self.index_path = index_path if index_path else "vector_index"

        try:
            logger.info(f"Loading sentence-transformer model: {model_name}...")
            self.model = SentenceTransformer(model_name)
            logger.info("Model loaded successfully.")

            # --- ส่วนที่แก้ไขใหม่ ---
            # พยายามโหลด index ที่มีอยู่ก่อน
            if self.load_index(self.index_path):
                logger.info(f"Successfully loaded index from {self.index_path}")
                # อัปเดต index ด้วยข้อมูลปัจจุบันและบันทึก
                self.rebuild_index()
            else:
                logger.info("No existing index found. Building a new one.")
                self._build_index()
                self.save_index(self.index_path)

        except Exception as e:
            logger.error(f"Failed to initialize VectorMemoryIndex: {e}", exc_info=True)
            self.model = None

    def _build_index(self):
        """
        สร้าง FAISS index จากความทรงจำที่มี
        """
        if not self.model or not self.memory_dict:
            return

        try:
            texts = []
            self.memory_ids_map = list(self.memory_dict.keys()) # เก็บ key 순서

            for memory_id in self.memory_ids_map:
                memory = self.memory_dict[memory_id]
                if 'content' in memory and memory['content']:
                    text = memory['content']
                else:
                    event = memory.get('event', '')
                    summary = memory.get('summary', '')
                    emotion = ', '.join(memory.get('emotion', []))
                    tags = ', '.join(memory.get('tags', []))
                    insight = memory.get('insight', '')
                    text = f"event: {event}; summary: {summary}; emotion: {emotion}; tags: {tags}; insight: {insight}"
                texts.append(text)
            
            if not texts:
                logger.warning("No texts to index.")
                return

            self.embeddings = self.model.encode(texts)
            
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIDMap(self.index) # ใช้ IndexIDMap เพื่อให้ลบและอัพเดทง่ายขึ้น
            
            # สร้าง ID สำหรับ FAISS index
            ids = np.arange(len(self.memory_ids_map))
            self.index.add_with_ids(self.embeddings.astype('float32'), ids)
            
            logger.info(f"Successfully built FAISS index with {len(texts)} memories")
            
        except Exception as e:
            logger.error(f"Error building FAISS index: {e}", exc_info=True)
            self.index = None
    
    # +++ เมธอดใหม่สำหรับบันทึก Index +++
    def save_index(self, folder_path: str):
        """
        บันทึก FAISS index, embeddings และ memory_ids_map ลงใน folder ที่กำหนด
        """
        if not self.index or self.embeddings is None:
            logger.warning("Index or embeddings not available to save.")
            return
        
        try:
            os.makedirs(folder_path, exist_ok=True)
            # บันทึก FAISS index
            faiss.write_index(self.index, os.path.join(folder_path, "vector.index"))
            
            # บันทึก memory_ids_map
            with open(os.path.join(folder_path, "memory_ids_map.pkl"), "wb") as f:
                pickle.dump(self.memory_ids_map, f)
            
            logger.info(f"Successfully saved index to {folder_path}")
        except Exception as e:
            logger.error(f"Error saving index to {folder_path}: {e}", exc_info=True)

    # +++ เมธอดใหม่สำหรับโหลด Index +++
    def load_index(self, folder_path: str) -> bool:
        """
        โหลด FAISS index และข้อมูลที่เกี่ยวข้องจาก folder ที่กำหนด
        """
        index_file = os.path.join(folder_path, "vector.index")
        map_file = os.path.join(folder_path, "memory_ids_map.pkl")

        if not os.path.exists(index_file) or not os.path.exists(map_file):
            logger.warning("Index files not found in the specified path.")
            return False
            
        try:
            # โหลด FAISS index
            self.index = faiss.read_index(index_file)

            # โหลด memory_ids_map
            with open(map_file, "rb") as f:
                self.memory_ids_map = pickle.load(f)

            # โหลด embeddings (สร้างใหม่เพื่อให้สอดคล้องกับ memory_dict)
            self._rebuild_embeddings()

            logger.info(f"Successfully loaded index and map from {folder_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading index from {folder_path}: {e}", exc_info=True)
            return False

    def _rebuild_embeddings(self):
        """สร้าง embeddings ขึ้นมาใหม่ตามลำดับของ memory_ids_map ที่โหลดมา"""
        texts = []
        for memory_id in self.memory_ids_map:
            memory = self.memory_dict.get(memory_id, {})
            # โค้ดสร้าง text เหมือนใน _build_index
            if 'content' in memory and memory['content']:
                text = memory['content']
            else:
                event = memory.get('event', '')
                summary = memory.get('summary', '')
                emotion = ', '.join(memory.get('emotion', []))
                tags = ', '.join(memory.get('tags', []))
                insight = memory.get('insight', '')
                text = f"event: {event}; summary: {summary}; emotion: {emotion}; tags: {tags}; insight: {insight}"
            texts.append(text)
        self.embeddings = self.model.encode(texts)


    def search(self, query: str, top_k: int = 1, exclude_ids: List[str] = None, emotion: str = 'neutral') -> List[Dict]:
        """
        ค้นหาความทรงจำที่เกี่ยวข้องกับ query
        """
        if not self.model or not self.index:
            logger.warning("Model or index not available")
            return []

        try:
            query_embedding = self.model.encode([query])[0]
            
            scores, indices = self.index.search(
                query_embedding.reshape(1, -1).astype('float32'), 
                top_k * 2
            )
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                    
                memory_id = self.memory_ids_map[idx] # แก้เป็นใช้ map ที่เราเก็บไว้
                
                if exclude_ids and memory_id in exclude_ids:
                    continue
                
                frequency_penalty = 0.8 ** self.memory_frequency[memory_id]
                
                memory_emotions = self.memory_dict[memory_id].get('emotion', ['neutral'])
                if isinstance(memory_emotions, list):
                    top_emotions = memory_emotions[:3]
                else:
                    top_emotions = [memory_emotions]
                weights = [self.emotion_weights.get(e, 1.0) for e in top_emotions]
                emotion_weight = sum(weights) / len(weights) if weights else 1.0
                
                final_score = float(score) * frequency_penalty * emotion_weight
                
                memory = self.memory_dict[memory_id]
                if 'content' in memory and memory['content']:
                    content = memory['content']
                else:
                    event = memory.get('event', '')
                    summary = memory.get('summary', '')
                    emotion = ', '.join(memory.get('emotion', []))
                    tags = ', '.join(memory.get('tags', []))
                    insight = memory.get('insight', '')
                    content = f"event: {event}; summary: {summary}; emotion: {emotion}; tags: {tags}; insight: {insight}"
                
                results.append({
                    'session_id': memory_id,
                    'score': final_score,
                    'content': content
                })
            
            results.sort(key=lambda x: x['score'], reverse=True)
            results = results[:top_k]
            
            for result in results:
                self.memory_frequency[result['session_id']] += 1
            
            return results
            
        except Exception as e:
            logger.error(f"Error during memory search: {e}", exc_info=True)
            return []
            
    # ส่วนที่เหลือของคลาส (update_memory, reset_frequency, etc.) สามารถคงไว้เหมือนเดิม
    # แต่การ update/remove จะซับซ้อนขึ้นเล็กน้อยกับ IndexIDMap และต้องมีการ rebuild index เป็นระยะๆ
    # เพื่อความง่ายในขั้นนี้ จะขอเน้นที่การ save/load ก่อนครับ

    def update_memory(self, memory_id: str, content: str, emotion: str = 'neutral'):
        """
        เพิ่มหรืออัปเดต memory ใน memory_dict และอัปเดต index พร้อม save index ใหม่
        """
        self.memory_dict[memory_id] = {'content': content, 'emotion': emotion}
        self.rebuild_index()

    def remove_memory(self, memory_id: str):
        """
        ลบ memory ออกจาก memory_dict และอัปเดต index พร้อม save index ใหม่
        """
        if memory_id in self.memory_dict:
            del self.memory_dict[memory_id]
            self.rebuild_index()

    def rebuild_index(self):
        """
        สร้าง index ใหม่ทั้งหมดจาก memory_dict ปัจจุบัน และ save index ใหม่ (ถ้า auto_save=True)
        """
        self._build_index()
        if self.auto_save:
            self.save_index(self.index_path)

    def force_save(self):
        """
        บังคับบันทึก index แม้ว่า auto_save จะเป็น False
        """
        self.save_index(self.index_path)
        logger.info("Force saved index")

    def toggle_auto_save(self, enabled: bool = None):
        """
        เปิด/ปิด auto-save หรือดูสถานะปัจจุบัน
        
        Args:
            enabled (bool): True=เปิด, False=ปิด, None=ดูสถานะ
        """
        if enabled is None:
            logger.info(f"Auto-save is currently: {'ON' if self.auto_save else 'OFF'}")
            return self.auto_save
        
        self.auto_save = enabled
        logger.info(f"Auto-save {'enabled' if enabled else 'disabled'}")
        return self.auto_save

    def reset_frequency(self):
        """
        รีเซ็ตความถี่ในการใช้ความทรงจำ (optionally save index)
        """
        self.memory_frequency.clear()
        logger.info("Memory frequency reset")
        self.save_index(self.index_path)

    def generate_memory_id(self) -> str:
        """
        สร้าง memory ID อัตโนมัติในรูปแบบ msg_YYYYMMDD_HHMMSS
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"msg_{timestamp}"

    def is_meaningful(self, text: str) -> bool:
        """
        ตรวจสอบว่าข้อความมีความหมายหรือไม่
        กรองข้อความสั้นๆ หรือไม่มีเนื้อหา
        """
        if not text or not text.strip():
            return False
        
        # ลบ whitespace และตรวจสอบความยาว
        clean_text = text.strip()
        if len(clean_text) < 10:
            return False
        
        # กรองข้อความที่ไม่มีความหมาย
        meaningless_patterns = [
            r'^[อืมฮืม]+$',  # อืม, ฮืม
            r'^[555]+$',      # 555
            r'^[โอเคโอเค]+$',  # โอเค
            r'^[ครับค่ะ]+$',   # ครับ, ค่ะ
            r'^[ใช่ไม่]+$',    # ใช่, ไม่
            r'^\s*$',         # whitespace เท่านั้น
        ]
        
        for pattern in meaningless_patterns:
            if re.match(pattern, clean_text, re.IGNORECASE):
                return False
        
        return True

    def add_memory(self, text: str, memory_id: Optional[str] = None, emotion: str = 'neutral') -> Optional[str]:
        """
        เพิ่ม memory ใหม่ด้วย auto generate ID
        
        Args:
            text (str): ข้อความที่จะเพิ่ม
            memory_id (Optional[str]): ID ที่กำหนดเอง ถ้าไม่ระบุจะสร้างให้
            emotion (str): อารมณ์ของข้อความ
            
        Returns:
            Optional[str]: memory_id ที่ใช้ หรือ None ถ้าเพิ่มไม่สำเร็จ
        """
        # ตรวจสอบว่าข้อความมีความหมายหรือไม่
        if not self.is_meaningful(text):
            logger.info(f"Skipping meaningless text: '{text[:50]}...'")
            return None
        
        # สร้าง memory_id ถ้าไม่ระบุ
        if memory_id is None:
            memory_id = self.generate_memory_id()
        
        # เพิ่ม memory ใหม่
        self.memory_dict[memory_id] = {
            'content': text,
            'emotion': emotion,
            'timestamp': datetime.now().isoformat()
        }
        
        # อัปเดต index
        self.rebuild_index()
        
        logger.info(f"Added memory: {memory_id} - '{text[:50]}...'")
        return memory_id

    def add_from_conversation(self, text: str, emotion: str = 'neutral') -> Optional[str]:
        """
        เพิ่มข้อความจากการสนทนา (wrapper สำหรับ add_memory)
        
        Args:
            text (str): ข้อความจากการสนทนา
            emotion (str): อารมณ์ของข้อความ
            
        Returns:
            Optional[str]: memory_id ที่สร้าง หรือ None ถ้าเพิ่มไม่สำเร็จ
        """
        return self.add_memory(text, emotion=emotion)

    def add_conversation_pair(self, user_input: str, assistant_response: str, emotion: str = 'neutral') -> Optional[str]:
        """
        เพิ่มคู่บทสนทนา (user + assistant)
        
        Args:
            user_input (str): ข้อความของผู้ใช้
            assistant_response (str): ข้อความของระบบ
            emotion (str): อารมณ์ของบทสนทนา
            
        Returns:
            Optional[str]: memory_id ที่สร้าง หรือ None ถ้าเพิ่มไม่สำเร็จ
        """
        conversation_text = f"User: {user_input} | Assistant: {assistant_response}"
        return self.add_memory(conversation_text, emotion=emotion)

    def get_memory_count(self) -> int:
        """
        ดูจำนวน memory ทั้งหมด
        """
        return len(self.memory_dict)

    def get_memory_info(self) -> Dict:
        """
        ดูข้อมูลสรุปของ memory system
        """
        return {
            'total_memories': len(self.memory_dict),
            'indexed_memories': len(self.memory_ids_map) if self.memory_ids_map else 0,
            'active_memories': len(self.active_memory_ids),
            'auto_save': self.auto_save,
            'index_path': self.index_path,
            'index_exists': self.index is not None,
            'model_loaded': self.model is not None
        }

    def list_recent_memories(self, limit: int = 10) -> List[Dict]:
        """
        ดู memory ล่าสุด
        
        Args:
            limit (int): จำนวน memory ที่ต้องการดู
            
        Returns:
            List[Dict]: รายการ memory ล่าสุด
        """
        memories = []
        for memory_id, memory_data in list(self.memory_dict.items())[-limit:]:
            memories.append({
                'id': memory_id,
                'content': memory_data.get('content', '')[:100] + '...' if len(memory_data.get('content', '')) > 100 else memory_data.get('content', ''),
                'emotion': memory_data.get('emotion', 'neutral'),
                'timestamp': memory_data.get('timestamp', 'N/A')
            })
        return memories

    def clear_all_memories(self):
        """
        ลบ memory ทั้งหมดและสร้าง index ใหม่
        """
        self.memory_dict.clear()
        self.memory_ids_map.clear()
        self.memory_frequency.clear()
        self.active_memory_ids.clear()
        self.rebuild_index()
        logger.info("All memories cleared")

    def export_memories(self, filepath: str):
        """
        ส่งออก memory ทั้งหมดเป็นไฟล์ JSON
        
        Args:
            filepath (str): path ของไฟล์ที่จะบันทึก
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.memory_dict, f, ensure_ascii=False, indent=2)
            logger.info(f"Memories exported to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export memories: {e}")

    def import_memories(self, filepath: str):
        """
        นำเข้า memory จากไฟล์ JSON
        
        Args:
            filepath (str): path ของไฟล์ที่จะโหลด
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                imported_memories = json.load(f)
            
            # รวม memory ใหม่เข้ากับที่มีอยู่
            self.memory_dict.update(imported_memories)
            self.rebuild_index()
            logger.info(f"Imported {len(imported_memories)} memories from {filepath}")
        except Exception as e:
            logger.error(f"Failed to import memories: {e}")

    def add_to_buffer(self, text: str):
        """เก็บบทสนทนาไว้ใน buffer จนพร้อมฝัง"""
        if not self.is_meaningful(text):
            logger.info(f"Buffer: Skipping meaningless text: '{text[:50]}...'")
            return

        self.conversation_buffer.append(text)
        logger.info(f"Buffer: Added text. Current size: {len(self.conversation_buffer)}/{self.buffer_limit}")
        if len(self.conversation_buffer) >= self.buffer_limit:
            self.flush_buffer_if_ready()

    def flush_buffer_if_ready(self):
        """ตรวจสอบและฝังความทรงจำเมื่อ buffer พร้อม"""
        if len(self.conversation_buffer) < self.buffer_limit:
            return

        combined_text = " ".join(self.conversation_buffer).strip()
        
        # ตรวจสอบอีกครั้งหลังรวมข้อความ
        if not self.is_meaningful(combined_text):
            logger.info("Buffer: Flushing buffer because combined text is not meaningful.")
            self.conversation_buffer.clear()
            return

        # สร้าง ID พิเศษสำหรับ memory ที่มาจาก buffer
        memory_id = f"conv_auto_{datetime.now().strftime('%Y%m%d_%H%M%S')}_#{self.memory_counter}_#auto_buffer_v0.1"
        self.memory_counter += 1
        
        # เพิ่ม memory ใหม่ (rebuild และ auto-save จะถูกจัดการในนี้)
        self.add_memory(combined_text, memory_id=memory_id)

        # ล้าง buffer หลังฝัง memory
        self.conversation_buffer.clear()
        logger.info(f"Buffer: Flushed and created new memory {memory_id}")

    def maybe_remember(self, text: str, emotion: str = 'neutral') -> Optional[str]:
        """
        ตรวจสอบว่าข้อความมีความหมายหรือไม่ ถ้าใช่ → เพิ่มเป็น memory ใหม่
        
        Args:
            text (str): ข้อความที่อาจจะจดจำ
            emotion (str): อารมณ์ของข้อความ
            
        Returns:
            Optional[str]: memory_id ที่สร้าง หรือ None ถ้าไม่จดจำ
        """
        if self.is_meaningful(text):
            return self.add_memory(text, emotion=emotion)
        else:
            logger.info(f"Decided not to remember meaningless text: '{text[:50]}...'")
            return None