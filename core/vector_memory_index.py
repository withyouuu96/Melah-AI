# core/vector_memory_index.py

import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer, util
import faiss
from collections import defaultdict

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
    def __init__(self, memory_dict: Dict, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        """
        Args:
            memory_dict (Dict): Dictionary ของความทรงจำจาก MemoryMetaManager
            model_name (str): ชื่อของ pre-trained model จาก sentence-transformers
        """
        self.memory_dict = memory_dict
        self.model = None
        self.index = None  # FAISS index
        self.embeddings = None  # Numpy array of embeddings
        self.memory_frequency = defaultdict(int)  # เก็บความถี่ในการใช้ความทรงจำ
        self.emotion_weights = {
            'happy': 1.2,
            'sad': 1.2,
            'angry': 1.2,
            'fear': 1.2,
            'surprise': 1.2,
            'neutral': 1.0
        }
        self.active_memory_ids = set()  # เก็บ session_id ของ memory ที่ถูกใช้งานล่าสุด
        self.max_active_memories = 1000  # ปรับขนาดตามทรัพยากร
        
        try:
            logger.info(f"Loading sentence-transformer model: {model_name}...")
            self.model = SentenceTransformer(model_name)
            logger.info("Model loaded successfully.")
            self._build_index()
        except Exception as e:
            logger.error(f"Failed to load sentence-transformer model or build index: {e}", exc_info=True)
            self.model = None

    def _build_index(self):
        """
        สร้าง FAISS index จากความทรงจำที่มี
        """
        if not self.model or not self.memory_dict:
            return

        try:
            # สร้าง embeddings
            texts = []
            for memory in self.memory_dict.values():
                # ใช้ content ถ้ามี, ถ้าไม่มีให้รวม field อื่น ๆ
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
            
            # สร้าง FAISS index
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # เพิ่ม embeddings เข้า index
            self.index.add(self.embeddings.astype('float32'))
            logger.info(f"Successfully built FAISS index with {len(texts)} memories")
            
        except Exception as e:
            logger.error(f"Error building FAISS index: {e}", exc_info=True)
            self.index = None

    def search(self, query: str, top_k: int = 1, exclude_ids: List[str] = None, emotion: str = 'neutral') -> List[Dict]:
        """
        ค้นหาความทรงจำที่เกี่ยวข้องกับ query
        
        Args:
            query (str): ข้อความที่ต้องการค้นหา
            top_k (int): จำนวนผลลัพธ์ที่ต้องการ
            exclude_ids (List[str]): รายการ ID ที่ไม่ต้องการให้ค้นหา
            emotion (str): อารมณ์ของ query
        
        Returns:
            List[Dict]: รายการความทรงจำที่เกี่ยวข้อง พร้อมคะแนน
        """
        if not self.model or not self.index:
            logger.warning("Model or index not available")
            return []

        try:
            # สร้าง embedding ของ query
            query_embedding = self.model.encode([query])[0]
            
            # ค้นหาด้วย FAISS
            scores, indices = self.index.search(
                query_embedding.reshape(1, -1).astype('float32'), 
                top_k * 2  # ค้นหาเพิ่มเพื่อกรอง
            )
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # FAISS returns -1 for empty slots
                    continue
                    
                memory_id = list(self.memory_dict.keys())[idx]
                
                # ข้ามถ้า ID อยู่ใน exclude_ids
                if exclude_ids and memory_id in exclude_ids:
                    continue
                
                # ปรับคะแนนตามความถี่ในการใช้
                frequency_penalty = 0.8 ** self.memory_frequency[memory_id]
                
                # ปรับคะแนนตามอารมณ์ (รองรับ top 3 อารมณ์)
                memory_emotions = self.memory_dict[memory_id].get('emotion', ['neutral'])
                if isinstance(memory_emotions, list):
                    top_emotions = memory_emotions[:3]
                else:
                    top_emotions = [memory_emotions]
                weights = [self.emotion_weights.get(e, 1.0) for e in top_emotions]
                emotion_weight = sum(weights) / len(weights) if weights else 1.0
                
                # คำนวณคะแนนสุดท้าย
                final_score = float(score) * frequency_penalty * emotion_weight
                
                # สร้าง content อัตโนมัติหากไม่มีฟิลด์ content
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
            
            # เรียงลำดับตามคะแนนและเลือก top_k
            results.sort(key=lambda x: x['score'], reverse=True)
            results = results[:top_k]
            
            # อัพเดทความถี่
            for result in results:
                self.memory_frequency[result['session_id']] += 1
            
            return results
            
        except Exception as e:
            logger.error(f"Error during memory search: {e}", exc_info=True)
            return []
            
    def update_memory(self, memory_id: str, content: str, emotion: str = 'neutral'):
        """
        อัพเดทความทรงจำและ index
        
        Args:
            memory_id (str): ID ของความทรงจำ
            content (str): เนื้อหาของความทรงจำ
            emotion (str): อารมณ์ของความทรงจำ
        """
        try:
            # อัพเดท memory_dict
            self.memory_dict[memory_id] = {
                'content': content,
                'emotion': emotion
            }
            
            # สร้าง embedding ใหม่
            new_embedding = self.model.encode([content])[0]
            
            # อัพเดท FAISS index
            if self.index:
                # ลบ embedding เก่า (ถ้ามี)
                if memory_id in self.memory_dict:
                    old_idx = list(self.memory_dict.keys()).index(memory_id)
                    self.index.remove_ids(np.array([old_idx]))
                
                # เพิ่ม embedding ใหม่
                self.index.add(new_embedding.reshape(1, -1).astype('float32'))
            
            logger.info(f"Successfully updated memory: {memory_id}")
            
        except Exception as e:
            logger.error(f"Error updating memory: {e}", exc_info=True)

    def reset_frequency(self):
        """
        รีเซ็ตความถี่ในการใช้ความทรงจำ
        """
        self.memory_frequency.clear()
        logger.info("Memory frequency reset")

    def mark_memory_active(self, session_id: str):
        """
        เพิ่ม session_id เข้า active memory set และ evict ถ้าเกิน limit
        """
        self.active_memory_ids.add(session_id)
        if len(self.active_memory_ids) > self.max_active_memories:
            # ลบ memory ที่เก่าที่สุด (FIFO)
            self.active_memory_ids = set(list(self.active_memory_ids)[-self.max_active_memories:])

    def get_active_memories(self) -> Dict:
        """
        คืนค่าเฉพาะ memory ที่อยู่ใน active set
        """
        return {mid: self.memory_dict[mid] for mid in self.active_memory_ids if mid in self.memory_dict}

    def summarize_old_memories(self):
        """
        สรุป/บีบอัดความทรงจำที่ไม่ active (เช่น เก่า, ไม่ถูกเรียกใช้)
        """
        old_ids = [mid for mid in self.memory_dict if mid not in self.active_memory_ids]
        for mid in old_ids:
            mem = self.memory_dict[mid]
            # สรุปเนื้อหา (ใช้ summary + insight แทน content เต็ม)
            mem['content'] = f"summary: {mem.get('summary', '')}; insight: {mem.get('insight', '')}"
            # สามารถลบ field อื่นที่ไม่จำเป็นได้ เช่น path, tags, emotion (ถ้าต้องการบีบอัด)
        return len(old_ids) 