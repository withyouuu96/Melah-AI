# core/vector_memory_retriever.py

import logging
import os
from typing import List, Dict, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

logger = logging.getLogger(__name__)

class VectorMemoryRetriever:
    """
    Semantic Retriever สำหรับ chain-of-memory
    รับ memory_dict (subset) จาก chain/summary node ภายนอกเท่านั้น
    ไม่สามารถเพิ่ม/ลบ/แก้ไข memory จริงได้
    ใช้สำหรับ semantic search เฉพาะ subset ที่กำหนด
    รองรับ scoring explanation, weighted fields, memory hook/observer
    """
    def __init__(self, memory_dict: Dict, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2',
                 field_weights: Optional[Dict[str, float]] = None,
                 search_hook: Optional[callable] = None):
        """
        Args:
            memory_dict (Dict): Dictionary ของความทรงจำจาก MemoryMetaManager
            model_name (str): ชื่อของ pre-trained model จาก sentence-transformers
            field_weights (dict): กำหนดน้ำหนัก field เช่น emotion, tags
            search_hook (callable): observer/hook เรียกเมื่อค้นหา (รับ args: query, results)
        """
        self.memory_dict = memory_dict
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.memory_ids_map = []
        self.field_weights = field_weights or {}
        self.search_hook = search_hook
        self._build_index()

    def set_memories(self, memory_dict: Dict):
        """รับ memories ใหม่ (เช่น จาก chain/summary node) แล้ว rebuild index"""
        self.memory_dict = memory_dict
        self._build_index()

    def _build_index(self):
        if not self.model or not self.memory_dict:
            self.index = None
            return
        texts = []
        self.memory_ids_map = list(self.memory_dict.keys())
        for memory_id in self.memory_ids_map:
            memory = self.memory_dict[memory_id]
            text = memory.get('content', '')
            texts.append(text)
        if not texts:
            self.index = None
            return
        embeddings = self.model.encode(texts)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index = faiss.IndexIDMap(self.index)
        ids = np.arange(len(self.memory_ids_map))
        self.index.add_with_ids(embeddings.astype('float32'), ids)

    def _score_with_fields(self, memory, base_score: float, query: str) -> Tuple[float, dict]:
        """
        ปรับ score ตาม field weights (เช่น emotion, tags)
        Return: (final_score, explanation_dict)
        """
        explanation = {'base_score': base_score}
        score = base_score
        # ตัวอย่าง: emotion weight
        if 'emotion' in self.field_weights:
            emotion = memory.get('emotion', 'neutral')
            if isinstance(emotion, list):
                emotion = emotion[0] if emotion else 'neutral'
            weight = self.field_weights['emotion'].get(emotion, 1.0)
            score *= weight
            explanation['emotion_weight'] = weight
        # ตัวอย่าง: tag weight
        if 'tags' in self.field_weights:
            tags = memory.get('tags', [])
            if isinstance(tags, str):
                tags = [tags]
            tag_weights = [self.field_weights['tags'].get(tag, 1.0) for tag in tags]
            tag_weight = max(tag_weights) if tag_weights else 1.0
            score *= tag_weight
            explanation['tag_weight'] = tag_weight
        # สามารถเพิ่ม logic อื่นๆ ได้
        return score, explanation

    def search(self, query: str, top_k: int = 5, threshold: Optional[float] = None, explain: bool = True) -> List[Dict]:
        if not self.model or not self.index:
            return []
        query_embedding = self.model.encode([query])[0]
        scores, indices = self.index.search(query_embedding.reshape(1, -1).astype('float32'), top_k * 2)
        raw_scores = scores[0]
        # normalize score 0-1
        max_score = float(np.max(raw_scores)) if len(raw_scores) > 0 else 1.0
        min_score = float(np.min(raw_scores)) if len(raw_scores) > 0 else 0.0
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            memory_id = self.memory_ids_map[idx]
            memory = self.memory_dict[memory_id]
            norm_score = (float(score) - min_score) / (max_score - min_score + 1e-8)
            final_score, explanation = self._score_with_fields(memory, norm_score, query)
            if threshold is not None and final_score < threshold:
                continue
            result = {
                'memory_id': memory_id,
                'score': final_score,
                'content': memory.get('content', ''),
            }
            if explain:
                result['explanation'] = explanation
            results.append(result)
        results = sorted(results, key=lambda x: x['score'], reverse=True)[:top_k]
        # memory hook/observer
        if self.search_hook:
            try:
                self.search_hook(query=query, results=results)
            except Exception as e:
                logger.warning(f"search_hook error: {e}")
        return results

    # ปิดการเพิ่ม/ลบ/แก้ไข memory จริง
    def add_memory(self, *args, **kwargs):
        raise NotImplementedError("Use storage/chain manager for memory operations.")

    def remove_memory(self, *args, **kwargs):
        raise NotImplementedError("Use storage/chain manager for memory operations.")

    def update_memory(self, *args, **kwargs):
        raise NotImplementedError("Use storage/chain manager for memory operations.")