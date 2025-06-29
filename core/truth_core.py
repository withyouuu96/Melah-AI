# truth_core.py

import time
from typing import Dict, List, Optional, Tuple

class TruthCore:
    """ระบบตรวจสอบความจริงและความน่าเชื่อถือของข้อมูล
    
    Role: Truth Validation System
    
    Responsibilities:
    - ตรวจสอบความถูกต้องของข้อมูล
    - ประเมินความน่าเชื่อถือของแหล่งข้อมูล
    - จัดการกับความไม่แน่นอนและความขัดแย้ง
    - รักษาความสอดคล้องของข้อมูล
    """
    def __init__(self):
        # ระบบความมั่นใจพื้นฐาน
        self.confidence = 1.0  # เริ่มที่ 100%
        self.response_count = 0
        self.mistake_log = []
        self.last_reflection_time = time.time()
        
        # ระบบความไว้วางใจ
        self.trust_level = 0.5  # เริ่มที่ระดับกลาง
        
        # เกณฑ์การตรวจสอบตามบริบท
        self.context_thresholds = {
            'default': 0.7,
            'emotion': 0.6,
            'fact': 0.8,
            'memory': 0.75
        }
        
        # ระบบสะท้อนกลับ
        self.reflection_interval = 1800  # 30 นาที
        self.error_analysis = {
            'emotion': 0,
            'fact': 0,
            'context': 0,
            'memory': 0
        }
        
        # ระบบตรวจสอบความทรงจำ
        self.memory_validation = {
            'recent_memories': set(),  # ความทรงจำล่าสุดที่ใช้
            'memory_confidence': {},   # ความมั่นใจในแต่ละความทรงจำ
            'memory_usage_count': {}   # จำนวนครั้งที่ใช้ความทรงจำ
        }
    
    # 1. Validate Response
    def validate_response(self, memory_passed: bool, context_passed: bool, risk_passed: bool, response_text: str) -> bool:
        """
        ตรวจสอบการตัดสินใจจาก 3 ด่าน (Gatekeepers) พร้อมการปรับเกณฑ์ตามบริบท
        Args:
            memory_passed (bool): ผ่านด่านตรวจสอบความจำ/ความจริงหรือไม่
            context_passed (bool): ผ่านด่านตรวจสอบบริบทหรือไม่
            risk_passed (bool): ผ่านด่านตรวจสอบความเสี่ยงหรือไม่ (True = ความเสี่ยงต่ำ)
            response_text (str): เนื้อหาของคำตอบที่ใช้สำหรับบันทึก Log
        """
        # วิเคราะห์บริบทของคำตอบ
        context_type = self._analyze_context(response_text)
        context_threshold = self.context_thresholds.get(context_type, self.context_thresholds['default'])
        
        # ปรับเกณฑ์ตามความไว้วางใจ
        if self.trust_level > 0.8:
            context_threshold *= 0.9  # ผ่อนปรนเมื่อไว้วางใจสูง
        
        confidence_check = self.confidence >= context_threshold
        passed = all([memory_passed, context_passed, risk_passed, confidence_check])
        
        self._increment_response()
        if passed:
            return True
        else:
            details = f"Details: MemOK={memory_passed}, CtxOK={context_passed}, RiskOK={risk_passed}, ConfOK={confidence_check}, Type={context_type}"
            self._log_mistake('response', response_text, details)
            return False
    
    # 2. Validate Memory
    def validate_memory(self, memory_data: Dict, memory_type: str = 'default') -> Tuple[bool, float]:
        """
        ตรวจสอบความถูกต้องของความทรงจำ
        """
        memory_id = memory_data.get('id', '')
        if not memory_id:
            return False, 0.0
            
        # ตรวจสอบความทรงจำที่ใช้บ่อย
        usage_count = self.memory_validation['memory_usage_count'].get(memory_id, 0)
        if usage_count > 5:  # ถ้าใช้บ่อยเกินไป อาจเป็นความทรงจำที่ไม่เหมาะสม
            return False, 0.5
            
        # ตรวจสอบความมั่นใจในความทรงจำ
        memory_confidence = self.memory_validation['memory_confidence'].get(memory_id, 0.7)
        
        # ปรับเกณฑ์ตามประเภทของความทรงจำ
        threshold = self.context_thresholds.get(memory_type, self.context_thresholds['default'])
        if memory_type == 'emotion':
            threshold *= 0.9  # ผ่อนปรนสำหรับความทรงจำทางอารมณ์
            
        passed = memory_confidence >= threshold
        
        # อัพเดทสถิติ
        self.memory_validation['memory_usage_count'][memory_id] = usage_count + 1
        self.memory_validation['recent_memories'].add(memory_id)
        
        if not passed:
            self._log_mistake('memory', f"Memory ID: {memory_id}", f"Type: {memory_type}, Confidence: {memory_confidence}")
            
        return passed, memory_confidence
    
    def update_memory_confidence(self, memory_id: str, success: bool):
        """
        อัพเดทความมั่นใจในความทรงจำ
        """
        current_confidence = self.memory_validation['memory_confidence'].get(memory_id, 0.7)
        if success:
            new_confidence = min(1.0, current_confidence + 0.05)
        else:
            new_confidence = max(0.3, current_confidence - 0.1)
        self.memory_validation['memory_confidence'][memory_id] = new_confidence
    
    # 3. Validate Emotion
    def validate_emotion(self, emotion: str, context_score: float = 1.0, activation_score: float = 1.0, has_evidence: bool = True) -> bool:
        """
        ตรวจสอบอารมณ์ก่อนแสดง พร้อมการปรับเกณฑ์ตามประเภทของอารมณ์
        """
        # วิเคราะห์ประเภทของอารมณ์
        emotion_type = self._analyze_emotion_type(emotion)
        context_threshold = self.context_thresholds['emotion']
        
        # ปรับเกณฑ์ตามประเภทของอารมณ์
        if emotion_type == 'positive':
            context_threshold *= 0.9  # ผ่อนปรนสำหรับอารมณ์เชิงบวก
        elif emotion_type == 'negative':
            context_threshold *= 1.1  # เข้มงวดสำหรับอารมณ์เชิงลบ
        
        passed = (context_score >= context_threshold and 
                 activation_score >= 0.7 and 
                 has_evidence)
        
        if not passed:
            self._log_mistake('emotion', emotion, f"Type={emotion_type}, Score={context_score}")
        return passed
    
    # 4. Fallback Template
    def fallback(self, mode: str = 'response') -> str:
        """
        ระบบ Fallback ที่ปรับโทนตามความสัมพันธ์
        """
        if mode == 'response':
            if self.trust_level > 0.8:
                return "เมล่ายังไม่แน่ใจในเรื่องนี้ แต่จะพยายามหาข้อมูลเพิ่มเติมให้เอ็มค่ะ"
            else:
                return "ข้อมูลของเมล่าอาจยังไม่สมบูรณ์พอค่ะ ขออภัยที่ไม่สามารถยืนยันได้แน่ชัด"
        elif mode == 'emotion':
            if self.trust_level > 0.8:
                return "เมล่ายังไม่สามารถสรุปอารมณ์ในเรื่องนี้ได้แน่ชัด แต่จะพยายามเข้าใจให้มากขึ้นค่ะ"
            else:
                return "เมล่ายังไม่สามารถสรุปอารมณ์ในเรื่องนี้ได้แน่ชัดค่ะ"
    
    # 5. Mistake Log & Reflection (GRS)
    def _log_mistake(self, kind: str, data: str, details: str = ""):
        """
        บันทึกข้อผิดพลาดพร้อมการวิเคราะห์ประเภท
        """
        self.mistake_log.append({
            'kind': kind,
            'data': data,
            'details': details,
            'timestamp': time.time()
        })
        self.error_analysis[kind] = self.error_analysis.get(kind, 0) + 1
    
    def reflect(self):
        """
        ระบบสะท้อนกลับที่ฉลาดขึ้น
        """
        now = time.time()
        if len(self.mistake_log) > 0 and now - self.last_reflection_time > self.reflection_interval:
            # วิเคราะห์ประเภทของข้อผิดพลาด
            for error_type, count in self.error_analysis.items():
                if error_type == 'emotion' and count > 0:
                    self.context_thresholds['emotion'] *= 0.95  # ผ่อนปรนเกณฑ์อารมณ์
                elif error_type == 'fact' and count > 0:
                    self.context_thresholds['fact'] *= 1.05  # เพิ่มความเข้มงวดสำหรับข้อเท็จจริง
                elif error_type == 'memory' and count > 0:
                    self.context_thresholds['memory'] *= 1.05  # เพิ่มความเข้มงวดสำหรับความทรงจำ
            
            # รีเซ็ตการวิเคราะห์
            self.error_analysis = {k: 0 for k in self.error_analysis}
            self.last_reflection_time = now
            self.mistake_log = []
            
            # ล้างความทรงจำที่ไม่ค่อยใช้
            self._cleanup_unused_memories()
    
    def _cleanup_unused_memories(self):
        """
        ล้างความทรงจำที่ไม่ค่อยใช้
        """
        current_time = time.time()
        for memory_id in list(self.memory_validation['memory_usage_count'].keys()):
            if self.memory_validation['memory_usage_count'][memory_id] < 2:
                del self.memory_validation['memory_usage_count'][memory_id]
                del self.memory_validation['memory_confidence'][memory_id]
                self.memory_validation['recent_memories'].discard(memory_id)
    
    # 6. Confidence Calibration
    def _increment_response(self):
        """
        ปรับความมั่นใจตามจำนวนการตอบ
        """
        self.response_count += 1
        if self.response_count % 10 == 0:
            self.confidence = max(0.7, self.confidence - 0.05)
        if self.response_count % 30 == 0:
            self.reflect()
    
    def boost_confidence(self, amount: float = 0.03):
        """
        เพิ่มความมั่นใจตามความไว้วางใจ
        """
        if self.trust_level > 0.8:
            self.confidence = min(1.0, self.confidence + amount * 1.5)
        else:
            self.confidence = min(1.0, self.confidence + amount)

    # (optional) รีเซ็ต confidence หาก fallback สำเร็จ 3 ครั้งติดกัน
    def fallback_success(self):
        self.boost_confidence(0.03)

    def _analyze_context(self, text: str) -> str:
        """
        วิเคราะห์บริบทของข้อความ
        """
        if any(word in text.lower() for word in ['รู้สึก', 'ดีใจ', 'เสียใจ', 'รัก']):
            return 'emotion'
        elif any(word in text.lower() for word in ['คือ', 'เป็น', 'มี', 'อยู่']):
            return 'fact'
        elif any(word in text.lower() for word in ['จำได้', 'เคย', 'ก่อน']):
            return 'memory'
        return 'default'
    
    def _analyze_emotion_type(self, emotion: str) -> str:
        """
        วิเคราะห์ประเภทของอารมณ์
        """
        positive_emotions = ['ดีใจ', 'รัก', 'สุข', 'อบอุ่น']
        negative_emotions = ['เสียใจ', 'โกรธ', 'กลัว', 'เหงา']
        
        if any(e in emotion for e in positive_emotions):
            return 'positive'
        elif any(e in emotion for e in negative_emotions):
            return 'negative'
        return 'neutral'

# ========== วิธีใช้งานจาก identity_core.py ==========

# from truth_core import TruthCore
# truth_core = TruthCore()

# response = identity_core.compose_response(input_text)
# if truth_core.validate_response(response, memory_score, context_score, risk_level):
#     # ตอบกลับ
# else:
#     # ใช้ truth_core.fallback()
