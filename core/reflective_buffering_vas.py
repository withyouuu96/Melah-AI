from .vas import ValueAffectiveSystem
from .hybrid_vas import HybridVAS, ContextType
from .medical_safe_vas import MedicalSafeVAS

class ReflectiveBufferingVAS:
    """
    ระบบกลางสำหรับผสาน VAS, HybridVAS, MedicalSafeVAS
    รองรับโหมด buffering เพื่อความแม่นยำสูงสุดใน context สำคัญ
    """
    def __init__(self, vas=None, hybrid_vas=None, medical_vas=None):
        self.vas = vas or ValueAffectiveSystem()
        self.hybrid_vas = hybrid_vas or HybridVAS()
        self.medical_vas = medical_vas or MedicalSafeVAS()
        self.buffer = []  # เก็บ input ที่ยังไม่ควรบันทึก VU ทันที

    def _convert_to_vas_input(self, input_data):
        """
        แปลง input_data จาก hybrid/medical format ให้เป็น input ของ VAS
        """
        # พยายาม map จาก decision_request หรือ event_description
        event_description = None
        if 'event_description' in input_data:
            event_description = input_data['event_description']
        elif 'decision_request' in input_data:
            # ใช้ type หรือ summary ของ decision_request
            dr = input_data['decision_request']
            if isinstance(dr, dict):
                event_description = dr.get('type', str(dr))
            else:
                event_description = str(dr)
        else:
            event_description = str(input_data)
        # factor หลัก ถ้าไม่มีให้ default 0.5
        emotion = input_data.get('emotion')
        significance = input_data.get('significance')
        novelty = input_data.get('novelty')
        long_term_impact = input_data.get('long_term_impact')
        # ลองดึงจาก user_preferences ถ้ามี
        prefs = input_data.get('user_preferences', {})
        if emotion is None:
            emotion = prefs.get('emotional_weight', 0.5)
        if significance is None:
            significance = prefs.get('significance', 0.5)
        if novelty is None:
            novelty = prefs.get('novelty', 0.5)
        if long_term_impact is None:
            long_term_impact = prefs.get('long_term_focus', 0.5)
        return dict(
            event_description=event_description,
            emotion=emotion,
            significance=significance,
            novelty=novelty,
            long_term_impact=long_term_impact
        )

    def process_input(self, context, input_data):
        """
        context: ContextType หรือ str
        input_data: dict (ข้อมูลที่ต้องใช้กับแต่ละระบบ)
        """
        if isinstance(context, str):
            try:
                context = ContextType[context.upper()]
            except Exception:
                pass
        # Medical/Critical: ใช้ medical_vas, buffer ไว้
        if context in [ContextType.MEDICAL, ContextType.SAFETY_CRITICAL]:
            decision = self.medical_vas.process_clinical_case(**input_data)
            self.buffer.append(('medical', input_data))
            return decision
        # High/Business/Legal: ใช้ hybrid_vas, buffer ไว้
        elif context in [ContextType.FINANCIAL, ContextType.LEGAL, ContextType.BUSINESS]:
            decision = self.hybrid_vas.make_decision(context, **input_data)
            self.buffer.append(('hybrid', context, input_data))
            return decision
        # อื่นๆ: ใช้ VAS ปกติ
        else:
            vas_input = self._convert_to_vas_input(input_data)
            vu = self.vas.evaluate_input(**vas_input)
            return vu

    def reflect_and_update(self):
        """
        เรียกเมื่อถึงจังหวะเหมาะสม (เช่น จบ session)
        จะนำ buffer ไปประเมินและบันทึก VU ใน VAS
        """
        for item in self.buffer:
            if item[0] == 'medical':
                self.vas.evaluate_input(**self._convert_to_vas_input(item[1]))
            elif item[0] == 'hybrid':
                _, context, input_data = item
                self.vas.evaluate_input(**self._convert_to_vas_input(input_data))
        self.buffer.clear()
