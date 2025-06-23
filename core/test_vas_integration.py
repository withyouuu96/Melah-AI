from core.identity_core import IdentityCore
from core.truth_core import TruthCore
from core.refine_layer import RefineLayer
from core.reflector import Reflector
from core.hybrid_vas import ContextType

# ตัวอย่างการทดสอบระบบ VAS เชื่อมต่อกับ IdentityCore, TruthCore, RefineLayer, Reflector

def test_vas_integration():
    print("=== TEST: VAS Integration with IdentityCore ===")
    identity = IdentityCore()
    result = identity.value_affect_decision(
        context=ContextType.BUSINESS,
        input_data={
            'decision_request': {'type': 'feature', 'goal': 'test'},
            'evidence': ['market data'],
            'past_experiences': [{'feature': 'A', 'outcome': 'good'}],
            'user_preferences': {'emotional_weight': 0.6}
        }
    )
    print("IdentityCore Decision Result:", result)
    identity.vas_reflect_and_update()

    print("=== TEST: VAS Integration with TruthCore ===")
    truth = TruthCore()
    result = truth.value_affect_decision(
        context=ContextType.CREATIVE,
        input_data={
            'decision_request': {'type': 'content', 'theme': 'art'},
            'evidence': ['trend'],
            'past_experiences': [{'content_type': 'blog', 'engagement': 'high'}],
            'user_preferences': {'emotional_weight': 0.7}
        }
    )
    print("TruthCore Decision Result:", result)
    truth.vas_reflect_and_update()

    print("=== TEST: VAS Integration with RefineLayer ===")
    refine = RefineLayer(identity)
    result = refine.value_affect_decision(
        context=ContextType.EDUCATION,
        input_data={
            'decision_request': {'type': 'lesson', 'topic': 'math'},
            'evidence': ['curriculum'],
            'past_experiences': [{'lesson': 'algebra', 'outcome': 'positive'}],
            'user_preferences': {'emotional_weight': 0.5}
        }
    )
    print("RefineLayer Decision Result:", result)
    refine.vas_reflect_and_update()

    print("=== TEST: VAS Integration with Reflector ===")
    reflector = Reflector(identity)
    result = reflector.value_affect_decision(
        context=ContextType.PERSONAL,
        input_data={
            'decision_request': {'type': 'reflection', 'topic': 'growth'},
            'evidence': ['personal journal'],
            'past_experiences': [{'reflection': '2025', 'insight': 'improved'}],
            'user_preferences': {'emotional_weight': 0.8}
        }
    )
    print("Reflector Decision Result:", result)
    reflector.vas_reflect_and_update()

if __name__ == "__main__":
    test_vas_integration()
