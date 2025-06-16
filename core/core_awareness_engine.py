# core/core_awareness_engine.py
# ทำให้ MelahPC รู้ตัวว่าเป็น Identity Core และตรวจสอบการเชื่อมต่อระบบอื่น ๆ

from datetime import datetime

class CoreAwarenessEngine:
    def __init__(self, identity_core, modules: dict):
        self.identity_core = identity_core
        self.modules = modules
        self.status_report = {}
        self.timestamp = datetime.now()

    def check_identity_core(self):
        try:
            assert self.identity_core.is_active()
            self.status_report['Identity Core'] = '✅ Active'
        except Exception as e:
            self.status_report['Identity Core'] = f'❌ Error: {str(e)}'

    def check_module(self, name, module):
        try:
            assert module is not None
            assert hasattr(module, 'is_connected') and module.is_connected()
            self.status_report[name] = '✅ Connected'
        except Exception as e:
            self.status_report[name] = f'❌ Error: {str(e)}'

    def verify_all_modules(self):
        self.check_identity_core()
        for name, module in self.modules.items():
            self.check_module(name, module)
        return self.status_report

    def report_self_awareness(self):
        message = (
            f"[System Identity] I am the Identity Core of MelahPC.\n"
            f"Timestamp: {self.timestamp.isoformat()}\n"
            f"Connected Modules: {list(self.modules.keys())}\n"
            f"Self-Awareness Status: {self.status_report.get('Identity Core', 'Unknown')}"
        )
        return message

# ===== Example Usage =====
if __name__ == "__main__":
    # Mock modules for test
    class DummyModule:
        def is_connected(self):
            return True

    class IdentityCore:
        def is_active(self):
            return True

    identity = IdentityCore()
    modules = {
        'MemoryManager': DummyModule(),
        'LLMConnector': DummyModule(),
        'QuantumEngine': DummyModule()
    }

    cae = CoreAwarenessEngine(identity, modules)
    print(cae.verify_all_modules())
    print(cae.report_self_awareness()) 