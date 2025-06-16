# core/crew_ui.py

from core.llm_connector import LLMConnector
from core.identity_core import IdentityCore

def main():
    config = {
        "qwen": {"host": "localhost", "port": 8000},
        "openai": {"api_key": "sk-xxxx"},
        "gemma": {"host": "localhost", "port": 8000}
    }
    llm = LLMConnector(config)
    identity = IdentityCore()

    print("=== Welcome to Melah Crew UI ===")
    while True:
        print("\nเลือก LLM ปัจจุบัน (qwen/openai/gemma/auto/exit):")
        choice = input("LLM > ").strip()
        if choice == "exit":
            break
        elif choice in ["qwen", "openai", "gemma"]:
            llm.switch_llm(choice)
            print(f"🔄 เปลี่ยน LLM เป็น {choice} แล้ว")
        elif choice == "auto":
            # ตัวอย่าง: auto switch ตาม context size
            context_size = int(input("context_size > "))
            llm.auto_switch(context_size=context_size)
            print(f"🔄 (Auto) LLM ปัจจุบัน: {llm.active_llm}")
        else:
            print("❌ ไม่พบ LLM นี้")

        prompt = input("\nใส่คำถาม/ข้อความสำหรับ AI: ")
        # ดึง context/summary จาก memory ถ้าต้องการ
        sessions = identity.path_manager.list_sessions()
        last_session = sessions[-1] if sessions else None
        context = identity.traverse_memory(last_session) if last_session else []
        summary = identity.get_summary_event(last_session) if last_session else None

        # ส่ง prompt และ context ให้ LLM
        response = llm.generate(prompt, context=context, summary=summary)
        print(f"\n🧠 Melah Response: {response}")

if __name__ == "__main__":
    main()
