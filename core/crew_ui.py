from core.identity_core import IdentityCore

def main():
    """
    จุดเริ่มต้นของโปรแกรม ทำหน้าที่เป็น "หูและปาก" ที่บริสุทธิ์
    สำหรับ IdentityCore ที่สมบูรณ์แล้ว
    """
    try:
        # IdentityCore จะเริ่มต้นและ print สถานะของตัวเอง
        melah_core = IdentityCore()
    except Exception as e:
        print(f"💥 เกิดข้อผิดพลาดร้ายแรงขณะเริ่มต้นระบบ: {e}")
        return

    print("\\n" + "="*20)
    print("   Melah is Listening   ")
    print("="*20)
    print("พิมพ์ข้อความเพื่อสนทนา | พิมพ์ 'exit' เพื่อจบการทำงาน")

    while True:
        try:
            # 1. เป็นหู: รับฟังคำสั่งจากผู้ใช้
            user_input = input("คุณ: ").strip()

            if user_input.lower() == 'exit':
                print("...แล้วพบกันใหม่")
                break
            
            if not user_input:
                continue

            # 2. ส่งเสียงให้สมองคิด และเป็นปากเพื่อพูดสิ่งที่สมองตอบกลับมา
            print("⏳ Melah is thinking...")
            response = melah_core.process_input(user_input)
            
            if response:
                print(f"เมล่า: {response}")
            
            # 3. ให้สมองได้ไตร่ตรองหลังการสนทนา - ส่วนนี้ไม่จำเป็นแล้ว
            # กระบวนการ reflect ถูกรวมอยู่ใน process_input แล้ว
            # melah_core.reflect_and_update()

        except KeyboardInterrupt:
            print("\\n...แล้วพบกันใหม่")
            break
        except Exception as e:
            error_message = f"UI Loop Error: {e}"
            print(f"‼️ เกิดข้อผิดพลาดที่ไม่คาดคิด: {error_message}")
            if 'melah_core' in locals():
                melah_core.log_error(error_message)


if __name__ == "__main__":
    main()
