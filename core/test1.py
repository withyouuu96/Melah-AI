from path_manager import PathManager

pm = PathManager(root_dir="memory_core")  # เปลี่ยน root_dir ตาม path จริง

# ทดสอบอ่าน session legacy
text = pm.read_session("archive/chat_sessions_legacy/Session_12.txt")
print("Session Content:", text[:300], "...")  # ดูแค่ 300 ตัวอักษรแรก

# ทดสอบอ่าน batch summary
batch = pm.get_batch_sessions_content("archive/session_summaries/summary_batch_3.txt")
for i, content in enumerate(batch):
    print(f"\nSession {i+1}: {content[:200]} ...")  # แสดง 200 ตัวอักษรแรกแต่ละ session
