# core/self_schema.py
# DEPRECATED: This module is replaced by core_mapper.py (self-awareness system)

import os
import ast
# from .auto_core_mapper import get_core_connection_brief  # ลบระบบเดิมออก

# ฟังก์ชันนี้จะถูกแทนที่ด้วยระบบใหม่
# def auto_discover_schema(base_dir="core"):
#     schema = []
#     for fname in os.listdir(base_dir):
#         path = os.path.join(base_dir, fname)
#         if not os.path.isfile(path):
#             continue
#         if fname.endswith(".py") and not fname.startswith("__"):
#             key = fname.replace(".py", "")
#             schema.append(key)
#     return schema

# def get_self_schema_brief():
#     return get_core_connection_brief()

# ===== Example Usage/Test =====
if __name__ == "__main__":
    pass 