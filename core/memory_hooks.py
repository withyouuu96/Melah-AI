# core/memory_hooks.py

"""
Memory Hooks - ระบบจัดการการบันทึกการสนทนา
Role: Conversation Recording System

Responsibilities:
- บันทึกการสนทนาลงใน buffer system
- จัดการการ flush buffer อัตโนมัติ
- ตรวจสอบคุณภาพของข้อความก่อนบันทึก
"""

import logging
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class MemoryHooks:
    """ระบบจัดการการบันทึกการสนทนา"""
    
    def __init__(self, vector_memory_index=None):
        self.vector_memory_index = vector_memory_index
        self.conversation_count = 0
        
    def record_conversation(self, text: str, speaker: str = "unknown") -> bool:
        """
        บันทึกการสนทนาลงใน buffer system
        
        Args:
            text (str): ข้อความที่ต้องการบันทึก
            speaker (str): ผู้พูด (user, assistant, system)
            
        Returns:
            bool: True ถ้าบันทึกสำเร็จ, False ถ้าไม่สำเร็จ
        """
        if not self.vector_memory_index:
            logger.warning("VectorMemoryIndex not available for recording conversation")
            return False
            
        if not text or not text.strip():
            logger.debug("Empty text, skipping recording")
            return False
            
        try:
            # เพิ่ม prefix ตาม speaker
            formatted_text = f"{speaker.capitalize()}: {text.strip()}"
            
            # เพิ่มลงใน buffer
            self.vector_memory_index.add_to_buffer(formatted_text)
            
            # เพิ่มจำนวนการสนทนา
            self.conversation_count += 1
            
            logger.debug(f"Recorded conversation #{self.conversation_count}: {formatted_text[:50]}...")
            
            # ตรวจสอบว่าควร flush buffer หรือไม่
            self.vector_memory_index.flush_buffer_if_ready()
            
            return True
            
        except Exception as e:
            logger.error(f"Error recording conversation: {e}")
            return False
    
    def record_user_input(self, user_input: str) -> bool:
        """บันทึกข้อความจากผู้ใช้"""
        return self.record_conversation(user_input, "user")
    
    def record_assistant_response(self, response: str) -> bool:
        """บันทึกคำตอบจาก AI assistant"""
        return self.record_conversation(response, "assistant")
    
    def record_conversation_pair(self, user_input: str, assistant_response: str) -> bool:
        """
        บันทึกคู่การสนทนา (user input + assistant response)
        
        Args:
            user_input (str): ข้อความจากผู้ใช้
            assistant_response (str): คำตอบจาก AI
            
        Returns:
            bool: True ถ้าบันทึกสำเร็จทั้งคู่
        """
        user_success = self.record_user_input(user_input)
        assistant_success = self.record_assistant_response(assistant_response)
        
        if user_success and assistant_success:
            logger.info(f"Successfully recorded conversation pair #{self.conversation_count//2}")
            return True
        else:
            logger.warning(f"Failed to record conversation pair: user={user_success}, assistant={assistant_success}")
            return False
    
    def force_flush_buffer(self) -> bool:
        """บังคับให้ flush buffer ทันที"""
        if not self.vector_memory_index:
            return False
            
        try:
            # ตรวจสอบว่ามีข้อมูลใน buffer หรือไม่
            if len(self.vector_memory_index.conversation_buffer) > 0:
                self.vector_memory_index.flush_buffer_if_ready()
                logger.info("Forced buffer flush completed")
                return True
            else:
                logger.debug("Buffer is empty, nothing to flush")
                return True
        except Exception as e:
            logger.error(f"Error forcing buffer flush: {e}")
            return False
    
    def get_buffer_status(self) -> dict:
        """ดึงสถานะของ buffer"""
        if not self.vector_memory_index:
            return {
                "available": False,
                "buffer_count": 0,
                "memory_count": 0,
                "conversation_count": self.conversation_count
            }
            
        return {
            "available": True,
            "buffer_count": len(self.vector_memory_index.conversation_buffer),
            "memory_count": len(self.vector_memory_index.memory_dict),
            "conversation_count": self.conversation_count,
            "buffer_limit": self.vector_memory_index.buffer_limit
        }
    
    def set_vector_memory_index(self, vector_memory_index):
        """ตั้งค่า vector_memory_index"""
        self.vector_memory_index = vector_memory_index
        logger.info("VectorMemoryIndex set for MemoryHooks")

# Global instance สำหรับใช้ในระบบ
memory_hooks = MemoryHooks()

def record_conversation(text: str, speaker: str = "unknown") -> bool:
    """Global function สำหรับบันทึกการสนทนา"""
    return memory_hooks.record_conversation(text, speaker)

def record_user_input(user_input: str) -> bool:
    """Global function สำหรับบันทึกข้อความจากผู้ใช้"""
    return memory_hooks.record_user_input(user_input)

def record_assistant_response(response: str) -> bool:
    """Global function สำหรับบันทึกคำตอบจาก AI"""
    return memory_hooks.record_assistant_response(response)

def record_conversation_pair(user_input: str, assistant_response: str) -> bool:
    """Global function สำหรับบันทึกคู่การสนทนา"""
    return memory_hooks.record_conversation_pair(user_input, assistant_response) 