import json
import time
import os
import logging
from datetime import datetime, timedelta
from collections import deque

# กำหนดค่าคงที่
LOG_PATH = "melah.log"  # log ที่ AI บันทึกเอง
AWARE_PATH = "self_aware.json"  # ใช้อัปเดตสภาพภายใน
MONITOR_LOG = "self_monitor.log"  # log ของระบบ monitor เอง
HISTORY_SIZE = 10  # จำนวนประวัติปัญหาที่เก็บ

# กำหนดคำหลักที่สื่อถึงปัญหา
ERROR_KEYWORDS = {
    "critical": ["fatal", "crash", "exception", "error", "failed"],
    "warning": ["timeout", "not responding", "slow", "warning"],
    "info": ["info", "debug", "trace"]
}

# กำหนดสถานะอารมณ์
EMOTION_STATES = {
    "critical": "unstable",
    "warning": "concerned",
    "ok": "calm"
}

# ตั้งค่า logging
logging.basicConfig(
    filename=MONITOR_LOG,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class IssueHistory:
    def __init__(self, max_size=HISTORY_SIZE):
        self.history = deque(maxlen=max_size)
    
    def add(self, issue):
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "module": issue["module"],
            "status": issue["status"],
            "message": issue["message"]
        })
    
    def get_all(self):
        return list(self.history)
    
    def get_recent(self, module_name=None, minutes=30):
        now = datetime.now()
        recent = []
        for issue in self.history:
            issue_time = datetime.fromisoformat(issue["timestamp"])
            if (now - issue_time) <= timedelta(minutes=minutes):
                if module_name is None or issue["module"] == module_name:
                    recent.append(issue)
        return recent

def read_recent_logs(log_path, max_lines=50):
    """อ่าน log ล่าสุดจากไฟล์"""
    try:
        if not os.path.exists(log_path):
            logging.warning(f"Log file not found: {log_path}")
            return []
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()[-max_lines:]
        return lines
    except Exception as e:
        logging.error(f"Error reading logs: {str(e)}")
        return []

def analyze_log(log_lines):
    """วิเคราะห์ log เพื่อหาปัญหา"""
    subsystem_status = {}
    try:
        for line in log_lines:
            line_lower = line.lower()
            severity = None
            
            # ตรวจสอบความรุนแรงของปัญหา
            for level, keywords in ERROR_KEYWORDS.items():
                if any(keyword in line_lower for keyword in keywords):
                    severity = level
                    break
            
            if severity:
                # แยกชื่อโมดูลจาก log
                parts = line.split("]")
                if len(parts) >= 2:
                    module_name = parts[1].strip().split(" ")[0]
                    if module_name not in subsystem_status:
                        subsystem_status[module_name] = {
                            "status": severity,
                            "last_seen": datetime.now().isoformat(),
                            "message": line.strip()
                        }
                    else:
                        # อัปเดตสถานะถ้าพบปัญหาที่รุนแรงกว่า
                        current_severity = subsystem_status[module_name]["status"]
                        if severity == "critical" or (severity == "warning" and current_severity == "info"):
                            subsystem_status[module_name].update({
                                "status": severity,
                                "last_seen": datetime.now().isoformat(),
                                "message": line.strip()
                            })
        
        return subsystem_status
    except Exception as e:
        logging.error(f"Error analyzing logs: {str(e)}")
        return {}

def determine_emotion(issues):
    """กำหนดอารมณ์ของ AI จากปัญหาที่พบ"""
    if not issues:
        return "calm"
    
    # นับจำนวนปัญหาตามระดับความรุนแรง
    severity_count = {"critical": 0, "warning": 0, "info": 0}
    for issue in issues.values():
        severity_count[issue["status"]] += 1
    
    # กำหนดอารมณ์ตามความรุนแรงของปัญหา
    if severity_count["critical"] > 0:
        return "unstable"
    elif severity_count["warning"] > 0:
        return "concerned"
    return "calm"

def update_self_aware(issues, issue_history):
    """อัปเดต self_aware.json ด้วยสถานะล่าสุด"""
    try:
        if not os.path.exists(AWARE_PATH):
            logging.warning(f"Self-aware file not found: {AWARE_PATH}")
            return

        with open(AWARE_PATH, "r", encoding="utf-8") as f:
            current = json.load(f)

        # อัปเดตสถานะของแต่ละโมดูล
        for system_name, system_data in current.get("core_systems", {}).items():
            if system_name in issues:
                issue_data = issues[system_name]
                system_data.update({
                    "status": issue_data["status"],
                    "last_issue": issue_data["last_seen"],
                    "issue_message": issue_data["message"]
                })
                # เพิ่มประวัติปัญหา
                issue_history.add({
                    "module": system_name,
                    "status": issue_data["status"],
                    "message": issue_data["message"]
                })
            else:
                # รีเซ็ตสถานะถ้าไม่มีปัญหา
                system_data.update({
                    "status": "ok",
                    "last_issue": None,
                    "issue_message": None
                })

        # กำหนดอารมณ์ของ AI
        current["self_emotion"] = determine_emotion(issues)

        # เพิ่มข้อมูลการตรวจสอบล่าสุด
        current["monitoring"] = {
            "last_check": datetime.now().isoformat(),
            "total_issues": len(issues),
            "monitor_version": "1.0",
            "issue_history": issue_history.get_all(),
            "status_summary": {
                "critical": sum(1 for i in issues.values() if i["status"] == "critical"),
                "warning": sum(1 for i in issues.values() if i["status"] == "warning"),
                "info": sum(1 for i in issues.values() if i["status"] == "info")
            }
        }

        with open(AWARE_PATH, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2, ensure_ascii=False)
            
        logging.info(f"Updated self-awareness with {len(issues)} issues")
    except Exception as e:
        logging.error(f"Error updating self-awareness: {str(e)}")

def run_once(issue_history):
    """ทำงานหนึ่งรอบของการตรวจสอบ"""
    try:
        logs = read_recent_logs(LOG_PATH)
        issues = analyze_log(logs)
        update_self_aware(issues, issue_history)
        return len(issues)
    except Exception as e:
        logging.error(f"Error in monitoring cycle: {str(e)}")
        return 0

def start_monitoring(interval=10):
    """เริ่มการทำงานของระบบ monitor"""
    logging.info("Starting self-monitoring system...")
    issue_history = IssueHistory()
    try:
        while True:
            issues_found = run_once(issue_history)
            if issues_found > 0:
                logging.warning(f"Found {issues_found} issues in last check")
            time.sleep(interval)
    except KeyboardInterrupt:
        logging.info("Monitoring stopped by user")
    except Exception as e:
        logging.error(f"Monitoring stopped due to error: {str(e)}")

if __name__ == "__main__":
    start_monitoring() 