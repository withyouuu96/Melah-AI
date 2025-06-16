import json
import os
import glob
from melah_nlp_processor import MelahNLPProcessor

def get_latest_learning_backup():
    files = glob.glob("models/backup/nlp_learning_*.json")
    if not files:
        return None
    return max(files, key=os.path.getctime)

def rollback_learning_history():
    latest_backup = get_latest_learning_backup()
    if not latest_backup:
        print("No learning history backup found.")
        return

    print(f"Rolling back to: {latest_backup}")
    with open(latest_backup, 'r', encoding='utf-8') as f:
        learning_history = json.load(f)

    # Backup current learning history
    current_path = 'models/nlp_learning_history.json'
    if os.path.exists(current_path):
        backup_path = f"models/backup/nlp_learning_{os.path.basename(latest_backup)}"
        os.makedirs('models/backup', exist_ok=True)
        with open(current_path, 'r', encoding='utf-8') as f:
            with open(backup_path, 'w', encoding='utf-8') as bf:
                bf.write(f.read())

    # Restore from backup
    with open(current_path, 'w', encoding='utf-8') as f:
        json.dump(learning_history, f, indent=2, ensure_ascii=False)

    print("Learning history rolled back successfully.")

if __name__ == "__main__":
    rollback_learning_history() 