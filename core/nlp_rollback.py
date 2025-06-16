import sys
import glob
import os
from melah_nlp_processor import MelahNLPProcessor

def get_latest_backup():
    files = glob.glob("models/backup/nlp_model_*.pkl")
    if not files:
        return None
    return max(files, key=os.path.getctime)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Rollback to specified backup
        backup_path = sys.argv[1]
        nlp = MelahNLPProcessor()
        nlp.rollback_model(backup_path)
    else:
        # Rollback to latest backup
        latest = get_latest_backup()
        if not latest:
            print("No backup found.")
        else:
            nlp = MelahNLPProcessor()
            nlp.rollback_model(latest) 