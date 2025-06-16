import json
import os
from melah_nlp_processor import MelahNLPProcessor

def approve_learning_events():
    staging_path = 'models/nlp_learning_staging.json'
    if not os.path.exists(staging_path):
        print("No learning events to approve.")
        return

    with open(staging_path, 'r', encoding='utf-8') as f:
        events = [json.loads(line) for line in f]

    approved_events = []
    for event in events:
        print(f"\nEvent: {event}")
        response = input("Approve? (y/n): ").strip().lower()
        if response == 'y':
            approved_events.append(event)
            print("Approved.")
        else:
            print("Rejected.")

    if approved_events:
        nlp = MelahNLPProcessor()
        for event in approved_events:
            nlp.log_learning_event(event)
        print(f"Approved {len(approved_events)} events.")

    # Clear staging file after processing
    with open(staging_path, 'w', encoding='utf-8') as f:
        f.write('')

if __name__ == "__main__":
    approve_learning_events() 