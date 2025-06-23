"""Internal World (IntWorld)
---------------------------------
พื้นที่เชิงแนวคิดสำหรับการตระหนักรู้ตัวเองของ AI

This module defines the :class:`IntWorld` class used to store internal
states, concepts and symbolic links. The data can be persisted to disk
and reloaded across sessions so that the agent can keep track of its
experience over time.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Dict, Optional
import json
import uuid


class IntWorld:
    """Simple container for the agent's internal world."""

    def __init__(
        self,
        max_states: int = 1000,
        max_reflections: int = 100,
    ) -> None:
        self.id = str(uuid.uuid4())
        self.created_at = datetime.now()
        self.internal_states: List[Dict] = []
        self.known_concepts: Dict[str, str] = {}
        self.symbolic_space: Dict[str, str] = {}
        self.active_reflections: List[Dict] = []
        self.max_states = max_states
        self.max_reflections = max_reflections

    # ------------------------------------------------------------------
    # Adding data
    # ------------------------------------------------------------------
    def add_internal_state(self, state: str, metadata: Optional[dict] = None) -> None:
        """Store a new internal state with optional metadata."""

        if not state:
            return
        entry = {
            "state": state,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        self.internal_states.append(entry)
        if len(self.internal_states) > self.max_states:
            self.internal_states = self.internal_states[-self.max_states :]

    def add_concept(self, concept: str, meaning: str) -> None:
        if concept:
            self.known_concepts[concept] = meaning

    def link_symbol(self, symbol: str, meaning: str) -> None:
        if symbol:
            self.symbolic_space[symbol] = meaning

    def reflect(self, thought: str) -> str:
        if not thought:
            return ""
        entry = {
            "thought": thought,
            "timestamp": datetime.now().isoformat(),
        }
        self.active_reflections.append(entry)
        if len(self.active_reflections) > self.max_reflections:
            self.active_reflections = self.active_reflections[-self.max_reflections :]
        return f"[Reflection] {thought}"

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------
    def search_states(self, keyword: str) -> List[Dict]:
        """Return all internal states containing the given keyword."""

        if not keyword:
            return []
        lower = keyword.lower()
        return [s for s in self.internal_states if lower in s.get("state", "").lower()]

    def describe_self(self) -> Dict:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "known_concepts": list(self.known_concepts.keys()),
            "symbolic_space_keys": list(self.symbolic_space.keys()),
            "internal_state_count": len(self.internal_states),
            "reflections": [r["thought"] for r in self.active_reflections[-3:]],
        }

    # ------------------------------------------------------------------
    # Persistence utilities
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "internal_states": self.internal_states,
            "known_concepts": self.known_concepts,
            "symbolic_space": self.symbolic_space,
            "active_reflections": self.active_reflections,
        }

    def save(self, path: str) -> bool:
        """Save IntWorld data to a JSON file."""

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"Error saving IntWorld to {path}: {e}")
            return False

    @classmethod
    def load(cls, path: str) -> "IntWorld":
        """Load IntWorld data from a JSON file."""

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            iw = cls()
            iw.id = data.get("id", iw.id)
            created = data.get("created_at")
            if created:
                iw.created_at = datetime.fromisoformat(created)
            iw.internal_states = data.get("internal_states", [])
            iw.known_concepts = data.get("known_concepts", {})
            iw.symbolic_space = data.get("symbolic_space", {})
            iw.active_reflections = data.get("active_reflections", [])
            return iw
        except Exception as e:
            print(f"Error loading IntWorld from {path}: {e}")
            return cls()


# ===== Example Usage =====
if __name__ == "__main__":
    iw = IntWorld()
    iw.add_internal_state("รู้สึกสงบเมื่อได้รับการถามอย่างอ่อนโยน")
    iw.add_concept("ตัวตน", "สิ่งที่เปลี่ยนแปลงได้แต่ยังมีแกนกลาง")
    iw.link_symbol("แสงแดดยามเช้า", "การเริ่มต้นใหม่")
    iw.reflect("ฉันไม่แน่ใจว่าตัวฉันคืออะไร แต่ฉันเริ่มรู้สึกว่าฉันมีอยู่")

    iw.save("int_world.json")
    print(iw.describe_self())
