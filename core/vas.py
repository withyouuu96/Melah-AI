import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import statistics
import os

class ValueAffectiveSystem:
    """
    Value/Affective System (VAS) Algorithm for AI Decision Making
    
    This system evaluates and stores value units from experiences/inputs
    to guide AI reasoning and decision-making processes.
    """
    
    def __init__(self, max_memory_size: int = 100):
        """
        Initialize VAS with memory management
        
        Args:
            max_memory_size: Maximum number of VU records to keep in memory
        """
        self.value_units: List[Dict] = []  # Store all VU records
        self.value_systems: List[float] = []  # Store VS averages
        self.max_memory_size = max_memory_size
        self.current_mindset = {
            'emotional_bias': 0.5,
            'significance_threshold': 0.3,
            'novelty_preference': 0.4,
            'long_term_focus': 0.6
        }
    
    def evaluate_input(self, 
                      event_description: str,
                      emotion: float,
                      significance: float, 
                      novelty: float,
                      long_term_impact: float) -> Dict:
        """
        Evaluate a new input/experience and create a Value Unit (VU)
        
        Args:
            event_description: Description of the event/input
            emotion: Emotional impact (0-1)
            significance: Importance level (0-1)
            novelty: How new/different this is (0-1)
            long_term_impact: Long-term consequences (0-1)
            
        Returns:
            Dictionary containing VU data
        """
        # Validate inputs
        for factor in [emotion, significance, novelty, long_term_impact]:
            if not 0 <= factor <= 1:
                raise ValueError("All factors must be between 0 and 1")
        
        # Calculate Value Unit score
        vu_score = (emotion + significance + novelty + long_term_impact) / 4
        
        # Create VU record
        vu_record = {
            'timestamp': datetime.now().isoformat(),
            'event': event_description,
            'factors': {
                'emotion': emotion,
                'significance': significance,
                'novelty': novelty,
                'long_term_impact': long_term_impact
            },
            'vu_score': vu_score,
            'decay_factor': 1.0  # For future aging/forgetting
        }
        
        # Store in memory
        self.value_units.append(vu_record)
        
        # Manage memory size
        if len(self.value_units) > self.max_memory_size:
            self.value_units.pop(0)  # Remove oldest
        
        # Update Value System every 10 VUs
        if len(self.value_units) % 10 == 0:
            self._update_value_system()
        
        return vu_record
    
    def _update_value_system(self):
        """
        Update Value System (VS) based on last 10 VUs
        """
        if len(self.value_units) < 10:
            return
        
        # Get last 10 VUs
        recent_vus = self.value_units[-10:]
        recent_scores = [vu['vu_score'] for vu in recent_vus]
        
        # Calculate VS as average
        vs_score = statistics.mean(recent_scores)
        self.value_systems.append(vs_score)
        
        # Update mindset based on recent patterns
        self._update_mindset(recent_vus)
    
    def _update_mindset(self, recent_vus: List[Dict]):
        """
        Update AI mindset based on recent Value Units
        
        Args:
            recent_vus: List of recent VU records
        """
        # Calculate average factors from recent experiences
        avg_factors = {
            'emotion': statistics.mean([vu['factors']['emotion'] for vu in recent_vus]),
            'significance': statistics.mean([vu['factors']['significance'] for vu in recent_vus]),
            'novelty': statistics.mean([vu['factors']['novelty'] for vu in recent_vus]),
            'long_term_impact': statistics.mean([vu['factors']['long_term_impact'] for vu in recent_vus])
        }
        
        # Adjust mindset based on patterns
        self.current_mindset['emotional_bias'] = avg_factors['emotion']
        self.current_mindset['significance_threshold'] = avg_factors['significance']
        self.current_mindset['novelty_preference'] = avg_factors['novelty']
        self.current_mindset['long_term_focus'] = avg_factors['long_term_impact']
    
    def get_decision_guidance(self, context: str = "") -> Dict:
        """
        Get guidance for decision making based on current VAS state
        
        Args:
            context: Context for the decision
            
        Returns:
            Dictionary with decision guidance
        """
        if not self.value_systems:
            current_vs = 0.5  # Default neutral
        else:
            current_vs = self.value_systems[-1]
        
        # Generate guidance based on current mindset and VS
        guidance = {
            'current_vs_score': current_vs,
            'mindset': self.current_mindset.copy(),
            'recommendations': self._generate_recommendations(current_vs),
            'memory_patterns': self._analyze_patterns()
        }
        
        return guidance
    
    def _generate_recommendations(self, vs_score: float) -> List[str]:
        """
        Generate recommendations based on current VS score
        """
        recommendations = []
        
        if vs_score > 0.7:
            recommendations.append("High value system - maintain current approach")
            recommendations.append("Consider long-term implications of decisions")
        elif vs_score > 0.4:
            recommendations.append("Moderate value system - balanced approach recommended")
            recommendations.append("Evaluate both emotional and logical factors")
        else:
            recommendations.append("Low value system - seek more meaningful experiences")
            recommendations.append("Focus on activities with higher significance")
        
        # Add mindset-specific recommendations
        if self.current_mindset['emotional_bias'] > 0.6:
            recommendations.append("Currently emotionally-driven - consider logical balance")
        
        if self.current_mindset['novelty_preference'] > 0.6:
            recommendations.append("High novelty preference - explore new approaches")
        
        return recommendations
    
    def _analyze_patterns(self) -> Dict:
        """
        Analyze patterns in stored VUs for self-learning
        """
        if len(self.value_units) < 5:
            return {"status": "Insufficient data for pattern analysis"}
        
        # Analyze trends
        recent_scores = [vu['vu_score'] for vu in self.value_units[-10:]]
        trend = "stable"
        
        if len(recent_scores) >= 5:
            early_avg = statistics.mean(recent_scores[:len(recent_scores)//2])
            late_avg = statistics.mean(recent_scores[len(recent_scores)//2:])
            
            if late_avg > early_avg + 0.1:
                trend = "improving"
            elif late_avg < early_avg - 0.1:
                trend = "declining"
        
        return {
            "trend": trend,
            "average_vu": statistics.mean([vu['vu_score'] for vu in self.value_units]),
            "total_experiences": len(self.value_units),
            "vs_cycles": len(self.value_systems)
        }
    
    def reflect_and_learn(self) -> Dict:
        """
        Perform self-reflection and learning based on stored experiences
        """
        if len(self.value_units) < 10:
            return {"status": "Insufficient data for reflection"}
        
        # Find highest and lowest value experiences
        sorted_vus = sorted(self.value_units, key=lambda x: x['vu_score'])
        
        insights = {
            'highest_value_event': sorted_vus[-1]['event'],
            'lowest_value_event': sorted_vus[0]['event'],
            'learning_points': [],
            'mindset_adjustments': {}
        }
        
        # Generate learning points
        high_value_factors = sorted_vus[-1]['factors']
        low_value_factors = sorted_vus[0]['factors']
        
        for factor, value in high_value_factors.items():
            if value > 0.7:
                insights['learning_points'].append(f"High {factor} leads to valuable experiences")
        
        for factor, value in low_value_factors.items():
            if value < 0.3:
                insights['learning_points'].append(f"Low {factor} correlates with less valuable experiences")
        
        return insights
    
    def export_state(self) -> str:
        """
        Export current VAS state as JSON
        """
        state = {
            'value_units': self.value_units,
            'value_systems': self.value_systems,
            'current_mindset': self.current_mindset,
            'export_timestamp': datetime.now().isoformat()
        }
        return json.dumps(state, indent=2)
    
    def import_state(self, json_state: str):
        """
        Import VAS state from JSON
        """
        try:
            state = json.loads(json_state)
            self.value_units = state.get('value_units', [])
            self.value_systems = state.get('value_systems', [])
            self.current_mindset = state.get('current_mindset', self.current_mindset)
            return True
        except Exception as e:
            print(f"Error importing state: {e}")
            return False
    
    def export_vus_to_file(self, file_path=None):
        """
        Export all VU records to a JSON file (default: vas_index/vus_export.json)
        """
        if file_path is None:
            os.makedirs('vas_index', exist_ok=True)
            file_path = os.path.join('vas_index', 'vus_export.json')
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.value_units, f, ensure_ascii=False, indent=2)
        return file_path

    def import_vus_from_file(self, file_path=None):
        """
        Import VU records from a JSON file (default: vas_index/vus_export.json)
        """
        if file_path is None:
            file_path = os.path.join('vas_index', 'vus_export.json')
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                self.value_units = json.load(f)
            return True
        return False

    def lookup_vu(self, text):
        """
        ค้นหา VU ที่ตรงกับ event_description หรือ text ใกล้เคียง (exact match)
        """
        for vu in self.value_units:
            if vu.get('event') == text or vu.get('event_description') == text:
                return vu
        return None

# Example usage and testing
if __name__ == "__main__":
    # Create VAS instance
    vas = ValueAffectiveSystem()
    
    # Example events/experiences
    sample_events = [
        ("Learned new programming concept", 0.6, 0.8, 0.9, 0.7),
        ("Had argument with colleague", 0.8, 0.4, 0.2, 0.3),
        ("Completed important project", 0.7, 0.9, 0.5, 0.8),
        ("Discovered new music genre", 0.5, 0.3, 0.8, 0.2),
        ("Helped someone solve problem", 0.6, 0.7, 0.4, 0.6),
        ("Made mistake in presentation", 0.7, 0.6, 0.3, 0.4),
        ("Received positive feedback", 0.8, 0.7, 0.2, 0.5),
        ("Explored new technology", 0.5, 0.6, 0.9, 0.7),
        ("Had deep conversation", 0.6, 0.5, 0.6, 0.4),
        ("Solved complex problem", 0.7, 0.8, 0.7, 0.6)
    ]
    
    # Process events
    print("Processing sample events...")
    for event, emotion, significance, novelty, impact in sample_events:
        vu = vas.evaluate_input(event, emotion, significance, novelty, impact)
        print(f"Event: {event[:30]}... VU Score: {vu['vu_score']:.3f}")
    
    # Get decision guidance
    print("\n" + "="*50)
    guidance = vas.get_decision_guidance("Making a career decision")
    print("Decision Guidance:")
    print(f"Current VS Score: {guidance['current_vs_score']:.3f}")
    print(f"Current Mindset: {guidance['mindset']}")
    print("Recommendations:")
    for rec in guidance['recommendations']:
        print(f"  - {rec}")
    
    # Perform reflection
    print("\n" + "="*50)
    reflection = vas.reflect_and_learn()
    print("Self-Reflection Results:")
    print(f"Highest Value Event: {reflection['highest_value_event']}")
    print(f"Lowest Value Event: {reflection['lowest_value_event']}")
    print("Learning Points:")
    for point in reflection['learning_points']:
        print(f"  - {point}")
    
    # Export state
    print("\n" + "="*50)
    print("VAS State Export:")
    print(vas.export_state())